"""Hierarchical Transformer for Banking Transaction Embeddings — see architecture/overview.md."""

import torch
import torch.nn as nn
from loguru import logger

from src.models.hier_transformer.encoder import TransactionEncoder
from src.models.hier_transformer.features import (
    DEFAULT_FEATURES,
    FeatureSpec,
    NumericFeature,
    NumericNormalizer,
    categorical_vocab_sizes,
    numeric_field_names,
)
from src.models.hier_transformer.field_transformer import FieldTransformer
from src.models.hier_transformer.sequence_encoder import SequenceTransformer
from src.models.hier_transformer.loss import MTMHead, ContrastiveHead


class TransactionTransformer(nn.Module):
    """Full hierarchical Transformer for transaction embeddings.

    Combines: TransactionEncoder → FieldTransformer → SequenceTransformer
    With optional pre-training heads (MTM + Contrastive).

    Total params: ~2.5M
    """

    def __init__(
        self,
        features: list[FeatureSpec] = None,
        d_field: int = 64,
        d_model: int = 128,
        n_frequencies: int = 16,
        field_n_layers: int = 2,
        field_n_heads: int = 4,
        seq_n_layers: int = 4,
        seq_n_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        time_alpha_init: float = 0.1,
        use_gradient_checkpointing: bool = True,
        pretrain: bool = True,
    ):
        if features is None:
            raise ValueError("'features' can't be None")
        super().__init__()

        # --- Backbone ---
        self.encoder = TransactionEncoder(features, d_field, n_frequencies)
        self.field_transformer = FieldTransformer(
            n_fields=self.encoder.n_fields,
            d_field=d_field,
            d_model=d_model,
            n_layers=field_n_layers,
            n_heads=field_n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.sequence_transformer = SequenceTransformer(
            d_model=d_model,
            n_layers=seq_n_layers,
            n_heads=seq_n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            time_alpha_init=time_alpha_init,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # --- Pre-training heads (removable) ---
        self.pretrain = pretrain
        if pretrain:
            self.mtm_head = MTMHead(
                d_model,
                vocab_sizes=categorical_vocab_sizes(features),
                numeric_names=numeric_field_names(features),
            )
            self.contrastive_head = ContrastiveHead(d_model)

    def _encode_client(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = batch.get("padding_mask", None)  # (B, T), True=padded
        field_embeddings = self.encoder(batch)
        transaction_embeddings = self.field_transformer(field_embeddings)
        h_cls = self.sequence_transformer(
            transaction_embeddings,
            delta_t=batch["delta_t"],
            padding_mask=padding_mask,
        )
        return h_cls, transaction_embeddings

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h_cls, transaction_embeddings = self._encode_client(batch)
        output = {"h_cls": h_cls}
        output["time_alpha"] = self.sequence_transformer.time_alpha
        if self.pretrain:
            output["mtm_preds"] = self.mtm_head(transaction_embeddings)
            output["contrastive_z"] = self.contrastive_head(h_cls)
            output["temperature"] = self.contrastive_head.temperature

        return output

    def get_client_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Inference-only: returns just the client embedding vector.

        Args:
            batch: dict with field tensors
        Returns:
            (B, d_model) — client embedding
        """
        with torch.no_grad():
            h_cls, _ = self._encode_client(batch)
            return h_cls


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count trainable parameters per sub-module."""
    counts = {}
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters() if p.requires_grad)
        counts[name] = n
    counts["total"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts


def count_saved_parameters(model_state: dict[str, torch.Tensor]) -> dict[str, int]:
    """Count tensor values persisted in a model state dict by sub-module."""
    counts: dict[str, int] = {}
    for name, value in model_state.items():
        if not torch.is_tensor(value):
            continue
        parts = name.split(".")
        module_name = parts[1] if parts[0] == "backbone" and len(parts) > 1 else parts[0]
        counts[module_name] = counts.get(module_name, 0) + value.numel()
    counts["total"] = sum(counts.values())
    return counts


class EmbeddingModel(nn.Module):
    """High-level wrapper around :class:`TransactionTransformer`.

    Adds production-friendly helpers (``embed`` for inference, ``save``/
    ``load`` for checkpoint round-trip) while keeping the underlying
    backbone fully accessible via ``self.backbone``.

    Subclasses can swap the backbone architecture by overriding
    ``backbone_cls`` (``save``/``load`` keep working since they go through
    ``cls``).
    """

    backbone_cls: type[TransactionTransformer] = TransactionTransformer

    def __init__(self, features: list[FeatureSpec], **kwargs):
        super().__init__()
        self.features = features
        self.backbone = self.backbone_cls(features=features, **kwargs)
        self._init_kwargs = kwargs  # remembered for ``load``

    def forward(self, batch: dict[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:
        return self.backbone(batch)

    def embed(self, batch: dict[str, "torch.Tensor"]) -> "torch.Tensor":
        """Inference-only: return the (B, d_model) client embedding."""
        return self.backbone.get_client_embedding(batch)

    def save(
        self,
        path,
        optimizer_state: dict | None = None,
        step: int | None = None,
    ) -> None:
        payload: dict = {"model_state": self.state_dict()}
        # Numeric normalizers are plain Python objects (not part of state_dict),
        # so persist their fitted stats explicitly — otherwise inference would
        # have to re-fit them on a different population than train time.
        norm_states = {
            f.name: f.normalizer.state_dict()
            for f in self.features
            if isinstance(f, NumericFeature)
            and f.normalizer is not None
            and f.normalizer.fitted
        }
        if norm_states:
            payload["normalizer_states"] = norm_states
        if optimizer_state is not None:
            payload["optimizer_state"] = optimizer_state
        if step is not None:
            payload["step"] = step
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path,
        features: list[FeatureSpec],
        map_location: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> "EmbeddingModel":
        state = torch.load(path, map_location=map_location, weights_only=False)
        # Restore fitted numeric normalizers onto the feature specs so the
        # encoder normalizes inference inputs exactly as it did at train time.
        norm_states = state.get("normalizer_states", {})
        for feat in features:
            if isinstance(feat, NumericFeature) and feat.name in norm_states:
                if feat.normalizer is None:
                    feat.normalizer = NumericNormalizer()
                feat.normalizer.load_state_dict(norm_states[feat.name])
        model = cls(features=features, **kwargs)
        model_state = state["model_state"]
        saved_param_stats = count_saved_parameters(model_state)
        logger.info(
            "Saved model parameters: {:,} total",
            saved_param_stats["total"],
        )
        logger.debug("Saved parameter breakdown: {}", saved_param_stats)
        time_alpha_key = "backbone.sequence_transformer.time_log_alpha"
        if strict and time_alpha_key not in model_state:
            model_state[time_alpha_key] = model.state_dict()[time_alpha_key]
        model.load_state_dict(model_state, strict=strict)
        return model
