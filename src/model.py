"""Hierarchical Transformer for Banking Transaction Embeddings — see architecture/overview.md."""

import torch
import torch.nn as nn

from .encoder import (
    TransactionEncoder,
    NumericFeature, CategoricalFeature, DatetimeFeature, DoubleHashFeature,
    FeatureSpec, categorical_vocab_sizes,
)
from .field_encoder import FieldTransformer
from .sequence_encoder import SequenceTransformer
from .loss import MTMHead, ContrastiveHead


# Default feature schema matching the original 13-field design:
#   importo(signed) → 2, saldo_post → 1, delta_t → 1, merchant(hash) → 1,
#   mcc/canale/macro_tipo/sotto_tipo/divisa → 5, timestamp → 3   total = 13
DEFAULT_FEATURES: list[FeatureSpec] = [
    NumericFeature("importo", signed=True),
    NumericFeature("saldo_post"),
    NumericFeature("delta_t"),
    DoubleHashFeature("merchant"),
    CategoricalFeature("mcc",        801),
    CategoricalFeature("canale",      11),
    CategoricalFeature("macro_tipo",   9),
    CategoricalFeature("sotto_tipo",  41),
    CategoricalFeature("divisa",       6),
    DatetimeFeature("timestamp"),
]


class TransactionTransformer(nn.Module):
    """Full hierarchical Transformer for transaction embeddings.

    Combines: TransactionEncoder → FieldTransformer → SequenceTransformer
    With optional pre-training heads (MTM + Contrastive).

    Total params: ~2.5M
    """

    def __init__(
        self,
        features: list[FeatureSpec] = DEFAULT_FEATURES,
        d_field: int = 64,
        d_model: int = 128,
        n_frequencies: int = 16,
        field_n_layers: int = 2,
        field_n_heads: int = 4,
        seq_n_layers: int = 4,
        seq_n_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
        pretrain: bool = True,
    ):
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
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # --- Pre-training heads (removable) ---
        self.pretrain = pretrain
        if pretrain:
            self.mtm_head = MTMHead(d_model, vocab_sizes=categorical_vocab_sizes(features))
            self.contrastive_head = ContrastiveHead(d_model)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: dict with field tensors (B, T) + 'delta_t' + optional 'padding_mask'
        Returns:
            dict with:
                'h_cls': (B, d_model) — client embedding
                if pretrain:
                    'mtm_preds': dict of field predictions
                    'contrastive_z': (B, d_proj) — normalized projections
        """
        padding_mask = batch.get("padding_mask", None)  # (B, T), True=padded

        # 1. Encode raw fields → (B, T, n_fields, d_field)
        field_embeddings = self.encoder(batch)

        # 2. Field Transformer → (B, T, d_model)
        transaction_embeddings = self.field_transformer(field_embeddings, padding_mask)

        # 3. Sequence Transformer → (B, d_model)
        h_cls = self.sequence_transformer(
            transaction_embeddings,
            delta_t=batch["delta_t"],
            padding_mask=padding_mask,
        )

        output = {"h_cls": h_cls}

        # 4. Pre-training heads
        if self.pretrain:
            output["mtm_preds"] = self.mtm_head(transaction_embeddings)
            output["contrastive_z"] = self.contrastive_head(h_cls)

        return output

    def get_client_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Inference-only: returns just the client embedding vector.

        Args:
            batch: dict with field tensors
        Returns:
            (B, d_model) — client embedding
        """
        with torch.no_grad():
            padding_mask = batch.get("padding_mask", None)
            field_emb = self.encoder(batch)
            tx_emb = self.field_transformer(field_emb, padding_mask)
            return self.sequence_transformer(tx_emb, batch["delta_t"], padding_mask)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count trainable parameters per sub-module."""
    counts = {}
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters() if p.requires_grad)
        counts[name] = n
    counts["total"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Instantiate model
    model = TransactionTransformer(pretrain=True).to(device)

    # Parameter count
    params = count_parameters(model)
    print("\nParameter count:")
    for name, n in params.items():
        print(f"  {name}: {n:,}")

    # Dummy batch (B=4, T=32)
    B, T = 4, 32
    # Unix timestamps in [2020-01-01, 2024-01-01] range; 0 = padding
    ts_base = 1577836800  # 2020-01-01 00:00 UTC
    batch = {
        # Numeric (float); importo may be negative (signed)
        "importo":    torch.randn(B, T, device=device) * 500,
        "saldo_post": torch.randn(B, T, device=device) * 1000,
        "delta_t":    torch.abs(torch.randn(B, T, device=device)) * 86400,
        # Merchant double-hash (keys: merchant_a, merchant_b)
        "merchant_a": torch.randint(1, 8192, (B, T), device=device),
        "merchant_b": torch.randint(1, 8192, (B, T), device=device),
        # Categoricals
        "mcc":        torch.randint(1, 800, (B, T), device=device),
        "canale":     torch.randint(1, 10,  (B, T), device=device),
        "macro_tipo": torch.randint(1, 8,   (B, T), device=device),
        "sotto_tipo": torch.randint(1, 40,  (B, T), device=device),
        "divisa":     torch.randint(1, 5,   (B, T), device=device),
        # Datetime as Unix timestamp (int64); encoder decomposes automatically
        "timestamp":  (torch.randint(0, 126_230_400, (B, T), device=device) + ts_base).long(),
        # Padding mask (last 8 transactions are padded)
        "padding_mask": torch.cat([
            torch.zeros(B, T - 8, dtype=torch.bool, device=device),
            torch.ones(B, 8, dtype=torch.bool, device=device),
        ], dim=1),
    }

    # Forward pass
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
        output = model(batch)

    print(f"\nh_cls shape: {output['h_cls'].shape}")
    if "contrastive_z" in output:
        print(f"contrastive_z shape: {output['contrastive_z'].shape}")
    print(f"MTM pred keys: {list(output.get('mtm_preds', {}).keys())}")

    # Test inference mode
    emb = model.get_client_embedding(batch)
    print(f"\nInference embedding shape: {emb.shape}")
    print("All shapes OK!")
