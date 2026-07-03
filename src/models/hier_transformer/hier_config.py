from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import config


_MODEL_ARCH_SECTION = "model.hier_transformer.architecture"
_MODEL_TRAINING_SECTION = "model.hier_transformer.training"
_MODEL_DATA_SECTION = "model.hier_transformer.data"
_MODEL_PATHS_SECTION = "model.hier_transformer.paths"
_MODEL_PERTURB_SECTION = "model.hier_transformer.perturbation"
_MODEL_RUNTIME_SECTION = "model.hier_transformer.runtime"


def _get_config_value(
    primary_section: str,
    key: str,
    *,
    value_type: type | tuple[type, ...],
    fallback_sections: tuple[str, ...] = (),
) -> Any:
    for section in (primary_section, *fallback_sections):
        try:
            return config.get(section, key, value_type=value_type)
        except KeyError:
            continue
    fallback_list = ", ".join((primary_section, *fallback_sections))
    raise KeyError(f"Key `{key}` not found in any of: {fallback_list}.")


def _get_config_value_with_aliases(
    primary_section: str,
    keys: tuple[str, ...],
    *,
    value_type: type | tuple[type, ...],
    fallback_sections: tuple[str, ...] = (),
) -> Any:
    searched: list[str] = []
    for section in (primary_section, *fallback_sections):
        for key in keys:
            searched.append(f"{section}.{key}")
            try:
                return config.get(section, key, value_type=value_type)
            except KeyError:
                continue
    raise KeyError(f"None of these config keys were found: {', '.join(searched)}.")


@dataclass
class ModelConfig:
    """Architecture hyper-parameters forwarded to ``TransactionTransformer``."""

    d_field: int = config.get(_MODEL_ARCH_SECTION, "d_field", value_type=int)
    d_model: int = config.get(_MODEL_ARCH_SECTION, "d_model", value_type=int)
    n_frequencies: int = config.get(_MODEL_ARCH_SECTION, "n_frequencies", value_type=int)
    field_n_layers: int = config.get(_MODEL_ARCH_SECTION, "field_n_layers", value_type=int)
    field_n_heads: int = config.get(_MODEL_ARCH_SECTION, "field_n_heads", value_type=int)
    seq_n_layers: int = config.get(_MODEL_ARCH_SECTION, "seq_n_layers", value_type=int)
    seq_n_heads: int = config.get(_MODEL_ARCH_SECTION, "seq_n_heads", value_type=int)
    dim_feedforward: int = config.get(
        _MODEL_ARCH_SECTION,
        "dim_feedforward",
        value_type=int,
    )
    dropout: float = config.get(_MODEL_ARCH_SECTION, "dropout", value_type=float)
    time_alpha_init: float = config.get(
        _MODEL_ARCH_SECTION,
        "time_alpha_init",
        default=0.1,
        value_type=float,
    )


@dataclass
class PathsConfig:
    """Input/output paths shared by training, prediction and artifact export."""

    train_path: Path = config.get(_MODEL_PATHS_SECTION, "train_path", value_type=Path)
    pred_input_path: Path = config.get(
        _MODEL_PATHS_SECTION,
        "pred_input_path",
        value_type=Path,
    )
    pred_output_path: Path = config.get(
        _MODEL_PATHS_SECTION,
        "pred_output_path",
        value_type=Path,
    )
    model_output_dir: Path = config.get(
        _MODEL_PATHS_SECTION,
        "model_output_dir",
        value_type=Path,
    )

    def __post_init__(self) -> None:
        if not self.pred_output_path.suffix:
            raise ValueError(
                "`model.hier_transformer.paths.pred_output_path` must be a file path.",
            )


@dataclass
class DataPipelineConfig:
    """Window sampling and batching parameters shared by train and prediction."""

    seq_len: int = _get_config_value(
        _MODEL_DATA_SECTION,
        "seq_len",
        value_type=int,
        fallback_sections=(_MODEL_TRAINING_SECTION,),
    )
    pred_windows_per_client: int = _get_config_value_with_aliases(
        _MODEL_DATA_SECTION,
        ("pred_windows_per_client", "inference_windows_per_client", "windows_per_client"),
        value_type=int,
        fallback_sections=(_MODEL_TRAINING_SECTION,),
    )
    clients_per_batch: int = _get_config_value(
        _MODEL_DATA_SECTION,
        "clients_per_batch",
        value_type=int,
        fallback_sections=(_MODEL_TRAINING_SECTION,),
    )
    train_windows_per_client: int = _get_config_value_with_aliases(
        _MODEL_DATA_SECTION,
        ("train_windows_per_client", "train_views_per_client", "views_per_client"),
        value_type=int,
        fallback_sections=(_MODEL_TRAINING_SECTION,),
    )

    @property
    def batch_size(self) -> int:
        return self.clients_per_batch * self.train_windows_per_client

    @property
    def windows_per_client(self) -> int:
        """Backward-compatible alias for inference window count."""
        return self.pred_windows_per_client

    @property
    def inference_windows_per_client(self) -> int:
        """Backward-compatible alias for prediction window count."""
        return self.pred_windows_per_client

    @property
    def views_per_client(self) -> int:
        """Backward-compatible alias for training view count."""
        return self.train_windows_per_client

    @property
    def train_views_per_client(self) -> int:
        """Backward-compatible alias for training window count."""
        return self.train_windows_per_client


@dataclass
class TrainingConfig:
    """Optimization and evaluation hyper-parameters for pre-training."""

    epochs: int = config.get(_MODEL_TRAINING_SECTION, "epochs", value_type=int)
    mask_prob: float = config.get(_MODEL_TRAINING_SECTION, "mask_prob", value_type=float)
    contrastive_weight: float = config.get(
        _MODEL_TRAINING_SECTION,
        "contrastive_weight",
        value_type=float,
    )
    lr: float = config.get(_MODEL_TRAINING_SECTION, "lr", value_type=float)
    weight_decay: float = config.get(
        _MODEL_TRAINING_SECTION,
        "weight_decay",
        value_type=float,
    )
    lr_gamma: float = config.get(_MODEL_TRAINING_SECTION, "lr_gamma", value_type=float)
    grad_clip: float = config.get(_MODEL_TRAINING_SECTION, "grad_clip", value_type=float)

    val_frac: float = config.get(_MODEL_TRAINING_SECTION, "val_frac", value_type=float)
    val_every: int = config.get(_MODEL_TRAINING_SECTION, "val_every", value_type=int)

    early_stopping_patience: int = config.get(
        _MODEL_TRAINING_SECTION,
        "early_stopping_patience",
        value_type=int,
    )
    early_stopping_min_delta: float = config.get(
        _MODEL_TRAINING_SECTION,
        "early_stopping_min_delta",
        value_type=float,
    )

    log_every: int = config.get(_MODEL_TRAINING_SECTION, "log_every", value_type=int)


@dataclass
class RuntimeConfig:
    """Runtime-only options such as device selection and randomness."""

    device: str | None = (
        _get_config_value(
            _MODEL_RUNTIME_SECTION,
            "device",
            value_type=str,
            fallback_sections=(_MODEL_TRAINING_SECTION,),
        )
        or None
    )
    seed: int = _get_config_value(
        _MODEL_RUNTIME_SECTION,
        "seed",
        value_type=int,
        fallback_sections=(_MODEL_TRAINING_SECTION,),
    )

    def __post_init__(self) -> None:
        if self.device == "gpu":
            self.device = "cuda"


@dataclass
class PerturbationConfig:
    """Settings for permutation-drift evaluation."""

    column: str | None = (
        config.get(_MODEL_PERTURB_SECTION, "column", value_type=str) or None
    )
    output_path: Path = config.get(
        _MODEL_PERTURB_SECTION,
        "output_path",
        value_type=Path,
    )
    classification_output_path: Path = config.get(
        _MODEL_PERTURB_SECTION,
        "classification_output_path",
        default=Path("model_artifacts/classification_perturbation.csv"),
        value_type=Path,
    )
    classification_test_size: float = config.get(
        _MODEL_PERTURB_SECTION,
        "classification_test_size",
        default=0.3,
        value_type=float,
    )
    classification_max_iter: int = config.get(
        _MODEL_PERTURB_SECTION,
        "classification_max_iter",
        default=1000,
        value_type=int,
    )
    classification_n_seeds: int = config.get(
        _MODEL_PERTURB_SECTION,
        "classification_n_seeds",
        default=5,
        value_type=int,
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.classification_test_size < 1.0:
            raise ValueError(
                "`model.hier_transformer.perturbation.classification_test_size` "
                "must be between 0 and 1.",
            )
        if self.classification_max_iter < 1:
            raise ValueError(
                "`model.hier_transformer.perturbation.classification_max_iter` "
                "must be >= 1.",
            )
        if self.classification_n_seeds < 1:
            raise ValueError(
                "`model.hier_transformer.perturbation.classification_n_seeds` "
                "must be >= 1.",
            )


@dataclass
class HierTransformerConfig:
    """Top-level configuration for the hierarchical Transformer pipeline."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
