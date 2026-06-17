from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import config


_MODEL_ARCH_SECTION = "model.hierTransformer.architecture"
_MODEL_TRAINING_SECTION = "model.hierTransformer.training"
_MODEL_PATHS_SECTION = "model.hierTransformer.paths"
_MODEL_PERTURB_SECTION = "model.hierTransformer.perturbation"
_MISSING = object()


def _cfg_int(section: str, key: str, default: int) -> int:
    value = config.get(section, key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{section}.{key}` must be an integer, got {type(value).__name__}.")
    return value


def _cfg_float(section: str, key: str, default: float) -> float:
    value = config.get(section, key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{section}.{key}` must be numeric, got {type(value).__name__}.")
    return float(value)


def _cfg_optional_str(section: str, key: str, default: str | None = None) -> str | None:
    value = config.get(section, key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"`{section}.{key}` must be a string, got {type(value).__name__}.")
    return value


def _cfg_path(section: str, key: str, default: Path) -> Path:
    value = config.get(section, key, str(default))
    if isinstance(value, Path):
        return value
    if not isinstance(value, str):
        raise TypeError(f"`{section}.{key}` must be a string path, got {type(value).__name__}.")
    return Path(value)


def _cfg_str(section: str, key: str, default: str) -> str:
    value = config.get(section, key, default)
    if not isinstance(value, str):
        raise TypeError(f"`{section}.{key}` must be a string, got {type(value).__name__}.")
    return value


def _cfg_path_with_aliases(section: str, keys: tuple[str, ...], default: Path) -> Path:
    for key in keys:
        try:
            value = config.get(section, key, _MISSING)
        except KeyError:
            continue
        if value is _MISSING:
            continue
        if isinstance(value, Path):
            return value
        if not isinstance(value, str):
            raise TypeError(f"`{section}.{key}` must be a string path, got {type(value).__name__}.")
        return Path(value)
    return Path(default)


def _cfg_train_path() -> Path:
    return _cfg_path_with_aliases(
        _MODEL_PATHS_SECTION,
        ("train_path", "csv_path"),
        Path("data") / "transactions.csv",
    )


def _cfg_pred_output_dir() -> Path:
    default = Path("data") / "pred"
    pred_dir = _cfg_path_with_aliases(_MODEL_PATHS_SECTION, ("pred_path",), default)
    if pred_dir != default:
        return pred_dir

    legacy_pred_csv_path = _cfg_path_with_aliases(_MODEL_PATHS_SECTION, ("pred_csv_path",), default)
    if legacy_pred_csv_path != default:
        return legacy_pred_csv_path.parent if legacy_pred_csv_path.suffix else legacy_pred_csv_path
    return pred_dir


@dataclass
class ModelConfig:
    """Architecture hyper-parameters for :class:`src.model.EmbeddingModel`.

    These are forwarded as keyword arguments to ``TransactionTransformer``.
    ``pretrain`` and ``use_gradient_checkpointing`` are intentionally left
    out: the first is always ``True`` during pre-training, the second is
    derived from the runtime device in :mod:`src.train`.
    """

    d_field: int = _cfg_int(_MODEL_ARCH_SECTION, "d_field", 64)          # dimensione dell'embedding di ogni singolo campo
    d_model: int = _cfg_int(_MODEL_ARCH_SECTION, "d_model", 128)         # dimensione interna del modello (transazione + sequenza)
    n_frequencies: int = _cfg_int(_MODEL_ARCH_SECTION, "n_frequencies", 16)    # n. di frequenze sin/cos del NumericEncoder
    field_n_layers: int = _cfg_int(_MODEL_ARCH_SECTION, "field_n_layers", 2)    # n. di layer del FieldTransformer (attenzione intra-transazione)
    field_n_heads: int = _cfg_int(_MODEL_ARCH_SECTION, "field_n_heads", 4)     # n. di teste di attenzione del FieldTransformer
    seq_n_layers: int = _cfg_int(_MODEL_ARCH_SECTION, "seq_n_layers", 4)      # n. di layer del SequenceTransformer (attenzione sulla sequenza)
    seq_n_heads: int = _cfg_int(_MODEL_ARCH_SECTION, "seq_n_heads", 8)       # n. di teste di attenzione del SequenceTransformer
    dim_feedforward: int = _cfg_int(_MODEL_ARCH_SECTION, "dim_feedforward", 512) # dimensione della FFN interna ai blocchi Transformer
    dropout: float = _cfg_float(_MODEL_ARCH_SECTION, "dropout", 0.1)       # probabilità di dropout in tutti i blocchi Transformer


@dataclass
class TrainingConfig:
    """All hyper-parameters and I/O paths for end-to-end pre-training."""

    train_path: Path = field(default_factory=_cfg_train_path)
    pred_path: Path = field(default_factory=_cfg_pred_output_dir)
    pred_file_name: str = _cfg_str(_MODEL_PATHS_SECTION, "pred_file_name", "window_embeddings.csv")
    ckpt_dir: Path = field(default_factory=lambda: _cfg_path(_MODEL_PATHS_SECTION, "ckpt_dir", Path("checkpoints")))

    # Perturbation analysis: which column to perturb ("" = all) and where to write the report.
    perturb_column: str | None = _cfg_optional_str(_MODEL_PERTURB_SECTION, "column", None)
    perturb_path: Path = field(default_factory=lambda: _cfg_path(_MODEL_PERTURB_SECTION, "folder", Path("data/pred/eval")))
    perturb_file_name: str = _cfg_str(_MODEL_PERTURB_SECTION, "file", "perturbation.csv")

    seq_len: int = _cfg_int(_MODEL_TRAINING_SECTION, "seq_len", 32)
    windows_per_client: int = _cfg_int(_MODEL_TRAINING_SECTION, "windows_per_client", 4)
    clients_per_batch: int = _cfg_int(_MODEL_TRAINING_SECTION, "clients_per_batch", 8)
    windows_per_pair: int = _cfg_int(_MODEL_TRAINING_SECTION, "windows_per_pair", 2)

    epochs: int = _cfg_int(_MODEL_TRAINING_SECTION, "epochs", 30)
    mask_prob: float = _cfg_float(_MODEL_TRAINING_SECTION, "mask_prob", 0.15)
    contrastive_weight: float = _cfg_float(_MODEL_TRAINING_SECTION, "contrastive_weight", 0.5)
    lr: float = _cfg_float(_MODEL_TRAINING_SECTION, "lr", 3e-4)
    weight_decay: float = _cfg_float(_MODEL_TRAINING_SECTION, "weight_decay", 0.01)  # regolarizzazione L2 di AdamW (anti-overfitting)
    lr_gamma: float = _cfg_float(_MODEL_TRAINING_SECTION, "lr_gamma", 0.95)      # fattore di decadimento esponenziale del LR, per epoca
    grad_clip: float = _cfg_float(_MODEL_TRAINING_SECTION, "grad_clip", 1.0)

    val_frac: float = _cfg_float(_MODEL_TRAINING_SECTION, "val_frac", 0.2)   # fraction of client_ids held out for validation (0 disables)
    val_every: int = _cfg_int(_MODEL_TRAINING_SECTION, "val_every", 1)      # run validation every N epochs (0 disables)

    early_stopping_patience: int = _cfg_int(_MODEL_TRAINING_SECTION, "early_stopping_patience", 5)    # stop after N validation checks with no val-loss improvement (0 disables)
    early_stopping_min_delta: float = _cfg_float(_MODEL_TRAINING_SECTION, "early_stopping_min_delta", 0.0)  # minimum val-loss decrease to count as an improvement

    log_every: int = _cfg_int(_MODEL_TRAINING_SECTION, "log_every", 5)
    device: str | None = _cfg_optional_str(_MODEL_TRAINING_SECTION, "device", None)
    seed: int = _cfg_int(_MODEL_TRAINING_SECTION, "seed", 0)

    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        self.train_path = Path(self.train_path)
        self.pred_path = Path(self.pred_path)
        self.perturb_path = Path(self.perturb_path)
        self.ckpt_dir = Path(self.ckpt_dir)
        if not self.pred_file_name:
            raise ValueError("`model.hierTransformer.paths.pred_file_name` cannot be empty.")

    @property
    def batch_size(self) -> int:
        return self.clients_per_batch * self.windows_per_pair
