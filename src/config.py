"""Training-time constants for :mod:`src.train`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Columns that ARE embedded by the TransactionEncoder.
FEATURE_COLS: list[str] = [
    "importo", "merchant", "mcc", "macro_tipo",
]


@dataclass
class DataConfig:
    """Column names that the data pipeline uses but does **not** embed.

    - ``client_col`` groups rows into clients (used by the sampler to
      build InfoNCE positive pairs).
    - ``timestamp_col`` is read only to compute the per-window
      ``delta_t`` tensor.
    - ``delta_t_col`` is the derived per-window time-gap (seconds)
      that feeds :class:`TimeAwarePositionalEncoding` directly.
    """

    client_col: str = "client_id"
    timestamp_col: str = "timestamp"
    delta_t_col: str = "delta_t"


@dataclass
class ModelConfig:
    """Architecture hyper-parameters for :class:`src.model.EmbeddingModel`.

    These are forwarded as keyword arguments to ``TransactionTransformer``.
    ``pretrain`` and ``use_gradient_checkpointing`` are intentionally left
    out: the first is always ``True`` during pre-training, the second is
    derived from the runtime device in :mod:`src.train`.
    """

    d_field: int = 64          # dimensione dell'embedding di ogni singolo campo
    d_model: int = 128         # dimensione interna del modello (transazione + sequenza)
    n_frequencies: int = 16    # n. di frequenze sin/cos del NumericEncoder
    field_n_layers: int = 2    # n. di layer del FieldTransformer (attenzione intra-transazione)
    field_n_heads: int = 4     # n. di teste di attenzione del FieldTransformer
    seq_n_layers: int = 4      # n. di layer del SequenceTransformer (attenzione sulla sequenza)
    seq_n_heads: int = 8       # n. di teste di attenzione del SequenceTransformer
    dim_feedforward: int = 512 # dimensione della FFN interna ai blocchi Transformer
    dropout: float = 0.1       # probabilità di dropout in tutti i blocchi Transformer


@dataclass
class TrainingConfig:
    """All hyper-parameters and I/O paths for end-to-end pre-training."""

    csv_path: Path = Path("data") / "transactions.csv"
    ckpt_dir: Path = Path("checkpoints")

    seq_len: int = 32
    windows_per_client: int = 4
    clients_per_batch: int = 8
    windows_per_pair: int = 2

    epochs: int = 30
    mask_prob: float = 0.15
    contrastive_weight: float = 0.5
    lr: float = 3e-4
    weight_decay: float = 0.01  # regolarizzazione L2 di AdamW (anti-overfitting)
    lr_gamma: float = 0.95      # fattore di decadimento esponenziale del LR, per epoca
    grad_clip: float = 1.0

    val_frac: float = 0.2   # fraction of client_ids held out for validation (0 disables)
    val_every: int = 1      # run validation every N epochs (0 disables)

    early_stopping_patience: int = 5    # stop after N validation checks with no val-loss improvement (0 disables)
    early_stopping_min_delta: float = 0.0  # minimum val-loss decrease to count as an improvement

    log_every: int = 5
    device: str | None = None
    seed: int = 0

    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def batch_size(self) -> int:
        return self.clients_per_batch * self.windows_per_pair
