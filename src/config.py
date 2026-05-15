"""Training-time constants for :mod:`src.train`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Columns that ARE embedded by the TransactionEncoder.
FEATURE_COLS: list[str] = [
    "importo", "saldo_post", "merchant", "mcc",
    "canale", "macro_tipo", "sotto_tipo", "divisa",
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
class TrainingArgs:
    """All hyper-parameters and I/O paths for end-to-end pre-training."""

    csv_path: Path = Path("data") / "transactions.csv"
    ckpt_dir: Path = Path("checkpoints")

    seq_len: int = 32
    windows_per_client: int = 4
    clients_per_batch: int = 8
    windows_per_pair: int = 2

    epochs: int = 2
    mask_prob: float = 0.15
    contrastive_weight: float = 0.5
    lr: float = 3e-4
    grad_clip: float = 1.0

    val_frac: float = 0.2   # fraction of client_ids held out for validation (0 disables)
    val_every: int = 1      # run validation every N epochs (0 disables)

    log_every: int = 5
    device: str | None = None
    seed: int = 0

    @property
    def batch_size(self) -> int:
        return self.clients_per_batch * self.windows_per_pair
