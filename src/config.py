"""Training-time constants for :mod:`src.train`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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

    log_every: int = 5
    device: str | None = None
    seed: int = 0

    @property
    def batch_size(self) -> int:
        return self.clients_per_batch * self.windows_per_pair
