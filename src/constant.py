

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """Canonical column names used across dataset generation and training.

    - ``client_col`` groups rows into clients (used by the sampler to
      build InfoNCE positive pairs).
    - ``cluster_col`` stores the synthetic cluster label used to generate
      each client.
    - ``timestamp_col`` is read only to compute the per-window
      ``delta_t`` tensor.
    - ``delta_t_col`` is the derived per-window time-gap (seconds)
      that feeds :class:`TimeAwarePositionalEncoding` directly.
    """

    client_col: str = "client_id"
    cluster_col: str = "cluster"
    timestamp_col: str = "timestamp"
    delta_t_col: str = "delta_t"
    amount_col: str = "importo"
    merchant_col: str = "merchant"
    cocau_col: str = "cocau"

    @property
    def feature_cols(self) -> list[str]:
        return [self.amount_col, self.merchant_col, self.cocau_col]

    @property
    def transaction_cols(self) -> list[str]:
        return [self.client_col, self.cluster_col, self.timestamp_col, *self.feature_cols]

    @property
    def transaction_sort_cols(self) -> list[str]:
        return [self.client_col, self.timestamp_col]


DATA_CONFIG = DataConfig()
FEATURE_COLS: list[str] = DATA_CONFIG.feature_cols
TRANSACTION_COLS: list[str] = DATA_CONFIG.transaction_cols
