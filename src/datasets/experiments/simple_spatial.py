"""Simple, self-contained synthetic transaction dataset.

This experiment is intentionally minimal and fully self-contained: every cluster
is defined explicitly by its amount mean/variance, transaction-count
mean/variance, preferred merchants, and preferred cocau.

Temporal model:
  * Day, hour, and second are drawn uniformly at random.
  * There is no cluster-specific temporal parameter: ``delta_t`` is random and
    non-informative in this experiment.

Each merchant carries its own cocau (intersected with the cluster's preferred
codes) and the transaction amount is the mean of the cluster and merchant amount
distributions, so its sign emerges naturally from the underlying normals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import config

from src.data_schema import DATA_CONFIG
from src.datasets.experiments.abc import SyntheticExperiment
from src.datasets.experiments.specs import (
    Merchant,
    MerchantConfig,
    SimpleClientType,
    _merchant_catalog,
    load_timing_config,
)

_SECTION = "synthetic.simple_spatial"

def simple_cluster_types(merchants: MerchantConfig) -> list[SimpleClientType]:
    """Return the explicit simple-spatial cluster definitions."""
    catalog = _merchant_catalog(merchants)

    def pool(names: list[str]) -> list[Merchant]:
        return [catalog[n] for n in names]

    return [
        SimpleClientType(
            name="cluster_1",
            n_clients=1000,
            amount_mean=60.0,
            amount_std=55.0,
            n_tx_mean=120.0,
            n_tx_std=30,
            merchants=pool(merchants.utilities + merchants.common_merchants),
            preferred_cocau=(8, 19, 36, 54, 72, 91, 101, 110, 129, 148, 167, 186, 210, 238, 260),
        ),
        SimpleClientType(
            name="cluster_2",
            n_clients=1000,
            amount_mean=60.0,
            amount_std=60.0,
            n_tx_mean=120.0,
            n_tx_std=60,
            merchants=pool(merchants.utilities + merchants.common_merchants),
            #different preferred cocau from cluster_1, but same merchants and similar amount distribution, so that the clusters are not trivially separable.
            preferred_cocau=(7, 11, 23, 24, 39, 45, 57, 66, 76, 88, 98, 101, 120, 123, 140, 161, 182, 203, 224),
        ),
        SimpleClientType(
            name="cluster_3",
            n_clients=1000,
            amount_mean=60.0,
            amount_std=45.0,
            n_tx_mean=120.0,
            n_tx_std=50,
            # similar amount but different merchants from cluster_1 and cluster_2, so that the clusters are not trivially separable.
            merchants=pool(merchants.payments + merchants.shopping),
            preferred_cocau=(7, 11, 23, 24, 39, 45, 57, 66, 76, 88, 98, 101, 120, 123, 140, 161, 182, 203, 224),
        ),
        SimpleClientType(
            name="cluster_4",
            n_clients=1000,
            amount_mean=100.0,
            amount_std=100.0,
            n_tx_mean=120.0,
            n_tx_std=10,
            # different amount distribution and different merchants from cluster_1, cluster_2 and cluster_3, so that the clusters are not trivially separable.
            merchants=pool(merchants.utilities + merchants.common_merchants),
            preferred_cocau=(7, 11, 23, 24, 39, 45, 57, 66, 76, 88, 98, 101, 120, 123, 140, 161, 182, 203, 224),
        ),

    ]


class SimpleSpatialExperiment(SyntheticExperiment):
    """Build a simple synthetic transaction dataset from explicit clusters."""

    @property
    def experiment(self) -> str:
        return "simple_spatial"

    @property
    def output_section(self) -> str:
        return _SECTION

    @staticmethod
    def simple_cluster_types(merchants: MerchantConfig) -> list[SimpleClientType]:
        """Return the fixed cluster definitions used by the experiment."""
        return simple_cluster_types(merchants)

    def __init__(self) -> None:
        self.data_config = DATA_CONFIG
        self.client_types = self.simple_cluster_types(MerchantConfig())
        if not self.client_types:
            raise ValueError("`client_types` cannot be empty.")
        self._amount_blend_factor = config.get(
            _SECTION,
            "amount_blend_factor",
            value_type=float,
        )
        self.timing = load_timing_config()

    def sample_timestamps(
        self,
        rng: np.random.Generator,
        n_tx: int,
    ) -> np.ndarray:
        """Sample timestamps with non-informative random day/hour/second."""
        days = rng.integers(0, self.timing.n_days, size=n_tx)
        hours = rng.integers(0, 24, size=n_tx)
        secs = rng.integers(0, 3600, size=n_tx)
        ts = self.timing.ts_base + days * self.timing.day + hours * 3600 + secs
        return np.sort(ts.astype(np.int64))

    def build(self, seed: int) -> pd.DataFrame:
        """Build one simple-spatial synthetic split."""
        rng = np.random.default_rng(seed)

        # Each cluster contributes exactly `n_clients` clients (total is their sum).
        assignments = np.repeat(
            np.arange(len(self.client_types)),
            [int(c.n_clients) for c in self.client_types],
        )
        rng.shuffle(assignments)
        n_clients = int(assignments.shape[0])

        # Precompute per-cluster sampling material once.
        cocau_pools = [self._cocau_pools(c.merchants, c.preferred_cocau) for c in self.client_types]
        merch_names = [np.array([m.name for m in c.merchants], dtype=object) for c in self.client_types]
        merch_means = [np.array([m.amount_mean for m in c.merchants], dtype=float) for c in self.client_types]
        merch_stds = [
            np.sqrt(np.array([m.amount_variance for m in c.merchants], dtype=float))
            for c in self.client_types
        ]

        client_col: list[np.ndarray] = []
        cluster_col: list[np.ndarray] = []
        ts_col: list[np.ndarray] = []
        amount_col: list[np.ndarray] = []
        merchant_col: list[np.ndarray] = []
        cocau_col: list[np.ndarray] = []

        for client_id in range(n_clients):
            k = int(assignments[client_id])
            c = self.client_types[k]
            n_tx = int(round(rng.normal(c.n_tx_mean, c.n_tx_std)))
            n_tx = max(1, n_tx)

            ts = self.sample_timestamps(rng, n_tx)

            m_idx = rng.integers(0, len(c.merchants), size=n_tx)
            cluster_sample = rng.normal(c.amount_mean, c.amount_std, size=n_tx)
            merchant_sample = rng.normal(merch_means[k][m_idx], merch_stds[k][m_idx])
            amounts = self._amount_blend_factor * (cluster_sample + merchant_sample)
            if c.negative_only:
                amounts = -np.abs(amounts)

            pools = cocau_pools[k]
            cocau = np.empty(n_tx, dtype=np.int64)
            for mi in np.unique(m_idx):
                mask = m_idx == mi
                cocau[mask] = rng.choice(pools[mi], size=int(mask.sum()))

            client_col.append(np.full(n_tx, client_id, dtype=np.int64))
            cluster_col.append(np.full(n_tx, c.name, dtype=object))
            ts_col.append(ts.astype(np.int64))
            amount_col.append(amounts.astype(float))
            merchant_col.append(merch_names[k][m_idx])
            cocau_col.append(cocau)

        dc = self.data_config
        df = pd.DataFrame(
            {
                dc.client_col: np.concatenate(client_col),
                dc.cluster_col: np.concatenate(cluster_col),
                dc.timestamp_col: np.concatenate(ts_col),
                dc.amount_col: np.concatenate(amount_col),
                dc.merchant_col: np.concatenate(merchant_col),
                dc.cocau_col: np.concatenate(cocau_col),
            },
            columns=dc.transaction_cols,
        )
        return df.sort_values(dc.transaction_sort_cols).reset_index(drop=True)