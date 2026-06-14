"""Simple, self-contained synthetic transaction dataset.

Unlike :mod:`vanilla`/`coherent` (which share the feature-rich
``SyntheticTransactionDatasetCore``), this experiment is intentionally minimal:
every cluster is defined explicitly by its amount mean/variance, transaction-count
mean/variance, preferred merchants, preferred cocau, and a single temporal
parameter — the inter-transaction **gap rate** ``λ``.

Temporal model (the cluster discriminator lives entirely in the gap):
  * The hour-of-day and second-of-hour are **not** modelled — they are drawn
    uniformly, so the absolute time-of-day carries no cluster signal.
  * The gap between consecutive transactions is drawn from an exponential
    distribution ``gap ~ Exponential(λ)`` (in days), where ``λ`` is the only
    per-cluster temporal parameter. Timestamps are the cumulative sum of those
    gaps, so the model's ``delta_t`` is exponential with a cluster-specific rate.

Merchant logic mirrors ``coherent``: the merchant carries its own cocau (intersected
with the cluster's preferred codes) and the transaction amount is the mean of the
cluster and merchant amount distributions, so its sign emerges naturally from the
underlying normals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...constant import DATA_CONFIG
from ..generators.merchant import MerchantConfig, _merchant_catalog
from ..utils.config import SPLITS, DatasetConfig
from ..utils.entities import Merchant
from .common import BaseSyntheticDataset


@dataclass(frozen=True)
class SimpleClientType:
    """Explicit, fully-specified cluster definition for the simple experiment.

    Input:
        name: cluster label (written to the `cluster` column).
        cluster_prob: nominal share of clients in this cluster (informational).
        n_clients: explicit number of clients belonging to this cluster.
        amount_mean: mean of the cluster amount distribution.
        amount_std: standard deviation of the cluster amount distribution.
        n_tx_mean: mean number of transactions per client.
        n_tx_std: standard deviation of the per-client transaction count.
        merchants: preferred merchant pool for this cluster.
        preferred_cocau: preferred categorical codes (intersected with merchant codes).
        gap_lambda: rate ``λ`` (per day) of the exponential inter-transaction gap;
            the mean gap is ``1 / gap_lambda`` days. This is the ONLY temporal
            differentiator between clusters.
        negative_only: if True, every transaction of this cluster is forced to be
            a debit (amount < 0); otherwise the sign emerges from the underlying
            cluster/merchant normals.
    Output:
        Frozen dataclass instance.
    What it does:
        Carries every parameter needed to generate one client segment with no
        hidden defaults or shared machinery.
    """

    name: str
    cluster_prob: float
    n_clients: int
    amount_mean: float
    amount_std: float
    n_tx_mean: float
    n_tx_std: float
    merchants: list[Merchant]
    preferred_cocau: tuple[int, ...]
    gap_lambda: float
    negative_only: bool = False


def simple_cluster_types(merchants: MerchantConfig) -> list[SimpleClientType]:
    """Return the default explicit clusters for the simple experiment.

    Input:
        merchants: merchant configuration providing themed name pools.
    Output:
        List of SimpleClientType objects (priors sum is normalized at use time).
    What it does:
        Reuses the merchant catalog (amount stats + cocau per theme) to build the
        merchant pools, and assigns each cluster a distinct amount level,
        transaction volume, and inter-transaction gap rate ``λ``.
    """
    catalog = _merchant_catalog(merchants)

    def pool(names: list[str]) -> list[Merchant]:
        return [catalog[n] for n in names]

    return [
        SimpleClientType(
            name="famiglia_giorno",
            cluster_prob=0.35,
            n_clients=1400,
            amount_mean=45.0,
            amount_std=55.0,
            n_tx_mean=90.0,
            n_tx_std=25.0,
            merchants=pool(merchants.groceries + merchants.payments + merchants.utilities),
            preferred_cocau=(8, 19, 36, 54, 72, 91, 101, 110, 129, 148, 167, 186, 210, 238, 260),
            gap_lambda=0.25,  # mean gap ≈ 4 days (frequent)
        ),
        SimpleClientType(
            name="giovane_sera",
            cluster_prob=0.30,
            n_clients=1200,
            amount_mean=32.0,
            amount_std=45.0,
            n_tx_mean=70.0,
            n_tx_std=20.0,
            merchants=pool(merchants.payments + merchants.shopping + ["Starbucks"]),
            preferred_cocau=(7, 11, 23, 24, 39, 45, 57, 66, 76, 88, 98, 101, 120, 123, 140, 161, 182, 203, 224),
            gap_lambda=0.1428,  # mean gap ≈ 7 days
        ),
        SimpleClientType(
            name="altospendente",
            cluster_prob=0.20,
            n_clients=800,
            amount_mean=200.0,
            amount_std=220.0,
            n_tx_mean=50.0,
            n_tx_std=15.0,
            merchants=pool(merchants.shopping + merchants.travel),
            preferred_cocau=(33, 52, 71, 90, 109, 120, 128, 147, 166, 185, 210, 230, 252, 274, 296),
            gap_lambda=0.0833,  # mean gap ≈ 12 days (infrequent)
            negative_only=True,  # this cluster makes debits only (amount < 0)
        ),
        SimpleClientType(
            name="mattiniero_utenze",
            cluster_prob=0.15,
            n_clients=600,
            amount_mean=60.0,
            amount_std=60.0,
            n_tx_mean=60.0,
            n_tx_std=18.0,
            merchants=pool(merchants.utilities + merchants.groceries),
            preferred_cocau=(8, 19, 36, 54, 72, 91, 101, 110, 129, 148, 167, 186, 210, 238, 260, 282),
            gap_lambda=0.1111,  # mean gap ≈ 9 days
        ),
    ]


class SimpleSyntheticTransactionDataset(BaseSyntheticDataset):
    """Build a simple synthetic transaction dataset from explicit clusters."""

    experiment = "simple_spatial"

    def __init__(
        self,
        config: DatasetConfig,
        client_types: list[SimpleClientType] | None = None,
        split: str = "train",
    ) -> None:
        """Initialize the simple dataset builder.

        Input:
            config: full dataset configuration (sampling/time/output are used).
            client_types: optional explicit clusters; defaults from merchant config.
            split: sampling split to draw (`train` or `pred`); selects the per-split
                seed so the two files are independent draws of the same clusters.
        Output:
            None.
        What it does:
            Stores config, resolves clusters, and seeds the shared RNG.
        """
        self.config = config
        self.split = split
        self.data_config = DATA_CONFIG
        self.client_types = (
            list(client_types)
            if client_types is not None
            else simple_cluster_types(config.merchants)
        )
        if not self.client_types:
            raise ValueError("`client_types` cannot be empty.")
        self._rng = np.random.default_rng(config.sampling_for(split).seed)

        time_cfg = config.time
        self._ts_base = int(time_cfg.ts_base)
        self._day = int(time_cfg.day)
        self._n_days = max(1, int(time_cfg.ts_range) // self._day)

    def _cocau_pools(self, c: SimpleClientType) -> list[np.ndarray]:
        """Per-merchant cocau pools: intersection with cluster, else merchant codes."""
        preferred = set(c.preferred_cocau)
        pools: list[np.ndarray] = []
        for m in c.merchants:
            inter = tuple(sorted(set(m.cocau) & preferred))
            pools.append(np.array(inter if inter else m.cocau, dtype=np.int64))
        return pools

    def _sample_timestamps(self, c: SimpleClientType, n_tx: int) -> np.ndarray:
        """Sample timestamps whose gaps are exponential with the cluster rate.

        Input:
            c: cluster definition providing the inter-transaction gap rate.
            n_tx: number of transactions to place.
        Output:
            Sorted int64 array of Unix-second timestamps of length ``n_tx``.
        What it does:
            Draws inter-transaction gaps (in days) from ``Exponential(1/gap_lambda)``
            and cumulates them from a random start day. The hour-of-day and
            second-of-hour are drawn uniformly and are NOT informative; only the
            cumulative gap (the model's ``delta_t``) carries the cluster signal.
        """
        rng = self._rng
        mean_gap_days = 1.0 / max(c.gap_lambda, 1e-9)

        # Inter-arrival gaps in days; the first transaction starts the sequence.
        gaps_days = rng.exponential(mean_gap_days, size=n_tx)
        gaps_days[0] = 0.0
        day_offsets = np.floor(np.cumsum(gaps_days)).astype(np.int64)

        # Random absolute start so dow/day-of-month decompositions stay uninformative.
        start_day = int(rng.integers(0, self._n_days))
        days = start_day + day_offsets

        hours = rng.integers(0, 24, size=n_tx)
        secs = rng.integers(0, 3600, size=n_tx)
        ts = self._ts_base + days * self._day + hours * 3600 + secs
        return np.sort(ts.astype(np.int64))

    def build(self) -> pd.DataFrame:
        """Build the full simple synthetic transaction dataset.

        Input:
            None.
        Output:
            Pandas DataFrame with the canonical transaction columns, sorted by
            (client_id, timestamp).
        What it does:
            Gives each cluster its explicit client count, samples a per-client
            transaction count, then vectorially samples timestamps (exponential
            gaps), merchants, amounts, and cocau for that client.
        """
        rng = self._rng

        # Each cluster contributes exactly `n_clients` clients (total is their sum).
        assignments = np.repeat(
            np.arange(len(self.client_types)),
            [int(c.n_clients) for c in self.client_types],
        )
        rng.shuffle(assignments)
        n_clients = int(assignments.shape[0])

        # Precompute per-cluster sampling material once.
        cocau_pools = [self._cocau_pools(c) for c in self.client_types]
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

            ts = self._sample_timestamps(c, n_tx)

            m_idx = rng.integers(0, len(c.merchants), size=n_tx)
            cluster_sample = rng.normal(c.amount_mean, c.amount_std, size=n_tx)
            merchant_sample = rng.normal(merch_means[k][m_idx], merch_stds[k][m_idx])
            amounts = 0.5 * (cluster_sample + merchant_sample)
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

    def _output_path(self) -> Path:
        """Resolve the split-suffixed simple-spatial output path from config."""
        return self.config.output.split_path(self.experiment, self.split)


def generate(
    config: DatasetConfig | None = None,
    client_types: list[SimpleClientType] | None = None,
) -> dict[str, Path]:
    """Generate and save the simple-spatial synthetic dataset for every split.

    Input:
        config: optional explicit dataset configuration.
        client_types: optional explicit cluster definitions.
    Output:
        Mapping of split name (`train`/`pred`) to the generated file path.
    What it does:
        Resolves the cluster definitions once, then materialises one file per split
        (`train` and `pred`) — independent draws from the same clusters.
    """
    resolved_config = config or DatasetConfig()
    shared_client_types = (
        client_types
        if client_types is not None
        else simple_cluster_types(resolved_config.merchants)
    )
    paths: dict[str, Path] = {}
    for split in SPLITS:
        std = SimpleSyntheticTransactionDataset(
            config=resolved_config,
            client_types=shared_client_types,
            split=split,
        )
        paths[split] = std.generate_and_save()
    return paths


if __name__ == "__main__":
    generate()
