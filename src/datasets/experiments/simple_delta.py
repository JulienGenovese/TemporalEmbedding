"""Delta-only validation dataset.

The dual of :mod:`simple_timing`, but targeting the signal the model *does*
consume — the inter-transaction gap ``delta_t``. Four clusters are **identical in
everything that is embedded as a value** (amount distribution, merchant pool,
cocau pool, transaction volume) **and** in the absolute time-of-day / day-of-week
(drawn uniformly, hence non-informative). They differ *only* in the **rate** ``λ``
of the exponential inter-transaction gap.

    Cluster A — "veloce":      gap_lambda 0.5000  → mean gap ≈ 2 days.
    Cluster B — "medio":       gap_lambda 0.1429  → mean gap ≈ 7 days.
    Cluster C — "lento":       gap_lambda 0.0500  → mean gap ≈ 20 days.
    Cluster D — "lentissimo":  gap_lambda 0.0200  → mean gap ≈ 50 days.

Because the spending profile is defined once on the dataset (a single
``SharedProfile``) and shared by every cluster, and the hour-of-day / day-of-week
carry no signal, the *only* thing separating the clusters is the distribution of
``delta_t``. This is a clean pass/fail probe of
:class:`TimeAwarePositionalEncoding`: a model that recovers the clusters is
genuinely exploiting the time-gap channel; one that collapses them is ignoring it.

Amount and cocau are drawn exactly like ``simple_spatial`` (mean of the
shared-cluster and merchant amount normals; cocau intersected with merchant
codes), so the value distributions match the rest of the synthetic suite.
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
class SharedProfile:
    """Spending profile shared by *every* delta cluster.

    Input:
        amount_mean: mean of the cluster amount distribution (shared).
        amount_std: standard deviation of the cluster amount distribution.
        n_tx_mean: mean number of transactions per client.
        n_tx_std: standard deviation of the per-client transaction count.
        merchants: merchant pool (shared across all clusters).
        preferred_cocau: preferred categorical codes (intersected with merchant codes).
    Output:
        Frozen dataclass instance.
    What it does:
        Holds the non-temporal half of the data-generating process so that it is
        physically impossible for any two clusters to differ on amount/merchant/
        cocau — they can only differ in their :class:`DeltaSignature`.
    """

    amount_mean: float
    amount_std: float
    n_tx_mean: float
    n_tx_std: float
    merchants: list[Merchant]
    preferred_cocau: tuple[int, ...]


@dataclass(frozen=True)
class DeltaSignature:
    """The *only* thing that distinguishes one delta cluster from another.

    Input:
        name: cluster label (written to the `cluster` column).
        n_clients: explicit number of clients belonging to this cluster.
        gap_lambda: rate ``λ`` (per day) of the exponential inter-transaction gap;
            the mean gap is ``1 / gap_lambda`` days. This is the ONLY differentiator
            between clusters.
    Output:
        Frozen dataclass instance.
    What it does:
        Fully specifies a cluster's gap fingerprint and how many clients it owns;
        carries no amount/merchant/time-of-day information by design.
    """

    name: str
    n_clients: int
    gap_lambda: float


def shared_profile(merchants: MerchantConfig) -> SharedProfile:
    """Build the single spending profile shared by all delta clusters.

    Input:
        merchants: merchant configuration providing themed name pools.
    Output:
        A :class:`SharedProfile` reused by every :class:`DeltaSignature`.
    What it does:
        Picks a fixed, mixed merchant pool (groceries + payments) and a fixed
        amount/volume level so that the clusters are value-indistinguishable.
    """
    catalog = _merchant_catalog(merchants)
    pool = [catalog[n] for n in merchants.groceries + merchants.payments]
    return SharedProfile(
        amount_mean=50.0,
        amount_std=55.0,
        n_tx_mean=120.0,
        n_tx_std=25.0,
        merchants=pool,
        preferred_cocau=(8, 14, 27, 41, 58, 73, 89, 101, 120, 133, 149, 168),
    )


def delta_signatures() -> list[DeltaSignature]:
    """Return the four delta-only clusters of the validation experiment.

    Input:
        None.
    Output:
        List with exactly four :class:`DeltaSignature` objects.
    What it does:
        Assigns four well-separated exponential gap rates so the ``delta_t``
        distributions are clearly distinct while everything else is shared.
    """
    return [
        DeltaSignature(name="veloce", n_clients=1000, gap_lambda=0.5),        # ≈ 2 days
        DeltaSignature(name="medio", n_clients=1000, gap_lambda=0.1429),      # ≈ 7 days
        DeltaSignature(name="lento", n_clients=1000, gap_lambda=0.05),        # ≈ 20 days
        DeltaSignature(name="lentissimo", n_clients=1000, gap_lambda=0.02),   # ≈ 50 days
    ]


class DeltaSyntheticTransactionDataset(BaseSyntheticDataset):
    """Build a delta-only synthetic dataset (clusters differ solely in gap rate)."""

    experiment = "simple_delta"

    def __init__(
        self,
        config: DatasetConfig,
        profile: SharedProfile | None = None,
        signatures: list[DeltaSignature] | None = None,
        split: str = "train",
    ) -> None:
        """Initialize the delta dataset builder.

        Input:
            config: full dataset configuration (sampling/time/output are used).
            profile: optional shared spending profile; defaults from merchant config.
            signatures: optional explicit delta clusters; defaults to the four-cluster probe.
            split: sampling split to draw (`train` or `pred`); selects the per-split
                seed so the two files are independent draws of the same signatures.
        Output:
            None.
        What it does:
            Stores config, resolves the shared profile and delta signatures, and
            seeds the shared RNG.
        """
        self.config = config
        self.split = split
        self.data_config = DATA_CONFIG
        self.profile = profile if profile is not None else shared_profile(config.merchants)
        self.signatures = list(signatures) if signatures is not None else delta_signatures()
        if not self.signatures:
            raise ValueError("`signatures` cannot be empty.")
        self._rng = np.random.default_rng(config.sampling_for(split).seed)

        time_cfg = config.time
        self._ts_base = int(time_cfg.ts_base)
        self._day = int(time_cfg.day)
        self._n_days = max(1, int(time_cfg.ts_range) // self._day)

    def _cocau_pools(self) -> list[np.ndarray]:
        """Per-merchant cocau pools for the shared profile.

        Input:
            None.
        Output:
            List of int64 arrays, one per shared merchant.
        What it does:
            Intersects each merchant's allowed codes with the profile's preferred
            codes (falling back to the merchant codes when the intersection is empty).
        """
        preferred = set(self.profile.preferred_cocau)
        pools: list[np.ndarray] = []
        for m in self.profile.merchants:
            inter = tuple(sorted(set(m.cocau) & preferred))
            pools.append(np.array(inter if inter else m.cocau, dtype=np.int64))
        return pools

    def _sample_timestamps(self, sig: DeltaSignature, n_tx: int) -> np.ndarray:
        """Sample timestamps whose gaps are exponential with the cluster rate.

        Input:
            sig: signature providing the inter-transaction gap rate ``λ``.
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
        mean_gap_days = 1.0 / max(sig.gap_lambda, 1e-9)

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
        """Build the delta-only synthetic transaction dataset.

        Input:
            None.
        Output:
            Pandas DataFrame with the canonical transaction columns, sorted by
            (client_id, timestamp).
        What it does:
            Assigns each cluster its client count, then for every client samples a
            transaction count and draws timestamps from that cluster's exponential
            gap rate while drawing amount/merchant/cocau from the *shared* profile —
            so the clusters are separable by ``delta_t`` alone.
        """
        rng = self._rng
        profile = self.profile

        assignments = np.repeat(
            np.arange(len(self.signatures)),
            [int(s.n_clients) for s in self.signatures],
        )
        rng.shuffle(assignments)
        n_clients = int(assignments.shape[0])

        # Shared (cluster-independent) sampling material — computed once.
        cocau_pools = self._cocau_pools()
        merch_names = np.array([m.name for m in profile.merchants], dtype=object)
        merch_means = np.array([m.amount_mean for m in profile.merchants], dtype=float)
        merch_stds = np.sqrt(np.array([m.amount_variance for m in profile.merchants], dtype=float))

        client_col: list[np.ndarray] = []
        cluster_col: list[np.ndarray] = []
        ts_col: list[np.ndarray] = []
        amount_col: list[np.ndarray] = []
        merchant_col: list[np.ndarray] = []
        cocau_col: list[np.ndarray] = []

        for client_id in range(n_clients):
            k = int(assignments[client_id])
            sig = self.signatures[k]
            n_tx = max(1, int(round(rng.normal(profile.n_tx_mean, profile.n_tx_std))))

            # --- timing: the ONLY cluster-dependent draw (gap rate) ---
            ts = self._sample_timestamps(sig, n_tx)

            # --- value: drawn from the SHARED profile, identical across clusters ---
            m_idx = rng.integers(0, len(profile.merchants), size=n_tx)
            cluster_sample = rng.normal(profile.amount_mean, profile.amount_std, size=n_tx)
            merchant_sample = rng.normal(merch_means[m_idx], merch_stds[m_idx])
            amounts = 0.5 * (cluster_sample + merchant_sample)

            cocau = np.empty(n_tx, dtype=np.int64)
            for mi in np.unique(m_idx):
                mask = m_idx == mi
                cocau[mask] = rng.choice(cocau_pools[mi], size=int(mask.sum()))

            client_col.append(np.full(n_tx, client_id, dtype=np.int64))
            cluster_col.append(np.full(n_tx, sig.name, dtype=object))
            ts_col.append(ts.astype(np.int64))
            amount_col.append(amounts.astype(float))
            merchant_col.append(merch_names[m_idx])
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
        """Resolve the split-suffixed simple-delta output path from config."""
        return self.config.output.split_path(self.experiment, self.split)


def generate(
    config: DatasetConfig | None = None,
    profile: SharedProfile | None = None,
    signatures: list[DeltaSignature] | None = None,
) -> dict[str, Path]:
    """Generate and save the simple-delta validation dataset for every split.

    Input:
        config: optional explicit dataset configuration.
        profile: optional shared spending profile.
        signatures: optional explicit delta cluster definitions.
    Output:
        Mapping of split name (`train`/`pred`) to the generated file path.
    What it does:
        Resolves the shared profile/signatures once, then materialises one file per
        split (`train` and `pred`) — independent draws from the same signatures.
    """
    resolved_config = config or DatasetConfig()
    shared_profile_obj = profile if profile is not None else shared_profile(resolved_config.merchants)
    shared_signatures = signatures if signatures is not None else delta_signatures()
    paths: dict[str, Path] = {}
    for split in SPLITS:
        std = DeltaSyntheticTransactionDataset(
            config=resolved_config,
            profile=shared_profile_obj,
            signatures=shared_signatures,
            split=split,
        )
        paths[split] = std.generate_and_save()
    return paths


if __name__ == "__main__":
    generate()
