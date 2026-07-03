"""Delta-only validation dataset.

Targets the signal the model *does*
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
:class:`TimeDeltaEncoding`: a model that recovers the clusters is
genuinely exploiting the time-gap channel; one that collapses them is ignoring it.

Amount and cocau are drawn exactly like ``simple_spatial`` (mean of the
shared-cluster and merchant amount normals; cocau intersected with merchant
codes), so the value distributions match the rest of the synthetic suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_schema import DATA_CONFIG
from src.datasets.experiments.abc import SyntheticExperiment
from src.datasets.experiments.specs import Merchant, MerchantConfig, _merchant_catalog, load_timing_config

_SECTION = "synthetic.simple_delta"

@dataclass(frozen=True)
class SharedProfile:
    """Spending profile shared by every delta cluster."""

    amount_mean: float
    amount_std: float
    n_tx_mean: float
    n_tx_std: float
    merchants: list[Merchant]
    preferred_cocau: tuple[int, ...]


@dataclass(frozen=True)
class DeltaSignature:
    """Cluster signature: only the inter-transaction gap rate differs."""

    name: str
    n_clients: int
    gap_lambda: float


def shared_profile(merchants: MerchantConfig) -> SharedProfile:
    """Build the single value profile shared by all delta clusters."""
    catalog = _merchant_catalog(merchants)
    pool = [catalog[n] for n in merchants.groceries + merchants.payments + merchants.travel + merchants.shopping]
    return SharedProfile(
        amount_mean=50.0,
        amount_std=55.0,
        n_tx_mean=150.0,
        n_tx_std=70,
        merchants=pool,
        preferred_cocau=(8, 14, 27, 41, 58, 73, 89, 101, 120, 133, 149, 168),
    )


def delta_signatures() -> list[DeltaSignature]:
    """Return four clusters separated only by gap rate."""
    return [
        DeltaSignature(name="veloce", n_clients=1000, gap_lambda=0.5),        # ≈ 2 days
        DeltaSignature(name="medio", n_clients=1000, gap_lambda=0.1429),      # ≈ 7 days
        DeltaSignature(name="lento", n_clients=1000, gap_lambda=0.05),        # ≈ 20 days
        DeltaSignature(name="lentissimo", n_clients=1000, gap_lambda=0.02),   # ≈ 50 days
    ]


class SimpleDeltaExperiment(SyntheticExperiment):
    """Build a delta-only synthetic dataset (clusters differ solely in gap rate)."""

    @property
    def experiment(self) -> str:
        return "simple_delta"

    @property
    def output_section(self) -> str:
        return _SECTION

    @staticmethod
    def shared_profile(merchants: MerchantConfig) -> SharedProfile:
        """Return the fixed shared profile used by the experiment."""
        return shared_profile(merchants)

    @staticmethod
    def delta_signatures() -> list[DeltaSignature]:
        """Return the fixed cluster signatures used by the experiment."""
        return delta_signatures()

    def __init__(self) -> None:
        self.data_config = DATA_CONFIG
        self.profile = self.shared_profile(MerchantConfig())
        self.signatures = self.delta_signatures()
        if not self.signatures:
            raise ValueError("`signatures` cannot be empty.")
        self.timing = load_timing_config()

    def sample_timestamps(
        self,
        rng: np.random.Generator,
        sig: DeltaSignature,
        n_tx: int,
    ) -> np.ndarray:
        """Sample timestamps whose gaps follow the cluster-specific rate."""
        mean_gap_days = 1.0 / max(sig.gap_lambda, 1e-9)

        gaps_days = rng.exponential(mean_gap_days, size=n_tx)
        gaps_days[0] = 0.0
        day_offsets = np.floor(np.cumsum(gaps_days)).astype(np.int64)

        # Random absolute start so dow/day-of-month decompositions stay uninformative.
        start_day = int(rng.integers(0, self.timing.n_days))
        days = start_day + day_offsets

        hours = rng.integers(0, 24, size=n_tx)
        secs = rng.integers(0, 3600, size=n_tx)
        ts = self.timing.ts_base + days * self.timing.day + hours * 3600 + secs
        return np.sort(ts.astype(np.int64))

    def build(self, seed: int) -> pd.DataFrame:
        """Build one delta-only synthetic split."""
        rng = np.random.default_rng(seed)
        profile = self.profile

        assignments = np.repeat(
            np.arange(len(self.signatures)),
            [int(s.n_clients) for s in self.signatures],
        )
        rng.shuffle(assignments)
        n_clients = int(assignments.shape[0])

        # Shared (cluster-independent) sampling material — computed once.
        cocau_pools = self._cocau_pools(profile.merchants, profile.preferred_cocau)
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
            ts = self.sample_timestamps(rng, sig, n_tx)

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