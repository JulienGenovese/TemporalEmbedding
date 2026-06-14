"""Timing-only validation dataset.

A controlled probe for the time-aware part of the model: four clusters that are
**identical in everything that is embedded as a value** (amount distribution,
merchant pool, cocau pool, transaction volume) and differ *only* in their
temporal signature — the hour-of-day and day-of-week at which transactions
happen.

    Cluster A — "mattiniero_feriale":        peak hour 09, peak Mon–Tue.
    Cluster B — "pranzo_infrasettimanale":   peak hour 13, peak Wed–Thu.
    Cluster C — "serale_weekend":            peak hour 20, peak Fri–Sat.
    Cluster D — "notturno_weekend":          peak hour 02, peak Sat–Sun.

Because the spending profile is defined once on the dataset (a single
``SharedProfile``) and shared by every cluster, the *only* signal separating
them is timing. A model that recovers the clusters from the resulting
embeddings is genuinely exploiting ``delta_t`` / the timestamp decomposition;
one that collapses them is blind to time. This makes the experiment a clean
pass/fail validation of :class:`TimeAwarePositionalEncoding` and the datetime
field encoder.

The amount and cocau are drawn exactly like ``simple``/``coherent`` (mean of the
shared-cluster and merchant amount normals; cocau intersected with merchant
codes), so sign emerges naturally and the value distributions match the rest of
the synthetic suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
    """Spending profile shared by *every* timing cluster.

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
        cocau — they can only differ in their :class:`TimingSignature`.
    """

    amount_mean: float
    amount_std: float
    n_tx_mean: float
    n_tx_std: float
    merchants: list[Merchant]
    preferred_cocau: tuple[int, ...]


@dataclass(frozen=True)
class TimingSignature:
    """The *only* thing that distinguishes one timing cluster from another.

    Input:
        name: cluster label (written to the `cluster` column).
        n_clients: explicit number of clients belonging to this cluster.
        hour_peak: peak hour-of-day (0-23) of the circular hourly intensity.
        hour_width: std (in hours) of the circular-Gaussian hourly bump.
        dow_weights: 7 non-negative day-of-week weights (index 0 = Monday … 6 = Sunday).
        dow_base: floor added to every day-of-week weight (keeps a low background rate).
    Output:
        Frozen dataclass instance.
    What it does:
        Fully specifies a cluster's temporal fingerprint and how many clients it
        owns; carries no amount/merchant information by design.
    """

    name: str
    n_clients: int
    hour_peak: float
    hour_width: float
    dow_weights: tuple[float, float, float, float, float, float, float]
    dow_base: float = 0.1


def shared_profile(merchants: MerchantConfig) -> SharedProfile:
    """Build the single spending profile shared by all timing clusters.

    Input:
        merchants: merchant configuration providing themed name pools.
    Output:
        A :class:`SharedProfile` reused by every :class:`TimingSignature`.
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


def timing_signatures() -> list[TimingSignature]:
    """Return the four timing-only clusters of the validation experiment.

    Input:
        None.
    Output:
        List with exactly four :class:`TimingSignature` objects.
    What it does:
        Encodes four sharp, non-overlapping temporal fingerprints that combine a
        distinct peak hour with a distinct preferred pair of weekdays:
        "mattiniero feriale" (morning, Mon–Tue), "pranzo infrasettimanale"
        (midday, Wed–Thu), "serale weekend" (evening, Fri–Sat) and "notturno
        weekend" (night, Sat–Sun). The day-of-week index follows
        ``datetime.weekday`` (0 = Monday … 6 = Sunday).
    """
    return [
        TimingSignature(
            name="mattiniero_feriale",
            n_clients=1000,
            hour_peak=9.0,
            hour_width=2.0,
            #            Mon  Tue  Wed  Thu  Fri  Sat  Sun
            dow_weights=(1.0, 0.9, 0.5, 0.4, 0.3, 0.1, 0.1),
        ),
        TimingSignature(
            name="pranzo_infrasettimanale",
            n_clients=1000,
            hour_peak=13.0,
            hour_width=2.0,
            #            Mon  Tue  Wed  Thu  Fri  Sat  Sun
            dow_weights=(0.3, 0.4, 1.0, 0.9, 0.4, 0.1, 0.1),
        ),
        TimingSignature(
            name="serale_weekend",
            n_clients=1000,
            hour_peak=20.0,
            hour_width=2.0,
            #            Mon  Tue  Wed  Thu  Fri  Sat  Sun
            dow_weights=(0.1, 0.1, 0.3, 0.4, 0.9, 1.0, 0.6),
        ),
        TimingSignature(
            name="notturno_weekend",
            n_clients=1000,
            hour_peak=2.0,
            hour_width=2.0,
            #            Mon  Tue  Wed  Thu  Fri  Sat  Sun
            dow_weights=(0.1, 0.1, 0.1, 0.2, 0.4, 0.9, 1.0),
        ),
    ]


class TimingSyntheticTransactionDataset(BaseSyntheticDataset):
    """Build a timing-only synthetic dataset (clusters differ solely in time)."""

    experiment = "simple_timing"

    def __init__(
        self,
        config: DatasetConfig,
        profile: SharedProfile | None = None,
        signatures: list[TimingSignature] | None = None,
        split: str = "train",
    ) -> None:
        """Initialize the timing dataset builder.

        Input:
            config: full dataset configuration (sampling/time/output are used).
            profile: optional shared spending profile; defaults from merchant config.
            signatures: optional explicit timing clusters; defaults to the two-cluster probe.
            split: sampling split to draw (`train` or `pred`); selects the per-split
                seed so the two files are independent draws of the same signatures.
        Output:
            None.
        What it does:
            Stores config, resolves the shared profile and timing signatures, seeds
            the RNG, and precomputes the day-of-week calendar over the horizon.
        """
        self.config = config
        self.split = split
        self.data_config = DATA_CONFIG
        self.profile = profile if profile is not None else shared_profile(config.merchants)
        self.signatures = list(signatures) if signatures is not None else timing_signatures()
        if not self.signatures:
            raise ValueError("`signatures` cannot be empty.")
        self._rng = np.random.default_rng(config.sampling_for(split).seed)

        time_cfg = config.time
        self._ts_base = int(time_cfg.ts_base)
        self._day = int(time_cfg.day)
        self._n_days = max(1, int(time_cfg.ts_range) // self._day)
        self._dow = self._calendar_dow()

    def _calendar_dow(self) -> np.ndarray:
        """Compute the day-of-week for every day in the horizon.

        Input:
            None.
        Output:
            Integer array (dow in [0, 6], 0 = Monday) of length ``n_days``.
        What it does:
            Walks the daily grid once via UTC datetimes so day-of-week intensity
            can be mapped onto real calendar days.
        """
        day_ts = self._ts_base + np.arange(self._n_days) * self._day
        dates = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in day_ts]
        return np.array([d.weekday() for d in dates], dtype=np.int64)

    def _hour_prob(self, sig: TimingSignature) -> np.ndarray:
        """Circular-Gaussian hourly intensity normalized over 24 hours.

        Input:
            sig: timing signature providing peak hour and width.
        Output:
            Length-24 probability vector summing to 1.
        What it does:
            Places a wrap-around Gaussian bump at ``hour_peak`` so e.g. hour 23
            and hour 0 are neighbours, giving a sharp, single-peak hour profile.
        """
        h = np.arange(24)
        d = np.abs(h - sig.hour_peak)
        d = np.minimum(d, 24.0 - d)  # circular distance on the 24h clock
        w = np.exp(-0.5 * (d / max(sig.hour_width, 1e-6)) ** 2)
        return w / w.sum()

    def _day_prob(self, sig: TimingSignature) -> np.ndarray:
        """Day-of-week intensity mapped onto the calendar, normalized over the horizon.

        Input:
            sig: timing signature providing the 7 day-of-week weights.
        Output:
            Length-``n_days`` probability vector summing to 1.
        What it does:
            Looks up each calendar day's weekday weight (plus a small floor) so
            transactions concentrate on the cluster's preferred days.
        """
        weights = sig.dow_base + np.asarray(sig.dow_weights, dtype=float)
        w = weights[self._dow]
        total = w.sum()
        if total <= 0.0:
            return np.full(self._n_days, 1.0 / self._n_days)
        return w / total

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

    def build(self) -> pd.DataFrame:
        """Build the timing-only synthetic transaction dataset.

        Input:
            None.
        Output:
            Pandas DataFrame with the canonical transaction columns, sorted by
            (client_id, timestamp).
        What it does:
            Assigns each cluster its client count, then for every client samples a
            transaction count and draws timestamps from that cluster's hour/day
            intensities while drawing amount/merchant/cocau from the *shared*
            profile — so the clusters are separable by timing alone.
        """
        rng = self._rng
        profile = self.profile

        assignments = np.repeat(
            np.arange(len(self.signatures)),
            [int(s.n_clients) for s in self.signatures],
        )
        rng.shuffle(assignments)
        n_clients = int(assignments.shape[0])

        hour_probs = [self._hour_prob(s) for s in self.signatures]
        day_probs = [self._day_prob(s) for s in self.signatures]

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

            # --- timing: the ONLY cluster-dependent draw ---
            days = rng.choice(self._n_days, size=n_tx, p=day_probs[k])
            hours = rng.choice(24, size=n_tx, p=hour_probs[k])
            secs = rng.integers(0, 3600, size=n_tx)
            ts = self._ts_base + days * self._day + hours * 3600 + secs

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
        """Resolve the split-suffixed simple-timing output path from config."""
        return self.config.output.split_path(self.experiment, self.split)


def generate(
    config: DatasetConfig | None = None,
    profile: SharedProfile | None = None,
    signatures: list[TimingSignature] | None = None,
) -> dict[str, Path]:
    """Generate and save the simple-timing validation dataset for every split.

    Input:
        config: optional explicit dataset configuration.
        profile: optional shared spending profile.
        signatures: optional explicit timing cluster definitions.
    Output:
        Mapping of split name (`train`/`pred`) to the generated file path.
    What it does:
        Resolves the shared profile/signatures once, then materialises one file per
        split (`train` and `pred`) — independent draws from the same signatures.
    """
    resolved_config = config or DatasetConfig()
    shared_profile_obj = profile if profile is not None else shared_profile(resolved_config.merchants)
    shared_signatures = signatures if signatures is not None else timing_signatures()
    paths: dict[str, Path] = {}
    for split in SPLITS:
        std = TimingSyntheticTransactionDataset(
            config=resolved_config,
            profile=shared_profile_obj,
            signatures=shared_signatures,
            split=split,
        )
        paths[split] = std.generate_and_save()
    return paths


if __name__ == "__main__":
    generate()
