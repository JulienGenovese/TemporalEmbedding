"""Calendar-only validation dataset.

This experiment is the negative-control counterpart of ``simple_delta``. All
clusters share the same value profile, transaction volume and inter-transaction
gap distribution. They differ only by the absolute calendar phase: transactions
happen weekly, but each cluster is anchored to a different day offset within the
week.

Because the hierarchical model consumes ``delta_t`` but does not include
``timestamp`` in ``DEFAULT_FEATURES``, these clusters should not be separable by
the current model. If they are separable, the signal is leaking through a
non-calendar channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...data_schema import DATA_CONFIG
from .abc import SyntheticExperiment
from .simple_delta import SharedProfile, shared_profile
from .specs import MerchantConfig, load_timing_config

_SECTION = "synthetic.simple_calendar"
_WEEK_DAYS = 7


@dataclass(frozen=True)
class CalendarSignature:
    """Cluster signature: only the absolute calendar phase differs."""

    name: str
    n_clients: int
    week_day_offset: int


def calendar_signatures() -> list[CalendarSignature]:
    """Return clusters separated only by their weekly calendar offset."""
    return [
        CalendarSignature(name="calendar_phase_0", n_clients=100, week_day_offset=0),
        CalendarSignature(name="calendar_phase_2", n_clients=100, week_day_offset=2),
        CalendarSignature(name="calendar_phase_4", n_clients=100, week_day_offset=4),
        CalendarSignature(name="calendar_phase_6", n_clients=100, week_day_offset=6),
    ]


class SimpleCalendarExperiment(SyntheticExperiment):
    """Build a calendar-only dataset with identical ``delta_t`` across clusters."""

    @property
    def experiment(self) -> str:
        return "simple_calendar"

    @property
    def output_section(self) -> str:
        return _SECTION

    @staticmethod
    def shared_profile(merchants: MerchantConfig) -> SharedProfile:
        """Return the fixed shared value profile used by the experiment."""
        return shared_profile(merchants)

    @staticmethod
    def calendar_signatures() -> list[CalendarSignature]:
        """Return the fixed calendar signatures used by the experiment."""
        return calendar_signatures()

    def __init__(self) -> None:
        self.data_config = DATA_CONFIG
        self.profile = self.shared_profile(MerchantConfig())
        self.signatures = self.calendar_signatures()
        if not self.signatures:
            raise ValueError("`signatures` cannot be empty.")
        for signature in self.signatures:
            if not 0 <= signature.week_day_offset < _WEEK_DAYS:
                raise ValueError("`week_day_offset` must be in [0, 6].")
        self.timing = load_timing_config()
        self._max_weekly_events = max(1, self.timing.n_days // _WEEK_DAYS)

    def sample_timestamps(
        self,
        rng: np.random.Generator,
        sig: CalendarSignature,
        n_tx: int,
    ) -> np.ndarray:
        """Sample weekly timestamps anchored to the cluster calendar phase."""
        max_start_week = max(1, self._max_weekly_events - n_tx + 1)
        start_week = int(rng.integers(0, max_start_week))

        day_offsets = (
            (start_week + np.arange(n_tx, dtype=np.int64)) * _WEEK_DAYS
            + sig.week_day_offset
        )
        seconds_within_day = rng.integers(0, self.timing.day, size=n_tx)
        ts = self.timing.ts_base + day_offsets * self.timing.day + seconds_within_day
        return ts.astype(np.int64)

    def build(self, seed: int) -> pd.DataFrame:
        """Build one calendar-only synthetic split."""
        rng = np.random.default_rng(seed)
        profile = self.profile

        assignments = np.repeat(
            np.arange(len(self.signatures)),
            [int(s.n_clients) for s in self.signatures],
        )
        rng.shuffle(assignments)
        n_clients = int(assignments.shape[0])

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
            n_tx = min(n_tx, self._max_weekly_events)

            # Timing is the only cluster-dependent draw; gaps stay weekly.
            ts = self.sample_timestamps(rng, sig, n_tx)

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
            ts_col.append(ts)
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
