"""Synthetic timestamp generation with homogeneous and non-homogeneous arrivals.

Overview:
- Main inputs:
  - n_tx: number of timestamps/transactions to generate.
  - ctype: client profile (hour distribution + arrival parameters).
  - noise_level: how much to mix cluster hours with a uniform distribution.
- Arrival strategy:
  - constant lambda -> homogeneous Poisson;
  - lambda(t) -> NHPP via Lewis-Shedler thinning.
- Output:
  - NumPy array of Unix timestamps (seconds), composed from:
    cumulative day + sampled hour + intra-hour seconds.
"""

from __future__ import annotations

from numbers import Real

import numpy as np

from .abc import Generator
from ..utils.config import TimeConfig
from ..utils.entities import ClientType, TimeLambda


class ArrivalGapSampler:
    """Inter-arrival sampler for homogeneous and non-homogeneous processes."""

    def __init__(self, rng: np.random.Generator, ts_range: int) -> None:
        """Initialize inter-arrival gap sampler.

        Input:
            rng: NumPy random generator used for all draws.
            ts_range: temporal horizon length in seconds.
        Output:
            None.
        What it does:
            Stores parameters needed to estimate/evaluate lambda(t) on the
            configured time horizon.
        """
        self._rng = rng
        self._ts_range = ts_range

    @staticmethod
    def _validate_lambda_value(value: float) -> float:
        """Validate one lambda intensity value.

        Input:
            value: candidate lambda(t) value.
        Output:
            The same float value if valid.
        What it does:
            Ensures value is finite and non-negative, otherwise raises ValueError.
        """
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("`lambda(t)` must be finite and non-negative.")
        return value

    def _eval_lambda(self, lam_fn: TimeLambda, t: float) -> float:
        """Evaluate lambda at time t, handling constant or callable forms.

        Input:
            lam_fn: constant lambda (float) or callable lambda(t).
            t: timestamp in seconds where lambda is evaluated.
        Output:
            Validated lambda(t) value (finite and >= 0).
        What it does:
            Unifies constant/function handling and validates every returned value.
        """
        if isinstance(lam_fn, Real):
            return self._validate_lambda_value(float(lam_fn))
        return self._validate_lambda_value(float(lam_fn(float(t))))

    def _estimate_lambda_max(self, lam_fn: TimeLambda, start: int) -> float:
        """Estimate an upper bound for lambda(t) used by thinning.

        Input:
            lam_fn: lambda(t) callable or constant value.
            start: start timestamp of simulation.
        Output:
            Estimated lambda_max > 0.
        What it does:
            Samples lambda on a grid over [start, start+ts_range], takes the
            observed maximum, adds a 5% safety margin, and validates positivity.
        """
        grid = np.linspace(start, start + self._ts_range, num=512, dtype=float)
        values = np.array([self._eval_lambda(lam_fn, float(t)) for t in grid], dtype=float)
        lam_max = float(values.max(initial=0.0)) * 1.05
        if lam_max <= 0.0:
            raise ValueError("`lambda(t)` must be strictly positive at least at one time point.")
        return lam_max

    def _sample_homogeneous(self, n_tx: int, lam: float) -> np.ndarray:
        """Sample inter-arrival gaps for a homogeneous Poisson process.

        Input:
            n_tx: number of events/transactions to generate.
            lam: constant intensity in events per second.
        Output:
            Int64 array of length n_tx with gaps in seconds.
        What it does:
            Draws n_tx exponential inter-arrivals with mean 1/lam.
        """
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError("Constant lambda must be finite and > 0.")
        return self._rng.exponential(scale=1.0 / lam, size=n_tx).astype(np.int64)

    def _sample_thinning(
        self,
        n_tx: int,
        start: int,
        lam_fn: TimeLambda,
        lam_max: float,
    ) -> np.ndarray:
        """Sample NHPP gaps with Lewis-Shedler thinning.

        Input:
            n_tx: number of accepted events to generate.
            start: simulation start time.
            lam_fn: lambda(t) intensity function.
            lam_max: upper bound for lambda(t) over the considered horizon.
        Output:
            Int64 array of length n_tx with accepted-event gaps in seconds.
        What it does:
            1) Proposes candidates from homogeneous process intensity lam_max.
            2) Accepts each candidate with probability lambda(t)/lam_max.
            3) Converts accepted absolute times to inter-arrival differences.
        """
        if not np.isfinite(lam_max) or lam_max <= 0.0:
            raise ValueError("`arrival_lambda_max` must be finite and > 0.")

        accepted = np.empty(n_tx, dtype=float)
        t = float(start)
        i = 0
        while i < n_tx:
            t += float(self._rng.exponential(scale=1.0 / lam_max))
            lam_t = self._eval_lambda(lam_fn, t)
            if lam_t > lam_max:
                raise ValueError("`lambda(t)` exceeded `arrival_lambda_max` during thinning.")
            if lam_t > 0.0 and self._rng.random() <= (lam_t / lam_max):
                accepted[i] = t
                i += 1

        return np.diff(np.concatenate(([float(start)], accepted))).astype(np.int64)

    def sample(self, n_tx: int, start: int, ctype: ClientType) -> np.ndarray:
        """Select gap sampling strategy according to client configuration.

        Input:
            n_tx: number of transactions to generate.
            start: start time in seconds.
            ctype: ClientType with temporal parameters (arrival_lambda,
                arrival_lambda_max).
        Output:
            Int64 gap array in seconds, length n_tx.
        What it does:
            - If arrival_lambda is constant: uses homogeneous Poisson.
            - If arrival_lambda is callable: uses NHPP with thinning.
        """
        lam = ctype.arrival_lambda
        if isinstance(lam, Real):
            return self._sample_homogeneous(n_tx, float(lam))

        lam_max = ctype.arrival_lambda_max
        if lam_max is None:
            lam_max = self._estimate_lambda_max(lam, start)
        return self._sample_thinning(n_tx, start, lam, float(lam_max))


class TimestampGenerator(Generator):
    """Unix timestamp generator: sampled gaps + hour profile + uniform seconds."""

    _UNIFORM_HOURS = np.full(24, 1.0 / 24)

    def __init__(self, rng: np.random.Generator, time: TimeConfig) -> None:
        """Initialize timestamp generator.

        Input:
            rng: shared random generator.
            time: time configuration (base, range, seconds per day).
        Output:
            None.
        What it does:
            Stores time constants and creates the arrival gap sampler used by
            `generate()`.
        """
        self._rng = rng
        self._ts_base = time.ts_base
        self._ts_range = time.ts_range
        self._day = time.day
        self._arrival_sampler = ArrivalGapSampler(rng, self._ts_range)

    def generate(self, n_tx: int, ctype: ClientType, noise_level: float) -> np.ndarray:
        """Generate Unix timestamps for n_tx transactions.

        Input:
            n_tx: number of timestamps to produce.
            ctype: client cluster with hour distribution and arrival settings.
            noise_level: noise intensity [0, 1] for cluster/uniform hour mixing.
        Output:
            NumPy array of Unix timestamps (seconds), ordered by cumulative day.
        What it does:
            1) Builds hour distribution as cluster/uniform mixture (max 40% uniform).
            2) Samples random start in first half of horizon.
            3) Samples temporal gaps via ArrivalGapSampler.
            4) Rebuilds final timestamps from day, hour, and intra-hour seconds.
        """
        # Mix cluster hour profile with uniform profile (40% uniform at noise=1).
        hours_p = (1.0 - 0.4 * noise_level) * ctype.hours + (0.4 * noise_level) * self._UNIFORM_HOURS
        hours_p = hours_p / hours_p.sum()

        start = self._ts_base + int(self._rng.integers(0, self._ts_range // 2))
        gaps = self._arrival_sampler.sample(n_tx, start, ctype)

        days = (start + np.cumsum(gaps)) // self._day
        hours = self._rng.choice(24, size=n_tx, p=hours_p)
        secs = self._rng.integers(0, 3600, size=n_tx)
        return days * self._day + hours * 3600 + secs
