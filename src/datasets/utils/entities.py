from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Arrival rate used by timestamp generation:
# - float for homogeneous Poisson process
# - callable lambda(t) for non-homogeneous process
TimeLambda = float | Callable[[float], float]


@dataclass(frozen=True)
class Merchant:
    """Merchant domain metadata.

    Input:
        name: merchant display name.
        amount_mean: expected transaction amount for this merchant.
        amount_variance: transaction amount variance for this merchant.
        cocau: allowed categorical transaction codes for this merchant.
    Output:
        Frozen dataclass instance.
    What it does:
        Stores stable merchant-level attributes consumed by synthetic generators.
    """

    name: str
    amount_mean: float
    amount_variance: float
    cocau: tuple[int, ...]


@dataclass(frozen=True)
class ClientType:
    """Client cluster definition used by all dataset generators.

    Input:
        name: cluster label.
        cluster_prob: prior probability used for client-to-cluster assignment.
        hours: 24-hour probability distribution of transaction hours.
        amount_mu: baseline lognormal mean (client spending profile).
        arrival_lambda: arrival rate (constant or function of time).
        merchants: merchant pool associated with this cluster.
        preferred_cocau: preferred categorical transaction codes.
        arrival_lambda_max: optional upper bound for lambda(t) in thinning.
    Output:
        Frozen dataclass instance.
    What it does:
        Encapsulates behavioral parameters for one synthetic client segment.
    """

    name: str
    cluster_prob: float
    hours: np.ndarray
    amount_mu: float
    arrival_lambda: TimeLambda
    merchants: list[Merchant]
    preferred_cocau: tuple[int, ...]
    arrival_lambda_max: float | None = None


@dataclass(frozen=True)
class Fingerprint:
    """Per-client stable fingerprint sampled inside a cluster.

    Input:
        fav_merchants: client-specific preferred merchants.
        amount_mu: client-specific spending baseline.
    Output:
        Frozen dataclass instance.
    What it does:
        Carries stable per-client traits used while generating transactions.
    """

    fav_merchants: np.ndarray
    amount_mu: float
