"""Client cluster definitions and cluster/fingerprint samplers."""

from __future__ import annotations

import numpy as np

from .merchant import MerchantConfig, _merchant_catalog
from ..utils.entities import ClientType, Fingerprint, Merchant
from .abc import Generator


def hour_cluster_probs(peak: int, width: float = 4.0) -> np.ndarray:
    """Build a normalized 24-hour probability curve centered on a peak hour.

    Input:
        peak: hour-of-day (0-23) where the curve is centered.
        width: Gaussian spread controlling how flat/peaked the profile is.
    Output:
        NumPy array of shape (24,) summing to 1.
    What it does:
        Creates an hour preference distribution used by client clusters.
    """
    h = np.arange(24)
    w = np.exp(-0.5 * ((h - peak) / width) ** 2) + 0.02
    return w / w.sum()


def cluster_types(merchants: MerchantConfig) -> list[ClientType]:
    """Return default client clusters based on merchant configuration.

    Input:
        merchants: merchant configuration defining available merchant pools.
    Output:
        List of ClientType objects with priors and behavioral parameters.
    What it does:
        Creates baseline clusters for synthetic generation, including hour
        profiles, spending levels, arrival rates, merchant pools, and
        preferred cocau codes.
    """
    hours_morning = hour_cluster_probs(9)  # commuters / utilities usage
    hours_midday = hour_cluster_probs(13)  # families and groceries
    hours_evening = hour_cluster_probs(20)  # evening lifestyle / shopping
    catalog = _merchant_catalog(merchants)

    def merchant_list(names: list[str]) -> list[Merchant]:
        """Map merchant names to Merchant metadata objects.

        Input:
            names: merchant name list.
        Output:
            Ordered list of Merchant objects.
        What it does:
            Resolves each merchant name through the local metadata catalog.
        """
        return [catalog[n] for n in names]

    return [
        ClientType(
            name="famiglia_giorno",
            cluster_prob=0.35,
            hours=hours_midday,
            amount_mu=3.4,
            arrival_lambda=1.0 / (1.5 * 86400),
            merchants=merchant_list(merchants.groceries + merchants.payments + merchants.utilities),
            preferred_cocau=(8, 19, 36, 54, 72, 91, 101, 110, 129, 148, 167, 186, 210, 238, 260),
        ),
        ClientType(
            name="giovane_sera",
            cluster_prob=0.30,
            hours=hours_evening,
            amount_mu=2.8,
            arrival_lambda=1.0 / (2.0 * 86400),
            merchants=merchant_list(merchants.payments + merchants.shopping + ["Starbucks"]),
            preferred_cocau=(7, 11, 23, 24, 39, 45, 57, 66, 76, 88, 98, 101, 120, 123, 140, 161, 182, 203, 224),
        ),
        ClientType(
            name="altospendente",
            cluster_prob=0.20,
            hours=hours_evening,
            amount_mu=4.2,
            arrival_lambda=1.0 / (3.0 * 86400),
            merchants=merchant_list(merchants.shopping + merchants.travel),
            preferred_cocau=(33, 52, 71, 90, 109, 120, 128, 147, 166, 185, 210, 230, 252, 274, 296),
        ),
        ClientType(
            name="mattiniero_utenze",
            cluster_prob=0.15,
            hours=hours_morning,
            amount_mu=3.3,
            arrival_lambda=1.0 / (2.5 * 86400),
            merchants=merchant_list(merchants.utilities + merchants.groceries),
            preferred_cocau=(8, 19, 36, 54, 72, 91, 101, 110, 129, 148, 167, 186, 210, 238, 260, 282),
        ),
    ]

def return_cluster():
    return cluster_types(MerchantConfig())


class ClusterToClientGenerator(Generator):
    """Assign each client to a ClientType via categorical sampling."""

    def __init__(self, rng: np.random.Generator, client_types: list[ClientType]) -> None:
        """Initialize cluster assignment sampler.

        Input:
            rng: NumPy random generator.
            client_types: ordered client cluster list with `cluster_prob` priors.
        Output:
            None.
        What it does:
            Precomputes normalized sampling probabilities for cluster assignment.
        """
        self._rng = rng
        self._types = client_types
        w = np.array([t.cluster_prob for t in client_types], dtype=float)
        self._p = w / w.sum()

    def generate(self, n_clients: int) -> np.ndarray:
        """Sample cluster index for each client.

        Input:
            n_clients: number of clients to assign.
        Output:
            NumPy array of cluster indices, shape `(n_clients,)`.
        What it does:
            Draws one categorical cluster assignment per client.
        """
        return self._rng.choice(len(self._types), size=n_clients, p=self._p)

    def type_for(self, idx: int) -> ClientType:
        """Return the ClientType for a previously sampled cluster index.

        Input:
            idx: cluster index.
        Output:
            Corresponding ClientType object.
        What it does:
            Provides index-to-object lookup after assignment.
        """
        return self._types[idx]


class FingerprintGenerator(Generator):
    """Generate per-client fingerprints inside a cluster."""

    def __init__(self, rng: np.random.Generator, sigma_spending: float) -> None:
        """Initialize fingerprint sampler.

        Input:
            rng: NumPy random generator.
            sigma_spending: standard deviation for spending baseline perturbation.
        Output:
            None.
        What it does:
            Stores sampling state and noise scale for per-client variability.
        """
        self._rng = rng
        self._sigma = sigma_spending

    def generate(self, ctype: ClientType) -> Fingerprint:
        """Sample a stable fingerprint for one client.

        Input:
            ctype: client cluster used as the sampling prior.
        Output:
            Fingerprint with favorite merchants and personalized amount_mu.
        What it does:
            Chooses a subset of preferred merchants and perturbs the cluster-level
            spending baseline to obtain client-specific behavior.
        """
        pool = ctype.merchants
        # Keep RNG evaluation order stable: integers() is evaluated before choice().
        fav = self._rng.choice(
            pool, size=min(self._rng.integers(3, 8), len(pool)), replace=False
        )
        amount_mu = ctype.amount_mu + self._rng.normal(0, self._sigma)
        return Fingerprint(fav_merchants=fav, amount_mu=amount_mu)
