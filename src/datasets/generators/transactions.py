"""Transaction-level generators: counts, amounts, and refunds."""

from __future__ import annotations

import numpy as np

from .abc import Generator
from ..utils.config import AmountConfig, SamplingConfig
from ..utils.entities import Merchant


class TransactionCountAllocator(Generator):
    """Allocate transaction counts per client using Dirichlet + Multinomial."""

    def __init__(self, rng: np.random.Generator, sampling: SamplingConfig) -> None:
        """Initialize per-client transaction count allocator.

        Input:
            rng: NumPy random generator.
            sampling: sampling configuration with Dirichlet alpha and min count.
        Output:
            None.
        What it does:
            Stores parameters used to allocate total transactions across clients.
        """
        self._rng = rng
        self._alpha = sampling.alpha_dirichlet
        self._min = sampling.min_tx_per_client

    def generate(self, n_transactions: int, n_clients: int) -> np.ndarray:
        """Allocate `n_transactions` across `n_clients`.

        Input:
            n_transactions: total number of transactions to distribute.
            n_clients: number of clients.
        Output:
            Integer NumPy array of length `n_clients`, summing to n_transactions.
        What it does:
            Samples client weights via Dirichlet, draws multinomial counts,
            enforces a minimum per client, then rebalances to preserve the total.
        """
        weights = self._rng.dirichlet(np.ones(n_clients) * self._alpha)
        counts = self._rng.multinomial(n_transactions, weights)
        counts = np.clip(counts, self._min, None)
        while counts.sum() > n_transactions:
            counts[counts.argmax()] -= 1
        while counts.sum() < n_transactions:
            counts[counts.argmin()] += 1
        return counts
    
    

class AmountGenerator(Generator):
    """Sample signed transaction amounts from a lognormal distribution."""

    def __init__(self, rng: np.random.Generator, amount: AmountConfig) -> None:
        """Initialize amount generator.

        Input:
            rng: NumPy random generator.
            amount: amount configuration with sign probability and sigma.
        Output:
            None.
        What it does:
            Stores parameters for amount sign and magnitude sampling.
        """
        self._rng = rng
        self._p_spend = amount.spending_probability
        self._sigma = amount.lognormal_sigma

    def generate(self, mu: float) -> tuple[float, int]:
        """Sample one signed amount.

        Input:
            mu: lognormal mean parameter.
        Output:
            Tuple `(amount, sign)` where sign is -1 for debit, +1 for credit.
        What it does:
            Draws magnitude from lognormal(mu, sigma) and applies sampled sign.
        """
        sign = -1 if self._rng.random() < self._p_spend else 1
        return sign * float(self._rng.lognormal(mean=mu, sigma=self._sigma)), sign


class CoherentAmountGenerator(Generator):
    """Sample client-merchant coherent amounts by mixing two lognormal samples."""

    def __init__(self, rng: np.random.Generator, amount: AmountConfig, merchant_weight: float = 0.5) -> None:
        """Initialize coherent amount generator.

        Input:
            rng: NumPy random generator.
            amount: amount configuration for sign probability and client sigma.
            merchant_weight: weight in [0, 1] assigned to merchant-driven sample.
        Output:
            None.
        What it does:
            Configures a blended amount model combining client and merchant effects.
        """
        self._rng = rng
        self._p_spend = amount.spending_probability
        self._client_sigma = amount.lognormal_sigma
        self._merchant_weight = float(np.clip(merchant_weight, 0.0, 1.0))

    def _merchant_lognormal_params(self, merchant: Merchant) -> tuple[float, float]:
        """Convert merchant mean/variance to lognormal `(mu, sigma)` parameters.

        Input:
            merchant: merchant metadata with amount moments.
        Output:
            Tuple `(mu, sigma)` for a numerically stable lognormal.
        What it does:
            Applies moment-matching transformation with lower bounds for stability.
        """
        mean = max(float(merchant.amount_mean), 1e-6)
        variance = max(float(merchant.amount_variance), 1e-6)
        sigma2 = float(np.log1p(variance / (mean ** 2)))
        sigma = float(np.sqrt(sigma2))
        mu = float(np.log(mean) - 0.5 * sigma2)
        return mu, max(sigma, 1e-6)

    def generate(self, client_mu: float, merchant: Merchant) -> tuple[float, int]:
        """Sample one coherent signed amount conditioned on client and merchant.

        Input:
            client_mu: client-side lognormal mean.
            merchant: merchant metadata used to derive merchant-side distribution.
        Output:
            Tuple `(amount, sign)` where sign is -1 for debit, +1 for credit.
        What it does:
            Samples a client amount and a merchant amount, blends them by
            `merchant_weight`, then applies sampled sign.
        """
        sign = -1 if self._rng.random() < self._p_spend else 1
        client_sample = float(self._rng.lognormal(mean=client_mu, sigma=self._client_sigma))
        merchant_mu, merchant_sigma = self._merchant_lognormal_params(merchant)
        merchant_sample = float(self._rng.lognormal(mean=merchant_mu, sigma=merchant_sigma))
        amount = (1.0 - self._merchant_weight) * client_sample + self._merchant_weight * merchant_sample
        return sign * float(amount), sign


class RefundGenerator(Generator):
    """Generate partial refunds/reversals for debit transactions."""

    def __init__(self, rng: np.random.Generator, day: int) -> None:
        """Initialize refund generator.

        Input:
            rng: NumPy random generator.
            day: seconds in one day.
        Output:
            None.
        What it does:
            Stores sampling state for refund timing and percentage.
        """
        self._rng = rng
        self._day = day

    def generate(self, ts: int, amount: float) -> tuple[int, float]:
        """Generate one refund event timestamp and amount.

        Input:
            ts: original transaction timestamp.
            amount: original signed amount (expected negative for debit).
        Output:
            Tuple `(refund_ts, refund_amt)` with refund 1-2 days later.
        What it does:
            Samples refund timing and a partial ratio in [30%, 100%], returning
            a credit amount that partially or fully offsets the original debit.
        """
        refund_ts = int(ts) + int(self._rng.integers(1, 3) * self._day)
        refund_amt = -amount * float(self._rng.uniform(0.3, 1.0))  # amount<0 -> refund>0
        return refund_ts, refund_amt