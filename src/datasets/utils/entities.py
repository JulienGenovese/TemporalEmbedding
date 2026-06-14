from __future__ import annotations

from dataclasses import dataclass


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
