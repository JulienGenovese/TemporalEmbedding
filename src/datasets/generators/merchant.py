"""Merchant pools, metadata catalog, and merchant selection logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from ..utils.entities import Merchant


@dataclass
class MerchantConfig:
    """Merchant pools grouped by theme.

    Input:
        Themed merchant lists and selection probabilities (defaults provided).
    Output:
        Mutable config dataclass with computed `merchant_pool`.
    What it does:
        Defines merchant universes used by cluster construction and selection.
    """

    groceries: list[str] = field(default_factory=lambda: [
        "Esselunga", "Conad", "Carrefour", "Lidl", "Aldi", "Coop"])
    shopping: list[str] = field(default_factory=lambda: [
        "Amazon", "Walmart", "IKEA", "Apple", "Zara", "H&M",
        "Decathlon", "Mediaworld", "Unieuro"])
    travel: list[str] = field(default_factory=lambda: [
        "Trenitalia", "Booking", "Airbnb", "RyanAir", "Easyjet"])
    utilities: list[str] = field(default_factory=lambda: [
        "TIM", "Vodafone", "Eni", "Shell"])
    payments: list[str] = field(default_factory=lambda: [
        "PayPal", "Satispay", "Bancomat", "Netflix", "Spotify", "Starbucks"])

    # "Universal" merchants that can appear for any client cluster.
    common_merchants: list[str] = field(default_factory=lambda: [
        "Amazon", "PayPal", "Netflix", "Spotify"])
    p_common_merchant: float = 0.15  # Constant probability of picking a universal merchant.

    merchant_pool: list[str] = field(init=False)  # Union of all themed merchant lists.

    def __post_init__(self) -> None:
        """Compute the complete merchant pool from all themes.

        Input:
            None.
        Output:
            None.
        What it does:
            Builds `merchant_pool`, consumed by global/off-pattern selection paths.
        """
        self.merchant_pool = (self.groceries + self.shopping + self.travel
                              + self.utilities + self.payments)


def _merchant_catalog(merchants: MerchantConfig) -> dict[str, Merchant]:
    """Build a merchant-name -> Merchant metadata catalog.

    Input:
        merchants: merchant configuration with themed name lists.
    Output:
        Dictionary keyed by merchant name with stable Merchant metadata values.
    What it does:
        Assigns default amount statistics and available cocau codes by merchant theme.
    """
    grouped_stats = (
        (merchants.groceries, 45.0, 180.0, (8, 14, 27, 41, 58, 73, 89, 101, 120, 133, 149, 168)),
        (merchants.shopping, 120.0, 950.0, (11, 24, 39, 57, 76, 98, 120, 140, 162, 185, 207, 229, 251)),
        (merchants.travel, 280.0, 4_000.0, (33, 52, 71, 90, 109, 128, 147, 166, 185, 210, 230, 252, 274, 296)),
        (merchants.utilities, 90.0, 300.0, (19, 36, 54, 72, 91, 110, 129, 148, 167, 186, 210, 238, 260, 282)),
        (merchants.payments, 22.0, 140.0, (7, 23, 45, 66, 88, 101, 123, 140, 161, 182, 203, 224, 245, 266, 287)),
    )
    return {
        name: Merchant(
            name=name,
            amount_mean=mean,
            amount_variance=variance,
            cocau=cocau,
        )
        for names, mean, variance, cocau in grouped_stats
        for name in names
    }

def merchant_pool() -> list[Merchant]:
    """Get the full merchant pool as a list of Merchant instances.

    Input:
        merchants: merchant configuration with themed name lists.
    Output:
        List of Merchant instances corresponding to the full merchant pool. 
    What it does:
        Converts the merchant pool from names to Merchant instances using the catalog.
    """
    catalog = _merchant_catalog(MerchantConfig())
    return catalog
