"""Shared synthetic experiment specifications."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import config

_TIMING_SECTION = "synthetic.timing"


@dataclass(frozen=True)
class Merchant:
    """Merchant-level metadata consumed by synthetic generators."""

    name: str
    amount_mean: float
    amount_variance: float
    cocau: tuple[int, ...]


@dataclass
class MerchantConfig:
    """Merchant pools grouped by theme."""

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

    common_merchants: list[str] = field(default_factory=lambda: [
        "Amazon", "PayPal", "Netflix", "Spotify"])
    p_common_merchant: float = 0.15

    merchant_pool: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.merchant_pool = (self.groceries + self.shopping + self.travel
                              + self.utilities + self.payments)


@dataclass(frozen=True)
class SimpleClientType:
    """Shared cluster definition used by simple spatial/timing experiments."""

    name: str
    n_clients: int
    amount_mean: float
    amount_std: float
    n_tx_mean: float
    n_tx_std: float
    merchants: list[Merchant]
    preferred_cocau: tuple[int, ...]
    gap_lambda: float = 0.25
    negative_only: bool = False
    cluster_prob: float | None = None


@dataclass(frozen=True)
class TimingConfig:
    """Shared synthetic timestamp configuration."""

    ts_base: int
    ts_range: int
    day: int

    @property
    def n_days(self) -> int:
        return max(1, self.ts_range // self.day)


def load_timing_config() -> TimingConfig:
    return TimingConfig(
        ts_base=config.get(_TIMING_SECTION, "ts_base", value_type=int),
        ts_range=config.get(_TIMING_SECTION, "ts_range", value_type=int),
        day=config.get(_TIMING_SECTION, "day", value_type=int),
    )


def _merchant_catalog(merchants: MerchantConfig) -> dict[str, Merchant]:
    """Build a merchant-name -> metadata catalog."""
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
    """Return the full merchant pool as metadata objects."""
    catalog = _merchant_catalog(MerchantConfig())
    return list(catalog.values())
