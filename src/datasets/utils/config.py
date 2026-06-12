

"""Configuration for synthetic dataset generation, split by subsystem.

This module defines:
  - SamplingConfig: dataset size and per-client allocation controls
  - NoiseConfig: single difficulty knob and derived noise dials
  - MerchantConfig: themed and shared merchant pools
  - TimeConfig: time origin and granularity constants
  - AmountConfig: amount distribution controls
  - CategoricalConfig: categorical ranges and noise controls
  - OutputConfig: output paths by experiment variant

DatasetConfig composes all sections and acts as the entry configuration object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..generators.merchant import MerchantConfig

from src.config import config



_SAMPLING_SECTION = "dataset.sampling"
_LEGACY_GENERAL_SECTIONS = ("syntheticData.general", "synteticData.general")
_EXPERIMENT_SECTIONS: dict[str, tuple[str, ...]] = {
    "vanilla": (
        "syntentic.vanilla",
        "syntetic.vanilla",
        "synthetic.vanilla",
        "syntheticData.output.vanilla",
        "synteticData.output.vanilla",
    ),
    "coherent": (
        "syntentic.coherent",
        "syntetic.coherent",
        "synthetic.coherent",
        "syntentic.cohernet",
        "syntetic.cohernet",
        "synthetic.cohernet",
        "syntheticData.output.coherent",
        "synteticData.output.coherent",
    ),
    "simple": (
        "syntentic.simple",
        "syntetic.simple",
        "synthetic.simple",
        "syntheticData.output.simple",
        "synteticData.output.simple",
    ),
    "timing": (
        "syntentic.timing",
        "syntetic.timing",
        "synthetic.timing",
        "syntheticData.output.timing",
        "synteticData.output.timing",
    ),
}
_SUPPORTED_OUTPUT_EXTENSIONS = {".csv", ".parquet"}
_MISSING = object()


def _sampling_int(key: str, default: int) -> int:
    """Read an integer value from the sampling config section.

    Input:
        key: sampling config key.
        default: fallback value if key is absent.
    Output:
        Integer config value.
    What it does:
        Fetches and validates that the resolved value is an integer (not bool).
    """
    try:
        value = config.get(_SAMPLING_SECTION, key, default)
    except KeyError:
        value = default
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{_SAMPLING_SECTION}.{key}` must be an integer, got {type(value).__name__}.")
    return value


def _sampling_float(key: str, default: float) -> float:
    """Read a numeric value from the sampling config section.

    Input:
        key: sampling config key.
        default: fallback value if key is absent.
    Output:
        Float config value.
    What it does:
        Fetches and validates numeric values (int/float, excluding bool).
    """
    try:
        value = config.get(_SAMPLING_SECTION, key, default)
    except KeyError:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{_SAMPLING_SECTION}.{key}` must be numeric, got {type(value).__name__}.")
    return float(value)


def _noise_level_float(default: float) -> float:
    """Read `noise_level`, prioritizing dataset.sampling."""
    try:
        value = config.get(_SAMPLING_SECTION, "noise_level", _MISSING)
    except KeyError:
        value = _MISSING
    if value is not _MISSING:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"`{_SAMPLING_SECTION}.noise_level` must be numeric, got {type(value).__name__}.")
        return float(value)
    return _coherent_float("noise_level", default)


def _coherent_float(key: str, default: float) -> float:
    """Read a numeric value for coherent synthetic generation settings."""
    sections = (
        *_EXPERIMENT_SECTIONS["coherent"],
        *_LEGACY_GENERAL_SECTIONS,
    )
    for section in sections:
        try:
            value = config.get(section, key, _MISSING)
        except KeyError:
            continue
        if value is _MISSING:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"`{section}.{key}` must be numeric, got {type(value).__name__}.")
        return float(value)
    return float(default)


def _output_path(variant: str, key: str, default: Path) -> Path:
    """Read and normalize an output path from config.

    Input:
        variant: experiment variant name (for example `vanilla`/`coherent`).
        key: output key inside the variant section.
        default: fallback path when config key is missing.
    Output:
        Path object.
    What it does:
        Resolves a path from config, validates type, and converts to `Path`.
    """
    sections = _EXPERIMENT_SECTIONS.get(variant)
    if sections is None:
        raise ValueError(f"Unsupported synthetic experiment `{variant}`.")

    for section in sections:
        try:
            out_path = config.get(section, key, _MISSING)
        except KeyError:
            continue

        if out_path is not _MISSING:
            if isinstance(out_path, Path):
                return out_path
            if not isinstance(out_path, str):
                raise TypeError(
                    f"`{section}.{key}` must be a string path, got {type(out_path).__name__}.",
                )
            return Path(out_path)

        folder = config.get(section, "folder", str(default.parent))
        name_file = config.get(section, "name_file", default.stem)
        ext = config.get(section, "ext", default.suffix or ".csv")

        if isinstance(folder, Path):
            folder_path = folder
        elif isinstance(folder, str):
            folder_path = Path(folder)
        else:
            raise TypeError(f"`{section}.folder` must be a string path, got {type(folder).__name__}.")

        if not isinstance(name_file, str):
            raise TypeError(f"`{section}.name_file` must be a string, got {type(name_file).__name__}.")
        if not name_file:
            raise ValueError(f"`{section}.name_file` cannot be empty.")

        if not isinstance(ext, str):
            raise TypeError(f"`{section}.ext` must be a string, got {type(ext).__name__}.")
        if not ext:
            raise ValueError(f"`{section}.ext` cannot be empty.")
        normalized_ext = (ext if ext.startswith(".") else f".{ext}").lower()
        if normalized_ext not in _SUPPORTED_OUTPUT_EXTENSIONS:
            raise ValueError(
                f"`{section}.ext` must be one of {sorted(_SUPPORTED_OUTPUT_EXTENSIONS)}, got `{ext}`.",
            )

        return folder_path / f"{name_file}{normalized_ext}"
    return Path(default)


@dataclass
class SamplingConfig:
    """Sampling volume and transaction allocation settings."""

    n_transactions: int = _sampling_int("n_transactions", 400_000)
    n_clients: int = _sampling_int("n_clients", 4_000)
    alpha_dirichlet: float = _sampling_float("alpha_dirichlet", 1.5)  # Smaller alpha => more skewed client activity.
    min_tx_per_client: int = _sampling_int("min_tx_per_client", 50)
    seed: int = _sampling_int("seed", 42)

@dataclass
class NoiseConfig:
    """Difficulty subsystem controlled by a single `noise_level` in [0, 1]."""

    noise_level: float = field(default_factory=lambda: _noise_level_float(0.9))
    # 0 = clean/separable, 1 = high noise

    p_offpattern: float = field(init=False)  # Probability of ignoring client fingerprint.
    p_global_merchant: float = field(init=False)  # Probability of drawing merchant from global pool.
    p_refund: float = field(init=False)  # Probability of partial refund after debit.
    sigma_spending: float = field(init=False)  # Sigma for client fingerprint spending perturbation.

    def __post_init__(self) -> None:
        """Derive all noise dials from `noise_level`.

        Input:
            None.
        Output:
            None.
        What it does:
            Clamps `noise_level` to [0, 1] and computes derived probabilities
            and spending-noise sigma used by other generators.
        """
        self.noise_level = max(0.0, min(1.0, float(self.noise_level)))
        self.p_offpattern = 0.2 * self.noise_level
        self.p_global_merchant = 0.3 * self.noise_level
        self.p_refund = 0.05 * self.noise_level
        self.sigma_spending = 0.3 + 0.4 * self.noise_level





@dataclass
class TimeConfig:
    """Temporal anchoring and granularity constants."""

    ts_base: int = 1_577_836_800  # 2020-01-01 00:00 UTC
    ts_range: int = 4 * 365 * 24 * 3600  # About 4 years in seconds
    day: int = 86_400


@dataclass
class AmountConfig:
    """Amount distribution settings (signed lognormal)."""

    spending_probability: float = 0.85  # Probability that transaction is debit (sign < 0)
    lognormal_sigma: float = 1.0  # Lognormal sigma for amount magnitude
    merchant_amount_weight: float = field(
        default_factory=lambda: _coherent_float("merchant_amount_weight", 0.5),
    )
    # Weight in [0, 1] for coherent mode amount blend:
    # 0.0 = client-driven only, 1.0 = merchant-driven only.

    def __post_init__(self) -> None:
        """Validate coherent-amount blend weight read from configuration."""
        self.merchant_amount_weight = float(self.merchant_amount_weight)
        if not 0.0 <= self.merchant_amount_weight <= 1.0:
            raise ValueError("`syntentic.coherent.merchant_amount_weight` must be in [0, 1].")


@dataclass
class CategoricalConfig:
    """Categorical feature vocabulary and noise settings.

    `cocau_vocab` uses right-exclusive bounds, where 0 is padding.
    `p_noise` defines base categorical noise rate.
    """

    cocau_vocab: tuple[int, int] = (0, 501)
    p_noise: float = 0.15


@dataclass
class OutputConfig:
    """Output base paths for each experiment variant."""

    vanilla_out_path: Path = field(
        default_factory=lambda: _output_path(
            "vanilla", "out_path", Path("data") / "transactions_vanilla.csv",
        ),
    )
    coherent_out_path: Path = field(
        default_factory=lambda: _output_path(
            "coherent", "out_path", Path("data") / "transactions_coherent.csv",
        ),
    )
    simple_out_path: Path = field(
        default_factory=lambda: _output_path(
            "simple", "out_path", Path("data") / "transactions_simple.csv",
        ),
    )
    timing_out_path: Path = field(
        default_factory=lambda: _output_path(
            "timing", "out_path", Path("data") / "transactions_timing.csv",
        ),
    )

    def __post_init__(self) -> None:
        """Normalize configured output paths to `Path` objects.

        Input:
            None.
        Output:
            None.
        What it does:
            Converts output path fields to `Path` for consistent downstream use.
        """
        self.vanilla_out_path = Path(self.vanilla_out_path)
        self.coherent_out_path = Path(self.coherent_out_path)
        self.simple_out_path = Path(self.simple_out_path)
        self.timing_out_path = Path(self.timing_out_path)

    def path_for(self, experiment: str, noise_level: float) -> Path:
        """Return output path for a given experiment variant.

        Input:
            experiment: variant name (`vanilla` or `coherent`).
            noise_level: global noise level used to suffix the output file name.
        Output:
            Path where that variant should be saved.
        What it does:
            Selects the correct output field, injects the noise level in file name,
            and validates supported variants.
        """
        if experiment == "vanilla":
            base_path = self.vanilla_out_path
        elif experiment == "coherent":
            base_path = self.coherent_out_path
        else:
            raise ValueError(f"Unsupported synthetic experiment `{experiment}`.")

        suffix = (base_path.suffix or ".csv").lower()
        if suffix not in _SUPPORTED_OUTPUT_EXTENSIONS:
            raise ValueError(
                f"Unsupported output extension `{suffix}` for `{base_path}`. "
                f"Supported: {sorted(_SUPPORTED_OUTPUT_EXTENSIONS)}.",
            )
        noise_label = f"{max(0.0, min(1.0, float(noise_level))):.2f}".replace(".", "_")
        return base_path.with_name(f"{base_path.stem}_noise_{noise_label}{suffix}")


@dataclass
class DatasetConfig:
    """Top-level composition of all synthetic dataset config sections."""

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    merchants: MerchantConfig = field(default_factory=MerchantConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    amount: AmountConfig = field(default_factory=AmountConfig)
    categorical: CategoricalConfig = field(default_factory=CategoricalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
