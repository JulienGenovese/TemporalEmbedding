

"""Configuration for synthetic dataset generation, split by subsystem.

This module defines:
  - SamplingConfig: dataset size and per-client allocation controls (per split)
  - MerchantConfig: themed and shared merchant pools
  - TimeConfig: time origin and granularity constants
  - OutputConfig: output paths by experiment variant

DatasetConfig composes all sections and acts as the entry configuration object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..generators.merchant import MerchantConfig

from src.config import config



_SAMPLING_SECTION = "dataset.sampling"
# Every synthetic dataset is materialised as two independent draws from the same
# distributions: a `train` split and a `pred` split. Each split has its own
# sub-section under `[dataset.sampling]` (volume + seed).
SPLITS: tuple[str, ...] = ("train", "pred")
_SPLIT_DEFAULTS: dict[str, dict[str, int | float]] = {
    "train": {
        "n_transactions": 400_000,
        "n_clients": 4_000,
        "alpha_dirichlet": 1.5,
        "min_tx_per_client": 50,
        "seed": 42,
    },
    "pred": {
        "n_transactions": 100_000,
        "n_clients": 1_000,
        "alpha_dirichlet": 1.5,
        "min_tx_per_client": 50,
        "seed": 1234,
    },
}
_EXPERIMENT_SECTIONS: dict[str, tuple[str, ...]] = {
    "simple_spatial": (
        "syntentic.simple_spatial",
        "syntetic.simple_spatial",
        "synthetic.simple_spatial",
        "syntheticData.output.simple_spatial",
        "synteticData.output.simple_spatial",
    ),
    "simple_timing": (
        "syntentic.simple_timing",
        "syntetic.simple_timing",
        "synthetic.simple_timing",
        "syntheticData.output.simple_timing",
        "synteticData.output.simple_timing",
    ),
    "simple_delta": (
        "syntentic.simple_delta",
        "syntetic.simple_delta",
        "synthetic.simple_delta",
        "syntheticData.output.simple_delta",
        "synteticData.output.simple_delta",
    ),
}
_SUPPORTED_OUTPUT_EXTENSIONS = {".csv", ".parquet"}
# Maps each experiment to its base output-path field on `OutputConfig`.
_OUTPUT_EXPERIMENT_FIELDS: dict[str, str] = {
    "simple_spatial": "simple_spatial_out_path",
    "simple_timing": "simple_timing_out_path",
    "simple_delta": "simple_delta_out_path",
}
_MISSING = object()


def _sampling_sections(split: str | None) -> tuple[str, ...]:
    """Return the config sections to read a sampling key from, most specific first.

    Input:
        split: split name (`train`/`pred`) or None for the legacy flat section.
    Output:
        Tuple of dotted section paths tried in order; the split sub-section takes
        precedence over the shared parent `[dataset.sampling]` section.
    What it does:
        Lets a split inherit any value not overridden under its own sub-section.
    """
    if split is None:
        return (_SAMPLING_SECTION,)
    return (f"{_SAMPLING_SECTION}.{split}", _SAMPLING_SECTION)


def _sampling_int(key: str, default: int, split: str | None = None) -> int:
    """Read an integer value from the (split-aware) sampling config section.

    Input:
        key: sampling config key.
        default: fallback value if key is absent in every candidate section.
        split: optional split name selecting the `[dataset.sampling.<split>]` sub-section.
    Output:
        Integer config value.
    What it does:
        Walks the candidate sections (split sub-section, then parent) and returns
        the first present value, validating it is an integer (not bool).
    """
    for section in _sampling_sections(split):
        try:
            value = config.get(section, key, _MISSING)
        except KeyError:
            continue
        if value is _MISSING:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"`{section}.{key}` must be an integer, got {type(value).__name__}.")
        return value
    return default


def _sampling_float(key: str, default: float, split: str | None = None) -> float:
    """Read a numeric value from the (split-aware) sampling config section.

    Input:
        key: sampling config key.
        default: fallback value if key is absent in every candidate section.
        split: optional split name selecting the `[dataset.sampling.<split>]` sub-section.
    Output:
        Float config value.
    What it does:
        Walks the candidate sections (split sub-section, then parent) and returns
        the first present value, validating it is numeric (int/float, excluding bool).
    """
    for section in _sampling_sections(split):
        try:
            value = config.get(section, key, _MISSING)
        except KeyError:
            continue
        if value is _MISSING:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"`{section}.{key}` must be numeric, got {type(value).__name__}.")
        return float(value)
    return default


def _output_path(variant: str, key: str, default: Path) -> Path:
    """Read and normalize an output path from config.

    Input:
        variant: experiment variant name (for example `simple_spatial`).
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
    """Sampling volume and transaction allocation settings for one split.

    A synthetic dataset is generated once per split (`train`/`pred`) from the same
    distributions; only the volume and `seed` carried here differ between splits,
    so the two files are independent draws of the same generative process.
    """

    n_transactions: int = 400_000
    n_clients: int = 4_000
    alpha_dirichlet: float = 1.5  # Smaller alpha => more skewed client activity.
    min_tx_per_client: int = 50
    seed: int = 42

    @classmethod
    def from_split(cls, split: str) -> "SamplingConfig":
        """Build the sampling config for a split from `[dataset.sampling.<split>]`.

        Input:
            split: split name (`train` or `pred`).
        Output:
            SamplingConfig populated from the split sub-section, falling back to the
            shared `[dataset.sampling]` section and then to per-split defaults.
        What it does:
            Reads each volume/seed key with split-aware precedence so the two splits
            can differ in size and seed while sharing whatever is left on the parent.
        """
        defaults = _SPLIT_DEFAULTS.get(split, _SPLIT_DEFAULTS["train"])
        return cls(
            n_transactions=_sampling_int("n_transactions", int(defaults["n_transactions"]), split),
            n_clients=_sampling_int("n_clients", int(defaults["n_clients"]), split),
            alpha_dirichlet=_sampling_float("alpha_dirichlet", float(defaults["alpha_dirichlet"]), split),
            min_tx_per_client=_sampling_int("min_tx_per_client", int(defaults["min_tx_per_client"]), split),
            seed=_sampling_int("seed", int(defaults["seed"]), split),
        )


@dataclass
class TimeConfig:
    """Temporal anchoring and granularity constants."""

    ts_base: int = 1_577_836_800  # 2020-01-01 00:00 UTC
    ts_range: int = 4 * 365 * 24 * 3600  # About 4 years in seconds
    day: int = 86_400


@dataclass
class OutputConfig:
    """Output base paths for each experiment variant."""

    simple_spatial_out_path: Path = field(
        default_factory=lambda: _output_path(
            "simple_spatial", "out_path", Path("data") / "transactions_simple_spatial.csv",
        ),
    )
    simple_timing_out_path: Path = field(
        default_factory=lambda: _output_path(
            "simple_timing", "out_path", Path("data") / "transactions_simple_timing.csv",
        ),
    )
    simple_delta_out_path: Path = field(
        default_factory=lambda: _output_path(
            "simple_delta", "out_path", Path("data") / "transactions_simple_delta.csv",
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
        self.simple_spatial_out_path = Path(self.simple_spatial_out_path)
        self.simple_timing_out_path = Path(self.simple_timing_out_path)
        self.simple_delta_out_path = Path(self.simple_delta_out_path)

    def split_path(self, experiment: str, split: str) -> Path:
        """Return the output path for one (experiment, split) combination.

        Input:
            experiment: variant name (`simple_spatial`, `simple_timing`, `simple_delta`).
            split: split name (`train` or `pred`) injected into the file stem.
        Output:
            Path where that experiment's split should be saved, e.g.
            `data/transactions_simple_spatial_pred.csv`.
        What it does:
            Selects the base path for the experiment, validates the extension, and
            builds `<stem>_<split><ext>`.
        """
        try:
            attr = _OUTPUT_EXPERIMENT_FIELDS[experiment]
        except KeyError:
            raise ValueError(f"Unsupported synthetic experiment `{experiment}`.")

        base_path: Path = getattr(self, attr)
        suffix = (base_path.suffix or ".csv").lower()
        if suffix not in _SUPPORTED_OUTPUT_EXTENSIONS:
            raise ValueError(
                f"Unsupported output extension `{suffix}` for `{base_path}`. "
                f"Supported: {sorted(_SUPPORTED_OUTPUT_EXTENSIONS)}.",
            )

        return base_path.with_name(f"{base_path.stem}_{split}{suffix}")


@dataclass
class DatasetConfig:
    """Top-level composition of all synthetic dataset config sections.

    Sampling is split per generation target: `sampling_train` and `sampling_pred`
    each carry their own volume + seed, while every other section (merchants, time)
    defines the shared distributions both splits draw from.
    """

    sampling_train: SamplingConfig = field(default_factory=lambda: SamplingConfig.from_split("train"))
    sampling_pred: SamplingConfig = field(default_factory=lambda: SamplingConfig.from_split("pred"))
    merchants: MerchantConfig = field(default_factory=MerchantConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def sampling_for(self, split: str) -> SamplingConfig:
        """Return the sampling config for a split.

        Input:
            split: split name (`train` or `pred`).
        Output:
            The matching :class:`SamplingConfig`.
        What it does:
            Routes to the per-split sampling config, raising on unknown splits.
        """
        if split == "train":
            return self.sampling_train
        if split == "pred":
            return self.sampling_pred
        raise ValueError(f"Unsupported split `{split}`. Expected one of {SPLITS}.")
