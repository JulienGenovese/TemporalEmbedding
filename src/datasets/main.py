"""Unified entrypoints for synthetic dataset generation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .experiments.simple_spatial import generate as generate_simple_spatial
from .experiments.simple_timing import generate as generate_simple_timing
from .experiments.simple_delta import generate as generate_simple_delta
from .utils.config import DatasetConfig

DatasetType = Literal["simple_spatial", "simple_timing", "simple_delta"]


def generate(
    dataset_type: DatasetType,
    config: DatasetConfig | None = None,
) -> dict[str, Path]:
    """Generate a synthetic dataset (train + pred splits) for the requested variant.

    Returns a mapping of split name (`train`/`pred`) to the generated file path.
    """
    if dataset_type == "simple_spatial":
        return generate_simple_spatial(config=config)
    if dataset_type == "simple_timing":
        return generate_simple_timing(config=config)
    if dataset_type == "simple_delta":
        return generate_simple_delta(config=config)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")
