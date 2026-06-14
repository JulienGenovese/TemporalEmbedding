"""Unified entrypoints for synthetic dataset generation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .experiments.coherent_sintetic import generate as generate_coherent
from .experiments.simple_spatial import generate as generate_simple_spatial
from .experiments.simple_timing import generate as generate_simple_timing
from .utils.config import DatasetConfig
from .utils.entities import ClientType
from .experiments.vanilla import generate as generate_vanilla

DatasetType = Literal["vanilla", "coherent", "simple_spatial", "simple_timing"]


def generate(
    dataset_type: DatasetType,
    config: DatasetConfig | None = None,
    client_types: list[ClientType] | None = None,
    merchant_amount_weight: float | None = None,
) -> dict[str, Path]:
    """Generate a synthetic dataset (train + pred splits) for the requested variant.

    Returns a mapping of split name (`train`/`pred`) to the generated file path.
    """
    if dataset_type == "vanilla":
        return generate_vanilla(
            config=config, 
            client_types=client_types,
            )
    if dataset_type == "coherent":
        return generate_coherent(
            merchant_amount_weight=merchant_amount_weight,
            config=config,
            client_types=client_types,
        )
    if dataset_type == "simple_spatial":
        return generate_simple_spatial(config=config)
    if dataset_type == "simple_timing":
        return generate_simple_timing(config=config)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")
