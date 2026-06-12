"""Unified entrypoints for synthetic dataset generation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .experiments.coherent_sintetic import generate as generate_coherent
from .utils.config import DatasetConfig
from .utils.entities import ClientType
from .experiments.vanilla import generate as generate_vanilla

DatasetType = Literal["vanilla", "coherent"]


def generate(
    dataset_type: DatasetType,
    config: DatasetConfig | None = None,
    client_types: list[ClientType] | None = None,
    merchant_amount_weight: float | None = None,
) -> Path:
    """Generate a synthetic dataset for the requested variant."""
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
    raise ValueError(f"Unsupported dataset type: {dataset_type}")
