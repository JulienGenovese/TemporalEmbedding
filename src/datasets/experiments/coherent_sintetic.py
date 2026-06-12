"""Coherent synthetic dataset entrypoint and wrappers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..generators.abc import Generator
from ..generators.transactions import CoherentAmountGenerator
from .common import AmountGeneratorBuilder, SyntheticTransactionDatasetCore
from ..utils.config import DatasetConfig
from ..utils.entities import ClientType


def _resolve_merchant_amount_weight(
    config: DatasetConfig,
    merchant_amount_weight: float | None,
) -> float:
    """Resolve coherent amount blend weight from argument or config.

    Input:
        config: dataset configuration source.
        merchant_amount_weight: optional runtime override.
    Output:
        Validated blend weight in [0, 1].
    What it does:
        Uses explicit argument when provided, otherwise falls back to
        `config.amount.merchant_amount_weight`.
    """
    resolved_weight = (
        config.amount.merchant_amount_weight
        if merchant_amount_weight is None
        else float(merchant_amount_weight)
    )
    if not 0.0 <= resolved_weight <= 1.0:
        raise ValueError("`merchant_amount_weight` must be in [0, 1].")
    return float(resolved_weight)


def _coherent_amount_builder(
    merchant_amount_weight: float,
) -> AmountGeneratorBuilder:
    """Create a builder for coherent amount generation.

    Input:
        merchant_amount_weight: blend weight for merchant-driven amount signal.
    Output:
        Factory function that builds a coherent amount generator.
    What it does:
        Captures the merchant weight and returns a closure used by the dataset
        core to instantiate the coherent amount generator.
    """
    def _builder(rng: np.random.Generator, config: DatasetConfig) -> Generator:
        """Instantiate the coherent amount generator.

        Input:
            rng: NumPy random generator.
            config: full dataset configuration.
        Output:
            CoherentAmountGenerator instance.
        What it does:
            Builds the coherent amount generator using captured merchant weight.
        """
        return CoherentAmountGenerator(
            rng=rng,
            amount=config.amount,
            merchant_weight=merchant_amount_weight,
        )

    return _builder


class CoherentSyntheticTransactionDataset(SyntheticTransactionDatasetCore):
    """Synthetic dataset variant blending client and merchant amount signals."""

    def __init__(
        self,
        config: DatasetConfig,
        client_types: list[ClientType] | None = None,
        merchant_amount_weight: float | None = None,
    ) -> None:
        """Initialize coherent synthetic dataset builder.

        Input:
            config: full dataset configuration.
            client_types: optional explicit client cluster definitions.
            merchant_amount_weight: optional merchant contribution override.
                If omitted, reads from `config.amount.merchant_amount_weight`.
        Output:
            None.
        What it does:
            Wires the shared dataset core with coherent amount generation.
        """
        resolved_weight = _resolve_merchant_amount_weight(config, merchant_amount_weight)
        super().__init__(
            config=config,
            client_types=client_types,
            amount_generator_builder=_coherent_amount_builder(resolved_weight),
            amount_requires_merchant=True,
            experiment="coherent",
        )


def generate(
    merchant_amount_weight: float | None = None,
    config: DatasetConfig | None = None,
    client_types: list[ClientType] | None = None,
) -> Path:
    """Generate and save the coherent synthetic dataset.

    Input:
        merchant_amount_weight: optional merchant contribution override.
            If omitted, reads from `config.amount.merchant_amount_weight`.
        config: optional explicit dataset configuration.
        client_types: optional explicit client cluster definitions.
    Output:
        Path to the generated CSV file.
    What it does:
        Creates config, builds coherent dataset variant, and saves it to disk.
    """
    resolved_config = config or DatasetConfig()
    std = CoherentSyntheticTransactionDataset(
        config=resolved_config,
        client_types=client_types,
        merchant_amount_weight=merchant_amount_weight,
    )
    return std.generate_and_save()


if __name__ == "__main__":
    generate()
