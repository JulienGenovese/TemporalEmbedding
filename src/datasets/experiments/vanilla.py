

"""Vanilla synthetic dataset entrypoint and wrapper classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.datasets.generators.cluster import cluster_types
from src.datasets.utils.entities import ClientType

from ..generators.transactions import (
    AmountGenerator,
)
from .common import SyntheticTransactionDatasetCore
from ..utils.config import DatasetConfig


def default_amount_generator_builder(
    amount_rng: np.random.Generator,
    cfg: DatasetConfig,
) -> AmountGenerator:
    """Build the default amount generator for vanilla mode.

    Input:
        amount_rng: NumPy random generator used by amount sampling.
        cfg: full dataset configuration.
    Output:
        Amount generator instance.
    What it does:
        Returns the standard lognormal amount generator that is independent of
        merchant metadata.
    """
    return AmountGenerator(amount_rng, cfg.amount)

class VanillaSyntheticTransactionDataset(SyntheticTransactionDatasetCore):
    """Synthetic dataset variant with client-only lognormal amounts."""

    def __init__(
        self,
        config: DatasetConfig,
        client_types: list[ClientType] | None = None,
    ) -> None:
        """Initialize vanilla synthetic dataset builder.

        Input:
            config: full dataset configuration.
            client_types: optional explicit client cluster definitions.
        Output:
            None.
        What it does:
            Wires the shared dataset core with vanilla amount generation behavior.
        """
        super().__init__(
            config=config,
            client_types=client_types,
            amount_generator_builder=default_amount_generator_builder,
            amount_requires_merchant=False,
            experiment="vanilla",
        )

def generate(
    config: DatasetConfig | None = None,
    client_types: list[ClientType] | None = None,
) -> Path:
    """Generate and save the vanilla synthetic dataset.

    Input:
        config: optional explicit dataset configuration.
        client_types: optional explicit client cluster definitions.
    Output:
        Path to the generated CSV file.
    What it does:
        Creates default config and clusters, builds the dataset, then saves it.
    """
    resolved_config = config or DatasetConfig()
    std = VanillaSyntheticTransactionDataset(
        config=resolved_config,
        client_types=(
            client_types
            if client_types is not None
            else cluster_types(resolved_config.merchants)
        ),
    )
    return std.generate_and_save()


if __name__ == "__main__":
    generate()
