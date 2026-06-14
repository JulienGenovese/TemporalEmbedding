

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
from ..utils.config import SPLITS, DatasetConfig


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
        split: str = "train",
    ) -> None:
        """Initialize vanilla synthetic dataset builder.

        Input:
            config: full dataset configuration.
            client_types: optional explicit client cluster definitions.
            split: sampling split to draw (`train` or `pred`).
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
            split=split,
        )

def generate(
    config: DatasetConfig | None = None,
    client_types: list[ClientType] | None = None,
) -> dict[str, Path]:
    """Generate and save the vanilla synthetic dataset for every split.

    Input:
        config: optional explicit dataset configuration.
        client_types: optional explicit client cluster definitions.
    Output:
        Mapping of split name (`train`/`pred`) to the generated file path.
    What it does:
        Builds the cluster definitions once, then materialises one file per split
        (`train` and `pred`) — independent draws from the same distributions.
    """
    resolved_config = config or DatasetConfig()
    shared_client_types = (
        client_types
        if client_types is not None
        else cluster_types(resolved_config.merchants)
    )
    paths: dict[str, Path] = {}
    for split in SPLITS:
        std = VanillaSyntheticTransactionDataset(
            config=resolved_config,
            client_types=shared_client_types,
            split=split,
        )
        paths[split] = std.generate_and_save()
    return paths


if __name__ == "__main__":
    generate()
