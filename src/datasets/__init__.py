"""Synthetic dataset package (vanilla/coherent builders and shared utilities)."""

from .experiments.coherent_sintetic import CoherentSyntheticTransactionDataset
from .main import DatasetType, generate
from .experiments.vanilla import VanillaSyntheticTransactionDataset

__all__ = [
    "CoherentSyntheticTransactionDataset",
    "VanillaSyntheticTransactionDataset",
    "DatasetType",
    "generate",
]
