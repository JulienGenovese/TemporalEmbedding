"""Abstract contracts for synthetic dataset experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import config
from ...data_schema import DATA_CONFIG, DataConfig
from .specs import Merchant


class SyntheticExperiment(ABC):
    """Single-class contract for synthetic dataset experiments."""

    splits: tuple[str, ...] = ("train", "pred")
    data_config: DataConfig = DATA_CONFIG
    supported_output_extensions: tuple[str, ...] = (".csv", ".parquet")

    @property
    @abstractmethod
    def experiment(self) -> str:
        """Logical experiment identifier used in logs."""
        raise NotImplementedError

    @property
    @abstractmethod
    def output_section(self) -> str:
        """Config section used to resolve dataset output paths."""
        raise NotImplementedError

    def generate(self) -> dict[str, Path]:
        """Generate every configured split and return a split -> output path map."""
        paths: dict[str, Path] = {}
        for split_index, split in enumerate(self.splits):
            paths[split] = self.save(self.build(self._split_seed(split_index)), split)
        return paths

    @abstractmethod
    def build(self, seed: int) -> pd.DataFrame:
        """Build the full transaction DataFrame for the requested split."""
        raise NotImplementedError

    @abstractmethod
    def sample_timestamps(self, rng: np.random.Generator, n_tx: int, **kwargs) -> np.ndarray:
        """Sample `n_tx` timestamps according to the experiment's temporal logic."""
        raise NotImplementedError
    
    def _cocau_pools(
        self,
        merchants: list[Merchant],
        preferred_cocau: tuple[int, ...],
    ) -> list[np.ndarray]:
        """Per-merchant cocau pools: intersection with cluster, else merchant codes."""
        preferred = set(preferred_cocau)
        pools: list[np.ndarray] = []
        for merchant in merchants:
            inter = tuple(sorted(set(merchant.cocau) & preferred))
            pools.append(np.array(inter if inter else merchant.cocau, dtype=np.int64))
        return pools

    def _output_path(self, split: str) -> Path:
        """Resolve the split-suffixed output path from experiment config."""
        base_path = config.get(self.output_section, "output", value_type=Path)
        suffix = (base_path.suffix or ".csv").lower()
        supported = {ext.lower() for ext in self.supported_output_extensions}
        if suffix not in supported:
            raise ValueError(
                f"Unsupported output extension `{suffix}` for `{base_path}`. Supported: {sorted(supported)}.",
            )
        return base_path.with_name(f"{base_path.stem}_{split}{suffix}")

    def _split_seed(self, split_index: int) -> int:
        """Derive a deterministic, distinct seed for each generated split."""
        base_seed = config.get(self.output_section, "seed", value_type=int)
        return base_seed + split_index

    def save(self, df: pd.DataFrame, split: str) -> Path:
        """Persist a generated dataset and log summary statistics."""
        out_path = self._output_path(split)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix.lower()
        if suffix == ".csv":
            df.to_csv(out_path, index=False)
        elif suffix == ".parquet":
            df.to_parquet(out_path, index=False)
        else:
            raise ValueError(
                f"Unsupported output extension `{suffix}` for `{out_path}`. Supported: ['.csv', '.parquet'].",
            )

        per_client = df.groupby(self.data_config.client_col).size()
        logger.info(
            "Saved {:,} rows × {} cols → {} (experiment={}, split={})",
            len(df),
            len(df.columns),
            out_path,
            self.experiment,
            split,
        )
        logger.info(
            "Client transactions: {} (min={}, max={}, mean={:.1f} tx/client)",
            df[self.data_config.client_col].nunique(),
            per_client.min(),
            per_client.max(),
            per_client.mean(),
        )
        return out_path