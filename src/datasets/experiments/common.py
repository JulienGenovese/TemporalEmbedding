"""Shared orchestration layer for synthetic transaction datasets.

Provides the common base every experiment inherits from. Concrete experiments
(:mod:`simple_spatial`, :mod:`simple_timing`, :mod:`simple_delta`) implement
``build()`` (the DataFrame) and ``_output_path()`` (where it is written);
serialization, directory creation and summary logging live here once.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import pandas as pd
from loguru import logger

from ...constant import DATA_CONFIG, DataConfig
from ..generators.abc import Build
from ..utils.config import DatasetConfig


class BaseSyntheticDataset(Build):
    """Common interface shared by every synthetic transaction dataset.

    Concrete experiments only implement `build()` (the DataFrame) and
    `_output_path()` (where it is written). Serialization, directory creation and
    summary logging are provided once here via `save()` / `generate_and_save()`,
    so no experiment reimplements them. Subclasses may override `_prepare_for_save`
    to decorate the frame and `_log_context` to add detail to the save log.
    """

    experiment: str = "base"
    split: str = "train"
    data_config: DataConfig = DATA_CONFIG
    config: DatasetConfig

    @abstractmethod
    def build(self) -> pd.DataFrame:
        """Build the full transaction DataFrame for the experiment."""
        raise NotImplementedError

    @abstractmethod
    def _output_path(self) -> Path:
        """Return the destination path for the generated dataset."""
        raise NotImplementedError

    def _prepare_for_save(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hook to decorate the frame before writing (default: unchanged)."""
        return df

    def _log_context(self) -> str:
        """Hook for extra context appended to the save summary log."""
        return ""

    def save(self, df: pd.DataFrame) -> Path:
        """Persist a generated dataset and log summary statistics.

        Input:
            df: generated transaction DataFrame.
        Output:
            Path of the saved dataset file.
        What it does:
            Resolves the experiment output path, writes CSV/parquet by extension,
            logs cardinality stats, and returns the final path.
        """
        out_path = self._output_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_to_save = self._prepare_for_save(df)
        suffix = out_path.suffix.lower()
        if suffix == ".csv":
            df_to_save.to_csv(out_path, index=False)
        elif suffix == ".parquet":
            df_to_save.to_parquet(out_path, index=False)
        else:
            raise ValueError(
                f"Unsupported output extension `{suffix}` for `{out_path}`. Supported: ['.csv', '.parquet'].",
            )

        per_client = df.groupby(self.data_config.client_col).size()
        logger.info(
            "Saved {:,} rows × {} cols → {} (experiment={}, split={}{})",
            len(df),
            len(df_to_save.columns),
            out_path,
            self.experiment,
            self.split,
            self._log_context(),
        )
        logger.info(
            "Client transactions: {} (min={}, max={}, mean={:.1f} tx/client)",
            df[self.data_config.client_col].nunique(),
            per_client.min(),
            per_client.max(),
            per_client.mean(),
        )
        return out_path

    def generate_and_save(self) -> Path:
        """Build the dataset and persist it in one call."""
        return self.save(self.build())
