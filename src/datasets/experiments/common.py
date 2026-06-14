"""Shared orchestration layer for synthetic transaction datasets.

This module wires reusable generator components and exposes a common core used
by both dataset variants (`vanilla` and `coherent`).
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from ...constant import DATA_CONFIG, DataConfig
from ..generators.abc import Build, Generator, TransactionRow
from ..generators.cluster import (
    ClusterToClientGenerator,
    FingerprintGenerator,
    cluster_types,
)
from ..generators.merchant import MerchantSelector
from ..generators.timestamp import TimestampGenerator
from ..generators.transactions import (
    AmountGenerator,
    RefundGenerator,
    TransactionCountAllocator,
)
from ..utils.config import DatasetConfig
from ..utils.entities import ClientType, Merchant

AmountGeneratorBuilder = Callable[[np.random.Generator, DatasetConfig], Generator]



class ClientTransactionGenerator:
    """Generate all transaction rows for a single client."""

    def __init__(
        self,
        rng: np.random.Generator,
        fingerprint_gen: Generator,
        timestamp_gen: Generator,
        amount_gen: Generator,
        merchant_gen: Generator,
        refund_gen: Generator,
        data_config: DataConfig,
        global_amount_mu: float,
        amount_requires_merchant: bool,
        p_offpattern: float,
        p_refund: float,
    ) -> None:
        """Initialize per-client transaction generation pipeline.

        Input:
            rng: shared NumPy random generator.
            fingerprint_gen: generator producing a stable client fingerprint.
            timestamp_gen: generator producing transaction timestamps.
            amount_gen: generator producing transaction amounts/signs.
            merchant_gen: generator selecting merchant metadata per transaction.
            refund_gen: generator creating optional refund events.
            data_config: output schema configuration (column names/order).
            global_amount_mu: fallback amount baseline for off-pattern behavior.
            amount_requires_merchant: whether amount generation needs merchant info.
            p_offpattern: probability that a transaction ignores client fingerprint.
            p_refund: probability of creating refund for debit transactions.
        Output:
            None.
        What it does:
            Stores all generation dependencies and noise controls for one client.
        """
        self._rng = rng
        self._fingerprint_gen = fingerprint_gen
        self._timestamp_gen = timestamp_gen
        self._amount_gen = amount_gen
        self._merchant_gen = merchant_gen
        self._refund_gen = refund_gen
        self._data_config = data_config
        self._global_amount_mu = global_amount_mu
        self._amount_requires_merchant = amount_requires_merchant
        self._p_offpattern = p_offpattern
        self._p_refund = p_refund

    def _pick_cocau(self, merchant: Merchant, ctype: ClientType) -> int:
        """Pick a cocau code compatible with merchant and client preferences.

        Input:
            merchant: selected merchant for the transaction.
            ctype: client cluster providing preferred cocau codes.
        Output:
            One integer cocau code.
        What it does:
            Uses intersection between merchant-allowed and cluster-preferred codes;
            if empty, falls back to merchant-allowed codes.
        """
        available = set(merchant.cocau)
        preferred = set(ctype.preferred_cocau)
        intersection = tuple(sorted(available & preferred))
        pool = intersection if intersection else merchant.cocau
        return int(self._rng.choice(pool))

    def _build_row(
        self,
        client_id: int,
        cluster: str,
        ts: int,
        amount: float,
        merchant: Merchant,
        cocau: int,
    ) -> TransactionRow:
        """Build one transaction row in output schema format.

        Input:
            client_id: synthetic client identifier.
            cluster: client cluster label.
            ts: transaction timestamp (Unix seconds).
            amount: signed transaction amount.
            merchant: selected merchant metadata.
            cocau: categorical code for the transaction.
        Output:
            Dictionary representing one transaction row.
        What it does:
            Maps generated values to configured output column names.
        """
        return {
            self._data_config.client_col: client_id,
            self._data_config.cluster_col: cluster,
            self._data_config.timestamp_col: int(ts),
            self._data_config.amount_col: float(amount),
            self._data_config.merchant_col: merchant.name,
            self._data_config.cocau_col: int(cocau),
        }

    def generate(
        self,
        client_id: int,
        n_tx: int,
        ctype: ClientType,
        noise_level: float,
    ) -> list[TransactionRow]:
        """Generate all rows for one client, including optional refunds.

        Input:
            client_id: synthetic client identifier.
            n_tx: number of primary transactions to generate.
            ctype: assigned client cluster.
            noise_level: global noise control used by timestamp generation.
        Output:
            List of transaction rows sorted by timestamp.
        What it does:
            1) Samples client fingerprint and timestamps.
            2) For each timestamp, samples off-pattern behavior, amount, merchant,
               and cocau.
            3) Optionally appends refund events for debit transactions.
            4) Sorts rows by timestamp before returning.
        """
        fp = self._fingerprint_gen.generate(ctype)
        timestamps = self._timestamp_gen.generate(n_tx, ctype, noise_level)
        cluster_name = ctype.name

        rows: list[TransactionRow] = []
        for ts in timestamps:
            off = self._rng.random() < self._p_offpattern
            mu = self._global_amount_mu if off else fp.amount_mu

            if self._amount_requires_merchant:
                merchant = self._merchant_gen.generate(off, ctype.merchants, fp.fav_merchants)
                amount, sign = self._amount_gen.generate(mu, merchant)
            else:
                amount, sign = self._amount_gen.generate(mu)
                merchant = self._merchant_gen.generate(off, ctype.merchants, fp.fav_merchants)

            cocau = self._pick_cocau(merchant, ctype)
            rows.append(self._build_row(client_id, cluster_name, int(ts), amount, merchant, cocau))

            if sign < 0 and self._rng.random() < self._p_refund:
                refund_ts, refund_amt = self._refund_gen.generate(ts, amount)
                rows.append(
                    self._build_row(
                        client_id,
                        cluster_name,
                        refund_ts,
                        refund_amt,
                        merchant,
                        cocau,
                    )
                )

        ts_col = self._data_config.timestamp_col
        rows.sort(key=lambda row: int(row[ts_col]))
        return rows


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


class SyntheticTransactionDatasetCore(BaseSyntheticDataset):
    """Common core used to build synthetic datasets across variants."""

    def __init__(
        self,
        config: DatasetConfig,
        client_types: list[ClientType] | None = None,
        amount_generator_builder: AmountGeneratorBuilder | None = None,
        amount_requires_merchant: bool = False,
        experiment: str = "vanilla",
        split: str = "train",
    ) -> None:
        """Initialize the dataset core and all component generators.

        Input:
            config: full dataset configuration.
            client_types: optional list of client clusters used by the generator.
                If omitted, default clusters are built from merchant config.
            amount_generator_builder: factory for amount generator implementation.
            amount_requires_merchant: whether amount generation depends on merchant.
            experiment: output variant name (`vanilla` or `coherent`).
            split: which sampling split to draw (`train` or `pred`); selects the
                per-split volume/seed while every distribution stays shared.
        Output:
            None.
        What it does:
            Creates shared RNG state, computes global priors, and wires all
            generators used during dataset construction.
        """
        if experiment not in {"vanilla", "coherent"}:
            raise ValueError(f"Unsupported synthetic experiment `{experiment}`.")

        self.config = config
        self.experiment = experiment
        self.split = split
        self.sampling = config.sampling_for(split)
        self.data_config = DATA_CONFIG
        self.client_types = (
            list(client_types)
            if client_types is not None
            else cluster_types(config.merchants)
        )
        if not self.client_types:
            raise ValueError("`client_types` cannot be empty.")
        self.seed = self.sampling.seed
        rng = np.random.default_rng(self.seed)
        self._rng = rng

        global_amount_mu = float(np.mean([t.amount_mu for t in self.client_types]))
        build_amount_generator = amount_generator_builder or (
            lambda amount_rng, dataset_cfg: AmountGenerator(amount_rng, dataset_cfg.amount)
        )

        self._count_allocator: Generator = TransactionCountAllocator(rng, self.sampling)
        self._cluster_assigner: Generator = ClusterToClientGenerator(rng, self.client_types)
        self._client_gen = ClientTransactionGenerator(
            rng=rng,
            fingerprint_gen=FingerprintGenerator(rng, config.noise.sigma_spending),
            timestamp_gen=TimestampGenerator(rng, config.time),
            amount_gen=build_amount_generator(rng, config),
            merchant_gen=MerchantSelector(rng, config.merchants, config.noise.p_global_merchant),
            refund_gen=RefundGenerator(rng, config.time.day),
            data_config=self.data_config,
            global_amount_mu=global_amount_mu,
            amount_requires_merchant=amount_requires_merchant,
            p_offpattern=config.noise.p_offpattern,
            p_refund=config.noise.p_refund,
        )

    def build(self) -> pd.DataFrame:
        """Build the full synthetic transaction dataset.

        Input:
            None.
        Output:
            Pandas DataFrame containing generated transactions.
        What it does:
            Allocates per-client transaction counts, assigns cluster types,
            generates each client's rows, and returns a sorted DataFrame.
        """
        sampling = self.sampling
        noise_level = self.config.noise.noise_level

        counts = self._count_allocator.generate(sampling.n_transactions, sampling.n_clients)
        type_idx = self._cluster_assigner.generate(sampling.n_clients)

        rows: list[TransactionRow] = []
        for client_id, n in enumerate(counts):
            ctype = self._cluster_assigner.type_for(type_idx[client_id])
            rows.extend(self._client_gen.generate(client_id, int(n), ctype, noise_level))

        df = pd.DataFrame(rows, columns=self.data_config.transaction_cols)
        return df.sort_values(self.data_config.transaction_sort_cols).reset_index(drop=True)

    def _output_path(self) -> Path:
        """Resolve the split- and noise-suffixed output path for this variant."""
        return self.config.output.split_path(
            self.experiment, self.split, self.config.noise.noise_level,
        )

    def _prepare_for_save(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach the `noise_level` column consumed downstream."""
        return df.assign(noise_level=self.config.noise.noise_level)

    def _log_context(self) -> str:
        """Add the noise level to the save summary log."""
        return f", noise_level={self.config.noise.noise_level:.2f}"
