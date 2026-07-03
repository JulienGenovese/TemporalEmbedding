"""Datasets and DataLoaders for transaction windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from ...data_schema import DataConfig as TransactionDataConfig
from .encoder import TransactionEncoder
from .features import DEFAULT_FEATURES, FeatureSpec
from .hier_config import HierTransformerConfig


class BaseTransactionDataset:
    """Shared dataframe -> per-client arrays -> padded sample logic."""

    def __init__(
        self,
        df: pd.DataFrame,
        client_col: str,
        timestamp_col: str,
        features: list[FeatureSpec],
        seq_len: int = 32,
    ):
        self.features = features
        self.seq_len = seq_len

        self.clients: list[dict[str, np.ndarray | int]] = []
        self.client_id_lookup: dict[int, object] = {}

        ordered = df.sort_values([client_col, timestamp_col])
        for code, (client_id, group) in enumerate(ordered.groupby(client_col)):
            self.client_id_lookup[code] = client_id
            entry: dict[str, np.ndarray | int] = {
                "client_id": code,
                "n": len(group),
                "timestamp": group[timestamp_col].to_numpy(np.int64, copy=True),
            }
            for feature in self.features:
                entry[feature.name] = TransactionEncoder.prepare_feature_column(
                    feature,
                    group[feature.name],
                )
            self.clients.append(entry)

    def _make_sample(
        self,
        client: dict[str, np.ndarray | int],
        start: int,
        length: int,
    ) -> dict[str, torch.Tensor]:
        end = start + length
        pad = self.seq_len - length

        ts = client["timestamp"][start:end].astype(np.int64)
        delta_t = np.zeros(length, dtype=np.float32)
        if length > 1:
            delta_t[1:] = np.clip(np.diff(ts).astype(np.float32), 0, None)

        sample: dict[str, torch.Tensor] = {
            "delta_t": _pad(delta_t, pad, np.float32),
            "timestamp": _pad(ts, pad, np.int64),
        }
        for feature in self.features:
            values = client[feature.name][start:end]
            dtype = np.float32 if values.dtype.kind == "f" else np.int64
            sample[feature.name] = _pad(values, pad, dtype)

        sample["padding_mask"] = torch.cat([
            torch.zeros(length, dtype=torch.bool),
            torch.ones(pad, dtype=torch.bool),
        ])
        return sample


class TrainTransactionDataset(BaseTransactionDataset, Dataset):
    """Per-client training dataset that emits multiple random windows per client."""

    def __init__(
        self,
        *args,
        train_windows_per_client: int = 2,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_windows_per_client = max(1, int(train_windows_per_client))
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.clients)

    def __getitem__(self, idx: int) -> tuple[list[dict[str, torch.Tensor]], int]:
        client = self.clients[idx]
        windows = [
            self._make_sample(client, *self._sample_window(client))
            for _ in range(self.train_windows_per_client)
        ]
        return windows, int(client["client_id"])

    def _sample_window(self, client: dict[str, np.ndarray | int]) -> tuple[int, int]:
        n = int(client["n"])
        if n <= self.seq_len:
            return 0, n
        start = int(self._rng.integers(0, n - self.seq_len + 1))
        return start, self.seq_len


class PredictionTransactionDataset(BaseTransactionDataset, Dataset):
    """Deterministic per-client windows for inference."""

    def __init__(
        self,
        *args,
        pred_windows_per_client: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pred_windows_per_client = max(1, int(pred_windows_per_client))
        self._windows = self._build_windows()

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[dict[str, torch.Tensor], int, int, int, int]:
        client_idx, window_index, start, length = self._windows[idx]
        client = self.clients[client_idx]

        end = start + length
        ts = client["timestamp"][start:end].astype(np.int64)
        sample = self._make_sample(client, start, length)
        left = int(ts.min()) if length > 0 else 0
        right = int(ts.max()) if length > 0 else 0
        return sample, int(client["client_id"]), window_index, left, right

    def _build_windows(self) -> list[tuple[int, int, int, int]]:
        windows: list[tuple[int, int, int, int]] = []
        for client_idx, client in enumerate(self.clients):
            n = int(client["n"])
            if n <= 0:
                continue
            starts = self._window_starts(n)
            length = min(n, self.seq_len)
            for window_index, start in enumerate(starts):
                windows.append((client_idx, window_index, int(start), length))
        return windows

    def _window_starts(self, n: int) -> np.ndarray:
        if n <= self.seq_len:
            return np.array([0], dtype=np.int64)
        n_possible = n - self.seq_len + 1
        n_windows = min(self.pred_windows_per_client, n_possible)
        return np.linspace(0, n_possible - 1, n_windows, dtype=np.int64)


def _pad(values: np.ndarray, pad: int, dtype) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(values, dtype=dtype))
    if pad:
        tensor = torch.cat([tensor, torch.zeros(pad, dtype=tensor.dtype)])
    return tensor


def collate_train(items: list[tuple[list[dict[str, torch.Tensor]], int]]):
    samples: list[dict[str, torch.Tensor]] = []
    client_ids: list[int] = []
    for windows, client_id in items:
        samples.extend(windows)
        client_ids.extend([client_id] * len(windows))

    keys = samples[0].keys()
    batch = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in keys}
    return batch, torch.tensor(client_ids, dtype=torch.long)


def collate_prediction(items):
    samples, client_codes, window_indices, lefts, rights = zip(*items)
    keys = samples[0].keys()
    batch = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in keys}
    return (
        batch,
        torch.tensor(client_codes, dtype=torch.long),
        torch.tensor(window_indices, dtype=torch.long),
        torch.tensor(lefts, dtype=torch.long),
        torch.tensor(rights, dtype=torch.long),
    )


class DataModule:
    """Callable bundle that produces training/validation DataLoaders."""

    def __init__(
        self,
        args: HierTransformerConfig,
        features: list[FeatureSpec] | None = None,
        data_config: TransactionDataConfig | None = None,
    ):
        self.args = args
        self.path_config = args.paths
        self.pipeline_config = args.data
        self.training_config = args.training
        self.runtime_config = args.runtime
        self.base_features = features if features is not None else DEFAULT_FEATURES
        self.data_config = data_config if data_config is not None else TransactionDataConfig()
        self.features: list[FeatureSpec] | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None

    def __call__(self) -> tuple[DataLoader, DataLoader | None, list[FeatureSpec]]:
        df = self._load_dataframe()
        train_df, val_df = self._split_clients(df)
        logger.info("Fitting per-feature normalizers/vocabularies")
        self.features = TransactionEncoder.fit_features(self.base_features, train_df)
        logger.debug(
            "Feature schema: {}",
            [(feature.__class__.__name__, feature.name) for feature in self.features],
        )

        self.train_loader = self._build_loader(train_df, tag="train")
        self.val_loader = self._build_loader(val_df, tag="val") if val_df is not None else None
        return self.train_loader, self.val_loader, self.features

    def _split_clients(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        cid_col = self.data_config.client_col
        all_clients = df[cid_col].unique()
        val_frac = float(self.training_config.val_frac)
        if val_frac <= 0:
            logger.info("val_frac=0: no validation split")
            return df, None

        rng = np.random.default_rng(self.runtime_config.seed)
        shuffled = rng.permutation(all_clients)
        n_val = int(round(len(shuffled) * val_frac))
        if n_val <= 0:
            logger.warning(
                "val_frac={:.3f} rounds to 0 clients (have {}); skipping val split",
                val_frac,
                len(shuffled),
            )
            return df, None

        val_ids = set(shuffled[:n_val].tolist())
        train_mask = ~df[cid_col].isin(val_ids)
        train_df = df[train_mask].reset_index(drop=True)
        val_df = df[~train_mask].reset_index(drop=True)
        logger.info(
            "Client split: {} train / {} val (val_frac={:.2f}, seed={})",
            train_df[cid_col].nunique(),
            val_df[cid_col].nunique(),
            val_frac,
            self.runtime_config.seed,
        )
        return train_df, val_df

    def _build_loader(self, df: pd.DataFrame, tag: str) -> DataLoader:
        dataset = self._build_dataset(df, tag=tag)
        loader = DataLoader(
            dataset,
            batch_size=max(1, self.pipeline_config.clients_per_batch),
            shuffle=(tag == "train"),
            drop_last=(
                tag == "train"
                and len(dataset) >= self.pipeline_config.clients_per_batch
            ),
            collate_fn=collate_train,
        )
        logger.info(
            "{} loader ready: {} batches/epoch, {} clients/batch, {} train windows/client",
            tag.capitalize(),
            len(loader),
            self.pipeline_config.clients_per_batch,
            self.pipeline_config.train_windows_per_client,
        )
        return loader

    def _load_dataframe(self, path: Path | None = None) -> pd.DataFrame:
        path = Path(path or self.path_config.train_path)
        logger.info("Loading dataframe from {}", path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        if path.is_dir():
            raise ValueError(f"{path} is a directory; pass a single .csv or .parquet file")
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix} ({path})")

        logger.info(
            "Loaded {:,} rows / {} clients",
            len(df),
            df[self.data_config.client_col].nunique(),
        )
        return df

    def _build_dataset(self, df: pd.DataFrame, tag: str = "train") -> TrainTransactionDataset:
        logger.info(
            "Building {} TrainTransactionDataset (seq_len={}, train_windows_per_client={})",
            tag,
            self.pipeline_config.seq_len,
            self.pipeline_config.train_windows_per_client,
        )
        dataset = TrainTransactionDataset(
            df=df,
            client_col=self.data_config.client_col,
            timestamp_col=self.data_config.timestamp_col,
            features=self.features or self.base_features,
            seq_len=self.pipeline_config.seq_len,
            train_windows_per_client=self.pipeline_config.train_windows_per_client,
            seed=self.runtime_config.seed,
        )
        logger.info("{} dataset built: {:,} clients", tag, len(dataset))
        return dataset
