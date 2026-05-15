"""Dataset & DataLoader: CSV of transactions → windowed batches per client.

Reads the file produced by :mod:`src.make_dataset`, groups rows by
``client_id``, and yields fixed-length sequence windows ready to feed into
:class:`src.model.TransactionTransformer`.

The two windows-per-client batch layout below guarantees in-batch positive
pairs for the InfoNCE contrastive head.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Sampler

from .encoder import (
    FeatureSpec, HighCardCategoricalFeature, NumericFeature,
)

from .config import FEATURE_COLS, DataConfig

if TYPE_CHECKING:
    from .config import TrainingConfig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TransactionDataset(Dataset):
    """Per-client windowed view over the transactions datafile.

    Each ``__getitem__`` returns a single fixed-length window (``seq_len``
    transactions) drawn from a single client.  Sequences shorter than
    ``seq_len`` are right-padded; ``padding_mask`` marks the padded slots.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        client_col: str,
        timestamp_col: str,
        feature_cols: list[str],
        seq_len: int = 32,
        windows_per_client: int = 4,
        seed: int = 0,
    ):
        self.client_col = client_col
        self.timestamp_col = timestamp_col
        self.feature_cols = list(feature_cols)
        self.seq_len = seq_len # quante transazioni per finestra
        self.windows_per_client = windows_per_client # quante finestre per ogni cliente
        self._rng = np.random.default_rng(seed)

        # Resolve per-feature numpy dtype from the source DataFrame.
        # float -> float32, integer -> int64, object/string -> int64 (FNV-1a hashed).
        self._col_dtypes: dict[str, type] = {}
        self._col_is_string: dict[str, bool] = {}
        for col in self.feature_cols:
            s = df[col]
            if pd.api.types.is_float_dtype(s):
                self._col_dtypes[col] = np.float32
                self._col_is_string[col] = False
            elif pd.api.types.is_integer_dtype(s):
                self._col_dtypes[col] = np.int64
                self._col_is_string[col] = False
            elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                self._col_dtypes[col] = np.int64
                self._col_is_string[col] = True
            else:
                raise TypeError(f"Unsupported dtype for column '{col}': {s.dtype}")

        # Pre-compute per-client field arrays once
        self.clients: list[dict[str, np.ndarray | int]] = []
        for cid, g in df.sort_values([client_col, timestamp_col]).groupby(client_col):
            entry: dict[str, np.ndarray | int] = {
                "client_id": int(cid),
                "n":         len(g),
                "timestamp": g[timestamp_col].to_numpy(np.int64, copy=True),
            }
            for col in self.feature_cols:
                if self._col_is_string[col]:
                    entry[col] = np.array(
                        [HighCardCategoricalFeature._to_int(v) for v in g[col]],
                        dtype=np.int64,
                    )
                else:
                    entry[col] = g[col].to_numpy(self._col_dtypes[col], copy=True)
            self.clients.append(entry)

        # Flat index: (client_idx, _) — start position is sampled at fetch time
        self._index: list[int] = [
            ci for ci in range(len(self.clients)) for _ in range(windows_per_client)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], int]:
        ci = self._index[idx]
        c = self.clients[ci]
        T = self.seq_len
        n = int(c["n"])

        if n <= T:
            start, length = 0, n
        else:
            start = int(self._rng.integers(0, n - T + 1))
            length = T
        end = start + length
        pad = T - length

        ts = c["timestamp"][start:end].astype(np.int64)
        # delta_t in seconds; first transaction of the window → 0
        delta_t = np.zeros(length, dtype=np.float32)
        if length > 1:
            delta_t[1:] = np.clip(np.diff(ts).astype(np.float32), 0, None)

        sample: dict[str, torch.Tensor] = {
            "delta_t":   _pad_float(delta_t, pad),
            "timestamp": _pad_long(ts, pad),
        }
        for col in self.feature_cols:
            arr = c[col][start:end]
            if self._col_dtypes[col] == np.float32:
                sample[col] = _pad_float(arr, pad)
            else:
                sample[col] = _pad_long(arr, pad)
        sample["padding_mask"] = torch.cat([
            torch.zeros(length, dtype=torch.bool),
            torch.ones(pad,    dtype=torch.bool),
        ])
        return sample, int(c["client_id"])


def _pad_float(arr: np.ndarray, pad: int) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    if pad:
        t = torch.cat([t, torch.zeros(pad, dtype=torch.float32)])
    return t


def _pad_long(arr: np.ndarray, pad: int) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(arr, dtype=np.int64))
    if pad:
        t = torch.cat([t, torch.zeros(pad, dtype=torch.int64)])
    return t


# ---------------------------------------------------------------------------
# Sampler — guarantees ≥2 windows per client in every batch (InfoNCE positives)
# ---------------------------------------------------------------------------

class PairedClientBatchSampler(Sampler[list[int]]):
    """Yields batches of ``clients_per_batch * windows_per_pair`` indices.

    Each batch contains ``windows_per_pair`` distinct windows per client
    so the InfoNCE head always sees positive pairs.
    """

    def __init__(
        self,
        dataset: TransactionDataset,
        clients_per_batch: int = 8,
        windows_per_pair: int = 2,
        seed: int = 0,
    ):
        if windows_per_pair > dataset.windows_per_client:
            raise ValueError("windows_per_pair must be ≤ dataset.windows_per_client")
        self.dataset = dataset
        self.clients_per_batch = clients_per_batch
        self.windows_per_pair = windows_per_pair
        self._rng = np.random.default_rng(seed)

        # Map client_idx → list of dataset indices that belong to it
        self._by_client: dict[int, list[int]] = {}
        for di, ci in enumerate(dataset._index):
            self._by_client.setdefault(ci, []).append(di)

        # Escludi clienti con troppo poche transazioni per produrre
        # `windows_per_pair` finestre con contenuto distinto.
        # Start ∈ [0, n - seq_len] → start distinti = max(1, n - seq_len + 1).
        min_transactions = dataset.seq_len + windows_per_pair - 1
        self._by_client = {
            ci: idxs for ci, idxs in self._by_client.items()
            if int(dataset.clients[ci]["n"]) >= min_transactions
        }
        if not self._by_client:
            raise ValueError(
                f"No clients have ≥{min_transactions} transactions "
                f"(seq_len={dataset.seq_len}, windows_per_pair={windows_per_pair})"
            )

    def __iter__(self):
        client_ids = list(self._by_client.keys())
        self._rng.shuffle(client_ids)

        for i in range(0, len(client_ids), self.clients_per_batch):
            chunk = client_ids[i:i + self.clients_per_batch]
            if len(chunk) < 2:  # need ≥2 clients for a meaningful contrastive batch
                continue
            batch: list[int] = []
            for ci in chunk:
                picks = self._rng.choice(
                    self._by_client[ci],
                    size=self.windows_per_pair,
                    replace=False, # finestre DISTINTE
                )
                batch.extend(int(p) for p in picks)
            yield batch

    def __len__(self) -> int:
        n_clients = len(self._by_client)
        return n_clients // self.clients_per_batch


# ---------------------------------------------------------------------------
# Collate — stacks (B,T) tensors and returns the client-id vector
# ---------------------------------------------------------------------------

def collate(items: list[tuple[dict[str, torch.Tensor], int]]):
    samples, client_ids = zip(*items)
    keys = samples[0].keys()
    batch = {k: torch.stack([s[k] for s in samples], dim=0) for k in keys}
    return batch, torch.tensor(client_ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# DataModule — callable wrapper around the load → fit → dataset → loader pipeline
# ---------------------------------------------------------------------------

class DataModule:
    """Callable bundle that produces a ready-to-iterate training loader.

    The CSV/parquet I/O and the per-feature normalizer fitting live
    *inside* this class (private methods).  The Dataset, Sampler and
    collate function are kept module-level on purpose: they are
    standalone PyTorch concepts reused by the DataLoader.

    Column names live in :class:`DataConfig` (non-embedded: client id,
    timestamp, delta_t) and in :data:`FEATURE_COLS` (embedded columns).
    """

    def __init__(
        self,
        args: "TrainingConfig",
        features: list[FeatureSpec] | None = None,
        data_config: DataConfig | None = None,
    ):
        from .model import DEFAULT_FEATURES  # local import: avoids cycle

        self.args = args
        self.base_features = features if features is not None else DEFAULT_FEATURES
        self.data_config = data_config if data_config is not None else DataConfig()
        self.features: list[FeatureSpec] | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None

    def __call__(self) -> tuple[DataLoader, DataLoader | None, list[FeatureSpec]]:
        df = self._load_dataframe()
        train_df, val_df = self._split_clients(df)
        self.features = self._fit_features(train_df)

        self.train_loader = self._build_loader(train_df, tag="train")
        self.val_loader = self._build_loader(val_df, tag="val") if val_df is not None else None
        return self.train_loader, self.val_loader, self.features

    def _split_clients(
        self, df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Hold out a deterministic fraction of clients for validation.

        Returns ``(train_df, val_df)``.  If ``args.val_frac == 0`` or the
        resulting val set is empty, ``val_df`` is ``None``.
        """
        cid_col = self.data_config.client_col
        all_clients = df[cid_col].unique()
        val_frac = float(self.args.val_frac)
        if val_frac <= 0:
            logger.info("val_frac=0 → no validation split")
            return df, None

        rng = np.random.default_rng(self.args.seed)
        shuffled = rng.permutation(all_clients)
        n_val = int(round(len(shuffled) * val_frac))
        if n_val <= 0:
            logger.warning(
                "val_frac={:.3f} rounds to 0 clients (have {}); skipping val split",
                val_frac, len(shuffled),
            )
            return df, None

        val_ids = set(shuffled[:n_val].tolist())
        train_mask = ~df[cid_col].isin(val_ids)
        train_df = df[train_mask].reset_index(drop=True)
        val_df = df[~train_mask].reset_index(drop=True)
        logger.info(
            "Client split: {} train / {} val (val_frac={:.2f}, seed={})",
            train_df[cid_col].nunique(), val_df[cid_col].nunique(),
            val_frac, self.args.seed,
        )
        return train_df, val_df

    def _build_loader(self, df: pd.DataFrame, tag: str) -> DataLoader:
        dataset = self._build_dataset(df, tag=tag)
        sampler = self._build_sampler(dataset)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)
        logger.info(
            "{} loader ready: {} batches/epoch, batch size {} "
            "({} clients × {} windows)",
            tag.capitalize(), len(sampler), self.args.batch_size,
            self.args.clients_per_batch, self.args.windows_per_pair,
        )
        return loader

    def _load_dataframe(self) -> pd.DataFrame:
        """Load transactions from a CSV/parquet file or a folder of either type.

        - File ``*.csv`` → :func:`pandas.read_csv`
        - File ``*.parquet`` → :func:`pandas.read_parquet`
        - Folder → all top-level ``*.csv`` are concatenated, or all ``*.parquet``
          are read together via :func:`pandas.read_parquet`.  Mixing both
          extensions in the same folder is rejected.
        """
        path = Path(self.args.csv_path)
        logger.info("Loading dataframe from {}", path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        if path.is_dir():
            parquets = sorted(path.glob("*.parquet"))
            csvs = sorted(path.glob("*.csv"))
            if parquets and csvs:
                raise ValueError(f"{path} contains both .parquet and .csv files")
            if parquets:
                df = pd.read_parquet(path)
            elif csvs:
                df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
            else:
                raise FileNotFoundError(f"No .csv or .parquet files in {path}")
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix} ({path})")

        logger.info(
            "Loaded {:,} rows / {} clients",
            len(df), df["client_id"].nunique(),
        )
        return df

    def _fit_features(self, df: pd.DataFrame) -> list[FeatureSpec]:
        """Fit per-feature normalizers from the training DataFrame.

        Only :class:`NumericFeature` specs are fitted (clip+log1p+z-score).
        Categorical features keep their explicit ``vocab_size``.
        """
        logger.info("Fitting per-feature normalizers/vocabularies")
        for feat in self.base_features:
            if isinstance(feat, NumericFeature) and feat.name in df.columns:
                feat.fit(df[feat.name].to_numpy(copy=True))
        logger.debug(
            "Feature schema: {}",
            [(f.__class__.__name__, f.name) for f in self.base_features],
        )
        return self.base_features

    def _build_dataset(self, df: pd.DataFrame, tag: str = "train") -> TransactionDataset:
        logger.info(
            "Building {} TransactionDataset (seq_len={}, windows_per_client={})",
            tag, self.args.seq_len, self.args.windows_per_client,
        )
        ds = TransactionDataset(
            df,
            client_col=self.data_config.client_col,
            timestamp_col=self.data_config.timestamp_col,
            feature_cols=FEATURE_COLS,
            seq_len=self.args.seq_len,
            windows_per_client=self.args.windows_per_client,
            seed=self.args.seed,
        )
        logger.info("{} dataset built: {:,} windows total", tag, len(ds))
        return ds

    def _build_sampler(self, dataset: TransactionDataset) -> PairedClientBatchSampler:
        return PairedClientBatchSampler(
            dataset,
            clients_per_batch=self.args.clients_per_batch,
            windows_per_pair=self.args.windows_per_pair,
            seed=self.args.seed,
        )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .config import TrainingConfig

    args = TrainingConfig()
    train_loader, val_loader, features = DataModule(args)()

    batch, client_ids = next(iter(train_loader))
    print(f"train batch size = {client_ids.size(0)}  unique clients = {client_ids.unique().numel()}")
    for k, v in batch.items():
        print(f"  {k:13s} {tuple(v.shape)}  {v.dtype}")
    if val_loader is not None:
        vb, vcid = next(iter(val_loader))
        print(f"val batch size = {vcid.size(0)}  unique clients = {vcid.unique().numel()}")
