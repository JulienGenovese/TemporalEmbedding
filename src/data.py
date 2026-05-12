"""Dataset & DataLoader: CSV of transactions → windowed batches per client.

Reads the file produced by :mod:`src.make_dataset`, groups rows by
``client_id``, and yields fixed-length sequence windows ready to feed into
:class:`src.model.TransactionTransformer`.

The two windows-per-client batch layout below guarantees in-batch positive
pairs for the InfoNCE contrastive head.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .encoder import (
    CategoricalFeature, DatetimeFeature, FeatureSpec,
    HighCardCategoricalFeature, NumericFeature,
)


# ---------------------------------------------------------------------------
# Feature fitting on the loaded DataFrame
# ---------------------------------------------------------------------------

def fit_features(df: pd.DataFrame, features: list[FeatureSpec]) -> list[FeatureSpec]:
    """Fit per-feature normalizers/vocabularies from the training DataFrame.

    Numeric features get a clip+log1p+z-score normalizer.  Categorical
    features keep their explicit ``vocab_size`` (synthetic data is generated
    inside that range).  ``delta_t`` is fitted from the inter-transaction
    gaps computed per client.
    """
    delta_t_values = (
        df.sort_values(["client_id", "timestamp"])
          .groupby("client_id")["timestamp"]
          .diff()
          .dropna()
          .clip(lower=0)
          .to_numpy()
    )

    for feat in features:
        if isinstance(feat, NumericFeature):
            if feat.name == "delta_t":
                feat.fit(delta_t_values)
            elif feat.name in df.columns:
                feat.fit(df[feat.name].to_numpy())
    return features


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
                "timestamp": g[timestamp_col].to_numpy(np.int64),
            }
            for col in self.feature_cols:
                if self._col_is_string[col]:
                    entry[col] = np.array(
                        [HighCardCategoricalFeature._to_int(v) for v in g[col]],
                        dtype=np.int64,
                    )
                else:
                    entry[col] = g[col].to_numpy(self._col_dtypes[col])
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
# Convenience
# ---------------------------------------------------------------------------

def load_dataframe(path: Path | str = Path("data") / "transactions.csv") -> pd.DataFrame:
    """Load transactions from a CSV/parquet file or a folder of either type.

    - File ``*.csv`` → :func:`pandas.read_csv`
    - File ``*.parquet`` → :func:`pandas.read_parquet`
    - Folder → all top-level ``*.csv`` are concatenated, or all ``*.parquet``
      are read together via :func:`pandas.read_parquet` (which natively handles
      a directory of parquet parts).  Mixing both extensions in the same folder
      is rejected.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    if path.is_dir():
        parquets = sorted(path.glob("*.parquet"))
        csvs = sorted(path.glob("*.csv"))
        if parquets and csvs:
            raise ValueError(f"{path} contains both .parquet and .csv files")
        if parquets:
            return pd.read_parquet(path)
        if csvs:
            return pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
        raise FileNotFoundError(f"No .csv or .parquet files in {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {path.suffix} ({path})")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    DEFAULT_FEATURES: list[FeatureSpec] = [
        NumericFeature("importo", signed=True),
        NumericFeature("saldo_post"),
        NumericFeature("delta_t"),
        HighCardCategoricalFeature("merchant"),
        CategoricalFeature("mcc",        801),
        CategoricalFeature("canale",      11),
        CategoricalFeature("macro_tipo",   9),
        CategoricalFeature("sotto_tipo",  41),
        CategoricalFeature("divisa",       6),
        DatetimeFeature("timestamp"),
    ]

    df = load_dataframe()
    features = fit_features(df, DEFAULT_FEATURES)
    ds = TransactionDataset(
        df,
        client_col="client_id",
        timestamp_col="timestamp",
        feature_cols=[
            "importo", "saldo_post", "merchant", "mcc",
            "canale", "macro_tipo", "sotto_tipo", "divisa",
        ],
        seq_len=32,
        windows_per_client=4,
    )
    sampler = PairedClientBatchSampler(ds, clients_per_batch=8, windows_per_pair=2)
    loader = DataLoader(ds, 
                        batch_sampler=sampler, 
                        collate_fn=collate)

    batch, client_ids = next(iter(loader))
    print(f"batch size = {client_ids.size(0)}  unique clients = {client_ids.unique().numel()}")
    for k, v in batch.items():
        print(f"  {k:13s} {tuple(v.shape)}  {v.dtype}")
