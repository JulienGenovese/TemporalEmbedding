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
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Sampler

from .encoder import (
    FeatureSpec, HighCardCategoricalFeature, NumericFeature,
)

from ...constant import FEATURE_COLS, DataConfig
from .hier_config import TrainingConfig


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
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Logical kind per feature column ('float' | 'int' | 'string'), resolved
        # once from the source DataFrame. At fetch time the stored array's own
        # dtype is enough to pick the pad function, so we only need this here.
        kinds = {col: _column_kind(df[col]) for col in self.feature_cols}

        # Pre-compute per-client field arrays once.
        # `client_id` is a factorized integer code (a per-client counter), so
        # non-integer ids (e.g. strings) work too; only ID *equality* matters
        # downstream (InfoNCE positive mask in src/loss.py).
        self.clients: list[dict[str, np.ndarray | int]] = []
        self.client_id_lookup: dict[int, object] = {}
        for code, (cid, g) in enumerate(
            df.sort_values([client_col, timestamp_col]).groupby(client_col)
        ):
            self.client_id_lookup[code] = cid
            entry: dict[str, np.ndarray | int] = {
                "client_id": code,
                "n":         len(g),
                "timestamp": g[timestamp_col].to_numpy(np.int64, copy=True),
            }
            for col in self.feature_cols:
                if kinds[col] == "float":
                    entry[col] = g[col].to_numpy(np.float32, copy=True)
                    continue
                if kinds[col] == "string":
                    arr = np.fromiter(
                        (HighCardCategoricalFeature._to_int(v) for v in g[col]),
                        dtype=np.int64, count=len(g),
                    )
                else:  # "int"
                    arr = g[col].to_numpy(np.int64)
                # Store integer columns at the smallest lossless width to cut RAM
                # (small-vocab categoricals → int8/int16; the 64-bit merchant hash
                # stays int64). _pad_long upcasts back to int64 at fetch, which is
                # what nn.Embedding consumes. astype(copy=True) also frees the
                # source DataFrame once the loop ends.
                entry[col] = arr.astype(_compact_int_dtype(arr))
            self.clients.append(entry)

        # Flat index: (client_idx, slot). The slot selects a disjoint bucket of
        # the start range in __getitem__, so distinct slots → distinct,
        # non-overlapping windows (see PairedClientBatchSampler).
        # Subclasses override _build_index() to change the windowing policy.
        self._index: list[tuple[int, int]] = self._build_index()

    def _build_index(self) -> list[tuple[int, int]]:
        """Flat ``(client_idx, slot)`` index — one entry per window to emit."""
        return [
            (ci, slot)
            for ci in range(len(self.clients))
            for slot in range(self.windows_per_client)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], int]:
        # return the client + window slot associated to the index
        ci, slot = self._index[idx]
        # return the features associated to the cliente
        c = self.clients[ci]

        T = self.seq_len
        n = int(c["n"])

        if n <= T:
            start, length = 0, n
        else:
            # Partition the start range [0, R) into disjoint buckets, one per
            # slot, and sample within this slot's bucket. Distinct slots land in
            # distinct buckets → distinct, non-overlapping windows (the sampler
            # guarantees R ≥ windows_per_client for eligible clients).
            R = n - T + 1
            k = min(self.windows_per_client, R)
            b = slot % k
            lo = (b * R) // k # lower bound
            hi = max(lo + 1, ((b + 1) * R) // k) # higher bound
            start = int(self._rng.integers(lo, hi))
            length = T
        end = start + length
        pad = T - length

        ts = c["timestamp"][start:end].astype(np.int64)
        # delta_t in seconds; first transaction of the window → 0
        delta_t = np.zeros(length, dtype=np.float32)
        if length > 1:
            delta_t[1:] = np.clip(np.diff(ts).astype(np.float32), 0, None)

        sample: dict[str, torch.Tensor] = {
            "delta_t":   _pad(delta_t, pad, np.float32),
            "timestamp": _pad(ts, pad, np.int64),
        }
        for col in self.feature_cols:
            arr = c[col][start:end]
            dtype = np.float32 if arr.dtype.kind == "f" else np.int64
            sample[col] = _pad(arr, pad, dtype)
        sample["padding_mask"] = torch.cat([
            torch.zeros(length, dtype=torch.bool),
            torch.ones(pad,    dtype=torch.bool),
        ])
        return sample, int(c["client_id"])


def _column_kind(s: pd.Series) -> str:
    """'float' | 'int' | 'string' — the only per-column info the Dataset needs.

    float → float32; int → int64; object/string → int64 (FNV-1a hashed).
    """
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return "string"
    raise TypeError(f"Unsupported dtype for column '{s.name}': {s.dtype}")


def _pad(arr: np.ndarray, pad: int, dtype) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(arr, dtype=dtype))
    if pad:
        t = torch.cat([t, torch.zeros(pad, dtype=t.dtype)])
    return t


def _compact_int_dtype(arr: np.ndarray) -> np.dtype:
    """Smallest signed int dtype that holds ``arr`` losslessly (≥ int8).

    IDs/indices are non-negative, but we stay signed so the int64 upcast in
    :func:`_pad_long` is unambiguous. Empty arrays default to ``int8``.
    """
    if arr.size == 0:
        return np.dtype(np.int8)
    lo, hi = int(arr.min()), int(arr.max())
    for dt in (np.int8, np.int16, np.int32):
        info = np.iinfo(dt)
        if lo >= info.min and hi <= info.max:
            return np.dtype(dt)
    return np.dtype(np.int64)


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
        for di, (ci, _slot) in enumerate(dataset._index):
            self._by_client.setdefault(ci, []).append(di)

        # Escludi clienti con troppo poche transazioni per produrre
        # `windows_per_client` finestre con contenuto distinto.
        # Start ∈ [0, n - seq_len] → start distinti R = n - seq_len + 1.
        # Richiediamo R ≥ windows_per_client così che ogni slot abbia il proprio
        # bucket disgiunto e qualunque coppia di slot distinti dia finestre
        # distinte (vedi TransactionDataset.__getitem__).
        min_transactions = dataset.seq_len + dataset.windows_per_client - 1
        self._by_client = {
            ci: idxs for ci, idxs in self._by_client.items()
            if int(dataset.clients[ci]["n"]) >= min_transactions
        }
        if not self._by_client:
            raise ValueError(
                f"No clients have ≥{min_transactions} transactions "
                f"(seq_len={dataset.seq_len}, "
                f"windows_per_client={dataset.windows_per_client})"
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


def _worker_init(worker_id: int) -> None:
    """Re-seed each DataLoader worker's RNG so num_workers>0 stays safe.

    Without this every forked worker would inherit an identical ``_rng`` and
    emit correlated samples. No-op when ``num_workers=0``.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._rng = np.random.default_rng(info.dataset._seed + worker_id)


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
        loader = DataLoader(
            dataset, batch_sampler=sampler, collate_fn=collate,
            worker_init_fn=_worker_init,
        )
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
        path = Path(self.args.train_path)
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
            len(df), df[self.data_config.client_col].nunique(),
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
    from ...constant import TrainingConfig

    args = TrainingConfig()
    train_loader, val_loader, features = DataModule(args)()

    batch, client_ids = next(iter(train_loader))
    print(f"train batch size = {client_ids.size(0)}  unique clients = {client_ids.unique().numel()}")
    for k, v in batch.items():
        print(f"  {k:13s} {tuple(v.shape)}  {v.dtype}")
    if val_loader is not None:
        vb, vcid = next(iter(val_loader))
        print(f"val batch size = {vcid.size(0)}  unique clients = {vcid.unique().numel()}")
