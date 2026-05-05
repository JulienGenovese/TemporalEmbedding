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
    """Per-client windowed view over the synthetic transactions CSV.

    Each ``__getitem__`` returns a single fixed-length window (``seq_len``
    transactions) drawn from a single client.  Sequences shorter than
    ``seq_len`` are right-padded; ``padding_mask`` marks the padded slots.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 32,
        windows_per_client: int = 4,
        seed: int = 0,
    ):
        self.seq_len = seq_len
        self.windows_per_client = windows_per_client
        self._rng = np.random.default_rng(seed)

        # Pre-compute per-client field arrays once
        self.clients: list[dict[str, np.ndarray | int]] = []
        for cid, g in df.sort_values(["client_id", "timestamp"]).groupby("client_id"):
            self.clients.append({
                "client_id":  int(cid),
                "n":          len(g),
                "timestamp":  g["timestamp"].to_numpy(np.int64),
                "importo":    g["importo"].to_numpy(np.float32),
                "saldo_post": g["saldo_post"].to_numpy(np.float32),
                "merchant":   np.array(
                    [HighCardCategoricalFeature._to_int(m) for m in g["merchant"]],
                    dtype=np.int64,
                ),
                "mcc":        g["mcc"].to_numpy(np.int64),
                "canale":     g["canale"].to_numpy(np.int64),
                "macro_tipo": g["macro_tipo"].to_numpy(np.int64),
                "sotto_tipo": g["sotto_tipo"].to_numpy(np.int64),
                "divisa":     g["divisa"].to_numpy(np.int64),
            })

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
            "importo":      _pad_float(c["importo"][start:end],    pad),
            "saldo_post":   _pad_float(c["saldo_post"][start:end], pad),
            "delta_t":      _pad_float(delta_t,                    pad),
            "merchant":     _pad_long (c["merchant"][start:end],   pad),
            "mcc":          _pad_long (c["mcc"][start:end],        pad),
            "canale":       _pad_long (c["canale"][start:end],     pad),
            "macro_tipo":   _pad_long (c["macro_tipo"][start:end], pad),
            "sotto_tipo":   _pad_long (c["sotto_tipo"][start:end], pad),
            "divisa":       _pad_long (c["divisa"][start:end],     pad),
            "timestamp":    _pad_long (ts,                         pad),
            "padding_mask": torch.cat([
                torch.zeros(length, dtype=torch.bool),
                torch.ones(pad,    dtype=torch.bool),
            ]),
        }
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
                    self._by_client[ci], size=self.windows_per_pair, replace=False,
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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `uv run python -m src.make_dataset` first."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from .model import DEFAULT_FEATURES

    df = load_dataframe()
    features = fit_features(df, DEFAULT_FEATURES)
    ds = TransactionDataset(df, seq_len=32, windows_per_client=4)
    sampler = PairedClientBatchSampler(ds, clients_per_batch=8, windows_per_pair=2)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate)

    batch, client_ids = next(iter(loader))
    print(f"batch size = {client_ids.size(0)}  unique clients = {client_ids.unique().numel()}")
    for k, v in batch.items():
        print(f"  {k:13s} {tuple(v.shape)}  {v.dtype}")
