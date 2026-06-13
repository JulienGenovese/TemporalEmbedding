"""Window-level client embeddings from a transactions dataset.

Loads a trained checkpoint, reads transactions like ``train.py``, sorts by
``client_id`` + ``timestamp``, splits each client timeline into distinct
windows, and writes one embedding per window with its time interval.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader

from ...constant import FEATURE_COLS
from .data import TransactionDataset, _pad_float, _pad_long
from .encoder import NumericFeature
from .hier_config import TrainingConfig
from .model import EmbeddingModel


class PredictionTransactionDataset(TransactionDataset):
    """Deterministic, per-client windows for inference.

    Keeps training-compatible tensor preparation, but:
    1. emits only distinct windows per client (no duplicated slots);
    2. returns temporal bounds (left=min_ts, right=max_ts) per window.
    """

    def _build_index(self) -> list[tuple[int, int]]:
        """Only distinct slots available per client (no duplicated windows)."""
        index: list[tuple[int, int]] = []
        for ci, client in enumerate(self.clients):
            n = int(client["n"])
            if n <= 0:
                continue
            if n <= self.seq_len:
                n_windows = 1
            else:
                n_windows = min(self.windows_per_client, n - self.seq_len + 1)
            for slot in range(n_windows):
                index.append((ci, slot))
        return index

    @staticmethod
    def _deterministic_window(
        n: int,
        seq_len: int,
        windows_per_client: int,
        slot: int,
    ) -> tuple[int, int]:
        """Return ``(start, length)`` for a deterministic distinct window."""
        if n <= seq_len:
            return 0, n

        # Same slot partitioning idea used in training, but deterministic.
        # Distinct slots map to distinct start buckets, so windows never coincide.
        r = n - seq_len + 1
        k = min(windows_per_client, r)
        lo = (slot * r) // k
        hi = max(lo + 1, ((slot + 1) * r) // k)
        start = (lo + hi - 1) // 2
        return int(start), seq_len

    def __getitem__(self, idx: int):
        ci, slot = self._index[idx]
        c = self.clients[ci]

        t = self.seq_len
        n = int(c["n"])
        start, length = self._deterministic_window(
            n=n,
            seq_len=t,
            windows_per_client=self.windows_per_client,
            slot=slot,
        )
        end = start + length
        pad = t - length

        ts = c["timestamp"][start:end].astype(np.int64)
        # delta_t in seconds; first transaction in window is always 0.
        delta_t = np.zeros(length, dtype=np.float32)
        if length > 1:
            delta_t[1:] = np.clip(np.diff(ts).astype(np.float32), 0, None)

        sample: dict[str, torch.Tensor] = {
            "delta_t": _pad_float(delta_t, pad),
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
            torch.ones(pad, dtype=torch.bool),
        ])

        left = int(ts.min()) if length > 0 else 0
        right = int(ts.max()) if length > 0 else 0
        return sample, int(c["client_id"]), int(slot), left, right


def collate_prediction(items):
    samples, client_codes, slots, lefts, rights = zip(*items)
    keys = samples[0].keys()
    batch = {k: torch.stack([s[k] for s in samples], dim=0) for k in keys}
    return (
        batch,
        torch.tensor(client_codes, dtype=torch.long),
        torch.tensor(slots, dtype=torch.long),
        torch.tensor(lefts, dtype=torch.long),
        torch.tensor(rights, dtype=torch.long),
    )


class Predictor:
    """Window-level embedding inference encapsulated as a callable object.

    Mirrors the :class:`~src.models.hier_transformer.train.Trainer` paradigm:
    owns the config, data module, model and loader; ``__call__`` runs the full
    prediction schedule (load → build model → embed → save) and returns the
    path of the written embeddings file.
    """

    def __init__(
        self,
        args: TrainingConfig | None = None,
        ckpt_path: Path | None = None,
        output_path: Path | None = None,
    ):
        from .data import DataModule  # local import: avoids import cycle

        self.args = args or TrainingConfig()
        self.ckpt_path = ckpt_path
        self.output_path = output_path
        self.device = self.args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device is : {}", self.device)

        self.data = DataModule(self.args)
        # Populated lazily during the run.
        self.model: EmbeddingModel | None = None
        self.dataset: PredictionTransactionDataset | None = None
        self.loader: DataLoader | None = None

    def __call__(self) -> Path:
        return self._predict()

    def _load_dataframe(self) -> pd.DataFrame:
        """Load and chronologically sort the transactions (like train time)."""
        df = self.data._load_dataframe()
        sort_cols = self.data.data_config.transaction_sort_cols
        df = df.sort_values(sort_cols).reset_index(drop=True)
        logger.info("Data sorted by {}", sort_cols)
        return df

    def _build_model(self, df: pd.DataFrame) -> EmbeddingModel:
        """Load the checkpoint and restore the train-time numeric normalizers.

        Feature specs start unfitted; :meth:`EmbeddingModel.load` restores the
        normalizer stats saved at train time so inference normalizes inputs
        exactly as training did.  Legacy checkpoints predate persisted stats —
        fall back to re-fitting on the prediction data (with a warning).
        """
        model_path = self.ckpt_path or (Path(self.args.ckpt_dir) / "model_final.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        features = self.data.base_features
        logger.info("Loading checkpoint from {}", model_path)
        model = EmbeddingModel.load(
            model_path,
            features=features,
            map_location=self.device,
            pretrain=True,
            use_gradient_checkpointing=(self.device != "cpu"),
            **asdict(self.args.model),
        ).to(self.device)
        model.eval()

        unfitted = [
            f.name for f in features
            if isinstance(f, NumericFeature)
            and (f.normalizer is None or not f.normalizer.fitted)
        ]
        if unfitted:
            logger.warning(
                "Checkpoint has no saved normalizer for {}; re-fitting on "
                "prediction data (legacy checkpoint — embeddings may differ "
                "slightly from train)",
                unfitted,
            )
            self.data._fit_features(df)

        return model

    def _build_loader(self, df: pd.DataFrame) -> DataLoader:
        self.dataset = PredictionTransactionDataset(
            df=df,
            client_col=self.data.data_config.client_col,
            timestamp_col=self.data.data_config.timestamp_col,
            feature_cols=FEATURE_COLS,
            seq_len=self.args.seq_len,
            windows_per_client=self.args.windows_per_client,
            seed=self.args.seed,
        )
        if len(self.dataset) == 0:
            raise ValueError("No windows generated from input dataset.")
        logger.info("Prediction dataset built: {:,} windows", len(self.dataset))
        return DataLoader(
            dataset=self.dataset,
            batch_size=max(1, self.args.batch_size),
            shuffle=False,
            collate_fn=collate_prediction,
        )

    def _embed(self) -> list[dict]:
        """Run the loader through the model → one row dict per window."""
        rows: list[dict] = []
        # model.embed → get_client_embedding already runs under torch.no_grad().
        for batch, client_codes, slots, lefts, rights in self.loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            emb = self.model.embed(batch).detach().cpu().numpy()

            for i in range(emb.shape[0]):
                code = int(client_codes[i].item())
                row = {
                    "client_id": self.dataset.client_id_lookup[code],
                    "client_code": code,
                    "window_slot": int(slots[i].item()),
                    "window_start_ts": int(lefts[i].item()),
                    "window_end_ts": int(rights[i].item()),
                }
                for j, value in enumerate(emb[i]):
                    row[f"emb_{j}"] = float(value)
                rows.append(row)
        return rows

    def _resolve_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        pred_file = Path(self.args.pred_file_name)
        if pred_file.name != self.args.pred_file_name:
            raise ValueError(
                "`model.hierTransformer.paths.pred_file_name` must be a file "
                "name, not a path."
            )
        if not pred_file.suffix:
            pred_file = pred_file.with_suffix(".csv")
        return Path(self.args.pred_path) / pred_file

    def _save(self, rows: list[dict]) -> Path:
        out = self._resolve_output_path()
        out.parent.mkdir(parents=True, exist_ok=True)

        result = pd.DataFrame(rows)
        result["window_start"] = pd.to_datetime(result["window_start_ts"], unit="s", utc=True)
        result["window_end"] = pd.to_datetime(result["window_end_ts"], unit="s", utc=True)
        if out.suffix.lower() == ".parquet":
            result.to_parquet(out, index=False)
        else:
            result.to_csv(out, index=False)

        logger.success(
            "Saved {:,} window embeddings for {:,} clients to {}",
            len(result),
            result["client_id"].nunique(),
            out.resolve(),
        )
        return out

    def _predict(self) -> Path:
        df = self._load_dataframe()
        self.model = self._build_model(df)
        self.loader = self._build_loader(df)
        rows = self._embed()
        return self._save(rows)


def main() -> Path:
    return Predictor()()


if __name__ == "__main__":
    main()
