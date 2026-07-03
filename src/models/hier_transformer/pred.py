"""Window-level client embeddings from a transactions dataset.

Loads a trained checkpoint, reads transactions like ``train.py``, sorts by
``client_id`` + ``timestamp``, splits each client timeline into distinct
windows, and writes one embedding per window with its time interval.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader

from .artifacts import (
    MODEL_CHECKPOINT_FILENAME,
    resolve_artifact_dir,
)
from .data import PredictionTransactionDataset, collate_prediction
from .encoder import TransactionEncoder
from .features import NumericFeature
from .hier_config import HierTransformerConfig
from .model import EmbeddingModel


class Predictor:
    """Window-level embedding inference encapsulated as a callable object.

    Mirrors the :class:`~src.models.hier_transformer.train.Trainer` paradigm:
    owns the config, data module, model and loader; ``__call__`` runs the full
    prediction schedule (load → build model → embed → save) and returns the
    path of the written embeddings file.
    """

    def __init__(
        self,
        args: HierTransformerConfig | None = None,
        ckpt_path: Path | None = None,
        output_path: Path | None = None,
    ):
        from .data import DataModule  # local import: avoids import cycle

        self.args = args or HierTransformerConfig()
        self.path_config = self.args.paths
        self.pipeline_config = self.args.data
        self.runtime_config = self.args.runtime
        self.model_config = self.args.model
        self.ckpt_path = ckpt_path
        self.output_path = output_path
        self.device = self.runtime_config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Device is : {}", self.device)

        self.data_module = DataModule(self.args)
        # Populated lazily during the run.
        self.model: EmbeddingModel | None = None
        self.dataset: PredictionTransactionDataset | None = None
        self.loader: DataLoader | None = None

    def __call__(self) -> Path:
        return self._predict()

    def _load_dataframe(self) -> pd.DataFrame:
        """Load and chronologically sort the transactions (like train time)."""
        df = self.data_module._load_dataframe(self.path_config.pred_input_path)
        sort_cols = self.data_module.data_config.transaction_sort_cols
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
        model_path = self._resolve_checkpoint_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        features = self.data_module.base_features
        logger.info("Loading checkpoint from {}", model_path)
        model = EmbeddingModel.load(
            model_path,
            features=features,
            map_location=self.device,
            pretrain=True,
            use_gradient_checkpointing=(self.device != "cpu"),
            **asdict(self.model_config),
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
            TransactionEncoder.fit_features(features, df)

        return model

    def _resolve_checkpoint_path(self) -> Path:
        if self.ckpt_path is not None:
            return self.ckpt_path

        artifacts_root = Path(self.path_config.model_output_dir)
        checkpoint_path = (
            resolve_artifact_dir(artifacts_root) / MODEL_CHECKPOINT_FILENAME
        )
        if checkpoint_path.exists():
            return checkpoint_path

        return artifacts_root / MODEL_CHECKPOINT_FILENAME

    def _build_loader(self, df: pd.DataFrame) -> DataLoader:
        self.dataset = PredictionTransactionDataset(
            df=df,
            client_col=self.data_module.data_config.client_col,
            timestamp_col=self.data_module.data_config.timestamp_col,
            features=self.data_module.base_features,
            seq_len=self.pipeline_config.seq_len,
            pred_windows_per_client=self.pipeline_config.pred_windows_per_client,
        )
        if len(self.dataset) == 0:
            raise ValueError("No windows generated from input dataset.")
        logger.info("Prediction dataset built: {:,} windows", len(self.dataset))
        return DataLoader(
            dataset=self.dataset,
            batch_size=max(1, self.pipeline_config.batch_size),
            shuffle=False,
            collate_fn=collate_prediction,
        )

    def prepare(self) -> tuple[pd.DataFrame, EmbeddingModel, DataLoader, PredictionTransactionDataset]:
        """Load prediction dataframe, model, loader and dataset."""
        df = self._load_dataframe()
        self.model = self._build_model(df)
        self.loader = self._build_loader(df)
        return df, self.model, self.loader, self.dataset

    def _embed(self) -> list[dict]:
        """Run the loader through the model → one row dict per window."""
        rows: list[dict] = []
        # model.embed → get_client_embedding already runs under torch.no_grad().
        for batch, client_codes, window_indices, lefts, rights in self.loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            emb = self.model.embed(batch).detach().cpu().numpy()

            for i in range(emb.shape[0]):
                code = int(client_codes[i].item())
                row = {
                    "client_id": self.dataset.client_id_lookup[code],
                    "client_code": code,
                    "window_index": int(window_indices[i].item()),
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
        return Path(self.path_config.pred_output_path)

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
        self.prepare()
        rows = self._embed()
        return self._save(rows)


def main() -> Path:
    return Predictor()()