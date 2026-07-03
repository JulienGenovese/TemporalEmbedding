"""Permutation sensibility analysis for client embeddings.

For each perturbable input column, values are shuffled across windows and the
model embeddings are recomputed. The reported score is the mean cosine distance
between clean and perturbed embeddings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.data_schema import DATA_CONFIG
from src.models.hier_transformer.hier_config import HierTransformerConfig
from src.models.hier_transformer.pred import Predictor

PERTURBABLE = [
    DATA_CONFIG.amount_col,
    DATA_CONFIG.merchant_col,
    DATA_CONFIG.cocau_col,
    DATA_CONFIG.delta_t_col,
]
N_SEEDS = 5


class SensibilityAnalyzer:
    """Permutation drift over perturbable input variables."""

    def __init__(
        self,
        args: HierTransformerConfig | None = None,
        ckpt_path: Path | None = None,
    ):
        self.predictor = Predictor(args=args, ckpt_path=ckpt_path)
        self.args = self.predictor.args
        self.device = self.predictor.device

        self.column = self.args.perturbation.column or None
        if self.column is not None and self.column not in PERTURBABLE:
            raise ValueError(
                f"Unknown column {self.column!r}; perturbable columns: {PERTURBABLE}",
            )
        self.output_path = Path(self.args.perturbation.output_path)

    def __call__(self) -> pd.DataFrame:
        _, model, loader, _ = self.predictor.prepare()
        full = self._materialize_loader(loader)
        clean = self._embed_all(model, full)

        targets = [self.column] if self.column else PERTURBABLE
        result = pd.DataFrame([
            self._score_column(column, model, full, clean)
            for column in targets
        ]).sort_values("drift", ascending=False).reset_index(drop=True)
        self._save(result)
        return result

    @staticmethod
    def _materialize_loader(loader) -> dict[str, torch.Tensor]:
        parts = []
        for batch, *_ in loader:
            parts.append(batch)
        keys = parts[0].keys()
        return {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}

    def _score_column(
        self,
        column: str,
        model,
        full: dict[str, torch.Tensor],
        clean: np.ndarray,
    ) -> dict:
        n = full["padding_mask"].shape[0]
        drifts = []
        for seed_offset in range(N_SEEDS):
            rng = np.random.default_rng(self.args.runtime.seed + seed_offset)
            perm = torch.from_numpy(rng.permutation(n))
            perturbed = dict(full)
            perturbed[column] = full[column][perm]
            emb = self._embed_all(model, perturbed)
            drifts.append(self._drift(clean, emb))
        return {
            "variable": column,
            "drift": float(np.mean(drifts)),
            "drift_std": float(np.std(drifts)),
        }

    def _embed_all(self, model, full: dict[str, torch.Tensor]) -> np.ndarray:
        batch_size = max(1, self.args.data.batch_size)
        embeddings = []
        for i in range(0, full["padding_mask"].shape[0], batch_size):
            chunk = {k: v[i : i + batch_size].to(self.device) for k, v in full.items()}
            embeddings.append(model.embed(chunk).detach().cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    @staticmethod
    def _drift(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return float(np.mean(1.0 - (a * b).sum(axis=1)))

    def _save(self, result: pd.DataFrame) -> Path:
        logger.info("Perturbation drift:\n{}", result.to_string(index=False))
        out = self.output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)
        logger.success("Saved perturbation report to {}", out.resolve())
        return out


def main() -> pd.DataFrame:
    return SensibilityAnalyzer()()
