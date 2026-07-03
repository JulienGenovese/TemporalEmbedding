"""Classification-based perturbation analysis for client embeddings.

The clean embeddings are used to train a logistic regression classifier for the
dataset cluster label. Each perturbable transaction column is then permuted,
embeddings are recomputed, and the same classifier is evaluated on the perturbed
embeddings to quantify downstream classification degradation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_schema import DATA_CONFIG
from src.models.hier_transformer.artifacts import dataset_name_from_path
from src.models.hier_transformer.hier_config import HierTransformerConfig
from src.models.hier_transformer.pred import Predictor
from src.eval.sensibility import PERTURBABLE

CLASSIFICATION_REPORT_FILENAME = "classification_perturbation.csv"


@dataclass(frozen=True)
class ClassificationBaseline:
    """Clean classifier state reused for each perturbation."""

    classifier: Any
    test_idx: np.ndarray
    labels: np.ndarray
    clean_pred: np.ndarray
    clean_accuracy: float
    clean_macro_f1: float


class ClassificationPerturbationAnalyzer:
    """Evaluate feature perturbations through a logistic cluster classifier."""

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
        self.output_path = self._resolve_output_path()

    def _resolve_output_path(self) -> Path:
        base_path = Path(self.args.perturbation.classification_output_path)
        dataset_name = dataset_name_from_path(self.args.paths.pred_input_path)
        if base_path.suffix:
            if base_path.parent.name == dataset_name:
                return base_path
            return base_path.parent / dataset_name / base_path.name
        return base_path / dataset_name / CLASSIFICATION_REPORT_FILENAME

    def __call__(self) -> pd.DataFrame:
        df, model, loader, dataset = self.predictor.prepare()
        full, client_codes = self._materialize_loader(loader)
        labels = self._labels_for_windows(df, dataset, client_codes)
        clean = self._embed_all(model, full)
        baseline = self._build_baseline(clean, labels, client_codes)

        targets = [self.column] if self.column else PERTURBABLE
        result = pd.DataFrame([
            self._score_column(column, model, full, baseline)
            for column in targets
        ]).sort_values("accuracy_drop", ascending=False).reset_index(drop=True)
        result.insert(1, "importance_rank", np.arange(1, len(result) + 1))
        self._save(result)
        return result

    @staticmethod
    def _materialize_loader(loader) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        parts = []
        client_codes = []
        for batch, batch_client_codes, *_ in loader:
            parts.append(batch)
            client_codes.append(batch_client_codes)
        keys = parts[0].keys()
        full = {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}
        return full, torch.cat(client_codes, dim=0).cpu().numpy()

    @staticmethod
    def _labels_for_windows(df: pd.DataFrame, dataset, client_codes: np.ndarray) -> np.ndarray:
        client_col = DATA_CONFIG.client_col
        cluster_col = DATA_CONFIG.cluster_col
        if cluster_col not in df.columns:
            raise ValueError(f"Column `{cluster_col}` not found in prediction dataset.")

        clusters_per_client = df.groupby(client_col)[cluster_col].nunique()
        inconsistent = clusters_per_client[clusters_per_client > 1]
        if not inconsistent.empty:
            examples = ", ".join(str(client) for client in inconsistent.index[:5])
            raise ValueError(
                "Classification perturbation expects one cluster per client; "
                f"found multiple clusters for clients: {examples}",
            )

        client_ids = [dataset.client_id_lookup[int(code)] for code in client_codes]
        label_by_client = df.groupby(client_col)[cluster_col].first()
        labels = pd.Series(client_ids).map(label_by_client).to_numpy()
        labels_array = np.asarray(labels)
        if np.unique(labels_array).size < 2:
            raise ValueError("At least two cluster labels are required for classification.")
        return labels_array

    def _split_indices(
        self,
        labels: np.ndarray,
        client_codes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = np.arange(labels.shape[0])
        client_labels = pd.DataFrame({
            "client_code": client_codes,
            "label": labels,
        }).drop_duplicates("client_code")
        counts = client_labels["label"].value_counts()
        n_classes = counts.shape[0]
        min_test_clients = n_classes
        max_test_clients = len(client_labels) - n_classes
        can_stratify = counts.min() >= 2 and max_test_clients >= min_test_clients
        if can_stratify:
            requested = int(
                round(
                    len(client_labels)
                    * self.args.perturbation.classification_test_size,
                ),
            )
            test_clients = min(
                max(requested, min_test_clients),
                max_test_clients,
            )
            _, test_client_codes = train_test_split(
                client_labels["client_code"].to_numpy(),
                test_size=test_clients,
                random_state=self.args.runtime.seed,
                stratify=client_labels["label"].to_numpy(),
            )
            test_mask = np.isin(client_codes, test_client_codes)
            return indices[~test_mask], indices[test_mask]

        logger.warning(
            "Not enough samples per cluster for a stratified split; evaluating on "
            "the same clean embeddings used to fit the classifier.",
        )
        return indices, indices

    def _build_baseline(
        self,
        clean: np.ndarray,
        labels: np.ndarray,
        client_codes: np.ndarray,
    ) -> ClassificationBaseline:
        train_idx, test_idx = self._split_indices(labels, client_codes)
        classifier = self._fit_classifier(clean[train_idx], labels[train_idx])
        clean_pred = classifier.predict(clean[test_idx])
        clean_accuracy, clean_macro_f1 = self._classification_scores(
            labels[test_idx],
            clean_pred,
        )
        return ClassificationBaseline(
            classifier=classifier,
            test_idx=test_idx,
            labels=labels,
            clean_pred=clean_pred,
            clean_accuracy=clean_accuracy,
            clean_macro_f1=clean_macro_f1,
        )

    def _fit_classifier(self, embeddings: np.ndarray, labels: np.ndarray):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=self.args.perturbation.classification_max_iter,
                random_state=self.args.runtime.seed,
            ),
        )
        classifier.fit(embeddings, labels)
        return classifier

    def _score_column(
        self,
        column: str,
        model,
        full: dict[str, torch.Tensor],
        baseline: ClassificationBaseline,
    ) -> dict:
        n = full["padding_mask"].shape[0]
        accuracies = []
        macro_f1_scores = []
        flip_rates = []

        for seed_offset in range(self.args.perturbation.classification_n_seeds):
            rng = np.random.default_rng(self.args.runtime.seed + seed_offset)
            perm = torch.from_numpy(rng.permutation(n))
            perturbed = dict(full)
            perturbed[column] = full[column][perm]
            emb = self._embed_all(model, perturbed)
            pred = baseline.classifier.predict(emb[baseline.test_idx])

            accuracy, macro_f1 = self._classification_scores(
                baseline.labels[baseline.test_idx],
                pred,
            )
            accuracies.append(accuracy)
            macro_f1_scores.append(macro_f1)
            flip_rates.append(float(np.mean(pred != baseline.clean_pred)))

        perturbed_accuracy = float(np.mean(accuracies))
        perturbed_macro_f1 = float(np.mean(macro_f1_scores))
        return {
            "variable": column,
            "clean_accuracy": baseline.clean_accuracy,
            "perturbed_accuracy": perturbed_accuracy,
            "accuracy_drop": float(baseline.clean_accuracy - perturbed_accuracy),
            "accuracy_drop_std": float(
                np.std(baseline.clean_accuracy - np.asarray(accuracies)),
            ),
            "clean_macro_f1": baseline.clean_macro_f1,
            "perturbed_macro_f1": perturbed_macro_f1,
            "macro_f1_drop": float(baseline.clean_macro_f1 - perturbed_macro_f1),
            "prediction_flip_rate": float(np.mean(flip_rates)),
            "prediction_flip_rate_std": float(np.std(flip_rates)),
        }

    @staticmethod
    def _classification_scores(labels: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
        return (
            float(accuracy_score(labels, pred)),
            float(f1_score(labels, pred, average="macro", zero_division=0)),
        )

    def _embed_all(self, model, full: dict[str, torch.Tensor]) -> np.ndarray:
        batch_size = max(1, self.args.data.batch_size)
        embeddings = []
        with torch.no_grad():
            for i in range(0, full["padding_mask"].shape[0], batch_size):
                chunk = {
                    key: value[i : i + batch_size].to(self.device)
                    for key, value in full.items()
                }
                embeddings.append(model.embed(chunk).detach().cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def _save(self, result: pd.DataFrame) -> Path:
        logger.info("Classification perturbation:\n{}", result.to_string(index=False))
        out = self.output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)
        logger.success("Saved classification perturbation report to {}", out.resolve())
        return out


def main() -> pd.DataFrame:
    return ClassificationPerturbationAnalyzer()()
