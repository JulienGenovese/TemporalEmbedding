"""Global, dataset-level perturbation analysis for client embeddings.

Measures how much each input variable influences the trained model by
*permutation*: the values of one variable are shuffled across windows (block
per-window — the whole sequence is swapped), everything else left intact, and
the embeddings are recomputed.  Two complementary metrics are reported:

- **delta_acc**  drop in cluster separability — accuracy of a logistic-regression
  probe (trained once on the clean embeddings and then frozen) at predicting the
  synthetic ``cluster`` label, ``acc_clean - acc_perturbed``.
- **drift**      mean cosine distance ``1 - cos(clean[i], perturbed[i])`` between
  each window's clean and perturbed embedding (1:1 correspondence).

Both are averaged over a few permutation seeds.  Reuses the prediction pipeline
(checkpoint + normalizers + deterministic windows) from :mod:`pred`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.linear_model import LogisticRegression

from ..constant import DATA_CONFIG
from ..models.hier_transformer.hier_config import TrainingConfig
from ..models.hier_transformer.pred import Predictor

# Variables that actually reach the model (schema features + the time gap).
PERTURBABLE = [
    DATA_CONFIG.amount_col,
    DATA_CONFIG.merchant_col,
    DATA_CONFIG.cocau_col,
    DATA_CONFIG.delta_t_col,
]
N_SEEDS = 5
TEST_FRAC = 0.3


class PerturbationAnalyzer:
    """Permutation-importance over the perturbable input variables."""

    def __init__(
        self,
        args: TrainingConfig | None = None,
        ckpt_path: Path | None = None,
    ):
        self.predictor = Predictor(args=args, ckpt_path=ckpt_path)
        self.args = self.predictor.args
        self.device = self.predictor.device

        # column + output path come from config.toml ([model.hierTransformer.perturbation]).
        self.column = self.args.perturb_column or None
        if self.column is not None and self.column not in PERTURBABLE:
            raise ValueError(
                f"Unknown column {self.column!r}; perturbable columns: {PERTURBABLE}"
            )
        self.output_path = Path(self.args.perturb_path) / self.args.perturb_file_name

    def __call__(self) -> pd.DataFrame:
        df = self.predictor._load_dataframe()
        model = self.predictor._build_model(df)
        loader = self.predictor._build_loader(df)
        dataset = self.predictor.dataset

        # 1. Materialize every window into one set of full-batch tensors.
        parts, codes = [], []
        for batch, client_codes, *_ in loader:
            parts.append(batch)
            codes.append(client_codes)
        keys = parts[0].keys()
        full = {k: torch.cat([p[k] for p in parts], dim=0) for k in keys}
        codes = torch.cat(codes).numpy()
        n = codes.shape[0]

        # 2. Per-window cluster label (constant per client).
        cluster_lookup = (
            df.groupby(DATA_CONFIG.client_col)[DATA_CONFIG.cluster_col].first().to_dict()
        )
        client_ids = np.array([dataset.client_id_lookup[c] for c in codes])
        clusters = np.array([cluster_lookup[cid] for cid in client_ids])

        # 3. Clean embeddings + frozen probe (split by client to avoid leakage).
        clean = self._embed_all(model, full)
        rng = np.random.default_rng(self.args.seed)
        unique = np.unique(client_ids)
        rng.shuffle(unique)
        test_clients = set(unique[: int(len(unique) * TEST_FRAC)])
        test = np.array([cid in test_clients for cid in client_ids])

        probe = LogisticRegression(max_iter=1000)
        probe.fit(clean[~test], clusters[~test])
        acc0 = probe.score(clean[test], clusters[test])
        logger.info("Baseline probe accuracy on {:,} test windows: {:.4f}", int(test.sum()), acc0)

        # 4. Permute each target column and measure the two metrics.
        targets = [self.column] if self.column else PERTURBABLE
        rows = []
        for col in targets:
            daccs, drifts = [], []
            for s in range(N_SEEDS):
                perm = torch.from_numpy(np.random.default_rng(self.args.seed + s).permutation(n))
                perturbed = dict(full)
                perturbed[col] = full[col][perm]
                emb = self._embed_all(model, perturbed)
                daccs.append(acc0 - probe.score(emb[test], clusters[test]))
                drifts.append(self._drift(clean, emb))
            rows.append({
                "variable": col,
                "delta_acc": float(np.mean(daccs)),
                "delta_acc_std": float(np.std(daccs)),
                "drift": float(np.mean(drifts)),
                "drift_std": float(np.std(drifts)),
            })

        result = pd.DataFrame(rows).sort_values("delta_acc", ascending=False).reset_index(drop=True)
        self._save(result)
        return result

    def _embed_all(self, model, full: dict[str, torch.Tensor]) -> np.ndarray:
        bs = max(1, self.args.batch_size)
        out = []
        for i in range(0, full["padding_mask"].shape[0], bs):
            chunk = {k: v[i : i + bs].to(self.device) for k, v in full.items()}
            out.append(model.embed(chunk).detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    @staticmethod
    def _drift(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return float(np.mean(1.0 - (a * b).sum(axis=1)))

    def _save(self, result: pd.DataFrame) -> Path:
        logger.info("Perturbation importance (higher delta_acc = more influential):\n{}", result.to_string(index=False))
        out = self.output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)
        logger.success("Saved perturbation report to {}", out.resolve())
        return out


def main() -> pd.DataFrame:
    return PerturbationAnalyzer()()


if __name__ == "__main__":
    main()
