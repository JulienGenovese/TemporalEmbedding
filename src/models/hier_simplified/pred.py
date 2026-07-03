"""Window-level embeddings from a trained simplified checkpoint."""

from __future__ import annotations

from pathlib import Path

from src.models.hier_simplified.config import load_config
from src.models.hier_simplified.features import SIMPLIFIED_FEATURES
from src.models.hier_simplified.model import SimplifiedEmbeddingModel
from src.models.hier_transformer.pred import Predictor


class SimplifiedPredictor(Predictor):
    model_cls = SimplifiedEmbeddingModel


def main() -> Path:
    return SimplifiedPredictor(args=load_config(), features=SIMPLIFIED_FEATURES)()
