"""Unified entrypoints for evaluation analyses."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal
from enum import Enum

from src.models.types import ModelType

PerturbationAnalysis = Literal["classification", "sensibility", "sensitivity"]
EvalPipeline = Callable[[], Path]

class PerturbationAnalysis(str, Enum):
    """Supported perturbation analysis modes."""

    CLASSIFICATION = "classification"
    SENSIBILITY = "sensibility"

def _classification_hier() -> Path:
    from src.eval.classification_perturbation import ClassificationPerturbationAnalyzer

    analyzer = ClassificationPerturbationAnalyzer()
    analyzer()
    return analyzer.output_path


def _sensibility_hier() -> Path:
    from src.eval.sensibility import SensibilityAnalyzer

    analyzer = SensibilityAnalyzer()
    analyzer()
    return analyzer.output_path


EVAL_PIPELINES: dict[str, dict[ModelType, EvalPipeline]] = {
    "classification": {
        ModelType.HIER: _classification_hier,
    },
    "sensibility": {
        ModelType.HIER: _sensibility_hier,
    },
    "sensitivity": {
        ModelType.HIER: _sensibility_hier,
    },
}


def main(
    analysis: PerturbationAnalysis | str = "sensibility",
    model_type: ModelType | str = ModelType.HIER,
) -> Path:
    normalized_analysis = analysis.lower()
    pipelines = EVAL_PIPELINES.get(normalized_analysis)
    if pipelines is None:
        supported_analyses = ", ".join(EVAL_PIPELINES)
        raise ValueError(
            f"Unsupported evaluation analysis: {analysis}. "
            f"Supported values: {supported_analyses}.",
        )

    try:
        normalized_model_type = ModelType(model_type)
    except ValueError as exc:
        supported_types = ", ".join(variant.value for variant in ModelType)
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported values: {supported_types}."
        ) from exc

    pipeline = pipelines.get(normalized_model_type)
    if pipeline is None:
        supported_types = ", ".join(variant.value for variant in pipelines)
        raise ValueError(
            f"Unsupported model type for {analysis}: {model_type}. "
            f"Supported values: {supported_types}.",
        )
    return pipeline()
