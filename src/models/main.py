"""Unified entrypoints for model train, prediction and evaluation pipelines."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from src.models.types import ModelType

ModelPipeline = Callable[[], Path]
ModelAction = Literal["train", "pred"]


def _train_hier() -> Path:
    from src.models.hier_transformer.train import main

    return main()


def _pred_hier() -> Path:
    from src.models.hier_transformer.pred import main

    return main()


PIPELINES: dict[str, dict[ModelType, ModelPipeline]] = {
    "train": {
        ModelType.HIER: _train_hier,
    },
    "pred": {
        ModelType.HIER: _pred_hier,
    },
}


def _run(
    action: ModelAction | str,
    model_type: ModelType | str,
) -> Path:
    normalized_action = action.lower()
    pipelines = PIPELINES.get(normalized_action)
    if pipelines is None:
        supported_actions = ", ".join(PIPELINES)
        raise ValueError(
            f"Unsupported action: {action}. Supported values: {supported_actions}."
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
            f"Unsupported model type for {action}: {model_type}. "
            f"Supported values: {supported_types}."
        )
    return pipeline()


def train(model_type: ModelType | str) -> Path:
    return _run("train", model_type)


def pred(model_type: ModelType | str) -> Path:
    return _run("pred", model_type)


def main(action: ModelAction | str, model_type: ModelType | str) -> Path:
    return _run(action, model_type)
