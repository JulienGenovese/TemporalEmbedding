"""Unified entrypoints for synthetic dataset generation."""

from __future__ import annotations

from pathlib import Path

from .experiments.abc import SyntheticExperiment
from .experiments.simple_calendar import SimpleCalendarExperiment
from .experiments.simple_delta import SimpleDeltaExperiment
from .experiments.simple_spatial import SimpleSpatialExperiment
from .types import DatasetType

EXPERIMENTS: dict[DatasetType, type[SyntheticExperiment]] = {
    DatasetType.SIMPLE_SPATIAL: SimpleSpatialExperiment,
    DatasetType.SIMPLE_DELTA: SimpleDeltaExperiment,
    DatasetType.SIMPLE_CALENDAR: SimpleCalendarExperiment,
}


def generate(
    dataset_type: DatasetType | str,
) -> dict[str, Path]:
    """Generate a synthetic dataset (train + pred splits) for the requested variant.

    Returns a mapping of split name (`train`/`pred`) to the generated file path.
    """
    try:
        normalized_dataset_type = DatasetType(dataset_type)
    except ValueError as exc:
        supported_types = ", ".join(variant.value for variant in DatasetType)
        raise ValueError(
            f"Unsupported dataset type: {dataset_type}. Supported values: {supported_types}."
        ) from exc

    experiment_class = EXPERIMENTS.get(normalized_dataset_type)
    if experiment_class is None:
        supported_types = ", ".join(variant.value for variant in EXPERIMENTS)
        raise ValueError(
            f"Unsupported dataset type: {dataset_type}. Supported values: {supported_types}."
        )
    return experiment_class().generate()
