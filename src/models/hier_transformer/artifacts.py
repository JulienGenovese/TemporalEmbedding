"""Helpers for hierarchical model artifact directories."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path


MODEL_CHECKPOINT_FILENAME = "model_final.pt"
HISTORY_FILENAME = "history.json"
TRAIN_EVAL_HISTORY_FILENAME = "train_eval_history.json"
VAL_HISTORY_FILENAME = "val_history.json"
RUN_METADATA_FILENAME = "run_metadata.json"
LATEST_DIRNAME = "latest"


def dataset_name_from_path(path: Path | str) -> str:
    """Derive a stable dataset slug from an input file name."""
    stem = Path(path).stem.lower()
    name = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    for prefix in ("transactions_", "transaction_", "synthetic_", "dataset_", "data_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for suffix in ("_train", "_training", "_pred", "_prediction"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = name.strip("_")
    if not name:
        raise ValueError(f"Cannot derive dataset name from path: {path}")
    return name


def training_timestamp(created_at: datetime | None = None) -> str:
    return (created_at or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")


def create_training_run_dir(
    artifacts_root: Path | str,
    train_path: Path | str,
    created_at: datetime | None = None,
) -> tuple[str, str, Path]:
    """Create ``<root>/<dataset>/<timestamp>`` and return its identifiers."""
    root = Path(artifacts_root)
    dataset = dataset_name_from_path(train_path)
    run_id = training_timestamp(created_at)
    dataset_dir = root / dataset
    run_dir = dataset_dir / run_id

    suffix = 2
    while run_dir.exists():
        run_dir = dataset_dir / f"{run_id}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return dataset, run_dir.name, run_dir


def replace_latest_artifact_dirs(
    artifacts_root: Path | str,
    dataset: str,
    run_dir: Path | str,
) -> None:
    """Copy the completed run to root-level and dataset-level ``latest`` dirs."""
    root = Path(artifacts_root)
    source = Path(run_dir)
    for target in (root / dataset / LATEST_DIRNAME, root / LATEST_DIRNAME):
        _replace_dir(source, target)


def resolve_artifact_dir(
    artifacts_root: Path | str,
    experiment: str | None = None,
) -> Path:
    """Resolve a configured artifact experiment to a concrete directory.

    Supported experiment values include ``latest``, ``simple_spatial``,
    ``simple_spatial/latest`` and ``simple_spatial/<timestamp>``.
    """
    root = Path(artifacts_root)
    if experiment is None:
        latest = root / LATEST_DIRNAME
        return latest if latest.exists() else root

    requested = Path(experiment)
    if requested.is_absolute():
        return requested

    if requested.parts == (LATEST_DIRNAME,):
        return root / LATEST_DIRNAME

    if len(requested.parts) == 1:
        dataset_latest = root / requested / LATEST_DIRNAME
        if dataset_latest.exists():
            return dataset_latest
        timestamp_matches = [
            dataset_dir / requested
            for dataset_dir in root.iterdir()
            if dataset_dir.is_dir() and (dataset_dir / requested).is_dir()
        ] if root.exists() else []
        if len(timestamp_matches) == 1:
            return timestamp_matches[0]
        if len(timestamp_matches) > 1:
            matches = ", ".join(str(path) for path in timestamp_matches)
            raise ValueError(
                f"Ambiguous experiment `{experiment}`. Use dataset/timestamp. "
                f"Matches: {matches}"
            )
        return root / requested

    return root / requested


def require_artifact_file(artifact_dir: Path | str, filename: str) -> Path:
    path = Path(artifact_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Artifact file not found: {path}")
    return path


def _replace_dir(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
