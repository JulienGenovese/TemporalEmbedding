"""Replay hierarchical training history into TensorBoard event files.

Reads a selected training artifact directory and writes TensorBoard runs under
the configured run directory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

from src.config import config
from src.models.hier_transformer.artifacts import (
    HISTORY_FILENAME,
    TRAIN_EVAL_HISTORY_FILENAME,
    VAL_HISTORY_FILENAME,
    require_artifact_file,
    resolve_artifact_dir,
)

# model.plot config section
_MODEL_PLOT_SECTION = "model.plot"
_MODEL_PATHS_SECTION = "model.hier_transformer.paths"


def _load_sibling(history_path: Path | str, fname: str) -> list[dict] | None:
    """Load a per-epoch eval file living next to ``history.json`` (or None)."""
    f = Path(history_path).with_name(fname)
    return json.loads(f.read_text()) if f.exists() else None


def _resolve_configured_plot_paths() -> tuple[Path, Path, Path]:
    configured_history_path = config.get(
        _MODEL_PLOT_SECTION,
        "history",
        value_type=Path,
    )
    runs_dir = config.get(_MODEL_PLOT_SECTION, "runs_dir", value_type=Path)
    artifacts_root = config.get(
        _MODEL_PATHS_SECTION,
        "model_output_dir",
        value_type=Path,
    )
    return configured_history_path, runs_dir, artifacts_root


def _resolve_plot_paths(experiment: str | None = None) -> tuple[Path, Path, Path]:
    configured_history_path, runs_dir, artifacts_root = _resolve_configured_plot_paths()
    artifact_dir = resolve_artifact_dir(artifacts_root, experiment)

    if experiment is None and not (artifact_dir / HISTORY_FILENAME).exists():
        if configured_history_path.exists():
            return configured_history_path, runs_dir / "latest", runs_dir

    history_path = require_artifact_file(artifact_dir, HISTORY_FILENAME)
    return (
        history_path,
        _run_dir_for_experiment(runs_dir, artifacts_root, artifact_dir),
        runs_dir,
    )


def _run_dir_for_experiment(
    runs_dir: Path,
    artifacts_root: Path,
    artifact_dir: Path,
) -> Path:
    try:
        relative_experiment = artifact_dir.resolve().relative_to(
            artifacts_root.resolve(),
        )
    except ValueError:
        relative_experiment = Path(artifact_dir.name)
    if relative_experiment == Path("."):
        relative_experiment = Path("latest")
    return runs_dir / relative_experiment


def tensorboard_ui_command(run_dir: Path | str) -> str:
    return f'uv run tensorboard --logdir "{Path(run_dir).resolve()}"'


def _tensorboard_process_args(log_dir: Path) -> list[str]:
    return [sys.executable, "-m", "tensorboard.main", "--logdir", str(log_dir)]


def serve_tensorboard(log_dir: Path | str) -> None:
    """Start the TensorBoard UI for the exports root directory."""
    resolved_log_dir = Path(log_dir).resolve()
    logger.info(
        "Starting TensorBoard UI with: {}",
        tensorboard_ui_command(resolved_log_dir),
    )
    try:
        subprocess.run(_tensorboard_process_args(resolved_log_dir), check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"TensorBoard UI exited with status {exc.returncode}."
        ) from exc


# ---------------------------------------------------------------------------
# TensorBoard export
# ---------------------------------------------------------------------------

class TensorBoardExporter:
    """Replays JSON training history into TensorBoard event files.

    The selected experiment is the TensorBoard run group. ``train`` / ``val`` /
    ``random`` are sub-runs inside that experiment so matching metric tags
    overlay on the same charts.

    Usable as a context manager.
    """

    # chart tag -> key in the per-epoch eval dict
    _EVAL_TAGS = {
        "loss": "loss",
        "loss_mtm": "loss_mtm",
        "loss_contrastive": "loss_contrastive",
        "accuracy": "infonce_acc",
        "lift": "infonce_lift",
    }

    def __init__(self, log_dir: Path | str):
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = Path(log_dir)
        self._SummaryWriter = SummaryWriter
        self._writers: dict[str, object] = {}

    def _run(self, name: str):
        """Lazily create one SummaryWriter per split sub-run."""
        if name not in self._writers:
            self._writers[name] = self._SummaryWriter(
                log_dir=str(self.log_dir / name)
            )
        return self._writers[name]

    def add_history(
        self,
        eval_history: list[dict],
        run: str,
        *,
        include_random_baseline: bool = False,
    ) -> None:
        """Write per-epoch eval scalars for one split sub-run."""
        writer = self._run(run)
        random = self._run("random") if include_random_baseline else None
        for e in eval_history:
            if "epoch" not in e:
                continue
            x = e["epoch"]
            for tag, key in self._EVAL_TAGS.items():
                if key in e:
                    writer.add_scalar(tag, e[key], x)
            if random is not None and "infonce_acc_random" in e:
                random.add_scalar("accuracy", e["infonce_acc_random"], x)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.flush()
            writer.close()

    def __enter__(self) -> "TensorBoardExporter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def export_tensorboard(experiment: str | None = None) -> Path:
    """Replay the configured hierarchical histories into a TensorBoard run.

    If ``experiment`` is provided, it selects an artifact subdirectory such as
    ``simple_spatial/latest`` or ``simple_spatial/<timestamp>``.
    """
    resolved_history_path, run_dir, tensorboard_log_dir = _resolve_plot_paths(experiment)
    if run_dir.exists():
        shutil.rmtree(run_dir)  # avoid merging stale event files

    logger.info("Exporting TensorBoard history from {}", resolved_history_path)
    logger.info("TensorBoard experiment run: {}", run_dir)
    with TensorBoardExporter(run_dir) as tb:
        train_eval = _load_sibling(resolved_history_path, TRAIN_EVAL_HISTORY_FILENAME)
        if train_eval:
            tb.add_history(train_eval, "train", include_random_baseline=True)
        val = _load_sibling(resolved_history_path, VAL_HISTORY_FILENAME)
        if val:
            tb.add_history(val, "val")
    logger.info(
        "Launch TensorBoard UI with: {}",
        tensorboard_ui_command(tensorboard_log_dir),
    )
    return tensorboard_log_dir

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> Path:
    return export_tensorboard()
