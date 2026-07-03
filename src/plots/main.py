"""Unified entrypoints for plot exports."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

PlotPipeline = Callable[[str | None], Path]
PlotServer = Callable[[Path], None]


def _tensorboard(experiment: str | None = None) -> Path:
    from .tensorboard import export_tensorboard

    return export_tensorboard(experiment=experiment)


def _serve_tensorboard(run_dir: Path) -> None:
    from .tensorboard import serve_tensorboard

    serve_tensorboard(run_dir)


PLOT_PIPELINES: dict[str, PlotPipeline] = {
    "tensorboard": _tensorboard,
}
PLOT_SERVERS: dict[str, PlotServer] = {
    "tensorboard": _serve_tensorboard,
}


def main(plot_type: str = "tensorboard", experiment: str | None = None) -> Path:
    normalized_plot_type = plot_type.lower()
    pipeline = PLOT_PIPELINES.get(normalized_plot_type)
    if pipeline is None:
        supported_types = ", ".join(PLOT_PIPELINES)
        raise ValueError(
            f"Unsupported plot type: {plot_type}. Supported values: {supported_types}."
        )
    return pipeline(experiment=experiment)


def serve(plot_type: str, run_dir: Path) -> None:
    normalized_plot_type = plot_type.lower()
    server = PLOT_SERVERS.get(normalized_plot_type)
    if server is None:
        supported_types = ", ".join(PLOT_SERVERS)
        raise ValueError(
            f"Unsupported plot UI type: {plot_type}. Supported values: {supported_types}."
        )
    server(run_dir)
