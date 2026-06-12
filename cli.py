"""Project CLI for dataset generation, model training, prediction and plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer
from loguru import logger

app = typer.Typer(add_completion=False)


def _syntetic(dataset_type: Literal["vanilla", "coherent", "simple", "timing"]) -> Path:
    from src.datasets.main import generate

    return generate(dataset_type=dataset_type)


def _train(model_type: str) -> Path:
    # `base` is currently routed to the available training pipeline.
    if model_type in {"hier", "base"}:
        if model_type == "base":
            logger.info("Model type 'base' currently uses the same training pipeline as 'hier'.")
        from src.models.hier_transformer.train import main

        return main()

    raise ValueError(f"Unsupported model type: {model_type}")


def _plot(model_type: str, history: str | None, runs_dir: str | None) -> None:
    # `base` and `hier` share the same exporter at the moment.
    if model_type not in {"hier", "base"}:
        raise ValueError(f"Unsupported model type: {model_type}")

    from src.plots.tensorboard import main

    if model_type == "base":
        logger.info("Model type 'base' currently uses the same plotting pipeline as 'hier'.")

    resolved_history = Path(history) if history else Path("checkpoints") / "history.json"
    resolved_runs_dir = Path(runs_dir) if runs_dir else Path("runs") / model_type
    main(history_path=resolved_history, runs_dir=resolved_runs_dir)


def _pred(model_type: str) -> Path:
    # `base` is currently routed to the available prediction pipeline.
    if model_type in {"hier", "base"}:
        if model_type == "base":
            logger.info("Model type 'base' currently uses the same prediction pipeline as 'hier'.")
        from src.models.hier_transformer.pred import main

        return main()

    raise ValueError(f"Unsupported model type: {model_type}")


@app.command("syntetic")
def generate_command(
    dataset_type: Literal["vanilla", "coherent", "simple", "timing"] = typer.Option(
        ...,
        "--type",
        help="Dataset syntetic to generate.",
    ),
) -> None:
    output_path = _syntetic(dataset_type)
    logger.success("Dataset generated: {}", output_path)

@app.command("train")
def train_command(
    model_type: Literal["base", "hier"] = typer.Option(
        "hier",
        "--type",
        help="Model variant to train.",
    ),
) -> None:
    ckpt_path = _train(model_type)
    logger.success("Training completed: {}", ckpt_path)


@app.command("pred")
def pred_command(
    model_type: Literal["base", "hier"] = typer.Option(
        "hier",
        "--type",
        help="Model variant to predict.",
    ),
) -> None:
    out_path = _pred(model_type)
    logger.success("Prediction completed: {}", out_path)


@app.command("plot")
def plot_command(
    model_type: Literal["base", "hier"] = typer.Option(
        "hier",
        "--type",
        help="Model variant to plot.",
    ),
    history: Path | None = typer.Option(
        None,
        "--history",
        help="Path to history.json (default: checkpoints/history.json).",
    ),
    runs_dir: Path | None = typer.Option(
        None,
        "--runs-dir",
        help="Parent directory for TensorBoard runs (default: runs/<type>).",
    ),
) -> None:
    _plot(model_type, str(history) if history else None, str(runs_dir) if runs_dir else None)


def main() -> None:
    import sys

    normalized_args = ["--type" if arg == "-type" else arg for arg in sys.argv[1:]]
    app(prog_name="py", args=normalized_args)


if __name__ == "__main__":
    main()
