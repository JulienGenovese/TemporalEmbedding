"""Project CLI for dataset generation, model training, prediction and plotting."""

from __future__ import annotations

from typing import Annotated, NoReturn

import typer
from loguru import logger

from src.datasets.main import generate as generate_dataset
from src.datasets.types import DatasetType
from src.eval import main as eval_main
from src.models import main as model_main
from src.models.types import ModelType
from src.plots import main as plot_main

app = typer.Typer(add_completion=False)

DatasetTypeOption = Annotated[
    DatasetType,
    typer.Option(
        "--type",
        "-t",
        help="Synthetic dataset to generate.",
    ),
]
ModelTypeOption = Annotated[
    ModelType,
    typer.Option(
        "--type",
        "-t",
        help="Model variant to use.",
    ),
]


PerturbationAnalysisOption = Annotated[
    eval_main.PerturbationAnalysis,
    typer.Option(
        "--analysis",
        "-a",
        help="Perturbation analysis to run.",
    ),
]


PlotTypeOption = Annotated[
    str,
    typer.Option(
        "--type",
        "-t",
        help="Plot backend to use.",
    ),
]
PlotServeOption = Annotated[
    bool,
    typer.Option(
        "--serve",
        help="Start the plot UI after exporting.",
    ),
]
PlotExperimentOption = Annotated[
    str | None,
    typer.Option(
        "--experiment",
        "-e",
        help=(
            "Artifact experiment to plot, e.g. latest, simple_spatial, "
            "or simple_spatial/2026-06-19_17-25-00."
        ),
    ),
]


def _exit_user_error(exc: Exception) -> NoReturn:
    typer.secho(str(exc), fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1) from exc


@app.command("synthetic")
def synthetic_command(dataset_type: DatasetTypeOption) -> None:
    try:
        output_paths = generate_dataset(dataset_type=dataset_type)
    except (ValueError, KeyError, TypeError) as exc:
        _exit_user_error(exc)
    for split, path in output_paths.items():
        logger.success("Dataset generated ({}): {}", split, path)


@app.command("train")
def train_command(model_type: ModelTypeOption = ModelType.HIER) -> None:
    try:
        output = model_main.main(action="train", model_type=model_type)
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        _exit_user_error(exc)
    logger.success("Training completed: {}", output)


@app.command("pred")
def pred_command(model_type: ModelTypeOption = ModelType.HIER) -> None:
    try:
        output = model_main.main(action="pred", model_type=model_type)
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        _exit_user_error(exc)
    logger.success("Prediction completed: {}", output)


@app.command("perturbation")
def perturbation_command(
    model_type: ModelTypeOption = ModelType.HIER,
    analysis: PerturbationAnalysisOption = eval_main.PerturbationAnalysis.SENSIBILITY,
) -> None:
    try:
        output = eval_main.main(
            analysis=analysis.value,
            model_type=model_type,
        )
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        _exit_user_error(exc)
    logger.success("Perturbation analysis completed: {}", output)


@app.command("plot")
def plot_command(
    plot_type: PlotTypeOption = "tensorboard",
    experiment: PlotExperimentOption = None,
    serve: PlotServeOption = False,
) -> None:
    try:
        output = plot_main.main(plot_type=plot_type, experiment=experiment)
        logger.success("TensorBoard export completed: {}", output)
        if serve:
            plot_main.serve(plot_type=plot_type, run_dir=output)
    except (ValueError, KeyError, TypeError, FileNotFoundError, RuntimeError) as exc:
        _exit_user_error(exc)


def main() -> None:
    app(prog_name="py")
