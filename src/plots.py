"""Plot training curves saved by :mod:`src.train`.

Reads ``checkpoints/history.json`` (one record per step) plus the sibling
per-epoch eval files ``train_eval_history.json`` / ``val_history.json`` and
writes, under ``checkpoints/plots/``:

  * ``training_curves.png`` — per-step training curves with the per-epoch
    train/val eval overlaid on the loss and accuracy panels (so train-vs-val
    divergence is visible at a glance); the accuracy panel also shows the
    random baseline and the normalized lift.
  * ``mtm_breakdown.png``  — per-field MTM loss.

With ``--tensorboard`` the same history is replayed into TensorBoard event
files, with one chart per metric and ``train`` / ``val`` (and the random
accuracy baseline) overlaid as legend-distinguished runs. By default it
overwrites ``runs/latest/`` (so ``tensorboard --logdir runs`` shows only the
most recent export); pass ``--archive`` to keep a timestamped
``runs/<timestamp>/`` folder instead.

    uv run python -m src.plots [--history ...] [--tensorboard] [--archive]
    tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from itertools import groupby
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-friendly: no display required
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loading / helpers
# ---------------------------------------------------------------------------

def load_history(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `uv run python -m src.train` first."
        )
    return json.loads(path.read_text())


def _load_sibling(history_path: Path | str, fname: str) -> list[dict] | None:
    """Load a per-epoch eval file living next to ``history.json`` (or None)."""
    f = Path(history_path).with_name(fname)
    return json.loads(f.read_text()) if f.exists() else None


def _column(history: list[dict], key: str) -> list[float]:
    return [h[key] for h in history]


def _smooth(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return values
    out: list[float] = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def _epoch_to_step(history: list[dict]) -> dict[int, int]:
    """Map each epoch to its last training step, to place per-epoch eval
    points on the per-step x-axis."""
    return {h["epoch"]: h["step"] for h in history if "epoch" in h}


def _eval_xy(eval_hist: list[dict] | None, key: str,
             epoch_step: dict[int, int]) -> tuple[list[int], list[float]]:
    """(x, y) for a per-epoch eval series, x mapped to step via ``epoch_step``."""
    if not eval_hist:
        return [], []
    pts = [(epoch_step[e["epoch"]], e[key]) for e in eval_hist
           if key in e and e["epoch"] in epoch_step]
    return [x for x, _ in pts], [y for _, y in pts]


def _draw(ax, title: str, curves: list[tuple], *,
          log: bool = False, xlabel: str = "step", legend: bool | None = None) -> None:
    """Plot ``curves`` (each ``(x, y, plot_kwargs)``) on ``ax`` and style it.

    ``legend`` defaults to auto: shown only when ≥2 curves carry a label.
    """
    n_labels = sum("label" in kw for _, _, kw in curves)
    for x, y, kw in curves:
        ax.plot(x, y, **kw)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    if log:
        ax.set_yscale("log")
    if (n_labels >= 2 if legend is None else legend) and n_labels:
        ax.legend(loc="best", fontsize=8)


def _save(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: list[dict],
    out_path: Path,
    smoothing: int = 3,
    train_eval: list[dict] | None = None,
    val: list[dict] | None = None,
) -> Path:
    """Per-step training curves; loss & accuracy panels overlay per-epoch
    train/val eval so divergence (overfitting) is visible at a glance."""
    steps = _column(history, "step")
    epoch_step = _epoch_to_step(history)

    def train_step(key: str) -> list[tuple]:
        """Per-step training series: faint raw + smoothed line (C0)."""
        raw = _column(history, key)
        curves = [(steps, raw, dict(color="C0", alpha=0.3))]
        if smoothing > 1:
            curves.append((steps, _smooth(raw, smoothing), dict(color="C0", label="train")))
        else:
            curves[0][2]["label"] = "train"
        return curves

    def eval_overlay(key: str) -> list[tuple]:
        """Per-epoch train-eval (C1) and val (C3) points, mapped to step."""
        curves = []
        for src, color, label in ((train_eval, "C1", "train (full)"), (val, "C3", "val")):
            x, y = _eval_xy(src, key, epoch_step)
            if x:
                curves.append((x, y, dict(color=color, marker="o", ms=3, label=label)))
        return curves

    # loss panels: per-step training + per-epoch train/val overlay
    panels: list[tuple[str, list[tuple], bool]] = [
        (title, train_step(key) + eval_overlay(key), log)
        for title, key, log in (
            ("Total loss", "loss", False),
            ("MTM loss", "loss_mtm", True),
            ("InfoNCE loss", "loss_contrastive", False),
        )
    ]

    # accuracy panel: train/val acc + random baseline + normalized lift
    acc = train_step("infonce_acc")
    x, y = _eval_xy(val, "infonce_acc", epoch_step)
    if x:
        acc.append((x, y, dict(color="C3", marker="o", ms=3, label="val")))
    if "infonce_acc_random" in history[0]:
        acc.append((steps, _column(history, "infonce_acc_random"),
                    dict(color="gray", ls="--", alpha=0.7, label="random")))
    if "infonce_lift" in history[0]:
        acc.append((steps, _smooth(_column(history, "infonce_lift"), smoothing),
                    dict(color="C2", label="lift")))
    panels.append(("InfoNCE acc / random / lift", acc, False))

    # training-only diagnostics (no eval counterpart)
    panels += [(title, train_step(key), False)
               for title, key in (("Temperature", "temperature"),
                                   ("Grad norm", "grad_norm"),
                                   ("Learning rate", "lr"))]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, (title, curves, log) in zip(axes.flat, panels):
        _draw(ax, title, curves, log=log)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")

    fig.suptitle("Training curves (lines = train, markers = per-epoch eval)", y=1.02)
    return _save(fig, out_path)


def plot_mtm_breakdown(history: list[dict], out_path: Path, smoothing: int = 3) -> Path | None:
    """Per-field MTM loss — one panel for categorical (CE), one for numeric (smooth-L1)."""
    if not history or not history[0].get("mtm_breakdown"):
        return None

    steps = _column(history, "step")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, prefix, title, log in (
        (axes[0], "cat_", "Categorical MTM (cross-entropy)", False),
        (axes[1], "num_", "Numeric MTM (smooth-L1, log scale)", True),
    ):
        keys = sorted({k for h in history for k in h["mtm_breakdown"] if k.startswith(prefix)})
        curves = [(steps, _smooth([h["mtm_breakdown"].get(k, float("nan")) for h in history], smoothing),
                   dict(label=k[4:])) for k in keys]
        _draw(ax, title, curves, log=log, legend=True)

    fig.suptitle("MTM loss per field", y=1.02)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# TensorBoard export
# ---------------------------------------------------------------------------

class TensorBoardExporter:
    """Replays JSON training history into TensorBoard event files.

    One chart per **metric** (the scalar *tag*, e.g. ``loss`` or ``accuracy``);
    ``train`` / ``val`` / ``random`` are written as separate TB *runs* (sub-
    directories) so they overlay on the same chart, distinguished by the run
    legend. The x-axis is the global training step throughout — per-epoch eval
    points are placed at the step that closed their epoch. Usable as a context
    manager.
    """

    # chart tag -> key in the per-step training-history dict
    _STEP_TAGS = {
        "loss": "loss",
        "loss_mtm": "loss_mtm",
        "loss_contrastive": "loss_contrastive",
        "accuracy": "infonce_acc",
        "lift": "infonce_lift",
        "temperature": "temperature",
        "grad_norm": "grad_norm",
        "lr": "lr",
    }
    # chart tag -> key in the per-epoch eval dict (metrics with an eval counterpart)
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
        """Lazily create one SummaryWriter per run sub-directory."""
        if name not in self._writers:
            self._writers[name] = self._SummaryWriter(
                log_dir=str(self.log_dir / name)
            )
        return self._writers[name]

    def add_train(self, history: list[dict]) -> None:
        """Per-**epoch-mean** training curve (run ``train``), the random-accuracy
        baseline (run ``random``, overlaid on the ``accuracy`` chart) and the
        per-field MTM breakdown.

        Per-step records are grouped by epoch and the mean of each metric is
        logged at the step that closed the epoch — so the curve is smooth and
        its points line up with the per-epoch ``val`` overlay."""
        train = self._run("train")
        random = self._run("random")

        def mean(vals: list[float]) -> float:
            return sum(vals) / len(vals)

        for _epoch, group in groupby(history, key=lambda h: h.get("epoch")):
            records = list(group)
            step = max(h["step"] for h in records)
            for tag, key in self._STEP_TAGS.items():
                vals = [h[key] for h in records if key in h]
                if vals:
                    train.add_scalar(tag, mean(vals), step)
            rnd = [h["infonce_acc_random"] for h in records
                   if "infonce_acc_random" in h]
            if rnd:
                random.add_scalar("accuracy", mean(rnd), step)
            fields: dict[str, list[float]] = {}
            for h in records:
                for field, val in h.get("mtm_breakdown", {}).items():
                    fields.setdefault(field, []).append(val)
            for field, vals in fields.items():
                train.add_scalar(f"mtm/{field}", mean(vals), step)

    def add_eval(self, eval_history: list[dict], run: str,
                 epoch_step: dict[int, int]) -> None:
        """Per-epoch eval scalars for ``run`` (e.g. ``val``), placed on the
        step x-axis so each metric overlays its training curve on one chart."""
        writer = self._run(run)
        for e in eval_history:
            step = epoch_step.get(e["epoch"])
            if step is None:
                continue
            for tag, key in self._EVAL_TAGS.items():
                if key in e:
                    writer.add_scalar(tag, e[key], step)

    def close(self) -> None:
        for w in self._writers.values():
            w.flush()
            w.close()

    def __enter__(self) -> "TensorBoardExporter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def export_tensorboard(history_path: Path | str, runs_dir: Path | str,
                       archive: bool = False) -> Path:
    """Replay ``history_path`` (and the sibling val-eval file) into a TB run.

    Default: overwrite ``runs_dir/latest`` so only the most recent export is
    shown. ``archive=True`` keeps a timestamped ``runs_dir/<timestamp>`` run.
    """
    history_path = Path(history_path)
    if archive:
        run_dir = Path(runs_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        run_dir = Path(runs_dir) / "latest"
        if run_dir.exists():
            shutil.rmtree(run_dir)  # avoid merging stale event files into 'latest'

    history = load_history(history_path)
    epoch_step = _epoch_to_step(history)
    with TensorBoardExporter(run_dir) as tb:
        tb.add_train(history)
        val = _load_sibling(history_path, "val_history.json")
        if val:
            tb.add_eval(val, "val", epoch_step)
    return run_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    history_path: Path | str = Path("checkpoints") / "history.json",
    out_dir: Path | str = Path("checkpoints") / "plots",
    smoothing: int = 3,
    tensorboard: bool = False,
    runs_dir: Path | str = Path("runs"),
    archive: bool = False,
) -> None:
    history = load_history(history_path)
    train_eval = _load_sibling(history_path, "train_eval_history.json")
    val = _load_sibling(history_path, "val_history.json")
    out_dir = Path(out_dir)

    p1 = plot_training_curves(history, out_dir / "training_curves.png", smoothing, train_eval, val)
    print(f"Saved {p1}")
    if (p2 := plot_mtm_breakdown(history, out_dir / "mtm_breakdown.png", smoothing)):
        print(f"Saved {p2}")
    else:
        print("No mtm_breakdown in history — skipping per-field plot.")

    if tensorboard:
        run_dir = export_tensorboard(history_path, runs_dir, archive=archive)
        print(f"TensorBoard events → {run_dir}  (tensorboard --logdir {runs_dir})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="checkpoints/history.json")
    parser.add_argument("--out-dir", default="checkpoints/plots")
    parser.add_argument("--smoothing", type=int, default=3)
    parser.add_argument("--tensorboard", action="store_true",
                        help="also export the history into TensorBoard event files")
    parser.add_argument("--runs-dir", default="runs",
                        help="parent directory for TensorBoard run folders")
    parser.add_argument("--archive", action="store_true",
                        help="keep a timestamped run instead of overwriting runs/latest")
    args = parser.parse_args()
    main(args.history, args.out_dir, args.smoothing,
         tensorboard=args.tensorboard, runs_dir=args.runs_dir, archive=args.archive)
