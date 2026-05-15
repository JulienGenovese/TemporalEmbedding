"""Plot training curves saved by :mod:`src.train`.

Reads ``checkpoints/history.json`` (one record per training step) and writes
PNGs under ``checkpoints/plots/``:

  * ``training_curves.png`` — total / MTM / InfoNCE losses + InfoNCE top-1
    accuracy + learnable temperature + gradient norm
  * ``mtm_breakdown.png``  — per-field MTM loss (one curve per cat_*/num_* head)

With ``--tensorboard`` the same history is also replayed into TensorBoard
event files under ``runs/<timestamp>/`` via :class:`TensorBoardExporter`.

Usage:
    uv run python -m src.plots                                  # default paths
    uv run python -m src.plots --history path/to/history.json
    uv run python -m src.plots --tensorboard                    # + TB export
    tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-friendly: no display required
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_history(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `uv run python -m src.train` first."
        )
    return json.loads(path.read_text())


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


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: list[dict],
    out_path: Path,
    smoothing: int = 3,
) -> Path:
    """Six-panel summary of the run."""
    steps = _column(history, "step")
    panels = [
        ("Total loss",        "loss",             False),
        ("MTM loss",          "loss_mtm",         True),   # often huge → log scale
        ("InfoNCE loss",      "loss_contrastive", False),
        ("InfoNCE top-1 acc", "infonce_acc",      False),
        ("Temperature",       "temperature",      False),
        ("Grad norm",         "grad_norm",        False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, key, log) in zip(axes.flat, panels):
        raw = _column(history, key)
        ax.plot(steps, raw, alpha=0.35, label="raw")
        if smoothing > 1:
            ax.plot(steps, _smooth(raw, smoothing),
                    color="C0", label=f"ma({smoothing})")
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if log:
            ax.set_yscale("log")
        if smoothing > 1:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle("Training curves", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mtm_breakdown(
    history: list[dict],
    out_path: Path,
    smoothing: int = 3,
) -> Path | None:
    """Per-field MTM loss curves — one panel for categorical, one for numeric."""
    if not history or not history[0].get("mtm_breakdown"):
        return None

    steps = _column(history, "step")
    cat_keys = sorted({k for h in history for k in h["mtm_breakdown"] if k.startswith("cat_")})
    num_keys = sorted({k for h in history for k in h["mtm_breakdown"] if k.startswith("num_")})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for key in cat_keys:
        vals = [h["mtm_breakdown"].get(key, float("nan")) for h in history]
        axes[0].plot(steps, _smooth(vals, smoothing), label=key[4:])
    axes[0].set_title("Categorical MTM (cross-entropy)")
    axes[0].set_xlabel("step")
    axes[0].grid(True, alpha=0.3)
    if cat_keys:
        axes[0].legend(loc="best", fontsize=8)

    for key in num_keys:
        vals = [h["mtm_breakdown"].get(key, float("nan")) for h in history]
        axes[1].plot(steps, _smooth(vals, smoothing), label=key[4:])
    axes[1].set_title("Numeric MTM (smooth-L1, log scale)")
    axes[1].set_xlabel("step")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    if num_keys:
        axes[1].legend(loc="best", fontsize=8)

    fig.suptitle("MTM loss per field", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# TensorBoard export
# ---------------------------------------------------------------------------

class TensorBoardExporter:
    """Replays JSON training history into TensorBoard event files.

    Post-hoc counterpart of the matplotlib plots above: it consumes the
    same artifacts written by :mod:`src.train` (``history.json`` and the
    optional ``train_eval_history.json`` / ``val_history.json``) and emits
    event files — no changes to the training loop required.

    Scalar layout (so TensorBoard groups them sensibly):
      * ``train/*``      — per-step scalars, x-axis = global step
      * ``mtm/*``        — per-field MTM losses, x-axis = global step
      * ``eval_train/*`` — epoch-level eval over the train loader
      * ``eval_val/*``   — epoch-level eval over the val loader

    Usable as a context manager so the writer is always flushed/closed.
    """

    _STEP_KEYS = (
        "loss", "loss_mtm", "loss_contrastive",
        "infonce_acc", "temperature", "grad_norm", "lr",
    )
    _EVAL_KEYS = ("loss", "loss_mtm", "loss_contrastive", "infonce_acc")

    def __init__(self, log_dir: Path | str):
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = Path(log_dir)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def add_history(self, history: list[dict]) -> None:
        """Per-step scalars + per-field MTM breakdown."""
        for h in history:
            step = h["step"]
            for key in self._STEP_KEYS:
                if key in h:
                    self._writer.add_scalar(f"train/{key}", h[key], step)
            for key, val in h.get("mtm_breakdown", {}).items():
                self._writer.add_scalar(f"mtm/{key}", val, step)

    def add_eval(self, eval_history: list[dict], tag: str) -> None:
        """Epoch-level eval scalars under ``<tag>/*`` (x-axis = epoch)."""
        for e in eval_history:
            epoch = e["epoch"]
            for key in self._EVAL_KEYS:
                if key in e:
                    self._writer.add_scalar(f"{tag}/{key}", e[key], epoch)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()

    def __enter__(self) -> "TensorBoardExporter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def export_tensorboard(history_path: Path | str, runs_dir: Path | str) -> Path:
    """Replay ``history_path`` (and sibling eval files) into a fresh TB run."""
    history_path = Path(history_path)
    run_dir = Path(runs_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    with TensorBoardExporter(run_dir) as tb:
        tb.add_history(load_history(history_path))
        for fname, tag in (
            ("train_eval_history.json", "eval_train"),
            ("val_history.json",        "eval_val"),
        ):
            f = history_path.with_name(fname)
            if f.exists():
                tb.add_eval(json.loads(f.read_text()), tag)
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
) -> None:
    history = load_history(history_path)
    out_dir = Path(out_dir)

    p1 = plot_training_curves(history, out_dir / "training_curves.png", smoothing)
    print(f"Saved {p1}")
    p2 = plot_mtm_breakdown(history, out_dir / "mtm_breakdown.png", smoothing)
    if p2:
        print(f"Saved {p2}")
    else:
        print("No mtm_breakdown in history — skipping per-field plot.")

    if tensorboard:
        run_dir = export_tensorboard(history_path, runs_dir)
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
    args = parser.parse_args()
    main(args.history, args.out_dir, args.smoothing,
         tensorboard=args.tensorboard, runs_dir=args.runs_dir)
