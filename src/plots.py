"""Plot training curves saved by :mod:`src.train`.

Reads ``checkpoints/history.json`` (one record per training step) and writes
PNGs under ``checkpoints/plots/``:

  * ``training_curves.png`` — total / MTM / InfoNCE losses + InfoNCE top-1
    accuracy + learnable temperature + gradient norm
  * ``mtm_breakdown.png``  — per-field MTM loss (one curve per cat_*/num_* head)

Usage:
    uv run python -m src.plots                                  # default paths
    uv run python -m src.plots --history path/to/history.json
"""

from __future__ import annotations

import argparse
import json
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
# Entry point
# ---------------------------------------------------------------------------

def main(
    history_path: Path | str = Path("checkpoints") / "history.json",
    out_dir: Path | str = Path("checkpoints") / "plots",
    smoothing: int = 3,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="checkpoints/history.json")
    parser.add_argument("--out-dir", default="checkpoints/plots")
    parser.add_argument("--smoothing", type=int, default=3)
    args = parser.parse_args()
    main(args.history, args.out_dir, args.smoothing)
