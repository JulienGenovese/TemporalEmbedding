"""Replay training history into TensorBoard event files.

Reads ``checkpoints/history.json`` (one record per step) plus the sibling
``train_eval_history.json`` / ``val_history.json`` and writes TensorBoard
runs under ``runs/latest/`` (overwritten at every export).

    uv run python -m src.plots [--history ...]
    tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

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

# ---------------------------------------------------------------------------
# TensorBoard export
# ---------------------------------------------------------------------------

class TensorBoardExporter:
    """Replays JSON training history into TensorBoard event files.

    One chart per **metric** (the scalar *tag*, e.g. ``loss`` or ``accuracy``);
    ``train`` / ``val`` / ``random`` are written as separate TB *runs* (sub-
    directories) so they overlay on the same chart, distinguished by the run
    legend.

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
        """Lazily create one SummaryWriter per run sub-directory."""
        if name not in self._writers:
            self._writers[name] = self._SummaryWriter(
                log_dir=str(self.log_dir / name)
            )
        return self._writers[name]

    def add_train(
        self,
        eval_history: list[dict],
    ) -> None:
        """Per-epoch train scalars for run ``train`` plus random baseline."""
        train = self._run("train")
        random = self._run("random")
        for e in eval_history:
            if "epoch" not in e:
                continue
            x = e["epoch"]
            for tag, key in self._EVAL_TAGS.items():
                if key in e:
                    train.add_scalar(tag, e[key], x)
            if "infonce_acc_random" in e:
                random.add_scalar("accuracy", e["infonce_acc_random"], x)

    def add_eval(self, eval_history: list[dict], run: str) -> None:
        """Per-epoch eval scalars for ``run`` (e.g. ``val``)."""
        writer = self._run(run)
        for e in eval_history:
            if "epoch" not in e:
                continue
            x = e["epoch"]
            for tag, key in self._EVAL_TAGS.items():
                if key in e:
                    writer.add_scalar(tag, e[key], x)

    def close(self) -> None:
        for w in self._writers.values():
            w.flush()
            w.close()

    def __enter__(self) -> "TensorBoardExporter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

def export_tensorboard(history_path: Path | str, runs_dir: Path | str) -> Path:
    """Replay ``history_path`` (and the sibling val-eval file) into a TB run.

    Always overwrites ``runs_dir/latest`` so only the most recent export is
    shown.
    """
    history_path = Path(history_path)
    run_dir = Path(runs_dir) / "latest"
    if run_dir.exists():
        shutil.rmtree(run_dir) # avoid merging stale event files into 'latest'

    load_history(history_path)
    with TensorBoardExporter(run_dir) as tb:
        train_eval = _load_sibling(history_path, "train_eval_history.json")
        if train_eval:
            tb.add_train(train_eval)
        val = _load_sibling(history_path, "val_history.json")
        if val:
            tb.add_eval(val, "val")
    return run_dir

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    history_path: Path | str = Path("checkpoints") / "history.json",
    runs_dir: Path | str = Path("runs"),
) -> None:
    run_dir = export_tensorboard(history_path, runs_dir)
    print(f"TensorBoard events → {run_dir} (tensorboard --logdir {runs_dir})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="checkpoints/history.json")
    parser.add_argument("--runs-dir", default="runs",
                        help="parent directory for TensorBoard run folders")
    args = parser.parse_args()
    main(args.history, runs_dir=args.runs_dir)