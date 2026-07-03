#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


SUPPORTED_EXPERIMENTS = ("simple_spatial", "simple_delta", "simple_calendar")
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a complete synthetic experiment: data generation, training, "
            "prediction and perturbation analyses."
        ),
    )
    parser.add_argument(
        "experiment",
        choices=SUPPORTED_EXPERIMENTS,
        help="Synthetic experiment to run.",
    )
    return parser.parse_args()


def experiment_paths(experiment: str) -> dict[tuple[str, str], Path]:
    with CONFIG_PATH.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    base_output = Path(raw_config["synthetic"][experiment]["output"])
    suffix = base_output.suffix or ".csv"
    train_path = base_output.with_name(f"{base_output.stem}_train{suffix}")
    pred_input_path = base_output.with_name(f"{base_output.stem}_pred{suffix}")

    return {
        ("model.hier_transformer.paths", "train_path"): train_path,
        ("model.hier_transformer.paths", "pred_input_path"): pred_input_path,
        ("model.hier_transformer.paths", "pred_output_path"): (
            Path("data/output") / f"pred_{experiment}_embeddings.csv"
        ),
        ("model.hier_transformer.perturbation", "output_path"): (
            Path("model_artifacts") / experiment / "perturbation.csv"
        ),
    }


def rewrite_config_paths(updates: dict[tuple[str, str], Path]) -> None:
    section: str | None = None
    replaced: set[tuple[str, str]] = set()
    rewritten_lines: list[str] = []

    for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines(keepends=True):
        line_body = line.rstrip("\r\n")
        line_ending = line[len(line_body):]

        section_match = re.match(r"^\s*\[([^\]]+)\]\s*$", line_body)
        if section_match:
            section = section_match.group(1)

        key_match = re.match(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*).*$", line_body)
        if key_match and section is not None:
            indent, key, separator = key_match.groups()
            update_key = (section, key)
            if update_key in updates:
                value = updates[update_key].as_posix()
                line = f'{indent}{key}{separator}"{value}"{line_ending}'
                replaced.add(update_key)

        rewritten_lines.append(line)

    missing = set(updates) - replaced
    if missing:
        missing_keys = ", ".join(f"{section}.{key}" for section, key in sorted(missing))
        raise KeyError(f"Missing config keys: {missing_keys}")

    CONFIG_PATH.write_text("".join(rewritten_lines), encoding="utf-8")


def run_step(label: str, command: list[str]) -> None:
    print(f"==> {label}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    updates = experiment_paths(args.experiment)
    original_config = CONFIG_PATH.read_text(encoding="utf-8")

    try:
        print(f"==> Configuring paths for {args.experiment}", flush=True)
        rewrite_config_paths(updates)
        for (_, key), value in updates.items():
            print(f"{key}={value.as_posix()}", flush=True)

        run_step(
            f"Generating {args.experiment} synthetic dataset",
            ["uv", "run", "py", "synthetic", "--type", args.experiment],
        )
        run_step("Training hierarchical model", ["uv", "run", "py", "train", "--type", "hier"])
        run_step("Predicting embeddings", ["uv", "run", "py", "pred", "--type", "hier"])
        #run_step(
        #    "Running sensibility perturbation analysis",
        #    ["uv", "run", "py", "perturbation", "--type", "hier", "--analysis", "sensibility"],
        #)
        run_step(
            "Running classification perturbation analysis",
            ["uv", "run", "py", "perturbation", "--type", "hier", "--analysis", "classification"],
        )
        print(f"==> {args.experiment} experiment completed", flush=True)
        return 0
    finally:
        CONFIG_PATH.write_text(original_config, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
