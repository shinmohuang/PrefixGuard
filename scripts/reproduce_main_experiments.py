from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/main_experiments.json")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the public main monitor reproduction pipeline: prepare datasets, "
            "train configured seeds, evaluate locked test splits, and summarize metrics."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=["prepare", "verify", "train", "eval", "summarize", "all"],
        default="all",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Subset of dataset families to run. Defaults to all families in the manifest.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional seed subset, applied to every selected run.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override manifest epochs. Use --epochs 1 for a smoke run.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _selected_runs(config: dict[str, Any], families: list[str] | None) -> list[dict[str, Any]]:
    runs = list(config["runs"])
    if families is None:
        return runs
    requested = set(families)
    available = set(config["datasets"])
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"Unknown families: {unknown}. Available: {sorted(available)}")
    return [run for run in runs if run["family"] in requested]


def _selected_seeds(run: dict[str, Any], seeds: list[int] | None) -> list[int]:
    run_seeds = [int(seed) for seed in run["seeds"]]
    if seeds is None:
        return run_seeds
    requested = set(int(seed) for seed in seeds)
    return [seed for seed in run_seeds if seed in requested]


def _append_common_train_args(
    command: list[str],
    *,
    run: dict[str, Any],
    seed: int,
    dataset: str,
    output_dir: str,
    device: str,
    epochs: int | None,
) -> list[str]:
    manifest_epochs = int(run["epochs"] if epochs is None else epochs)
    return [
        *command,
        "--dataset",
        dataset,
        "--output-dir",
        output_dir,
        *run["train_args"],
        "--num-symbols",
        str(run["num_symbols"]),
        "--q-max",
        str(run["q_max"]),
        "--epochs",
        str(manifest_epochs),
        "--horizon",
        str(run["horizon"]),
        "--seed",
        str(seed),
        "--device",
        device,
    ]


def build_commands(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    selected_runs = _selected_runs(config, args.families)
    selected_families = sorted({run["family"] for run in selected_runs})
    commands: list[dict[str, Any]] = []

    if args.stage in {"prepare", "all"}:
        for family in selected_families:
            dataset = config["datasets"][family]
            commands.append(
                {
                    "stage": "prepare",
                    "family": family,
                    "command": [
                        args.python,
                        "scripts/prepare_source_raw_baseline_datasets.py",
                        "--only",
                        dataset["prepare_only"],
                    ],
                }
            )

    if args.stage in {"verify", "all"}:
        commands.append(
            {
                "stage": "verify",
                "family": ",".join(selected_families),
                "command": [
                    args.python,
                    "scripts/verify_dataset_artifacts.py",
                    "--families",
                    *selected_families,
                    "--json",
                ],
            }
        )

    if args.stage in {"train", "all"}:
        for run in selected_runs:
            dataset = config["datasets"][run["family"]]["artifact"]
            for seed in _selected_seeds(run, args.seeds):
                output_dir = run["output_template"].format(seed=seed)
                if args.skip_existing and (Path(output_dir) / "best_checkpoint.pt").exists():
                    continue
                commands.append(
                    {
                        "stage": "train",
                        "family": run["family"],
                        "run_id": run["id"],
                        "seed": seed,
                        "command": _append_common_train_args(
                            [args.python, "scripts/train_differentiable_automaton.py"],
                            run=run,
                            seed=seed,
                            dataset=dataset,
                            output_dir=output_dir,
                            device=args.device,
                            epochs=args.epochs,
                        ),
                    }
                )

    if args.stage in {"eval", "all"}:
        for run in selected_runs:
            dataset = config["datasets"][run["family"]]["artifact"]
            for seed in _selected_seeds(run, args.seeds):
                output_dir = run["output_template"].format(seed=seed)
                eval_dir = f"{output_dir}_test_locked"
                metrics_path = Path(eval_dir) / "best_metrics.json"
                if args.skip_existing and metrics_path.exists():
                    continue
                commands.append(
                    {
                        "stage": "eval",
                        "family": run["family"],
                        "run_id": run["id"],
                        "seed": seed,
                        "command": [
                            args.python,
                            "scripts/evaluate_differentiable_automaton.py",
                            "--dataset",
                            dataset,
                            "--checkpoint",
                            f"{output_dir}/best_checkpoint.pt",
                            "--eval-split",
                            "test",
                            "--device",
                            args.device,
                            *run["eval_args"],
                            "--output",
                            str(metrics_path),
                        ],
                    }
                )

    if args.stage in {"summarize", "all"}:
        commands.append(
            {
                "stage": "summarize",
                "family": ",".join(selected_families),
                "command": [
                    args.python,
                    "scripts/summarize_main_experiments.py",
                    "--config",
                    str(args.config),
                ],
            }
        )
    return commands


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    commands = build_commands(args, config)
    if args.dry_run:
        print(json.dumps({"commands": commands}, indent=2, sort_keys=True))
        return
    for item in commands:
        print(json.dumps(item, sort_keys=True), flush=True)
        subprocess.run(item["command"], check=True)


if __name__ == "__main__":
    main()
