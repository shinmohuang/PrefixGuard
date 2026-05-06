from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from monitor_symbolization.data.io import load_trajectories
from monitor_symbolization.experiment_paths import sanity_output_dir
from monitor_symbolization.data.prefixes import summarize_prefix_dataset
from monitor_symbolization.models.warning_head_names import WARNING_MODEL_CHOICES
from monitor_symbolization.training import (
    DifferentiableTrainingConfig,
    train_differentiable_automaton,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the M0 differentiable automaton sanity experiment on toy or subsetted data."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/toy/trajectories.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ["DIFF_AUTOMATON_OUTPUT_DIR"])
        if "DIFF_AUTOMATON_OUTPUT_DIR" in os.environ
        else sanity_output_dir(),
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--num-symbols", type=int, default=8)
    parser.add_argument("--q-max", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument(
        "--warning-model-type",
        choices=WARNING_MODEL_CHOICES,
        default=DifferentiableTrainingConfig().warning_model_type,
    )
    parser.add_argument(
        "--selection-metric",
        choices=["discrete-trusted", "direct-soft"],
        default="direct-soft",
    )
    parser.add_argument(
        "--step-view-frontend",
        choices=["inferred", "webarena", "tau2bench", "skillsbench"],
        default="inferred",
    )
    parser.add_argument(
        "--step-view-text-mode",
        choices=[
            "full",
            "transfer-full",
            "drop-tool",
            "drop-status",
            "drop-args",
            "drop-result",
            "drop-args-result",
            "observation-only",
        ],
        default="full",
    )
    return parser.parse_args()


def _limit_split(trajectories, subset_size: int):
    if subset_size <= 0:
        return trajectories
    kept = []
    split_counts: dict[str, int] = {}
    for trajectory in trajectories:
        count = split_counts.get(trajectory.split, 0)
        if count >= subset_size:
            continue
        kept.append(trajectory)
        split_counts[trajectory.split] = count + 1
    return kept


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = _limit_split(load_trajectories(args.dataset), args.subset_size)
    train_trajectories = [trajectory for trajectory in trajectories if trajectory.split == "train"]
    val_trajectories = [trajectory for trajectory in trajectories if trajectory.split == "val"]
    if not train_trajectories or not val_trajectories:
        raise ValueError("Sanity run requires at least train and val trajectories")

    dataset_summary = summarize_prefix_dataset(trajectories, horizon=args.horizon)
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = train_differentiable_automaton(
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        output_dir=args.output_dir / "differentiable_automaton",
        config=DifferentiableTrainingConfig(
            warning_model_type=args.warning_model_type,
            selection_metric=args.selection_metric,
            encoder_type="tfidf",
            step_view_frontend=args.step_view_frontend,
            step_view_text_mode=args.step_view_text_mode,
            epochs=args.epochs,
            num_symbols=args.num_symbols,
            q_max=args.q_max,
            horizon=args.horizon,
            batch_size=2,
            hidden_dim=64,
            symbol_embedding_dim=32,
            device="cpu",
            dfa_backend="aalpy-rpni",
            derive_train_fit_cal=False,
            fit_split="train",
            cal_split="train",
        ),
    )
    payload = {
        "dataset": str(args.dataset),
        "subset_size": args.subset_size,
        "dataset_summary": dataset_summary,
        "training_result": {
            "best_epoch": result.best_epoch,
            "checkpoint_path": result.checkpoint_path,
            "best_metrics": result.best_metrics,
        },
        "warning_model_type": args.warning_model_type,
        "selection_metric": args.selection_metric,
        "step_view_frontend": args.step_view_frontend,
        "step_view_text_mode": args.step_view_text_mode,
    }
    (args.output_dir / "sanity_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
