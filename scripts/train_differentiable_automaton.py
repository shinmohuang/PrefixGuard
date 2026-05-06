from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from monitor_symbolization.data.io import (
    load_trajectories,
    resolve_fit_cal_splits,
    trajectories_for_split,
)
from monitor_symbolization.data.prefixes import summarize_prefix_dataset
from monitor_symbolization.monitor.hardening import available_hardening_strategies
from monitor_symbolization.models.warning_head_names import WARNING_MODEL_CHOICES
from monitor_symbolization.training import (
    DifferentiableTrainingConfig,
    train_differentiable_automaton,
)


PUBLIC_TOY_DATASET = Path("data/toy/trajectories.jsonl")
PUBLIC_TRAINING_OUTPUT_DIR = Path("outputs/training_public/differentiable_automaton")


def parse_args() -> argparse.Namespace:
    defaults = DifferentiableTrainingConfig()
    parser = argparse.ArgumentParser(
        description="Train the differentiable finite-state monitor on a split trajectory JSONL."
    )
    parser.add_argument("--dataset", type=Path, default=PUBLIC_TOY_DATASET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ["DIFF_AUTOMATON_OUTPUT_DIR"])
        if "DIFF_AUTOMATON_OUTPUT_DIR" in os.environ
        else PUBLIC_TRAINING_OUTPUT_DIR,
    )
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--warning-model-type",
        choices=WARNING_MODEL_CHOICES,
        default=defaults.warning_model_type,
    )
    parser.add_argument(
        "--selection-metric",
        choices=["discrete-trusted", "direct-soft"],
        default="direct-soft",
    )
    parser.add_argument(
        "--encoder-type",
        choices=["transformer", "tfidf", "hybrid"],
        default="tfidf",
    )
    parser.add_argument(
        "--representation-mode",
        choices=["legacy", "source-raw", "reduced-dense", "hybrid"],
        default=defaults.representation_mode,
    )
    parser.add_argument(
        "--step-view-frontend",
        choices=["inferred", "webarena", "tau2bench", "skillsbench", "terminalbench"],
        default=defaults.step_view_frontend,
    )
    parser.add_argument(
        "--tau2-refinement-profile",
        choices=[
            "none",
            "collecting-subtypes-v1",
            "collecting-subtypes-v2",
            "collecting-subtypes-v3",
            "semantic-tool-role-v1",
            "semantic-tool-role-obligation-v1",
            "semantic-tool-role-obligation-state-v1",
            "semantic-full-v1",
        ],
        default=defaults.tau2_refinement_profile or "none",
    )
    parser.add_argument(
        "--skillsbench-process-profile",
        choices=["none", "phase-v1", "process-full-v1"],
        default=defaults.skillsbench_process_profile or "none",
    )
    parser.add_argument(
        "--step-view-text-mode",
        choices=[
            "full",
            "transfer-full",
            "lexical",
            "drop-tool",
            "drop-status",
            "drop-args",
            "drop-result",
            "drop-args-result",
            "observation-only",
        ],
        default=defaults.step_view_text_mode,
    )
    parser.add_argument(
        "--transformer-stepview-view",
        choices=["dense", "lexical", "transfer-full-lexical", "fieldwise", "grouped-fieldwise"],
        default=defaults.transformer_stepview_view,
    )
    parser.add_argument("--encoder-name", type=str, default=defaults.encoder_name)
    parser.add_argument("--encoder-max-length", type=int, default=defaults.encoder_max_length)
    parser.add_argument("--encoder-batch-size", type=int, default=defaults.encoder_batch_size)
    parser.add_argument("--encoder-max-features", type=int, default=defaults.encoder_max_features)
    parser.add_argument(
        "--tfidf-metadata-sidechannel",
        choices=[
            "none",
            "tau2-semantic-v1",
            "tau2-semantic-v2",
            "skillsbench-process-v1",
            "skillsbench-task-v1",
            "skillsbench-exec-v2",
            "terminalbench-task-v1",
            "terminalbench-meta-v1",
        ],
        default=defaults.tfidf_metadata_sidechannel or "none",
    )
    parser.add_argument(
        "--tfidf-metadata-sidechannel-scale",
        type=float,
        default=defaults.tfidf_metadata_sidechannel_scale,
    )
    parser.add_argument(
        "--tfidf-sparse-runtime",
        action=argparse.BooleanOptionalAction,
        default=defaults.tfidf_sparse_runtime,
    )
    parser.add_argument("--max-observation-lines", type=int, default=defaults.max_observation_lines)
    parser.add_argument(
        "--fine-tune-encoder",
        action=argparse.BooleanOptionalAction,
        default=defaults.fine_tune_encoder,
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--num-symbols", type=int, default=8)
    parser.add_argument("--q-max", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--fit-split", type=str, default="train")
    parser.add_argument("--cal-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument(
        "--derive-train-fit-cal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Derive train-internal fit/cal splits from --fit-split. Disabled by "
            "default so the included toy dataset can train without resampling."
        ),
    )
    parser.add_argument("--train-fit-ratio", type=float, default=defaults.train_fit_ratio)
    parser.add_argument("--train-cal-ratio", type=float, default=defaults.train_cal_ratio)
    parser.add_argument("--protocol-split-seed", type=int, default=defaults.protocol_split_seed)
    parser.add_argument(
        "--legacy-reproduction",
        action=argparse.BooleanOptionalAction,
        default=defaults.legacy_reproduction,
    )
    parser.add_argument(
        "--final-best-paired-validation",
        action=argparse.BooleanOptionalAction,
        default=defaults.final_best_paired_validation,
    )
    parser.add_argument(
        "--skip-final-paired-validation",
        action=argparse.BooleanOptionalAction,
        default=defaults.skip_final_paired_validation,
    )
    parser.add_argument("--trusted-state-min-count", type=int, default=defaults.trusted_state_min_count)
    parser.add_argument(
        "--state-risk-smoothing-alpha",
        type=float,
        default=defaults.state_risk_smoothing_alpha,
    )
    parser.add_argument("--calibration-bins", type=int, default=defaults.calibration_bins)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--symbol-embedding-dim", type=int, default=32)
    parser.add_argument("--tau-sym-start", type=float, default=defaults.tau_sym_start)
    parser.add_argument("--tau-sym-end", type=float, default=defaults.tau_sym_end)
    parser.add_argument("--tau-trans-start", type=float, default=defaults.tau_trans_start)
    parser.add_argument("--tau-trans-end", type=float, default=defaults.tau_trans_end)
    parser.add_argument("--anneal-scheme", choices=["linear", "cosine"], default=defaults.anneal_scheme)
    parser.add_argument("--lambda-prefix", type=float, default=defaults.lambda_prefix)
    parser.add_argument("--lambda-balance", type=float, default=defaults.lambda_balance)
    parser.add_argument(
        "--batched-differentiable-runtime",
        action=argparse.BooleanOptionalAction,
        default=defaults.batched_differentiable_runtime,
    )
    parser.add_argument(
        "--length-bucketed-batches",
        action=argparse.BooleanOptionalAction,
        default=defaults.length_bucketed_batches,
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dfa-backend",
        choices=["legacy", "aalpy", "aalpy-edsm", "aalpy-rpni"],
        default="aalpy-rpni",
    )
    parser.add_argument(
        "--hardening-strategy",
        choices=available_hardening_strategies(),
        default=defaults.hardening_strategy,
    )
    parser.add_argument("--hardening-threshold", type=float, default=defaults.hardening_threshold)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = load_trajectories(args.dataset)
    train_trajectories, cal_trajectories, protocol_split_summary = resolve_fit_cal_splits(
        trajectories,
        fit_split=args.fit_split,
        cal_split=args.cal_split,
        derive_train_fit_cal=args.derive_train_fit_cal and not args.legacy_reproduction,
        train_fit_ratio=args.train_fit_ratio,
        train_cal_ratio=args.train_cal_ratio,
        protocol_split_seed=args.protocol_split_seed,
    )
    val_trajectories = trajectories_for_split(trajectories, args.val_split, role="val split")
    dataset_summary = summarize_prefix_dataset(trajectories, horizon=args.horizon)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "input_dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "protocol_split_summary.json").write_text(
        json.dumps(protocol_split_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "dataset_summary": dataset_summary,
                "fit_split_protocol": protocol_split_summary,
                "val_split": args.val_split,
                "legacy_reproduction": args.legacy_reproduction,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    result = train_differentiable_automaton(
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        output_dir=args.output_dir,
        config=DifferentiableTrainingConfig(
            warning_model_type=args.warning_model_type,
            selection_metric=args.selection_metric,
            encoder_type=args.encoder_type,
            representation_mode=args.representation_mode,
            step_view_frontend=args.step_view_frontend,
            tau2_refinement_profile=None
            if args.tau2_refinement_profile == "none"
            else args.tau2_refinement_profile,
            skillsbench_process_profile=None
            if args.skillsbench_process_profile == "none"
            else args.skillsbench_process_profile,
            step_view_text_mode=args.step_view_text_mode,
            transformer_stepview_view=args.transformer_stepview_view,
            encoder_name=args.encoder_name,
            fine_tune_encoder=args.fine_tune_encoder,
            encoder_max_length=args.encoder_max_length,
            encoder_batch_size=args.encoder_batch_size,
            encoder_max_features=args.encoder_max_features,
            tfidf_metadata_sidechannel=None
            if args.tfidf_metadata_sidechannel == "none"
            else args.tfidf_metadata_sidechannel,
            tfidf_metadata_sidechannel_scale=args.tfidf_metadata_sidechannel_scale,
            tfidf_sparse_runtime=args.tfidf_sparse_runtime,
            max_observation_lines=args.max_observation_lines,
            hidden_dim=args.hidden_dim,
            symbol_embedding_dim=args.symbol_embedding_dim,
            epochs=args.epochs,
            num_symbols=args.num_symbols,
            q_max=args.q_max,
            horizon=args.horizon,
            seed=args.seed,
            trusted_state_min_count=args.trusted_state_min_count,
            state_risk_smoothing_alpha=args.state_risk_smoothing_alpha,
            calibration_bins=args.calibration_bins,
            fit_split=args.fit_split,
            cal_split=args.cal_split,
            val_split=args.val_split,
            derive_train_fit_cal=args.derive_train_fit_cal and not args.legacy_reproduction,
            train_fit_ratio=args.train_fit_ratio,
            train_cal_ratio=args.train_cal_ratio,
            protocol_split_seed=args.protocol_split_seed,
            legacy_reproduction=args.legacy_reproduction,
            final_best_paired_validation=args.final_best_paired_validation,
            skip_final_paired_validation=args.skip_final_paired_validation,
            batch_size=args.batch_size,
            tau_sym_start=args.tau_sym_start,
            tau_sym_end=args.tau_sym_end,
            tau_trans_start=args.tau_trans_start,
            tau_trans_end=args.tau_trans_end,
            anneal_scheme=args.anneal_scheme,
            lambda_prefix=args.lambda_prefix,
            lambda_balance=args.lambda_balance,
            batched_differentiable_runtime=args.batched_differentiable_runtime,
            length_bucketed_batches=args.length_bucketed_batches,
            device=args.device,
            dfa_backend=args.dfa_backend,
            hardening_strategy=args.hardening_strategy,
            hardening_threshold=args.hardening_threshold,
        ),
        cal_trajectories=cal_trajectories,
        resume_from=args.resume_from,
    )
    payload = {
        "dataset": str(args.dataset),
        "best_epoch": result.best_epoch,
        "checkpoint_path": result.checkpoint_path,
        "best_metrics": result.best_metrics,
        "output_dir": result.output_dir,
    }
    (Path(result.output_dir) / "train_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
