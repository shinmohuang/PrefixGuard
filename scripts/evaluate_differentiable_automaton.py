from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from monitor_symbolization.data.io import (
    load_trajectories,
    resolve_fit_cal_splits,
    trajectories_for_split,
)
from monitor_symbolization.data.serialization import build_step_payload
from monitor_symbolization.models.differentiable_automaton import (
    DifferentiableFiniteStateSurrogate,
    FlatPrefixRiskHead,
    GruPrefixRiskHead,
    TransformerPrefixRiskHead,
)
from monitor_symbolization.models.encoders import (
    DEFAULT_TRANSFORMER_MAX_LENGTH,
    DEFAULT_TRANSFORMER_MODEL,
    HybridStepEncoder,
    TfidfSegmentEncoder,
    TransformerSegmentEncoder,
)
from monitor_symbolization.models.symbolizer import GumbelEventSymbolizer
from monitor_symbolization.models.warning_head_names import normalize_warning_model_type
from monitor_symbolization.monitor.evaluation import (
    evaluate_paired_differentiable_monitor,
    evaluate_soft_differentiable_monitor,
)
from monitor_symbolization.monitor.hardening import available_hardening_strategies


PUBLIC_TOY_DATASET = Path("data/toy/trajectories.jsonl")
PUBLIC_EVALUATION_OUTPUT = Path("outputs/evaluation_public/differentiable_automaton_test.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained differentiable automaton checkpoint."
    )
    parser.add_argument("--dataset", type=Path, default=PUBLIC_TOY_DATASET)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-type", choices=["transformer", "tfidf", "hybrid"], default=None)
    parser.add_argument(
        "--representation-mode",
        choices=["legacy", "source-raw", "reduced-dense", "hybrid"],
        default=None,
    )
    parser.add_argument(
        "--step-view-frontend",
        choices=["inferred", "webarena", "tau2bench", "skillsbench", "terminalbench"],
        default=None,
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
        default=None,
    )
    parser.add_argument(
        "--skillsbench-process-profile",
        choices=["none", "phase-v1", "process-full-v1"],
        default=None,
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
        default=None,
    )
    parser.add_argument(
        "--transformer-stepview-view",
        choices=["dense", "lexical", "transfer-full-lexical", "fieldwise", "grouped-fieldwise"],
        default=None,
    )
    parser.add_argument("--encoder-name", type=str, default=None)
    parser.add_argument("--encoder-max-length", type=int, default=None)
    parser.add_argument("--encoder-batch-size", type=int, default=None)
    parser.add_argument("--encoder-max-features", type=int, default=None)
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
        default=None,
    )
    parser.add_argument("--tfidf-metadata-sidechannel-scale", type=float, default=None)
    parser.add_argument("--max-observation-lines", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ["DIFF_AUTOMATON_OUTPUT_DIR"]) / "paired_eval.json"
        if "DIFF_AUTOMATON_OUTPUT_DIR" in os.environ
        else PUBLIC_EVALUATION_OUTPUT,
    )
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--fit-split", type=str, default=None)
    parser.add_argument("--cal-split", type=str, default=None)
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--derive-train-fit-cal",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--train-fit-ratio", type=float, default=None)
    parser.add_argument("--train-cal-ratio", type=float, default=None)
    parser.add_argument("--protocol-split-seed", type=int, default=None)
    parser.add_argument(
        "--legacy-reproduction",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--trusted-state-min-count", type=int, default=10)
    parser.add_argument("--state-risk-smoothing-alpha", type=float, default=5.0)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--soft-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate only the learned soft monitor and skip discrete DFA induction.",
    )
    parser.add_argument(
        "--dfa-backend",
        choices=["legacy", "aalpy", "aalpy-edsm", "aalpy-rpni"],
        default="aalpy-rpni",
    )
    parser.add_argument(
        "--hardening-strategy",
        choices=available_hardening_strategies(),
        default=None,
    )
    parser.add_argument("--hardening-threshold", type=float, default=None)
    return parser.parse_args()


def _build_encoder(checkpoint: dict, args: argparse.Namespace, train_trajectories):
    checkpoint_config = checkpoint["config"]
    encoder_type = args.encoder_type or checkpoint_config.get("encoder_type", "tfidf")
    representation_mode = args.representation_mode or checkpoint_config.get(
        "representation_mode",
        "reduced-dense",
    )
    step_view_frontend = args.step_view_frontend or checkpoint_config.get(
        "step_view_frontend",
        "inferred",
    )
    tau2_refinement_profile = args.tau2_refinement_profile
    if tau2_refinement_profile is None:
        tau2_refinement_profile = checkpoint_config.get("tau2_refinement_profile")
    if tau2_refinement_profile == "none":
        tau2_refinement_profile = None
    skillsbench_process_profile = args.skillsbench_process_profile
    if skillsbench_process_profile is None:
        skillsbench_process_profile = checkpoint_config.get("skillsbench_process_profile")
    if skillsbench_process_profile == "none":
        skillsbench_process_profile = None
    step_view_text_mode = args.step_view_text_mode or checkpoint_config.get(
        "step_view_text_mode",
        "full",
    )
    transformer_stepview_view = args.transformer_stepview_view or checkpoint_config.get(
        "transformer_stepview_view",
        "dense",
    )
    encoder_name = args.encoder_name or checkpoint_config.get(
        "encoder_name",
        DEFAULT_TRANSFORMER_MODEL,
    )
    encoder_max_length = args.encoder_max_length or checkpoint_config.get(
        "encoder_max_length",
        DEFAULT_TRANSFORMER_MAX_LENGTH,
    )
    encoder_batch_size = args.encoder_batch_size or checkpoint_config.get("encoder_batch_size", 1)
    encoder_max_features = args.encoder_max_features or checkpoint_config.get(
        "encoder_max_features",
        4096,
    )
    tfidf_metadata_sidechannel = args.tfidf_metadata_sidechannel
    if tfidf_metadata_sidechannel is None:
        tfidf_metadata_sidechannel = checkpoint_config.get("tfidf_metadata_sidechannel")
    if tfidf_metadata_sidechannel == "none":
        tfidf_metadata_sidechannel = None
    tfidf_metadata_sidechannel_scale = args.tfidf_metadata_sidechannel_scale
    if tfidf_metadata_sidechannel_scale is None:
        tfidf_metadata_sidechannel_scale = checkpoint_config.get(
            "tfidf_metadata_sidechannel_scale",
            1.0,
        )
    tfidf_sparse_runtime = checkpoint_config.get("tfidf_sparse_runtime", False)
    max_observation_lines = args.max_observation_lines or checkpoint_config.get(
        "max_observation_lines",
        8,
    )

    if encoder_type == "transformer":
        encoder = TransformerSegmentEncoder(
            model_name=encoder_name,
            fine_tune=False,
            batch_size=encoder_batch_size,
            max_length=encoder_max_length,
            step_view_text_mode=transformer_stepview_view,
        )
        return (
            encoder,
            encoder_type,
            representation_mode,
            max_observation_lines,
            step_view_frontend,
            tau2_refinement_profile,
            skillsbench_process_profile,
            step_view_text_mode,
            transformer_stepview_view,
            tfidf_metadata_sidechannel,
            tfidf_metadata_sidechannel_scale,
        )

    if encoder_type == "hybrid":
        encoder = HybridStepEncoder(
            model_name=encoder_name,
            lexical_max_features=encoder_max_features,
            step_view_text_mode=step_view_text_mode,
            lexical_metadata_sidechannel_mode=tfidf_metadata_sidechannel,
            lexical_metadata_sidechannel_scale=tfidf_metadata_sidechannel_scale,
            transformer_step_view_mode=transformer_stepview_view,
            fine_tune=False,
            batch_size=encoder_batch_size,
            max_length=encoder_max_length,
        )
        encoder.fit(
            build_step_payload(
                step,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                dataset_name=step_view_frontend,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
            )
            for trajectory in train_trajectories
            for step in trajectory.steps
        )
        return (
            encoder,
            encoder_type,
            representation_mode,
            max_observation_lines,
            step_view_frontend,
            tau2_refinement_profile,
            skillsbench_process_profile,
            step_view_text_mode,
            transformer_stepview_view,
            tfidf_metadata_sidechannel,
            tfidf_metadata_sidechannel_scale,
        )

    encoder = TfidfSegmentEncoder(
        max_features=encoder_max_features,
        step_view_text_mode=step_view_text_mode,
        sparse_output=tfidf_sparse_runtime,
        metadata_sidechannel_mode=tfidf_metadata_sidechannel,
        metadata_sidechannel_scale=tfidf_metadata_sidechannel_scale,
    )
    artifact_state = checkpoint.get("encoder_artifact_state")
    if artifact_state is not None:
        encoder.load_artifact_state(artifact_state)
    else:
        encoder.fit(
            build_step_payload(
                step,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                dataset_name=step_view_frontend,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
            )
            for trajectory in train_trajectories
            for step in trajectory.steps
        )
    return (
        encoder,
        encoder_type,
        representation_mode,
        max_observation_lines,
        step_view_frontend,
        tau2_refinement_profile,
        skillsbench_process_profile,
        step_view_text_mode,
        transformer_stepview_view,
        tfidf_metadata_sidechannel,
        tfidf_metadata_sidechannel_scale,
    )


def _build_warning_model(config: dict):
    warning_model_type = normalize_warning_model_type(
        config.get("warning_model_type", "soft-fsm")
    )
    if warning_model_type == "soft-fsm":
        return DifferentiableFiniteStateSurrogate(
            num_symbols=config["num_symbols"],
            num_states=config["q_max"],
        )
    if warning_model_type == "symbol-flat":
        return FlatPrefixRiskHead(
            num_symbols=config["num_symbols"],
            num_states=config["q_max"],
        )
    if warning_model_type == "symbol-gru":
        return GruPrefixRiskHead(
            num_symbols=config["num_symbols"],
            num_states=config["q_max"],
        )
    if warning_model_type == "symbol-transformer":
        return TransformerPrefixRiskHead(
            num_symbols=config["num_symbols"],
            num_states=config["q_max"],
        )
    raise ValueError(f"Unsupported warning_model_type: {warning_model_type}")


def _soft_only_payload(
    *,
    soft_output: dict,
    automaton,
    config: dict,
    args: argparse.Namespace,
    hardening_strategy: str,
    hardening_threshold: float | None,
) -> dict:
    soft_metrics = soft_output["metrics"]
    validation_mode = (
        "soft-only-locked-test" if args.eval_split == "test" else "soft-only-evaluation"
    )
    return {
        "method": getattr(automaton, "method_name", "differentiable_automaton"),
        "validation_mode": validation_mode,
        "dfa_backend": "not-run",
        "hardening_strategy": str(hardening_strategy),
        "hardening_threshold": (
            None if hardening_threshold is None else float(hardening_threshold)
        ),
        "num_symbols": int(config["num_symbols"]),
        "legacy_reproduction": bool(args.legacy_reproduction),
        "soft_metrics": soft_metrics,
        "summary": {
            "method": getattr(automaton, "method_name", "differentiable_automaton"),
            "validation_mode": validation_mode,
            "soft_auroc": float(soft_metrics["auroc"]),
            "soft_auprc": float(soft_metrics["auprc"]),
            "soft_detection_latency": float(soft_metrics["detection_latency"]),
            "soft_alert_lead_time": float(soft_metrics["alert_lead_time"]),
            "soft_threshold": float(soft_metrics["threshold"]),
            "soft_state_count": int(soft_metrics["soft_state_count"]),
            "soft_calibration_error": float(soft_metrics["calibration_error"]),
            "soft_brier_score": float(soft_metrics["brier_score"]),
            "num_symbols": int(config["num_symbols"]),
            "dfa_backend": "not-run",
        },
    }


def evaluate_checkpoint_with_loaded_trajectories(
    args: argparse.Namespace,
    trajectories,
) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    fit_split = args.fit_split or config.get("fit_split", "train")
    cal_split = args.cal_split or config.get("cal_split", fit_split)
    derive_train_fit_cal = (
        args.derive_train_fit_cal
        if args.derive_train_fit_cal is not None
        else bool(config.get("derive_train_fit_cal", False))
    )
    train_fit_ratio = (
        args.train_fit_ratio
        if args.train_fit_ratio is not None
        else float(config.get("train_fit_ratio", 0.8))
    )
    train_cal_ratio = (
        args.train_cal_ratio
        if args.train_cal_ratio is not None
        else float(config.get("train_cal_ratio", 0.2))
    )
    protocol_split_seed = (
        args.protocol_split_seed
        if args.protocol_split_seed is not None
        else int(config.get("protocol_split_seed", 1))
    )
    hardening_strategy = args.hardening_strategy or config.get("hardening_strategy", "argmax")
    hardening_threshold = (
        args.hardening_threshold
        if args.hardening_threshold is not None
        else config.get("hardening_threshold")
    )
    train_trajectories, cal_trajectories, protocol_split_summary = resolve_fit_cal_splits(
        trajectories,
        fit_split=fit_split,
        cal_split=cal_split,
        derive_train_fit_cal=derive_train_fit_cal and not args.legacy_reproduction,
        train_fit_ratio=train_fit_ratio,
        train_cal_ratio=train_cal_ratio,
        protocol_split_seed=protocol_split_seed,
    )
    eval_trajectories = trajectories_for_split(trajectories, args.eval_split, role="eval split")
    print(
        json.dumps(
            {
                "fit_split_protocol": protocol_split_summary,
                "eval_split": args.eval_split,
                "legacy_reproduction": args.legacy_reproduction,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    (
        encoder,
        encoder_type,
        representation_mode,
        max_observation_lines,
        step_view_frontend,
        tau2_refinement_profile,
        skillsbench_process_profile,
        step_view_text_mode,
        transformer_stepview_view,
        tfidf_metadata_sidechannel,
        tfidf_metadata_sidechannel_scale,
    ) = _build_encoder(
        checkpoint,
        args,
        train_trajectories,
    )
    encoder_state_dict = checkpoint.get("encoder_state_dict")
    if encoder_type in {"transformer", "hybrid"} and encoder_state_dict is not None:
        encoder.load_state_dict(encoder_state_dict)

    symbolizer = GumbelEventSymbolizer(
        input_dim=encoder.output_dim,
        hidden_dim=config["hidden_dim"],
        num_symbols=config["num_symbols"],
        symbol_embedding_dim=config["symbol_embedding_dim"],
    )
    symbolizer.load_state_dict(checkpoint["symbolizer_state_dict"])
    automaton = _build_warning_model(config)
    automaton.load_state_dict(
        checkpoint.get("warning_model_state_dict", checkpoint["automaton_state_dict"])
    )

    device = torch.device(args.device)
    symbolizer = symbolizer.to(device)
    automaton = automaton.to(device)
    horizon = args.horizon or config.get("horizon", 3)
    symbol_temperature = float(checkpoint.get("tau_sym", config.get("tau_sym_end", 1.0)))
    transition_temperature = float(
        checkpoint.get("tau_trans", config.get("tau_trans_end", 1.0))
    )
    if args.soft_only:
        soft_output = evaluate_soft_differentiable_monitor(
            train_trajectories=train_trajectories,
            eval_trajectories=eval_trajectories,
            encoder=encoder,
            symbolizer=symbolizer,
            automaton=automaton,
            horizon=horizon,
            device=device,
            symbol_temperature=symbol_temperature,
            transition_temperature=transition_temperature,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        metrics = _soft_only_payload(
            soft_output=soft_output,
            automaton=automaton,
            config=config,
            args=args,
            hardening_strategy=hardening_strategy,
            hardening_threshold=hardening_threshold,
        )
    else:
        metrics = evaluate_paired_differentiable_monitor(
            train_trajectories=train_trajectories,
            eval_trajectories=eval_trajectories,
            cal_trajectories=cal_trajectories,
            encoder=encoder,
            symbolizer=symbolizer,
            automaton=automaton,
            horizon=horizon,
            num_symbols=config["num_symbols"],
            device=device,
            symbol_temperature=symbol_temperature,
            transition_temperature=transition_temperature,
            dfa_backend=args.dfa_backend,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            trusted_state_min_count=args.trusted_state_min_count,
            state_risk_smoothing_alpha=args.state_risk_smoothing_alpha,
            calibration_bins=args.calibration_bins,
            legacy_reproduction=args.legacy_reproduction,
            hardening_strategy=hardening_strategy,
            hardening_threshold=hardening_threshold,
            allow_discrete_failure=config.get("selection_metric") == "direct-soft",
        )
    metrics["dataset"] = str(args.dataset)
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["eval_split"] = args.eval_split
    metrics["fit_split_protocol"] = protocol_split_summary
    metrics["step_view_frontend"] = step_view_frontend
    metrics["tau2_refinement_profile"] = tau2_refinement_profile
    metrics["skillsbench_process_profile"] = skillsbench_process_profile
    metrics["tfidf_metadata_sidechannel"] = tfidf_metadata_sidechannel
    metrics["tfidf_metadata_sidechannel_scale"] = tfidf_metadata_sidechannel_scale
    metrics["representation_mode"] = representation_mode
    metrics["encoder_type"] = encoder_type
    metrics["step_view_text_mode"] = step_view_text_mode
    metrics["transformer_stepview_view"] = transformer_stepview_view
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": metrics.get("summary", {})}, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    args = parse_args()
    trajectories = load_trajectories(args.dataset)
    evaluate_checkpoint_with_loaded_trajectories(args, trajectories)


if __name__ == "__main__":
    main()
