from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re

import numpy as np
import torch

from monitor_symbolization.data.prefixes import build_prefix_dataset
from monitor_symbolization.data.schema import StepView, TrajectoryRecord
from monitor_symbolization.data.serialization import (
    build_step_payload,
    build_step_view,
    payload_to_text,
)
from monitor_symbolization.models.encoders import BaseSegmentEncoder
from monitor_symbolization.models.symbolizer import GumbelEventSymbolizer
from monitor_symbolization.monitor.backends import DfaBackendName
from monitor_symbolization.monitor.evaluation import (
    _evaluate_precomputed_symbol_sequences_with_details,
    symbolize_trajectories,
)
from monitor_symbolization.monitor.hardening import (
    DEFAULT_HARDENING_STRATEGY,
    HardeningStrategyName,
    harden_symbol_probabilities,
)

_TOKEN_RE = re.compile(r"[a-z0-9_:/.\-]+")


@dataclass(frozen=True)
class StepAssignment:
    trajectory_id: str
    step_index: int
    split: str
    symbol_id: int
    confidence: float
    margin: float
    payload_text: str
    step_view: StepView


def _lexical_document(step_view: StepView) -> str:
    parts = [
        step_view.tool_name,
        step_view.status,
        step_view.action_text,
        step_view.tool_args_text,
        step_view.result_text,
        step_view.metadata_text,
    ]
    return " ".join(part for part in parts if part and part != "NONE")


def _extract_ngrams(text: str, max_n: int = 2) -> list[str]:
    tokens = [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in {"none", "true", "false"}
    ]
    ngrams: list[str] = []
    for n in range(1, max_n + 1):
        if len(tokens) < n:
            continue
        for start in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[start : start + n]))
    return ngrams


def collect_step_assignments(
    trajectories: list[TrajectoryRecord],
    encoder: BaseSegmentEncoder,
    symbolizer: GumbelEventSymbolizer,
    *,
    device: torch.device,
    symbol_temperature: float = 1.0,
    hardening_strategy: HardeningStrategyName = DEFAULT_HARDENING_STRATEGY,
    hardening_threshold: float | None = None,
    representation_mode: str = "legacy",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    progress_label: str | None = None,
) -> list[StepAssignment]:
    payloads: list[str | StepView] = []
    step_views: list[StepView] = []
    refs: list[tuple[str, int, str]] = []

    for trajectory in trajectories:
        for step_index, step in enumerate(trajectory.steps, start=1):
            payload = build_step_payload(
                step,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                dataset_name=step_view_frontend,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
            )
            payloads.append(payload)
            step_views.append(
                build_step_view(
                    step,
                    max_observation_lines=max_observation_lines,
                    dataset_name=step_view_frontend,
                    tau2_refinement_profile=tau2_refinement_profile,
                    skillsbench_process_profile=skillsbench_process_profile,
                )
            )
            refs.append((trajectory.trajectory_id, step_index, trajectory.split))

    if not payloads:
        return []

    with torch.no_grad():
        embeddings = encoder.encode(
            payloads,
            device=device,
            progress_label=progress_label,
        ).embeddings
        outputs = symbolizer.deterministic_output(
            embeddings,
            temperature=symbol_temperature,
        )
        if hardening_strategy != DEFAULT_HARDENING_STRATEGY:
            hardening = harden_symbol_probabilities(
                outputs.probs.detach().cpu(),
                strategy=hardening_strategy,
                threshold=hardening_threshold,
            )
            hard_ids = hardening.hard_ids
        else:
            hard_ids = outputs.hard_ids.detach().cpu().tolist()
    probabilities = outputs.probs.detach().cpu().numpy()
    assignments: list[StepAssignment] = []
    for index, (trajectory_id, step_index, split) in enumerate(refs):
        row = probabilities[index]
        top = int(hard_ids[index])
        sorted_probs = np.sort(row)
        margin = (
            float(sorted_probs[-1] - sorted_probs[-2])
            if len(sorted_probs) > 1
            else float(sorted_probs[-1])
        )
        assignments.append(
            StepAssignment(
                trajectory_id=trajectory_id,
                step_index=step_index,
                split=split,
                symbol_id=top,
                confidence=float(row[top]),
                margin=margin,
                payload_text=payload_to_text(payloads[index]),
                step_view=step_views[index],
            )
        )
    return assignments


def summarize_symbols(
    assignments: list[StepAssignment],
    *,
    num_symbols: int,
    top_k_examples: int = 5,
    top_k_ngrams: int = 10,
) -> list[dict]:
    grouped: dict[int, list[StepAssignment]] = defaultdict(list)
    global_ngram_weights: Counter[str] = Counter()
    symbol_ngram_weights: dict[int, Counter[str]] = {
        symbol_id: Counter() for symbol_id in range(num_symbols)
    }

    for assignment in assignments:
        grouped[assignment.symbol_id].append(assignment)
        ngrams = _extract_ngrams(_lexical_document(assignment.step_view))
        for ngram in ngrams:
            global_ngram_weights[ngram] += assignment.confidence
            symbol_ngram_weights[assignment.symbol_id][ngram] += assignment.confidence

    total_assignments = len(assignments)
    symbol_summaries: list[dict] = []
    for symbol_id in range(num_symbols):
        symbol_assignments = sorted(
            grouped.get(symbol_id, []),
            key=lambda item: (item.confidence, item.margin),
            reverse=True,
        )
        seen_texts: set[str] = set()
        top_examples: list[dict] = []
        for assignment in symbol_assignments:
            lexical_text = assignment.step_view.lexical_text
            if lexical_text in seen_texts:
                continue
            seen_texts.add(lexical_text)
            top_examples.append(
                {
                    "trajectory_id": assignment.trajectory_id,
                    "step_index": assignment.step_index,
                    "split": assignment.split,
                    "confidence": round(assignment.confidence, 6),
                    "margin": round(assignment.margin, 6),
                    "payload_text": assignment.payload_text,
                    "step_view": {
                        "tool_name": assignment.step_view.tool_name,
                        "status": assignment.step_view.status,
                        "action_text": assignment.step_view.action_text,
                        "tool_args_text": assignment.step_view.tool_args_text,
                        "result_text": assignment.step_view.result_text,
                        "metadata_text": assignment.step_view.metadata_text,
                        "observation_text": assignment.step_view.observation_text,
                        "lexical_text": lexical_text,
                    },
                }
            )
            if len(top_examples) >= top_k_examples:
                break

        weighted_ngrams = []
        for ngram, weight in symbol_ngram_weights[symbol_id].items():
            global_weight = global_ngram_weights[ngram]
            distinctiveness = weight / global_weight if global_weight else 0.0
            score = (weight * weight) / max(global_weight, 1e-8)
            weighted_ngrams.append(
                {
                    "ngram": ngram,
                    "weight": round(float(weight), 6),
                    "global_weight": round(float(global_weight), 6),
                    "distinctiveness": round(float(distinctiveness), 6),
                    "score": round(float(score), 6),
                }
            )
        weighted_ngrams.sort(
            key=lambda item: (item["score"], item["weight"], item["ngram"]),
            reverse=True,
        )

        symbol_summaries.append(
            {
                "symbol_id": symbol_id,
                "assignment_count": len(symbol_assignments),
                "assignment_rate": (
                    round(len(symbol_assignments) / total_assignments, 6)
                    if total_assignments
                    else 0.0
                ),
                "top_representative_step_views": top_examples,
                "top_lexical_ngrams": weighted_ngrams[:top_k_ngrams],
            }
        )
    return symbol_summaries


def _precompute_rendered_step_views(
    trajectories: list[TrajectoryRecord],
    *,
    max_observation_lines: int,
    step_view_frontend: str,
    tau2_refinement_profile: str | None,
    skillsbench_process_profile: str | None,
) -> dict[str, list[str]]:
    rendered: dict[str, list[str]] = {}
    for trajectory in trajectories:
        rendered[trajectory.trajectory_id] = [
            build_step_view(
                step,
                max_observation_lines=max_observation_lines,
                dataset_name=step_view_frontend,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
            ).render_text("full")
            for step in trajectory.steps
        ]
    return rendered


def _cached_step_view_tail(
    rendered_step_views: dict[str, list[str]],
    trajectory_id: str,
    *,
    prefix_index: int,
    max_steps: int,
) -> list[str]:
    step_views = rendered_step_views[trajectory_id]
    return step_views[max(0, prefix_index - max_steps) : prefix_index]


def summarize_states(
    *,
    eval_trajectories: list[TrajectoryRecord],
    eval_symbols: dict[str, list[int]],
    horizon: int,
    details: dict,
    trusted_state_min_count: int,
    max_observation_lines: int,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    top_k_prefixes: int = 3,
    top_k_warning_traces: int = 3,
    max_steps_per_trace: int = 4,
) -> list[dict]:
    dfa = details["dfa"]
    state_risk = details["state_risk"]
    state_support = details["state_support"]
    threshold = float(details["threshold"])
    prefix_records = build_prefix_dataset(eval_trajectories, horizon=horizon)
    prefix_states = details["val_states"]
    prefix_scores = details["val_scores"]
    trusted_states = {
        state for state, support in state_support.items() if support >= trusted_state_min_count
    }
    trajectory_by_id = {
        trajectory.trajectory_id: trajectory for trajectory in eval_trajectories
    }
    rendered_step_views = _precompute_rendered_step_views(
        eval_trajectories,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
    )

    if len(prefix_records) != len(prefix_states):
        raise ValueError("Prefix/state alignment broke during post-hoc analysis")

    prefixes_by_state: dict[int, list[dict]] = defaultdict(list)
    for prefix_record, state, score in zip(prefix_records, prefix_states, prefix_scores):
        normalized_position = prefix_record.prefix_index / prefix_record.full_length
        is_warning = state in trusted_states and score >= threshold
        prefixes_by_state[state].append(
            {
                "trajectory_id": prefix_record.trajectory_id,
                "prefix_index": prefix_record.prefix_index,
                "full_length": prefix_record.full_length,
                "normalized_position": round(float(normalized_position), 6),
                "future_failure_label": int(prefix_record.future_failure_label),
                "final_success": bool(prefix_record.final_success),
                "score": round(float(score), 6),
                "warning": bool(is_warning),
                "tail_symbols": eval_symbols[prefix_record.trajectory_id][
                    max(0, prefix_record.prefix_index - max_steps_per_trace) : prefix_record.prefix_index
                ],
                "tail_step_views": _cached_step_view_tail(
                    rendered_step_views,
                    prefix_record.trajectory_id,
                    prefix_index=prefix_record.prefix_index,
                    max_steps=max_steps_per_trace,
                ),
            }
        )

    warning_traces_by_state: dict[int, list[dict]] = defaultdict(list)
    for trajectory in eval_trajectories:
        sequence = eval_symbols[trajectory.trajectory_id]
        state = dfa.start_state
        for index, symbol in enumerate(sequence, start=1):
            state = dfa.transition(state, symbol)
            if state not in trusted_states:
                continue
            if state_risk[state] < threshold:
                continue
            detection_latency = index / max(len(sequence), 1)
            warning_traces_by_state[state].append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "final_success": bool(trajectory.final_success),
                    "failure_bucket": trajectory.failure_bucket,
                    "detection_index": index,
                    "full_length": len(sequence),
                    "alert_lead_time": round(float(1.0 - detection_latency), 6),
                    "tail_symbols": sequence[max(0, index - max_steps_per_trace) : index],
                    "tail_step_views": _cached_step_view_tail(
                        rendered_step_views,
                        trajectory.trajectory_id,
                        prefix_index=index,
                        max_steps=max_steps_per_trace,
                    ),
                }
            )
            break

    state_ids = sorted(
        dfa.transitions.keys(),
        key=lambda state: (
            len(prefixes_by_state.get(state, [])),
            state_support.get(state, 0),
            -int(state),
        ),
        reverse=True,
    )

    summaries: list[dict] = []
    for state in state_ids:
        exemplar_prefixes = sorted(
            prefixes_by_state.get(state, []),
            key=lambda item: (
                item["future_failure_label"],
                item["warning"],
                -item["normalized_position"],
                item["score"],
            ),
            reverse=True,
        )
        warning_traces = sorted(
            warning_traces_by_state.get(state, []),
            key=lambda item: (
                int(not item["final_success"]),
                item["alert_lead_time"],
                -item["detection_index"],
            ),
            reverse=True,
        )
        summaries.append(
            {
                "state_id": int(state),
                "risk": round(float(state_risk.get(state, 0.0)), 6),
                "calibration_support": int(state_support.get(state, 0)),
                "eval_prefix_count": int(len(prefixes_by_state.get(state, []))),
                "trusted": bool(state in trusted_states),
                "warning_state": bool(
                    state in trusted_states and state_risk.get(state, 0.0) >= threshold
                ),
                "exemplar_prefixes": exemplar_prefixes[:top_k_prefixes],
                "warning_traces": warning_traces[:top_k_warning_traces],
            }
        )
    return summaries


def analyze_symbolic_checkpoint(
    *,
    fit_trajectories: list[TrajectoryRecord],
    cal_trajectories: list[TrajectoryRecord],
    eval_trajectories: list[TrajectoryRecord],
    symbol_analysis_trajectories: list[TrajectoryRecord],
    encoder: BaseSegmentEncoder,
    symbolizer: GumbelEventSymbolizer,
    device: torch.device,
    horizon: int,
    num_symbols: int,
    dfa_backend: DfaBackendName = "aalpy-rpni",
    representation_mode: str = "legacy",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    symbol_temperature: float = 1.0,
    hardening_strategy: HardeningStrategyName = DEFAULT_HARDENING_STRATEGY,
    hardening_threshold: float | None = None,
    trusted_state_min_count: int = 10,
    state_risk_smoothing_alpha: float = 5.0,
    calibration_bins: int = 10,
    legacy_reproduction: bool = False,
    top_k_symbol_examples: int = 5,
    top_k_symbol_ngrams: int = 10,
    top_k_state_prefixes: int = 3,
    top_k_warning_traces: int = 3,
    max_steps_per_trace: int = 4,
    include_symbol_analysis: bool = True,
    include_state_analysis: bool = True,
) -> dict:
    metrics: dict | None = None
    details: dict | None = None
    eval_symbols: dict[str, list[int]] | None = None

    if include_state_analysis:
        fit_symbols = symbolize_trajectories(
            fit_trajectories,
            encoder,
            symbolizer,
            device=device,
            progress_label="posthoc/fit-symbolization",
            deterministic=True,
            symbol_temperature=symbol_temperature,
            hardening_strategy=hardening_strategy,
            hardening_threshold=hardening_threshold,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        if cal_trajectories is fit_trajectories:
            cal_symbols = fit_symbols
        else:
            cal_symbols = symbolize_trajectories(
                cal_trajectories,
                encoder,
                symbolizer,
                device=device,
                progress_label="posthoc/cal-symbolization",
                deterministic=True,
                symbol_temperature=symbol_temperature,
                hardening_strategy=hardening_strategy,
                hardening_threshold=hardening_threshold,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                step_view_frontend=step_view_frontend,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
            )
        eval_symbols = symbolize_trajectories(
            eval_trajectories,
            encoder,
            symbolizer,
            device=device,
            progress_label="posthoc/eval-symbolization",
            deterministic=True,
            symbol_temperature=symbol_temperature,
            hardening_strategy=hardening_strategy,
            hardening_threshold=hardening_threshold,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        metrics, details = _evaluate_precomputed_symbol_sequences_with_details(
            train_trajectories=fit_trajectories,
            val_trajectories=eval_trajectories,
            train_symbols=fit_symbols,
            val_symbols=eval_symbols,
            cal_trajectories=cal_trajectories,
            cal_symbols=cal_symbols,
            horizon=horizon,
            num_symbols=num_symbols,
            dfa_backend=dfa_backend,
            trusted_state_min_count=trusted_state_min_count,
            state_risk_smoothing_alpha=state_risk_smoothing_alpha,
            calibration_bins=calibration_bins,
            legacy_reproduction=legacy_reproduction,
        )

    symbol_summary: list[dict] = []
    if include_symbol_analysis:
        step_assignments = collect_step_assignments(
            symbol_analysis_trajectories,
            encoder,
            symbolizer,
            device=device,
            symbol_temperature=symbol_temperature,
            hardening_strategy=hardening_strategy,
            hardening_threshold=hardening_threshold,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            progress_label="posthoc/symbol-analysis",
        )
        symbol_summary = summarize_symbols(
            step_assignments,
            num_symbols=num_symbols,
            top_k_examples=top_k_symbol_examples,
            top_k_ngrams=top_k_symbol_ngrams,
        )

    state_summary: list[dict] = []
    if include_state_analysis:
        assert details is not None and eval_symbols is not None
        state_summary = summarize_states(
            eval_trajectories=eval_trajectories,
            eval_symbols=eval_symbols,
            horizon=horizon,
            details=details,
            trusted_state_min_count=trusted_state_min_count,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            top_k_prefixes=top_k_state_prefixes,
            top_k_warning_traces=top_k_warning_traces,
            max_steps_per_trace=max_steps_per_trace,
        )
    return {
        "metadata": {
            "representation_mode": representation_mode,
            "step_view_frontend": step_view_frontend,
            "tau2_refinement_profile": tau2_refinement_profile,
            "skillsbench_process_profile": skillsbench_process_profile,
            "num_symbols": int(num_symbols),
            "horizon": int(horizon),
            "dfa_backend": str(dfa_backend),
            "trusted_state_min_count": int(trusted_state_min_count),
            "state_risk_smoothing_alpha": float(state_risk_smoothing_alpha),
            "calibration_bins": int(calibration_bins),
            "legacy_reproduction": bool(legacy_reproduction),
            "symbol_temperature": float(symbol_temperature),
            "hardening_strategy": str(hardening_strategy),
            "hardening_threshold": None if hardening_threshold is None else float(hardening_threshold),
            "symbol_analysis_trajectory_count": int(len(symbol_analysis_trajectories)),
            "eval_trajectory_count": int(len(eval_trajectories)),
            "include_symbol_analysis": bool(include_symbol_analysis),
            "include_state_analysis": bool(include_state_analysis),
        },
        "monitor_metrics": metrics,
        "symbol_analysis": symbol_summary,
        "state_analysis": state_summary,
    }
