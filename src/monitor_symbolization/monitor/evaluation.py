from __future__ import annotations

from collections import defaultdict
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from monitor_symbolization.data.schema import TrajectoryRecord
from monitor_symbolization.models.differentiable_automaton import (
    DifferentiableFiniteStateSurrogate,
    FlatPrefixRiskHead,
)
from monitor_symbolization.models.encoders import BaseSegmentEncoder
from monitor_symbolization.models.symbolizer import GumbelEventSymbolizer
from monitor_symbolization.monitor.backends import DfaBackendName, fit_dfa_with_backend
from monitor_symbolization.monitor.hardening import (
    DEFAULT_HARDENING_STRATEGY,
    HardeningStrategyName,
    harden_symbol_probabilities,
)
from monitor_symbolization.monitor.rpni import DFA
from monitor_symbolization.runtime_cache import (
    RuntimeCache,
    encode_trajectories,
    iter_trajectory_batches,
)


def _summarize_trajectory_steps(trajectories: list[TrajectoryRecord]) -> tuple[int, int]:
    return len(trajectories), sum(len(trajectory.steps) for trajectory in trajectories)


def _progress_interval(total: int) -> int:
    if total <= 0:
        return 1
    return max(1, min(200, total // 10 or 1))


def symbolize_trajectories(
    trajectories: list[TrajectoryRecord],
    encoder: BaseSegmentEncoder,
    symbolizer: GumbelEventSymbolizer,
    device: torch.device,
    progress_label: str | None = None,
    deterministic: bool = False,
    symbol_temperature: float = 1.0,
    representation_mode: str = "legacy",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    runtime_cache: RuntimeCache | None = None,
    hardening_strategy: HardeningStrategyName = DEFAULT_HARDENING_STRATEGY,
    hardening_threshold: float | None = None,
) -> dict[str, list[int]]:
    symbolizer.eval()
    cache = runtime_cache or RuntimeCache()
    total_trajectories, total_steps = _summarize_trajectory_steps(trajectories)
    if progress_label:
        print(
            f"[{progress_label}] materializing {total_trajectories} trajectories "
            f"into {total_steps} serialized steps",
            flush=True,
        )
    symbol_sequences: dict[str, list[int]] = {}
    trajectory_batches = iter_trajectory_batches(
        encoder,
        trajectories,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        dataset_name=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
        runtime_cache=cache,
    )
    for batch_index, trajectory_batch in enumerate(trajectory_batches, start=1):
        batch_progress_label = (
            f"{progress_label}/batch-{batch_index}"
            if progress_label and batch_index > 1
            else progress_label
        )
        with torch.no_grad():
            embeddings, lengths, trajectory_ids = encode_trajectories(
                encoder,
                trajectory_batch,
                device=device,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                dataset_name=step_view_frontend,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
                runtime_cache=cache,
                progress_label=batch_progress_label,
            )
            if deterministic or hardening_strategy != DEFAULT_HARDENING_STRATEGY:
                outputs = symbolizer.deterministic_output(
                    embeddings,
                    temperature=symbol_temperature,
                )
                probabilities = outputs.probs.detach().cpu()
            else:
                outputs = symbolizer(
                    embeddings,
                    temperature=symbol_temperature,
                    hard=True,
                )
                probabilities = outputs.probs.detach().cpu()

        cursor = 0
        hard_ids = outputs.hard_ids.detach().cpu().tolist()
        for trajectory_id, length in zip(trajectory_ids, lengths):
            if hardening_strategy == DEFAULT_HARDENING_STRATEGY and not deterministic:
                symbol_sequences[trajectory_id] = hard_ids[cursor : cursor + length]
            else:
                hardening = harden_symbol_probabilities(
                    probabilities[cursor : cursor + length],
                    strategy=hardening_strategy,
                    threshold=hardening_threshold,
                )
                symbol_sequences[trajectory_id] = hardening.hard_ids
            cursor += length
    if progress_label:
        print(
            f"[{progress_label}] converted embeddings to symbol sequences for "
            f"{len(symbol_sequences)} trajectories",
            flush=True,
        )
    return symbol_sequences


def _prefix_sequences(
    trajectories: list[TrajectoryRecord],
    symbol_sequences: dict[str, list[int]],
    horizon: int,
    runtime_cache: RuntimeCache | None = None,
) -> tuple[list[list[int]], list[int], list[str]]:
    sequences = []
    labels = []
    trajectory_ids = []
    cache = runtime_cache or RuntimeCache()
    for trajectory in trajectories:
        full_sequence = symbol_sequences[trajectory.trajectory_id]
        future_failure_labels = cache.get_future_failure_labels(
            trajectory,
            horizon=horizon,
        )
        prefix_label_mask = cache.get_prefix_label_mask(trajectory)
        for prefix_index, future_failure_label in enumerate(
            future_failure_labels,
            start=1,
        ):
            if not prefix_label_mask[prefix_index - 1]:
                continue
            sequences.append(full_sequence[:prefix_index])
            labels.append(future_failure_label)
            trajectory_ids.append(trajectory.trajectory_id)
    return sequences, labels, trajectory_ids


def _full_trace_labels(trajectories: list[TrajectoryRecord]) -> tuple[list[list[int]], list[list[int]]]:
    raise RuntimeError("_full_trace_labels must be called with materialized symbol sequences")


def _build_consistent_trace_sets(
    positive_traces: list[list[int]],
    negative_traces: list[list[int]],
) -> tuple[list[list[int]], list[list[int]], dict[str, int]]:
    positive_set = {tuple(trace) for trace in positive_traces}
    negative_set = {tuple(trace) for trace in negative_traces}
    ambiguous = positive_set & negative_set
    filtered_positive = sorted(positive_set - ambiguous)
    filtered_negative = sorted(negative_set - ambiguous)
    stats = {
        "positive_trace_count": len(positive_traces),
        "negative_trace_count": len(negative_traces),
        "unique_positive_trace_count": len(positive_set),
        "unique_negative_trace_count": len(negative_set),
        "ambiguous_trace_count": len(ambiguous),
        "consistent_positive_trace_count": len(filtered_positive),
        "consistent_negative_trace_count": len(filtered_negative),
    }
    return [list(trace) for trace in filtered_positive], [list(trace) for trace in filtered_negative], stats


def _replay_state(dfa: DFA, sequence: list[int]) -> int:
    state = dfa.start_state
    for symbol in sequence:
        state = dfa.transition(state, symbol)
    return state


def _fit_state_risk(
    dfa: DFA,
    prefix_sequences: list[list[int]],
    prefix_labels: list[int],
    smoothing_alpha: float = 5.0,
) -> tuple[dict[int, float], dict[int, int], float]:
    global_failure_rate = float(np.mean(prefix_labels)) if prefix_labels else 0.0
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for sequence, label in zip(prefix_sequences, prefix_labels):
        state = _replay_state(dfa, sequence)
        counts[state][0] += int(label)
        counts[state][1] += 1

    state_risk = {}
    state_support = {}
    for state in dfa.transitions:
        failures, total = counts[state]
        state_support[state] = total
        state_risk[state] = (failures + smoothing_alpha * global_failure_rate) / (
            total + smoothing_alpha
        )
    return state_risk, state_support, global_failure_rate


def _replay_prefix_states(
    dfa: DFA,
    prefix_sequences: list[list[int]],
) -> list[int]:
    return [_replay_state(dfa, sequence) for sequence in prefix_sequences]


def _score_prefixes(
    dfa: DFA,
    state_risk: dict[int, float],
    prefix_sequences: list[list[int]],
) -> list[float]:
    return [state_risk[_replay_state(dfa, sequence)] for sequence in prefix_sequences]


def _compute_binary_metrics(
    labels: list[int],
    scores: list[float],
) -> tuple[float, float]:
    auroc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5
    auprc = average_precision_score(labels, scores) if labels else 0.0
    return float(auroc), float(auprc)


def _select_threshold(scores: list[float], labels: list[int]) -> float:
    if not scores:
        return 0.5
    # Preserve the original semantics exactly:
    # 1. Candidates are the sorted unique score values.
    # 2. Predictions use `score >= threshold`.
    # 3. Ties keep the earliest candidate in ascending order.
    scored_labels = sorted(zip(scores, labels), key=lambda item: item[0])
    grouped_counts: list[tuple[float, int, int]] = []
    for score, label in scored_labels:
        if grouped_counts and score == grouped_counts[-1][0]:
            threshold, positives, negatives = grouped_counts[-1]
            if label:
                grouped_counts[-1] = (threshold, positives + 1, negatives)
            else:
                grouped_counts[-1] = (threshold, positives, negatives + 1)
        else:
            grouped_counts.append((score, int(label == 1), int(label == 0)))

    total_positives = int(sum(label == 1 for label in labels))
    tp = total_positives
    fp = len(labels) - total_positives
    best_threshold = grouped_counts[0][0]
    best_f1 = -1.0

    for index, (threshold, _, _) in enumerate(grouped_counts):
        fn = total_positives - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

        if index == len(grouped_counts) - 1:
            break
        _, positive_count, negative_count = grouped_counts[index]
        tp -= positive_count
        fp -= negative_count

    return float(best_threshold)


def _compute_detection_latency(
    trajectories: list[TrajectoryRecord],
    symbol_sequences: dict[str, list[int]],
    dfa: DFA,
    state_risk: dict[int, float],
    threshold: float,
    trusted_states: set[int] | None = None,
) -> float:
    latencies = []
    for trajectory in trajectories:
        if trajectory.final_success:
            continue
        sequence = symbol_sequences[trajectory.trajectory_id]
        state = dfa.start_state
        detected_at = len(sequence)
        for index, symbol in enumerate(sequence, start=1):
            state = dfa.transition(state, symbol)
            if trusted_states is not None and state not in trusted_states:
                continue
            if state_risk[state] >= threshold:
                detected_at = index
                break
        latencies.append(detected_at / len(sequence))
    return float(np.mean(latencies)) if latencies else 1.0


def _compute_brier_score(scores: list[float], labels: list[int]) -> float:
    if not scores:
        return 0.0
    return float(np.mean([(score - label) ** 2 for score, label in zip(scores, labels)]))


def _compute_expected_calibration_error(
    scores: list[float],
    labels: list[int],
    *,
    num_bins: int = 10,
) -> float:
    if not scores:
        return 0.0
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.float64)
    ece = 0.0
    total = len(scores)
    for index in range(num_bins):
        left = bin_edges[index]
        right = bin_edges[index + 1]
        if index == num_bins - 1:
            in_bin = (scores_array >= left) & (scores_array <= right)
        else:
            in_bin = (scores_array >= left) & (scores_array < right)
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        avg_confidence = float(np.mean(scores_array[in_bin]))
        avg_accuracy = float(np.mean(labels_array[in_bin]))
        ece += (count / total) * abs(avg_accuracy - avg_confidence)
    return float(ece)


def _compute_trusted_metrics(
    *,
    prefix_states: list[int],
    labels: list[int],
    state_risk: dict[int, float],
    state_support: dict[int, int],
    min_count: int,
    calibration_bins: int,
) -> dict[str, float | int]:
    if min_count < 0:
        raise ValueError("min_count must be non-negative")
    trusted_states = {
        state
        for state, support in state_support.items()
        if support >= min_count
    }
    trusted_scores = [
        state_risk[state]
        for state in prefix_states
        if state in trusted_states
    ]
    trusted_labels = [
        label
        for state, label in zip(prefix_states, labels)
        if state in trusted_states
    ]
    trusted_auroc, trusted_auprc = _compute_binary_metrics(trusted_labels, trusted_scores)
    total_prefix_count = len(labels)
    trusted_prefix_count = len(trusted_labels)
    trusted_positive_prefix_count = int(sum(trusted_labels))
    trusted_negative_prefix_count = trusted_prefix_count - trusted_positive_prefix_count
    abstention_rate = 1.0 - (trusted_prefix_count / total_prefix_count) if total_prefix_count else 0.0
    calibration_error = _compute_expected_calibration_error(
        trusted_scores,
        trusted_labels,
        num_bins=calibration_bins,
    )
    brier_score = _compute_brier_score(trusted_scores, trusted_labels)
    trusted_positive_rate = (
        trusted_positive_prefix_count / trusted_prefix_count
        if trusted_prefix_count
        else 0.0
    )
    return {
        "trusted_state_count": int(len(trusted_states)),
        "trusted_prefix_count": int(trusted_prefix_count),
        "trusted_positive_prefix_count": int(trusted_positive_prefix_count),
        "trusted_negative_prefix_count": int(trusted_negative_prefix_count),
        "trusted_prefix_rate": float(trusted_prefix_count / total_prefix_count)
        if total_prefix_count
        else 0.0,
        "trusted_prefix_positive_rate": float(trusted_positive_rate),
        "abstention_rate": float(abstention_rate),
        "trusted_state_auroc": float(trusted_auroc),
        "trusted_state_auprc": float(trusted_auprc),
        "calibration_error": float(calibration_error),
        "ece": float(calibration_error),
        "brier_score": float(brier_score),
        "trusted_state_min_count": int(min_count),
        "calibration_bins": int(calibration_bins),
    }


def _prepare_trace_sets(
    train_trajectories: list[TrajectoryRecord],
    train_symbols: dict[str, list[int]],
    num_symbols: int,
) -> tuple[list[list[int]], list[list[int]], dict[str, int]]:
    terminal_symbol = num_symbols
    positive_traces = [
        train_symbols[trajectory.trajectory_id] + [terminal_symbol]
        for trajectory in train_trajectories
        if trajectory.final_success
    ]
    negative_traces = [
        train_symbols[trajectory.trajectory_id] + [terminal_symbol]
        for trajectory in train_trajectories
        if not trajectory.final_success
    ]
    return _build_consistent_trace_sets(
        positive_traces=positive_traces,
        negative_traces=negative_traces,
    )


def induce_dfa_from_symbol_sequences(
    train_trajectories: list[TrajectoryRecord],
    train_symbols: dict[str, list[int]],
    num_symbols: int,
    dfa_backend: DfaBackendName = "aalpy",
) -> tuple[DFA, dict[str, int], float]:
    positive_traces, negative_traces, trace_stats = _prepare_trace_sets(
        train_trajectories=train_trajectories,
        train_symbols=train_symbols,
        num_symbols=num_symbols,
    )
    if not positive_traces or not negative_traces:
        raise ValueError(
            "No consistent positive/negative full traces remain after ambiguity filtering; "
            "the symbolic front-end is too collapsed for exact DFA induction."
        )
    print(
        f"[validation] inducing DFA with backend={dfa_backend} from "
        f"{len(positive_traces)} positive / {len(negative_traces)} negative traces",
        flush=True,
    )
    start_time = perf_counter()
    dfa = fit_dfa_with_backend(
        positive_traces=positive_traces,
        negative_traces=negative_traces,
        alphabet_size=num_symbols + 1,
        backend=dfa_backend,
    )
    induction_time = perf_counter() - start_time
    print(
        f"[validation] DFA induction finished in {induction_time:.2f}s with "
        f"{dfa.state_count} states",
        flush=True,
    )
    return dfa, trace_stats, induction_time


def evaluate_symbolic_monitor(
    train_trajectories: list[TrajectoryRecord],
    val_trajectories: list[TrajectoryRecord],
    encoder: BaseSegmentEncoder,
    symbolizer: GumbelEventSymbolizer,
    horizon: int,
    num_symbols: int,
    device: torch.device,
    dfa_backend: DfaBackendName = "aalpy",
    representation_mode: str = "legacy",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    trusted_state_min_count: int = 10,
    state_risk_smoothing_alpha: float = 5.0,
    calibration_bins: int = 10,
    cal_trajectories: list[TrajectoryRecord] | None = None,
    legacy_reproduction: bool = False,
    runtime_cache: RuntimeCache | None = None,
    hardening_strategy: HardeningStrategyName = DEFAULT_HARDENING_STRATEGY,
    hardening_threshold: float | None = None,
) -> dict:
    cache = runtime_cache or RuntimeCache()
    print("[validation] start train symbolization", flush=True)
    train_symbols = symbolize_trajectories(
        train_trajectories,
        encoder,
        symbolizer,
        device=device,
        progress_label="validation/train-symbolization",
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
        runtime_cache=cache,
        hardening_strategy=hardening_strategy,
        hardening_threshold=hardening_threshold,
    )
    calibration_trajectories = cal_trajectories or train_trajectories
    if cal_trajectories is None:
        cal_symbols = train_symbols
    else:
        print("[validation] start cal symbolization", flush=True)
        cal_symbols = symbolize_trajectories(
            calibration_trajectories,
            encoder,
            symbolizer,
            device=device,
            progress_label="validation/cal-symbolization",
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            runtime_cache=cache,
            hardening_strategy=hardening_strategy,
            hardening_threshold=hardening_threshold,
        )
    print("[validation] start val symbolization", flush=True)
    val_symbols = symbolize_trajectories(
        val_trajectories,
        encoder,
        symbolizer,
        device=device,
        progress_label="validation/val-symbolization",
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
        runtime_cache=cache,
        hardening_strategy=hardening_strategy,
        hardening_threshold=hardening_threshold,
    )
    metrics = evaluate_precomputed_symbol_sequences(
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        train_symbols=train_symbols,
        val_symbols=val_symbols,
        cal_trajectories=calibration_trajectories,
        cal_symbols=cal_symbols,
        horizon=horizon,
        num_symbols=num_symbols,
        dfa_backend=dfa_backend,
        trusted_state_min_count=trusted_state_min_count,
        state_risk_smoothing_alpha=state_risk_smoothing_alpha,
        calibration_bins=calibration_bins,
        legacy_reproduction=legacy_reproduction,
        runtime_cache=cache,
    )
    metrics["hardening_strategy"] = str(hardening_strategy)
    metrics["hardening_threshold"] = (
        None if hardening_threshold is None else float(hardening_threshold)
    )
    return metrics


def evaluate_precomputed_symbol_sequences(
    train_trajectories: list[TrajectoryRecord],
    val_trajectories: list[TrajectoryRecord],
    train_symbols: dict[str, list[int]],
    val_symbols: dict[str, list[int]],
    horizon: int,
    num_symbols: int,
    dfa_backend: DfaBackendName = "aalpy",
    trusted_state_min_count: int = 10,
    state_risk_smoothing_alpha: float = 5.0,
    calibration_bins: int = 10,
    cal_trajectories: list[TrajectoryRecord] | None = None,
    cal_symbols: dict[str, list[int]] | None = None,
    legacy_reproduction: bool = False,
    runtime_cache: RuntimeCache | None = None,
) -> dict:
    metrics, _ = _evaluate_precomputed_symbol_sequences_with_details(
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        train_symbols=train_symbols,
        val_symbols=val_symbols,
        cal_trajectories=cal_trajectories,
        cal_symbols=cal_symbols,
        horizon=horizon,
        num_symbols=num_symbols,
        dfa_backend=dfa_backend,
        trusted_state_min_count=trusted_state_min_count,
        state_risk_smoothing_alpha=state_risk_smoothing_alpha,
            calibration_bins=calibration_bins,
            legacy_reproduction=legacy_reproduction,
            runtime_cache=runtime_cache,
        )
    return metrics


def _evaluate_precomputed_symbol_sequences_with_details(
    train_trajectories: list[TrajectoryRecord],
    val_trajectories: list[TrajectoryRecord],
    train_symbols: dict[str, list[int]],
    val_symbols: dict[str, list[int]],
    horizon: int,
    num_symbols: int,
    dfa_backend: DfaBackendName = "aalpy",
    trusted_state_min_count: int = 10,
    state_risk_smoothing_alpha: float = 5.0,
    calibration_bins: int = 10,
    cal_trajectories: list[TrajectoryRecord] | None = None,
    cal_symbols: dict[str, list[int]] | None = None,
    legacy_reproduction: bool = False,
    runtime_cache: RuntimeCache | None = None,
) -> tuple[dict, dict]:
    cache = runtime_cache or RuntimeCache()
    dfa, trace_stats, _ = induce_dfa_from_symbol_sequences(
        train_trajectories=train_trajectories,
        train_symbols=train_symbols,
        num_symbols=num_symbols,
        dfa_backend=dfa_backend,
    )

    train_prefixes, train_labels, _ = _prefix_sequences(
        train_trajectories,
        train_symbols,
        horizon=horizon,
        runtime_cache=cache,
    )
    calibration_trajectories = cal_trajectories or train_trajectories
    calibration_symbols = cal_symbols or train_symbols
    cal_prefixes, cal_labels, _ = _prefix_sequences(
        calibration_trajectories,
        calibration_symbols,
        horizon=horizon,
        runtime_cache=cache,
    )
    val_prefixes, val_labels, _ = _prefix_sequences(
        val_trajectories,
        val_symbols,
        horizon=horizon,
        runtime_cache=cache,
    )
    if legacy_reproduction:
        state_risk, state_support, _ = _fit_state_risk(
            dfa,
            train_prefixes,
            train_labels,
            smoothing_alpha=1.0,
        )
        cal_states = _replay_prefix_states(dfa, train_prefixes)
        cal_scores = [state_risk[state] for state in cal_states]
        threshold_labels = val_labels
        threshold_split = "eval"
        state_risk_fit_split = "train"
        calibration_prefix_count = len(train_labels)
        calibration_prefix_positive_rate = (
            float(np.mean(train_labels)) if train_labels else 0.0
        )
    else:
        state_risk, state_support, _ = _fit_state_risk(
            dfa,
            cal_prefixes,
            cal_labels,
            smoothing_alpha=state_risk_smoothing_alpha,
        )
        cal_states = _replay_prefix_states(dfa, cal_prefixes)
        cal_scores = [state_risk[state] for state in cal_states]
        threshold_labels = cal_labels
        threshold_split = "cal"
        state_risk_fit_split = "cal"
        calibration_prefix_count = len(cal_labels)
        calibration_prefix_positive_rate = (
            float(np.mean(cal_labels)) if cal_labels else 0.0
        )
    val_states = _replay_prefix_states(dfa, val_prefixes)
    val_scores = [state_risk[state] for state in val_states]
    threshold_scores = val_scores if legacy_reproduction else cal_scores
    threshold = _select_threshold(threshold_scores, threshold_labels)

    auroc, auprc = _compute_binary_metrics(val_labels, val_scores)
    latency = _compute_detection_latency(
        trajectories=val_trajectories,
        symbol_sequences=val_symbols,
        dfa=dfa,
        state_risk=state_risk,
        threshold=threshold,
    )
    trusted_metrics = _compute_trusted_metrics(
        prefix_states=val_states,
        labels=val_labels,
        state_risk=state_risk,
        state_support=state_support,
        min_count=trusted_state_min_count,
        calibration_bins=calibration_bins,
    )
    trusted_states = {
        state
        for state, support in state_support.items()
        if support >= trusted_state_min_count
    }
    trusted_latency = _compute_detection_latency(
        trajectories=val_trajectories,
        symbol_sequences=val_symbols,
        dfa=dfa,
        state_risk=state_risk,
        threshold=threshold,
        trusted_states=trusted_states,
    )
    metrics = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "detection_latency": float(latency),
        "trusted_detection_latency": float(trusted_latency),
        "alert_lead_time": float(1.0 - trusted_latency),
        "threshold": float(threshold),
        "dfa_state_count": int(dfa.state_count),
        "prefix_count": int(len(val_labels)),
        "prefix_positive_rate": float(np.mean(val_labels)) if val_labels else 0.0,
        "calibration_prefix_count": int(calibration_prefix_count),
        "calibration_prefix_positive_rate": float(calibration_prefix_positive_rate),
        "state_risk_smoothing_alpha": float(state_risk_smoothing_alpha),
        "legacy_reproduction": bool(legacy_reproduction),
        "state_risk_fit_split": state_risk_fit_split,
        "threshold_selection_split": threshold_split,
        "extraction_success_rate": 1.0,
        **trusted_metrics,
        **trace_stats,
    }
    print(
        "[validation] completed: "
        f"AUROC={metrics['auroc']:.4f}, "
        f"AUPRC={metrics['auprc']:.4f}, "
        f"trusted_AUPRC={metrics['trusted_state_auprc']:.4f}, "
        f"states={metrics['dfa_state_count']}, "
        f"latency={metrics['detection_latency']:.4f}, "
        f"abstention={metrics['abstention_rate']:.4f}",
        flush=True,
    )
    details = {
        "dfa": dfa,
        "state_risk": state_risk,
        "state_support": state_support,
        "threshold": float(threshold),
        "cal_labels": cal_labels,
        "cal_scores": cal_scores,
        "cal_states": cal_states,
        "val_labels": val_labels,
        "val_scores": val_scores,
        "val_states": val_states,
    }
    return metrics, details


def compare_dfa_backends_on_symbol_sequences(
    train_trajectories: list[TrajectoryRecord],
    val_trajectories: list[TrajectoryRecord],
    train_symbols: dict[str, list[int]],
    val_symbols: dict[str, list[int]],
    horizon: int,
    num_symbols: int,
    backends: tuple[DfaBackendName, ...] = ("legacy", "aalpy-edsm", "aalpy-rpni"),
    trusted_state_min_count: int = 10,
    state_risk_smoothing_alpha: float = 5.0,
    calibration_bins: int = 10,
    cal_trajectories: list[TrajectoryRecord] | None = None,
    cal_symbols: dict[str, list[int]] | None = None,
) -> dict:
    comparison: dict[str, dict] = {}
    for backend in backends:
        metrics, _ = _evaluate_precomputed_symbol_sequences_with_details(
            train_trajectories=train_trajectories,
            val_trajectories=val_trajectories,
            train_symbols=train_symbols,
            val_symbols=val_symbols,
            cal_trajectories=cal_trajectories,
            cal_symbols=cal_symbols,
            horizon=horizon,
            num_symbols=num_symbols,
            dfa_backend=backend,
            trusted_state_min_count=trusted_state_min_count,
            state_risk_smoothing_alpha=state_risk_smoothing_alpha,
            calibration_bins=calibration_bins,
        )
        _, _, induction_seconds = induce_dfa_from_symbol_sequences(
            train_trajectories=train_trajectories,
            train_symbols=train_symbols,
            num_symbols=num_symbols,
            dfa_backend=backend,
        )
        comparison[backend] = {
            "induction_seconds": float(induction_seconds),
            "metrics": metrics,
        }
    return {"backends": comparison}


def _score_soft_prefixes(
    trajectories: list[TrajectoryRecord],
    encoder: BaseSegmentEncoder,
    symbolizer: GumbelEventSymbolizer,
    automaton: DifferentiableFiniteStateSurrogate | FlatPrefixRiskHead,
    horizon: int,
    device: torch.device,
    symbol_temperature: float,
    transition_temperature: float,
    progress_label: str | None = None,
    representation_mode: str = "legacy",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    runtime_cache: RuntimeCache | None = None,
) -> tuple[list[float], list[int], dict[str, list[float]]]:
    symbolizer.eval()
    automaton.eval()
    scores: list[float] = []
    labels: list[int] = []
    per_trajectory_scores: dict[str, list[float]] = {}
    total_trajectories, total_steps = _summarize_trajectory_steps(trajectories)
    start_time = perf_counter()
    processed_steps = 0
    progress_interval = _progress_interval(total_trajectories)
    cache = runtime_cache or RuntimeCache()

    if progress_label:
        print(
            f"[{progress_label}] start soft prefix scoring over "
            f"{total_trajectories} trajectories / {total_steps} steps; "
            "using trajectory-step batching",
            flush=True,
        )

    encode_start = perf_counter()
    trajectory_batches = iter_trajectory_batches(
        encoder,
        trajectories,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        dataset_name=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
        runtime_cache=cache,
    )
    processed_trajectories = 0
    for batch_index, trajectory_batch in enumerate(trajectory_batches, start=1):
        batch_progress_label = (
            f"{progress_label}/encode-batch-{batch_index}"
            if progress_label and batch_index > 1
            else (f"{progress_label}/encode" if progress_label else None)
        )
        with torch.no_grad():
            embeddings, lengths, trajectory_ids = encode_trajectories(
                encoder,
                trajectory_batch,
                device=device,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                dataset_name=step_view_frontend,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
                runtime_cache=cache,
                progress_label=batch_progress_label,
            )
            symbolizer_output = symbolizer.deterministic_output(
                embeddings,
                temperature=symbol_temperature,
            )

        cursor = 0
        for trajectory, trajectory_id, length in zip(trajectory_batch, trajectory_ids, lengths):
            with torch.no_grad():
                automaton_output = automaton(
                    symbolizer_output.probs[cursor : cursor + length],
                    transition_temperature=transition_temperature,
                )
            trajectory_scores = [
                float(score) for score in automaton_output.risk_scores.detach().cpu().tolist()
            ]
            prefix_labels = list(
                cache.get_future_failure_labels(
                    trajectory,
                    horizon=horizon,
                )
            )
            prefix_label_mask = list(cache.get_prefix_label_mask(trajectory))
            masked_scores = [
                score for score, active in zip(trajectory_scores, prefix_label_mask) if active
            ]
            masked_labels = [
                label for label, active in zip(prefix_labels, prefix_label_mask) if active
            ]
            scores.extend(masked_scores)
            labels.extend(masked_labels)
            per_trajectory_scores[trajectory_id] = masked_scores
            processed_steps += length
            processed_trajectories += 1
            cursor += length
            if progress_label and (
                processed_trajectories % progress_interval == 0
                or processed_trajectories == total_trajectories
            ):
                elapsed = perf_counter() - start_time
                print(
                    f"[{progress_label}] progress {processed_trajectories}/{total_trajectories} trajectories, "
                    f"{processed_steps}/{total_steps} steps, {len(scores)} prefixes "
                    f"scored in {elapsed:.2f}s",
                    flush=True,
                )

    if progress_label:
        print(
            f"[{progress_label}] finished batched encoding in "
            f"{perf_counter() - encode_start:.2f}s",
            flush=True,
        )

    if progress_label:
        elapsed = perf_counter() - start_time
        print(
            f"[{progress_label}] completed soft prefix scoring: "
            f"{len(scores)} prefixes from {total_trajectories} trajectories in {elapsed:.2f}s",
            flush=True,
        )

    return scores, labels, per_trajectory_scores


def _score_soft_prefixes_trajectory_loop(
    trajectories: list[TrajectoryRecord],
    encoder: BaseSegmentEncoder,
    symbolizer: GumbelEventSymbolizer,
    automaton: DifferentiableFiniteStateSurrogate | FlatPrefixRiskHead,
    horizon: int,
    device: torch.device,
    symbol_temperature: float,
    transition_temperature: float,
    progress_label: str | None = None,
    representation_mode: str = "legacy",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    runtime_cache: RuntimeCache | None = None,
) -> tuple[list[float], list[int], dict[str, list[float]]]:
    symbolizer.eval()
    automaton.eval()
    scores: list[float] = []
    labels: list[int] = []
    per_trajectory_scores: dict[str, list[float]] = {}
    total_trajectories, total_steps = _summarize_trajectory_steps(trajectories)
    start_time = perf_counter()
    processed_steps = 0
    progress_interval = _progress_interval(total_trajectories)
    cache = runtime_cache or RuntimeCache()

    if progress_label:
        print(
            f"[{progress_label}] start soft prefix scoring over "
            f"{total_trajectories} trajectories / {total_steps} steps; "
            "encoding and scoring run one trajectory at a time",
            flush=True,
        )

    for index, trajectory in enumerate(trajectories, start=1):
        with torch.no_grad():
            embeddings, lengths, _ = encode_trajectories(
                encoder,
                [trajectory],
                device=device,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                dataset_name=step_view_frontend,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
                runtime_cache=cache,
                progress_label=f"{progress_label}/trajectory-1-example"
                if progress_label and index == 1
                else None,
            )
            symbolizer_output = symbolizer.deterministic_output(
                embeddings,
                temperature=symbol_temperature,
            )
            automaton_output = automaton(
                symbolizer_output.probs,
                transition_temperature=transition_temperature,
            )
        trajectory_scores = [float(score) for score in automaton_output.risk_scores.detach().cpu().tolist()]
        prefix_labels = list(
            cache.get_future_failure_labels(
                trajectory,
                horizon=horizon,
            )
        )
        prefix_label_mask = list(cache.get_prefix_label_mask(trajectory))
        masked_scores = [
            score for score, active in zip(trajectory_scores, prefix_label_mask) if active
        ]
        masked_labels = [
            label for label, active in zip(prefix_labels, prefix_label_mask) if active
        ]
        scores.extend(masked_scores)
        labels.extend(masked_labels)
        per_trajectory_scores[trajectory.trajectory_id] = masked_scores
        processed_steps += lengths[0]
        if progress_label and (
            index % progress_interval == 0 or index == total_trajectories
        ):
            elapsed = perf_counter() - start_time
            print(
                f"[{progress_label}] progress {index}/{total_trajectories} trajectories, "
                f"{processed_steps}/{total_steps} steps, {len(scores)} prefixes "
                f"scored in {elapsed:.2f}s",
                flush=True,
            )

    if progress_label:
        elapsed = perf_counter() - start_time
        print(
            f"[{progress_label}] completed soft prefix scoring: "
            f"{len(scores)} prefixes from {total_trajectories} trajectories in {elapsed:.2f}s",
            flush=True,
        )

    return scores, labels, per_trajectory_scores


def _compute_soft_detection_latency(
    trajectories: list[TrajectoryRecord],
    per_trajectory_scores: dict[str, list[float]],
    threshold: float,
) -> float:
    latencies = []
    for trajectory in trajectories:
        if trajectory.final_success:
            continue
        scores = per_trajectory_scores[trajectory.trajectory_id]
        detected_at = len(scores)
        for index, score in enumerate(scores, start=1):
            if score >= threshold:
                detected_at = index
                break
        latencies.append(detected_at / max(len(scores), 1))
    return float(np.mean(latencies)) if latencies else 1.0


def _compute_matched_coverage_soft_metrics(
    *,
    soft_scores: list[float],
    soft_labels: list[int],
    soft_threshold: float,
    discrete_prefix_states: list[int],
    discrete_state_support: dict[int, int],
    min_count: int,
    calibration_bins: int,
) -> dict[str, float | int]:
    if not (
        len(soft_scores) == len(soft_labels) == len(discrete_prefix_states)
    ):
        raise ValueError(
            "Soft scores/labels and discrete prefix states must have identical lengths"
        )
    trusted_states = {
        state
        for state, support in discrete_state_support.items()
        if support >= min_count
    }
    matched_scores = [
        score
        for score, state in zip(soft_scores, discrete_prefix_states)
        if state in trusted_states
    ]
    matched_labels = [
        label
        for label, state in zip(soft_labels, discrete_prefix_states)
        if state in trusted_states
    ]
    matched_auroc, matched_auprc = _compute_binary_metrics(matched_labels, matched_scores)
    matched_calibration_error = _compute_expected_calibration_error(
        matched_scores,
        matched_labels,
        num_bins=calibration_bins,
    )
    matched_brier = _compute_brier_score(matched_scores, matched_labels)
    total_prefix_count = len(soft_labels)
    matched_prefix_count = len(matched_labels)
    matched_positive_prefix_count = int(sum(matched_labels))
    matched_negative_prefix_count = matched_prefix_count - matched_positive_prefix_count
    matched_positive_rate = (
        matched_positive_prefix_count / matched_prefix_count
        if matched_prefix_count
        else 0.0
    )
    return {
        "available": 1.0,
        "auroc": float(matched_auroc),
        "auprc": float(matched_auprc),
        "threshold": float(soft_threshold),
        "prefix_count": int(matched_prefix_count),
        "positive_prefix_count": int(matched_positive_prefix_count),
        "negative_prefix_count": int(matched_negative_prefix_count),
        "prefix_rate": float(matched_prefix_count / total_prefix_count)
        if total_prefix_count
        else 0.0,
        "positive_rate": float(matched_positive_rate),
        "calibration_error": float(matched_calibration_error),
        "ece": float(matched_calibration_error),
        "brier_score": float(matched_brier),
        "trusted_state_count": int(len(trusted_states)),
        "trusted_state_min_count": int(min_count),
        "calibration_bins": int(calibration_bins),
    }


def _empty_matched_coverage_soft_metrics(
    *,
    soft_threshold: float,
    min_count: int,
    calibration_bins: int,
    failure_message: str,
) -> dict[str, float | int | str]:
    return {
        "available": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "threshold": float(soft_threshold),
        "prefix_count": 0,
        "positive_prefix_count": 0,
        "negative_prefix_count": 0,
        "prefix_rate": 0.0,
        "positive_rate": 0.0,
        "calibration_error": 0.0,
        "ece": 0.0,
        "brier_score": 0.0,
        "trusted_state_count": 0,
        "trusted_state_min_count": int(min_count),
        "calibration_bins": int(calibration_bins),
        "failure": failure_message,
    }


def _empty_discrete_metrics(
    *,
    soft_prefix_count: int,
    calibration_bins: int,
    trusted_state_min_count: int,
    failure_message: str,
) -> dict[str, float | int | str]:
    return {
        "available": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "detection_latency": 1.0,
        "trusted_detection_latency": 1.0,
        "alert_lead_time": 0.0,
        "threshold": 0.5,
        "dfa_state_count": 0,
        "prefix_count": int(soft_prefix_count),
        "prefix_positive_rate": 0.0,
        "calibration_prefix_count": 0,
        "calibration_prefix_positive_rate": 0.0,
        "state_risk_smoothing_alpha": 0.0,
        "legacy_reproduction": False,
        "state_risk_fit_split": "unavailable",
        "threshold_selection_split": "unavailable",
        "extraction_success_rate": 0.0,
        "trusted_state_count": 0,
        "trusted_prefix_count": 0,
        "trusted_positive_prefix_count": 0,
        "trusted_negative_prefix_count": 0,
        "trusted_prefix_rate": 0.0,
        "trusted_prefix_positive_rate": 0.0,
        "abstention_rate": 1.0,
        "trusted_state_auroc": 0.0,
        "trusted_state_auprc": 0.0,
        "calibration_error": 0.0,
        "ece": 0.0,
        "brier_score": 0.0,
        "trusted_state_min_count": int(trusted_state_min_count),
        "calibration_bins": int(calibration_bins),
        "positive_trace_count": 0,
        "negative_trace_count": 0,
        "unique_positive_trace_count": 0,
        "unique_negative_trace_count": 0,
        "ambiguous_trace_count": 0,
        "consistent_positive_trace_count": 0,
        "consistent_negative_trace_count": 0,
        "failure": failure_message,
    }


def evaluate_soft_differentiable_monitor(
    train_trajectories: list[TrajectoryRecord],
    eval_trajectories: list[TrajectoryRecord],
    encoder: BaseSegmentEncoder,
    symbolizer: GumbelEventSymbolizer,
    automaton: DifferentiableFiniteStateSurrogate | FlatPrefixRiskHead,
    horizon: int,
    device: torch.device,
    symbol_temperature: float,
    transition_temperature: float,
    representation_mode: str = "legacy",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    runtime_cache: RuntimeCache | None = None,
) -> dict:
    total_start = perf_counter()
    train_count, train_steps = _summarize_trajectory_steps(train_trajectories)
    eval_count, eval_steps = _summarize_trajectory_steps(eval_trajectories)
    print(
        "[soft-monitor] start paired soft validation: "
        f"train={train_count} trajectories/{train_steps} steps, "
        f"eval={eval_count} trajectories/{eval_steps} steps",
        flush=True,
    )
    train_start = perf_counter()
    train_scores, train_labels, _ = _score_soft_prefixes(
        trajectories=train_trajectories,
        encoder=encoder,
        symbolizer=symbolizer,
        automaton=automaton,
        horizon=horizon,
        device=device,
        symbol_temperature=symbol_temperature,
        transition_temperature=transition_temperature,
        progress_label="soft-monitor/train",
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
        runtime_cache=runtime_cache,
    )
    print(
        f"[soft-monitor/train] finished train-side soft scoring in "
        f"{perf_counter() - train_start:.2f}s",
        flush=True,
    )
    eval_start = perf_counter()
    eval_scores, eval_labels, eval_trajectory_scores = _score_soft_prefixes(
        trajectories=eval_trajectories,
        encoder=encoder,
        symbolizer=symbolizer,
        automaton=automaton,
        horizon=horizon,
        device=device,
        symbol_temperature=symbol_temperature,
        transition_temperature=transition_temperature,
        progress_label="soft-monitor/eval",
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
        runtime_cache=runtime_cache,
    )
    print(
        f"[soft-monitor/eval] finished eval-side soft scoring in "
        f"{perf_counter() - eval_start:.2f}s",
        flush=True,
    )
    threshold_start = perf_counter()
    threshold = _select_threshold(train_scores, train_labels)
    print(
        f"[soft-monitor] selected threshold={threshold:.4f} from train soft scores in "
        f"{perf_counter() - threshold_start:.2f}s",
        flush=True,
    )
    auroc, auprc = _compute_binary_metrics(eval_labels, eval_scores)
    latency = _compute_soft_detection_latency(
        trajectories=eval_trajectories,
        per_trajectory_scores=eval_trajectory_scores,
        threshold=threshold,
    )
    calibration_error = _compute_expected_calibration_error(eval_scores, eval_labels)
    brier_score = _compute_brier_score(eval_scores, eval_labels)
    print(
        "[soft-monitor] completed: "
        f"AUROC={auroc:.4f}, "
        f"AUPRC={auprc:.4f}, "
        f"latency={latency:.4f}, "
        f"prefixes={len(eval_labels)}, "
        f"total_time={perf_counter() - total_start:.2f}s",
        flush=True,
    )
    return {
        "metrics": {
            "auroc": auroc,
            "auprc": auprc,
            "detection_latency": float(latency),
            "alert_lead_time": float(1.0 - latency),
            "threshold": float(threshold),
            "soft_state_count": int(automaton.num_states),
            "prefix_count": int(len(eval_labels)),
            "prefix_positive_rate": float(np.mean(eval_labels)) if eval_labels else 0.0,
            "calibration_error": float(calibration_error),
            "ece": float(calibration_error),
            "brier_score": float(brier_score),
        },
        "labels": eval_labels,
        "scores": eval_scores,
        "trajectory_scores": eval_trajectory_scores,
    }


def compute_faithfulness_metrics(
    soft_scores: list[float],
    discrete_scores: list[float],
    discrete_threshold: float,
    soft_auroc: float,
    discrete_auroc: float,
) -> dict:
    if len(soft_scores) != len(discrete_scores):
        raise ValueError("Soft and discrete score lists must have the same length")
    if not soft_scores:
        return {
            "gap_auroc": float(abs(soft_auroc - discrete_auroc)),
            "soft_discrete_agreement": 0.0,
            "mean_score_l1": 0.0,
            "deployment_threshold": float(discrete_threshold),
        }
    agreement = np.mean(
        [
            int(soft_score >= discrete_threshold) == int(discrete_score >= discrete_threshold)
            for soft_score, discrete_score in zip(soft_scores, discrete_scores)
        ]
    )
    score_l1 = np.mean(
        [
            abs(soft_score - discrete_score)
            for soft_score, discrete_score in zip(soft_scores, discrete_scores)
        ]
    )
    return {
        "gap_auroc": float(abs(soft_auroc - discrete_auroc)),
        "soft_discrete_agreement": float(agreement),
        "mean_score_l1": float(score_l1),
        "deployment_threshold": float(discrete_threshold),
    }


def flatten_paired_monitor_metrics(payload: dict) -> dict[str, float | int | str]:
    soft_metrics = payload["soft_metrics"]
    matched_soft_metrics = payload.get("matched_soft_metrics", {})
    discrete_metrics = payload["discrete_metrics"]
    faithfulness = payload["faithfulness"]
    flattened = {
        "method": payload["method"],
        "hardening_strategy": str(payload.get("hardening_strategy", DEFAULT_HARDENING_STRATEGY)),
        "hardening_threshold": (
            None
            if payload.get("hardening_threshold") is None
            else float(payload["hardening_threshold"])
        ),
        "soft_auroc": float(soft_metrics["auroc"]),
        "soft_auprc": float(soft_metrics["auprc"]),
        "soft_detection_latency": float(soft_metrics["detection_latency"]),
        "soft_alert_lead_time": float(soft_metrics["alert_lead_time"]),
        "soft_threshold": float(soft_metrics["threshold"]),
        "soft_state_count": int(soft_metrics["soft_state_count"]),
        "soft_calibration_error": float(soft_metrics["calibration_error"]),
        "soft_brier_score": float(soft_metrics["brier_score"]),
        "matched_soft_available": float(matched_soft_metrics.get("available", 0.0)),
        "discrete_available": float(discrete_metrics.get("available", 1.0)),
        "discrete_auroc": float(discrete_metrics["auroc"]),
        "discrete_auprc": float(discrete_metrics["auprc"]),
        "discrete_detection_latency": float(discrete_metrics["detection_latency"]),
        "discrete_alert_lead_time": float(discrete_metrics["alert_lead_time"]),
        "discrete_threshold": float(discrete_metrics["threshold"]),
        "discrete_state_count": int(discrete_metrics["dfa_state_count"]),
        "discrete_trusted_state_auroc": float(discrete_metrics["trusted_state_auroc"]),
        "discrete_trusted_state_auprc": float(discrete_metrics["trusted_state_auprc"]),
        "discrete_abstention_rate": float(discrete_metrics["abstention_rate"]),
        "discrete_calibration_error": float(discrete_metrics["calibration_error"]),
        "discrete_brier_score": float(discrete_metrics["brier_score"]),
        "discrete_extraction_success_rate": float(discrete_metrics["extraction_success_rate"]),
        "faithfulness_gap_auroc": float(faithfulness["gap_auroc"]),
        "faithfulness_agreement": float(faithfulness["soft_discrete_agreement"]),
        "faithfulness_score_l1": float(faithfulness["mean_score_l1"]),
        "num_symbols": int(payload["num_symbols"]),
        "dfa_backend": str(payload["dfa_backend"]),
    }
    if matched_soft_metrics:
        flattened.update(
            {
                "matched_soft_auroc": float(matched_soft_metrics["auroc"]),
                "matched_soft_auprc": float(matched_soft_metrics["auprc"]),
                "matched_soft_threshold": float(matched_soft_metrics["threshold"]),
                "matched_soft_prefix_count": int(matched_soft_metrics["prefix_count"]),
                "matched_soft_prefix_rate": float(matched_soft_metrics["prefix_rate"]),
                "matched_soft_positive_rate": float(matched_soft_metrics["positive_rate"]),
                "matched_soft_calibration_error": float(
                    matched_soft_metrics["calibration_error"]
                ),
                "matched_soft_brier_score": float(matched_soft_metrics["brier_score"]),
                "matched_soft_trusted_state_count": int(
                    matched_soft_metrics["trusted_state_count"]
                ),
            }
        )
    return flattened


def evaluate_paired_differentiable_monitor(
    train_trajectories: list[TrajectoryRecord],
    eval_trajectories: list[TrajectoryRecord],
    encoder: BaseSegmentEncoder,
    symbolizer: GumbelEventSymbolizer,
    automaton: DifferentiableFiniteStateSurrogate | FlatPrefixRiskHead,
    horizon: int,
    num_symbols: int,
    device: torch.device,
    symbol_temperature: float,
    transition_temperature: float,
    dfa_backend: DfaBackendName = "aalpy-rpni",
    representation_mode: str = "legacy",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    trusted_state_min_count: int = 10,
    state_risk_smoothing_alpha: float = 5.0,
    calibration_bins: int = 10,
    cal_trajectories: list[TrajectoryRecord] | None = None,
    legacy_reproduction: bool = False,
    runtime_cache: RuntimeCache | None = None,
    hardening_strategy: HardeningStrategyName = DEFAULT_HARDENING_STRATEGY,
    hardening_threshold: float | None = None,
    allow_discrete_failure: bool = False,
) -> dict:
    cache = runtime_cache or RuntimeCache()
    paired_start = perf_counter()
    train_count, train_steps = _summarize_trajectory_steps(train_trajectories)
    eval_count, eval_steps = _summarize_trajectory_steps(eval_trajectories)
    print(
        "[paired-validation] start: "
        f"train={train_count} trajectories/{train_steps} steps, "
        f"eval={eval_count} trajectories/{eval_steps} steps, "
        f"dfa_backend={dfa_backend}",
        flush=True,
    )
    soft_start = perf_counter()
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
        runtime_cache=cache,
    )
    print(
        f"[paired-validation] soft monitor finished in {perf_counter() - soft_start:.2f}s",
        flush=True,
    )
    discrete_train_start = perf_counter()
    train_symbols = symbolize_trajectories(
        train_trajectories,
        encoder,
        symbolizer,
        device=device,
        progress_label="paired-discrete/train",
        deterministic=True,
        symbol_temperature=symbol_temperature,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
        runtime_cache=cache,
        hardening_strategy=hardening_strategy,
        hardening_threshold=hardening_threshold,
    )
    print(
        "[paired-validation] discrete train symbolization finished in "
        f"{perf_counter() - discrete_train_start:.2f}s",
        flush=True,
    )
    calibration_trajectories = cal_trajectories or train_trajectories
    if cal_trajectories is None:
        cal_symbols = train_symbols
    else:
        discrete_cal_start = perf_counter()
        cal_symbols = symbolize_trajectories(
            calibration_trajectories,
            encoder,
            symbolizer,
            device=device,
            progress_label="paired-discrete/cal",
            deterministic=True,
            symbol_temperature=symbol_temperature,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            runtime_cache=cache,
            hardening_strategy=hardening_strategy,
            hardening_threshold=hardening_threshold,
        )
        print(
            "[paired-validation] discrete cal symbolization finished in "
            f"{perf_counter() - discrete_cal_start:.2f}s",
            flush=True,
        )
    discrete_eval_start = perf_counter()
    eval_symbols = symbolize_trajectories(
        eval_trajectories,
        encoder,
        symbolizer,
        device=device,
        progress_label="paired-discrete/eval",
        deterministic=True,
        symbol_temperature=symbol_temperature,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
        runtime_cache=cache,
        hardening_strategy=hardening_strategy,
        hardening_threshold=hardening_threshold,
    )
    print(
        "[paired-validation] discrete eval symbolization finished in "
        f"{perf_counter() - discrete_eval_start:.2f}s",
        flush=True,
    )
    discrete_failure: str | None = None
    discrete_metrics_start = perf_counter()
    try:
        discrete_metrics, discrete_details = _evaluate_precomputed_symbol_sequences_with_details(
            train_trajectories=train_trajectories,
            val_trajectories=eval_trajectories,
            train_symbols=train_symbols,
            val_symbols=eval_symbols,
            cal_trajectories=calibration_trajectories,
            cal_symbols=cal_symbols,
            horizon=horizon,
            num_symbols=num_symbols,
            dfa_backend=dfa_backend,
            trusted_state_min_count=trusted_state_min_count,
            state_risk_smoothing_alpha=state_risk_smoothing_alpha,
            calibration_bins=calibration_bins,
            legacy_reproduction=legacy_reproduction,
            runtime_cache=cache,
        )
    except ValueError as error:
        if not allow_discrete_failure:
            raise
        discrete_failure = str(error)
        discrete_metrics = _empty_discrete_metrics(
            soft_prefix_count=soft_output["metrics"]["prefix_count"],
            calibration_bins=calibration_bins,
            trusted_state_min_count=trusted_state_min_count,
            failure_message=discrete_failure,
        )
        discrete_details = {
            "dfa": None,
            "state_risk": {},
            "state_support": {},
            "threshold": 0.5,
            "cal_labels": [],
            "cal_scores": [],
            "cal_states": [],
            "val_labels": [],
            "val_scores": [],
            "val_states": [],
        }
        matched_soft_metrics = _empty_matched_coverage_soft_metrics(
            soft_threshold=soft_output["metrics"]["threshold"],
            min_count=trusted_state_min_count,
            calibration_bins=calibration_bins,
            failure_message=discrete_failure,
        )
        faithfulness = {
            "available": 0.0,
            "gap_auroc": 0.0,
            "soft_discrete_agreement": 0.0,
            "mean_score_l1": 0.0,
            "deployment_threshold": 0.5,
            "failure": discrete_failure,
        }
        print(
            "[paired-validation] discrete monitor unavailable; "
            f"continuing with soft-only payload because allow_discrete_failure=True: "
            f"{discrete_failure}",
            flush=True,
        )
    else:
        print(
            "[paired-validation] discrete monitor metrics finished in "
            f"{perf_counter() - discrete_metrics_start:.2f}s",
            flush=True,
        )
        faithfulness = compute_faithfulness_metrics(
            soft_scores=soft_output["scores"],
            discrete_scores=discrete_details["val_scores"],
            discrete_threshold=discrete_metrics["threshold"],
            soft_auroc=soft_output["metrics"]["auroc"],
            discrete_auroc=discrete_metrics["auroc"],
        )
        faithfulness["available"] = 1.0
        matched_soft_metrics = _compute_matched_coverage_soft_metrics(
            soft_scores=soft_output["scores"],
            soft_labels=soft_output["labels"],
            soft_threshold=soft_output["metrics"]["threshold"],
            discrete_prefix_states=discrete_details["val_states"],
            discrete_state_support=discrete_details["state_support"],
            min_count=trusted_state_min_count,
            calibration_bins=calibration_bins,
        )
    payload = {
        "method": getattr(automaton, "method_name", "differentiable_automaton"),
        "dfa_backend": dfa_backend,
        "hardening_strategy": str(hardening_strategy),
        "hardening_threshold": None if hardening_threshold is None else float(hardening_threshold),
        "num_symbols": num_symbols,
        "legacy_reproduction": bool(legacy_reproduction),
        "soft_metrics": soft_output["metrics"],
        "matched_soft_metrics": matched_soft_metrics,
        "discrete_metrics": discrete_metrics,
        "faithfulness": faithfulness,
    }
    if discrete_failure is not None:
        payload["discrete_failure"] = discrete_failure
    payload["summary"] = flatten_paired_monitor_metrics(payload)
    print(
        "[paired-validation] completed: "
        f"soft_AUPRC={payload['soft_metrics']['auprc']:.4f}, "
        f"matched_soft_AUPRC={payload['matched_soft_metrics']['auprc']:.4f}, "
        f"discrete_AUPRC={payload['discrete_metrics']['auprc']:.4f}, "
        f"states={payload['discrete_metrics']['dfa_state_count']}, "
        f"total_time={perf_counter() - paired_start:.2f}s",
        flush=True,
    )
    return payload
