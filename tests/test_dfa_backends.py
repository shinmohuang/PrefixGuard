from __future__ import annotations

from monitor_symbolization.data.schema import StepRecord, TrajectoryRecord
from monitor_symbolization.monitor.backends import adapt_traces_for_aalpy, fit_dfa_with_backend
from monitor_symbolization.monitor.evaluation import (
    compare_dfa_backends_on_symbol_sequences,
    evaluate_precomputed_symbol_sequences,
)


def _accepts(dfa, sequence: list[int]) -> bool:
    state = dfa.start_state
    for symbol in sequence:
        state = dfa.transition(state, symbol)
    return state in dfa.accepting_states


def _trajectory(trajectory_id: str, final_success: bool, split: str) -> TrajectoryRecord:
    steps = (
        StepRecord(context="ctx", action_text="a0", tool_name="click", result_text="ok", status="ok"),
        StepRecord(context="ctx", action_text="a1", tool_name="type", result_text="ok", status="ok"),
    )
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        task_id=trajectory_id,
        final_success=final_success,
        failure_bucket="NONE" if final_success else "TASK_FAILURE",
        steps=steps,
        split=split,
    )


def test_adapt_traces_for_aalpy_preserves_trace_order_and_labels() -> None:
    positive_traces = [[0, 1], [0, 1, 1]]
    negative_traces = [[1], [1, 0]]

    adapted = adapt_traces_for_aalpy(
        positive_traces=positive_traces,
        negative_traces=negative_traces,
    )

    assert adapted == [
        ((0, 1), True),
        ((0, 1, 1), True),
        ((1,), False),
        ((1, 0), False),
    ]


def test_legacy_and_aalpy_backends_conform_to_training_traces() -> None:
    positive_traces = [[0, 1], [0, 1, 1]]
    negative_traces = [[1], [1, 0]]

    for backend in ("legacy", "aalpy", "aalpy-edsm", "aalpy-rpni"):
        dfa = fit_dfa_with_backend(
            positive_traces=positive_traces,
            negative_traces=negative_traces,
            alphabet_size=3,
            backend=backend,
        )
        assert dfa.state_count >= 2
        assert all(_accepts(dfa, trace) for trace in positive_traces)
        assert all(not _accepts(dfa, trace) for trace in negative_traces)


def test_explicit_aalpy_backend_matches_default_evaluation() -> None:
    train_trajectories = [
        _trajectory("train-success", final_success=True, split="train"),
        _trajectory("train-failure", final_success=False, split="train"),
    ]
    val_trajectories = [
        _trajectory("val-success", final_success=True, split="val"),
        _trajectory("val-failure", final_success=False, split="val"),
    ]
    train_symbols = {
        "train-success": [0, 0],
        "train-failure": [1, 1],
    }
    val_symbols = {
        "val-success": [0, 0],
        "val-failure": [1, 1],
    }

    default_metrics = evaluate_precomputed_symbol_sequences(
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        train_symbols=train_symbols,
        val_symbols=val_symbols,
        horizon=1,
        num_symbols=2,
    )
    aalpy_metrics = evaluate_precomputed_symbol_sequences(
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        train_symbols=train_symbols,
        val_symbols=val_symbols,
        horizon=1,
        num_symbols=2,
        dfa_backend="aalpy",
    )

    assert default_metrics == aalpy_metrics


def test_evaluation_reports_trusted_state_and_calibration_metrics() -> None:
    train_trajectories = [
        _trajectory("train-success", final_success=True, split="train"),
        _trajectory("train-failure", final_success=False, split="train"),
    ]
    val_trajectories = [
        _trajectory("val-success", final_success=True, split="val"),
        _trajectory("val-failure", final_success=False, split="val"),
    ]
    train_symbols = {
        "train-success": [0, 0],
        "train-failure": [1, 1],
    }
    val_symbols = {
        "val-success": [0, 0],
        "val-failure": [1, 1],
    }

    metrics = evaluate_precomputed_symbol_sequences(
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        train_symbols=train_symbols,
        val_symbols=val_symbols,
        horizon=1,
        num_symbols=2,
        trusted_state_min_count=1,
        state_risk_smoothing_alpha=5.0,
        calibration_bins=5,
    )

    assert metrics["trusted_state_auroc"] == 1.0
    assert metrics["trusted_state_auprc"] == 1.0
    assert metrics["abstention_rate"] == 0.0
    assert metrics["alert_lead_time"] == 0.5
    assert metrics["trusted_detection_latency"] == 0.5
    assert metrics["extraction_success_rate"] == 1.0
    assert metrics["trusted_state_min_count"] == 1
    assert metrics["calibration_bins"] == 5
    assert 0.0 <= metrics["calibration_error"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_trusted_state_metrics_can_fully_abstain_when_support_is_too_low() -> None:
    train_trajectories = [
        _trajectory("train-success", final_success=True, split="train"),
        _trajectory("train-failure", final_success=False, split="train"),
    ]
    val_trajectories = [
        _trajectory("val-success", final_success=True, split="val"),
        _trajectory("val-failure", final_success=False, split="val"),
    ]
    train_symbols = {
        "train-success": [0, 0],
        "train-failure": [1, 1],
    }
    val_symbols = {
        "val-success": [0, 0],
        "val-failure": [1, 1],
    }

    metrics = evaluate_precomputed_symbol_sequences(
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        train_symbols=train_symbols,
        val_symbols=val_symbols,
        horizon=1,
        num_symbols=2,
        trusted_state_min_count=10,
    )

    assert metrics["trusted_prefix_count"] == 0
    assert metrics["abstention_rate"] == 1.0
    assert metrics["trusted_state_auroc"] == 0.5
    assert metrics["trusted_state_auprc"] == 0.0
    assert metrics["alert_lead_time"] == 0.0


def test_calibration_split_is_used_for_threshold_and_support_statistics() -> None:
    train_trajectories = [
        _trajectory("fit-success", final_success=True, split="fit"),
        _trajectory("fit-failure", final_success=False, split="fit"),
    ]
    cal_trajectories = [
        _trajectory("cal-success", final_success=True, split="cal"),
    ]
    val_trajectories = [
        _trajectory("val-success", final_success=True, split="val"),
        _trajectory("val-failure", final_success=False, split="val"),
    ]
    fit_symbols = {
        "fit-success": [0, 0],
        "fit-failure": [1, 1],
    }
    cal_symbols = {
        "cal-success": [0, 0],
    }
    val_symbols = {
        "val-success": [0, 0],
        "val-failure": [1, 1],
    }

    metrics = evaluate_precomputed_symbol_sequences(
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        train_symbols=fit_symbols,
        val_symbols=val_symbols,
        cal_trajectories=cal_trajectories,
        cal_symbols=cal_symbols,
        horizon=1,
        num_symbols=2,
        trusted_state_min_count=1,
    )

    assert metrics["calibration_prefix_count"] == 2
    assert metrics["calibration_prefix_positive_rate"] == 0.0
    assert metrics["prefix_count"] == 4
    assert metrics["prefix_positive_rate"] == 0.5


def test_compare_dfa_backends_reports_metrics_and_runtime() -> None:
    train_trajectories = [
        _trajectory("train-success", final_success=True, split="train"),
        _trajectory("train-failure", final_success=False, split="train"),
    ]
    val_trajectories = [
        _trajectory("val-success", final_success=True, split="val"),
        _trajectory("val-failure", final_success=False, split="val"),
    ]
    train_symbols = {
        "train-success": [0, 0],
        "train-failure": [1, 1],
    }
    val_symbols = {
        "val-success": [0, 0],
        "val-failure": [1, 1],
    }

    comparison = compare_dfa_backends_on_symbol_sequences(
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        train_symbols=train_symbols,
        val_symbols=val_symbols,
        horizon=1,
        num_symbols=2,
    )

    assert set(comparison["backends"]) == {"legacy", "aalpy-edsm", "aalpy-rpni"}
    for backend in ("legacy", "aalpy-edsm", "aalpy-rpni"):
        assert comparison["backends"][backend]["induction_seconds"] >= 0.0
        assert "dfa_state_count" in comparison["backends"][backend]["metrics"]


def test_aalpy_alias_matches_aalpy_edsm_backend() -> None:
    positive_traces = [[0, 1], [0, 1, 1]]
    negative_traces = [[1], [1, 0]]

    alias_dfa = fit_dfa_with_backend(
        positive_traces=positive_traces,
        negative_traces=negative_traces,
        alphabet_size=3,
        backend="aalpy",
    )
    edsm_dfa = fit_dfa_with_backend(
        positive_traces=positive_traces,
        negative_traces=negative_traces,
        alphabet_size=3,
        backend="aalpy-edsm",
    )

    assert alias_dfa.state_count == edsm_dfa.state_count
    assert alias_dfa.transitions == edsm_dfa.transitions
