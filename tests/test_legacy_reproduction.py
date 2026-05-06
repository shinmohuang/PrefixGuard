from __future__ import annotations

import random

from monitor_symbolization.data.prefixes import build_prefix_dataset
from monitor_symbolization.data.schema import StepRecord, TrajectoryRecord
from monitor_symbolization.monitor import evaluation


def _trajectory(
    trajectory_id: str,
    split: str,
    final_success: bool,
    num_steps: int,
) -> TrajectoryRecord:
    steps = tuple(
        StepRecord(
            context=f"ctx-{trajectory_id}",
            action_text=f"step-{index}",
            tool_name="click",
            result_text="ok",
            status="ok",
        )
        for index in range(num_steps)
    )
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        task_id=trajectory_id,
        final_success=final_success,
        failure_bucket="NONE" if final_success else "TASK_FAILURE",
        steps=steps,
        split=split,
    )


def _symbols(trajectories: list[TrajectoryRecord]) -> dict[str, list[int]]:
    return {
        trajectory.trajectory_id: ([0] * len(trajectory.steps) if trajectory.final_success else [1] * len(trajectory.steps))
        for trajectory in trajectories
    }


def test_legacy_reproduction_uses_train_for_state_risk_and_eval_for_threshold(
    monkeypatch,
) -> None:
    train_trajectories = [
        _trajectory("train-success", "train", True, 2),
        _trajectory("train-failure", "train", False, 2),
    ]
    cal_trajectories = [
        _trajectory("cal-success", "cal", True, 3),
        _trajectory("cal-failure", "cal", False, 3),
    ]
    val_trajectories = [
        _trajectory("val-success", "val", True, 4),
        _trajectory("val-failure", "val", False, 4),
    ]

    counts = {
        "train": len(build_prefix_dataset(train_trajectories, horizon=1)),
        "cal": len(build_prefix_dataset(cal_trajectories, horizon=1)),
        "val": len(build_prefix_dataset(val_trajectories, horizon=1)),
    }
    recorded: dict[str, list[int] | int | float] = {}

    def fake_fit_state_risk(dfa, prefix_sequences, prefix_labels, smoothing_alpha=5.0):
        recorded["fit_label_count"] = len(prefix_labels)
        recorded["fit_smoothing_alpha"] = float(smoothing_alpha)
        return (
            {state: 0.5 for state in dfa.transitions},
            {state: 10 for state in dfa.transitions},
            0.5,
        )

    def fake_select_threshold(scores, labels):
        recorded.setdefault("threshold_label_counts", []).append(len(labels))
        return 0.5

    monkeypatch.setattr(evaluation, "_fit_state_risk", fake_fit_state_risk)
    monkeypatch.setattr(evaluation, "_select_threshold", fake_select_threshold)

    protocol_metrics = evaluation.evaluate_precomputed_symbol_sequences(
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        train_symbols=_symbols(train_trajectories),
        val_symbols=_symbols(val_trajectories),
        cal_trajectories=cal_trajectories,
        cal_symbols=_symbols(cal_trajectories),
        horizon=1,
        num_symbols=2,
        trusted_state_min_count=1,
        legacy_reproduction=False,
    )
    assert recorded["fit_label_count"] == counts["cal"]
    assert recorded["fit_smoothing_alpha"] == 5.0
    assert recorded["threshold_label_counts"] == [counts["cal"]]
    assert protocol_metrics["state_risk_fit_split"] == "cal"
    assert protocol_metrics["threshold_selection_split"] == "cal"

    recorded.clear()
    legacy_metrics = evaluation.evaluate_precomputed_symbol_sequences(
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        train_symbols=_symbols(train_trajectories),
        val_symbols=_symbols(val_trajectories),
        cal_trajectories=cal_trajectories,
        cal_symbols=_symbols(cal_trajectories),
        horizon=1,
        num_symbols=2,
        trusted_state_min_count=1,
        legacy_reproduction=True,
    )
    assert recorded["fit_label_count"] == counts["train"]
    assert recorded["fit_smoothing_alpha"] == 1.0
    assert recorded["threshold_label_counts"] == [counts["val"]]
    assert legacy_metrics["state_risk_fit_split"] == "train"
    assert legacy_metrics["threshold_selection_split"] == "eval"
    assert legacy_metrics["legacy_reproduction"] is True


def _legacy_select_threshold(scores: list[float], labels: list[int]) -> float:
    if not scores:
        return 0.5
    candidates = sorted(set(scores))
    best_threshold = candidates[0]
    best_f1 = -1.0
    for threshold in candidates:
        predictions = [int(score >= threshold) for score in scores]
        tp = sum(int(pred == 1 and label == 1) for pred, label in zip(predictions, labels))
        fp = sum(int(pred == 1 and label == 0) for pred, label in zip(predictions, labels))
        fn = sum(int(pred == 0 and label == 1) for pred, label in zip(predictions, labels))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return float(best_threshold)


def test_select_threshold_matches_legacy_bruteforce() -> None:
    explicit_cases = [
        ([], []),
        ([0.1], [0]),
        ([0.1], [1]),
        ([0.2, 0.2, 0.8, 0.8], [0, 1, 0, 1]),
        ([0.3, 0.3, 0.3], [0, 1, 1]),
        ([0.1, 0.4, 0.4, 0.9], [0, 1, 0, 1]),
        ([0.9, 0.8, 0.7, 0.1], [1, 0, 1, 0]),
    ]
    rng = random.Random(0)
    random_cases = []
    for _ in range(100):
        length = rng.randint(1, 50)
        scores = [round(rng.random(), 3) for _ in range(length)]
        labels = [rng.randint(0, 1) for _ in range(length)]
        random_cases.append((scores, labels))

    for scores, labels in explicit_cases + random_cases:
        assert evaluation._select_threshold(scores, labels) == _legacy_select_threshold(scores, labels)
