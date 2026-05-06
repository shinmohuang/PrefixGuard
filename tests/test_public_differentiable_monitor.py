from __future__ import annotations

import torch

from monitor_symbolization.data.schema import StepRecord, TrajectoryRecord
from monitor_symbolization.models.differentiable_automaton import (
    DifferentiableFiniteStateSurrogate,
)
from monitor_symbolization.monitor.evaluation import _compute_soft_detection_latency


def _trajectory(
    trajectory_id: str,
    final_success: bool,
    length: int,
) -> TrajectoryRecord:
    steps = tuple(
        StepRecord(
            context="ctx",
            action_text=f"step-{index}",
            tool_name="click",
            result_text="ok",
            status="ok",
        )
        for index in range(length)
    )
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        task_id=trajectory_id,
        final_success=final_success,
        failure_bucket="NONE" if final_success else "TASK_FAILURE",
        steps=steps,
        split="test",
    )


def test_differentiable_automaton_state_probs_stay_on_simplex() -> None:
    automaton = DifferentiableFiniteStateSurrogate(num_symbols=3, num_states=4)
    symbol_probs = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ],
        dtype=torch.float32,
    )

    output = automaton(symbol_probs, transition_temperature=0.5)

    assert output.state_probs.shape == (3, 4)
    assert torch.allclose(output.state_probs.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert output.risk_scores.shape == (3,)


def test_soft_detection_latency_uses_first_alert_position_on_failures() -> None:
    trajectories = [
        _trajectory("success-clean", final_success=True, length=3),
        _trajectory("fail-early", final_success=False, length=3),
        _trajectory("fail-late", final_success=False, length=3),
    ]
    scores = {
        "success-clean": [0.1, 0.1, 0.1],
        "fail-early": [0.9, 0.1, 0.1],
        "fail-late": [0.1, 0.1, 0.9],
    }

    latency = _compute_soft_detection_latency(
        trajectories=trajectories,
        per_trajectory_scores=scores,
        threshold=0.5,
    )

    assert latency == 2 / 3
