from __future__ import annotations

from collections import OrderedDict

from monitor_symbolization.data.schema import TrajectoryRecord


def build_heuristic_symbol_sequences(
    trajectories: list[TrajectoryRecord],
) -> tuple[dict[str, list[int]], dict[str, int]]:
    vocabulary: "OrderedDict[str, int]" = OrderedDict()
    symbol_sequences: dict[str, list[int]] = {}
    for trajectory in trajectories:
        sequence = []
        for step in trajectory.steps:
            tool = step.tool_name or "NO_TOOL"
            status = step.status or "NO_STATUS"
            symbol_key = f"{tool}::{status}"
            if symbol_key not in vocabulary:
                vocabulary[symbol_key] = len(vocabulary)
            sequence.append(vocabulary[symbol_key])
        symbol_sequences[trajectory.trajectory_id] = sequence
    return symbol_sequences, dict(vocabulary)
