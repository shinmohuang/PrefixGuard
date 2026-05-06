from __future__ import annotations

from collections import Counter

from monitor_symbolization.data.schema import FutureSignature, PrefixRecord, TrajectoryRecord
from monitor_symbolization.data.serialization import serialize_step

EXPLICIT_FUTURE_FAILURE_LABELS_KEY = "explicit_future_failure_labels"
PREFIX_LABEL_MASK_KEY = "prefix_label_mask"


def _trajectory_step_count(trajectory: TrajectoryRecord) -> int:
    prefix_index = getattr(trajectory, "prefix_index", None)
    shuffled_indices = getattr(trajectory, "shuffled_step_indices", None)
    source_trajectory = getattr(trajectory, "source_trajectory", None)
    if (
        source_trajectory is not None
        and prefix_index is not None
        and shuffled_indices is not None
    ):
        return int(prefix_index)
    return len(trajectory.steps)


def bucket_remaining_steps(remaining_steps: int) -> str:
    if remaining_steps <= 0:
        return "0"
    if remaining_steps == 1:
        return "1"
    if remaining_steps <= 3:
        return "2-3"
    if remaining_steps <= 7:
        return "4-7"
    if remaining_steps <= 15:
        return "8-15"
    return "16+"


def build_prefix_dataset(
    trajectories: list[TrajectoryRecord],
    horizon: int,
) -> list[PrefixRecord]:
    prefixes: list[PrefixRecord] = []
    for trajectory in trajectories:
        serialized_steps = tuple(serialize_step(step) for step in trajectory.steps)
        full_length = len(serialized_steps)
        future_failure_labels = future_failure_labels_for_trajectory(
            trajectory,
            horizon=horizon,
        )
        future_signature_keys = future_signature_keys_for_trajectory(
            trajectory,
            horizon=horizon,
        )
        for prefix_index, (future_failure, signature_key) in enumerate(
            zip(future_failure_labels, future_signature_keys),
            start=1,
        ):
            signature = FutureSignature(
                terminal_label=signature_key[0],
                remaining_steps_bin=signature_key[1],
                failure_bucket=signature_key[2],
            )
            prefixes.append(
                PrefixRecord(
                    trajectory_id=trajectory.trajectory_id,
                    split=trajectory.split,
                    prefix_index=prefix_index,
                    serialized_steps=serialized_steps[:prefix_index],
                    future_signature=signature,
                    future_failure_label=future_failure,
                    final_success=trajectory.final_success,
                    full_length=full_length,
                )
            )
    return prefixes


def future_failure_labels_for_trajectory(
    trajectory: TrajectoryRecord,
    horizon: int,
) -> tuple[int, ...]:
    explicit_labels = trajectory.metadata.get(EXPLICIT_FUTURE_FAILURE_LABELS_KEY)
    if explicit_labels is not None:
        labels = tuple(int(label) for label in explicit_labels)
        step_count = _trajectory_step_count(trajectory)
        if len(labels) != step_count:
            raise ValueError(
                f"Trajectory '{trajectory.trajectory_id}' has "
                f"{EXPLICIT_FUTURE_FAILURE_LABELS_KEY} length {len(labels)} "
                f"but {step_count} steps"
            )
        if any(label not in (0, 1) for label in labels):
            raise ValueError(
                f"Trajectory '{trajectory.trajectory_id}' has non-binary "
                f"{EXPLICIT_FUTURE_FAILURE_LABELS_KEY}: {labels}"
            )
        return labels
    full_length = _trajectory_step_count(trajectory)
    return tuple(
        int((not trajectory.final_success) and (full_length - prefix_index) <= horizon)
        for prefix_index in range(1, full_length + 1)
    )


def prefix_label_mask_for_trajectory(trajectory: TrajectoryRecord) -> tuple[bool, ...]:
    explicit_mask = trajectory.metadata.get(PREFIX_LABEL_MASK_KEY)
    if explicit_mask is None:
        return tuple(True for _ in trajectory.steps)
    mask = tuple(bool(value) for value in explicit_mask)
    step_count = _trajectory_step_count(trajectory)
    if len(mask) != step_count:
        raise ValueError(
            f"Trajectory '{trajectory.trajectory_id}' has {PREFIX_LABEL_MASK_KEY} "
            f"length {len(mask)} but {step_count} steps"
        )
    if not any(mask):
        raise ValueError(
            f"Trajectory '{trajectory.trajectory_id}' has an empty active "
            f"{PREFIX_LABEL_MASK_KEY}"
        )
    return mask


def future_signature_keys_for_trajectory(
    trajectory: TrajectoryRecord,
    horizon: int,
) -> tuple[tuple[str, str, str], ...]:
    full_length = _trajectory_step_count(trajectory)
    terminal_label = "SUCCESS" if trajectory.final_success else "FAILURE"
    signatures: list[tuple[str, str, str]] = []
    for prefix_index in range(1, full_length + 1):
        remaining_steps = full_length - prefix_index
        future_failure = int((not trajectory.final_success) and remaining_steps <= horizon)
        signatures.append(
            (
                terminal_label,
                bucket_remaining_steps(remaining_steps),
                trajectory.failure_bucket if future_failure else "NONE",
            )
        )
    return tuple(signatures)


def summarize_prefix_dataset(trajectories: list[TrajectoryRecord], horizon: int = 3) -> dict:
    prefixes = build_prefix_dataset(trajectories, horizon=horizon)
    split_counts = Counter(trajectory.split for trajectory in trajectories)
    prefix_counts = Counter(prefix.split for prefix in prefixes)
    future_counts = Counter(prefix.future_signature.as_key() for prefix in prefixes)
    return {
        "num_trajectories": len(trajectories),
        "num_prefixes": len(prefixes),
        "split_counts": dict(split_counts),
        "prefix_counts": dict(prefix_counts),
        "num_future_signatures": len(future_counts),
        "future_signature_topk": [
            {"signature": list(signature), "count": count}
            for signature, count in future_counts.most_common(10)
        ],
    }
