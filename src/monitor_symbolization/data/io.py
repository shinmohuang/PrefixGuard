from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from monitor_symbolization.data.schema import StepRecord, TrajectoryRecord
from monitor_symbolization.data.skillsbench_manifest import (
    load_skillsbench_records_from_split_manifest,
    looks_like_skillsbench_split_manifest_row,
)
from monitor_symbolization.data.terminalbench import (
    load_terminalbench_records_from_split_manifest,
    looks_like_terminalbench_split_manifest_row,
)


def _require(obj: dict, key: str):
    if key not in obj:
        raise ValueError(f"Missing required field '{key}' in object: {obj}")
    return obj[key]


def _parse_step(obj: dict) -> StepRecord:
    return StepRecord(
        context=str(_require(obj, "context")),
        action_text=str(_require(obj, "action_text")),
        tool_name=None if obj.get("tool_name") in (None, "") else str(obj["tool_name"]),
        tool_args=dict(obj.get("tool_args", {})),
        result_text=str(obj.get("result_text", "")),
        status=None if obj.get("status") in (None, "") else str(obj["status"]),
        source_raw_text=None
        if obj.get("source_raw_text") in (None, "")
        else str(obj["source_raw_text"]),
    )


def _parse_trajectory(obj: dict) -> TrajectoryRecord:
    steps = tuple(_parse_step(step) for step in _require(obj, "steps"))
    if not steps:
        raise ValueError(f"Trajectory '{obj.get('trajectory_id')}' has no steps")
    return TrajectoryRecord(
        trajectory_id=str(_require(obj, "trajectory_id")),
        task_id=str(_require(obj, "task_id")),
        final_success=bool(_require(obj, "final_success")),
        failure_bucket=str(obj.get("failure_bucket", "NONE")),
        steps=steps,
        split=str(obj.get("split", "train")),
        metadata=dict(obj.get("metadata", {})),
    )


def load_trajectories(path: str | Path) -> list[TrajectoryRecord]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    trajectories: list[TrajectoryRecord] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if isinstance(obj, dict) and looks_like_skillsbench_split_manifest_row(obj):
                records, _summary = load_skillsbench_records_from_split_manifest(dataset_path)
                return [_parse_trajectory(record) for record in records]
            if isinstance(obj, dict) and looks_like_terminalbench_split_manifest_row(obj):
                records, _summary = load_terminalbench_records_from_split_manifest(dataset_path)
                return [_parse_trajectory(record) for record in records]
            trajectories.append(_parse_trajectory(obj))
    if not trajectories:
        raise ValueError(f"No trajectories found in dataset: {dataset_path}")
    return trajectories


def trajectories_for_split(
    trajectories: list[TrajectoryRecord],
    split_name: str,
    *,
    role: str | None = None,
) -> list[TrajectoryRecord]:
    selected = [trajectory for trajectory in trajectories if trajectory.split == split_name]
    if selected:
        return selected
    available = sorted({trajectory.split for trajectory in trajectories})
    label = role or "split"
    raise ValueError(
        f"No trajectories found for {label} '{split_name}'. "
        f"Available splits: {available}"
    )


@dataclass(frozen=True)
class DerivedTrainCalibrationSplit:
    fit_trajectories: list[TrajectoryRecord]
    cal_trajectories: list[TrajectoryRecord]
    summary: dict


def _task_bucket(task_id: str, seed: int) -> float:
    payload = f"{seed}:{task_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") / 2**64


def derive_train_fit_cal_split(
    trajectories: list[TrajectoryRecord],
    *,
    outer_train_split: str = "train",
    fit_ratio: float = 0.8,
    cal_ratio: float = 0.2,
    seed: int = 1,
) -> DerivedTrainCalibrationSplit:
    if fit_ratio <= 0.0 or cal_ratio <= 0.0:
        raise ValueError(
            f"fit_ratio and cal_ratio must both be positive, got {fit_ratio} and {cal_ratio}"
        )
    total = fit_ratio + cal_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"fit_ratio + cal_ratio must sum to 1.0, got {total}")

    outer_train = trajectories_for_split(
        trajectories,
        outer_train_split,
        role="outer train split",
    )
    task_to_inner_split: dict[str, str] = {}
    fit_trajectories: list[TrajectoryRecord] = []
    cal_trajectories: list[TrajectoryRecord] = []
    task_counts = {"fit": 0, "cal": 0}

    for trajectory in outer_train:
        inner_split = task_to_inner_split.get(trajectory.task_id)
        if inner_split is None:
            bucket = _task_bucket(trajectory.task_id, seed=seed)
            inner_split = "fit" if bucket < fit_ratio else "cal"
            task_to_inner_split[trajectory.task_id] = inner_split
            task_counts[inner_split] += 1
        if inner_split == "fit":
            fit_trajectories.append(trajectory)
        else:
            cal_trajectories.append(trajectory)

    if not fit_trajectories or not cal_trajectories:
        raise ValueError(
            "Derived train-internal fit/cal split is empty for one side; "
            f"outer_train_split={outer_train_split}, fit_count={len(fit_trajectories)}, "
            f"cal_count={len(cal_trajectories)}, seed={seed}"
        )

    summary = {
        "mode": "train_internal_fit_cal",
        "outer_train_split": outer_train_split,
        "seed": int(seed),
        "ratios": {
            "fit": float(fit_ratio),
            "cal": float(cal_ratio),
        },
        "task_counts": {
            "fit": int(task_counts["fit"]),
            "cal": int(task_counts["cal"]),
        },
        "trajectory_counts": {
            "fit": int(len(fit_trajectories)),
            "cal": int(len(cal_trajectories)),
        },
        "num_unique_tasks": int(len(task_to_inner_split)),
    }
    return DerivedTrainCalibrationSplit(
        fit_trajectories=fit_trajectories,
        cal_trajectories=cal_trajectories,
        summary=summary,
    )


def resolve_fit_cal_splits(
    trajectories: list[TrajectoryRecord],
    *,
    fit_split: str,
    cal_split: str,
    derive_train_fit_cal: bool,
    train_fit_ratio: float = 0.8,
    train_cal_ratio: float = 0.2,
    protocol_split_seed: int = 1,
) -> tuple[list[TrajectoryRecord], list[TrajectoryRecord], dict]:
    if derive_train_fit_cal and fit_split == cal_split:
        derived = derive_train_fit_cal_split(
            trajectories,
            outer_train_split=fit_split,
            fit_ratio=train_fit_ratio,
            cal_ratio=train_cal_ratio,
            seed=protocol_split_seed,
        )
        return (
            derived.fit_trajectories,
            derived.cal_trajectories,
            derived.summary,
        )

    fit_trajectories = trajectories_for_split(trajectories, fit_split, role="fit split")
    cal_trajectories = trajectories_for_split(trajectories, cal_split, role="cal split")
    summary = {
        "mode": "explicit_dataset_splits",
        "fit_split": fit_split,
        "cal_split": cal_split,
        "trajectory_counts": {
            "fit": int(len(fit_trajectories)),
            "cal": int(len(cal_trajectories)),
        },
    }
    return fit_trajectories, cal_trajectories, summary
