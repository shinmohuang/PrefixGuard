from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable

from monitor_symbolization.data.prefixes import (
    EXPLICIT_FUTURE_FAILURE_LABELS_KEY,
    PREFIX_LABEL_MASK_KEY,
    future_failure_labels_for_trajectory,
)
from monitor_symbolization.data.schema import StepRecord, TrajectoryRecord


@dataclass(frozen=True)
class ScrambledPrefixView:
    """A no-leakage scrambled prefix represented as indices into a source trajectory."""

    source_trajectory: TrajectoryRecord
    prefix_index: int
    shuffled_step_indices: tuple[int, ...]
    label: int
    runtime_cache_disabled: bool = False

    def __post_init__(self) -> None:
        if self.prefix_index < 1 or self.prefix_index > len(self.source_trajectory.steps):
            raise ValueError(
                f"prefix_index must be in [1, {len(self.source_trajectory.steps)}], "
                f"got {self.prefix_index}"
            )
        indices = tuple(int(index) for index in self.shuffled_step_indices)
        if len(indices) != self.prefix_index:
            raise ValueError(
                f"Scrambled prefix '{self.trajectory_id}' has {len(indices)} indices "
                f"for prefix length {self.prefix_index}"
            )
        if sorted(indices) != list(range(self.prefix_index)):
            raise ValueError(
                f"Scrambled prefix '{self.trajectory_id}' must be a permutation of "
                f"visible source steps 0..{self.prefix_index - 1}"
            )
        if int(self.label) not in (0, 1):
            raise ValueError(
                f"Scrambled prefix '{self.trajectory_id}' has non-binary label {self.label}"
            )
        object.__setattr__(self, "shuffled_step_indices", indices)
        object.__setattr__(self, "label", int(self.label))

    @property
    def trajectory_id(self) -> str:
        return f"{self.source_trajectory.trajectory_id}::prefix{self.prefix_index}"

    @property
    def task_id(self) -> str:
        return self.source_trajectory.task_id

    @property
    def final_success(self) -> bool:
        return self.source_trajectory.final_success

    @property
    def failure_bucket(self) -> str:
        return self.source_trajectory.failure_bucket

    @property
    def split(self) -> str:
        return self.source_trajectory.split

    @cached_property
    def steps(self) -> tuple[StepRecord, ...]:
        return tuple(
            self.source_trajectory.steps[index]
            for index in self.shuffled_step_indices
        )

    @cached_property
    def metadata(self) -> dict[str, Any]:
        labels = [0 for _ in range(self.prefix_index)]
        labels[-1] = self.label
        mask = [False for _ in range(self.prefix_index)]
        mask[-1] = True
        metadata = {
            **self.source_trajectory.metadata,
            "source_trajectory_id": self.source_trajectory.trajectory_id,
            "source_prefix_index": self.prefix_index,
            EXPLICIT_FUTURE_FAILURE_LABELS_KEY: labels,
            PREFIX_LABEL_MASK_KEY: mask,
        }
        if self.runtime_cache_disabled:
            metadata["disable_runtime_cache"] = True
        return metadata


class ScrambledIndexTrajectoryList(list[ScrambledPrefixView]):
    """Index-view scrambled prefixes with source-level encoder/cache controls."""

    disable_runtime_precompute = True

    def __init__(
        self,
        items: Iterable[ScrambledPrefixView] = (),
        *,
        source_trajectories: Iterable[TrajectoryRecord] = (),
    ) -> None:
        super().__init__(items)
        self.source_trajectories = list(source_trajectories)
        self.encoder_fit_trajectories = self.source_trajectories


def is_scrambled_prefix_view(trajectory: object) -> bool:
    return isinstance(trajectory, ScrambledPrefixView)


def trajectory_step_count(trajectory: TrajectoryRecord | ScrambledPrefixView) -> int:
    if is_scrambled_prefix_view(trajectory):
        return trajectory.prefix_index
    return len(trajectory.steps)


def build_scrambled_prefix_view(
    traj: TrajectoryRecord,
    *,
    prefix_index: int,
    label: int,
    seed: int,
    disable_runtime_cache: bool = False,
) -> ScrambledPrefixView:
    rng = random.Random(f"{seed}:{traj.trajectory_id}:{prefix_index}")
    shuffled_indices = list(range(prefix_index))
    rng.shuffle(shuffled_indices)
    return ScrambledPrefixView(
        source_trajectory=traj,
        prefix_index=prefix_index,
        shuffled_step_indices=tuple(shuffled_indices),
        label=label,
        runtime_cache_disabled=disable_runtime_cache,
    )


def build_scrambled_index_split(
    trajs: list[TrajectoryRecord],
    seed: int,
    *,
    horizon: int,
    disable_runtime_cache: bool = False,
) -> ScrambledIndexTrajectoryList:
    views = ScrambledIndexTrajectoryList(source_trajectories=trajs)
    for traj in trajs:
        labels = future_failure_labels_for_trajectory(traj, horizon=horizon)
        for prefix_index, label in enumerate(labels, start=1):
            views.append(
                build_scrambled_prefix_view(
                    traj,
                    prefix_index=prefix_index,
                    label=label,
                    seed=seed,
                    disable_runtime_cache=disable_runtime_cache,
                )
            )
    return views
