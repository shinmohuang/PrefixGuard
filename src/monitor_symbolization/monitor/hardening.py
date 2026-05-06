from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch


HardeningStrategyName = Literal["argmax", "sticky-margin", "sticky-confidence"]
DEFAULT_HARDENING_STRATEGY: HardeningStrategyName = "argmax"


def available_hardening_strategies() -> tuple[HardeningStrategyName, ...]:
    return ("argmax", "sticky-margin", "sticky-confidence")


@dataclass(frozen=True)
class HardeningResult:
    hard_ids: list[int]
    override_count: int
    override_rate: float
    switch_count: int
    switch_rate: float
    mean_confidence: float
    mean_margin: float

    def to_summary(self) -> dict[str, float | int | list[int]]:
        return asdict(self)


def _as_numpy(probabilities: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(probabilities, torch.Tensor):
        return probabilities.detach().cpu().numpy()
    return np.asarray(probabilities, dtype=np.float64)


def harden_symbol_probabilities(
    probabilities: np.ndarray | torch.Tensor,
    *,
    strategy: HardeningStrategyName = "argmax",
    threshold: float | None = None,
) -> HardeningResult:
    probs = _as_numpy(probabilities)
    if probs.ndim != 2:
        raise ValueError(f"Expected [T, K] probabilities, got shape {tuple(probs.shape)}")
    if probs.shape[0] == 0:
        return HardeningResult(
            hard_ids=[],
            override_count=0,
            override_rate=0.0,
            switch_count=0,
            switch_rate=0.0,
            mean_confidence=0.0,
            mean_margin=0.0,
        )

    top_ids = probs.argmax(axis=-1).astype(int)
    top_values = probs[np.arange(probs.shape[0]), top_ids]
    sorted_probs = np.sort(probs, axis=-1)
    if probs.shape[1] > 1:
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    else:
        margins = sorted_probs[:, -1]

    if strategy == "argmax":
        hard_ids = top_ids.tolist()
    else:
        if threshold is None:
            raise ValueError(f"strategy '{strategy}' requires a threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must lie in [0, 1], got {threshold}")
        hard_ids: list[int] = []
        for index, candidate in enumerate(top_ids.tolist()):
            if index == 0:
                hard_ids.append(candidate)
                continue
            if strategy == "sticky-margin" and float(margins[index]) < threshold:
                hard_ids.append(hard_ids[-1])
                continue
            if strategy == "sticky-confidence" and float(top_values[index]) < threshold:
                hard_ids.append(hard_ids[-1])
                continue
            hard_ids.append(candidate)

    override_count = int(sum(int(hard != top) for hard, top in zip(hard_ids, top_ids.tolist())))
    switch_count = int(
        sum(int(current != previous) for previous, current in zip(hard_ids, hard_ids[1:]))
    )
    total_steps = len(hard_ids)
    return HardeningResult(
        hard_ids=hard_ids,
        override_count=override_count,
        override_rate=float(override_count / total_steps),
        switch_count=switch_count,
        switch_rate=float(switch_count / max(total_steps - 1, 1)),
        mean_confidence=float(np.mean(top_values)),
        mean_margin=float(np.mean(margins)),
    )
