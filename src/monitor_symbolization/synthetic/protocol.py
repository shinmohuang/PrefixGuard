from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from monitor_symbolization.data.schema import StepRecord, TrajectoryRecord


@dataclass(frozen=True)
class SyntheticProtocolSpec:
    train_size: int = 24
    val_size: int = 12
    test_size: int = 12
    min_steps: int = 3
    max_steps: int = 6
    label_noise: float = 0.0
    lexical_variation: float = 0.2
    symbol_corruption: float = 0.0
    seed: int = 13


@dataclass(frozen=True)
class SyntheticProtocolDataset:
    trajectories: list[TrajectoryRecord]
    ground_truth_symbols: dict[str, list[int]]
    spec: SyntheticProtocolSpec


_LEXICAL_VARIANTS = {
    0: [
        ("inspect portal", "open", "page stable"),
        ("open dashboard", "browse", "healthy response"),
        ("check workspace", "inspect", "ready state"),
    ],
    1: [
        ("fill credential form", "type", "pending auth"),
        ("submit login", "click", "auth in progress"),
        ("confirm identity", "submit", "credential accepted"),
    ],
    2: [
        ("query item catalog", "search", "result list ready"),
        ("filter records", "search", "filtered list"),
        ("inspect candidate rows", "inspect", "records visible"),
    ],
    3: [
        ("commit purchase", "click", "task complete"),
        ("finalize workflow", "submit", "success recorded"),
        ("close transaction", "click", "execution finished"),
    ],
    4: [
        ("observe timeout", "wait", "page stalled"),
        ("retry failed action", "retry", "still failing"),
        ("emit error report", "report", "fatal failure"),
    ],
}


def _sample_variant(symbol_id: int, rng: random.Random, lexical_variation: float) -> tuple[str, str, str]:
    variants = _LEXICAL_VARIANTS[symbol_id]
    if lexical_variation <= 0:
        return variants[0]
    return variants[rng.randrange(len(variants))]


def _maybe_corrupt_symbol(symbol_id: int, rng: random.Random, corruption_rate: float) -> int:
    if corruption_rate <= 0 or rng.random() >= corruption_rate:
        return symbol_id
    alternatives = [candidate for candidate in _LEXICAL_VARIANTS if candidate != symbol_id]
    return rng.choice(alternatives)


def _success_symbols(length: int) -> list[int]:
    core = [0, 1, 2, 3]
    return core[: max(2, min(length, len(core)))]


def _failure_symbols(length: int) -> list[int]:
    if length <= 3:
        return [0, 1, 4][:length]
    prefix = [0, 1, 2]
    return prefix + [4] * max(length - len(prefix), 0)


def _materialize_steps(
    sequence: list[int],
    rng: random.Random,
    lexical_variation: float,
) -> tuple[StepRecord, ...]:
    steps = []
    for index, symbol_id in enumerate(sequence, start=1):
        action_text, tool_name, result_text = _sample_variant(symbol_id, rng, lexical_variation)
        steps.append(
            StepRecord(
                context=f"synthetic protocol prefix {index}",
                action_text=action_text,
                tool_name=tool_name,
                result_text=result_text,
                status="ok" if symbol_id != 4 else "error",
            )
        )
    return tuple(steps)


def generate_synthetic_protocol_dataset(
    spec: SyntheticProtocolSpec,
) -> SyntheticProtocolDataset:
    rng = random.Random(spec.seed)
    trajectories: list[TrajectoryRecord] = []
    ground_truth_symbols: dict[str, list[int]] = {}
    split_sizes = {
        "train": spec.train_size,
        "val": spec.val_size,
        "test": spec.test_size,
    }

    for split, size in split_sizes.items():
        for item_index in range(size):
            base_length = rng.randint(spec.min_steps, spec.max_steps)
            is_success = (item_index % 2 == 0)
            logical_symbols = (
                _success_symbols(base_length)
                if is_success
                else _failure_symbols(base_length)
            )
            if spec.label_noise > 0 and rng.random() < spec.label_noise:
                is_success = not is_success
            observed_symbols = [
                _maybe_corrupt_symbol(symbol_id, rng, spec.symbol_corruption)
                for symbol_id in logical_symbols
            ]
            trajectory_id = f"{split}-{item_index:03d}"
            trajectories.append(
                TrajectoryRecord(
                    trajectory_id=trajectory_id,
                    task_id=f"synthetic-{item_index:03d}",
                    final_success=is_success,
                    failure_bucket="NONE" if is_success else "TASK_FAILURE",
                    steps=_materialize_steps(
                        observed_symbols,
                        rng=rng,
                        lexical_variation=spec.lexical_variation,
                    ),
                    split=split,
                    metadata={
                        "ground_truth_symbols": logical_symbols,
                        "observed_symbols": observed_symbols,
                    },
                )
            )
            ground_truth_symbols[trajectory_id] = logical_symbols

    return SyntheticProtocolDataset(
        trajectories=trajectories,
        ground_truth_symbols=ground_truth_symbols,
        spec=spec,
    )


def write_synthetic_trajectories_jsonl(
    dataset: SyntheticProtocolDataset,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for trajectory in dataset.trajectories:
            payload = {
                "trajectory_id": trajectory.trajectory_id,
                "task_id": trajectory.task_id,
                "final_success": trajectory.final_success,
                "failure_bucket": trajectory.failure_bucket,
                "split": trajectory.split,
                "metadata": trajectory.metadata,
                "steps": [
                    {
                        "context": step.context,
                        "action_text": step.action_text,
                        "tool_name": step.tool_name,
                        "tool_args": step.tool_args,
                        "result_text": step.result_text,
                        "status": step.status,
                    }
                    for step in trajectory.steps
                ],
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def summarize_synthetic_recovery(
    dataset: SyntheticProtocolDataset,
    extracted_symbols: dict[str, list[int]],
) -> dict:
    exact_matches = 0
    token_matches = 0
    token_total = 0
    acceptance_matches = 0
    total = 0
    for trajectory in dataset.trajectories:
        if trajectory.trajectory_id not in extracted_symbols:
            continue
        truth = dataset.ground_truth_symbols[trajectory.trajectory_id]
        predicted = extracted_symbols[trajectory.trajectory_id]
        total += 1
        if truth == predicted:
            exact_matches += 1
        overlap = min(len(truth), len(predicted))
        token_matches += sum(int(left == right) for left, right in zip(truth[:overlap], predicted[:overlap]))
        token_total += max(len(truth), len(predicted))
        predicted_accepts = predicted[-1] != 4 if predicted else True
        truth_accepts = trajectory.final_success
        acceptance_matches += int(predicted_accepts == truth_accepts)
    return {
        "trajectory_exact_match": exact_matches / max(total, 1),
        "symbol_token_accuracy": token_matches / max(token_total, 1),
        "ground_truth_acceptance_agreement": acceptance_matches / max(total, 1),
        "num_trajectories": total,
        "spec": asdict(dataset.spec),
    }
