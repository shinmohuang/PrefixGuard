from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NativeBenchmarkRecord:
    dataset: str
    record_id: str
    split: str
    artifact_type: str
    evaluator_type: str
    outcome_source: str
    instruction: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _require(obj: dict, key: str) -> Any:
    if key not in obj:
        raise ValueError(f"Missing required field '{key}' in native benchmark record: {obj}")
    return obj[key]


def _parse_native_benchmark_record(obj: dict) -> NativeBenchmarkRecord:
    return NativeBenchmarkRecord(
        dataset=str(_require(obj, "dataset")),
        record_id=str(_require(obj, "record_id")),
        split=str(_require(obj, "split")),
        artifact_type=str(_require(obj, "artifact_type")),
        evaluator_type=str(_require(obj, "evaluator_type")),
        outcome_source=str(_require(obj, "outcome_source")),
        instruction=str(obj.get("instruction", "")),
        source_path=str(_require(obj, "source_path")),
        metadata=dict(obj.get("metadata", {})),
    )


def load_native_benchmark_records(path: str | Path) -> list[NativeBenchmarkRecord]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Native benchmark manifest not found: {dataset_path}")

    records: list[NativeBenchmarkRecord] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            records.append(_parse_native_benchmark_record(obj))
    if not records:
        raise ValueError(f"No native benchmark records found in manifest: {dataset_path}")
    return records


def summarize_native_benchmark_records(records: list[NativeBenchmarkRecord]) -> dict:
    dataset_counts: dict[str, int] = {}
    artifact_counts: dict[str, int] = {}
    evaluator_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for record in records:
        dataset = str(record.dataset)
        artifact_type = str(record.artifact_type)
        evaluator_type = str(record.evaluator_type)
        outcome_source = str(record.outcome_source)
        split_key = f"{dataset}:{record.split}"
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        artifact_counts[artifact_type] = artifact_counts.get(artifact_type, 0) + 1
        evaluator_counts[evaluator_type] = evaluator_counts.get(evaluator_type, 0) + 1
        outcome_counts[outcome_source] = outcome_counts.get(outcome_source, 0) + 1
        split_counts[split_key] = split_counts.get(split_key, 0) + 1
    return {
        "num_records": len(records),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "artifact_type_counts": dict(sorted(artifact_counts.items())),
        "evaluator_type_counts": dict(sorted(evaluator_counts.items())),
        "outcome_source_counts": dict(sorted(outcome_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
    }
