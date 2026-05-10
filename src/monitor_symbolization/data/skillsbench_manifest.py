from __future__ import annotations

import importlib.util
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLSBENCH_IMPORTER_PATH = REPO_ROOT / "scripts" / "import_skillsbench_traces.py"


@lru_cache(maxsize=1)
def _skillsbench_importer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_prefixguard_skillsbench_importer",
        SKILLSBENCH_IMPORTER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load SkillsBench importer: {SKILLSBENCH_IMPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def looks_like_skillsbench_split_manifest_row(obj: dict[str, Any]) -> bool:
    return (
        obj.get("dataset") == "SkillsBench"
        and str(obj.get("trajectory_id") or "").startswith("skillsbench::")
        and bool(str(obj.get("split") or "").strip())
        and bool(str(obj.get("trial_dir") or "").strip())
    )


def _resolve_trial_dir(trial_dir: str) -> Path:
    path = Path(trial_dir)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _read_skillsbench_split_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid SkillsBench split manifest JSON on line {line_no}: {exc}") from exc
            if not isinstance(obj, dict) or not looks_like_skillsbench_split_manifest_row(obj):
                raise ValueError(
                    "SkillsBench split manifest rows must include dataset=SkillsBench, "
                    f"trajectory_id, split, and trial_dir; line={line_no}"
                )
            trajectory_id = str(obj["trajectory_id"])
            if trajectory_id in seen:
                raise ValueError(f"Duplicate SkillsBench trajectory_id in split manifest: {trajectory_id}")
            seen.add(trajectory_id)
            row = {
                "dataset": "SkillsBench",
                "trajectory_id": trajectory_id,
                "task_id": str(obj.get("task_id") or ""),
                "split": str(obj["split"]),
                "trial_dir": str(obj["trial_dir"]),
            }
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in SkillsBench split manifest: {manifest_path}")
    return rows


def load_skillsbench_records_from_split_manifest(
    manifest_path: str | Path,
    *,
    history_window: int = 6,
    max_action_chars: int = 1200,
    max_result_chars: int = 2400,
    max_context_chars: int = 4000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_skillsbench_split_manifest(manifest_path)
    importer = _skillsbench_importer()
    records: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    trace_formats: Counter[str] = Counter()
    missing_source_raw: list[str] = []

    for row in rows:
        record = importer._parse_trial(
            _resolve_trial_dir(row["trial_dir"]),
            split=row["split"],
            history_window=history_window,
            max_action_chars=max_action_chars,
            max_result_chars=max_result_chars,
            max_context_chars=max_context_chars,
        )
        if record["trajectory_id"] != row["trajectory_id"]:
            raise ValueError(
                "SkillsBench manifest trajectory_id does not match parsed trial: "
                f"{row['trajectory_id']} != {record['trajectory_id']}"
            )
        if row["task_id"] and record["task_id"] != row["task_id"]:
            raise ValueError(
                "SkillsBench manifest task_id does not match parsed trial: "
                f"{row['task_id']} != {record['task_id']}"
            )
        for index, step in enumerate(record.get("steps", [])):
            if not str(step.get("source_raw_text") or "").strip() and len(missing_source_raw) < 5:
                missing_source_raw.append(f"{record['trajectory_id']}:{index}")
        record["split"] = row["split"]
        records.append(record)
        split_counts[row["split"]] += 1
        trace_formats[str(record.get("metadata", {}).get("trace_format", "UNKNOWN"))] += 1

    if missing_source_raw:
        raise ValueError(f"SkillsBench source_raw_text missing for steps: {missing_source_raw}")

    summary = {
        "dataset": "SkillsBench",
        "mode": "source_raw_split_manifest",
        "manifest_path": str(Path(manifest_path)),
        "num_manifest_rows": len(rows),
        "num_rebuilt_records": len(records),
        "num_steps": sum(len(record.get("steps", [])) for record in records),
        "num_steps_with_source_raw_text": sum(
            1
            for record in records
            for step in record.get("steps", [])
            if str(step.get("source_raw_text") or "").strip()
        ),
        "split_counts": dict(sorted(split_counts.items())),
        "source_raw_truncation": importer._source_raw_protocol_summary(),
        "trace_formats": dict(sorted(trace_formats.items())),
    }
    return records, summary
