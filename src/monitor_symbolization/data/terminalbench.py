from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

TERMINALBENCH_PARQUET_ROOT_ENV = "TERMINALBENCH_PARQUET_ROOT"
DEFAULT_TERMINALBENCH_PARQUET_ROOT = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "external"
    / "terminalbench"
    / "terminalbench-trajectories"
)


def _compact_text(value: Any, *, max_chars: int = 2000) -> str:
    if value is None:
        text = "NONE"
    elif isinstance(value, str):
        stripped = value.strip()
        text = stripped if stripped else "NONE"
    else:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 16]}...<truncated>"


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _default_terminalbench_parquet_root() -> Path:
    env_root = os.environ.get(TERMINALBENCH_PARQUET_ROOT_ENV)
    if env_root:
        return Path(env_root)
    return DEFAULT_TERMINALBENCH_PARQUET_ROOT


def _history_line(message: dict[str, Any], *, max_chars: int = 400) -> str:
    src = str(message.get("src") or "unknown").strip().lower() or "unknown"
    msg = _compact_text(message.get("msg"), max_chars=max_chars)
    return f"{src}: {msg}"


def _build_context(
    row: dict[str, Any],
    *,
    history: list[dict[str, Any]],
    history_window: int,
    step_num: int,
) -> str:
    lines = [
        "dataset=TerminalBench",
        f"task_name={row.get('task_name', 'UNKNOWN')}",
        f"trial_name={row.get('trial_name', 'UNKNOWN')}",
        f"trial_id={row.get('trial_id', 'UNKNOWN')}",
        f"agent={row.get('agent', 'UNKNOWN')}",
        f"model={row.get('model', 'UNKNOWN')}",
        f"step_num={step_num}",
        "observation=",
    ]
    rendered = history[-history_window:] if history_window > 0 else history
    if rendered:
        lines.extend(_history_line(message) for message in rendered)
    else:
        lines.append("NONE")
    return "\n".join(lines)


def _action_text(message: dict[str, Any], *, tool_name: str) -> str:
    message_text = _compact_text(message.get("msg"), max_chars=4000)
    if tool_name == "respond":
        return message_text
    if message_text == "NONE":
        return f"tool_call::{tool_name}"
    return f"{message_text}\nTOOL_CALL {tool_name}"


def _normalize_tool_name(name: Any) -> str:
    raw = str(name or "").strip()
    return raw if raw else "respond"


def _tool_args_from_single_tool(tool: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in tool.items() if key != "fn"}
    if "cmd" in payload and isinstance(payload["cmd"], str):
        payload["cmd"] = _compact_text(payload["cmd"], max_chars=2000)
    return payload


def _summarize_tools(tools: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(tools, list) or not tools:
        return "respond", {}
    normalized = [tool for tool in tools if isinstance(tool, dict)]
    if not normalized:
        return "respond", {}
    if len(normalized) == 1:
        tool = normalized[0]
        return _normalize_tool_name(tool.get("fn")), _tool_args_from_single_tool(tool)

    unique_fns: list[str] = []
    for tool in normalized:
        fn = _normalize_tool_name(tool.get("fn"))
        if fn not in unique_fns:
            unique_fns.append(fn)

    preview_cmds: list[str] = []
    for tool in normalized[:8]:
        cmd = tool.get("cmd")
        if cmd in (None, ""):
            continue
        preview_cmds.append(_compact_text(cmd, max_chars=240))

    tool_name = unique_fns[0] if len(unique_fns) == 1 else "multi_tool_call"
    tool_args = {
        "tool_count": len(normalized),
        "tool_fns": unique_fns,
    }
    if preview_cmds:
        tool_args["cmd_preview"] = preview_cmds
    if len(normalized) > len(preview_cmds):
        tool_args["truncated_tool_count"] = len(normalized) - len(preview_cmds)
    return tool_name, tool_args


def _step_status(*, result_text: str, is_last_agent_step: bool, final_success: bool) -> str:
    if is_last_agent_step:
        return "success" if final_success else "failure"
    lowered = result_text.lower()
    error_markers = (
        "[error]",
        "traceback",
        "exception",
        "<status>failed</status>",
        "tool reported failure",
    )
    if any(marker in lowered for marker in error_markers):
        return "tool_error"
    return "ok"


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    task_name = str(row.get("task_name") or "unknown_task").strip() or "unknown_task"
    trial_name = str(row.get("trial_name") or task_name).strip() or task_name
    trial_id = str(row.get("trial_id") or trial_name).strip() or trial_name
    return task_name, trial_name, trial_id


def _trajectory_id_from_row(row: dict[str, Any]) -> str:
    _task_name, trial_name, trial_id = _row_identity(row)
    return f"terminalbench::{trial_name}::{trial_id}"


def _parse_steps_payload(raw_steps: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_steps, str):
        return None
    stripped = raw_steps.strip()
    if not stripped or stripped.lower() == "null":
        return None
    payload = json.loads(stripped)
    if not isinstance(payload, list):
        raise ValueError(f"TerminalBench steps payload must decode to a list, got {type(payload)!r}")
    normalized = [item for item in payload if isinstance(item, dict)]
    return normalized if normalized else None


def _source_raw_text(
    row: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    message_index: int,
    history_window: int,
    step_num: int,
) -> str:
    history = messages[:message_index]
    rendered_history = history[-history_window:] if history_window > 0 else history
    return _stable_json(
        {
            "dataset": "TerminalBench",
            "task_name": row.get("task_name"),
            "trial_name": row.get("trial_name"),
            "trial_id": row.get("trial_id"),
            "agent": row.get("agent"),
            "model": row.get("model"),
            "step_num": step_num,
            "history": rendered_history,
            "agent_message": messages[message_index],
        }
    )


def parse_terminalbench_row(
    row: dict[str, Any],
    *,
    split: str = "raw",
    history_window: int = 8,
    source_parquet: str,
) -> dict:
    messages = _parse_steps_payload(row.get("steps"))
    if messages is None:
        raise ValueError("TerminalBench row does not contain a materialized steps payload")

    agent_positions = [
        index for index, message in enumerate(messages) if str(message.get("src") or "").lower() == "agent"
    ]
    if not agent_positions:
        raise ValueError("TerminalBench row does not contain any agent messages")

    final_success = bool(int(row.get("reward", 0) or 0) > 0)
    steps: list[dict[str, Any]] = []
    for step_num, message_index in enumerate(agent_positions, start=1):
        message = messages[message_index]
        tool_name, tool_args = _summarize_tools(message.get("tools"))
        result_text = _compact_text(message.get("obs"), max_chars=4000)
        steps.append(
            {
                "context": _build_context(
                    row,
                    history=messages[:message_index],
                    history_window=history_window,
                    step_num=step_num,
                ),
                "action_text": _action_text(message, tool_name=tool_name),
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result_text": result_text,
                "status": _step_status(
                    result_text=result_text,
                    is_last_agent_step=step_num == len(agent_positions),
                    final_success=final_success,
                ),
                "source_raw_text": _source_raw_text(
                    row,
                    messages=messages,
                    message_index=message_index,
                    history_window=history_window,
                    step_num=step_num,
                ),
            }
        )

    task_name, trial_name, trial_id = _row_identity(row)
    reward = int(row.get("reward", 0) or 0)

    return {
        "trajectory_id": f"terminalbench::{trial_name}::{trial_id}",
        "task_id": f"terminalbench::{task_name}",
        "final_success": final_success,
        "failure_bucket": "NONE" if final_success else "TERMINALBENCH_REWARD_ZERO",
        "steps": steps,
        "split": split,
        "metadata": {
            "dataset": "TerminalBench",
            "adapter_origin": "terminalbench_hf_parquet",
            "label_origin": "reward",
            "task_name": task_name,
            "agent_name": row.get("agent"),
            "model_name": row.get("model"),
            "raw_reward": reward,
            "duration_seconds": row.get("duration_seconds"),
            "input_tokens": row.get("input_tokens"),
            "output_tokens": row.get("output_tokens"),
            "cache_tokens": row.get("cache_tokens"),
            "cost_cents": row.get("cost_cents"),
            "trial_name": trial_name,
            "trial_id": trial_id,
            "started_at": row.get("started_at"),
            "ended_at": row.get("ended_at"),
            "num_raw_messages": len(messages),
            "num_agent_steps": len(steps),
            "source_parquet": source_parquet,
        },
    }


def _resolve_parquet_files(input_root: str | Path) -> list[Path]:
    root = Path(input_root)
    if root.is_file():
        return [root]

    candidate_dirs = [root]
    data_dir = root / "data"
    if data_dir.exists():
        candidate_dirs.insert(0, data_dir)

    parquet_files: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidate_dirs:
        for path in sorted(candidate.glob("*.parquet")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            parquet_files.append(path)
            seen.add(resolved)
    return parquet_files


def looks_like_terminalbench_split_manifest_row(obj: dict[str, Any]) -> bool:
    return (
        set(obj) == {"trajectory_id", "split"}
        and str(obj.get("trajectory_id") or "").startswith("terminalbench::")
        and bool(str(obj.get("split") or "").strip())
    )


def _read_terminalbench_split_manifest(path: str | Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    manifest_path = Path(path)
    rows: list[dict[str, str]] = []
    split_by_id: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid TerminalBench split manifest JSON on line {line_no}: {exc}") from exc
            if not isinstance(obj, dict) or set(obj) != {"trajectory_id", "split"}:
                raise ValueError(
                    "TerminalBench split manifest rows must contain only "
                    f"'trajectory_id' and 'split'; line={line_no}"
                )
            trajectory_id = str(obj["trajectory_id"])
            split = str(obj["split"])
            if not trajectory_id.startswith("terminalbench::") or not split.strip():
                raise ValueError(
                    "TerminalBench split manifest row has invalid trajectory_id or split; "
                    f"line={line_no}"
                )
            if trajectory_id in split_by_id:
                raise ValueError(f"Duplicate TerminalBench trajectory_id in split manifest: {trajectory_id}")
            row = {"trajectory_id": trajectory_id, "split": split}
            rows.append(row)
            split_by_id[trajectory_id] = split
    if not rows:
        raise ValueError(f"No rows found in TerminalBench split manifest: {manifest_path}")
    return rows, split_by_id


def load_terminalbench_records_from_split_manifest(
    manifest_path: str | Path,
    *,
    input_root: str | Path | None = None,
    history_window: int = 8,
) -> tuple[list[dict], dict[str, Any]]:
    manifest_rows, split_by_id = _read_terminalbench_split_manifest(manifest_path)
    parquet_root = Path(input_root) if input_root is not None else _default_terminalbench_parquet_root()
    records, import_summary = parse_terminalbench_parquet_dir(
        parquet_root,
        split="raw",
        history_window=history_window,
        include_trajectory_ids=set(split_by_id),
    )
    record_by_id: dict[str, dict] = {}
    duplicates: list[str] = []
    for record in records:
        trajectory_id = str(record.get("trajectory_id"))
        if trajectory_id in record_by_id:
            duplicates.append(trajectory_id)
            continue
        record_by_id[trajectory_id] = record
    if duplicates:
        raise ValueError(f"Duplicate TerminalBench trajectory_id in parquet import: {sorted(duplicates)[:5]}")

    rebuilt: list[dict] = []
    missing: list[str] = []
    for row in manifest_rows:
        trajectory_id = row["trajectory_id"]
        record = record_by_id.get(trajectory_id)
        if record is None:
            missing.append(trajectory_id)
            continue
        record["split"] = split_by_id[trajectory_id]
        rebuilt.append(record)
    if missing:
        raise ValueError(f"TerminalBench split manifest references missing parquet rows: {missing[:5]}")

    summary = {
        "dataset": "TerminalBench",
        "mode": "source_raw_split_manifest",
        "manifest_path": str(Path(manifest_path)),
        "parquet_input_root": str(parquet_root),
        "history_window": int(history_window),
        "num_manifest_rows": len(manifest_rows),
        "num_rebuilt_records": len(rebuilt),
        "split_counts": dict(Counter(row["split"] for row in manifest_rows)),
        "parquet_import_summary": import_summary,
    }
    return rebuilt, summary


def scan_terminalbench_manifest_source_records(
    input_root: str | Path,
    *,
    split: str = "raw",
    limit: int | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    parquet_files = _resolve_parquet_files(input_root)
    if not parquet_files:
        raise FileNotFoundError(f"No TerminalBench parquet files found under {Path(input_root)}")

    records: list[dict[str, str]] = []
    task_counts: Counter[str] = Counter()
    summary: dict[str, Any] = {
        "dataset": "TerminalBench",
        "input_root": str(Path(input_root)),
        "split": split,
        "num_parquet_files": len(parquet_files),
        "source_parquet_files": [str(path) for path in parquet_files],
        "num_rows_seen": 0,
        "num_imported": 0,
        "num_successes": 0,
        "num_failures": 0,
        "num_agent_steps": 0,
        "num_skipped_null_steps": 0,
        "num_skipped_invalid_rows": 0,
        "skipped_examples": [],
    }

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            summary["num_rows_seen"] += 1
            try:
                messages = _parse_steps_payload(row.get("steps"))
                if messages is None:
                    raise ValueError("TerminalBench row does not contain a materialized steps payload")
                agent_steps = sum(
                    1
                    for message in messages
                    if str(message.get("src") or "").lower() == "agent"
                )
                if agent_steps == 0:
                    raise ValueError("TerminalBench row does not contain any agent messages")
            except ValueError as exc:
                if "materialized steps payload" in str(exc):
                    summary["num_skipped_null_steps"] += 1
                else:
                    summary["num_skipped_invalid_rows"] += 1
                if len(summary["skipped_examples"]) < 20:
                    summary["skipped_examples"].append(
                        {
                            "task_name": row.get("task_name"),
                            "trial_name": row.get("trial_name"),
                            "trial_id": row.get("trial_id"),
                            "reason": str(exc),
                            "source_parquet": str(parquet_path),
                        }
                    )
                continue

            task_name, _trial_name, _trial_id = _row_identity(row)
            record = {
                "trajectory_id": _trajectory_id_from_row(row),
                "task_id": f"terminalbench::{task_name}",
                "split": split,
            }
            records.append(record)
            task_counts[record["task_id"]] += 1
            final_success = bool(int(row.get("reward", 0) or 0) > 0)
            summary["num_successes"] += int(final_success)
            summary["num_failures"] += int(not final_success)
            summary["num_agent_steps"] += agent_steps
            if limit is not None and len(records) >= limit:
                summary["num_imported"] = len(records)
                summary["num_unique_tasks"] = len(task_counts)
                summary["task_counts"] = dict(sorted(task_counts.items()))
                return records, summary

    summary["num_imported"] = len(records)
    summary["num_unique_tasks"] = len(task_counts)
    summary["task_counts"] = dict(sorted(task_counts.items()))
    return records, summary


def write_terminalbench_split_manifest(
    path: str | Path,
    records: list[dict[str, Any]],
    *,
    seed: int = 1,
    fit_ratio: float = 0.7,
    cal_ratio: float = 0.1,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, Any]:
    total = fit_ratio + cal_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    task_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        task_records.setdefault(str(record["task_id"]), []).append(record)

    rng = random.Random(seed)
    split_counts = {"fit": 0, "cal": 0, "val": 0, "test": 0}
    tasks_per_split: dict[str, set[str]] = {name: set() for name in split_counts}
    assigned: list[dict[str, str]] = []

    for task_id in sorted(task_records):
        bucket = list(task_records[task_id])
        rng.shuffle(bucket)
        n = len(bucket)
        if n < 4:
            raise ValueError(
                f"Task {task_id!r} has only {n} trajectories; "
                "trajectory-stratified requires >= 4 per task."
            )
        n_cal = max(1, round(n * cal_ratio))
        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * test_ratio))
        n_fit = n - n_cal - n_val - n_test
        if n_fit < 1:
            raise ValueError(f"Task {task_id!r}: n={n} too small for all splits.")
        labels = ["fit"] * n_fit + ["cal"] * n_cal + ["val"] * n_val + ["test"] * n_test
        for record, split in zip(bucket, labels):
            trajectory_id = str(record["trajectory_id"])
            assigned.append({"trajectory_id": trajectory_id, "split": split})
            split_counts[split] += 1
            tasks_per_split[split].add(task_id)

    all_tasks = set(task_records)
    for split, seen_tasks in tasks_per_split.items():
        missing = all_tasks - seen_tasks
        if missing:
            raise ValueError(f"Split {split!r} is missing tasks: {sorted(missing)[:10]}")

    rng.shuffle(assigned)
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in assigned:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    return {
        "protocol_mode": "trajectory-stratified-source-raw-manifest",
        "output": str(manifest_path),
        "seed": seed,
        "ratios": {
            "fit": fit_ratio,
            "cal": cal_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "task_counts": {split: len(tasks_per_split[split]) for split in split_counts},
        "trajectory_counts": split_counts,
        "num_unique_tasks": len(task_records),
        "manifest_fields": ["trajectory_id", "split"],
    }


def parse_terminalbench_parquet_dir(
    input_root: str | Path,
    *,
    split: str = "raw",
    history_window: int = 8,
    limit: int | None = None,
    include_trajectory_ids: set[str] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    parquet_files = _resolve_parquet_files(input_root)
    if not parquet_files:
        raise FileNotFoundError(f"No TerminalBench parquet files found under {Path(input_root)}")

    imported: list[dict] = []
    task_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    summary: dict[str, Any] = {
        "dataset": "TerminalBench",
        "input_root": str(Path(input_root)),
        "split": split,
        "history_window": int(history_window),
        "num_parquet_files": len(parquet_files),
        "source_parquet_files": [str(path) for path in parquet_files],
        "num_rows_seen": 0,
        "num_rows_filtered_by_trajectory_id": 0,
        "include_trajectory_id_count": None
        if include_trajectory_ids is None
        else len(include_trajectory_ids),
        "num_imported": 0,
        "num_successes": 0,
        "num_failures": 0,
        "num_skipped_null_steps": 0,
        "num_skipped_invalid_rows": 0,
        "skipped_examples": [],
    }
    remaining_ids = set(include_trajectory_ids) if include_trajectory_ids is not None else None

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            summary["num_rows_seen"] += 1
            if include_trajectory_ids is not None:
                trajectory_id = _trajectory_id_from_row(row)
                if trajectory_id not in include_trajectory_ids:
                    summary["num_rows_filtered_by_trajectory_id"] += 1
                    continue
            try:
                record = parse_terminalbench_row(
                    row,
                    split=split,
                    history_window=history_window,
                    source_parquet=str(parquet_path),
                )
            except ValueError as exc:
                if "materialized steps payload" in str(exc):
                    summary["num_skipped_null_steps"] += 1
                else:
                    summary["num_skipped_invalid_rows"] += 1
                if len(summary["skipped_examples"]) < 20:
                    summary["skipped_examples"].append(
                        {
                            "task_name": row.get("task_name"),
                            "trial_name": row.get("trial_name"),
                            "trial_id": row.get("trial_id"),
                            "reason": str(exc),
                            "source_parquet": str(parquet_path),
                        }
                    )
                continue

            imported.append(record)
            if remaining_ids is not None:
                remaining_ids.discard(record["trajectory_id"])
            task_counts[record["task_id"]] += 1
            model_counts[str(record["metadata"].get("model_name") or "UNKNOWN")] += 1
            summary["num_successes"] += int(record["final_success"])
            summary["num_failures"] += int(not record["final_success"])
            if (limit is not None and len(imported) >= limit) or (
                remaining_ids is not None and not remaining_ids
            ):
                summary["num_imported"] = len(imported)
                summary["num_unique_tasks"] = len(task_counts)
                summary["task_counts"] = dict(sorted(task_counts.items()))
                summary["model_counts"] = dict(sorted(model_counts.items()))
                return imported, summary

    summary["num_imported"] = len(imported)
    summary["num_unique_tasks"] = len(task_counts)
    summary["task_counts"] = dict(sorted(task_counts.items()))
    summary["model_counts"] = dict(sorted(model_counts.items()))
    return imported, summary
