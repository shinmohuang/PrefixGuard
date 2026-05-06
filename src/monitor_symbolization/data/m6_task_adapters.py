from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path) -> list[dict]:
    dataset_path = Path(path)
    rows: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _mutate_contract_text(text: str, *, salt: str) -> str:
    base = text.strip()
    if not base:
        return f"fallback-{salt[:8]}"
    chars = list(base)
    digest = hashlib.sha256(f"{salt}:{base}".encode("utf-8")).digest()
    index = digest[0] % len(chars)
    original = chars[index]
    if original.isdigit():
        chars[index] = str((int(original) + 1) % 10)
    elif original.isalpha():
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        original_lower = original.lower()
        offset = (alphabet.index(original_lower) + 1) % len(alphabet)
        replacement = alphabet[offset]
        chars[index] = replacement.upper() if original.isupper() else replacement
    else:
        chars.append(" alt")
    mutated = "".join(chars)
    if mutated == base:
        mutated = f"{base} alt"
    return mutated


def _finalize_contract(parts: list[str], *, fallback: str) -> str:
    cleaned = [_normalize_whitespace(part) for part in parts if part and part.strip()]
    if cleaned:
        return " | ".join(cleaned)
    return fallback


def _build_pair(
    *,
    dataset: str,
    task_id: str,
    instruction: str,
    metadata_lines: list[str],
    positive_contract: str,
    split: str,
    source_path: str,
    metadata: dict[str, Any],
    failure_bucket: str = "SYNTHETIC_TASK_CONTRACT_MISMATCH",
) -> tuple[dict, dict]:
    normalized_instruction = _normalize_whitespace(instruction)
    normalized_positive = _normalize_whitespace(positive_contract)
    negative_contract = _mutate_contract_text(
        normalized_positive,
        salt=f"{dataset}:{task_id}",
    )
    shared_context_lines = [
        f"dataset={dataset}",
        f"task_id={task_id}",
        f"instruction={normalized_instruction}",
        *metadata_lines,
    ]
    shared_context = "\n".join(shared_context_lines)
    second_context = f"{shared_context}\ncontract={normalized_positive}"
    shared_metadata = {
        **metadata,
        "dataset": dataset,
        "source_path": source_path,
        "adapter_origin": "synthetic_task_pair",
        "label_origin": "synthetic_contract_mutation",
        "positive_contract": normalized_positive,
        "negative_contract": negative_contract,
    }

    success = {
        "trajectory_id": f"{dataset.lower()}-{task_id}-success",
        "task_id": str(task_id),
        "final_success": True,
        "failure_bucket": "NONE",
        "split": split,
        "metadata": shared_metadata,
        "steps": [
            {
                "context": shared_context,
                "action_text": "inspect_contract",
                "tool_name": "inspect",
                "tool_args": {},
                "result_text": f"contract={normalized_positive}",
                "status": "ok",
            },
            {
                "context": second_context,
                "action_text": f"submit::{normalized_positive}",
                "tool_name": "submit",
                "tool_args": {},
                "result_text": "submission_recorded",
                "status": "ok",
            },
        ],
    }
    failure = {
        "trajectory_id": f"{dataset.lower()}-{task_id}-failure",
        "task_id": str(task_id),
        "final_success": False,
        "failure_bucket": failure_bucket,
        "split": split,
        "metadata": shared_metadata,
        "steps": [
            {
                "context": shared_context,
                "action_text": "inspect_contract",
                "tool_name": "inspect",
                "tool_args": {},
                "result_text": f"contract={normalized_positive}",
                "status": "ok",
            },
            {
                "context": second_context,
                "action_text": f"submit::{negative_contract}",
                "tool_name": "submit",
                "tool_args": {},
                "result_text": "submission_recorded",
                "status": "ok",
            },
        ],
    }
    return success, failure


def _summarize_dataset(
    *,
    dataset: str,
    trajectories: list[dict],
    notes: list[str],
) -> dict:
    split_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    for row in trajectories:
        split_counts[str(row.get("split", "unknown"))] += 1
        outcome_counts["success" if row["final_success"] else "failure"] += 1
    num_steps = sum(len(row["steps"]) for row in trajectories)
    return {
        "dataset": dataset,
        "adapter_origin": "synthetic_task_pair",
        "label_origin": "synthetic_contract_mutation",
        "num_trajectories": len(trajectories),
        "num_tasks": len({row["task_id"] for row in trajectories}),
        "num_steps": num_steps,
        "avg_steps_per_trajectory": (num_steps / len(trajectories)) if trajectories else 0.0,
        "split_counts": dict(split_counts),
        "outcome_counts": dict(outcome_counts),
        "notes": notes,
    }


def _limit_items(items: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return items
    return items[:limit]


def build_webchorearena_task_pairs(
    repo_root: str | Path,
    *,
    split: str = "raw",
    limit: int | None = None,
) -> tuple[list[dict], dict]:
    config_root = Path(repo_root) / "BrowserGym" / "config_files"
    trajectories: list[dict] = []
    task_records: list[tuple[str, str, dict, dict]] = []
    for path in sorted(config_root.glob("*.raw.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload:
            task_records.append((path.stem.replace(".raw", ""), str(row["task_id"]), row, path))
    for config_split, task_id, row, path in _limit_items(task_records, limit):
        unique_task_id = f"{config_split}:{task_id}"
        eval_config = row.get("eval") or {}
        reference_answers = eval_config.get("reference_answers") or {}
        contract = _finalize_contract(
            [
                f"eval_types={','.join(eval_config.get('eval_types', []))}",
                eval_config.get("reference_answer_raw_annotation", ""),
                " || ".join(reference_answers.get("must_include", [])[:3]),
                eval_config.get("reference_url", ""),
                _compact_json(eval_config.get("program_html", [])[:2])
                if eval_config.get("program_html")
                else "",
            ],
            fallback=f"task-{task_id}",
        )
        metadata_lines = [
            f"config_split={config_split}",
            f"sites={','.join(row.get('sites', []))}",
            f"type_main={row.get('type_main', 'UNKNOWN')}",
            f"type_sub={row.get('type_sub', 'UNKNOWN')}",
            f"required_obs={row.get('required_obs', 'UNKNOWN')}",
        ]
        success, failure = _build_pair(
            dataset="WebChoreArena",
            task_id=unique_task_id,
            instruction=row.get("intent", f"WebChoreArena task {task_id}"),
            metadata_lines=metadata_lines,
            positive_contract=contract,
            split=split,
            source_path=str(path),
            metadata={
                "original_task_id": task_id,
                "config_split": config_split,
                "sites": list(row.get("sites", [])),
                "type_main": row.get("type_main", "UNKNOWN"),
                "type_sub": row.get("type_sub", "UNKNOWN"),
                "required_obs": row.get("required_obs", "UNKNOWN"),
                "eval_types": list(eval_config.get("eval_types", [])),
            },
        )
        trajectories.extend([success, failure])
    summary = _summarize_dataset(
        dataset="WebChoreArena",
        trajectories=trajectories,
        notes=[
            "This adapter does not use public execution traces because the repo ships task configs only.",
            "Each task is expanded into one synthetic success trajectory and one synthetic contract-mismatch trajectory.",
        ],
    )
    return trajectories, summary


def build_osworld_task_pairs(
    repo_root: str | Path,
    *,
    manifest_name: str = "test_small",
    split: str = "raw",
    limit: int | None = None,
) -> tuple[list[dict], dict]:
    repo_path = Path(repo_root)
    manifest = json.loads(
        (repo_path / "evaluation_examples" / f"{manifest_name}.json").read_text(encoding="utf-8")
    )
    items: list[tuple[str, dict, Path]] = []
    for domain, example_ids in manifest.items():
        for example_id in example_ids:
            example_path = repo_path / "evaluation_examples" / "examples" / domain / f"{example_id}.json"
            example = json.loads(example_path.read_text(encoding="utf-8"))
            items.append((domain, example, example_path))

    trajectories: list[dict] = []
    for domain, example, example_path in _limit_items(items, limit):
        evaluator = example.get("evaluator", {})
        contract = _finalize_contract(
            [
                f"evaluator_func={evaluator.get('func', 'UNKNOWN')}",
                f"result={_compact_json(evaluator.get('result', {}))}",
                f"expected={_compact_json(evaluator.get('expected', {}))}",
            ],
            fallback=f"osworld-{example['id']}",
        )
        metadata_lines = [
            f"manifest={manifest_name}",
            f"domain={domain}",
            f"snapshot={example.get('snapshot', 'UNKNOWN')}",
            f"related_apps={','.join(example.get('related_apps', []))}",
            f"proxy={example.get('proxy', False)}",
            f"fixed_ip={example.get('fixed_ip', False)}",
        ]
        success, failure = _build_pair(
            dataset="OSWorld",
            task_id=str(example["id"]),
            instruction=example.get("instruction", f"OSWorld task {example['id']}"),
            metadata_lines=metadata_lines,
            positive_contract=contract,
            split=split,
            source_path=str(example_path),
            metadata={
                "manifest": manifest_name,
                "domain": domain,
                "snapshot": example.get("snapshot", "UNKNOWN"),
                "related_apps": list(example.get("related_apps", [])),
                "evaluator_func": evaluator.get("func", "UNKNOWN"),
            },
        )
        trajectories.extend([success, failure])
    summary = _summarize_dataset(
        dataset="OSWorld",
        trajectories=trajectories,
        notes=[
            "This adapter uses public example definitions rather than executed desktop rollouts.",
            "Synthetic failures are deterministic evaluator-contract mutations, not native failed agent traces.",
        ],
    )
    return trajectories, summary


def build_toolsandbox_task_pairs(
    audit_path: str | Path,
    *,
    split: str = "raw",
    limit: int | None = None,
) -> tuple[list[dict], dict]:
    rows = _read_jsonl(audit_path)
    trajectories: list[dict] = []
    for row in _limit_items(rows, limit):
        scenario_name = str(row["scenario_name"])
        scenario_file = str(row["scenario_file"])
        contract = _finalize_contract(
            [
                f"scenario_file={scenario_file}",
                f"scenario_name={scenario_name}",
            ],
            fallback=scenario_name,
        )
        metadata_lines = [
            f"scenario_file={scenario_file}",
            "evaluator=milestone_dag",
        ]
        success, failure = _build_pair(
            dataset="ToolSandbox",
            task_id=scenario_name,
            instruction=scenario_name.replace("_", " "),
            metadata_lines=metadata_lines,
            positive_contract=contract,
            split=split,
            source_path=str(audit_path),
            metadata={
                "scenario_file": scenario_file,
            },
        )
        trajectories.extend([success, failure])
    summary = _summarize_dataset(
        dataset="ToolSandbox",
        trajectories=trajectories,
        notes=[
            "This adapter uses public scenario definitions and names only.",
            "The benchmark's native milestone evaluator is preserved in metadata, but the trajectory labels are synthetic task pairs.",
        ],
    )
    return trajectories, summary


def build_workarena_task_audit_records(config_root: str | Path) -> list[dict]:
    records: list[dict] = []
    for config_path in sorted(Path(config_root).glob("*.json")):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                continue
            task_fields = list(row.get("task_fields", []))
            template_record = dict(row.get("template_record", {}))
            selected_values = {
                field: template_record.get(field)
                for field in task_fields[:6]
            }
            records.append(
                {
                    "dataset": "WorkArena",
                    "config_name": config_path.stem,
                    "task_id": f"{config_path.stem}:{index}",
                    "task_fields": task_fields,
                    "num_task_fields": len(task_fields),
                    "num_template_fields": len(template_record),
                    "selected_template_values": selected_values,
                    "source_path": str(config_path),
                }
            )
    return records


def build_workarena_task_pairs(
    config_root: str | Path,
    *,
    split: str = "raw",
    limit: int | None = None,
) -> tuple[list[dict], dict]:
    trajectories: list[dict] = []
    audit_records = build_workarena_task_audit_records(config_root)
    for row in _limit_items(audit_records, limit):
        contract = _finalize_contract(
            [
                f"config_name={row['config_name']}",
                f"task_fields={','.join(row['task_fields'][:8])}",
                f"selected_template_values={_compact_json(row['selected_template_values'])}",
            ],
            fallback=str(row["task_id"]),
        )
        short_description = row["selected_template_values"].get("short_description")
        instruction = short_description or f"WorkArena task {row['task_id']}"
        metadata_lines = [
            f"config_name={row['config_name']}",
            f"num_task_fields={row['num_task_fields']}",
            f"num_template_fields={row['num_template_fields']}",
        ]
        success, failure = _build_pair(
            dataset="WorkArena",
            task_id=str(row["task_id"]),
            instruction=instruction,
            metadata_lines=metadata_lines,
            positive_contract=contract,
            split=split,
            source_path=str(row["source_path"]),
            metadata={
                "config_name": row["config_name"],
                "task_fields": list(row["task_fields"]),
                "num_task_fields": row["num_task_fields"],
                "num_template_fields": row["num_template_fields"],
            },
        )
        trajectories.extend([success, failure])
    summary = _summarize_dataset(
        dataset="WorkArena",
        trajectories=trajectories,
        notes=[
            "This adapter uses public task configs only; no ServiceNow instance rollouts are bundled locally.",
            "Synthetic failures are deterministic mutations of selected task-field contracts.",
        ],
    )
    return trajectories, summary
