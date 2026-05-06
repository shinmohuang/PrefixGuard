from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _bundle_fields_from_stem(stem: str) -> tuple[str, str, str, str]:
    parts = stem.split("_")
    if len(parts) >= 5:
        agent_tag = parts[0]
        domain_tag = parts[1]
        policy_variant = parts[2]
        user_tag = parts[3]
        return agent_tag, domain_tag, policy_variant, user_tag
    return "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"


def _compact_text(value: Any, *, max_chars: int = 800) -> str:
    if value is None:
        text = "NONE"
    elif isinstance(value, str):
        text = " ".join(value.split()) or "NONE"
    else:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 16]}...<truncated>"


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _stringify_tool_payload(value: Any, *, max_chars: int = 1600) -> str:
    if value is None:
        text = "NONE"
    elif isinstance(value, str):
        text = value.strip() or "NONE"
    else:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 16]}...<truncated>"


def _render_history_line(message: dict[str, Any], *, max_chars: int = 400) -> str:
    role = str(message.get("role", "unknown")).lower()
    if role == "tool":
        name = message.get("requestor") or message.get("id") or "tool"
        prefix = f"tool[{name}]"
    else:
        prefix = role
    content = _compact_text(message.get("content"), max_chars=max_chars)
    return f"{prefix}: {content}"


def _render_follow_up_messages(
    messages: list[dict[str, Any]],
    *,
    max_messages: int = 6,
) -> str:
    if not messages:
        return "NONE"
    lines = [_render_history_line(message) for message in messages[:max_messages]]
    if len(messages) > max_messages:
        lines.append(f"... {len(messages) - max_messages} more message(s)")
    return "\n".join(lines)


def _build_context(
    *,
    dataset: str,
    domain: str,
    task_id: str,
    trial: int,
    policy_variant: str,
    agent_model: str,
    user_model: str,
    task: dict[str, Any],
    history: list[dict[str, Any]],
    history_window: int,
) -> str:
    lines = [
        f"dataset={dataset}",
        f"domain={domain}",
        f"task_id={task_id}",
        f"trial={trial}",
        f"policy_variant={policy_variant}",
        f"agent_model={agent_model}",
        f"user_model={user_model}",
        f"task_description={_compact_text(task.get('description'), max_chars=600)}",
    ]
    user_scenario = task.get("user_scenario")
    if user_scenario:
        lines.append(f"user_scenario={_compact_text(user_scenario, max_chars=600)}")
    ticket = task.get("ticket")
    if ticket:
        lines.append(f"ticket={_compact_text(ticket, max_chars=600)}")
    lines.append("observation=")
    recent_history = history[-history_window:] if history_window > 0 else history
    if recent_history:
        lines.extend(_render_history_line(message) for message in recent_history)
    else:
        lines.append("NONE")
    return "\n".join(lines)


def _normalize_failure_basis(value: Any) -> str:
    if value in (None, "", [], ()):
        return ""
    parsed = value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ""
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                parsed = raw
        else:
            parsed = raw

    if isinstance(parsed, (list, tuple, set)):
        parts = [str(item).strip().upper().replace(" ", "_") for item in parsed if str(item).strip()]
    else:
        parts = [str(parsed).strip().upper().replace(" ", "_")]

    cleaned = []
    for part in parts:
        normalized = part.replace("-", "_").replace("/", "_")
        normalized = normalized.strip("_")
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return "_".join(cleaned)


def _failure_bucket(*, final_success: bool, reward_info: dict[str, Any], termination_reason: str) -> str:
    if final_success:
        return "NONE"
    basis = _normalize_failure_basis(reward_info.get("reward_basis"))
    if basis:
        return f"TAU2_{basis}"
    normalized_reason = termination_reason.strip().upper().replace("-", "_").replace(" ", "_")
    if normalized_reason:
        return f"TAU2_{normalized_reason}"
    return "TAU2_REWARD_ZERO"


def _tool_action_text(content: Any, tool_name: str) -> str:
    content_text = _compact_text(content, max_chars=500)
    if content_text == "NONE":
        return f"tool_call::{tool_name}"
    return f"{content_text}\nTOOL_CALL {tool_name}"


def _assistant_action_text(content: Any) -> str:
    content_text = _compact_text(content, max_chars=800)
    if content_text == "NONE":
        return "assistant_response"
    return content_text


def _build_steps(
    *,
    messages: list[dict[str, Any]],
    dataset: str,
    domain: str,
    task_id: str,
    task: dict[str, Any],
    trial: int,
    policy_variant: str,
    agent_model: str,
    user_model: str,
    history_window: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    assistant_indices = [index for index, message in enumerate(messages) if message.get("role") == "assistant"]
    for assistant_pos, start_index in enumerate(assistant_indices):
        next_index = (
            assistant_indices[assistant_pos + 1]
            if assistant_pos + 1 < len(assistant_indices)
            else len(messages)
        )
        assistant_message = messages[start_index]
        follow_ups = messages[start_index + 1 : next_index]
        recent_history = messages[:start_index]
        rendered_history = recent_history[-history_window:] if history_window > 0 else recent_history
        source_raw_text = _stable_json(
            {
                "dataset": dataset,
                "domain": domain,
                "task_id": task_id,
                "trial": trial,
                "policy_variant": policy_variant,
                "agent_model": agent_model,
                "user_model": user_model,
                "task": {
                    "id": task.get("id"),
                    "description": task.get("description"),
                    "user_scenario": task.get("user_scenario"),
                    "ticket": task.get("ticket"),
                },
                "history": rendered_history,
                "assistant_message": assistant_message,
                "follow_ups": follow_ups,
            }
        )
        context = _build_context(
            dataset=dataset,
            domain=domain,
            task_id=task_id,
            trial=trial,
            policy_variant=policy_variant,
            agent_model=agent_model,
            user_model=user_model,
            task=task,
            history=messages[:start_index],
            history_window=history_window,
        )
        tool_calls = assistant_message.get("tool_calls") or []
        if tool_calls:
            tool_messages_by_id: dict[str, list[dict[str, Any]]] = {}
            unmatched_tool_messages: list[dict[str, Any]] = []
            for message in follow_ups:
                if message.get("role") != "tool":
                    continue
                tool_id = str(message.get("id") or "").strip()
                if tool_id:
                    tool_messages_by_id.setdefault(tool_id, []).append(message)
                else:
                    unmatched_tool_messages.append(message)

            for tool_call in tool_calls:
                tool_name = str(tool_call.get("name") or "tool_call")
                call_id = str(tool_call.get("id") or "").strip()
                matched = list(tool_messages_by_id.get(call_id, ()))
                if not matched and unmatched_tool_messages:
                    matched = [unmatched_tool_messages.pop(0)]
                status = "tool_error" if any(bool(message.get("error")) for message in matched) else "ok"
                steps.append(
                    {
                        "context": context,
                        "action_text": _tool_action_text(assistant_message.get("content"), tool_name),
                        "tool_name": tool_name,
                        "tool_args": dict(tool_call.get("arguments") or {}),
                        "result_text": _render_follow_up_messages(matched),
                        "status": status,
                        "source_raw_text": source_raw_text,
                    }
                )
            continue

        steps.append(
            {
                "context": context,
                "action_text": _assistant_action_text(assistant_message.get("content")),
                "tool_name": "respond",
                "tool_args": {},
                "result_text": _render_follow_up_messages(follow_ups),
                "status": "ok",
                "source_raw_text": source_raw_text,
            }
        )

    if not steps:
        raise ValueError(f"Unable to extract assistant-driven steps for task {task_id}")
    return steps


def parse_tau2_result_file(
    path: str | Path,
    *,
    split: str = "raw",
    history_window: int = 12,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_path = Path(path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    info = dict(payload.get("info", {}))
    environment_info = dict(info.get("environment_info", {}))
    agent_info = dict(info.get("agent_info", {}))
    user_info = dict(info.get("user_info", {}))
    domain = str(environment_info.get("domain_name") or result_path.stem.split("_")[1])
    _, stem_domain, policy_variant, _ = _bundle_fields_from_stem(result_path.stem)
    policy_variant = str(policy_variant or "UNKNOWN")
    agent_model = str(agent_info.get("llm") or agent_info.get("implementation") or "UNKNOWN")
    user_model = str(user_info.get("llm") or user_info.get("implementation") or "UNKNOWN")
    tasks_by_id = {str(task.get("id")): task for task in payload.get("tasks", [])}
    if domain == "UNKNOWN" and stem_domain != "UNKNOWN":
        domain = stem_domain

    imported: list[dict[str, Any]] = []
    summary = {
        "source_file": str(result_path),
        "dataset": "TAU2Bench",
        "domain": domain,
        "policy_variant": policy_variant,
        "agent_model": agent_model,
        "user_model": user_model,
        "num_imported": 0,
        "num_steps": 0,
        "num_successes": 0,
        "num_failures": 0,
        "failure_buckets": {},
        "missing_task_ids": [],
    }
    failure_counter: Counter[str] = Counter()
    missing_task_ids: set[str] = set()

    for simulation in payload.get("simulations", []):
        task_id = str(simulation.get("task_id"))
        task = tasks_by_id.get(task_id)
        if task is None:
            missing_task_ids.add(task_id)
            continue
        reward_info = dict(simulation.get("reward_info", {}))
        reward = float(reward_info.get("reward", 0.0) or 0.0)
        final_success = reward >= 1.0 - 1e-9
        termination_reason = str(simulation.get("termination_reason") or "UNKNOWN")
        steps = _build_steps(
            messages=list(simulation.get("messages", [])),
            dataset="TAU2Bench",
            domain=domain,
            task_id=f"tau2-{domain}-{task_id}",
            task=task,
            trial=int(simulation.get("trial", 0) or 0),
            policy_variant=policy_variant,
            agent_model=agent_model,
            user_model=user_model,
            history_window=history_window,
        )
        steps[-1]["status"] = "success" if final_success else "failure"
        failure_bucket = _failure_bucket(
            final_success=final_success,
            reward_info=reward_info,
            termination_reason=termination_reason,
        )
        record = {
            "trajectory_id": (
                f"tau2-{domain}-{result_path.stem}-task{task_id}-trial{simulation.get('trial', 0)}"
            ),
            "task_id": f"tau2-{domain}-{task_id}",
            "final_success": final_success,
            "failure_bucket": failure_bucket,
            "split": split,
            "metadata": {
                "dataset": "TAU2Bench",
                "adapter_origin": "tau2_results_final",
                "label_origin": "official_reward_info.reward",
                "source_file": str(result_path),
                "source_stem": result_path.stem,
                "simulation_id": str(simulation.get("id")),
                "task_numeric_id": task_id,
                "trial": int(simulation.get("trial", 0) or 0),
                "seed": int(simulation.get("seed", 0) or 0),
                "timestamp": simulation.get("timestamp"),
                "duration": simulation.get("duration"),
                "termination_reason": termination_reason,
                "raw_reward": reward,
                "reward_basis": reward_info.get("reward_basis"),
                "reward_breakdown": reward_info.get("reward_breakdown"),
                "domain": domain,
                "policy_variant": policy_variant,
                "agent_model": agent_model,
                "agent_implementation": agent_info.get("implementation"),
                "user_model": user_model,
                "user_implementation": user_info.get("implementation"),
                "git_commit": info.get("git_commit"),
                "num_trials_in_bundle": info.get("num_trials"),
                "task_description": task.get("description"),
                "user_scenario": task.get("user_scenario"),
            },
            "steps": steps,
        }
        imported.append(record)
        summary["num_steps"] += len(steps)
        summary["num_successes"] += int(final_success)
        summary["num_failures"] += int(not final_success)
        failure_counter[failure_bucket] += 1
        if limit is not None and len(imported) >= limit:
            break

    summary["num_imported"] = len(imported)
    summary["failure_buckets"] = dict(sorted(failure_counter.items()))
    summary["missing_task_ids"] = sorted(missing_task_ids)
    return imported, summary


def parse_tau2_results_dir(
    root: str | Path,
    *,
    pattern: str = "*.json",
    split: str = "raw",
    history_window: int = 12,
    limit_files: int | None = None,
    limit_trajectories: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root_path = Path(root)
    imported: list[dict[str, Any]] = []
    aggregate_failure_buckets: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    files_parsed: list[str] = []
    total_steps = 0
    total_successes = 0
    total_failures = 0

    for file_index, path in enumerate(sorted(root_path.glob(pattern))):
        if limit_files is not None and file_index >= limit_files:
            break
        remaining = None
        if limit_trajectories is not None:
            remaining = max(limit_trajectories - len(imported), 0)
            if remaining == 0:
                break
        records, summary = parse_tau2_result_file(
            path,
            split=split,
            history_window=history_window,
            limit=remaining,
        )
        imported.extend(records)
        total_steps += int(summary["num_steps"])
        total_successes += int(summary["num_successes"])
        total_failures += int(summary["num_failures"])
        aggregate_failure_buckets.update(summary["failure_buckets"])
        domains[summary["domain"]] += int(summary["num_imported"])
        files_parsed.append(str(path))

    payload = {
        "dataset": "TAU2Bench",
        "input_root": str(root_path),
        "pattern": pattern,
        "split": split,
        "num_files_parsed": len(files_parsed),
        "source_files": files_parsed,
        "num_trajectories": len(imported),
        "num_steps": total_steps,
        "avg_steps_per_trajectory": (total_steps / len(imported)) if imported else 0.0,
        "num_successes": total_successes,
        "num_failures": total_failures,
        "domain_counts": dict(sorted(domains.items())),
        "failure_buckets": dict(sorted(aggregate_failure_buckets.items())),
    }
    return imported, payload
