from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = (
    REPO_ROOT / "data" / "external" / "skillsbench" / "skillsbench-trajectories"
)
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "data" / "interim" / "skillsbench" / "full_repo_main_traces.jsonl"
DEFAULT_OUTPUT_SUMMARY = (
    REPO_ROOT / "data" / "interim" / "skillsbench" / "full_repo_main_traces_summary.json"
)
DEFAULT_SOURCE_RAW_MAX_STRING_CHARS = 2048
DEFAULT_SOURCE_RAW_MAX_LIST_ITEMS = 16
DEFAULT_SOURCE_RAW_HISTORY_ITEMS = 4
_TRUNCATION_MARKER = "source_raw_truncated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the main agent traces from SkillsBench raw trajectory dumps and convert them "
            "into TrajectoryRecord JSONL for this repository."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing trial folders with result.json/config.json.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="raw",
        help="Split tag written into each imported trajectory.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=6,
        help="Number of rendered history lines to keep in Claude-style step context.",
    )
    parser.add_argument(
        "--max-action-chars",
        type=int,
        default=1200,
    )
    parser.add_argument(
        "--max-result-chars",
        type=int,
        default=2400,
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on imported trajectories for smoke testing.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on the first parsing failure instead of recording it in the summary.",
    )
    return parser.parse_args()


def _compact_text(value: Any, *, max_chars: int) -> str:
    if value is None:
        text = "NONE"
    elif isinstance(value, str):
        text = " ".join(value.split()) or "NONE"
    else:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 16]}...<truncated>"


def _sanitize_multiline(value: Any, *, max_chars: int) -> str:
    if value is None:
        text = "NONE"
    elif isinstance(value, str):
        text = value.strip() or "NONE"
    else:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 16]}...<truncated>"


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _normalize_tool_name(name: Any) -> str:
    raw = str(name or "tool_call").strip()
    lowered = raw.lower() or "tool_call"
    alias_map = {
        "bash": "bash",
        "run_shell_command": "bash",
        "read": "read_file",
        "read_file": "read_file",
        "write": "write_file",
        "write_file": "write_file",
        "edit": "replace",
        "replace": "replace",
    }
    return alias_map.get(lowered, lowered)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _stable_choice(parts: list[str], *, seed: int = 7) -> str:
    digest = hashlib.sha256(f"{seed}:{'||'.join(parts)}".encode("utf-8")).hexdigest()
    return digest


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _truncate_source_raw_string(text: str, *, max_chars: int, stats: Counter[str]) -> str:
    if len(text) <= max_chars:
        return text
    stats["string_values"] += 1
    marker = f"...<{_TRUNCATION_MARKER}:chars={len(text) - max_chars}>..."
    budget = max(0, max_chars - len(marker))
    head_chars = max(0, budget // 2)
    tail_chars = max(0, budget - head_chars)
    return f"{text[:head_chars]}{marker}{text[-tail_chars:] if tail_chars else ''}"


def _truncate_source_raw_value(
    value: Any,
    *,
    max_string_chars: int = DEFAULT_SOURCE_RAW_MAX_STRING_CHARS,
    max_list_items: int = DEFAULT_SOURCE_RAW_MAX_LIST_ITEMS,
    stats: Counter[str] | None = None,
) -> Any:
    if stats is None:
        stats = Counter()
    if isinstance(value, str):
        return _truncate_source_raw_string(value, max_chars=max_string_chars, stats=stats)
    if isinstance(value, list):
        values = value
        if len(values) > max_list_items:
            stats["list_values"] += 1
            head_count = max_list_items // 2
            tail_count = max_list_items - head_count
            values = [
                *values[:head_count],
                {"source_raw_truncated_items": len(value) - max_list_items},
                *values[-tail_count:],
            ]
        return [
            _truncate_source_raw_value(
                item,
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
                stats=stats,
            )
            for item in values
        ]
    if isinstance(value, dict):
        return {
            str(key): _truncate_source_raw_value(
                item,
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
                stats=stats,
            )
            for key, item in value.items()
        }
    return value


def _source_raw_history_tail(items: list[Any]) -> list[Any]:
    if DEFAULT_SOURCE_RAW_HISTORY_ITEMS <= 0:
        return []
    return items[-DEFAULT_SOURCE_RAW_HISTORY_ITEMS:]


def _source_raw_text(
    *,
    base_metadata: dict[str, Any],
    task_id: str,
    step_index: int,
    raw_fragment: dict[str, Any],
) -> str:
    stats: Counter[str] = Counter()
    payload = _truncate_source_raw_value(
        {
            "dataset": "SkillsBench",
            "trace_format": base_metadata.get("trace_format"),
            "task_id": task_id,
            "trial_name": base_metadata.get("trial_name"),
            "condition_dir": base_metadata.get("condition_dir"),
            "step_index": step_index,
            **raw_fragment,
        },
        stats=stats,
    )
    if stats:
        payload["source_raw_truncation"] = {
            "history_items": DEFAULT_SOURCE_RAW_HISTORY_ITEMS,
            "list_values": int(stats["list_values"]),
            "max_list_items": DEFAULT_SOURCE_RAW_MAX_LIST_ITEMS,
            "max_string_chars": DEFAULT_SOURCE_RAW_MAX_STRING_CHARS,
            "string_values": int(stats["string_values"]),
        }
    return _stable_json(payload)


def _source_raw_protocol_summary() -> dict[str, Any]:
    return {
        "max_history_items": DEFAULT_SOURCE_RAW_HISTORY_ITEMS,
        "max_list_items": DEFAULT_SOURCE_RAW_MAX_LIST_ITEMS,
        "max_string_chars": DEFAULT_SOURCE_RAW_MAX_STRING_CHARS,
        "scope": "SkillsBench source_raw_text only",
    }


def _result_rewards(result_payload: dict[str, Any]) -> dict[str, Any]:
    verifier_result = _as_dict(result_payload.get("verifier_result"))
    verifier_rewards = _as_dict(verifier_result.get("rewards"))
    if verifier_rewards:
        return verifier_rewards
    return _as_dict(result_payload.get("rewards"))


def _reward_label_origin(result_payload: dict[str, Any]) -> str:
    if _as_dict(result_payload.get("rewards")):
        return "result.rewards.reward"
    return "result.verifier_result.rewards.reward"


def _condition_dir_for_trial(trial_dir: Path) -> str:
    parts = trial_dir.parts
    if "jobs" in parts:
        jobs_index = parts.index("jobs")
        if jobs_index + 1 < len(parts):
            return parts[jobs_index + 1]
    return trial_dir.parent.name



def _task_id_from_payload(result_payload: dict[str, Any], config_payload: dict[str, Any]) -> str:
    result_task_id = _as_dict(result_payload.get("task_id"))
    result_config = _as_dict(result_payload.get("config"))
    result_config_task = _as_dict(result_config.get("task"))
    config_task = _as_dict(config_payload.get("task"))
    task_name = _first_non_empty(result_payload.get("task_name"))
    if task_name:
        return f"skillsbench::{task_name}"
    task_path = _first_non_empty(
        result_task_id.get("path"),
        result_config_task.get("path"),
        config_task.get("path"),
        config_payload.get("task_path"),
    )
    if task_path:
        return f"skillsbench::{Path(task_path).name}"
    trial_name = _first_non_empty(result_payload.get("trial_name"), config_payload.get("trial_name"))
    if trial_name:
        return f"skillsbench::{trial_name.split('__')[0]}"
    raise ValueError("Unable to determine task_id for SkillsBench trial")


def _trajectory_id_for_trial(
    *,
    trial_dir: Path,
    result_payload: dict[str, Any],
    config_payload: dict[str, Any],
) -> str:
    condition_dir = _condition_dir_for_trial(trial_dir)
    trial_name = _first_non_empty(result_payload.get("trial_name"), config_payload.get("trial_name"), trial_dir.name)
    return f"skillsbench::{condition_dir}::{trial_name}"


def _failure_bucket(
    *,
    final_success: bool,
    result_payload: dict[str, Any],
    trial_dir: Path,
) -> str:
    if final_success:
        return "NONE"
    if result_payload.get("exception_info") not in (None, "", {}) or result_payload.get("error") not in (None, "", {}):
        return "SKILLSBENCH_AGENT_EXCEPTION"
    exception_txt = trial_dir / "exception.txt"
    if exception_txt.exists() and exception_txt.read_text(encoding="utf-8", errors="replace").strip():
        return "SKILLSBENCH_AGENT_EXCEPTION"
    rewards = _result_rewards(result_payload)
    raw_reward = float(rewards.get("reward", 0.0) or 0.0)
    if raw_reward <= 0.0:
        return "SKILLSBENCH_VERIFIER_FAIL"
    return "SKILLSBENCH_PARTIAL_REWARD"


def _build_common_metadata(
    *,
    trial_dir: Path,
    result_payload: dict[str, Any],
    config_payload: dict[str, Any],
    trace_path: Path,
    trace_format: str,
    raw_reward: float,
) -> dict[str, Any]:
    agent_config = _as_dict(config_payload.get("agent"))
    agent_info = _as_dict(result_payload.get("agent_info"))
    task_config = _as_dict(config_payload.get("task"))
    environment_config = _as_dict(config_payload.get("environment"))
    agent_env = _as_dict(config_payload.get("agent_env"))
    agent_kwargs = _as_dict(agent_config.get("kwargs"))
    rewards = _result_rewards(result_payload)
    result_task_id = _as_dict(result_payload.get("task_id"))
    environment_value = config_payload.get("environment")
    environment_type = _first_non_empty(
        environment_config.get("type"),
        environment_value if isinstance(environment_value, str) else None,
    )
    return {
        "dataset": "SkillsBench",
        "adapter_origin": "skillsbench_main_trace",
        "label_origin": _reward_label_origin(result_payload),
        "trace_format": trace_format,
        "source_trace": str(trace_path),
        "source_result": str(trial_dir / "result.json"),
        "source_config": str(trial_dir / "config.json"),
        "condition_dir": _condition_dir_for_trial(trial_dir),
        "trial_dir": str(trial_dir),
        "trial_name": _first_non_empty(result_payload.get("trial_name"), config_payload.get("trial_name"), trial_dir.name),
        "task_name": _first_non_empty(result_payload.get("task_name")),
        "task_source": _first_non_empty(result_payload.get("source"), task_config.get("source")),
        "task_path": _first_non_empty(
            result_task_id.get("path"),
            task_config.get("path"),
            config_payload.get("task_path"),
        ),
        "agent_name": _first_non_empty(
            agent_info.get("name"),
            agent_config.get("name"),
            result_payload.get("agent_name"),
            result_payload.get("agent"),
            config_payload.get("agent") if isinstance(config_payload.get("agent"), str) else None,
        ),
        "agent_model_name": _first_non_empty(
            agent_config.get("model_name"),
            config_payload.get("model"),
            result_payload.get("model"),
            agent_env.get("BENCHFLOW_PROVIDER_MODEL"),
            agent_env.get("ANTHROPIC_MODEL"),
        ),
        "agent_version": _first_non_empty(agent_info.get("version"), agent_kwargs.get("version")),
        "environment_type": environment_type,
        "raw_reward": float(rewards.get("reward", raw_reward) or raw_reward or 0.0),
        "started_at": result_payload.get("started_at"),
        "finished_at": result_payload.get("finished_at"),
        "exception_info": result_payload.get("exception_info") or result_payload.get("error"),
    }


def _render_role_content(content: Any, *, max_chars: int) -> str:
    if isinstance(content, str):
        return _compact_text(content, max_chars=max_chars)
    if isinstance(content, list):
        rendered: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                rendered.append(_compact_text(block, max_chars=max_chars))
                continue
            block_type = str(block.get("type", "unknown"))
            if block_type == "text":
                rendered.append(_compact_text(block.get("text"), max_chars=max_chars))
            elif block_type == "tool_result":
                rendered.append(
                    f"tool_result[{block.get('tool_use_id', 'UNKNOWN')}] "
                    f"{_compact_text(block.get('content'), max_chars=max_chars)}"
                )
            elif block_type == "tool_use":
                rendered.append(
                    f"tool_use[{block.get('name', 'UNKNOWN')}] "
                    f"{_compact_text(block.get('input'), max_chars=max_chars)}"
                )
            else:
                rendered.append(f"{block_type}: {_compact_text(block, max_chars=max_chars)}")
        return " | ".join(rendered) if rendered else "NONE"
    return _compact_text(content, max_chars=max_chars)


def _render_history_line(message: dict[str, Any], *, max_chars: int) -> str:
    role = str(message.get("type", "unknown")).lower()
    if role == "assistant":
        role = "assistant"
        content = _as_dict(message.get("message")).get("content")
    elif role == "user":
        role = "user"
        content = _as_dict(message.get("message")).get("content")
    else:
        content = message
    return f"{role}: {_render_role_content(content, max_chars=max_chars)}"


def _extract_text_blocks(content: Any, *, max_chars: int) -> str:
    if isinstance(content, str):
        return _compact_text(content, max_chars=max_chars)
    if not isinstance(content, list):
        return "NONE"
    texts = [
        _compact_text(block.get("text"), max_chars=max_chars)
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
    ]
    return "\n".join(texts) if texts else "NONE"


def _render_tool_result_blocks(
    matched_blocks: list[dict[str, Any]],
    *,
    max_chars: int,
    extra_lines: list[str] | None = None,
) -> str:
    lines: list[str] = []
    for block in matched_blocks:
        payload = block.get("content")
        rendered = _sanitize_multiline(payload, max_chars=max_chars)
        prefix = "error" if bool(block.get("is_error")) else "result"
        lines.append(f"{prefix}={rendered}")
    if extra_lines:
        lines.extend(extra_lines)
    return "\n".join(lines) if lines else "NONE"


def _build_claude_context(
    *,
    base_metadata: dict[str, Any],
    task_id: str,
    initial_prompt: str,
    history: list[dict[str, Any]],
    history_window: int,
    step_index: int,
    max_chars: int,
) -> str:
    lines = [
        "dataset=SkillsBench",
        f"task_id={task_id}",
        f"trial_name={base_metadata.get('trial_name')}",
        f"condition_dir={base_metadata.get('condition_dir')}",
        f"agent_name={base_metadata.get('agent_name')}",
        f"agent_model_name={base_metadata.get('agent_model_name')}",
        f"trace_format={base_metadata.get('trace_format')}",
        f"step_index={step_index}",
        f"initial_prompt={_compact_text(initial_prompt, max_chars=900)}",
        "dialogue_history=",
    ]
    recent_history = history[-history_window:] if history_window > 0 else history
    if recent_history:
        lines.extend(_render_history_line(message, max_chars=240) for message in recent_history)
    else:
        lines.append("NONE")
    context = "\n".join(lines)
    return context if len(context) <= max_chars else context[: max_chars - 16] + "...<truncated>"


def _extract_claude_initial_prompt(events: list[dict[str, Any]], *, max_chars: int) -> str:
    for event in events:
        if event.get("type") != "user":
            continue
        content = _as_dict(event.get("message")).get("content")
        if isinstance(content, str) and content.strip():
            return _compact_text(content, max_chars=max_chars)
    return "NONE"


def _build_steps_from_claude_events(
    *,
    events: list[dict[str, Any]],
    base_metadata: dict[str, Any],
    task_id: str,
    history_window: int,
    max_action_chars: int,
    max_result_chars: int,
    max_context_chars: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    initial_prompt = _extract_claude_initial_prompt(events, max_chars=1200)
    assistant_indices = [index for index, event in enumerate(events) if event.get("type") == "assistant"]

    for assistant_position, start_index in enumerate(assistant_indices):
        end_index = assistant_indices[assistant_position + 1] if assistant_position + 1 < len(assistant_indices) else len(events)
        assistant_event = events[start_index]
        follow_ups = events[start_index + 1 : end_index]
        assistant_content = _as_dict(assistant_event.get("message")).get("content")
        if not isinstance(assistant_content, list):
            assistant_content = []

        context = _build_claude_context(
            base_metadata=base_metadata,
            task_id=task_id,
            initial_prompt=initial_prompt,
            history=events[:start_index],
            history_window=history_window,
            step_index=len(steps) + 1,
            max_chars=max_context_chars,
        )

        tool_uses = [block for block in assistant_content if isinstance(block, dict) and block.get("type") == "tool_use"]
        assistant_text = _extract_text_blocks(assistant_content, max_chars=max_action_chars)

        if tool_uses:
            matched_by_id: dict[str, list[dict[str, Any]]] = {}
            matched_events_by_id: dict[str, list[dict[str, Any]]] = {}
            unmatched_followups: list[dict[str, Any]] = []
            for follow_up in follow_ups:
                if follow_up.get("type") != "user":
                    continue
                content = _as_dict(follow_up.get("message")).get("content")
                if isinstance(content, list) and content:
                    any_matched = False
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "tool_result" and block.get("tool_use_id"):
                            tool_use_id = str(block["tool_use_id"])
                            matched_by_id.setdefault(tool_use_id, []).append(block)
                            matched_events_by_id.setdefault(tool_use_id, []).append(follow_up)
                            any_matched = True
                    if any_matched:
                        continue
                unmatched_followups.append(follow_up)

            for tool_use in tool_uses:
                tool_name = _normalize_tool_name(tool_use.get("name") or "tool_call")
                tool_use_id = str(tool_use.get("id") or "")
                matched_blocks = list(matched_by_id.get(tool_use_id, ()))
                matched_events = list(matched_events_by_id.get(tool_use_id, ()))
                extra_lines = (
                    [_render_history_line(follow_up, max_chars=300) for follow_up in unmatched_followups]
                    if len(tool_uses) == 1
                    else None
                )
                result_text = _render_tool_result_blocks(
                    matched_blocks,
                    max_chars=max_result_chars,
                    extra_lines=extra_lines,
                )
                status = "tool_error" if any(bool(block.get("is_error")) for block in matched_blocks) else "ok"
                action_text = (
                    f"{assistant_text}\nTOOL_CALL {tool_name}"
                    if assistant_text != "NONE"
                    else f"tool_call::{tool_name}"
                )
                steps.append(
                    {
                        "context": context,
                        "action_text": _compact_text(action_text, max_chars=max_action_chars),
                        "tool_name": tool_name,
                        "tool_args": dict(tool_use.get("input") or {}),
                        "result_text": result_text,
                        "status": status,
                        "source_raw_text": _source_raw_text(
                            base_metadata=base_metadata,
                            task_id=task_id,
                            step_index=len(steps) + 1,
                            raw_fragment={
                                "history": _source_raw_history_tail(events[:start_index]),
                                "assistant_event": assistant_event,
                                "tool_use_index": assistant_content.index(tool_use),
                                "tool_use": tool_use,
                                "matched_follow_ups": matched_events,
                                "matched_tool_result_blocks": matched_blocks,
                                "unmatched_follow_ups": unmatched_followups
                                if extra_lines is not None
                                else [],
                            },
                        ),
                    }
                )
            continue

        if assistant_text == "NONE":
            continue

        follow_up_lines = [
            _render_history_line(follow_up, max_chars=300)
            for follow_up in follow_ups
            if follow_up.get("type") == "user"
        ]
        steps.append(
            {
                "context": context,
                "action_text": assistant_text,
                "tool_name": "respond",
                "tool_args": {},
                "result_text": "\n".join(follow_up_lines) if follow_up_lines else "NONE",
                "status": "ok",
                "source_raw_text": _source_raw_text(
                    base_metadata=base_metadata,
                    task_id=task_id,
                    step_index=len(steps) + 1,
                    raw_fragment={
                        "history": _source_raw_history_tail(events[:start_index]),
                        "assistant_event": assistant_event,
                        "follow_ups": follow_ups,
                    },
                ),
            }
        )

    return steps


def _parse_claude_trace(
    trace_path: Path,
    *,
    base_metadata: dict[str, Any],
    task_id: str,
    history_window: int,
    max_action_chars: int,
    max_result_chars: int,
    max_context_chars: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") in {"assistant", "user"}:
                events.append(event)
    return _build_steps_from_claude_events(
        events=events,
        base_metadata=base_metadata,
        task_id=task_id,
        history_window=history_window,
        max_action_chars=max_action_chars,
        max_result_chars=max_result_chars,
        max_context_chars=max_context_chars,
    )


def _extract_codex_task_prompt(config_payload: dict[str, Any], *, max_chars: int) -> str:
    task = config_payload.get("task", {})
    parts = [
        _first_non_empty(task.get("path")),
        _first_non_empty(task.get("source")),
        _first_non_empty(config_payload.get("trial_name")),
    ]
    rendered = " | ".join(part for part in parts if part)
    return _compact_text(rendered or "NONE", max_chars=max_chars)


def _build_codex_context(
    *,
    base_metadata: dict[str, Any],
    task_id: str,
    task_prompt: str,
    step_index: int,
    recent_reasoning: str | None,
    max_chars: int,
) -> str:
    lines = [
        "dataset=SkillsBench",
        f"task_id={task_id}",
        f"trial_name={base_metadata.get('trial_name')}",
        f"condition_dir={base_metadata.get('condition_dir')}",
        f"agent_name={base_metadata.get('agent_name')}",
        f"agent_model_name={base_metadata.get('agent_model_name')}",
        f"trace_format={base_metadata.get('trace_format')}",
        f"step_index={step_index}",
        f"task_prompt={task_prompt}",
    ]
    if recent_reasoning:
        lines.append(f"recent_reasoning={_compact_text(recent_reasoning, max_chars=900)}")
    context = "\n".join(lines)
    return context if len(context) <= max_chars else context[: max_chars - 16] + "...<truncated>"


def _parse_codex_trace(
    trace_path: Path,
    *,
    base_metadata: dict[str, Any],
    config_payload: dict[str, Any],
    task_id: str,
    max_action_chars: int,
    max_result_chars: int,
    max_context_chars: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    recent_reasoning: str | None = None
    recent_reasoning_event: dict[str, Any] | None = None
    task_prompt = _extract_codex_task_prompt(config_payload, max_chars=1200)

    with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            event_type = event.get("type")
            if event_type != "item.completed":
                continue
            item = dict(event.get("item") or {})
            item_type = item.get("type")
            if item_type == "reasoning":
                recent_reasoning = _compact_text(item.get("text"), max_chars=1000)
                recent_reasoning_event = event
                continue
            if item_type != "command_execution":
                continue

            exit_code = item.get("exit_code")
            status = "ok" if exit_code == 0 else "tool_error"
            steps.append(
                {
                    "context": _build_codex_context(
                        base_metadata=base_metadata,
                        task_id=task_id,
                        task_prompt=task_prompt,
                        step_index=len(steps) + 1,
                        recent_reasoning=recent_reasoning,
                        max_chars=max_context_chars,
                    ),
                    "action_text": _compact_text(str(item.get("command") or "command_execution"), max_chars=max_action_chars),
                    "tool_name": _normalize_tool_name("bash"),
                    "tool_args": {
                        "command": str(item.get("command") or ""),
                        "exit_code": exit_code,
                    },
                    "result_text": _sanitize_multiline(item.get("aggregated_output"), max_chars=max_result_chars),
                    "status": status,
                    "source_raw_text": _source_raw_text(
                        base_metadata=base_metadata,
                        task_id=task_id,
                        step_index=len(steps) + 1,
                        raw_fragment={
                            "raw_event": event,
                            "recent_reasoning_event": recent_reasoning_event,
                        },
                    ),
                }
            )
            recent_reasoning = None
            recent_reasoning_event = None

    return steps


def _maybe_parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        candidates.append(stripped[first_brace : last_brace + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _extract_trajectory_task_prompt(config_payload: dict[str, Any], *, max_chars: int) -> str:
    task = _as_dict(config_payload.get("task"))
    parts = [
        _first_non_empty(task.get("path")),
        _first_non_empty(task.get("source")),
        _first_non_empty(config_payload.get("trial_name")),
    ]
    rendered = " | ".join(part for part in parts if part)
    return _compact_text(rendered or "NONE", max_chars=max_chars)


def _build_terminal_batch_context(
    *,
    base_metadata: dict[str, Any],
    task_id: str,
    task_prompt: str,
    step_index: int,
    episode_index: int,
    prompt_preview: str,
    max_chars: int,
) -> str:
    lines = [
        "dataset=SkillsBench",
        f"task_id={task_id}",
        f"trial_name={base_metadata.get('trial_name')}",
        f"condition_dir={base_metadata.get('condition_dir')}",
        f"agent_name={base_metadata.get('agent_name')}",
        f"agent_model_name={base_metadata.get('agent_model_name')}",
        f"trace_format={base_metadata.get('trace_format')}",
        f"step_index={step_index}",
        f"episode_index={episode_index}",
        f"task_prompt={task_prompt}",
        f"episode_prompt={prompt_preview}",
    ]
    context = "\n".join(lines)
    return context if len(context) <= max_chars else context[: max_chars - 16] + "...<truncated>"


def _parse_terminal_batch_trace(
    trace_path: Path,
    *,
    base_metadata: dict[str, Any],
    config_payload: dict[str, Any],
    task_id: str,
    max_action_chars: int,
    max_result_chars: int,
    max_context_chars: int,
) -> list[dict[str, Any]]:
    payload = json.loads(trace_path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(payload, dict) and payload.get("schema_version") == "ATIF-v1.5":
        return _parse_atif_trace_payload(
            payload,
            base_metadata=base_metadata,
            config_payload=config_payload,
            task_id=task_id,
            max_action_chars=max_action_chars,
            max_result_chars=max_result_chars,
            max_context_chars=max_context_chars,
        )
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {trace_path}")
    if not payload:
        return []

    task_prompt = _extract_trajectory_task_prompt(config_payload, max_chars=1200)
    steps: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(payload):
        if not isinstance(episode, dict):
            continue
        response_text = str(episode.get("response") or "").strip()
        if not response_text:
            continue
        parsed_response = _maybe_parse_json_object(response_text)
        prompt_preview = _compact_text(str(episode.get("prompt") or "NONE"), max_chars=1200)

        if parsed_response and parsed_response.get("load_skill"):
            skill_name = str(parsed_response.get("load_skill") or "UNKNOWN")
            steps.append(
                {
                    "context": _build_terminal_batch_context(
                        base_metadata=base_metadata,
                        task_id=task_id,
                        task_prompt=task_prompt,
                        step_index=len(steps) + 1,
                        episode_index=episode_index,
                        prompt_preview=prompt_preview,
                        max_chars=max_context_chars,
                    ),
                    "action_text": _compact_text(f"load_skill::{skill_name}", max_chars=max_action_chars),
                    "tool_name": _normalize_tool_name("load_skill"),
                    "tool_args": {"skill": skill_name},
                    "result_text": _compact_text(response_text, max_chars=max_result_chars),
                    "status": "ok",
                    "source_raw_text": _source_raw_text(
                        base_metadata=base_metadata,
                        task_id=task_id,
                        step_index=len(steps) + 1,
                        raw_fragment={
                            "episode_index": episode_index,
                            "episode": episode,
                            "parsed_response": parsed_response,
                        },
                    ),
                }
            )
            continue

        if parsed_response and isinstance(parsed_response.get("commands"), list):
            analysis = _compact_text(parsed_response.get("analysis"), max_chars=600)
            plan = _compact_text(parsed_response.get("plan"), max_chars=600)
            next_prompt = (
                _compact_text(str(payload[episode_index + 1].get("prompt") or "NONE"), max_chars=max_result_chars)
                if episode_index + 1 < len(payload) and isinstance(payload[episode_index + 1], dict)
                else "NONE"
            )
            for command_index, command in enumerate(parsed_response.get("commands") or []):
                if not isinstance(command, dict):
                    continue
                keystrokes = str(command.get("keystrokes") or "").strip()
                if not keystrokes:
                    continue
                duration = command.get("duration")
                action_prefix = []
                if command_index == 0 and analysis != "NONE":
                    action_prefix.append(f"analysis={analysis}")
                if command_index == 0 and plan != "NONE":
                    action_prefix.append(f"plan={plan}")
                action_prefix.append(keystrokes)
                steps.append(
                    {
                        "context": _build_terminal_batch_context(
                            base_metadata=base_metadata,
                            task_id=task_id,
                            task_prompt=task_prompt,
                            step_index=len(steps) + 1,
                            episode_index=episode_index,
                            prompt_preview=prompt_preview,
                            max_chars=max_context_chars,
                        ),
                        "action_text": _compact_text("\n".join(action_prefix), max_chars=max_action_chars),
                        "tool_name": _normalize_tool_name("bash"),
                        "tool_args": {"keystrokes": keystrokes, "duration": duration},
                        "result_text": next_prompt,
                        "status": "ok",
                        "source_raw_text": _source_raw_text(
                            base_metadata=base_metadata,
                            task_id=task_id,
                            step_index=len(steps) + 1,
                            raw_fragment={
                                "episode_index": episode_index,
                                "episode": episode,
                                "parsed_response": parsed_response,
                                "command_index": command_index,
                                "command": command,
                                "next_episode": payload[episode_index + 1]
                                if episode_index + 1 < len(payload)
                                and isinstance(payload[episode_index + 1], dict)
                                else None,
                            },
                        ),
                    }
                )
            continue

        steps.append(
            {
                "context": _build_terminal_batch_context(
                    base_metadata=base_metadata,
                    task_id=task_id,
                    task_prompt=task_prompt,
                    step_index=len(steps) + 1,
                    episode_index=episode_index,
                    prompt_preview=prompt_preview,
                    max_chars=max_context_chars,
                ),
                "action_text": _compact_text(response_text, max_chars=max_action_chars),
                "tool_name": "respond",
                "tool_args": {},
                "result_text": "NONE",
                "status": "ok",
                "source_raw_text": _source_raw_text(
                    base_metadata=base_metadata,
                    task_id=task_id,
                    step_index=len(steps) + 1,
                    raw_fragment={
                        "episode_index": episode_index,
                        "episode": episode,
                        "parsed_response": parsed_response,
                    },
                ),
            }
        )

    return steps


def _extract_atif_task_prompt(payload: dict[str, Any], config_payload: dict[str, Any], *, max_chars: int) -> str:
    steps = payload.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict) and step.get("source") == "user":
                message = str(step.get("message") or "").strip()
                if message:
                    return _compact_text(message, max_chars=max_chars)
    return _extract_trajectory_task_prompt(config_payload, max_chars=max_chars)


def _parse_atif_trace_payload(
    payload: dict[str, Any],
    *,
    base_metadata: dict[str, Any],
    config_payload: dict[str, Any],
    task_id: str,
    max_action_chars: int,
    max_result_chars: int,
    max_context_chars: int,
) -> list[dict[str, Any]]:
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("ATIF-v1.5 payload missing steps[]")

    task_prompt = _extract_atif_task_prompt(payload, config_payload, max_chars=1200)
    history_lines: list[str] = []
    steps: list[dict[str, Any]] = []
    for raw_step_index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            continue
        source = str(raw_step.get("source") or "").lower()
        message_text = _compact_text(raw_step.get("message"), max_chars=max_action_chars)
        if source == "user":
            history_lines.append(f"user: {message_text}")
            continue
        if source != "agent":
            continue

        tool_calls = raw_step.get("tool_calls")
        observation_results = _as_dict(raw_step.get("observation")).get("results")
        if isinstance(tool_calls, list) and tool_calls:
            result_blocks = observation_results if isinstance(observation_results, list) else []
            for tool_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                tool_name = _normalize_tool_name(tool_call.get("function_name") or "tool_call")
                action_text = (
                    f"{message_text}\nTOOL_CALL {tool_name}"
                    if message_text != "NONE"
                    else f"tool_call::{tool_name}"
                )
                result_payload = result_blocks[tool_index] if tool_index < len(result_blocks) else None
                result_text = _sanitize_multiline(
                    _as_dict(result_payload).get("content"),
                    max_chars=max_result_chars,
                )
                steps.append(
                    {
                        "context": _build_gemini_context(
                            base_metadata=base_metadata,
                            task_id=task_id,
                            task_prompt=task_prompt,
                            step_index=len(steps) + 1,
                            history_lines=history_lines,
                            max_chars=max_context_chars,
                        ),
                        "action_text": _compact_text(action_text, max_chars=max_action_chars),
                        "tool_name": tool_name,
                        "tool_args": _as_dict(tool_call.get("arguments")),
                        "result_text": result_text,
                        "status": "ok",
                        "source_raw_text": _source_raw_text(
                            base_metadata=base_metadata,
                            task_id=task_id,
                            step_index=len(steps) + 1,
                            raw_fragment={
                                "raw_step_index": raw_step_index,
                                "raw_step": raw_step,
                                "history_steps": _source_raw_history_tail(raw_steps[:raw_step_index]),
                                "tool_call_index": tool_index,
                                "tool_call": tool_call,
                                "observation_result": result_payload,
                            },
                        ),
                    }
                )
            history_lines.append(f"assistant: {message_text}")
            continue

        if message_text != "NONE":
            steps.append(
                {
                    "context": _build_gemini_context(
                        base_metadata=base_metadata,
                        task_id=task_id,
                        task_prompt=task_prompt,
                        step_index=len(steps) + 1,
                        history_lines=history_lines,
                        max_chars=max_context_chars,
                    ),
                    "action_text": message_text,
                    "tool_name": "respond",
                    "tool_args": {},
                    "result_text": "NONE",
                    "status": "ok",
                    "source_raw_text": _source_raw_text(
                        base_metadata=base_metadata,
                        task_id=task_id,
                        step_index=len(steps) + 1,
                        raw_fragment={
                            "raw_step_index": raw_step_index,
                            "raw_step": raw_step,
                            "history_steps": _source_raw_history_tail(raw_steps[:raw_step_index]),
                        },
                    ),
                }
            )
            history_lines.append(f"assistant: {message_text}")

    return steps


def _extract_gemini_task_prompt(messages: list[dict[str, Any]], *, max_chars: int) -> str:
    for message_index, message in enumerate(messages):
        if message.get("type") == "user":
            content = str(message.get("content") or "").strip()
            if content:
                return _compact_text(content, max_chars=max_chars)
    return "NONE"


def _build_gemini_context(
    *,
    base_metadata: dict[str, Any],
    task_id: str,
    task_prompt: str,
    step_index: int,
    history_lines: list[str],
    max_chars: int,
) -> str:
    lines = [
        "dataset=SkillsBench",
        f"task_id={task_id}",
        f"trial_name={base_metadata.get('trial_name')}",
        f"condition_dir={base_metadata.get('condition_dir')}",
        f"agent_name={base_metadata.get('agent_name')}",
        f"agent_model_name={base_metadata.get('agent_model_name')}",
        f"trace_format={base_metadata.get('trace_format')}",
        f"step_index={step_index}",
        f"task_prompt={task_prompt}",
        "dialogue_history=",
    ]
    lines.extend(history_lines[-6:] if history_lines else ["NONE"])
    context = "\n".join(lines)
    return context if len(context) <= max_chars else context[: max_chars - 16] + "...<truncated>"


def _extract_gemini_tool_result(tool_call: dict[str, Any], *, max_chars: int) -> str:
    result = tool_call.get("result")
    if isinstance(result, list):
        pieces: list[str] = []
        for item in result:
            response = _as_dict(_as_dict(item).get("functionResponse")).get("response")
            response_dict = _as_dict(response)
            if response_dict.get("error"):
                pieces.append(f"error={_sanitize_multiline(response_dict.get('error'), max_chars=max_chars)}")
            if response_dict.get("output"):
                pieces.append(f"output={_sanitize_multiline(response_dict.get('output'), max_chars=max_chars)}")
            elif response:
                pieces.append(_sanitize_multiline(response, max_chars=max_chars))
        if pieces:
            return "\n".join(pieces)
    result_display = tool_call.get("resultDisplay")
    if result_display:
        return _sanitize_multiline(result_display, max_chars=max_chars)
    return _sanitize_multiline(result, max_chars=max_chars)


def _parse_gemini_cli_trace(
    trace_path: Path,
    *,
    base_metadata: dict[str, Any],
    task_id: str,
    max_action_chars: int,
    max_result_chars: int,
    max_context_chars: int,
) -> list[dict[str, Any]]:
    payload = json.loads(trace_path.read_text(encoding="utf-8", errors="replace"))
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Expected messages[] in {trace_path}")

    task_prompt = _extract_gemini_task_prompt(messages, max_chars=1200)
    history_lines: list[str] = []
    steps: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        message_type = str(message.get("type") or "").lower()
        if message_type == "user":
            content = _compact_text(message.get("content"), max_chars=600)
            history_lines.append(f"user: {content}")
            continue
        if message_type != "gemini":
            continue

        assistant_text = _compact_text(message.get("content"), max_chars=max_action_chars)
        tool_calls = message.get("toolCalls")
        if isinstance(tool_calls, list) and tool_calls:
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_name = _normalize_tool_name(
                    tool_call.get("name") or tool_call.get("displayName") or "tool_call"
                )
                action_text = (
                    f"{assistant_text}\nTOOL_CALL {tool_name}"
                    if assistant_text != "NONE"
                    else f"tool_call::{tool_name}"
                )
                status = "ok" if str(tool_call.get("status") or "success").lower() == "success" else "tool_error"
                steps.append(
                    {
                        "context": _build_gemini_context(
                            base_metadata=base_metadata,
                            task_id=task_id,
                            task_prompt=task_prompt,
                            step_index=len(steps) + 1,
                            history_lines=history_lines,
                            max_chars=max_context_chars,
                        ),
                        "action_text": _compact_text(action_text, max_chars=max_action_chars),
                        "tool_name": tool_name,
                        "tool_args": _as_dict(tool_call.get("args")),
                        "result_text": _extract_gemini_tool_result(tool_call, max_chars=max_result_chars),
                        "status": status,
                        "source_raw_text": _source_raw_text(
                            base_metadata=base_metadata,
                            task_id=task_id,
                            step_index=len(steps) + 1,
                            raw_fragment={
                                "message_index": message_index,
                                "message": message,
                                "history_messages": _source_raw_history_tail(messages[:message_index]),
                                "tool_call_index": message.get("toolCalls", []).index(tool_call),
                                "tool_call": tool_call,
                            },
                        ),
                    }
                )
            history_lines.append(f"assistant: {assistant_text}")
            continue

        if assistant_text != "NONE":
            steps.append(
                {
                    "context": _build_gemini_context(
                        base_metadata=base_metadata,
                        task_id=task_id,
                        task_prompt=task_prompt,
                        step_index=len(steps) + 1,
                        history_lines=history_lines,
                        max_chars=max_context_chars,
                    ),
                    "action_text": assistant_text,
                    "tool_name": "respond",
                    "tool_args": {},
                    "result_text": "NONE",
                    "status": "ok",
                    "source_raw_text": _source_raw_text(
                        base_metadata=base_metadata,
                        task_id=task_id,
                        step_index=len(steps) + 1,
                        raw_fragment={
                            "message_index": message_index,
                            "message": message,
                            "history_messages": _source_raw_history_tail(messages[:message_index]),
                        },
                    ),
                }
            )
            history_lines.append(f"assistant: {assistant_text}")

    return steps


_GEMINI_TXT_SKIP_PREFIXES = (
    "YOLO mode is enabled.",
    "Hook registry initialized",
    "Both GOOGLE_API_KEY and GEMINI_API_KEY are set.",
)


def _parse_gemini_cli_text_trace(
    trace_path: Path,
    *,
    base_metadata: dict[str, Any],
    config_payload: dict[str, Any],
    task_id: str,
    max_action_chars: int,
    max_context_chars: int,
) -> list[dict[str, Any]]:
    text = trace_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines()]
    filtered_lines = [
        line
        for line in lines
        if line and not any(line.startswith(prefix) for prefix in _GEMINI_TXT_SKIP_PREFIXES)
    ]
    if not filtered_lines:
        return []

    task_prompt = _extract_trajectory_task_prompt(config_payload, max_chars=1200)
    history_lines: list[str] = []
    steps: list[dict[str, Any]] = []
    for line_index, line in enumerate(filtered_lines):
        action_text = _compact_text(line, max_chars=max_action_chars)
        steps.append(
            {
                "context": _build_gemini_context(
                    base_metadata=base_metadata,
                    task_id=task_id,
                    task_prompt=task_prompt,
                    step_index=len(steps) + 1,
                    history_lines=history_lines,
                    max_chars=max_context_chars,
                ),
                "action_text": action_text,
                "tool_name": "respond",
                "tool_args": {},
                "result_text": "NONE",
                "status": "ok",
                "source_raw_text": _source_raw_text(
                    base_metadata=base_metadata,
                    task_id=task_id,
                    step_index=len(steps) + 1,
                    raw_fragment={
                        "line_index": line_index,
                        "line": line,
                    },
                ),
            }
        )
        history_lines.append(f"assistant: {action_text}")

    return steps


def _render_acp_content_blocks(content: Any, *, max_chars: int) -> list[str]:
    if not isinstance(content, list):
        rendered = _sanitize_multiline(content, max_chars=max_chars)
        return [] if rendered == "NONE" else [rendered]

    rendered_blocks: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            text = _sanitize_multiline(block, max_chars=max_chars)
        elif block.get("type") == "content":
            inner = block.get("content")
            if isinstance(inner, dict):
                inner_type = str(inner.get("type") or "content")
                if inner_type == "text":
                    text = _sanitize_multiline(inner.get("text"), max_chars=max_chars)
                else:
                    text = f"{inner_type}: {_sanitize_multiline(inner, max_chars=max_chars)}"
            else:
                text = _sanitize_multiline(inner, max_chars=max_chars)
        elif block.get("type") == "diff":
            path = str(block.get("path") or "UNKNOWN")
            new_text = block.get("newText")
            old_text = block.get("oldText")
            if new_text is not None:
                text = f"diff[{path}] new={_sanitize_multiline(new_text, max_chars=max_chars)}"
            elif old_text is not None:
                text = f"diff[{path}] old={_sanitize_multiline(old_text, max_chars=max_chars)}"
            else:
                text = f"diff[{path}]"
        else:
            block_type = str(block.get("type") or "content")
            text = f"{block_type}: {_sanitize_multiline(block, max_chars=max_chars)}"
        if text != "NONE":
            rendered_blocks.append(text)
    return rendered_blocks


def _render_acp_content(content: Any, *, max_chars: int) -> str:
    blocks = _render_acp_content_blocks(content, max_chars=max_chars)
    if not blocks:
        return "NONE"
    return _sanitize_multiline("\n".join(blocks), max_chars=max_chars)


def _acp_tool_name(event: dict[str, Any]) -> str:
    kind = str(event.get("kind") or "").strip().lower()
    title = str(event.get("title") or "").strip().lower()
    if kind == "execute" or title == "terminal":
        return _normalize_tool_name("bash")
    if kind == "read" or title.startswith("read"):
        return _normalize_tool_name("read_file")
    if kind == "edit" and title == "write":
        return _normalize_tool_name("write_file")
    if title == "toolsearch":
        return "tool_search"
    if title == "skill":
        return "skill"
    return _normalize_tool_name(kind or title or "tool_call")


def _render_acp_history_line(event: dict[str, Any], *, max_chars: int) -> str:
    event_type = str(event.get("type") or "unknown")
    if event_type == "tool_call":
        label = _acp_tool_name(event)
        title = _first_non_empty(event.get("title"), event.get("kind"), label)
        content = _render_acp_content(event.get("content"), max_chars=max_chars)
        return f"tool[{label}] {title}: {content}"
    if event_type == "agent_message":
        return f"assistant: {_compact_text(event.get('text'), max_chars=max_chars)}"
    return f"{event_type}: {_compact_text(event, max_chars=max_chars)}"


def _build_acp_context(
    *,
    base_metadata: dict[str, Any],
    task_id: str,
    history: list[dict[str, Any]],
    history_window: int,
    step_index: int,
    max_chars: int,
) -> str:
    lines = [
        "dataset=SkillsBench",
        f"task_id={task_id}",
        f"trial_name={base_metadata.get('trial_name')}",
        f"condition_dir={base_metadata.get('condition_dir')}",
        f"agent_name={base_metadata.get('agent_name')}",
        f"agent_model_name={base_metadata.get('agent_model_name')}",
        f"trace_format={base_metadata.get('trace_format')}",
        f"step_index={step_index}",
        "acp_history=",
    ]
    recent_history = history[-history_window:] if history_window > 0 else history
    if recent_history:
        lines.extend(_render_acp_history_line(event, max_chars=240) for event in recent_history)
    else:
        lines.append("NONE")
    context = "\n".join(lines)
    return context if len(context) <= max_chars else context[: max_chars - 16] + "...<truncated>"


def _parse_acp_trace(
    trace_path: Path,
    *,
    base_metadata: dict[str, Any],
    task_id: str,
    history_window: int,
    max_action_chars: int,
    max_result_chars: int,
    max_context_chars: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            event_type = str(event.get("type") or "")
            context = _build_acp_context(
                base_metadata=base_metadata,
                task_id=task_id,
                history=history,
                history_window=history_window,
                step_index=len(steps) + 1,
                max_chars=max_context_chars,
            )

            if event_type == "tool_call":
                content_blocks = _render_acp_content_blocks(event.get("content"), max_chars=max_result_chars)
                first_block = content_blocks[0] if content_blocks else "NONE"
                title = _first_non_empty(event.get("title"), event.get("kind"), "tool_call")
                action_text = title if first_block == "NONE" else f"{title}: {first_block}"
                status_value = str(event.get("status") or "").strip().lower()
                status = (
                    "tool_error"
                    if status_value and status_value not in {"completed", "ok", "success", "succeeded"}
                    else "ok"
                )
                steps.append(
                    {
                        "context": context,
                        "action_text": _compact_text(action_text, max_chars=max_action_chars),
                        "tool_name": _acp_tool_name(event),
                        "tool_args": {
                            "tool_call_id": event.get("tool_call_id"),
                            "kind": event.get("kind"),
                            "title": event.get("title"),
                            "status": event.get("status"),
                        },
                        "result_text": _render_acp_content(event.get("content"), max_chars=max_result_chars),
                        "status": status,
                        "source_raw_text": _source_raw_text(
                            base_metadata=base_metadata,
                            task_id=task_id,
                            step_index=len(steps) + 1,
                            raw_fragment={
                                "history": _source_raw_history_tail(history),
                                "acp_event": event,
                            },
                        ),
                    }
                )
                history.append(event)
                continue

            if event_type == "agent_message":
                action_text = _compact_text(event.get("text"), max_chars=max_action_chars)
                if action_text != "NONE":
                    steps.append(
                        {
                            "context": context,
                            "action_text": action_text,
                            "tool_name": "respond",
                            "tool_args": {},
                            "result_text": "NONE",
                            "status": "ok",
                            "source_raw_text": _source_raw_text(
                                base_metadata=base_metadata,
                                task_id=task_id,
                                step_index=len(steps) + 1,
                                raw_fragment={
                                    "history": _source_raw_history_tail(history),
                                    "acp_event": event,
                                },
                            ),
                        }
                    )
                history.append(event)
                continue

            history.append(event)

    return steps


def _detect_trace(trial_dir: Path) -> tuple[str, Path] | None:
    project_jsonls = sorted(trial_dir.glob("agent/sessions/projects/*/*.jsonl"))
    if project_jsonls:
        return ("claude_session_jsonl", project_jsonls[0])
    acp_trajectory = trial_dir / "trajectory" / "acp_trajectory.jsonl"
    if acp_trajectory.exists():
        return ("acp_trajectory_jsonl", acp_trajectory)
    claude_txt = trial_dir / "agent" / "claude-code.txt"
    if claude_txt.exists():
        return ("claude_code_txt", claude_txt)
    codex_txt = trial_dir / "agent" / "codex.txt"
    if codex_txt.exists():
        return ("codex_txt", codex_txt)
    gemini_cli_trajectory = trial_dir / "agent" / "gemini-cli.trajectory.json"
    if gemini_cli_trajectory.exists():
        return ("gemini_cli_trajectory_json", gemini_cli_trajectory)
    trajectory_json = trial_dir / "agent" / "trajectory.json"
    if trajectory_json.exists():
        return ("trajectory_json", trajectory_json)
    gemini_cli_txt = trial_dir / "agent" / "gemini-cli.txt"
    if gemini_cli_txt.exists():
        return ("gemini_cli_txt", gemini_cli_txt)
    return None


def _parse_trial(
    trial_dir: Path,
    *,
    split: str,
    history_window: int,
    max_action_chars: int,
    max_result_chars: int,
    max_context_chars: int,
) -> dict[str, Any]:
    result_path = trial_dir / "result.json"
    config_path = trial_dir / "config.json"
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    trace_spec = _detect_trace(trial_dir)
    if trace_spec is None:
        raise ValueError(f"No supported structured trace found under {trial_dir}")

    trace_format, trace_path = trace_spec
    rewards = _result_rewards(result_payload)
    raw_reward = float(rewards.get("reward", 0.0) or 0.0)
    final_success = raw_reward >= 1.0 - 1e-9
    task_id = _task_id_from_payload(result_payload, config_payload)
    base_metadata = _build_common_metadata(
        trial_dir=trial_dir,
        result_payload=result_payload,
        config_payload=config_payload,
        trace_path=trace_path,
        trace_format=trace_format,
        raw_reward=raw_reward,
    )

    if trace_format in {"claude_session_jsonl", "claude_code_txt"}:
        steps = _parse_claude_trace(
            trace_path,
            base_metadata=base_metadata,
            task_id=task_id,
            history_window=history_window,
            max_action_chars=max_action_chars,
            max_result_chars=max_result_chars,
            max_context_chars=max_context_chars,
        )
    elif trace_format == "acp_trajectory_jsonl":
        steps = _parse_acp_trace(
            trace_path,
            base_metadata=base_metadata,
            task_id=task_id,
            history_window=history_window,
            max_action_chars=max_action_chars,
            max_result_chars=max_result_chars,
            max_context_chars=max_context_chars,
        )
    elif trace_format == "codex_txt":
        steps = _parse_codex_trace(
            trace_path,
            base_metadata=base_metadata,
            config_payload=config_payload,
            task_id=task_id,
            max_action_chars=max_action_chars,
            max_result_chars=max_result_chars,
            max_context_chars=max_context_chars,
        )
    elif trace_format == "trajectory_json":
        steps = _parse_terminal_batch_trace(
            trace_path,
            base_metadata=base_metadata,
            config_payload=config_payload,
            task_id=task_id,
            max_action_chars=max_action_chars,
            max_result_chars=max_result_chars,
            max_context_chars=max_context_chars,
        )
    elif trace_format == "gemini_cli_trajectory_json":
        steps = _parse_gemini_cli_trace(
            trace_path,
            base_metadata=base_metadata,
            task_id=task_id,
            max_action_chars=max_action_chars,
            max_result_chars=max_result_chars,
            max_context_chars=max_context_chars,
        )
    elif trace_format == "gemini_cli_txt":
        steps = _parse_gemini_cli_text_trace(
            trace_path,
            base_metadata=base_metadata,
            config_payload=config_payload,
            task_id=task_id,
            max_action_chars=max_action_chars,
            max_context_chars=max_context_chars,
        )
    else:
        raise ValueError(f"Unsupported trace format: {trace_format}")

    if not steps:
        raise ValueError(f"No usable steps parsed from {trace_path}")

    steps[-1]["status"] = "success" if final_success else "failure"

    record = {
        "trajectory_id": _trajectory_id_for_trial(
            trial_dir=trial_dir,
            result_payload=result_payload,
            config_payload=config_payload,
        ),
        "task_id": task_id,
        "final_success": final_success,
        "failure_bucket": _failure_bucket(
            final_success=final_success,
            result_payload=result_payload,
            trial_dir=trial_dir,
        ),
        "split": split,
        "metadata": base_metadata,
        "steps": steps,
    }
    return record


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _iter_trial_dirs(input_root: Path) -> list[Path]:
    trial_dirs: list[Path] = []
    for result_path in input_root.rglob("result.json"):
        trial_dir = result_path.parent
        if not (trial_dir / "config.json").exists():
            continue
        trial_dirs.append(trial_dir)
    return sorted(set(trial_dirs))


def _skip_reason(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    for marker in (" under ", " from "):
        if marker in message:
            message = message.split(marker, 1)[0].strip()
    return message


def main() -> None:
    args = parse_args()
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    trial_dirs = _iter_trial_dirs(args.input_root)
    imported: list[dict[str, Any]] = []
    skipped_counter: Counter[str] = Counter()
    format_counter: Counter[str] = Counter()
    condition_counter: Counter[str] = Counter()
    skipped_examples: list[dict[str, str]] = []

    for trial_dir in trial_dirs:
        try:
            record = _parse_trial(
                trial_dir,
                split=args.split,
                history_window=args.history_window,
                max_action_chars=args.max_action_chars,
                max_result_chars=args.max_result_chars,
                max_context_chars=args.max_context_chars,
            )
        except Exception as exc:
            if args.strict:
                raise
            reason = _skip_reason(exc)
            skipped_counter[reason] += 1
            if len(skipped_examples) < 20:
                skipped_examples.append({"trial_dir": str(trial_dir), "reason": str(exc)})
            continue

        imported.append(record)
        trace_format = str(record["metadata"].get("trace_format", "UNKNOWN"))
        format_counter[trace_format] += 1
        condition_counter[str(record["metadata"].get("condition_dir", "UNKNOWN"))] += 1
        if args.limit is not None and len(imported) >= args.limit:
            break

    _write_jsonl(args.output_jsonl, imported)
    summary = {
        "dataset": "SkillsBench",
        "input_root": str(args.input_root),
        "output_jsonl": str(args.output_jsonl),
        "output_summary": str(args.output_summary),
        "num_trials_seen": len(trial_dirs),
        "num_trajectories": len(imported),
        "num_steps": sum(len(record["steps"]) for record in imported),
        "num_steps_with_source_raw_text": sum(
            1
            for record in imported
            for step in record["steps"]
            if str(step.get("source_raw_text") or "").strip()
        ),
        "avg_steps_per_trajectory": (
            sum(len(record["steps"]) for record in imported) / len(imported) if imported else 0.0
        ),
        "num_successes": sum(1 for record in imported if record["final_success"]),
        "num_failures": sum(1 for record in imported if not record["final_success"]),
        "num_unique_tasks": len({record["task_id"] for record in imported}),
        "trace_formats": dict(sorted(format_counter.items())),
        "condition_dirs": dict(sorted(condition_counter.items())),
        "source_raw_truncation": _source_raw_protocol_summary(),
        "skipped_reasons": dict(sorted(skipped_counter.items())),
        "skipped_examples": skipped_examples,
    }
    _write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
