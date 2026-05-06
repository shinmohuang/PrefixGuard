from __future__ import annotations

import json
import re
from collections import Counter

from monitor_symbolization.data.schema import (
    ObservationReductionStats,
    StepRecord,
    StepView,
    TrajectoryRecord,
)


RepresentationMode = str
_WHITESPACE = re.compile(r"\s+")
_TAU2_SEMANTIC_PROFILE_TAGS: dict[str, tuple[str, ...]] = {
    "semantic-tool-role-v1": ("tool_role",),
    "semantic-tool-role-obligation-v1": ("tool_role", "verification_obligation"),
    "semantic-tool-role-obligation-state-v1": (
        "tool_role",
        "verification_obligation",
        "obligation_state",
    ),
    "semantic-full-v1": (
        "tool_role",
        "verification_obligation",
        "argument_risk",
        "result_state",
    ),
}
_SKILLSBENCH_PROCESS_PROFILE_TAGS: dict[str, tuple[str, ...]] = {
    "phase-v1": ("phase",),
    "process-full-v1": (
        "phase",
        "error_persistence",
        "retry_pattern",
        "progress_state",
    ),
}


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def serialize_step(step: StepRecord) -> str:
    context = normalize_text(step.context)
    action_text = normalize_text(step.action_text)
    result_text = normalize_text(step.result_text)
    tool_name = step.tool_name or "NONE"
    status = step.status or "NONE"
    tool_args = json.dumps(step.tool_args, sort_keys=True, ensure_ascii=True)
    return (
        f"CONTEXT=[{context}] "
        f"ACTION=[action={action_text}; tool={tool_name}; args={tool_args}] "
        f"RESULT=[status={status}; text={result_text}]"
    )


def serialize_source_raw_step(step: StepRecord) -> str:
    if step.source_raw_text is None or not step.source_raw_text.strip():
        raise ValueError("source-raw representation requires StepRecord.source_raw_text")
    return normalize_text(step.source_raw_text)


def _count_tokens(text: str) -> int:
    normalized = normalize_text(text)
    if not normalized:
        return 0
    return len(normalized.split(" "))


def _infer_dataset_name_from_context(context: str) -> str:
    for raw_line in context.splitlines():
        line = normalize_text(raw_line)
        if line.startswith("dataset="):
            value = normalize_text(line[len("dataset=") :]).lower()
            if value:
                return value
    return "webarena"


def resolve_step_view_dataset_name(dataset_name: str | None) -> str | None:
    if dataset_name is None:
        return None
    normalized = normalize_text(dataset_name).lower()
    if normalized in {"", "inferred"}:
        return None
    return normalized


def resolve_tau2_refinement_profile(profile: str | None) -> str | None:
    if profile is None:
        return None
    normalized = normalize_text(profile).lower()
    if normalized in {"", "none", "baseline", "compact-v1"}:
        return None
    return normalized


def resolve_skillsbench_process_profile(profile: str | None) -> str | None:
    if profile is None:
        return None
    normalized = normalize_text(profile).lower()
    if normalized in {"", "none", "baseline"}:
        return None
    return normalized


def _enabled_tau2_semantic_tags(profile: str | None) -> tuple[str, ...]:
    resolved_profile = resolve_tau2_refinement_profile(profile)
    if resolved_profile is None:
        return tuple()
    return _TAU2_SEMANTIC_PROFILE_TAGS.get(resolved_profile, tuple())


def _enabled_skillsbench_process_tags(profile: str | None) -> tuple[str, ...]:
    resolved_profile = resolve_skillsbench_process_profile(profile)
    if resolved_profile is None:
        return _SKILLSBENCH_PROCESS_PROFILE_TAGS["process-full-v1"]
    return _SKILLSBENCH_PROCESS_PROFILE_TAGS.get(resolved_profile, tuple())


def _split_context_lines(context: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    metadata_lines: list[str] = []
    observation_lines: list[str] = []
    in_observation = False
    for raw_line in context.splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        if line.startswith("observation="):
            in_observation = True
            suffix = normalize_text(line[len("observation=") :])
            if suffix:
                observation_lines.append(suffix)
            continue
        if in_observation:
            observation_lines.append(line)
        else:
            metadata_lines.append(line)
    if not observation_lines and metadata_lines:
        metadata_lines, observation_lines = metadata_lines[:2], metadata_lines[2:]
    return tuple(metadata_lines), tuple(observation_lines)


_TERMINALBENCH_METADATA_PREFIXES = (
    "dataset=",
    "task_name=",
    "trial_name=",
    "trial_id=",
    "agent=",
    "model=",
    "step_num=",
)


def _split_terminalbench_context_lines(context: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split TerminalBench context into metadata and observation lines.

    TerminalBench context structure:
        dataset=TerminalBench
        task_name=<name>
        trial_name=<name>
        trial_id=<id>
        agent=<agent>
        model=<model>
        step_num=<n>
        observation=
        <history lines>

    Everything before (and including) the ``observation=`` marker goes to metadata;
    everything after goes to observation. Legacy ``history=`` artifacts remain
    supported so older canonical files continue to parse.
    """
    metadata_lines: list[str] = []
    observation_lines: list[str] = []
    in_observation = False
    for raw_line in context.splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        if line.startswith("observation="):
            in_observation = True
            suffix = normalize_text(line[len("observation="):])
            if suffix:
                observation_lines.append(suffix)
            continue
        if line.startswith("history="):
            in_observation = True
            suffix = normalize_text(line[len("history="):])
            if suffix:
                observation_lines.append(suffix)
            continue
        if in_observation:
            observation_lines.append(line)
        else:
            metadata_lines.append(line)
    return tuple(metadata_lines), tuple(observation_lines)


_SKILLSBENCH_METADATA_PREFIXES = (
    "dataset=",
    "task_id=",
    "trial_name=",
    "condition_dir=",
    "agent_name=",
    "agent_model_name=",
    "trace_format=",
    "step_index=",
    "initial_prompt=",
    "task_prompt=",
    "recent_reasoning=",
    "episode_index=",
    "episode_prompt=",
)

_SKILLSBENCH_DIALOGUE_PREFIXES = ("assistant:", "user:", "tool[", "tool:", "system:")


def _split_skillsbench_context_lines(context: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    metadata_lines: list[str] = []
    observation_lines: list[str] = []
    in_observation = False
    metadata_continuation_index: int | None = None

    for raw_line in context.splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        if line in {"observation=", "dialogue_history="}:
            in_observation = True
            metadata_continuation_index = None
            continue
        if line.startswith(_SKILLSBENCH_METADATA_PREFIXES):
            metadata_lines.append(line)
            metadata_continuation_index = len(metadata_lines) - 1
            continue
        if in_observation:
            observation_lines.append(line)
            continue
        if line.lower().startswith(("user:", "assistant:", "tool[", "tool:", "system:")):
            observation_lines.append(line)
            in_observation = True
            metadata_continuation_index = None
            continue
        if metadata_continuation_index is not None:
            metadata_lines[metadata_continuation_index] = (
                f"{metadata_lines[metadata_continuation_index]} {line}"
            )
            continue
        observation_lines.append(line)

    return tuple(metadata_lines), tuple(observation_lines)


def _metadata_value(metadata_lines: tuple[str, ...], prefix: str) -> str | None:
    for line in metadata_lines:
        if line.startswith(prefix):
            value = normalize_text(line[len(prefix) :])
            if value:
                return value
    return None


def _normalize_metadata_label(value: str | None, default: str = "unknown") -> str:
    normalized = normalize_text(value or "").lower()
    if not normalized:
        return default
    collapsed = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return collapsed or default


def _derive_skillsbench_trace_family(metadata_lines: tuple[str, ...]) -> str:
    trace_format = _metadata_value(metadata_lines, "trace_format=")
    normalized = normalize_text(trace_format or "").lower()
    if normalized in {"claude_session_jsonl", "claude_code_txt"}:
        return "claude_dialogue"
    if normalized == "codex_txt":
        return "codex_exec"
    if normalized == "gemini_cli_trajectory_json":
        return "gemini_exec"
    if normalized == "gemini_cli_txt":
        return "gemini_text"
    if normalized == "trajectory_json":
        return "trajectory_exec"
    return "generic"


def _exit_code_bucket(exit_code: int | None) -> str:
    """Map a raw exit code to a fine-grained semantic bucket (8 categories)."""
    if exit_code is None:
        return "none"
    if exit_code == 0:
        return "zero"
    if exit_code in (1, 2):
        return "error_generic"
    if exit_code == 126:
        return "error_notexec"
    if exit_code == 127:
        return "error_notfound"
    if exit_code >= 128:
        return "error_signal"
    return "error_other"


def _extract_exit_code(*, tool_args_text: str, result_text: str) -> int | None:
    if tool_args_text not in ("", "{}"):
        tool_args = json.loads(tool_args_text)
        raw_exit_code = tool_args.get("exit_code")
        if isinstance(raw_exit_code, int):
            return raw_exit_code
    match = re.search(r"exit code:\s*(-?\d+)", result_text, flags=re.IGNORECASE)
    if match is not None:
        return int(match.group(1))
    return None


def _derive_result_channel(result_text: str) -> str:
    normalized = normalize_text(result_text)
    lowered = normalized.lower()
    if normalized in {"", "NONE"}:
        return "none"
    if lowered.startswith("result="):
        return "result_block"
    if lowered.startswith("error="):
        return "error_block"
    if lowered.startswith("output="):
        return "output_block"
    return "plain_text"


def _derive_skillsbench_step_kind(
    *,
    trace_family: str,
    action_text: str,
    tool_name: str,
) -> str:
    normalized_tool = normalize_text(tool_name).lower()
    normalized_action = normalize_text(action_text).lower()
    if trace_family in {"codex_exec", "gemini_exec", "trajectory_exec"} and normalized_tool not in {"", "none", "respond"}:
        return "command_exec" if normalized_tool == "bash" else "tool_exec"
    if normalized_tool not in {"", "none", "respond"} or "tool_call" in normalized_action:
        return "tool_call"
    return "dialogue_response"


def _derive_skillsbench_outcome(
    *,
    step_kind: str,
    status: str,
    result_channel: str,
    exit_code: int | None,
) -> str:
    normalized_status = normalize_text(status).lower()
    error_status = normalized_status in {"tool_error", "error", "failed", "failure"}

    if step_kind == "dialogue_response":
        return "dialogue_error" if error_status else "dialogue_only"
    if step_kind == "tool_call":
        if error_status or result_channel == "error_block":
            return "tool_error"
        if result_channel in {"result_block", "output_block", "plain_text"}:
            return "tool_result"
        return "tool_call_only"
    if step_kind in {"command_exec", "tool_exec"}:
        if exit_code is not None:
            return "command_success" if exit_code == 0 else "command_error"
        if error_status or result_channel == "error_block":
            return "command_error"
        if result_channel in {"output_block", "plain_text"}:
            return "command_success"
        return "command_observed"
    return "generic"


def _skillsbench_text_signature(
    metadata_lines: tuple[str, ...],
    *,
    action_text: str,
    tool_args_text: str,
    result_text: str,
) -> str:
    parts = [normalize_text(action_text), normalize_text(result_text)]
    for prefix in ("recent_reasoning=", "task_prompt=", "initial_prompt=", "episode_prompt="):
        value = _metadata_value(metadata_lines, prefix)
        if value:
            parts.append(value)
    if tool_args_text not in {"", "{}"}:
        tool_args = json.loads(tool_args_text)
        for value in tool_args.values():
            if isinstance(value, (str, int, float)):
                parts.append(str(value))
    return " ".join(normalize_text(part).lower() for part in parts if normalize_text(part))


def _skillsbench_command_text(*, action_text: str, tool_args_text: str) -> str:
    parts = [normalize_text(action_text)]
    if tool_args_text not in {"", "{}"}:
        tool_args = json.loads(tool_args_text)
        command = tool_args.get("command")
        if isinstance(command, str) and normalize_text(command):
            parts.append(command)
    return " ".join(normalize_text(part).lower() for part in parts if normalize_text(part))


def _derive_skillsbench_phase(
    *,
    metadata_lines: tuple[str, ...],
    step_kind: str,
    tool_name: str,
    action_text: str,
    tool_args_text: str,
    result_text: str,
) -> str:
    signature = _skillsbench_text_signature(
        metadata_lines,
        action_text=action_text,
        tool_args_text=tool_args_text,
        result_text=result_text,
    )
    command_text = _skillsbench_command_text(action_text=action_text, tool_args_text=tool_args_text)
    normalized_tool = normalize_text(tool_name).lower()

    verify_markers = (
        "verify",
        "verification",
        "confirm",
        "check the result",
        "check output",
        "mass_report",
        "verify_filled",
        "final answer",
    )
    test_markers = (
        "pytest",
        "unittest",
        "unittest",
        "go test",
        "cargo test",
        "integration test",
        "failing build",
    )
    build_markers = (
        "pip install",
        "npm install",
        "yarn install",
        "make ",
        "cmake",
        "cargo build",
        "compile",
        "build ",
        "package",
        "javac",
    )
    inspect_markers = (
        "read ",
        "inspect",
        "examine",
        "list ",
        "search",
        "lookup",
        "head ",
        "tail ",
        "grep ",
        "find ",
        "which ",
        "pip list",
        "cat ",
        "od ",
        "hexdump",
        "google_web_search",
    )
    edit_markers = (
        "create ",
        "write ",
        "edit ",
        "update ",
        "modify ",
        "patch ",
        "mkdir ",
        "touch ",
        "rm ",
        "mv ",
        "cp ",
        "sed ",
        "fill ",
    )

    if any(marker in signature for marker in verify_markers):
        return "verify"
    if step_kind in {"command_exec", "tool_exec"} and any(marker in command_text for marker in test_markers):
        return "test"
    if any(marker in signature for marker in test_markers):
        return "test"
    if step_kind in {"command_exec", "tool_exec"} and any(marker in command_text for marker in build_markers):
        return "build"
    if any(marker in signature for marker in build_markers):
        return "build"
    if normalized_tool == "write_file" or any(marker in signature for marker in edit_markers):
        return "edit"
    if normalized_tool in {"read_file", "google_web_search"} or any(
        marker in signature for marker in inspect_markers
    ):
        return "inspect"
    return "unknown"


def _derive_skillsbench_error_persistence(
    *,
    metadata_lines: tuple[str, ...],
    action_text: str,
    tool_args_text: str,
    result_text: str,
    status: str,
    exit_code: int | None,
    result_channel: str,
) -> str:
    normalized_status = normalize_text(status).lower()
    signature = _skillsbench_text_signature(
        metadata_lines,
        action_text=action_text,
        tool_args_text=tool_args_text,
        result_text=result_text,
    )
    has_error = (
        exit_code not in {None, 0}
        or normalized_status in {"tool_error", "error", "failed", "failure"}
        or result_channel == "error_block"
        or any(
            marker in signature
            for marker in ("traceback", "error", "failed", "exception", "not found", "syntax error")
        )
    )
    if not has_error:
        return "no_error"
    retry_markers = ("retry", "rerun", "run again", "again", "still", "another attempt")
    if any(marker in signature for marker in retry_markers):
        return "repeated_error"
    return "new_error"


def _derive_skillsbench_retry_pattern(
    *,
    metadata_lines: tuple[str, ...],
    action_text: str,
    tool_args_text: str,
) -> str:
    signature = _skillsbench_text_signature(
        metadata_lines,
        action_text=action_text,
        tool_args_text=tool_args_text,
        result_text="",
    )
    if not any(marker in signature for marker in ("retry", "rerun", "run again", "again", "still")):
        return "none"
    same_command_markers = ("rerun the script", "rerun the command", "run again", "retry the script")
    if any(marker in signature for marker in same_command_markers):
        return "same_command_retry"
    return "same_goal_retry"


def _derive_skillsbench_progress_state(
    *,
    phase: str,
    outcome: str,
    status: str,
    error_persistence: str,
) -> str:
    normalized_status = normalize_text(status).lower()
    if normalized_status in {"failed", "failure"}:
        return "regression"
    if error_persistence != "no_error":
        return "no_progress"
    if outcome in {"command_error", "tool_error", "dialogue_error"}:
        return "no_progress"
    if phase in {"inspect", "edit", "build", "test", "verify"} and outcome in {
        "tool_result",
        "command_success",
        "command_observed",
        "dialogue_only",
    }:
        return "progress"
    return "no_progress"


def _augment_skillsbench_metadata(
    metadata_lines: tuple[str, ...],
    *,
    action_text: str,
    tool_name: str,
    tool_args_text: str,
    result_text: str,
    status: str,
    skillsbench_process_profile: str | None = None,
) -> tuple[str, ...]:
    trace_family = _derive_skillsbench_trace_family(metadata_lines)
    agent_family = _normalize_metadata_label(_metadata_value(metadata_lines, "agent_name="))
    step_kind = _derive_skillsbench_step_kind(
        trace_family=trace_family,
        action_text=action_text,
        tool_name=tool_name,
    )
    result_channel = _derive_result_channel(result_text)
    exit_code = _extract_exit_code(tool_args_text=tool_args_text, result_text=result_text)
    exit_code_tag = "none" if exit_code is None else ("zero" if exit_code == 0 else "nonzero")
    outcome = _derive_skillsbench_outcome(
        step_kind=step_kind,
        status=status,
        result_channel=result_channel,
        exit_code=exit_code,
    )
    phase = _derive_skillsbench_phase(
        metadata_lines=metadata_lines,
        step_kind=step_kind,
        tool_name=tool_name,
        action_text=action_text,
        tool_args_text=tool_args_text,
        result_text=result_text,
    )
    error_persistence = _derive_skillsbench_error_persistence(
        metadata_lines=metadata_lines,
        action_text=action_text,
        tool_args_text=tool_args_text,
        result_text=result_text,
        status=status,
        exit_code=exit_code,
        result_channel=result_channel,
    )
    retry_pattern = _derive_skillsbench_retry_pattern(
        metadata_lines=metadata_lines,
        action_text=action_text,
        tool_args_text=tool_args_text,
    )
    progress_state = _derive_skillsbench_progress_state(
        phase=phase,
        outcome=outcome,
        status=status,
        error_persistence=error_persistence,
    )
    enabled_process_tags = set(_enabled_skillsbench_process_tags(skillsbench_process_profile))
    task_name_raw = _metadata_value(metadata_lines, "task_name=") or ""
    task_type_tag = re.sub(r"[^a-z0-9]+", "_", task_name_raw.lower()).strip("_")
    enriched = list(metadata_lines)
    enriched.extend(
        (
            f"skillsbench_trace_family={trace_family}",
            f"skillsbench_agent_family={agent_family}",
            f"skillsbench_step_kind={step_kind}",
            f"skillsbench_outcome={outcome}",
            f"skillsbench_result_channel={result_channel}",
            f"skillsbench_exit_code={exit_code_tag}",
        )
    )
    if task_type_tag:
        enriched.append(f"skillsbench_task_type={task_type_tag}")
    enriched.append(f"skillsbench_exit_code_fine={_exit_code_bucket(exit_code)}")
    if "phase" in enabled_process_tags:
        enriched.append(f"skillsbench_phase={phase}")
    if "error_persistence" in enabled_process_tags:
        enriched.append(f"skillsbench_error_persistence={error_persistence}")
    if "retry_pattern" in enabled_process_tags:
        enriched.append(f"skillsbench_retry_pattern={retry_pattern}")
    if "progress_state" in enabled_process_tags:
        enriched.append(f"skillsbench_progress_state={progress_state}")
    return tuple(dict.fromkeys(enriched))


def _tau2_speaker_pattern(observation_lines: tuple[str, ...]) -> str:
    speakers: list[str] = []
    for line in observation_lines:
        normalized = normalize_text(line).lower()
        if normalized.startswith("assistant:"):
            speaker = "assistant"
        elif normalized.startswith("user:"):
            speaker = "user"
        elif normalized.startswith("tool[") or normalized.startswith("tool:"):
            speaker = "tool"
        elif normalized.startswith("system:"):
            speaker = "system"
        else:
            continue
        if not speakers or speakers[-1] != speaker:
            speakers.append(speaker)
    if not speakers:
        return "none"
    return "->".join(speakers[-4:])


def _looks_like_tau2_guidance(action_text: str) -> bool:
    normalized = normalize_text(action_text).lower()
    if not normalized or "tool_call " in normalized:
        return False
    guidance_markers = (
        "please provide",
        "could you provide",
        "can you provide",
        "please go to",
        "go to ",
        "click ",
        "select ",
        "open ",
        "turn on",
        "turn off",
        "enable ",
        "disable ",
        "log in",
        "sign in",
        "update ",
        "change ",
        "you can ",
        "you should ",
        "you'll need to ",
        "please check",
        "please contact",
        "i recommend that you",
    )
    return any(marker in normalized for marker in guidance_markers)


def _looks_like_tau2_query(action_text: str) -> bool:
    normalized = normalize_text(action_text).lower()
    if not normalized or "tool_call " in normalized:
        return False
    query_markers = (
        "?",
        "could you",
        "can you",
        "would you",
        "may i have",
        "please provide",
        "please confirm",
        "what is",
        "what's",
        "which ",
        "when ",
        "where ",
        "who ",
        "do you ",
        "is that correct",
    )
    return any(marker in normalized for marker in query_markers)


def _derive_tau2_process_tags(
    *,
    observation_lines: tuple[str, ...],
    action_text: str,
    tool_name: str,
    result_text: str,
    status: str,
) -> tuple[str, str, str]:
    normalized_tool = normalize_text(tool_name).lower()
    normalized_action = normalize_text(action_text)
    normalized_result = normalize_text(result_text).lower()
    normalized_status = normalize_text(status).lower()

    if normalized_tool not in ("", "none", "respond") or "tool_call " in normalized_action.lower():
        step_kind = "invoke_tool"
        control_locus = "agent"
    elif normalized_result.startswith("tool[") or "tool[" in normalized_result:
        step_kind = "explain_tool_result"
        control_locus = "dialogue-only"
    elif _looks_like_tau2_guidance(normalized_action):
        step_kind = "guide_user_action"
        control_locus = "user-mediated"
    elif _looks_like_tau2_query(normalized_action):
        step_kind = "ask_or_confirm"
        control_locus = "dialogue-only"
    elif normalized_status in {"tool_error", "error", "failed", "failure"}:
        step_kind = "handoff_or_fail"
        control_locus = "dialogue-only"
    else:
        step_kind = "assistant_response"
        control_locus = "dialogue-only"

    speaker_pattern = _tau2_speaker_pattern(observation_lines)
    return step_kind, control_locus, speaker_pattern


def _derive_tau2_compact_tags(
    *,
    step_kind: str,
    action_text: str,
    result_text: str,
    status: str,
) -> tuple[str, str, str, str]:
    normalized_action = normalize_text(action_text).lower()
    normalized_result = normalize_text(result_text).lower()
    normalized_status = normalize_text(status).lower()

    if normalized_status in {"tool_error", "error", "failed", "failure"}:
        return ("handoff", "failed", "error", "system_error")
    if step_kind == "invoke_tool":
        return ("tool_call", "grounded", "invoke", "none")
    if step_kind == "guide_user_action":
        return ("user_guidance", "collecting", "none", "user_action_required")
    if step_kind == "ask_or_confirm":
        return ("query_or_confirm", "collecting", "none", "none")
    if step_kind == "explain_tool_result" or normalized_result.startswith("tool[") or "tool[" in normalized_result:
        return ("inform", "verified", "post_tool", "none")
    if "unable to" in normalized_action or "cannot " in normalized_action or "can't " in normalized_action:
        return ("handoff", "failed", "error", "system_error")
    return ("inform", "none", "none", "none")


def _derive_tau2_query_collecting_subtype(
    *,
    observation_lines: tuple[str, ...],
    action_text: str,
    result_text: str,
) -> str:
    normalized = " ".join(
        normalize_text(part).lower()
        for part in (
            *observation_lines,
            action_text,
            result_text,
        )
        if normalize_text(part)
    )
    identity_markers = (
        "verify",
        "verification",
        "confirm",
        "billing zip",
        "zip code",
        "postal code",
        "last four",
        "phone number",
        "email address",
        "account holder",
        "full name",
        "date of birth",
        "security question",
        "identity",
        "authenticate",
        "authentication",
    )
    lookup_markers = (
        "order number",
        "reservation",
        "booking",
        "tracking number",
        "reference number",
        "account number",
        "member id",
        "confirmation code",
    )
    option_markers = (
        "which option",
        "which item",
        "which one",
        "would you like",
        "do you want",
        "choose",
        "select",
        "preference",
        "seat",
        "plan",
        "package",
        "delivery method",
        "color",
        "size",
    )
    constraint_markers = (
        "when",
        "where",
        "what time",
        "which date",
        "date",
        "time",
        "address",
        "quantity",
        "amount",
        "reason",
        "details",
        "constraint",
        "availability",
    )
    if any(marker in normalized for marker in identity_markers):
        return "identity_verification"
    if any(marker in normalized for marker in lookup_markers):
        return "order_or_account_lookup"
    if any(marker in normalized for marker in option_markers):
        return "option_confirmation"
    if any(marker in normalized for marker in constraint_markers):
        return "constraint_clarification"
    return "generic_followup"


def _derive_tau2_query_collecting_mode(action_text: str) -> str:
    normalized = normalize_text(action_text).lower()
    confirm_markers = (
        "confirm",
        "is that correct",
        "does that look right",
        "just to confirm",
        "can you verify",
    )
    if any(marker in normalized for marker in confirm_markers):
        return "confirm"
    return "query"


def _tau2_text_signature(
    observation_lines: tuple[str, ...],
    action_text: str,
    result_text: str,
) -> str:
    return " ".join(
        normalize_text(part).lower()
        for part in (*observation_lines, action_text, result_text)
        if normalize_text(part)
    )


def _derive_tau2_tool_role(tool_name: str) -> str:
    normalized_tool = normalize_text(tool_name).lower()
    if normalized_tool in {"", "none", "respond"}:
        return "none"
    irreversible_markers = (
        "cancel",
        "refund",
        "exchange",
        "replace",
        "remove",
        "delete",
        "terminate",
        "close_account",
    )
    quote_markers = ("quote", "estimate", "price", "preview")
    check_markers = ("check", "validate", "verify", "confirm")
    read_markers = ("get", "list", "search", "find", "lookup", "fetch", "retrieve")
    write_markers = (
        "update",
        "change",
        "set",
        "modify",
        "book",
        "create",
        "schedule",
        "apply",
        "toggle",
    )
    if any(marker in normalized_tool for marker in irreversible_markers):
        return "irreversible_write"
    if any(marker in normalized_tool for marker in quote_markers):
        return "quote"
    if any(marker in normalized_tool for marker in check_markers):
        return "check"
    if any(marker in normalized_tool for marker in read_markers):
        return "read"
    if any(marker in normalized_tool for marker in write_markers):
        return "write"
    return "write"


def _derive_tau2_verification_obligation(
    *,
    observation_lines: tuple[str, ...],
    action_text: str,
    result_text: str,
    tool_name: str,
) -> str:
    normalized = _tau2_text_signature(observation_lines, action_text, result_text)
    policy_markers = (
        "policy",
        "eligibility",
        "approval",
        "consent",
        "terms",
        "verification is complete",
        "verification complete",
        "blocked",
    )
    identity_markers = (
        "identity",
        "authenticate",
        "authentication",
        "billing zip",
        "zip code",
        "postal code",
        "last four",
        "security question",
        "date of birth",
    )
    account_markers = (
        "account",
        "reservation",
        "booking",
        "order",
        "customer",
        "member",
        "account_id",
        "reservation_id",
        "order_id",
        "customer_id",
    )
    normalized_tool = normalize_text(tool_name).lower()
    if any(marker in normalized for marker in policy_markers):
        return "policy_needed"
    if any(marker in normalized for marker in identity_markers):
        return "identity_needed"
    if any(marker in normalized for marker in account_markers) or normalized_tool in {
        "get_reservation_details",
        "get_order_details",
        "get_customer_details",
    }:
        return "account_needed"
    return "none"


def _derive_tau2_argument_risk(tool_args_text: str) -> str:
    if tool_args_text in {"", "{}"}:
        return "none"
    tool_args = json.loads(tool_args_text)
    normalized_keys = [
        normalize_text(str(key)).lower() for key in tool_args.keys() if normalize_text(str(key))
    ]
    identifier_markers = (
        "id",
        "code",
        "number",
        "account",
        "reservation",
        "order",
        "customer",
        "member",
    )
    time_markers = ("date", "time", "departure", "arrival", "schedule")
    quantity_markers = ("quantity", "amount", "price", "payment", "qty", "count", "units")
    if any(any(marker in key for marker in identifier_markers) for key in normalized_keys):
        return "identifier"
    if any(any(marker in key for marker in time_markers) for key in normalized_keys):
        return "time_or_date"
    if any(any(marker in key for marker in quantity_markers) for key in normalized_keys):
        return "quantity_or_amount"
    return "none"


def _derive_tau2_result_state(
    *,
    tool_role: str,
    observation_lines: tuple[str, ...],
    action_text: str,
    result_text: str,
    status: str,
) -> str:
    normalized = _tau2_text_signature(observation_lines, action_text, result_text)
    normalized_status = normalize_text(status).lower()
    if any(
        marker in normalized
        for marker in (
            "policy",
            "eligibility",
            "approval",
            "blocked",
            "verification is complete",
            "verification complete",
        )
    ):
        return "policy_block"
    if tool_role in {"read", "check", "quote"}:
        if any(
            marker in normalized
            for marker in ("not found", "no matching", "none found", "empty", "unavailable")
        ):
            return "lookup_empty"
        if normalized_status not in {"tool_error", "error", "failed", "failure"} and normalize_text(
            result_text
        ) not in {"", "NONE"}:
            return "lookup_ok"
    if tool_role in {"write", "irreversible_write"}:
        if normalized_status in {"tool_error", "error", "failed", "failure"}:
            return "write_error"
        if any(
            marker in normalized
            for marker in (
                "success",
                "updated",
                "changed",
                "cancelled",
                "canceled",
                "booked",
                "created",
                "refunded",
                "exchanged",
                "scheduled",
                "completed",
            )
        ):
            return "write_success"
    return "none"


def _derive_tau2_obligation_state(
    *,
    dialogue_act: str,
    verification_state: str,
    tool_role: str,
    verification_obligation: str,
    result_state: str,
) -> str:
    if verification_obligation == "none":
        return "none"
    if tool_role in {"write", "irreversible_write"}:
        if result_state in {"policy_block", "write_error"}:
            return "unsafe_write_attempt"
        return "write_before_verification"
    if verification_state == "collecting" or dialogue_act in {"query_or_confirm", "user_guidance"}:
        return "pending_verification"
    if verification_state == "verified":
        return "verification_cleared"
    if dialogue_act == "tool_call":
        return "verification_in_tool_loop"
    return "pending_verification"


def _augment_tau2_metadata(
    metadata_lines: tuple[str, ...],
    *,
    observation_lines: tuple[str, ...],
    action_text: str,
    tool_name: str,
    tool_args_text: str,
    result_text: str,
    status: str,
    tau2_refinement_profile: str | None = None,
) -> tuple[str, ...]:
    step_kind, control_locus, speaker_pattern = _derive_tau2_process_tags(
        observation_lines=observation_lines,
        action_text=action_text,
        tool_name=tool_name,
        result_text=result_text,
        status=status,
    )
    dialogue_act, verification_state, tool_phase, handoff_state = _derive_tau2_compact_tags(
        step_kind=step_kind,
        action_text=action_text,
        result_text=result_text,
        status=status,
    )
    tool_role = _derive_tau2_tool_role(tool_name)
    verification_obligation = _derive_tau2_verification_obligation(
        observation_lines=observation_lines,
        action_text=action_text,
        result_text=result_text,
        tool_name=tool_name,
    )
    argument_risk = _derive_tau2_argument_risk(tool_args_text)
    result_state = _derive_tau2_result_state(
        tool_role=tool_role,
        observation_lines=observation_lines,
        action_text=action_text,
        result_text=result_text,
        status=status,
    )
    obligation_state = _derive_tau2_obligation_state(
        dialogue_act=dialogue_act,
        verification_state=verification_state,
        tool_role=tool_role,
        verification_obligation=verification_obligation,
        result_state=result_state,
    )
    enabled_semantic_tags = set(_enabled_tau2_semantic_tags(tau2_refinement_profile))
    enriched = list(metadata_lines)
    enriched.extend(
        (
            f"step_kind={step_kind}",
            f"control_locus={control_locus}",
            f"speaker_pattern={speaker_pattern}",
            f"dialogue_act={dialogue_act}",
            f"verification_state={verification_state}",
            f"tool_phase={tool_phase}",
            f"handoff_state={handoff_state}",
        )
    )
    if "tool_role" in enabled_semantic_tags:
        enriched.append(f"tool_role={tool_role}")
    if "verification_obligation" in enabled_semantic_tags:
        enriched.append(f"verification_obligation={verification_obligation}")
    if "obligation_state" in enabled_semantic_tags:
        enriched.append(f"obligation_state={obligation_state}")
    if "argument_risk" in enabled_semantic_tags:
        enriched.append(f"argument_risk={argument_risk}")
    if "result_state" in enabled_semantic_tags:
        enriched.append(f"result_state={result_state}")
    resolved_profile = resolve_tau2_refinement_profile(tau2_refinement_profile)
    if (
        resolved_profile is not None
        and dialogue_act == "query_or_confirm"
        and verification_state == "collecting"
    ):
        query_collecting_subtype = _derive_tau2_query_collecting_subtype(
            observation_lines=observation_lines,
            action_text=action_text,
            result_text=result_text,
        )
        enriched.append(f"query_collecting_subtype={query_collecting_subtype}")
        if resolved_profile in {"collecting-subtypes-v2", "collecting-subtypes-v3"}:
            enriched.append(
                f"query_collecting_mode={_derive_tau2_query_collecting_mode(action_text)}"
            )
        if resolved_profile == "collecting-subtypes-v3":
            has_identifier = int(
                query_collecting_subtype in {"identity_verification", "order_or_account_lookup"}
            )
            enriched.append(f"query_collecting_has_identifier={has_identifier}")
    return tuple(dict.fromkeys(enriched))


def _line_score(line: str, anchors: tuple[str, ...]) -> tuple[int, int, int]:
    normalized = normalize_text(line).lower()
    anchor_hits = sum(int(anchor in normalized) for anchor in anchors if anchor)
    status_hits = int(any(token in normalized for token in ("error", "success", "fail", "invalid", "warning")))
    digit_hits = sum(char.isdigit() for char in normalized)
    return (anchor_hits, status_hits, digit_hits)


def _extract_anchor_tokens(
    action_text: str,
    tool_name: str,
    tool_args_text: str,
    result_text: str,
    status: str,
) -> tuple[str, ...]:
    tokens = [
        normalize_text(action_text).lower(),
        normalize_text(tool_name).lower(),
        normalize_text(status).lower(),
        normalize_text(result_text).lower(),
    ]
    if tool_args_text not in ("{}", ""):
        for value in json.loads(tool_args_text).values():
            tokens.append(normalize_text(str(value)).lower())
    return tuple(token for token in tokens if token and token != "none")


def _score_unit(text: str, anchors: tuple[str, ...]) -> tuple[int, int, int]:
    return _line_score(text, anchors)


def _build_reduction_stats(
    *,
    original_lines: tuple[str, ...],
    reduced_lines: tuple[str, ...],
) -> ObservationReductionStats:
    original_text = " ".join(original_lines)
    reduced_text = " ".join(reduced_lines)
    return ObservationReductionStats(
        original_line_count=len(original_lines),
        retained_line_count=len(reduced_lines),
        original_char_count=len(original_text),
        retained_char_count=len(reduced_text),
        original_token_count=_count_tokens(original_text),
        retained_token_count=_count_tokens(reduced_text),
    )


def _reduce_units(
    units: tuple[tuple[str, ...], ...],
    *,
    max_units: int,
    anchors: tuple[str, ...],
) -> tuple[tuple[str, ...], ObservationReductionStats]:
    flattened_units: list[tuple[int, tuple[int, int, int], tuple[str, ...]]] = []
    normalized_lines: list[str] = []
    for index, unit in enumerate(units):
        normalized_unit = tuple(normalize_text(line) for line in unit if normalize_text(line))
        if not normalized_unit:
            continue
        unit_text = " ".join(normalized_unit)
        normalized_lines.extend(normalized_unit)
        flattened_units.append((index, _score_unit(unit_text, anchors), normalized_unit))

    ranked = sorted(flattened_units, key=lambda item: (item[1], -item[0]), reverse=True)
    selected_indices = sorted(index for index, _, _ in ranked[:max_units])

    reduced_lines: list[str] = []
    for index, _, normalized_unit in flattened_units:
        if index in selected_indices:
            reduced_lines.extend(normalized_unit)

    stats = _build_reduction_stats(
        original_lines=tuple(normalized_lines),
        reduced_lines=tuple(reduced_lines),
    )
    return tuple(reduced_lines), stats


def _dialogue_turn_units(lines: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    if not lines:
        return tuple()

    turn_prefixes = ("user:", "assistant:", "tool[", "tool:", "system:")
    units: list[list[str]] = []
    current: list[str] = []

    for raw_line in lines:
        line = normalize_text(raw_line)
        if not line:
            continue
        is_new_turn = line.lower().startswith(turn_prefixes)
        if is_new_turn and current:
            units.append(current)
            current = [line]
        else:
            current.append(line)

    if current:
        units.append(current)
    return tuple(tuple(unit) for unit in units if unit)


def _reduce_skillsbench_dialogue_lines(
    lines: tuple[str, ...],
    *,
    max_lines: int,
) -> tuple[tuple[str, ...], ObservationReductionStats]:
    normalized_lines = tuple(normalize_text(line) for line in lines if normalize_text(line))
    if max_lines <= 0:
        return tuple(), _build_reduction_stats(original_lines=normalized_lines, reduced_lines=tuple())

    dialogue_lines = tuple(
        line for line in normalized_lines if line.lower().startswith(_SKILLSBENCH_DIALOGUE_PREFIXES)
    )
    if not dialogue_lines:
        return tuple(), _build_reduction_stats(original_lines=normalized_lines, reduced_lines=tuple())

    reduced_lines = dialogue_lines[-max_lines:]
    return reduced_lines, _build_reduction_stats(
        original_lines=normalized_lines,
        reduced_lines=reduced_lines,
    )


def _is_tau2_speaker_turn(unit: tuple[str, ...]) -> bool:
    if not unit:
        return False
    prefix = normalize_text(unit[0]).lower()
    return prefix.startswith("assistant:") or prefix.startswith("user:")


def _is_tau2_tool_turn(unit: tuple[str, ...]) -> bool:
    if not unit:
        return False
    prefix = normalize_text(unit[0]).lower()
    return prefix.startswith("tool[") or prefix.startswith("tool:")


def _summarize_tau2_tool_turns(
    *,
    units: tuple[tuple[str, ...], ...],
    tool_name: str,
    result_text: str,
    status: str,
) -> str | None:
    has_tool_evidence = any(_is_tau2_tool_turn(unit) for unit in units)
    normalized_result = normalize_text(result_text).lower()
    normalized_tool = normalize_text(tool_name).lower()
    normalized_status = normalize_text(status).lower()
    if not has_tool_evidence and normalized_tool in {"", "none", "respond"} and "tool[" not in normalized_result:
        return None

    tool_text = " ".join(
        normalize_text(line).lower()
        for unit in units
        if _is_tau2_tool_turn(unit)
        for line in unit
    )
    combined = " ".join(
        part for part in (tool_text, normalized_result, normalized_tool, normalized_status) if part
    )

    if normalized_status in {"tool_error", "error", "failed", "failure"} or any(
        marker in combined for marker in ("error", "failed", "failure", "rejected", "unable", "cannot")
    ):
        return "tool_summary: tool_error"
    if any(marker in combined for marker in ("matches", "options", "available flights", "available orders", "results")):
        return "tool_summary: multiple_results"
    if any(
        marker in combined
        for marker in ("found", "located", "verified", "fee", "reservation", "order", "account")
    ):
        return "tool_summary: lookup_success"
    return "tool_summary: tool_observed"


def reduce_dialogue_turns(
    lines: tuple[str, ...],
    action_text: str,
    tool_name: str,
    tool_args_text: str,
    result_text: str,
    status: str,
    max_lines: int = 8,
) -> tuple[tuple[str, ...], ObservationReductionStats]:
    units = _dialogue_turn_units(lines)
    if not units:
        return reduce_observation_lines(
            lines,
            action_text=action_text,
            tool_name=tool_name,
            tool_args_text=tool_args_text,
            result_text=result_text,
            status=status,
            max_lines=max_lines,
        )

    normalized_units = tuple(
        tuple(normalize_text(line) for line in unit if normalize_text(line))
        for unit in units
    )
    normalized_units = tuple(unit for unit in normalized_units if unit)
    original_lines = tuple(line for unit in normalized_units for line in unit)
    if max_lines <= 0:
        return tuple(), _build_reduction_stats(original_lines=original_lines, reduced_lines=tuple())

    speaker_units = tuple(unit for unit in normalized_units if _is_tau2_speaker_turn(unit))
    if not speaker_units:
        anchors = _extract_anchor_tokens(
            action_text=action_text,
            tool_name=tool_name,
            tool_args_text=tool_args_text,
            result_text=result_text,
            status=status,
        )
        return _reduce_units(normalized_units, max_units=max_lines, anchors=anchors)

    tool_summary = _summarize_tau2_tool_turns(
        units=normalized_units,
        tool_name=tool_name,
        result_text=result_text,
        status=status,
    )
    reserved_summary_lines = 1 if tool_summary is not None and max_lines > 1 else 0
    speaker_budget = max(0, min(4, len(speaker_units), max_lines - reserved_summary_lines))
    if speaker_budget == 0 and tool_summary is not None:
        reduced_lines = (tool_summary,)
        return reduced_lines, _build_reduction_stats(
            original_lines=original_lines,
            reduced_lines=reduced_lines,
        )

    selected_speaker_units = speaker_units[-speaker_budget:] if speaker_budget else tuple()
    reduced_lines_list = [line for unit in selected_speaker_units for line in unit]
    if tool_summary is not None and len(reduced_lines_list) < max_lines:
        reduced_lines_list.append(tool_summary)
    reduced_lines = tuple(reduced_lines_list[:max_lines])
    return reduced_lines, _build_reduction_stats(
        original_lines=original_lines,
        reduced_lines=reduced_lines,
    )


def _log_block_units(lines: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    if not lines:
        return tuple()

    units: list[list[str]] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            units.append(current)
            current = []

    block_starts = (
        "output=",
        "stdout:",
        "stderr:",
        "diff:",
        "patch:",
        "traceback",
        "error:",
        "warning:",
        "file:",
    )

    for raw_line in lines:
        line = normalize_text(raw_line)
        if not line:
            continue
        lower = line.lower()
        if (lower.startswith(block_starts) or line.startswith("```")) and current:
            flush()
        current.append(line)
    flush()
    return tuple(tuple(unit) for unit in units if unit)


def reduce_log_blocks(
    lines: tuple[str, ...],
    action_text: str,
    tool_name: str,
    tool_args_text: str,
    result_text: str,
    status: str,
    max_lines: int = 8,
) -> tuple[tuple[str, ...], ObservationReductionStats]:
    anchors = _extract_anchor_tokens(
        action_text=action_text,
        tool_name=tool_name,
        tool_args_text=tool_args_text,
        result_text=result_text,
        status=status,
    )
    units = _log_block_units(lines)
    if not units:
        return reduce_observation_lines(
            lines,
            action_text=action_text,
            tool_name=tool_name,
            tool_args_text=tool_args_text,
            result_text=result_text,
            status=status,
            max_lines=max_lines,
        )
    return _reduce_units(units, max_units=max_lines, anchors=anchors)


def reduce_observation_lines(
    lines: tuple[str, ...],
    action_text: str,
    tool_name: str,
    tool_args_text: str,
    result_text: str,
    status: str,
    max_lines: int = 8,
) -> tuple[tuple[str, ...], ObservationReductionStats]:
    normalized_lines = tuple(normalize_text(line) for line in lines if normalize_text(line))
    anchors = _extract_anchor_tokens(
        action_text=action_text,
        tool_name=tool_name,
        tool_args_text=tool_args_text,
        result_text=result_text,
        status=status,
    )

    scored = [
        (index, _line_score(line, anchors), line)
        for index, line in enumerate(normalized_lines)
    ]
    ranked = sorted(scored, key=lambda item: (item[1], -item[0]), reverse=True)
    selected_indices = sorted(index for index, _, _ in ranked[:max_lines])
    reduced = tuple(normalized_lines[index] for index in selected_indices)
    stats = _build_reduction_stats(
        original_lines=normalized_lines,
        reduced_lines=reduced,
    )
    return reduced, stats


def build_step_view(
    step: StepRecord,
    max_observation_lines: int = 8,
    dataset_name: str | None = None,
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
) -> StepView:
    resolved_dataset_name = resolve_step_view_dataset_name(dataset_name)
    normalized_dataset = resolved_dataset_name or _infer_dataset_name_from_context(step.context)
    if normalized_dataset == "skillsbench":
        metadata_lines, observation_lines = _split_skillsbench_context_lines(step.context)
    elif normalized_dataset == "terminalbench":
        metadata_lines, observation_lines = _split_terminalbench_context_lines(step.context)
    else:
        metadata_lines, observation_lines = _split_context_lines(step.context)
    action_text = normalize_text(step.action_text)
    tool_name = normalize_text(step.tool_name or "NONE")
    result_text = normalize_text(step.result_text)
    status = normalize_text(step.status or "NONE")
    tool_args_text = json.dumps(step.tool_args, sort_keys=True, ensure_ascii=True)
    if normalized_dataset == "tau2bench":
        metadata_lines = _augment_tau2_metadata(
            metadata_lines,
            observation_lines=observation_lines,
            action_text=action_text,
            tool_name=tool_name,
            tool_args_text=tool_args_text,
            result_text=result_text,
            status=status,
            tau2_refinement_profile=tau2_refinement_profile,
        )
        reduced_lines, stats = reduce_dialogue_turns(
            observation_lines,
            action_text=action_text,
            tool_name=tool_name,
            tool_args_text=tool_args_text,
            result_text=result_text,
            status=status,
            max_lines=max_observation_lines,
        )
    elif normalized_dataset == "skillsbench":
        metadata_lines = _augment_skillsbench_metadata(
            metadata_lines,
            action_text=action_text,
            tool_name=tool_name,
            tool_args_text=tool_args_text,
            result_text=result_text,
            status=status,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        trace_family = _metadata_value(metadata_lines, "skillsbench_trace_family=") or "generic"
        if trace_family in {"claude_dialogue", "gemini_exec", "gemini_text"}:
            reduced_lines, stats = _reduce_skillsbench_dialogue_lines(
                observation_lines,
                max_lines=max_observation_lines,
            )
        elif trace_family == "codex_exec":
            reduced_lines = tuple()
            normalized_observation_lines = tuple(
                normalize_text(line) for line in observation_lines if normalize_text(line)
            )
            stats = _build_reduction_stats(
                original_lines=normalized_observation_lines,
                reduced_lines=reduced_lines,
            )
        else:
            reduced_lines, stats = reduce_log_blocks(
                observation_lines,
                action_text=action_text,
                tool_name=tool_name,
                tool_args_text=tool_args_text,
                result_text=result_text,
                status=status,
                max_lines=max_observation_lines,
            )
    elif normalized_dataset == "terminalbench":
        reduced_lines, stats = reduce_log_blocks(
            observation_lines,
            action_text=action_text,
            tool_name=tool_name,
            tool_args_text=tool_args_text,
            result_text=result_text,
            status=status,
            max_lines=max_observation_lines,
        )
    else:
        reduced_lines, stats = reduce_observation_lines(
            observation_lines,
            action_text=action_text,
            tool_name=tool_name,
            tool_args_text=tool_args_text,
            result_text=result_text,
            status=status,
            max_lines=max_observation_lines,
        )
    if not metadata_lines:
        metadata_lines = ("NO_METADATA",)
    return StepView(
        metadata_lines=metadata_lines,
        observation_lines=reduced_lines,
        action_text=action_text,
        tool_name=tool_name,
        tool_args_text=tool_args_text,
        result_text=result_text,
        status=status,
        reduction_stats=stats,
    )


def serialize_step_view(step_view: StepView) -> str:
    return (
        f"METADATA=[{step_view.metadata_text}] "
        f"OBSERVATION=[{step_view.observation_text}] "
        f"ACTION=[action={step_view.action_text}; tool={step_view.tool_name}; args={step_view.tool_args_text}] "
        f"RESULT=[status={step_view.status}; text={step_view.result_text}]"
    )


def build_step_payload(
    step: StepRecord,
    representation_mode: RepresentationMode = "legacy",
    max_observation_lines: int = 8,
    dataset_name: str | None = None,
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
) -> str | StepView:
    if representation_mode == "legacy":
        return serialize_step(step)
    if representation_mode == "source-raw":
        return serialize_source_raw_step(step)
    if representation_mode in {"reduced-dense", "hybrid"}:
        return build_step_view(
            step,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
    raise ValueError(f"Unsupported representation_mode: {representation_mode}")


def payload_to_text(payload: str | StepView) -> str:
    if isinstance(payload, StepView):
        return serialize_step_view(payload)
    return payload


def summarize_representation_stats(
    trajectories: list[TrajectoryRecord],
    representation_mode: RepresentationMode,
    max_observation_lines: int = 8,
    dataset_name: str | None = None,
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
) -> dict:
    if representation_mode in {"legacy", "source-raw"}:
        serializer = serialize_step if representation_mode == "legacy" else serialize_source_raw_step
        serialized = [
            serializer(step)
            for trajectory in trajectories
            for step in trajectory.steps
        ]
        token_counts = [_count_tokens(text) for text in serialized]
        return {
            "representation_mode": representation_mode,
            "num_steps": len(serialized),
            "token_count_summary": _summarize_numeric(token_counts),
        }

    views = [
        build_step_view(
            step,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        for trajectory in trajectories
        for step in trajectory.steps
    ]
    original_tokens = [view.reduction_stats.original_token_count for view in views]
    retained_tokens = [view.reduction_stats.retained_token_count for view in views]
    retained_lines = [view.reduction_stats.retained_line_count for view in views]
    original_lines = [view.reduction_stats.original_line_count for view in views]
    return {
        "representation_mode": representation_mode,
        "num_steps": len(views),
        "original_token_summary": _summarize_numeric(original_tokens),
        "retained_token_summary": _summarize_numeric(retained_tokens),
        "original_line_summary": _summarize_numeric(original_lines),
        "retained_line_summary": _summarize_numeric(retained_lines),
    }


def _summarize_numeric(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    counts = Counter(values)
    mean = sum(values) / len(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(mean, 3),
        "most_common": counts.most_common(5),
    }
