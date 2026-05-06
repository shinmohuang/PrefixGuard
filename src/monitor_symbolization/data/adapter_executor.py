from __future__ import annotations

import json
from typing import Any

from monitor_symbolization.data.adapter_spec import (
    ACTION_SOURCE_SELECTORS,
    ARGS_SOURCE_SELECTORS,
    METADATA_SOURCE_SELECTORS,
    OBSERVATION_SOURCE_SELECTORS,
    RESULT_SOURCE_SELECTORS,
    STATUS_SOURCE_SELECTORS,
    TOOL_SOURCE_SELECTORS,
    AdapterSpec,
    AdapterSpecValidationError,
)
from monitor_symbolization.data.schema import StepRecord, StepView
from monitor_symbolization.data.serialization import (
    _split_context_lines,
    normalize_text,
    reduce_dialogue_turns,
    reduce_log_blocks,
    reduce_observation_lines,
)


SUPPORTED_METADATA_SOURCES = set(METADATA_SOURCE_SELECTORS)
SUPPORTED_OBSERVATION_SOURCES = set(OBSERVATION_SOURCE_SELECTORS)
SUPPORTED_ACTION_SOURCES = set(ACTION_SOURCE_SELECTORS)
SUPPORTED_TOOL_SOURCES = set(TOOL_SOURCE_SELECTORS)
SUPPORTED_ARGS_SOURCES = set(ARGS_SOURCE_SELECTORS)
SUPPORTED_RESULT_SOURCES = set(RESULT_SOURCE_SELECTORS)
SUPPORTED_STATUS_SOURCES = set(STATUS_SOURCE_SELECTORS)


def validate_executor_supported_spec(spec: AdapterSpec) -> None:
    unsupported_metadata = sorted(set(spec.field_mapping.metadata_sources) - SUPPORTED_METADATA_SOURCES)
    if unsupported_metadata:
        raise AdapterSpecValidationError(
            f"Unsupported metadata_sources for executor: {unsupported_metadata}"
        )
    if spec.field_mapping.observation_source not in SUPPORTED_OBSERVATION_SOURCES:
        raise AdapterSpecValidationError(
            f"Unsupported observation_source for executor: {spec.field_mapping.observation_source}"
        )
    if spec.field_mapping.action_source not in SUPPORTED_ACTION_SOURCES:
        raise AdapterSpecValidationError(
            f"Unsupported action_source for executor: {spec.field_mapping.action_source}"
        )
    if spec.field_mapping.tool_source not in SUPPORTED_TOOL_SOURCES:
        raise AdapterSpecValidationError(
            f"Unsupported tool_source for executor: {spec.field_mapping.tool_source}"
        )
    if spec.field_mapping.args_source not in SUPPORTED_ARGS_SOURCES:
        raise AdapterSpecValidationError(
            f"Unsupported args_source for executor: {spec.field_mapping.args_source}"
        )
    if spec.field_mapping.result_source not in SUPPORTED_RESULT_SOURCES:
        raise AdapterSpecValidationError(
            f"Unsupported result_source for executor: {spec.field_mapping.result_source}"
        )
    if spec.field_mapping.status_source not in SUPPORTED_STATUS_SOURCES:
        raise AdapterSpecValidationError(
            f"Unsupported status_source for executor: {spec.field_mapping.status_source}"
        )


def _field_text(step: StepRecord, selector: str) -> str:
    if selector == "field:action_text":
        return normalize_text(step.action_text)
    if selector == "field:result_text":
        return normalize_text(step.result_text)
    if selector == "field:tool_name":
        return normalize_text(step.tool_name or "NONE")
    if selector == "field:status":
        return normalize_text(step.status or "NONE")
    if selector == "none":
        return "NONE"
    raise AdapterSpecValidationError(f"Unsupported text selector: {selector}")


def _field_args(step: StepRecord, selector: str) -> dict[str, Any]:
    if selector == "field:tool_args":
        return dict(step.tool_args)
    if selector == "none":
        return {}
    raise AdapterSpecValidationError(f"Unsupported args selector: {selector}")


def _select_metadata_lines(step: StepRecord, selector: str) -> tuple[str, ...]:
    metadata_lines, _ = _split_context_lines(step.context)
    if selector == "context:metadata_block":
        return tuple(normalize_text(line) for line in metadata_lines if normalize_text(line))
    if selector.startswith("context:line:"):
        prefix = selector[len("context:line:") :]
        return tuple(
            normalize_text(line)
            for line in metadata_lines
            if normalize_text(line).startswith(prefix)
        )
    value = _field_text(step, selector)
    return () if value == "NONE" else (value,)


def _select_observation_lines(step: StepRecord, selector: str) -> tuple[str, ...]:
    metadata_lines, observation_lines = _split_context_lines(step.context)
    if selector == "context:observation_block":
        return tuple(normalize_text(line) for line in observation_lines if normalize_text(line))
    if selector == "context:all_lines":
        return tuple(
            normalize_text(line)
            for line in (*metadata_lines, *observation_lines)
            if normalize_text(line)
        )
    value = _field_text(step, selector)
    return () if value == "NONE" else tuple(line for line in value.splitlines() if normalize_text(line))


def _normalize_tool_name(tool_name: str, spec: AdapterSpec) -> str:
    normalized = normalize_text(tool_name or "NONE")
    if spec.policies.tool_normalization.lowercase:
        normalized = normalized.lower()
    alias_map = spec.policies.tool_normalization.alias_map
    if normalized in alias_map:
        normalized = alias_map[normalized]
    return normalize_text(normalized or "NONE") or "NONE"


def _derive_status(step: StepRecord, spec: AdapterSpec, result_text: str) -> str:
    source_status = _field_text(step, spec.field_mapping.status_source)
    if spec.policies.status_policy.mode == "native":
        return source_status
    if spec.policies.status_policy.mode == "none":
        return "NONE"
    if source_status not in ("", "NONE") and spec.policies.status_policy.mode == "native_or_derive":
        return source_status
    result_lower = result_text.lower()
    for pattern in spec.policies.status_policy.derive_from_result_patterns:
        if pattern.pattern.lower() in result_lower:
            return pattern.status
    if spec.policies.status_policy.mode == "derive":
        return "unknown"
    return source_status


def build_step_view_from_spec(
    step: StepRecord,
    spec: AdapterSpec,
    *,
    max_observation_units: int | None = None,
) -> StepView:
    validate_executor_supported_spec(spec)
    metadata_lines: list[str] = []
    for selector in spec.field_mapping.metadata_sources:
        metadata_lines.extend(_select_metadata_lines(step, selector))
    deduped_metadata = tuple(dict.fromkeys(line for line in metadata_lines if line))

    observation_lines = _select_observation_lines(step, spec.field_mapping.observation_source)
    action_text = _field_text(step, spec.field_mapping.action_source)
    tool_name = _normalize_tool_name(_field_text(step, spec.field_mapping.tool_source), spec)
    tool_args = _field_args(step, spec.field_mapping.args_source)
    tool_args_text = json.dumps(tool_args, sort_keys=True, ensure_ascii=True)
    result_text = _field_text(step, spec.field_mapping.result_source)
    status = _derive_status(step, spec, result_text)
    max_units = max_observation_units or spec.policies.max_observation_units

    if spec.policies.reducer_kind == "dialogue_turns":
        reduced_lines, stats = reduce_dialogue_turns(
            observation_lines,
            action_text=action_text,
            tool_name=tool_name,
            tool_args_text=tool_args_text,
            result_text=result_text,
            status=status,
            max_lines=max_units,
        )
    elif spec.policies.reducer_kind == "log_blocks":
        reduced_lines, stats = reduce_log_blocks(
            observation_lines,
            action_text=action_text,
            tool_name=tool_name,
            tool_args_text=tool_args_text,
            result_text=result_text,
            status=status,
            max_lines=max_units,
        )
    else:
        reduced_lines, stats = reduce_observation_lines(
            observation_lines,
            action_text=action_text,
            tool_name=tool_name,
            tool_args_text=tool_args_text,
            result_text=result_text,
            status=status,
            max_lines=max_units,
        )

    if not deduped_metadata:
        deduped_metadata = ("NO_METADATA",)
    return StepView(
        metadata_lines=deduped_metadata,
        observation_lines=reduced_lines,
        action_text=action_text or "NONE",
        tool_name=tool_name or "NONE",
        tool_args_text=tool_args_text,
        result_text=result_text or "NONE",
        status=status or "NONE",
        reduction_stats=stats,
    )
