from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRACE_FAMILIES = {
    "browser",
    "visual-browser",
    "desktop",
    "conversation-tooluse",
    "coding-agent",
    "unknown",
}
OBSERVATION_UNITS = {"line", "dialogue_turn", "log_block", "kv_block", "none"}
REDUCER_KINDS = {"lexical_lines", "dialogue_turns", "log_blocks", "kv_blocks", "none"}
STATUS_MODES = {"native", "native_or_derive", "derive", "none"}
DERIVED_STATUSES = {"ok", "tool_error", "failed", "warning", "unknown"}
METADATA_SOURCE_SELECTORS = (
    "context:metadata_block",
    "context:line:dataset=",
    "context:line:task_id=",
    "context:line:domain=",
    "context:line:policy_variant=",
    "context:line:agent_model=",
    "context:line:user_model=",
    "context:line:trial_name=",
    "context:line:condition_dir=",
    "context:line:agent_name=",
    "context:line:agent_model_name=",
    "context:line:trace_format=",
    "context:line:task_prompt=",
    "field:tool_name",
    "field:status",
)
OBSERVATION_SOURCE_SELECTORS = (
    "context:observation_block",
    "context:all_lines",
    "field:action_text",
    "field:result_text",
    "none",
)
ACTION_SOURCE_SELECTORS = ("field:action_text", "field:result_text", "none")
TOOL_SOURCE_SELECTORS = ("field:tool_name", "none")
ARGS_SOURCE_SELECTORS = ("field:tool_args", "none")
RESULT_SOURCE_SELECTORS = ("field:result_text", "field:action_text", "none")
STATUS_SOURCE_SELECTORS = ("field:status", "none")


class AdapterSpecValidationError(ValueError):
    pass


def _reject_extra_keys(obj: dict[str, Any], *, allowed: set[str], label: str) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        raise AdapterSpecValidationError(f"{label} has unsupported keys: {extra}")


def _expect_dict(obj: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise AdapterSpecValidationError(f"{label} must be an object, got {type(obj).__name__}")
    return obj


def _expect_string(obj: dict[str, Any], key: str, *, label: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AdapterSpecValidationError(f"{label}.{key} must be a non-empty string")
    return value


def _expect_enum(obj: dict[str, Any], key: str, *, label: str, allowed: set[str]) -> str:
    value = _expect_string(obj, key, label=label)
    if value not in allowed:
        raise AdapterSpecValidationError(
            f"{label}.{key} must be one of {sorted(allowed)}, got {value!r}"
        )
    return value


def _expect_list_of_strings(obj: dict[str, Any], key: str, *, label: str) -> list[str]:
    value = obj.get(key)
    if not isinstance(value, list) or not value:
        raise AdapterSpecValidationError(f"{label}.{key} must be a non-empty list of strings")
    cleaned: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise AdapterSpecValidationError(f"{label}.{key}[{index}] must be a non-empty string")
        cleaned.append(item)
    return cleaned


def _expect_float_01(obj: dict[str, Any], key: str, *, label: str) -> float:
    value = obj.get(key)
    if not isinstance(value, (int, float)):
        raise AdapterSpecValidationError(f"{label}.{key} must be a number in [0, 1]")
    cast_value = float(value)
    if cast_value < 0.0 or cast_value > 1.0:
        raise AdapterSpecValidationError(f"{label}.{key} must be in [0, 1], got {cast_value}")
    return cast_value


@dataclass(frozen=True)
class AdapterFieldMapping:
    metadata_sources: tuple[str, ...]
    observation_source: str
    action_source: str
    tool_source: str
    args_source: str
    result_source: str
    status_source: str

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "AdapterFieldMapping":
        _reject_extra_keys(
            obj,
            allowed={
                "metadata_sources",
                "observation_source",
                "action_source",
                "tool_source",
                "args_source",
                "result_source",
                "status_source",
            },
            label="field_mapping",
        )
        return cls(
            metadata_sources=tuple(_expect_list_of_strings(obj, "metadata_sources", label="field_mapping")),
            observation_source=_expect_string(obj, "observation_source", label="field_mapping"),
            action_source=_expect_string(obj, "action_source", label="field_mapping"),
            tool_source=_expect_string(obj, "tool_source", label="field_mapping"),
            args_source=_expect_string(obj, "args_source", label="field_mapping"),
            result_source=_expect_string(obj, "result_source", label="field_mapping"),
            status_source=_expect_string(obj, "status_source", label="field_mapping"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_sources": list(self.metadata_sources),
            "observation_source": self.observation_source,
            "action_source": self.action_source,
            "tool_source": self.tool_source,
            "args_source": self.args_source,
            "result_source": self.result_source,
            "status_source": self.status_source,
        }


@dataclass(frozen=True)
class ToolAliasRule:
    source: str
    target: str

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "ToolAliasRule":
        _reject_extra_keys(
            obj,
            allowed={"source", "target"},
            label="policies.tool_normalization.aliases[]",
        )
        return cls(
            source=_expect_string(obj, "source", label="policies.tool_normalization.aliases[]"),
            target=_expect_string(obj, "target", label="policies.tool_normalization.aliases[]"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target}


@dataclass(frozen=True)
class ToolNormalizationPolicy:
    lowercase: bool
    aliases: tuple[ToolAliasRule, ...]

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "ToolNormalizationPolicy":
        _reject_extra_keys(obj, allowed={"lowercase", "aliases"}, label="policies.tool_normalization")
        lowercase = obj.get("lowercase")
        if not isinstance(lowercase, bool):
            raise AdapterSpecValidationError("policies.tool_normalization.lowercase must be boolean")
        aliases_obj = obj.get("aliases", {})
        aliases: list[ToolAliasRule] = []
        if isinstance(aliases_obj, dict):
            for key, value in aliases_obj.items():
                if not isinstance(key, str) or not key.strip():
                    raise AdapterSpecValidationError("tool alias keys must be non-empty strings")
                if not isinstance(value, str) or not value.strip():
                    raise AdapterSpecValidationError("tool alias values must be non-empty strings")
                aliases.append(ToolAliasRule(source=key, target=value))
        elif isinstance(aliases_obj, list):
            aliases = [
                ToolAliasRule.from_dict(_expect_dict(item, label="policies.tool_normalization.aliases[]"))
                for item in aliases_obj
            ]
        else:
            raise AdapterSpecValidationError(
                "policies.tool_normalization.aliases must be an object or a list of {source,target} rules"
            )
        return cls(lowercase=lowercase, aliases=tuple(aliases))

    def to_dict(self) -> dict[str, Any]:
        return {
            "lowercase": self.lowercase,
            "aliases": [alias.to_dict() for alias in self.aliases],
        }

    @property
    def alias_map(self) -> dict[str, str]:
        return {alias.source: alias.target for alias in self.aliases}


@dataclass(frozen=True)
class DerivedStatusPattern:
    pattern: str
    status: str

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "DerivedStatusPattern":
        _reject_extra_keys(
            obj,
            allowed={"pattern", "status"},
            label="policies.status_policy.derive_from_result_patterns[]",
        )
        return cls(
            pattern=_expect_string(obj, "pattern", label="policies.status_policy.derive_from_result_patterns[]"),
            status=_expect_enum(
                obj,
                "status",
                label="policies.status_policy.derive_from_result_patterns[]",
                allowed=DERIVED_STATUSES,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"pattern": self.pattern, "status": self.status}


@dataclass(frozen=True)
class StatusPolicy:
    mode: str
    derive_from_result_patterns: tuple[DerivedStatusPattern, ...]

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "StatusPolicy":
        _reject_extra_keys(
            obj,
            allowed={"mode", "derive_from_result_patterns"},
            label="policies.status_policy",
        )
        patterns_obj = obj.get("derive_from_result_patterns", [])
        if not isinstance(patterns_obj, list):
            raise AdapterSpecValidationError(
                "policies.status_policy.derive_from_result_patterns must be a list"
            )
        return cls(
            mode=_expect_enum(obj, "mode", label="policies.status_policy", allowed=STATUS_MODES),
            derive_from_result_patterns=tuple(
                DerivedStatusPattern.from_dict(_expect_dict(item, label="derived_status_pattern"))
                for item in patterns_obj
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "derive_from_result_patterns": [item.to_dict() for item in self.derive_from_result_patterns],
        }


@dataclass(frozen=True)
class AdapterPolicies:
    observation_unit: str
    reducer_kind: str
    max_observation_units: int
    tool_normalization: ToolNormalizationPolicy
    status_policy: StatusPolicy

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "AdapterPolicies":
        _reject_extra_keys(
            obj,
            allowed={
                "observation_unit",
                "reducer_kind",
                "max_observation_units",
                "tool_normalization",
                "status_policy",
            },
            label="policies",
        )
        max_units = obj.get("max_observation_units")
        if not isinstance(max_units, int) or max_units <= 0:
            raise AdapterSpecValidationError("policies.max_observation_units must be a positive integer")
        return cls(
            observation_unit=_expect_enum(obj, "observation_unit", label="policies", allowed=OBSERVATION_UNITS),
            reducer_kind=_expect_enum(obj, "reducer_kind", label="policies", allowed=REDUCER_KINDS),
            max_observation_units=max_units,
            tool_normalization=ToolNormalizationPolicy.from_dict(
                _expect_dict(obj.get("tool_normalization"), label="policies.tool_normalization")
            ),
            status_policy=StatusPolicy.from_dict(
                _expect_dict(obj.get("status_policy"), label="policies.status_policy")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_unit": self.observation_unit,
            "reducer_kind": self.reducer_kind,
            "max_observation_units": self.max_observation_units,
            "tool_normalization": self.tool_normalization.to_dict(),
            "status_policy": self.status_policy.to_dict(),
        }


@dataclass(frozen=True)
class AdapterConfidence:
    metadata: float
    observation: float
    action: float
    tool: float
    args: float
    result: float
    status: float

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "AdapterConfidence":
        _reject_extra_keys(
            obj,
            allowed={"metadata", "observation", "action", "tool", "args", "result", "status"},
            label="confidence",
        )
        return cls(
            metadata=_expect_float_01(obj, "metadata", label="confidence"),
            observation=_expect_float_01(obj, "observation", label="confidence"),
            action=_expect_float_01(obj, "action", label="confidence"),
            tool=_expect_float_01(obj, "tool", label="confidence"),
            args=_expect_float_01(obj, "args", label="confidence"),
            result=_expect_float_01(obj, "result", label="confidence"),
            status=_expect_float_01(obj, "status", label="confidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "observation": self.observation,
            "action": self.action,
            "tool": self.tool,
            "args": self.args,
            "result": self.result,
            "status": self.status,
        }


@dataclass(frozen=True)
class AdapterSpec:
    dataset_name: str
    version: str
    trace_family: str
    field_mapping: AdapterFieldMapping
    policies: AdapterPolicies
    confidence: AdapterConfidence
    notes: tuple[str, ...]

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "AdapterSpec":
        obj = _expect_dict(obj, label="adapter_spec")
        _reject_extra_keys(
            obj,
            allowed={"dataset_name", "version", "trace_family", "field_mapping", "policies", "confidence", "notes"},
            label="adapter_spec",
        )
        version = _expect_string(obj, "version", label="adapter_spec")
        if version != "v1":
            raise AdapterSpecValidationError(f"adapter_spec.version must be 'v1', got {version!r}")
        notes_obj = obj.get("notes", [])
        if not isinstance(notes_obj, list):
            raise AdapterSpecValidationError("adapter_spec.notes must be a list of strings")
        notes: list[str] = []
        for index, item in enumerate(notes_obj):
            if not isinstance(item, str):
                raise AdapterSpecValidationError(f"adapter_spec.notes[{index}] must be a string")
            notes.append(item)
        return cls(
            dataset_name=_expect_string(obj, "dataset_name", label="adapter_spec"),
            version=version,
            trace_family=_expect_enum(obj, "trace_family", label="adapter_spec", allowed=TRACE_FAMILIES),
            field_mapping=AdapterFieldMapping.from_dict(
                _expect_dict(obj.get("field_mapping"), label="adapter_spec.field_mapping")
            ),
            policies=AdapterPolicies.from_dict(
                _expect_dict(obj.get("policies"), label="adapter_spec.policies")
            ),
            confidence=AdapterConfidence.from_dict(
                _expect_dict(obj.get("confidence"), label="adapter_spec.confidence")
            ),
            notes=tuple(notes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "trace_family": self.trace_family,
            "field_mapping": self.field_mapping.to_dict(),
            "policies": self.policies.to_dict(),
            "confidence": self.confidence.to_dict(),
            "notes": list(self.notes),
        }


def load_adapter_spec(path: str | Path) -> AdapterSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return AdapterSpec.from_dict(payload)


def save_adapter_spec(spec: AdapterSpec, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(spec.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def adapter_spec_json_schema() -> dict[str, Any]:
    return {
        "name": "stepview_adapter_spec",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "dataset_name",
                "version",
                "trace_family",
                "field_mapping",
                "policies",
                "confidence",
                "notes",
            ],
            "properties": {
                "dataset_name": {"type": "string"},
                "version": {"type": "string", "enum": ["v1"]},
                "trace_family": {"type": "string", "enum": sorted(TRACE_FAMILIES)},
                "field_mapping": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "metadata_sources",
                        "observation_source",
                        "action_source",
                        "tool_source",
                        "args_source",
                        "result_source",
                        "status_source",
                    ],
                    "properties": {
                        "metadata_sources": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "enum": list(METADATA_SOURCE_SELECTORS)},
                        },
                        "observation_source": {
                            "type": "string",
                            "enum": list(OBSERVATION_SOURCE_SELECTORS),
                        },
                        "action_source": {"type": "string", "enum": list(ACTION_SOURCE_SELECTORS)},
                        "tool_source": {"type": "string", "enum": list(TOOL_SOURCE_SELECTORS)},
                        "args_source": {"type": "string", "enum": list(ARGS_SOURCE_SELECTORS)},
                        "result_source": {"type": "string", "enum": list(RESULT_SOURCE_SELECTORS)},
                        "status_source": {"type": "string", "enum": list(STATUS_SOURCE_SELECTORS)},
                    },
                },
                "policies": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "observation_unit",
                        "reducer_kind",
                        "max_observation_units",
                        "tool_normalization",
                        "status_policy",
                    ],
                    "properties": {
                        "observation_unit": {"type": "string", "enum": sorted(OBSERVATION_UNITS)},
                        "reducer_kind": {"type": "string", "enum": sorted(REDUCER_KINDS)},
                        "max_observation_units": {"type": "integer", "minimum": 1, "maximum": 32},
                        "tool_normalization": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["lowercase", "aliases"],
                            "properties": {
                                "lowercase": {"type": "boolean"},
                                "aliases": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["source", "target"],
                                        "properties": {
                                            "source": {"type": "string"},
                                            "target": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                        "status_policy": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["mode", "derive_from_result_patterns"],
                            "properties": {
                                "mode": {"type": "string", "enum": sorted(STATUS_MODES)},
                                "derive_from_result_patterns": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["pattern", "status"],
                                        "properties": {
                                            "pattern": {"type": "string"},
                                            "status": {"type": "string", "enum": sorted(DERIVED_STATUSES)},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
                "confidence": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "metadata",
                        "observation",
                        "action",
                        "tool",
                        "args",
                        "result",
                        "status",
                    ],
                    "properties": {
                        "metadata": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "observation": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "action": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "tool": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "args": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "result": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "status": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
                "notes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    }
