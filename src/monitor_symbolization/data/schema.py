from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StepRecord:
    context: str
    action_text: str
    tool_name: str | None
    tool_args: dict[str, Any] = field(default_factory=dict)
    result_text: str = ""
    status: str | None = None
    source_raw_text: str | None = None


@dataclass(frozen=True)
class ObservationReductionStats:
    original_line_count: int
    retained_line_count: int
    original_char_count: int
    retained_char_count: int
    original_token_count: int
    retained_token_count: int


@dataclass(frozen=True)
class StepView:
    metadata_lines: tuple[str, ...]
    observation_lines: tuple[str, ...]
    action_text: str
    tool_name: str
    tool_args_text: str
    result_text: str
    status: str
    reduction_stats: ObservationReductionStats

    @property
    def metadata_text(self) -> str:
        return " | ".join(self.metadata_lines) if self.metadata_lines else "NONE"

    def filtered_metadata_lines(
        self,
        exclude_prefixes: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        if not exclude_prefixes:
            return self.metadata_lines
        return tuple(
            line
            for line in self.metadata_lines
            if not any(line.startswith(prefix) for prefix in exclude_prefixes)
        )

    @property
    def observation_text(self) -> str:
        return " | ".join(self.observation_lines) if self.observation_lines else "NONE"

    @property
    def lexical_text(self) -> str:
        return self.render_text("full")

    def render_text(
        self,
        mode: str = "full",
        *,
        exclude_metadata_prefixes: tuple[str, ...] = (),
    ) -> str:
        if mode == "lexical":
            mode = "full"
        metadata_lines = self.filtered_metadata_lines(exclude_metadata_prefixes)
        metadata_text = " | ".join(metadata_lines) if metadata_lines else "NONE"
        if mode == "full":
            return (
                f"tool={self.tool_name} "
                f"status={self.status} "
                f"action={self.action_text} "
                f"args={self.tool_args_text} "
                f"result={self.result_text} "
                f"meta={metadata_text}"
            )
        if mode == "transfer-full":
            return (
                f"tool={self.tool_name} "
                f"status={self.status} "
                f"action={self.action_text} "
                f"args={self.tool_args_text} "
                f"result={self.result_text} "
                f"meta={metadata_text} "
                f"observation={self.observation_text}"
            )
        if mode == "drop-tool":
            return (
                "tool=NONE "
                f"status={self.status} "
                f"action={self.action_text} "
                f"args={self.tool_args_text} "
                f"result={self.result_text} "
                f"meta={metadata_text}"
            )
        if mode == "drop-status":
            return (
                f"tool={self.tool_name} "
                "status=NONE "
                f"action={self.action_text} "
                f"args={self.tool_args_text} "
                f"result={self.result_text} "
                f"meta={metadata_text}"
            )
        if mode == "drop-args":
            return (
                f"tool={self.tool_name} "
                f"status={self.status} "
                f"action={self.action_text} "
                "args={} "
                f"result={self.result_text} "
                f"meta={metadata_text}"
            )
        if mode == "drop-result":
            return (
                f"tool={self.tool_name} "
                f"status={self.status} "
                f"action={self.action_text} "
                f"args={self.tool_args_text} "
                "result=NONE "
                f"meta={metadata_text}"
            )
        if mode == "drop-args-result":
            return (
                f"tool={self.tool_name} "
                f"status={self.status} "
                f"action={self.action_text} "
                "args={} "
                "result=NONE "
                f"meta={metadata_text}"
            )
        if mode == "observation-only":
            return self.observation_text
        raise ValueError(f"Unsupported StepView text mode: {mode}")

    @property
    def dense_chunks(self) -> tuple[str, ...]:
        chunks = list(self.metadata_lines)
        chunks.extend(self.observation_lines)
        if not chunks:
            chunks.append("NO_OBSERVATION")
        return tuple(chunks)

    def field_chunks(self) -> tuple[str, ...]:
        return (
            f"tool={self.tool_name}",
            f"status={self.status}",
            f"action={self.action_text}",
            f"args={self.tool_args_text}",
            f"result={self.result_text}",
            f"meta={self.metadata_text}",
            f"observation={self.observation_text}",
        )

    def field_route_groups(self) -> tuple[tuple[str, ...], ...]:
        return (
            (
                f"tool={self.tool_name}",
                f"status={self.status}",
                f"args={self.tool_args_text}",
            ),
            (
                f"action={self.action_text}",
                f"result={self.result_text}",
            ),
            (
                f"meta={self.metadata_text}",
                f"observation={self.observation_text}",
            ),
        )


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    task_id: str
    final_success: bool
    failure_bucket: str
    steps: tuple[StepRecord, ...]
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FutureSignature:
    terminal_label: str
    remaining_steps_bin: str
    failure_bucket: str

    def as_key(self) -> tuple[str, str, str]:
        return (self.terminal_label, self.remaining_steps_bin, self.failure_bucket)


@dataclass(frozen=True)
class PrefixRecord:
    trajectory_id: str
    split: str
    prefix_index: int
    serialized_steps: tuple[str, ...]
    future_signature: FutureSignature
    future_failure_label: int
    final_success: bool
    full_length: int
