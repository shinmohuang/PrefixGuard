from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup


MERGED_LOG_CONFIG_RE = re.compile(r"\[Config file\]:\s+config_files/(\d+)\.json")
MERGED_LOG_RESULT_RE = re.compile(r"\[Result\]\s+\((PASS|FAIL)\)\s+config_files/(\d+)\.json")
ACTION_NAME_RE = re.compile(r"^`{3}\s*([a-zA-Z_]+)|^([a-zA-Z_]+)")
ACTION_ARG_RE = re.compile(r"\[([^\]]*)\]")
FENCED_ACTION_RE = re.compile(r"```(.*?)```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`\n][^`]*)`")
LEADING_STEP_RE = re.compile(r"^\s*\d+\.\s*")
KNOWN_ACTIONS = (
    "click",
    "type",
    "goto",
    "scroll",
    "stop",
    "go_back",
    "go_forward",
    "new_tab",
    "tab_focus",
    "press",
    "wait",
    "hover",
)


@dataclass(frozen=True)
class ExecutionTraceStep:
    step_id: int
    context: str
    action_text: str
    tool_name: str
    tool_args: dict
    result_text: str
    status: str
    source_raw_text: str


def load_webarena_task_metadata(path: str | Path) -> dict[int, dict]:
    dataset_path = Path(path)
    tasks = json.loads(dataset_path.read_text(encoding="utf-8"))
    return {int(task["task_id"]): task for task in tasks}


def _source_name_from_archive_path(path: str | Path) -> str:
    archive_name = Path(path).name
    stem = archive_name.removesuffix(".zip")
    if stem.startswith("release_v1.0_"):
        stem = stem[len("release_v1.0_") :]
    return stem.replace(".", "")


def _find_archive_member(
    archive: zipfile.ZipFile,
    expected_name: str,
) -> str:
    names = archive.namelist()
    if expected_name in names:
        return expected_name

    suffix = f"/{expected_name}"
    matches = [name for name in names if name.endswith(suffix)]
    if not matches:
        raise ValueError(
            f"Archive is missing expected file '{expected_name}': {archive.filename}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Archive contains multiple candidates for '{expected_name}': {matches[:5]}"
        )
    return matches[0]


def parse_merged_log(
    content: str,
    *,
    allow_missing_results: bool = False,
) -> tuple[dict[int, bool], list[int]]:
    labels: dict[int, bool] = {}
    seen_configs: set[int] = set()
    for line in content.splitlines():
        config_match = MERGED_LOG_CONFIG_RE.search(line)
        if config_match:
            seen_configs.add(int(config_match.group(1)))
        result_match = MERGED_LOG_RESULT_RE.search(line)
        if result_match:
            labels[int(result_match.group(2))] = result_match.group(1) == "PASS"
    missing = sorted(seen_configs.difference(labels))
    if missing and not allow_missing_results:
        raise ValueError(f"Merged log is missing terminal results for task ids: {missing[:10]}")
    return labels, missing


def _extract_texts(soup: BeautifulSoup, name: str, class_name: str) -> list[str]:
    return [
        tag.get_text("\n", strip=True)
        for tag in soup.find_all(name, {"class": class_name})
    ]


def _normalize_url(raw_url: str) -> str:
    if raw_url.startswith("URL: "):
        return raw_url[len("URL: ") :].strip()
    return raw_url.strip()


def _normalize_action_line(line: str) -> str:
    return LEADING_STEP_RE.sub("", line).strip()


def _is_action_line(line: str) -> bool:
    normalized = _normalize_action_line(line)
    return any(
        normalized == action or normalized.startswith(f"{action} ")
        for action in KNOWN_ACTIONS
    )


def _contains_action_line(text: str) -> bool:
    return any(_is_action_line(line) for line in text.splitlines())


def _extract_action_block(text: str) -> str:
    action_lines = [
        _normalize_action_line(line)
        for line in text.splitlines()
        if _is_action_line(line)
    ]
    return "\n".join(action_lines).strip()


def _extract_inline_action_block(text: str) -> str:
    inline_segments = [
        segment.strip().replace("\\n", "\n")
        for segment in INLINE_CODE_RE.findall(text)
    ]
    action_segments = [
        segment
        for segment in inline_segments
        if _contains_action_line(segment)
    ]
    return "\n".join(action_segments).strip()


def _normalize_action(raw_prediction: str) -> str:
    action = raw_prediction.replace("\\n", "\n").strip()
    fenced_segments = [segment.strip() for segment in FENCED_ACTION_RE.findall(action)]
    if fenced_segments:
        action_like_segments = [
            segment.replace("\\n", "\n").strip()
            for segment in fenced_segments
            if _contains_action_line(segment.replace("\\n", "\n"))
        ]
        if action_like_segments:
            action = action_like_segments[-1]
        else:
            action = fenced_segments[-1]
    elif action.startswith("```") and action.endswith("```"):
        action = action[3:-3].strip()
    else:
        extracted = _extract_inline_action_block(action)
        if not extracted:
            extracted = _extract_action_block(action)
        if extracted:
            action = extracted
    return action.strip()


def _extract_action_object_field(
    action_object: str,
    field_name: str,
    next_field_name: str | None = None,
) -> str:
    marker = f"'{field_name}': "
    start = action_object.find(marker)
    if start < 0:
        return ""
    value_start = start + len(marker)
    if value_start >= len(action_object):
        return ""
    quote = action_object[value_start]
    if quote not in ("'", '"'):
        return ""

    i = value_start + 1
    while i < len(action_object):
        char = action_object[i]
        if char == "\\":
            i += 2
            continue
        if char == quote:
            tail = action_object[i + 1 :]
            if next_field_name is None:
                if tail.lstrip().startswith("}"):
                    return action_object[value_start + 1 : i]
            else:
                expected = f", '{next_field_name}':"
                if tail.startswith(expected):
                    return action_object[value_start + 1 : i]
        i += 1
    return ""


def _extract_action_object_answer(action_object: str) -> str:
    return _extract_action_object_field(
        action_object=action_object,
        field_name="answer",
        next_field_name="raw_prediction",
    )


def _extract_action_object_raw_prediction(action_object: str) -> str:
    return _extract_action_object_field(
        action_object=action_object,
        field_name="raw_prediction",
        next_field_name=None,
    )


def _extract_action_name(action_text: str) -> str:
    for line in action_text.splitlines():
        candidate = _normalize_action_line(line)
        if not candidate:
            continue
        match = ACTION_NAME_RE.match(candidate)
        if not match:
            continue
        action_name = match.group(1) or match.group(2)
        if action_name is not None and action_name in KNOWN_ACTIONS:
            return action_name
    raise ValueError(f"Unable to parse action name from prediction: {action_text}")


def _canonicalize_prediction(
    prediction: str,
    action_object: str,
) -> tuple[str, str]:
    normalized = _normalize_action(prediction)
    if normalized:
        try:
            _extract_action_name(normalized)
            return normalized, "ok"
        except ValueError:
            pass

    answer = _extract_action_object_answer(action_object).strip()
    if answer:
        return "stop [EARLY_STOP]", "early_stop"

    if prediction.strip():
        return "stop [INVALID_ACTION]", "invalid_action"
    return "stop [INVALID_ACTION]", "invalid_action"


def _extract_action_args(action_text: str) -> list[str]:
    return ACTION_ARG_RE.findall(action_text)


def _build_context(intent: str, current_url: str, observation: str) -> str:
    return "\n".join(
        [
            f"task_intent={intent}",
            f"current_url={current_url}",
            "observation=",
            observation,
        ]
    )


def _build_result(next_url: str | None, next_observation: str | None) -> str:
    if next_url is None or next_observation is None:
        return ""
    return "\n".join(
        [
            f"next_url={next_url}",
            "next_observation=",
            next_observation,
        ]
    )


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _build_source_raw_text(
    *,
    task_id: int,
    step_id: int,
    intent: str,
    current_url: str,
    observation: str,
    action_object: str,
    next_url: str | None,
    next_observation: str | None,
    source_name: str,
    render_file: str,
) -> str:
    return _stable_json(
        {
            "dataset": "WebArenaExecution",
            "source_name": source_name,
            "render_file": render_file,
            "task_id": task_id,
            "step_id": step_id,
            "intent": intent,
            "current_url": current_url,
            "observation": observation,
            "action_object": action_object,
            "next_url": next_url,
            "next_observation": next_observation,
        }
    )


def parse_execution_render_html(
    html: str,
    *,
    task_id: int,
    final_success: bool,
    task_metadata: dict[int, dict] | None = None,
    split: str = "coldstart",
    source_name: str = "execution",
    render_file: str = "",
    source_archive: str | None = None,
) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    observations = _extract_texts(soup, "div", "state_obv")
    urls = [_normalize_url(text) for text in _extract_texts(soup, "h3", "url")]
    action_objects = _extract_texts(soup, "div", "action_object")
    predictions = [
        _normalize_action(_extract_action_object_raw_prediction(action_object))
        for action_object in action_objects
    ]

    if not observations:
        raise ValueError(f"Render file {render_file or task_id} contains no observations")
    if len(predictions) != len(action_objects):
        raise ValueError(
            f"Render file {render_file or task_id} has inconsistent action parsing lengths: "
            f"{len(predictions)} predictions, {len(action_objects)} action objects"
        )
    if len(observations) != len(urls):
        raise ValueError(
            f"Render file {render_file or task_id} has inconsistent observation/url lengths: "
            f"{len(observations)} observations, {len(urls)} urls"
        )
    if len(observations) not in {len(action_objects), len(action_objects) + 1}:
        raise ValueError(
            f"Render file {render_file or task_id} has inconsistent sequence lengths: "
            f"{len(observations)} observations, {len(urls)} urls, "
            f"{len(predictions)} predictions, {len(action_objects)} action objects"
        )

    task = task_metadata.get(task_id, {}) if task_metadata else {}
    intent = str(task.get("intent", "UNKNOWN_INTENT"))
    steps: list[ExecutionTraceStep] = []
    for step_id, (observation, current_url, prediction, action_object) in enumerate(
        zip(observations, urls, predictions, action_objects)
    ):
        next_url = urls[step_id + 1] if step_id + 1 < len(urls) else None
        next_observation = observations[step_id + 1] if step_id + 1 < len(observations) else None
        canonical_prediction, status = _canonicalize_prediction(
            prediction=prediction,
            action_object=action_object,
        )
        action_name = _extract_action_name(canonical_prediction)
        action_args = _extract_action_args(canonical_prediction)
        steps.append(
            ExecutionTraceStep(
                step_id=step_id,
                context=_build_context(intent=intent, current_url=current_url, observation=observation),
                action_text=canonical_prediction,
                tool_name=action_name,
                tool_args={
                    "arguments": action_args,
                    "raw_prediction": prediction,
                    "raw_answer": _extract_action_object_answer(action_object),
                    "action_object": action_object,
                },
                result_text=_build_result(next_url=next_url, next_observation=next_observation),
                status=status,
                source_raw_text=_build_source_raw_text(
                    task_id=task_id,
                    step_id=step_id,
                    intent=intent,
                    current_url=current_url,
                    observation=observation,
                    action_object=action_object,
                    next_url=next_url,
                    next_observation=next_observation,
                    source_name=source_name,
                    render_file=render_file,
                ),
            )
        )

    metadata = {
        "intent": intent,
        "intent_template_id": task.get("intent_template_id"),
        "source_name": source_name,
        "render_file": render_file,
    }
    if source_archive is not None:
        metadata["source_archive"] = source_archive

    return {
        "trajectory_id": f"{source_name}-{task_id}",
        "task_id": task_id,
        "final_success": final_success,
        "failure_bucket": "NONE" if final_success else "TASK_FAILURE",
        "split": split,
        "metadata": metadata,
        "steps": [
            {
                "step_id": step.step_id,
                "context": step.context,
                "action_text": step.action_text,
                "tool_name": step.tool_name,
                "tool_args": step.tool_args,
                "result_text": step.result_text,
                "status": step.status,
                "source_raw_text": step.source_raw_text,
            }
            for step in steps
        ],
    }


def parse_execution_render(
    archive: zipfile.ZipFile,
    task_id: int,
    final_success: bool,
    task_metadata: dict[int, dict] | None = None,
    split: str = "coldstart",
    source_name: str = "execution",
    archive_root: str = "",
) -> dict:
    render_name = f"render_{task_id}.html"
    render_member = f"{archive_root}{render_name}"
    if render_member not in archive.namelist():
        raise ValueError(f"Archive is missing expected render file: {render_member}")

    html = archive.read(render_member).decode("utf-8", errors="replace")
    return parse_execution_render_html(
        html,
        task_id=task_id,
        final_success=final_success,
        task_metadata=task_metadata,
        split=split,
        source_name=source_name,
        render_file=render_member,
        source_archive=archive.filename,
    )


def parse_webarena_execution_archive(
    archive_path: str | Path,
    task_metadata: dict[int, dict] | None = None,
    limit: int | None = None,
    split: str = "coldstart",
    allow_missing_results: bool = False,
) -> tuple[list[dict], list[int]]:
    zip_path = Path(archive_path)
    source_name = _source_name_from_archive_path(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        merged_log_member = _find_archive_member(archive, "merged_log.txt")
        archive_root = merged_log_member[: -len("merged_log.txt")]
        labels, missing = parse_merged_log(
            archive.read(merged_log_member).decode("utf-8", errors="replace"),
            allow_missing_results=allow_missing_results,
        )
        task_ids = sorted(labels)
        if limit is not None:
            task_ids = task_ids[:limit]
        records = [
            parse_execution_render(
                archive=archive,
                task_id=task_id,
                final_success=labels[task_id],
                task_metadata=task_metadata,
                split=split,
                source_name=source_name,
                archive_root=archive_root,
            )
            for task_id in task_ids
        ]
        return records, missing
