from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HumanTraceStep:
    step_id: int
    context: str
    action_text: str
    tool_name: str
    tool_args: dict
    result_text: str
    status: str


def load_webarena_task_metadata(path: str | Path) -> dict[int, dict]:
    dataset_path = Path(path)
    tasks = json.loads(dataset_path.read_text(encoding="utf-8"))
    return {int(task["task_id"]): task for task in tasks}


def _extract_initial_url(trace_network_lines: list[str]) -> str:
    for line in trace_network_lines:
        obj = json.loads(line)
        if obj.get("type") != "resource-snapshot":
            continue
        snapshot = obj.get("snapshot", {})
        request = snapshot.get("request", {})
        url = request.get("url")
        if url:
            return str(url)
    return "UNKNOWN_URL"


def _extract_navigation_url(log_lines: list[str], current_url: str) -> str:
    for line in log_lines:
        stripped = line.strip()
        if stripped.startswith("navigated to "):
            parts = stripped.split('"')
            if len(parts) >= 2:
                return parts[1]
    return current_url


def parse_webarena_human_trace(
    zip_path: str | Path,
    task_metadata: dict[int, dict] | None = None,
) -> dict:
    archive_path = Path(zip_path)
    task_id = int(archive_path.name.split(".")[0])
    task = task_metadata.get(task_id, {}) if task_metadata else {}

    with zipfile.ZipFile(archive_path) as archive:
        trace_lines = archive.read("trace.trace").decode("utf-8", errors="replace").splitlines()
        network_lines = archive.read("trace.network").decode("utf-8", errors="replace").splitlines()

    current_url = _extract_initial_url(network_lines)
    before_by_call: dict[str, dict] = {}
    steps: list[HumanTraceStep] = []

    for line in trace_lines:
        obj = json.loads(line)
        event_type = obj.get("type")
        if event_type == "before":
            before_by_call[obj["callId"]] = obj
        elif event_type == "after":
            call_id = obj["callId"]
            before = before_by_call.get(call_id)
            if before is None:
                continue
            params = dict(before.get("params", {}))
            api_name = str(before.get("apiName", before.get("method", "unknown")))
            tool_name = str(before.get("method", "unknown"))
            logs = [str(item) for item in obj.get("log", [])]
            next_url = _extract_navigation_url(logs, current_url)
            context = (
                f"task_intent={task.get('intent', 'UNKNOWN_INTENT')} "
                f"current_url={current_url}"
            )
            result_text = " | ".join(logs)
            steps.append(
                HumanTraceStep(
                    step_id=len(steps),
                    context=context,
                    action_text=api_name,
                    tool_name=tool_name,
                    tool_args=params,
                    result_text=result_text,
                    status="ok",
                )
            )
            current_url = next_url

    return {
        "trajectory_id": f"human-{task_id}",
        "task_id": task_id,
        "intent": task.get("intent", "UNKNOWN_INTENT"),
        "intent_template_id": task.get("intent_template_id"),
        "source_zip": str(archive_path),
        "label_available": False,
        "final_success": None,
        "failure_bucket": None,
        "split": "coldstart",
        "steps": [
            {
                "step_id": step.step_id,
                "context": step.context,
                "action_text": step.action_text,
                "tool_name": step.tool_name,
                "tool_args": step.tool_args,
                "result_text": step.result_text,
                "status": step.status,
            }
            for step in steps
        ],
    }
