from __future__ import annotations

import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from monitor_symbolization.experiment_paths import REPO_ROOT


DEFAULT_POLL_INTERVAL_SECONDS = 30
DEFAULT_CANCELED_JOB_RETENTION_SECONDS = 24 * 60 * 60
DEFAULT_EXTERNAL_GPU_MEMORY_THRESHOLD_MIB = 1024
DEFAULT_EXTERNAL_GPU_UTILIZATION_THRESHOLD = 10


@dataclass(frozen=True)
class SchedulerPaths:
    root: Path
    state_path: Path
    lock_path: Path
    runtime_dir: Path
    logs_dir: Path

    @classmethod
    def default(cls, root: Path = REPO_ROOT) -> "SchedulerPaths":
        scheduler_root = root / "outputs" / "scheduler"
        return cls(
            root=root,
            state_path=scheduler_root / "state.json",
            lock_path=scheduler_root / "state.lock",
            runtime_dir=scheduler_root / "runtime",
            logs_dir=root / "logs" / "scheduler",
        )

    def ensure_dirs(self) -> None:
        (self.root / "outputs").mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _default_state(gpu_ids: list[int] | None = None) -> dict[str, Any]:
    return {
        "config": {
            "gpu_ids": list(gpu_ids or []),
            "canceled_job_retention_seconds": DEFAULT_CANCELED_JOB_RETENTION_SECONDS,
        },
        "jobs": {},
    }


def _parse_timestamp(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return None


def _ensure_state_defaults(
    state: dict[str, Any],
    gpu_ids: list[int] | None = None,
) -> dict[str, Any]:
    config = state.setdefault("config", {})
    config.setdefault("gpu_ids", list(gpu_ids or []))
    config.setdefault(
        "canceled_job_retention_seconds",
        DEFAULT_CANCELED_JOB_RETENTION_SECONDS,
    )
    state.setdefault("jobs", {})
    return state


def _prune_expired_jobs(state: dict[str, Any]) -> None:
    jobs = state.get("jobs", {})
    retention_seconds = int(
        state.get("config", {}).get(
            "canceled_job_retention_seconds",
            DEFAULT_CANCELED_JOB_RETENTION_SECONDS,
        )
    )
    if retention_seconds < 0:
        return
    cutoff = datetime.now().astimezone() - timedelta(seconds=retention_seconds)
    expired_job_ids: list[str] = []
    for job_id, job in jobs.items():
        if job.get("status") != "canceled":
            continue
        reference_time = _parse_timestamp(
            str(job.get("finished_at") or job.get("updated_at") or "")
        )
        if reference_time is None:
            continue
        if reference_time <= cutoff:
            expired_job_ids.append(job_id)
    for job_id in expired_job_ids:
        jobs.pop(job_id, None)


@contextmanager
def _locked_state(paths: SchedulerPaths, gpu_ids: list[int] | None = None):
    paths.ensure_dirs()
    with paths.lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if paths.state_path.exists():
            state = json.loads(paths.state_path.read_text(encoding="utf-8"))
        else:
            state = _default_state(gpu_ids=gpu_ids)
        _ensure_state_defaults(state, gpu_ids=gpu_ids)
        if gpu_ids is not None and not state["config"].get("gpu_ids"):
            state["config"]["gpu_ids"] = list(gpu_ids)
        _prune_expired_jobs(state)
        yield state
        paths.state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _ensure_job_defaults(job: dict[str, Any]) -> dict[str, Any]:
    job.setdefault("env", {})
    job.setdefault("depends_on_paths", [])
    job.setdefault("depends_on_jobs", [])
    job.setdefault("preferred_gpus", [])
    job.setdefault("priority", 0)
    job.setdefault("device", "gpu")
    job.setdefault("status", "queued")
    job.setdefault("assigned_gpu", None)
    job.setdefault("pid", None)
    job.setdefault("pgid", None)
    job.setdefault("tmux_session", None)
    job.setdefault("return_code", None)
    job.setdefault("log_path", "")
    job.setdefault("runtime_dir", "")
    job.setdefault("launcher_path", "")
    job.setdefault("returncode_path", "")
    job.setdefault("notes", "")
    return job


def _job_sort_key(job: dict[str, Any]) -> tuple[int, str]:
    return (-int(job.get("priority", 0)), str(job.get("created_at", "")))


def _pid_exists(pid: int | None) -> bool:
    if pid is None:
        return False
    proc_stat_path = Path(f"/proc/{pid}/stat")
    if proc_stat_path.exists():
        fields = proc_stat_path.read_text(encoding="utf-8").split()
        if len(fields) >= 3 and fields[2] == "Z":
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _tmux_session_name(job_id: str) -> str:
    normalized = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_"
        for char in job_id
    )
    return f"exp_{normalized}"[:64]


def _tmux_session_exists(session_name: str | None) -> bool:
    if not session_name:
        return False
    result = subprocess.run(  # noqa: S603
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _choose_gpu(
    job: dict[str, Any],
    free_gpus: list[int],
    *,
    allow_nonpreferred_fallback: bool = False,
) -> int | None:
    preferred = [int(gpu) for gpu in job.get("preferred_gpus", [])]
    if preferred:
        for gpu in preferred:
            if gpu in free_gpus:
                return gpu
        if allow_nonpreferred_fallback:
            return free_gpus[0] if free_gpus else None
        return None
    return free_gpus[0] if free_gpus else None


def _query_gpu_usage() -> dict[int, dict[str, int]]:
    result = subprocess.run(  # noqa: S603
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {}
    usage: dict[int, dict[str, int]] = {}
    for raw_line in result.stdout.splitlines():
        parts = [chunk.strip() for chunk in raw_line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpu_id = int(parts[0])
            memory_used_mib = int(parts[1])
            utilization_gpu = int(parts[2])
        except ValueError:
            continue
        usage[gpu_id] = {
            "memory_used_mib": memory_used_mib,
            "utilization_gpu": utilization_gpu,
        }
    return usage


def _externally_busy_gpu_ids(
    gpu_ids: list[int],
    *,
    memory_threshold_mib: int,
    utilization_threshold: int,
) -> set[int]:
    usage = _query_gpu_usage()
    busy_gpu_ids: set[int] = set()
    for gpu_id in gpu_ids:
        snapshot = usage.get(int(gpu_id))
        if snapshot is None:
            continue
        if (
            int(snapshot.get("memory_used_mib", 0)) >= int(memory_threshold_mib)
            or int(snapshot.get("utilization_gpu", 0)) >= int(utilization_threshold)
        ):
            busy_gpu_ids.add(int(gpu_id))
    return busy_gpu_ids


def _dependencies_ready(job: dict[str, Any], jobs: dict[str, dict[str, Any]]) -> bool:
    for path_text in job.get("depends_on_paths", []):
        if not Path(path_text).exists():
            return False
    for dep_id in job.get("depends_on_jobs", []):
        dep = jobs.get(dep_id)
        if dep is None or dep.get("status") != "completed":
            return False
    return True


def _dependencies_failed(job: dict[str, Any], jobs: dict[str, dict[str, Any]]) -> bool:
    for dep_id in job.get("depends_on_jobs", []):
        dep = jobs.get(dep_id)
        if dep is not None and dep.get("status") in {"failed", "canceled", "blocked"}:
            return True
    return False


def _write_launcher_script(
    *,
    job: dict[str, Any],
    assigned_gpu: int | None,
    runtime_dir: Path,
) -> tuple[Path, Path]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    launcher_path = runtime_dir / "launcher.sh"
    returncode_path = runtime_dir / "returncode.json"
    if returncode_path.exists():
        returncode_path.unlink()
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env_lines = []
    if assigned_gpu is not None:
        env_lines.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(assigned_gpu))}")
    for key, value in sorted(job.get("env", {}).items()):
        env_lines.append(f"export {key}={shlex.quote(str(value))}")
    command_literal = " ".join(shlex.quote(token) for token in job["command"])
    launcher = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -uo pipefail",
            f"cd {shlex.quote(job['workdir'])}",
            f"exec > {shlex.quote(str(log_path))} 2>&1",
            *env_lines,
            "set +e",
            command_literal,
            "status=$?",
            f"{shlex.quote(sys.executable)} - <<'PY' "
            f"{shlex.quote(str(returncode_path))} "
            "${status}",
            "import json, sys",
            "from pathlib import Path",
            "Path(sys.argv[1]).write_text(json.dumps({'return_code': int(sys.argv[2])}), encoding='utf-8')",
            "PY",
            "exit ${status}",
        ]
    )
    launcher_path.write_text(f"{launcher}\n", encoding="utf-8")
    launcher_path.chmod(0o755)
    return launcher_path, returncode_path


def _finalize_completed_jobs(state: dict[str, Any]) -> None:
    jobs = state.get("jobs", {})
    for job in jobs.values():
        _ensure_job_defaults(job)
        if job["status"] != "running":
            continue
        returncode_path = Path(job["returncode_path"]) if job.get("returncode_path") else None
        if returncode_path and returncode_path.exists():
            payload = json.loads(returncode_path.read_text(encoding="utf-8"))
            return_code = int(payload["return_code"])
            job["return_code"] = return_code
            job["finished_at"] = _now()
            job["updated_at"] = job["finished_at"]
            job["pid"] = None
            job["pgid"] = None
            if return_code == 0:
                job["status"] = "completed"
            else:
                cancel_requested = bool(job.get("cancel_requested"))
                job["status"] = "canceled" if cancel_requested else "failed"
            continue
        tmux_session = job.get("tmux_session")
        if not _tmux_session_exists(tmux_session) and not _pid_exists(job.get("pid")):
            job["return_code"] = None
            job["finished_at"] = _now()
            job["updated_at"] = job["finished_at"]
            job["pid"] = None
            job["pgid"] = None
            cancel_requested = bool(job.get("cancel_requested"))
            job["status"] = "canceled" if cancel_requested else "failed"


def _launch_job(paths: SchedulerPaths, job: dict[str, Any], assigned_gpu: int | None) -> None:
    runtime_dir = paths.runtime_dir / job["id"]
    launcher_path, returncode_path = _write_launcher_script(
        job=job,
        assigned_gpu=assigned_gpu,
        runtime_dir=runtime_dir,
    )
    session_name = _tmux_session_name(job["id"])
    subprocess.run(  # noqa: S603
        ["tmux", "new-session", "-d", "-s", session_name, "/bin/bash", str(launcher_path)],
        cwd=job["workdir"],
        check=True,
    )
    pane_pid_result = subprocess.run(  # noqa: S603
        ["tmux", "display-message", "-p", "-t", session_name, "#{pane_pid}"],
        cwd=job["workdir"],
        check=True,
        capture_output=True,
        text=True,
    )
    pane_pid_text = pane_pid_result.stdout.strip()
    pane_pid = int(pane_pid_text) if pane_pid_text else None
    job["status"] = "running"
    job["assigned_gpu"] = assigned_gpu
    job["tmux_session"] = session_name
    job["pid"] = pane_pid
    if pane_pid is not None:
        try:
            job["pgid"] = os.getpgid(pane_pid)
        except ProcessLookupError:
            job["pgid"] = None
    else:
        job["pgid"] = None
    job["runtime_dir"] = str(runtime_dir)
    job["launcher_path"] = str(launcher_path)
    job["returncode_path"] = str(returncode_path)
    job["started_at"] = _now()
    job["updated_at"] = job["started_at"]


class ExperimentScheduler:
    def __init__(
        self,
        *,
        paths: SchedulerPaths | None = None,
        gpu_ids: list[int] | None = None,
        respect_external_gpu_usage: bool = False,
        allow_nonpreferred_gpu_fallback: bool = False,
        external_gpu_memory_threshold_mib: int = DEFAULT_EXTERNAL_GPU_MEMORY_THRESHOLD_MIB,
        external_gpu_utilization_threshold: int = DEFAULT_EXTERNAL_GPU_UTILIZATION_THRESHOLD,
    ) -> None:
        self.paths = paths or SchedulerPaths.default()
        self.gpu_ids = list(gpu_ids or [])
        self.respect_external_gpu_usage = bool(respect_external_gpu_usage)
        self.allow_nonpreferred_gpu_fallback = bool(allow_nonpreferred_gpu_fallback)
        self.external_gpu_memory_threshold_mib = int(external_gpu_memory_threshold_mib)
        self.external_gpu_utilization_threshold = int(external_gpu_utilization_threshold)

    def init_state(self, gpu_ids: list[int] | None = None) -> dict[str, Any]:
        ids = self.gpu_ids if gpu_ids is None else gpu_ids
        with _locked_state(self.paths, gpu_ids=ids) as state:
            if ids:
                state.setdefault("config", {})["gpu_ids"] = list(ids)
            return json.loads(json.dumps(state))

    def read_state(self) -> dict[str, Any]:
        with _locked_state(self.paths, gpu_ids=self.gpu_ids) as state:
            return json.loads(json.dumps(state))

    def submit_job(
        self,
        *,
        name: str,
        command: list[str],
        workdir: str | Path | None = None,
        env: dict[str, str] | None = None,
        depends_on_paths: list[str] | None = None,
        depends_on_jobs: list[str] | None = None,
        preferred_gpus: list[int] | None = None,
        priority: int = 0,
        device: str = "gpu",
        log_path: str | Path | None = None,
        job_id: str | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        normalized_workdir = str(Path(workdir or self.paths.root).resolve())
        normalized_job_id = job_id or f"job_{uuid.uuid4().hex[:10]}"
        if log_path is None:
            log_path = self.paths.logs_dir / f"{normalized_job_id}.log"
        payload = _ensure_job_defaults(
            {
                "id": normalized_job_id,
                "name": name,
                "command": list(command),
                "workdir": normalized_workdir,
                "env": dict(env or {}),
                "depends_on_paths": list(depends_on_paths or []),
                "depends_on_jobs": list(depends_on_jobs or []),
                "preferred_gpus": list(preferred_gpus or []),
                "priority": int(priority),
                "device": device,
                "log_path": str(Path(log_path)),
                "created_at": _now(),
                "updated_at": _now(),
                "notes": notes,
            }
        )
        with _locked_state(self.paths, gpu_ids=self.gpu_ids) as state:
            jobs = state.setdefault("jobs", {})
            if normalized_job_id in jobs:
                raise ValueError(f"Job id already exists: {normalized_job_id}")
            jobs[normalized_job_id] = payload
            return json.loads(json.dumps(payload))

    def import_jobs(
        self,
        *,
        spec_path: str | Path,
        skip_existing: bool = False,
    ) -> list[dict[str, Any]]:
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        jobs_spec = spec.get("jobs", [])
        if not isinstance(jobs_spec, list):
            raise ValueError("Job spec must contain a top-level 'jobs' list")

        imported: list[dict[str, Any]] = []
        for item in jobs_spec:
            if not isinstance(item, dict):
                raise ValueError("Each job spec entry must be an object")
            job_id = item.get("job_id")
            if job_id is not None:
                current_state = self.read_state()
                if job_id in current_state.get("jobs", {}):
                    if skip_existing:
                        imported.append(current_state["jobs"][job_id])
                        continue
                    raise ValueError(f"Job id already exists: {job_id}")
            imported.append(
                self.submit_job(
                    name=item["name"],
                    job_id=item.get("job_id"),
                    command=list(item["command"]),
                    workdir=item.get("workdir", self.paths.root),
                    env=dict(item.get("env", {})),
                    depends_on_paths=list(item.get("depends_on_paths", [])),
                    depends_on_jobs=list(item.get("depends_on_jobs", [])),
                    preferred_gpus=list(item.get("preferred_gpus", [])),
                    priority=int(item.get("priority", 0)),
                    device=item.get("device", "gpu"),
                    log_path=item.get("log_path"),
                    notes=item.get("notes", ""),
                )
            )
        return imported

    def remove_job(self, job_id: str) -> dict[str, Any]:
        with _locked_state(self.paths, gpu_ids=self.gpu_ids) as state:
            jobs = state.setdefault("jobs", {})
            job = jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown job: {job_id}")
            if job.get("status") == "running":
                raise ValueError(f"Job is running; use cancel_job instead: {job_id}")
            removed = jobs.pop(job_id)
            return json.loads(json.dumps(removed))

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with _locked_state(self.paths, gpu_ids=self.gpu_ids) as state:
            jobs = state.setdefault("jobs", {})
            job = jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown job: {job_id}")
            _ensure_job_defaults(job)
            status = job.get("status")
            if status == "queued":
                job["status"] = "canceled"
                job["finished_at"] = _now()
                job["updated_at"] = job["finished_at"]
                return json.loads(json.dumps(job))
            if status != "running":
                return json.loads(json.dumps(job))
            tmux_session = job.get("tmux_session")
            if tmux_session and _tmux_session_exists(tmux_session):
                subprocess.run(  # noqa: S603
                    ["tmux", "kill-session", "-t", str(tmux_session)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            pgid = job.get("pgid")
            if pgid is not None:
                try:
                    os.killpg(int(pgid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif job.get("pid") is not None:
                try:
                    os.kill(int(job["pid"]), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            job["cancel_requested"] = True
            job["updated_at"] = _now()
            return json.loads(json.dumps(job))

    def tick(self) -> dict[str, Any]:
        with _locked_state(self.paths, gpu_ids=self.gpu_ids) as state:
            jobs: dict[str, dict[str, Any]] = state.setdefault("jobs", {})
            for job in jobs.values():
                _ensure_job_defaults(job)
            _finalize_completed_jobs(state)

            for job in jobs.values():
                if job["status"] == "queued" and _dependencies_failed(job, jobs):
                    job["status"] = "blocked"
                    job["finished_at"] = _now()
                    job["updated_at"] = job["finished_at"]

            gpu_ids = [int(gpu) for gpu in state.get("config", {}).get("gpu_ids", [])]
            busy_gpu_ids = {
                int(job["assigned_gpu"])
                for job in jobs.values()
                if job["status"] == "running" and job.get("assigned_gpu") is not None
            }
            free_gpu_ids = [gpu for gpu in gpu_ids if gpu not in busy_gpu_ids]
            if self.respect_external_gpu_usage and free_gpu_ids:
                externally_busy = _externally_busy_gpu_ids(
                    free_gpu_ids,
                    memory_threshold_mib=self.external_gpu_memory_threshold_mib,
                    utilization_threshold=self.external_gpu_utilization_threshold,
                )
                free_gpu_ids = [gpu for gpu in free_gpu_ids if gpu not in externally_busy]

            queued_jobs = sorted(
                [job for job in jobs.values() if job["status"] == "queued"],
                key=_job_sort_key,
            )
            for job in queued_jobs:
                if not _dependencies_ready(job, jobs):
                    continue
                assigned_gpu = None
                if job.get("device", "gpu") == "gpu":
                    assigned_gpu = _choose_gpu(
                        job,
                        free_gpu_ids,
                        allow_nonpreferred_fallback=self.allow_nonpreferred_gpu_fallback,
                    )
                    if assigned_gpu is None:
                        continue
                _launch_job(self.paths, job, assigned_gpu)
                if assigned_gpu is not None:
                    free_gpu_ids.remove(assigned_gpu)
            return json.loads(json.dumps(state))

    def run_loop(self, poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS) -> None:
        while True:
            self.tick()
            time.sleep(max(int(poll_interval_seconds), 1))


def parse_env_assignments(assignments: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE env assignment, got: {item}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def parse_gpu_ids(text: str | None) -> list[int]:
    if text is None or text == "":
        return []
    return [int(chunk) for chunk in text.split(",") if chunk.strip()]


def _status_rank(status: str) -> int:
    order = {
        "running": 0,
        "queued": 1,
        "blocked": 2,
        "failed": 3,
        "canceled": 4,
        "completed": 5,
    }
    return order.get(status, 99)


def _truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return f"{text[:width - 3]}..."


def _job_waiting_summary(job: dict[str, Any]) -> str:
    dep_jobs = list(job.get("depends_on_jobs", []))
    dep_paths = list(job.get("depends_on_paths", []))
    parts = []
    if dep_jobs:
        parts.append(f"jobs:{len(dep_jobs)}")
    if dep_paths:
        parts.append(f"paths:{len(dep_paths)}")
    return ",".join(parts) if parts else "-"


def format_jobs_table(
    state: dict[str, Any],
    *,
    visible_statuses: set[str] | None = None,
) -> str:
    jobs = state.get("jobs", {})
    if not jobs:
        return "No jobs in scheduler."

    sorted_jobs = sorted(
        jobs.values(),
        key=lambda item: (
            _status_rank(str(item.get("status", ""))),
            _job_sort_key(item),
        ),
    )
    counts: dict[str, int] = {}
    for job in sorted_jobs:
        status = str(job.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1

    if visible_statuses is None:
        filtered_jobs = sorted_jobs
    else:
        filtered_jobs = [
            job for job in sorted_jobs if str(job.get("status", "unknown")) in visible_statuses
        ]

    headers = ["STATUS", "GPU", "JOB ID", "NAME", "WAITING", "TMUX", "LOG"]
    rows = []
    for job in filtered_jobs:
        rows.append(
            [
                str(job.get("status", "")),
                "-" if job.get("assigned_gpu") is None else str(job.get("assigned_gpu")),
                str(job.get("id", "")),
                str(job.get("name", "")),
                _job_waiting_summary(job),
                str(job.get("tmux_session") or "-"),
                str(job.get("log_path", "")),
            ]
        )

    widths = []
    max_widths = [10, 3, 256, 28, 14, 24, 28]
    for column_index, header in enumerate(headers):
        content_width = max(len(header), *(len(row[column_index]) for row in rows)) if rows else len(header)
        widths.append(min(content_width, max_widths[column_index]))

    def render_row(values: list[str]) -> str:
        padded = []
        for column_index, (value, width) in enumerate(zip(values, widths)):
            rendered = value if headers[column_index] == "JOB ID" else _truncate(value, width)
            padded.append(rendered.ljust(width))
        return "  ".join(padded).rstrip()

    if visible_statuses is None:
        summary_counts = counts
    else:
        summary_counts = {
            status: counts[status]
            for status in sorted(counts, key=_status_rank)
            if status in visible_statuses
        }
    summary = "Jobs: " + ", ".join(
        f"{status}={summary_counts[status]}" for status in sorted(summary_counts, key=_status_rank)
    )
    hidden_counts = {
        status: counts[status]
        for status in sorted(counts, key=_status_rank)
        if status not in summary_counts
    }
    if hidden_counts:
        summary += " | hidden: " + ", ".join(
            f"{status}={hidden_counts[status]}" for status in sorted(hidden_counts, key=_status_rank)
        )

    lines = [summary]
    if rows:
        lines.extend([render_row(headers), render_row(["-" * width for width in widths])])
        lines.extend(render_row(row) for row in rows)
    else:
        lines.append("No jobs match the current filter.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Persistent GPU-aware experiment scheduler.")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root for scheduler state and default workdir.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids used by the scheduler, e.g. '0,1,2'.",
    )
    parser.add_argument(
        "--respect-external-gpu-usage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Avoid launching queued GPU jobs onto GPUs that are currently busy outside the scheduler.",
    )
    parser.add_argument(
        "--allow-nonpreferred-gpu-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Try each job's preferred GPUs first, then fall back to any idle scheduler GPU "
            "when all preferred GPUs are busy."
        ),
    )
    parser.add_argument(
        "--external-gpu-memory-threshold-mib",
        type=int,
        default=DEFAULT_EXTERNAL_GPU_MEMORY_THRESHOLD_MIB,
        help="Treat a GPU as externally busy when memory.used is at least this many MiB.",
    )
    parser.add_argument(
        "--external-gpu-utilization-threshold",
        type=int,
        default=DEFAULT_EXTERNAL_GPU_UTILIZATION_THRESHOLD,
        help="Treat a GPU as externally busy when utilization.gpu is at least this percentage.",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize scheduler state.")
    init_parser.add_argument("--init-gpus", type=str, default=None)

    submit_parser = subparsers.add_parser("submit", help="Submit a job to the scheduler.")
    submit_parser.add_argument("--name", required=True)
    submit_parser.add_argument("--job-id", default=None)
    submit_parser.add_argument("--workdir", type=Path, default=REPO_ROOT)
    submit_parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    submit_parser.add_argument("--preferred-gpus", type=str, default=None)
    submit_parser.add_argument("--depends-on-path", action="append", default=[])
    submit_parser.add_argument("--depends-on-job", action="append", default=[])
    submit_parser.add_argument("--env", action="append", default=[])
    submit_parser.add_argument("--priority", type=int, default=0)
    submit_parser.add_argument("--log-path", type=Path, default=None)
    submit_parser.add_argument("--notes", type=str, default="")
    submit_parser.add_argument("cmd", nargs=argparse.REMAINDER)

    import_parser = subparsers.add_parser("import-spec", help="Import jobs from a JSON spec.")
    import_parser.add_argument("--spec-path", type=Path, required=True)
    import_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose ids already exist in scheduler state.",
    )

    list_parser = subparsers.add_parser("list", help="List jobs in scheduler state.")
    list_group = list_parser.add_mutually_exclusive_group()
    list_group.add_argument(
        "--all",
        action="store_true",
        help="Show all jobs, including completed/canceled/failed entries.",
    )
    list_group.add_argument(
        "--completed",
        action="store_true",
        help="Show only completed jobs.",
    )

    tick_parser = subparsers.add_parser("tick", help="Run one scheduling iteration.")
    tick_parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL_SECONDS)

    daemon_parser = subparsers.add_parser("daemon", help="Run the scheduler loop.")
    daemon_parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL_SECONDS)

    remove_parser = subparsers.add_parser("remove", help="Remove a non-running job.")
    remove_parser.add_argument("job_id")

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a queued or running job.")
    cancel_parser.add_argument("job_id")

    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    gpu_ids = parse_gpu_ids(args.gpus)
    scheduler = ExperimentScheduler(
        paths=SchedulerPaths.default(root=root),
        gpu_ids=gpu_ids,
        respect_external_gpu_usage=args.respect_external_gpu_usage,
        allow_nonpreferred_gpu_fallback=args.allow_nonpreferred_gpu_fallback,
        external_gpu_memory_threshold_mib=args.external_gpu_memory_threshold_mib,
        external_gpu_utilization_threshold=args.external_gpu_utilization_threshold,
    )

    if args.command_name == "init":
        scheduler.init_state(
            gpu_ids=parse_gpu_ids(getattr(args, "init_gpus", None)) or gpu_ids
        )
        print(json.dumps(scheduler.read_state()["config"], indent=2, sort_keys=True))
        return 0

    if args.command_name == "submit":
        if args.cmd[:1] == ["--"]:
            command = args.cmd[1:]
        else:
            command = args.cmd
        if not command:
            raise SystemExit("submit requires a command after '--'")
        job = scheduler.submit_job(
            name=args.name,
            job_id=args.job_id,
            command=command,
            workdir=args.workdir,
            device=args.device,
            preferred_gpus=parse_gpu_ids(args.preferred_gpus),
            depends_on_paths=args.depends_on_path,
            depends_on_jobs=args.depends_on_job,
            env=parse_env_assignments(args.env),
            priority=args.priority,
            log_path=args.log_path,
            notes=args.notes,
        )
        print(json.dumps(job, indent=2, sort_keys=True))
        return 0

    if args.command_name == "import-spec":
        jobs = scheduler.import_jobs(
            spec_path=args.spec_path,
            skip_existing=args.skip_existing,
        )
        print(json.dumps(jobs, indent=2, sort_keys=True))
        return 0

    if args.command_name == "list":
        state = scheduler.read_state()
        if args.all:
            visible_statuses = None
        elif args.completed:
            visible_statuses = {"completed"}
        else:
            visible_statuses = {"running", "queued"}
        print(format_jobs_table(state, visible_statuses=visible_statuses))
        return 0

    if args.command_name == "tick":
        state = scheduler.tick()
        print(format_jobs_table(state))
        return 0

    if args.command_name == "daemon":
        scheduler.run_loop(poll_interval_seconds=args.poll_interval)
        return 0

    if args.command_name == "remove":
        removed = scheduler.remove_job(args.job_id)
        print(json.dumps(removed, indent=2, sort_keys=True))
        return 0

    if args.command_name == "cancel":
        canceled = scheduler.cancel_job(args.job_id)
        print(json.dumps(canceled, indent=2, sort_keys=True))
        return 0

    raise SystemExit(f"Unsupported command: {args.command_name}")


if __name__ == "__main__":
    sys.exit(main())
