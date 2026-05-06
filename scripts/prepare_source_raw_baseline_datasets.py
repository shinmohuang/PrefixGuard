"""Prepare source-raw dataset artifacts for matched StepView baselines."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from monitor_symbolization.data.tau2_bench import parse_tau2_results_dir  # noqa: E402
from monitor_symbolization.data.terminalbench import (  # noqa: E402
    scan_terminalbench_manifest_source_records,
    write_terminalbench_split_manifest,
)
from monitor_symbolization.data.webarena_execution import (  # noqa: E402
    _find_archive_member,
    _source_name_from_archive_path,
    load_webarena_task_metadata,
    parse_execution_render,
    parse_merged_log,
)


WEB_ARENA_ARCHIVES = (
    "data/external/webarena/execution_v1/072023_release_v1/release_v1.0_gpt3.5direct.zip",
    "data/external/webarena/execution_v1/072023_release_v1/release_v1.0_gpt3.5dreasoning.zip",
    "data/external/webarena/execution_v1/072023_release_v1/release_v1.0_gpt4.zip",
    "data/external/webarena/execution_v2/112023_release_v2/v2_919_gpt35_16k_cot.zip",
    "data/external/webarena/execution_v2/112023_release_v2/v2_919_gpt35_16k_cot_na.zip",
    "data/external/webarena/execution_v2/112023_release_v2/v2_919_gpt35_16k_direct.zip",
    "data/external/webarena/execution_v2/112023_release_v2/v2_919_gpt35_16k_direct_na.zip",
    "data/external/webarena/execution_v2/112023_release_v2/v2_919_gpt4_8k_cot.zip",
    "data/external/webarena/execution_v2/112023_release_v2/v2_919_text_bison_001_cot.zip",
)

WEB_ARENA_FULL = Path("data/interim/webarena/source_raw/execution_union_v1_v2_source_raw_labeled_full.jsonl")
WEB_ARENA_FULL_SUMMARY = Path(
    "data/interim/webarena/source_raw/execution_union_v1_v2_source_raw_labeled_full_summary.json"
)
WEB_ARENA_SPLIT = Path("data/interim/webarena/source_raw/execution_union_v1_v2_source_raw_labeled_split.jsonl")
WEB_ARENA_SPLIT_SUMMARY = Path(
    "data/interim/webarena/source_raw/execution_union_v1_v2_source_raw_labeled_split_summary.json"
)

TAU2_FULL = Path("data/interim/tau2_bench/source_raw/results_final_source_raw.jsonl")
TAU2_FULL_SUMMARY = Path("data/interim/tau2_bench/source_raw/results_final_source_raw_summary.json")
TAU2_SPLIT = Path("data/interim/tau2_bench/source_raw/results_final_source_raw_outer_train_val_test.jsonl")
TAU2_SPLIT_SUMMARY = Path(
    "data/interim/tau2_bench/source_raw/results_final_source_raw_outer_train_val_test_summary.json"
)

TERMINAL_FULL = Path("data/interim/terminalbench/source_raw/terminalbench_trajectories_source_raw_full.jsonl")
TERMINAL_FULL_SUMMARY = Path(
    "data/interim/terminalbench/source_raw/terminalbench_trajectories_source_raw_full_summary.json"
)
TERMINAL_SPLIT = Path(
    "data/interim/terminalbench/source_raw/"
    "terminalbench_trajectories_source_raw_traj_split_manifest.jsonl"
)
TERMINAL_SPLIT_SUMMARY = Path(
    "data/interim/terminalbench/source_raw/terminalbench_trajectories_source_raw_traj_split_summary.json"
)

SKILLS_CANONICAL_SPLIT = Path("data/interim/skillsbench/full_repo_main_traces_split.jsonl")
SKILLS_FULL_SUMMARY = Path("data/interim/skillsbench/source_raw/full_repo_main_traces_source_raw_manifest_summary.json")
SKILLS_SPLIT = Path("data/interim/skillsbench/source_raw/full_repo_main_traces_source_raw_split_manifest.jsonl")
SKILLS_SPLIT_SUMMARY = Path(
    "data/interim/skillsbench/source_raw/full_repo_main_traces_source_raw_split_manifest_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild source-raw WebArena/TAU2/TerminalBench/SkillsBench artifacts."
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--only",
        choices=["all", "webarena", "tau2", "terminalbench", "skillsbench"],
        default="all",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _count_source_raw_steps(records: list[dict[str, Any]]) -> int:
    return sum(
        1
        for record in records
        for step in record.get("steps", [])
        if str(step.get("source_raw_text") or "").strip()
    )


def _require_source_raw(records: list[dict[str, Any]], *, dataset_name: str) -> None:
    missing: list[str] = []
    for record in records:
        for index, step in enumerate(record.get("steps", [])):
            if not str(step.get("source_raw_text") or "").strip():
                missing.append(f"{record.get('trajectory_id')}:{index}")
                if len(missing) >= 5:
                    break
        if len(missing) >= 5:
            break
    if missing:
        raise ValueError(f"{dataset_name} source_raw_text missing for steps: {missing}")


def _validate_source_raw_jsonl(path: Path, *, dataset_name: str) -> dict[str, Any]:
    num_records = 0
    num_steps = 0
    num_source_raw_steps = 0
    missing: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            num_records += 1
            for index, step in enumerate(record.get("steps", [])):
                num_steps += 1
                if str(step.get("source_raw_text") or "").strip():
                    num_source_raw_steps += 1
                elif len(missing) < 5:
                    missing.append(f"{record.get('trajectory_id', line_no)}:{index}")
    if missing:
        raise ValueError(f"{dataset_name} source_raw_text missing for steps: {missing}")
    return {
        "num_records": num_records,
        "num_steps": num_steps,
        "num_steps_with_source_raw_text": num_source_raw_steps,
    }


def _run_split(root: Path, args: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(root / "scripts/create_task_grouped_split.py"), *args],
        cwd=root,
        check=True,
    )


def _remove_transient_full(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _parse_webarena_archive(
    archive_path: Path,
    *,
    task_metadata: dict[int, dict],
) -> tuple[list[dict[str, Any]], list[int], dict[int, str]]:
    source_name = _source_name_from_archive_path(archive_path)
    records: list[dict[str, Any]] = []
    render_errors: dict[int, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        merged_log_member = _find_archive_member(archive, "merged_log.txt")
        archive_root = merged_log_member[: -len("merged_log.txt")]
        labels, missing = parse_merged_log(
            archive.read(merged_log_member).decode("utf-8", errors="replace"),
            allow_missing_results=True,
        )
        for task_id in sorted(labels):
            try:
                records.append(
                    parse_execution_render(
                        archive=archive,
                        task_id=task_id,
                        final_success=labels[task_id],
                        task_metadata=task_metadata,
                        split="raw",
                        source_name=source_name,
                        archive_root=archive_root,
                    )
                )
            except ValueError as exc:
                render_errors[task_id] = str(exc)
    return records, missing, render_errors


def prepare_webarena(root: Path) -> dict[str, Any]:
    task_metadata = load_webarena_task_metadata(root / "data/external/webarena/test.raw.json")
    records: list[dict[str, Any]] = []
    missing_results: dict[str, list[int]] = {}
    render_errors: dict[str, dict[int, str]] = {}
    per_source_counts: Counter[str] = Counter()
    per_source_successes: Counter[str] = Counter()

    for archive_text in WEB_ARENA_ARCHIVES:
        archive_path = root / archive_text
        archive_records, missing, archive_render_errors = _parse_webarena_archive(
            archive_path,
            task_metadata=task_metadata,
        )
        records.extend(archive_records)
        if missing:
            missing_results[archive_path.name] = list(missing)
        if archive_render_errors:
            render_errors[archive_path.name] = archive_render_errors
        for record in archive_records:
            source_name = str(record.get("metadata", {}).get("source_name", "UNKNOWN"))
            per_source_counts[source_name] += 1
            per_source_successes[source_name] += int(bool(record.get("final_success")))

    _require_source_raw(records, dataset_name="WebArena")
    _write_jsonl(root / WEB_ARENA_FULL, records)
    full_summary = {
        "dataset": "WebArena",
        "mode": "source_raw",
        "archives": list(WEB_ARENA_ARCHIVES),
        "output": str(WEB_ARENA_FULL),
        "num_records": len(records),
        "num_steps": sum(len(record.get("steps", [])) for record in records),
        "num_steps_with_source_raw_text": _count_source_raw_steps(records),
        "per_source_counts": dict(sorted(per_source_counts.items())),
        "per_source_successes": dict(sorted(per_source_successes.items())),
        "missing_result_task_ids": missing_results,
        "render_errors": render_errors,
    }
    _write_json(root / WEB_ARENA_FULL_SUMMARY, full_summary)
    _run_split(
        root,
        [
            "--input",
            str(WEB_ARENA_FULL),
            "--output",
            str(WEB_ARENA_SPLIT),
            "--summary-output",
            str(WEB_ARENA_SPLIT_SUMMARY),
            "--protocol-mode",
            "outer-train-val-test",
            "--seed",
            "13",
            "--val-ratio",
            "0.1",
            "--test-ratio",
            "0.1",
        ],
    )
    _remove_transient_full(root / WEB_ARENA_FULL)
    full_summary["full_jsonl_retained"] = False
    _write_json(root / WEB_ARENA_FULL_SUMMARY, full_summary)
    return full_summary


def prepare_tau2(root: Path) -> dict[str, Any]:
    records, summary = parse_tau2_results_dir(
        root / "data/external/tau2_bench/results/final",
        pattern="*.json",
        split="raw",
        history_window=12,
    )
    _require_source_raw(records, dataset_name="TAU2Bench")
    _write_jsonl(root / TAU2_FULL, records)
    full_summary = {
        **summary,
        "mode": "source_raw",
        "output": str(TAU2_FULL),
        "num_steps_with_source_raw_text": _count_source_raw_steps(records),
    }
    _write_json(root / TAU2_FULL_SUMMARY, full_summary)
    _run_split(
        root,
        [
            "--input",
            str(TAU2_FULL),
            "--output",
            str(TAU2_SPLIT),
            "--summary-output",
            str(TAU2_SPLIT_SUMMARY),
            "--protocol-mode",
            "outer-train-val-test",
            "--seed",
            "1",
            "--val-ratio",
            "0.15",
            "--test-ratio",
            "0.15",
        ],
    )
    _remove_transient_full(root / TAU2_FULL)
    full_summary["full_jsonl_retained"] = False
    _write_json(root / TAU2_FULL_SUMMARY, full_summary)
    return full_summary


def prepare_terminalbench(root: Path) -> dict[str, Any]:
    records, summary = scan_terminalbench_manifest_source_records(
        root / "data/external/terminalbench/terminalbench-trajectories",
        split="raw",
    )
    full_summary = {
        **summary,
        "mode": "source_raw_split_manifest",
        "output": str(TERMINAL_SPLIT),
        "full_jsonl_retained": False,
        "source_raw_materialization": "load_time_from_parquet",
    }
    _write_json(root / TERMINAL_FULL_SUMMARY, full_summary)
    split_summary = write_terminalbench_split_manifest(
        root / TERMINAL_SPLIT,
        records,
        seed=1,
        fit_ratio=0.7,
        cal_ratio=0.1,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    _write_json(root / TERMINAL_SPLIT_SUMMARY, split_summary)
    full_summary["split_summary"] = split_summary
    _write_json(root / TERMINAL_FULL_SUMMARY, full_summary)
    return full_summary


def prepare_skillsbench(root: Path) -> dict[str, Any]:
    input_path = root / SKILLS_CANONICAL_SPLIT
    if not input_path.exists():
        raw_path = root / "data/interim/skillsbench/full_repo_main_traces.jsonl"
        if not raw_path.exists():
            raise FileNotFoundError(
                "SkillsBench prepare requires either "
                f"{SKILLS_CANONICAL_SPLIT} or {raw_path.relative_to(root)}. "
                "Run scripts/import_skillsbench_traces.py first."
            )
        _run_split(
            root,
            [
                "--input",
                "data/interim/skillsbench/full_repo_main_traces.jsonl",
                "--output",
                str(SKILLS_CANONICAL_SPLIT),
                "--summary-output",
                "data/interim/skillsbench/full_repo_main_traces_split_summary.json",
                "--protocol-mode",
                "outer-train-val-test",
                "--seed",
                "13",
                "--val-ratio",
                "0.1",
                "--test-ratio",
                "0.1",
            ],
        )
    output_path = root / SKILLS_SPLIT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_counts: Counter[str] = Counter()
    trace_formats: Counter[str] = Counter()
    task_ids: set[str] = set()
    num_rows = 0
    missing_trial_dir: list[str] = []
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {})
            trial_dir = str(metadata.get("trial_dir") or "")
            if not trial_dir:
                missing_trial_dir.append(str(record.get("trajectory_id") or line_no))
                continue
            split = str(record.get("split", "train"))
            task_id = str(record["task_id"])
            row = {
                "dataset": "SkillsBench",
                "trajectory_id": str(record["trajectory_id"]),
                "task_id": task_id,
                "split": split,
                "trial_dir": trial_dir,
            }
            dst.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            num_rows += 1
            split_counts[split] += 1
            task_ids.add(task_id)
            trace_formats[str(metadata.get("trace_format", "UNKNOWN"))] += 1
    if missing_trial_dir:
        raise ValueError(f"SkillsBench manifest missing trial_dir for rows: {missing_trial_dir[:5]}")
    split_summary = {
        "dataset": "SkillsBench",
        "mode": "source_raw_split_manifest",
        "protocol_mode": "reuse-canonical-split",
        "input": str(SKILLS_CANONICAL_SPLIT),
        "output": str(SKILLS_SPLIT),
        "manifest_fields": ["dataset", "trajectory_id", "task_id", "split", "trial_dir"],
        "num_manifest_rows": num_rows,
        "num_unique_tasks": len(task_ids),
        "trajectory_counts": dict(sorted(split_counts.items())),
        "trace_formats": dict(sorted(trace_formats.items())),
        "source_raw_materialization": "load_time_from_raw_trial_dir",
    }
    _write_json(root / SKILLS_SPLIT_SUMMARY, split_summary)
    full_summary = {
        **split_summary,
        "full_jsonl_retained": False,
        "split_summary": split_summary,
    }
    _write_json(root / SKILLS_FULL_SUMMARY, full_summary)
    return full_summary


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    summaries: dict[str, dict[str, Any]] = {}
    if args.only in {"all", "webarena"}:
        summaries["webarena"] = prepare_webarena(root)
    if args.only in {"all", "tau2"}:
        summaries["tau2"] = prepare_tau2(root)
    if args.only in {"all", "terminalbench"}:
        summaries["terminalbench"] = prepare_terminalbench(root)
    if args.only in {"all", "skillsbench"}:
        summaries["skillsbench"] = prepare_skillsbench(root)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
