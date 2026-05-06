#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from monitor_symbolization.data.skillsbench_clean_monitor import clean_skillsbench_monitor_trajectory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cleaned SkillsBench monitor dataset.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def _update_summary_counts(record: dict[str, Any], counts: dict[str, Any]) -> None:
    counts["num_trajectories"] += 1
    counts["num_steps"] += len(record.get("steps", []))
    if record.get("final_success"):
        counts["num_successes"] += 1
    else:
        counts["num_failures"] += 1
    counts["unique_tasks"].add(record.get("task_id"))
    trace_format = str(record.get("metadata", {}).get("trace_format") or "UNKNOWN")
    counts["trace_formats"][trace_format] += 1
    split = record.get("split")
    if split is not None:
        counts["trajectory_counts_by_split"][split] += 1
        counts["tasks_by_split"].setdefault(split, set()).add(record.get("task_id"))


def main() -> None:
    args = _parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    input_counts = {
        "num_trajectories": 0,
        "num_steps": 0,
    }
    output_counts: dict[str, Any] = {
        "num_trajectories": 0,
        "num_steps": 0,
        "num_successes": 0,
        "num_failures": 0,
        "unique_tasks": set(),
        "trace_formats": Counter(),
        "trajectory_counts_by_split": Counter(),
        "tasks_by_split": {},
    }
    dropped_by_category: Counter[str] = Counter()
    rewritten_by_category: Counter[str] = Counter()
    replacement_counts: Counter[str] = Counter()
    residual_counts: Counter[str] = Counter()
    rewritten_trajectories = 0
    dropped_trajectories = 0

    with args.input_jsonl.open("r", encoding="utf-8") as src, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            input_counts["num_trajectories"] += 1
            record = json.loads(line)
            input_counts["num_steps"] += len(record.get("steps", []))

            cleaned_record, stats = clean_skillsbench_monitor_trajectory(record)
            if stats["dropped"]:
                dropped_trajectories += 1
                dropped_by_category.update(stats["drop_categories"])
                continue

            if stats["rewrite_categories"]:
                rewritten_trajectories += 1
                rewritten_by_category.update(stats["rewrite_categories"])
            replacement_counts.update(stats["replacement_counts"])
            residual_counts.update(stats["residual_counts"])
            _update_summary_counts(cleaned_record, output_counts)
            dst.write(json.dumps(cleaned_record, ensure_ascii=False) + "\n")

    task_counts_by_split = {
        split: len(task_ids) for split, task_ids in output_counts["tasks_by_split"].items()
    }
    summary = {
        "dataset": "SkillsBench-clean-monitor",
        "profile": "skillsbench-clean-monitor-v1",
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "input_trajectories": input_counts["num_trajectories"],
        "input_steps": input_counts["num_steps"],
        "output_trajectories": output_counts["num_trajectories"],
        "output_steps": output_counts["num_steps"],
        "num_successes": output_counts["num_successes"],
        "num_failures": output_counts["num_failures"],
        "num_unique_tasks": len(output_counts["unique_tasks"]),
        "num_dropped_trajectories": dropped_trajectories,
        "num_rewritten_trajectories": rewritten_trajectories,
        "dropped_by_category": dict(dropped_by_category),
        "rewritten_by_category": dict(rewritten_by_category),
        "replacement_counts": dict(replacement_counts),
        "residual_counts": dict(residual_counts),
        "trace_formats": dict(output_counts["trace_formats"]),
        "trajectory_counts_by_split": dict(output_counts["trajectory_counts_by_split"]),
        "task_counts_by_split": task_counts_by_split,
    }

    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
