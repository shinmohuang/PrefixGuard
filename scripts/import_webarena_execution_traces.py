from __future__ import annotations

import argparse
import json
from pathlib import Path

from monitor_symbolization.data.webarena_execution import (
    load_webarena_task_metadata,
    parse_webarena_execution_archive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import labeled WebArena execution traces.")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "data/external/webarena/execution_v1/072023_release_v1/release_v1.0_gpt3.5direct.zip"
        ),
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=Path("data/external/webarena/test.raw.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/webarena/gpt35direct_labeled_coldstart.jsonl"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=32,
        help="Maximum number of labeled trajectories to export.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="coldstart",
    )
    parser.add_argument(
        "--allow-missing-results",
        action="store_true",
        help="Skip tasks that appear in merged_log.txt without a terminal PASS/FAIL result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    task_metadata = load_webarena_task_metadata(args.task_config)
    imported, missing = parse_webarena_execution_archive(
        archive_path=args.archive,
        task_metadata=task_metadata,
        limit=args.limit,
        split=args.split,
        allow_missing_results=args.allow_missing_results,
    )

    with args.output.open("w", encoding="utf-8") as handle:
        for record in imported:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary = {
        "num_trajectories": len(imported),
        "num_steps": sum(len(record["steps"]) for record in imported),
        "avg_steps_per_trajectory": (
            sum(len(record["steps"]) for record in imported) / len(imported)
            if imported
            else 0.0
        ),
        "num_successes": sum(1 for record in imported if record["final_success"]),
        "num_failures": sum(1 for record in imported if not record["final_success"]),
        "num_missing_results": len(missing),
        "missing_result_task_ids": missing[:20],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
