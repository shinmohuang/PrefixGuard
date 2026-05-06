from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from monitor_symbolization.data.tau2_bench import parse_tau2_results_dir  # noqa: E402


EXTERNAL_ROOT = REPO_ROOT / "data" / "external" / "tau2_bench"
INTERIM_ROOT = REPO_ROOT / "data" / "interim" / "tau2_bench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import bundled TAU2-bench final-result raw traces into TrajectoryRecord JSONL."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=EXTERNAL_ROOT / "results" / "final",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=INTERIM_ROOT / "results_final_raw.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=INTERIM_ROOT / "results_final_raw_summary.json",
    )
    parser.add_argument("--split", type=str, default="raw")
    parser.add_argument("--history-window", type=int, default=12)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--limit-trajectories", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    imported, summary = parse_tau2_results_dir(
        args.input_root,
        pattern=args.pattern,
        split=args.split,
        history_window=args.history_window,
        limit_files=args.limit_files,
        limit_trajectories=args.limit_trajectories,
    )

    with args.output.open("w", encoding="utf-8") as handle:
        for record in imported:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    payload = {
        **summary,
        "output": str(args.output),
        "history_window": int(args.history_window),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
