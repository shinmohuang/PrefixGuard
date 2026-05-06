from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from monitor_symbolization.data.terminalbench import parse_terminalbench_parquet_dir  # noqa: E402


EXTERNAL_ROOT = REPO_ROOT / "data" / "external" / "terminalbench" / "terminalbench-trajectories"
INTERIM_ROOT = REPO_ROOT / "data" / "interim" / "terminalbench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Hugging Face TerminalBench trajectories and convert materialized rows "
            "into TrajectoryRecord JSONL for this repository."
        )
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="yoonholee/terminalbench-trajectories",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=EXTERNAL_ROOT,
        help="Local HF snapshot root containing README.md and data/*.parquet.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=INTERIM_ROOT / "terminalbench_trajectories_full.jsonl",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=INTERIM_ROOT / "terminalbench_trajectories_full_summary.json",
    )
    parser.add_argument("--split", type=str, default="raw")
    parser.add_argument("--history-window", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing local snapshot under --input-root.",
    )
    return parser.parse_args()


def _download_snapshot(repo_id: str, input_root: Path) -> dict[str, str | list[str]]:
    input_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(input_root),
        allow_patterns=["README.md", "data/*.parquet"],
    )
    return {
        "repo_id": repo_id,
        "local_dir": str(input_root),
        "snapshot_path": str(snapshot_path),
        "allow_patterns": ["README.md", "data/*.parquet"],
    }


def main() -> None:
    args = parse_args()
    download_info = None
    if not args.skip_download:
        download_info = _download_snapshot(args.repo_id, args.input_root)

    imported, summary = parse_terminalbench_parquet_dir(
        args.input_root,
        split=args.split,
        history_window=args.history_window,
        limit=args.limit,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in imported:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary_payload = {
        **summary,
        "repo_id": args.repo_id,
        "output_jsonl": str(args.output_jsonl),
        "output_summary": str(args.output_summary),
        "download": download_info,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
