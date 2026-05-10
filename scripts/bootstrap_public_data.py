from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import Request
from urllib.request import urlopen
from urllib.request import urlretrieve

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_FAMILIES = ("webarena", "tau2", "terminalbench", "skillsbench")
PUBLIC_DOWNLOADABLE_FAMILIES = ALL_FAMILIES

WEB_ARENA_TEST_RAW_URL = (
    "https://raw.githubusercontent.com/web-arena-x/webarena/"
    "dce04686a56253aefba7b18a4fa0937cf1dc987b/config_files/test.raw.json"
)
WEB_ARENA_GDRIVE_FOLDERS = (
    (
        "https://drive.google.com/drive/folders/18Oww0fAgwhuSjSzxUNgzBUlC6M9IZZB2?usp=sharing",
        "data/external/webarena/execution_v1/072023_release_v1",
    ),
    (
        "https://drive.google.com/drive/folders/1H4wkzDkY2ufiC63DISMXllri0j-ipWcs?usp=sharing",
        "data/external/webarena/execution_v2/112023_release_v2",
    ),
)

TAU2_DATA_REPO = "HuggingFaceH4/tau2-bench-data"
TAU2_DATA_REVISION = "60e37c7a19672769a6034c45a5c8b36e7cd3768b"
TAU2_RESULTS_API_URL = (
    "https://api.github.com/repos/sierra-research/tau2-bench/contents/"
    "data/tau2/results/final?ref=main"
)
TAU2_RESULTS_TREE_URL = "https://github.com/sierra-research/tau2-bench/tree/main/data/tau2/results/final"
TERMINALBENCH_REPO = "yoonholee/terminalbench-trajectories"
SKILLSBENCH_REPO = "benchflow/skillsbench-trajectories-apr2026"
SKILLSBENCH_REVISION = "841dfc7d248bb0b1cd35fa65bb993a1eba0d1d2f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download/stage public benchmark data and rebuild data/interim artifacts "
            "for PrefixGuard reproduction."
        )
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["all"],
        choices=["public", "all", *ALL_FAMILIES],
        help=(
            "'all' restores WebArena, tau2, TerminalBench, and SkillsBench. "
            "'public' is kept as an alias for the same public-downloadable set."
        ),
    )
    parser.add_argument(
        "--tau2-results-dir",
        type=Path,
        default=None,
        help=(
            "Optional local override for completed tau2 evaluation result JSON files. "
            "If omitted, files are downloaded from sierra-research/tau2-bench."
        ),
    )
    parser.add_argument(
        "--tau2-link-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="How to stage --tau2-results-dir/*.json into data/external/tau2_bench/results/final.",
    )
    parser.add_argument("--hf-max-workers", type=int, default=2)
    parser.add_argument(
        "--download-tau2-definitions",
        action="store_true",
        help=(
            "Also mirror HuggingFaceH4/tau2-bench-data benchmark definitions. "
            "The monitor importer only needs the GitHub results/final JSON files."
        ),
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run scripts/verify_dataset_artifacts.py after prepare.",
    )
    parser.add_argument(
        "--strict-verify",
        action="store_true",
        help="Fail if verification row counts or SHA-256 checksums differ.",
    )
    parser.add_argument(
        "--after-prepare",
        choices=["none", "dry-run", "train", "all"],
        default="dry-run",
        help=(
            "What to do after data preparation. 'dry-run' prints the reproduction "
            "commands; 'train' starts training; 'all' runs train/eval/summarize."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without downloading, writing, or preparing data.",
    )
    return parser.parse_args()


def _resolve_families(raw_families: list[str]) -> list[str]:
    if "all" in raw_families:
        return list(ALL_FAMILIES)
    selected: list[str] = []
    for family in raw_families:
        if family == "public":
            selected.extend(PUBLIC_DOWNLOADABLE_FAMILIES)
        elif family not in selected:
            selected.append(family)
    return list(dict.fromkeys(selected))


def _log(action: str, payload: dict) -> None:
    print(json.dumps({"action": action, **payload}, indent=2, sort_keys=True), flush=True)


def _run(root: Path, command: list[str], *, dry_run: bool) -> None:
    _log("run", {"command": command})
    if dry_run:
        return
    subprocess.run(command, cwd=root, check=True)


def _download_url(url: str, output_path: Path, *, dry_run: bool) -> None:
    _log("download_url", {"url": url, "output": str(output_path)})
    if dry_run:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, output_path)


def _download_hf_dataset(
    *,
    repo_id: str,
    local_dir: Path,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
    max_workers: int = 2,
    dry_run: bool,
) -> None:
    payload = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "revision": revision,
        "allow_patterns": allow_patterns,
        "max_workers": max_workers,
    }
    _log("download_hf_dataset", payload)
    if dry_run:
        return
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        max_workers=max_workers,
    )


def _download_github_directory_files(
    *,
    api_url: str,
    output_dir: Path,
    suffix: str,
    dry_run: bool,
) -> None:
    _log(
        "download_github_directory",
        {
            "api_url": api_url,
            "output_dir": str(output_dir),
            "suffix": suffix,
        },
    )
    if dry_run:
        return
    request = Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "PrefixGuard-data-bootstrap",
        },
    )
    with urlopen(request) as response:
        entries = json.loads(response.read().decode("utf-8"))
    if not isinstance(entries, list):
        raise RuntimeError(f"GitHub API did not return a directory listing: {api_url}")

    files = [
        entry
        for entry in entries
        if entry.get("type") == "file"
        and str(entry.get("name", "")).endswith(suffix)
        and entry.get("download_url")
    ]
    if not files:
        raise RuntimeError(f"No {suffix} files found in GitHub directory: {api_url}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in sorted(files, key=lambda item: str(item.get("name"))):
        _download_url(str(entry["download_url"]), output_dir / str(entry["name"]), dry_run=False)


def _download_webarena(root: Path, args: argparse.Namespace) -> None:
    if args.skip_download:
        _log("reuse_webarena_downloads", {"path": "data/external/webarena"})
        return
    _download_url(
        WEB_ARENA_TEST_RAW_URL,
        root / "data/external/webarena/test.raw.json",
        dry_run=args.dry_run,
    )
    for folder_url, output_dir in WEB_ARENA_GDRIVE_FOLDERS:
        _run(
            root,
            [
                sys.executable,
                "-m",
                "gdown",
                "--folder",
                "--remaining-ok",
                "--continue",
                folder_url,
                "-O",
                output_dir,
            ],
            dry_run=args.dry_run,
        )


def _stage_local_tau2_results(root: Path, args: argparse.Namespace) -> None:
    destination = root / "data/external/tau2_bench/results/final"
    source_dir = args.tau2_results_dir.expanduser().resolve()
    source_files = sorted(source_dir.glob("*.json"))
    if not source_files:
        raise SystemExit(f"No *.json files found in --tau2-results-dir: {source_dir}")
    _log(
        "stage_tau2_results",
        {
            "source": str(source_dir),
            "destination": str(destination),
            "num_files": len(source_files),
            "mode": args.tau2_link_mode,
        },
    )
    if args.dry_run:
        return
    destination.mkdir(parents=True, exist_ok=True)
    for source_file in source_files:
        target = destination / source_file.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if args.tau2_link_mode == "symlink":
            target.symlink_to(source_file)
        else:
            shutil.copy2(source_file, target)


def _download_tau2_results(root: Path, args: argparse.Namespace) -> None:
    destination = root / "data/external/tau2_bench/results/final"
    if args.tau2_results_dir is not None:
        _stage_local_tau2_results(root, args)
        return

    existing = sorted(destination.glob("*.json"))
    if args.skip_download:
        if existing:
            _log(
                "reuse_tau2_results",
                {
                    "path": str(destination.relative_to(root)),
                    "num_files": len(existing),
                },
            )
            return
        raise SystemExit(
            "tau2 was selected with --skip-download, but no local "
            "data/external/tau2_bench/results/final/*.json files were found."
        )

    _log(
        "tau2_results_source",
        {
            "source": TAU2_RESULTS_TREE_URL,
            "destination": str(destination.relative_to(root)),
        },
    )
    _download_github_directory_files(
        api_url=TAU2_RESULTS_API_URL,
        output_dir=destination,
        suffix=".json",
        dry_run=args.dry_run,
    )


def _download_tau2(root: Path, args: argparse.Namespace) -> None:
    if args.download_tau2_definitions and not args.skip_download:
        _download_hf_dataset(
            repo_id=TAU2_DATA_REPO,
            revision=TAU2_DATA_REVISION,
            local_dir=root / "data/external/tau2_bench/tau2_data",
            max_workers=args.hf_max_workers,
            dry_run=args.dry_run,
        )
    _download_tau2_results(root, args)


def _download_terminalbench(root: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "scripts/import_terminalbench_trajectories.py",
        "--repo-id",
        TERMINALBENCH_REPO,
        "--input-root",
        "data/external/terminalbench/terminalbench-trajectories",
        "--output-jsonl",
        "data/interim/terminalbench/terminalbench_trajectories_full.jsonl",
        "--output-summary",
        "data/interim/terminalbench/terminalbench_trajectories_full_summary.json",
    ]
    if args.skip_download:
        command.append("--skip-download")
    _run(root, command, dry_run=args.dry_run)


def _download_skillsbench(root: Path, args: argparse.Namespace) -> None:
    input_root = root / "data/external/skillsbench/skillsbench-trajectories"
    if args.skip_download:
        _log("reuse_skillsbench_downloads", {"path": str(input_root.relative_to(root))})
    else:
        _download_hf_dataset(
            repo_id=SKILLSBENCH_REPO,
            revision=SKILLSBENCH_REVISION,
            local_dir=input_root,
            max_workers=args.hf_max_workers,
            dry_run=args.dry_run,
        )
    _run(
        root,
        [
            sys.executable,
            "scripts/import_skillsbench_traces.py",
            "--input-root",
            str(input_root.relative_to(root)),
            "--output-jsonl",
            "data/interim/skillsbench/full_repo_main_traces.jsonl",
            "--output-summary",
            "data/interim/skillsbench/full_repo_main_traces_summary.json",
        ],
        dry_run=args.dry_run,
    )


def _download_family(root: Path, family: str, args: argparse.Namespace) -> None:
    if family == "webarena":
        _download_webarena(root, args)
    elif family == "tau2":
        _download_tau2(root, args)
    elif family == "terminalbench":
        _download_terminalbench(root, args)
    elif family == "skillsbench":
        _download_skillsbench(root, args)
    else:
        raise ValueError(f"Unsupported family: {family}")


def _prepare_family(root: Path, family: str, args: argparse.Namespace) -> None:
    if args.skip_prepare:
        _log("skip_prepare", {"family": family})
        return
    _run(
        root,
        [
            sys.executable,
            "scripts/prepare_source_raw_baseline_datasets.py",
            "--only",
            family,
        ],
        dry_run=args.dry_run,
    )


def _verify(root: Path, families: list[str], args: argparse.Namespace) -> None:
    if not args.verify:
        return
    command = [
        sys.executable,
        "scripts/verify_dataset_artifacts.py",
        "--families",
        *families,
        "--json",
    ]
    _log("verify", {"command": command, "strict": args.strict_verify})
    if args.dry_run:
        return
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        message = (
            "Artifact verification failed. This is expected if an upstream dataset "
            "snapshot differs from the authors' checksum, or if tau2 result JSONs "
            "do not match the authors' run-result bundle."
        )
        if args.strict_verify:
            raise SystemExit(message)
        _log("verify_warning", {"message": message})


def _after_prepare(root: Path, families: list[str], args: argparse.Namespace) -> None:
    if args.after_prepare == "none":
        return
    if args.after_prepare == "dry-run":
        command = [
            sys.executable,
            "scripts/reproduce_main_experiments.py",
            "--stage",
            "all",
            "--families",
            *families,
            "--dry-run",
        ]
        _run(root, command, dry_run=args.dry_run)
        return

    stages = ["train"] if args.after_prepare == "train" else ["train", "eval", "summarize"]
    for stage in stages:
        command = [
            sys.executable,
            "scripts/reproduce_main_experiments.py",
            "--stage",
            stage,
            "--families",
            *families,
        ]
        if stage in {"train", "eval"}:
            command.extend(["--device", args.device])
        _run(root, command, dry_run=args.dry_run)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    families = _resolve_families(args.families)
    _log(
        "bootstrap_start",
        {
            "root": str(root),
            "families": families,
            "skip_download": args.skip_download,
            "skip_prepare": args.skip_prepare,
            "dry_run": args.dry_run,
        },
    )
    for family in families:
        _download_family(root, family, args)
    for family in families:
        _prepare_family(root, family, args)
    _verify(root, families, args)
    _after_prepare(root, families, args)
    _log("bootstrap_complete", {"families": families})


if __name__ == "__main__":
    main()
