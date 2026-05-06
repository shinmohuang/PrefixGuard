from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

from monitor_symbolization.data.webarena_execution import parse_execution_render_html


_RESULT_LINE_RE = re.compile(
    r"\[Result\]\s+\((PASS|FAIL)\)\s+config_files/([^/]+)/(\d+)\.json"
)

_BENCHMARK_SPECS = (
    {
        "site": "classifieds",
        "config_dir": "classifieds_visual",
        "folder": "classifieds_gpt4v_som",
        "config_file": "test_classifieds.raw.json",
    },
    {
        "site": "reddit",
        "config_dir": "reddit_visual",
        "folder": "reddit_gpt4v_som",
        "config_file": "test_reddit.raw.json",
    },
    {
        "site": "shopping",
        "config_dir": "shopping_visual",
        "folder": "shopping_gpt4v_som",
        "config_file": "test_shopping.raw.json",
    },
)


def load_visualwebarena_task_metadata(config_root: str | Path) -> dict[str, dict[int, dict]]:
    root = Path(config_root)
    metadata_by_site: dict[str, dict[int, dict]] = {}
    for spec in _BENCHMARK_SPECS:
        config_path = root / spec["config_file"]
        tasks = json.loads(config_path.read_text(encoding="utf-8"))
        metadata_by_site[spec["site"]] = {
            int(task["task_id"]): task for task in tasks if isinstance(task, dict)
        }
    return metadata_by_site


def parse_visualwebarena_results(
    content: str,
    *,
    expected_config_dir: str,
) -> dict[int, bool]:
    labels: dict[int, bool] = {}
    for outcome, config_dir, task_id in _RESULT_LINE_RE.findall(content):
        if config_dir != expected_config_dir:
            continue
        labels[int(task_id)] = outcome == "PASS"
    return labels


def parse_visualwebarena_execution_tarball(
    archive_path: str | Path,
    *,
    task_metadata_by_site: dict[str, dict[int, dict]],
    split: str = "raw",
    allow_missing_results: bool = True,
    limit: int | None = None,
) -> tuple[list[dict], dict]:
    tar_path = Path(archive_path)
    imported: list[dict] = []
    summary = {
        "source_archive": str(tar_path),
        "num_imported": 0,
        "num_successes": 0,
        "num_failures": 0,
        "missing_result_task_ids": {},
    }

    with tarfile.open(tar_path) as archive:
        members_by_name = {
            member.name: member for member in archive.getmembers() if member.isfile()
        }
        for spec in _BENCHMARK_SPECS:
            folder = spec["folder"]
            site = spec["site"]
            config_dir = spec["config_dir"]
            results_member = members_by_name.get(f"gpt4v_som/{folder}/results.txt")
            if results_member is None:
                raise ValueError(f"Missing VisualWebArena results.txt for {folder}")
            results_text = archive.extractfile(results_member).read().decode(
                "utf-8", errors="replace"
            )
            labels = parse_visualwebarena_results(
                results_text,
                expected_config_dir=config_dir,
            )

            html_members = {
                int(name.split("render_")[-1].split(".html")[0]): name
                for name in members_by_name
                if name.startswith(f"gpt4v_som/{folder}/render_") and name.endswith(".html")
            }
            missing_task_ids = sorted(set(html_members).difference(labels))
            if missing_task_ids and not allow_missing_results:
                raise ValueError(
                    f"VisualWebArena results are missing labels for {site}: {missing_task_ids[:10]}"
                )
            summary["missing_result_task_ids"][site] = missing_task_ids

            for task_id in sorted(labels):
                render_member = html_members.get(task_id)
                if render_member is None:
                    continue
                html = archive.extractfile(render_member).read().decode(
                    "utf-8", errors="replace"
                )
                record = parse_execution_render_html(
                    html,
                    task_id=task_id,
                    final_success=labels[task_id],
                    task_metadata=task_metadata_by_site[site],
                    split=split,
                    source_name=f"visualwebarena-{site}-gpt4v_som",
                    render_file=render_member,
                    source_archive=str(tar_path),
                )
                record["trajectory_id"] = f"visualwebarena-{site}-gpt4v_som-{task_id}"
                record["task_id"] = f"visualwebarena-{site}-{task_id}"
                metadata = dict(record.get("metadata", {}))
                metadata.update(
                    {
                        "dataset": "VisualWebArena",
                        "site": site,
                        "benchmark_folder": folder,
                        "original_task_id": task_id,
                        "result_source": f"gpt4v_som/{folder}/results.txt",
                    }
                )
                record["metadata"] = metadata
                imported.append(record)
                summary["num_successes"] += int(record["final_success"])
                summary["num_failures"] += int(not record["final_success"])
                if limit is not None and len(imported) >= limit:
                    summary["num_imported"] = len(imported)
                    return imported, summary

    summary["num_imported"] = len(imported)
    return imported, summary
