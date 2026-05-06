from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(script_name: str):
    script_path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_experiment_manifest_lists_core_real_datasets() -> None:
    manifest_path = REPO_ROOT / "configs" / "main_experiments.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert set(manifest["datasets"]) == {"webarena", "tau2", "terminalbench", "skillsbench"}
    assert {run["family"] for run in manifest["runs"]} == set(manifest["datasets"])
    assert all("m6" not in json.dumps(run).lower() for run in manifest["runs"])


def test_reproduce_main_dry_run_emits_train_and_eval_commands() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_main_experiments.py",
            "--families",
            "webarena",
            "--seeds",
            "1",
            "--stage",
            "all",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    commands = [" ".join(item["command"]) for item in payload["commands"]]

    assert any("prepare_source_raw_baseline_datasets.py" in command for command in commands)
    assert any("train_differentiable_automaton.py" in command for command in commands)
    assert any("evaluate_differentiable_automaton.py" in command for command in commands)


def test_verify_dataset_artifacts_has_expected_checksums() -> None:
    module = _load_script_module("verify_dataset_artifacts.py")
    expected = module.expected_artifacts()

    assert expected["webarena"].sha256 == "756ac7d4e9b5797e69bea90e5ffd27ca85ebd1752b5a74f566eb050e4dcf3819"
    assert expected["tau2"].rows == 10832
    assert expected["terminalbench"].rows == 34397
    assert expected["skillsbench"].rows == 10951
