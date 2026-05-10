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


def test_bootstrap_tau2_dry_run_uses_public_github_results() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/bootstrap_public_data.py",
            "--families",
            "tau2",
            "--dry-run",
            "--after-prepare",
            "all",
            "--no-verify",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert (
        "https://github.com/sierra-research/tau2-bench/tree/main/data/tau2/results/final"
        in result.stdout
    )
    assert "data/tau2/results/final?ref=main" in result.stdout
    assert "scripts/reproduce_main_experiments.py" in result.stdout
    assert '"train"' in result.stdout
    assert '"eval"' in result.stdout
    assert '"summarize"' in result.stdout


def test_verify_dataset_artifacts_has_expected_checksums() -> None:
    module = _load_script_module("verify_dataset_artifacts.py")
    expected = module.expected_artifacts()

    assert expected["webarena"].sha256 == "756ac7d4e9b5797e69bea90e5ffd27ca85ebd1752b5a74f566eb050e4dcf3819"
    assert expected["tau2"].rows == 10832
    assert expected["terminalbench"].rows == 34397
    assert expected["skillsbench"].rows == 10951


def test_skillsbench_importer_reads_public_acp_trace(tmp_path: Path) -> None:
    module = _load_script_module("import_skillsbench_traces.py")
    trial_dir = (
        tmp_path
        / "jobs"
        / "opus47-with-skills-t1"
        / "2026-04-22__01-27-25"
        / "toy-task__12345678"
    )
    (trial_dir / "trajectory").mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "toy-task",
                "trial_name": "toy-task__12345678",
                "rewards": {"reward": 1.0},
                "agent": "claude-agent-acp",
                "agent_name": "@zed-industries/claude-agent-acp",
                "model": "claude-opus-4-7",
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "config.json").write_text(
        json.dumps(
            {
                "task_path": "/workspace/repos/skillsbench/tasks/toy-task",
                "agent": "claude-agent-acp",
                "model": "claude-opus-4-7",
                "environment": "daytona",
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "trajectory" / "acp_trajectory.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool_call_id": "call-1",
                        "kind": "execute",
                        "title": "Terminal",
                        "status": "completed",
                        "content": [
                            {"type": "content", "content": {"type": "text", "text": "Run checker"}},
                            {"type": "content", "content": {"type": "text", "text": "```console\nok\n```"}},
                        ],
                    }
                ),
                json.dumps({"type": "agent_message", "text": "Done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    record = module._parse_trial(
        trial_dir,
        split="raw",
        history_window=4,
        max_action_chars=1200,
        max_result_chars=2400,
        max_context_chars=4000,
    )

    assert record["final_success"] is True
    assert record["metadata"]["trace_format"] == "acp_trajectory_jsonl"
    assert record["metadata"]["raw_reward"] == 1.0
    assert record["metadata"]["condition_dir"] == "opus47-with-skills-t1"
    assert [step["tool_name"] for step in record["steps"]] == ["bash", "respond"]
    assert record["steps"][-1]["status"] == "success"
    assert all(step["source_raw_text"] for step in record["steps"])
