from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sanity_cli_accepts_step_view_frontend(monkeypatch) -> None:
    module = _load_script_module("run_differentiable_automaton_sanity.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_differentiable_automaton_sanity.py",
            "--step-view-frontend",
            "tau2bench",
        ],
    )

    args = module.parse_args()

    assert args.step_view_frontend == "tau2bench"


def test_sanity_cli_accepts_transfer_full_text_mode(monkeypatch) -> None:
    module = _load_script_module("run_differentiable_automaton_sanity.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_differentiable_automaton_sanity.py",
            "--step-view-text-mode",
            "transfer-full",
        ],
    )

    args = module.parse_args()

    assert args.step_view_text_mode == "transfer-full"
