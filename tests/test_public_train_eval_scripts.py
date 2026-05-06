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


def test_public_train_script_defaults_to_toy_cpu_protocol(monkeypatch) -> None:
    module = _load_script_module("train_differentiable_automaton.py")
    monkeypatch.setattr(sys, "argv", ["train_differentiable_automaton.py"])

    args = module.parse_args()

    assert args.dataset == Path("data/toy/trajectories.jsonl")
    assert args.device == "cpu"
    assert args.selection_metric == "direct-soft"
    assert args.derive_train_fit_cal is False
    assert args.fit_split == "train"
    assert args.cal_split == "train"
    assert args.val_split == "val"


def test_public_eval_script_defaults_to_test_split_cpu(monkeypatch) -> None:
    module = _load_script_module("evaluate_differentiable_automaton.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_differentiable_automaton.py",
            "--checkpoint",
            "outputs/training_public/differentiable_automaton/best_checkpoint.pt",
        ],
    )

    args = module.parse_args()

    assert args.dataset == Path("data/toy/trajectories.jsonl")
    assert args.eval_split == "test"
    assert args.device == "cpu"
    assert args.dfa_backend == "aalpy-rpni"
