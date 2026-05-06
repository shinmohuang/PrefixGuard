from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "logs"
OUTPUT_DIR = REPO_ROOT / "outputs"
TRAINING_OUTPUT_DIR = OUTPUT_DIR / "training"
EVALUATION_OUTPUT_DIR = OUTPUT_DIR / "evaluation"
COMPARISON_OUTPUT_DIR = OUTPUT_DIR / "comparison"
SANITY_OUTPUT_DIR = OUTPUT_DIR / "sanity"
SYNTHETIC_OUTPUT_DIR = OUTPUT_DIR / "synthetic"
DEFAULT_MONITOR_DATASET = REPO_ROOT / "data" / "interim" / "tau2_bench" / "results_final_outer_train_val_test.jsonl"
SKILLSBENCH_DATASET_DIR = REPO_ROOT / "data" / "interim" / "skillsbench"
DEFAULT_SKILLSBENCH_FULL_DATASET = SKILLSBENCH_DATASET_DIR / "full_repo_main_traces.jsonl"
DEFAULT_SKILLSBENCH_SPLIT_DATASET = SKILLSBENCH_DATASET_DIR / "full_repo_main_traces_split.jsonl"


def ensure_experiment_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SANITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def default_output_dir(run_name: str) -> Path:
    ensure_experiment_dirs()
    return OUTPUT_DIR / run_name


def default_log_path(run_name: str) -> Path:
    ensure_experiment_dirs()
    return LOG_DIR / f"{run_name}.log"


def training_output_dir(run_name: str = "differentiable_automaton") -> Path:
    ensure_experiment_dirs()
    return TRAINING_OUTPUT_DIR / run_name


def evaluation_output_path(name: str = "differentiable_automaton_eval.json") -> Path:
    ensure_experiment_dirs()
    return EVALUATION_OUTPUT_DIR / name


def comparison_output_path(name: str = "symbolization_methods.json") -> Path:
    ensure_experiment_dirs()
    return COMPARISON_OUTPUT_DIR / name


def sanity_output_dir(run_name: str = "differentiable_automaton") -> Path:
    ensure_experiment_dirs()
    return SANITY_OUTPUT_DIR / run_name


def synthetic_output_dir(run_name: str = "protocol_benchmark") -> Path:
    ensure_experiment_dirs()
    return SYNTHETIC_OUTPUT_DIR / run_name


def default_monitor_dataset_path() -> Path:
    return DEFAULT_MONITOR_DATASET


def skillsbench_dataset_dir() -> Path:
    SKILLSBENCH_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    return SKILLSBENCH_DATASET_DIR


def default_skillsbench_full_dataset_path() -> Path:
    return DEFAULT_SKILLSBENCH_FULL_DATASET


def default_skillsbench_split_dataset_path() -> Path:
    return DEFAULT_SKILLSBENCH_SPLIT_DATASET
