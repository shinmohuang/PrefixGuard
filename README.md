# Monitor Symbolization

Anonymous review artifact for monitor-aware event symbolization experiments.

This repository contains the public code path needed to inspect the trajectory
schema, preprocess the real benchmark artifacts, train and evaluate the
differentiable automaton monitor, and run self-contained regression tests for
the DFA backend, calibration protocol, and differentiable monitor components.

## Scope

Included:

- `src/monitor_symbolization/`: package source.
- `configs/main_experiments.json`: real-data main monitor experiment manifest.
- `scripts/import_*` and `scripts/prepare_source_raw_baseline_datasets.py`: real-data import and preprocessing entry points.
- `scripts/train_differentiable_automaton.py`: split-aware training entry point.
- `scripts/evaluate_differentiable_automaton.py`: checkpoint evaluation entry point.
- `scripts/reproduce_main_experiments.py`: prepare/train/evaluate/summarize driver.
- `scripts/run_differentiable_automaton_sanity.py`: CPU sanity experiment.
- `data/toy/trajectories.jsonl`: small synthetic train/val/test dataset.
- `DATASETS.md`: reconstruction notes for excluded real benchmark datasets.
- `tests/`: selected self-contained regression tests.
- `pyproject.toml` and `uv.lock`: pinned environment metadata.

Excluded:

- raw benchmark traces and intermediate datasets;
- generated experiment outputs, figures, logs, and checkpoints;
- internal planning notes and agent/worktree metadata.

The toy run is intended to verify the implementation path and reproducibility
mechanics. It is not a substitute for the full benchmark datasets used in the
paper. See `DATASETS.md` for the source, preprocessing, checksum, and expected
path contract for excluded real datasets.

## Setup

With `uv`:

```bash
uv sync --group dev
```

With standard Python tooling:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

Python 3.10 or newer is required.

## Quick Checks

Inspect the toy dataset:

```bash
python -m monitor_symbolization inspect data/toy/trajectories.jsonl
```

Run the CPU sanity experiment:

```bash
python scripts/run_differentiable_automaton_sanity.py \
  --dataset data/toy/trajectories.jsonl \
  --selection-metric direct-soft \
  --epochs 1 \
  --output-dir outputs/sanity_public
```

Expected output files:

- `outputs/sanity_public/dataset_summary.json`
- `outputs/sanity_public/sanity_results.json`
- `outputs/sanity_public/differentiable_automaton/`

Train the differentiable monitor on the included toy data:

```bash
python scripts/train_differentiable_automaton.py \
  --dataset data/toy/trajectories.jsonl \
  --epochs 1 \
  --output-dir outputs/training_public/differentiable_automaton
```

Evaluate the resulting checkpoint on the toy test split:

```bash
python scripts/evaluate_differentiable_automaton.py \
  --dataset data/toy/trajectories.jsonl \
  --checkpoint outputs/training_public/differentiable_automaton/best_checkpoint.pt \
  --eval-split test \
  --output outputs/evaluation_public/differentiable_automaton_test.json
```

Dry-run the real-data main monitor reproduction plan:

```bash
python scripts/reproduce_main_experiments.py \
  --stage all \
  --families webarena tau2 terminalbench skillsbench \
  --dry-run
```

After staging/downloading the raw datasets described in `DATASETS.md`, rebuild
and verify canonical artifacts:

```bash
python scripts/reproduce_main_experiments.py --stage prepare
python scripts/verify_dataset_artifacts.py
```

Run the configured main monitor experiments:

```bash
python scripts/reproduce_main_experiments.py --stage train --device cuda
python scripts/reproduce_main_experiments.py --stage eval --device cuda
python scripts/reproduce_main_experiments.py --stage summarize
```

For a cheap executable check of the same driver, override the epoch count and
select one family/seed:

```bash
python scripts/reproduce_main_experiments.py \
  --families webarena \
  --seeds 1 \
  --epochs 1 \
  --device cpu \
  --dry-run
```

Run the public regression subset:

```bash
python -m pytest \
  tests/test_dfa_backends.py \
  tests/test_legacy_reproduction.py \
  tests/test_public_differentiable_monitor.py \
  tests/test_differentiable_automaton_sanity_cli.py \
  tests/test_public_train_eval_scripts.py \
  tests/test_public_main_reproduction.py
```

## Reproducibility Notes

The sanity script uses CPU execution and the training default seed `13`. The
toy dataset already contains explicit `train`, `val`, and `test` split labels.
The script trains on `train`, selects/evaluates on `val`, and writes all outputs
under the requested output directory. The toy sanity run validates the direct
soft monitor path. Exact DFA backend behavior is covered separately by
`tests/test_dfa_backends.py`.

The public train/evaluate scripts default to the included toy dataset and CPU
execution. The real-data reproduction driver reads `configs/main_experiments.json`
and supplies the dataset path, split protocol, seeds, model head, and locked-test
output paths explicitly.

The public main driver covers the core differentiable-monitor experiments on
WebArena, tau2-bench, TerminalBench, and SkillsBench. It intentionally excludes
M6/native adapters and API-based LLM judge baselines.

See `REPRODUCIBILITY.md` for the exact public artifact contract and
`DATASETS.md` for real-dataset reconstruction notes.
