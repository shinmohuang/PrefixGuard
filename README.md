# Monitor Symbolization

Anonymous review artifact for monitor-aware event symbolization experiments.

This repository contains the minimal public code path needed to inspect the
trajectory schema, train the differentiable automaton monitor on a toy
trajectory set, and run self-contained regression tests for the DFA backend,
calibration protocol, and differentiable monitor components.

## Scope

Included:

- `src/monitor_symbolization/`: package source.
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

Run the public regression subset:

```bash
python -m pytest \
  tests/test_dfa_backends.py \
  tests/test_legacy_reproduction.py \
  tests/test_public_differentiable_monitor.py \
  tests/test_differentiable_automaton_sanity_cli.py
```

## Reproducibility Notes

The sanity script uses CPU execution and the training default seed `13`. The
toy dataset already contains explicit `train`, `val`, and `test` split labels.
The script trains on `train`, selects/evaluates on `val`, and writes all outputs
under the requested output directory. The toy sanity run validates the direct
soft monitor path. Exact DFA backend behavior is covered separately by
`tests/test_dfa_backends.py`.

See `REPRODUCIBILITY.md` for the exact public artifact contract and
`DATASETS.md` for real-dataset reconstruction notes.
