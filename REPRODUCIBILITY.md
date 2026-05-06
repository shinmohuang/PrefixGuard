# Reproducibility Contract

## Artifact Goal

This branch is a minimal anonymous review artifact. It is designed to let a
reviewer execute the core monitor-symbolization code path without private raw
datasets, local experiment queues, generated figures, or internal notes.

## Public Data

The repository includes one synthetic dataset:

- `data/toy/trajectories.jsonl`

`DATASETS.md` records the source, preprocessing, checksum, and expected path
contract for the excluded real benchmark datasets.

Each row is a trajectory with:

- `trajectory_id`
- `task_id`
- `final_success`
- `failure_bucket`
- `split`
- ordered `steps`

The included split labels are part of the artifact. The public sanity command
does not resample or alter them.

## Determinism

The public sanity script uses:

- CPU device;
- training seed `13`;
- `fit_split="train"`;
- `cal_split="train"`;
- `selection_metric="direct-soft"` unless overridden;
- `dfa_backend="aalpy-rpni"`.

Floating-point details may vary slightly across PyTorch, BLAS, and platform
versions. The regression tests check protocol behavior and invariants rather
than exact training-loss traces.

## Validation Commands

```bash
python -m monitor_symbolization inspect data/toy/trajectories.jsonl
```

```bash
python scripts/run_differentiable_automaton_sanity.py \
  --dataset data/toy/trajectories.jsonl \
  --selection-metric direct-soft \
  --epochs 1 \
  --output-dir outputs/sanity_public
```

```bash
python scripts/train_differentiable_automaton.py \
  --dataset data/toy/trajectories.jsonl \
  --epochs 1 \
  --output-dir outputs/training_public/differentiable_automaton
```

```bash
python scripts/evaluate_differentiable_automaton.py \
  --dataset data/toy/trajectories.jsonl \
  --checkpoint outputs/training_public/differentiable_automaton/best_checkpoint.pt \
  --eval-split test \
  --output outputs/evaluation_public/differentiable_automaton_test.json
```

```bash
python -m pytest \
  tests/test_dfa_backends.py \
  tests/test_legacy_reproduction.py \
  tests/test_public_differentiable_monitor.py \
  tests/test_differentiable_automaton_sanity_cli.py \
  tests/test_public_train_eval_scripts.py
```

The sanity script is intentionally documented as a direct-soft monitor smoke
run on tiny data. Exact DFA induction is validated by the DFA backend tests,
which use hand-constructed symbol sequences rather than learned toy symbols.

## Exclusions

The public branch intentionally excludes:

- raw WebArena, Tau2Bench, TerminalBench, and SkillsBench traces;
- extra M6/native benchmark adapters and task-only adapter datasets;
- downloaded external data;
- local caches and imported intermediate datasets;
- training logs, scheduler queues, checkpoints, and generated outputs;
- paper drafts, proof notes, refinement logs, and generated figures.

These exclusions do not change the code-level protocol in the included tests or
toy sanity run. They only remove data and artifacts that are not suitable for an
anonymous public review repository.
