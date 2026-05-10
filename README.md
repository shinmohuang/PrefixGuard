[![arXiv](https://img.shields.io/badge/arXiv-2605.06455-b31b1b.svg)](https://arxiv.org/abs/2605.06455)

# PrefixGuard

PrefixGuard turns agent execution traces into online failure-warning monitors.
Given step-by-step trajectories from a web, tool-use, terminal, or coding agent,
it learns a compact event abstraction and trains a differentiable finite-state
monitor that can raise prefix-time risk alerts before the final task outcome is
known.

This repository contains the public implementation used for the PrefixGuard
monitoring pipeline:

- canonical agent-trace JSONL loading and inspection;
- benchmark importers for WebArena, tau2-bench, TerminalBench, and SkillsBench;
- StepView-based trace representations;
- differentiable automaton monitor training and evaluation;
- DFA backend, calibration, and public smoke/regression tests;
- a small toy dataset that runs without external downloads.

Raw benchmark trajectories, generated checkpoints, logs, and paper figures are
not bundled. See [DATASETS.md](DATASETS.md) for reconstruction notes and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the public reproduction contract.

## Installation

Python 3.10 or newer is required.

With `uv`:

```bash
uv sync --group dev
source .venv/bin/activate
```

If you do not activate the environment, prefix the Python commands below with
`uv run`.

With standard Python tooling:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

The package also installs a small CLI:

```bash
monitor-symbolization --help
python -m monitor_symbolization --help
```

## Quick Start

Inspect the included toy dataset:

```bash
python -m monitor_symbolization inspect data/toy/trajectories.jsonl
```

Train a monitor on the toy train/validation split:

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

For an even smaller end-to-end sanity run:

```bash
python scripts/run_differentiable_automaton_sanity.py \
  --dataset data/toy/trajectories.jsonl \
  --selection-metric direct-soft \
  --epochs 1 \
  --output-dir outputs/sanity_public
```

## Use Your Own Agent Traces

PrefixGuard expects one JSON object per completed trajectory. Convert your raw
agent logs into this canonical JSONL format before training:

```json
{"trajectory_id":"run-0001","task_id":"calendar-task-17","final_success":false,"failure_bucket":"TOOL_ARGUMENT_ERROR","split":"train","metadata":{"dataset":"my_agent_benchmark","agent":"my-agent-v1"},"steps":[{"context":"calendar page open; event form visible","action_text":"create meeting for 9am without an attendee","tool_name":"create_calendar_event","tool_args":{"time":"09:00","attendees":[]},"result_text":"tool rejected the request because attendees is empty","status":"error","source_raw_text":"optional raw step text"}]}
```

Required trajectory fields:

- `trajectory_id`: unique id for a full agent run.
- `task_id`: task id. Keep the same task out of both train and test when
  evaluating out-of-distribution generalization.
- `final_success`: final task success label.
- `steps`: ordered, non-empty list of step records.

Recommended trajectory fields:

- `split`: usually `train`, `val`, or `test`. If omitted, records default to
  `train`.
- `failure_bucket`: `NONE` for successful runs and a stable failure category
  for failed runs.
- `metadata`: dataset, agent, environment, benchmark version, or source file
  provenance.

Required step fields:

- `context`: compact observation or state visible at this step.
- `action_text`: natural-language or normalized action description.
- `tool_name`: tool/API/action type, or `null` if unavailable.

Recommended step fields:

- `tool_args`: normalized tool arguments as a JSON object.
- `result_text`: immediate action result.
- `status`: stable step status such as `ok`, `error`, `success`, or `failure`.
- `source_raw_text`: optional full raw step serialization, required only for
  `--representation-mode source-raw`.

Validate your converted dataset:

```bash
python -m monitor_symbolization inspect data/my_agent/trajectories.jsonl
```

Run a CPU smoke train:

```bash
python scripts/train_differentiable_automaton.py \
  --dataset data/my_agent/trajectories.jsonl \
  --output-dir outputs/my_agent/differentiable_automaton \
  --fit-split train \
  --cal-split train \
  --val-split val \
  --selection-metric direct-soft \
  --encoder-type tfidf \
  --representation-mode reduced-dense \
  --step-view-frontend inferred \
  --epochs 1 \
  --device cpu
```

For a locked OOD report, keep `test` untouched until the final evaluation:

```bash
python scripts/evaluate_differentiable_automaton.py \
  --dataset data/my_agent/trajectories.jsonl \
  --checkpoint outputs/my_agent/differentiable_automaton/best_checkpoint.pt \
  --eval-split test \
  --output outputs/my_agent/differentiable_automaton_test.json
```

Before reporting results, check that train/validation/test splits are non-empty,
that both successes and failures appear in the train and validation data, and
that no future outcome information is included in per-step fields.

## Reproducing Benchmark Experiments

The real benchmark traces are not bundled in this repository. To download the
public upstream sources, rebuild `data/interim/<dataset>`, verify the expected
artifacts, and print the full reproduction plan without training, run:

```bash
python scripts/bootstrap_public_data.py --families all --after-prepare dry-run
```

The bootstrap script downloads/stages:

- WebArena execution archives from the official Google Drive links;
- tau2 completed run-result JSON files from
  `sierra-research/tau2-bench/data/tau2/results/final`;
- TerminalBench trajectories from Hugging Face;
- SkillsBench trajectories from Hugging Face.

Use `--skip-download` to rebuild from an existing `data/external/` mirror, and
`--strict-verify` when you want checksum mismatches to fail the command instead
of being reported as warnings.

To launch the full benchmark reproduction after the same data preparation step:

```bash
python scripts/bootstrap_public_data.py \
  --families all \
  --after-prepare all \
  --device cuda \
  --strict-verify
```

This runs dataset preparation, artifact verification, training, locked test
evaluation, and summary generation. It requires network access for the first
download and a CUDA-capable machine for the main training runs. For a smaller
run, pass a subset such as `--families tau2` or `--families webarena`.

Equivalent manual steps are:

```bash
python scripts/reproduce_main_experiments.py --stage prepare
python scripts/verify_dataset_artifacts.py
python scripts/reproduce_main_experiments.py --stage train --device cuda
python scripts/reproduce_main_experiments.py --stage eval --device cuda
python scripts/reproduce_main_experiments.py --stage summarize
```

To inspect the planned commands without running them:

```bash
python scripts/reproduce_main_experiments.py \
  --stage all \
  --families webarena tau2 terminalbench skillsbench \
  --dry-run
```

For a cheap driver check:

```bash
python scripts/reproduce_main_experiments.py \
  --families webarena \
  --seeds 1 \
  --epochs 1 \
  --device cpu \
  --dry-run
```

The main experiment manifest is
[configs/main_experiments.json](configs/main_experiments.json).

## Repository Layout

```text
src/monitor_symbolization/        Python package source
scripts/                          import, preprocessing, training, evaluation
configs/main_experiments.json     benchmark experiment manifest
scripts/bootstrap_public_data.py   one-command public data reconstruction
data/toy/trajectories.jsonl       self-contained toy dataset
tests/                            public regression tests
DATASETS.md                       real-data reconstruction notes
REPRODUCIBILITY.md                reproduction contract and caveats
```

## Tests

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

## Outputs

Training writes checkpoints and metadata under the requested output directory,
including:

- `best_checkpoint.pt`
- `last_checkpoint.pt`
- `input_dataset_summary.json`
- `protocol_split_summary.json`
- `train_result.json`

Evaluation writes a JSON metrics file to the path passed with `--output`.

## Citation

If you use this repository in academic work, please cite:

```bibtex
@article{huang2026prefixguard,
  title={PrefixGuard: From LLM-Agent Traces to Online Failure-Warning Monitors},
  author={Huang, Xinmiao and Hu, Jinwei and Roy, Rajarshi and Wu, Changshun and Dong, Yi and Huang, Xiaowei},
  journal={arXiv preprint arXiv:2605.06455},
  year={2026},
  doi={10.48550/arXiv.2605.06455},
  url={https://arxiv.org/abs/2605.06455}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
