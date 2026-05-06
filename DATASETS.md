# Dataset Reconstruction Notes

This anonymous review branch intentionally does not redistribute real benchmark
traces. The excluded datasets are large, externally licensed, or locally
mirrored research artifacts. This file records the reconstruction contract for
the real datasets used outside the minimal public branch.

The public branch includes only:

```text
data/toy/trajectories.jsonl
sha256: adc23ed75d19c037c8ca7149ee861a4f49ff2457cfc47846991855f09f488f41
```

Use the commands in `README.md` and `REPRODUCIBILITY.md` to exercise the public
toy-data path.

## General Layout

The full research tree uses this data convention:

```text
data/external/<dataset>/...   # downloaded or locally staged raw data
data/interim/<dataset>/...    # imported JSONL files and split manifests
```

The minimal anonymous branch preserves the package code and toy data only. The
preprocessing scripts listed below are from the full research tree and are not
shipped in this stripped branch unless separately restored by the authors.

To verify a reconstructed artifact:

```bash
sha256sum <expected-file>
wc -l <expected-file>
```

## WebArena

Source:

- Project page: `https://webarena.dev/`
- Code and benchmark metadata: `https://github.com/web-arena-x/webarena`
- Execution trace releases used by the full pipeline are the WebArena v1/v2
  trace archives staged under `data/external/webarena/`.

Download and staging:

Download the WebArena execution trajectory archives from the official WebArena
resources linked by the project repository, then place the zip files at the
paths below. The importer expects these filenames and does not download them
automatically.

Expected raw inputs:

```text
data/external/webarena/test.raw.json
data/external/webarena/execution_v1/072023_release_v1/release_v1.0_gpt3.5direct.zip
data/external/webarena/execution_v1/072023_release_v1/release_v1.0_gpt3.5dreasoning.zip
data/external/webarena/execution_v1/072023_release_v1/release_v1.0_gpt4.zip
data/external/webarena/execution_v2/112023_release_v2/v2_919_gpt35_16k_cot.zip
data/external/webarena/execution_v2/112023_release_v2/v2_919_gpt35_16k_cot_na.zip
data/external/webarena/execution_v2/112023_release_v2/v2_919_gpt35_16k_direct.zip
data/external/webarena/execution_v2/112023_release_v2/v2_919_gpt35_16k_direct_na.zip
data/external/webarena/execution_v2/112023_release_v2/v2_919_gpt4_8k_cot.zip
data/external/webarena/execution_v2/112023_release_v2/v2_919_text_bison_001_cot.zip
```

Preprocessing command in the full tree:

```bash
python scripts/prepare_source_raw_baseline_datasets.py --only webarena
```

The script imports each archive with `scripts/import_webarena_execution_traces.py`,
merges the imported JSONL files, and writes a deterministic task-grouped
train/validation/test split with seed `13`, validation ratio `0.1`, and test
ratio `0.1`.

Expected artifact:

```text
data/interim/webarena/source_raw/execution_union_v1_v2_source_raw_labeled_split.jsonl
rows: 4430
sha256: 756ac7d4e9b5797e69bea90e5ffd27ca85ebd1752b5a74f566eb050e4dcf3819
```

## tau2-bench

Source:

- Code and benchmark definitions: `https://github.com/sierra-research/tau2-bench`
- The full tree stages tau2-bench run result JSON files under
  `data/external/tau2_bench/results/final/`.

Download and staging:

Clone or download the official tau2-bench repository for benchmark definitions.
Generate or obtain the run result JSON files for the evaluated agents, then
place them under the path below. The full-tree importer reads local JSON files
and does not contact the upstream repository.

Expected raw inputs:

```text
data/external/tau2_bench/results/final/*.json
```

Preprocessing command in the full tree:

```bash
python scripts/prepare_source_raw_baseline_datasets.py --only tau2
```

The script imports raw result JSON files with
`scripts/import_tau2_bench_results.py` using history window `12`, then writes a
deterministic task-grouped train/validation/test split with seed `1`,
validation ratio `0.15`, and test ratio `0.15`.

Expected artifact:

```text
data/interim/tau2_bench/source_raw/results_final_source_raw_outer_train_val_test.jsonl
rows: 10832
sha256: 002c1a34d290b03c916b354c7b839a8ffefc4c90838f647c711e0689a97c3002
```

## TerminalBench Trajectories

Source:

- Hugging Face dataset: `https://huggingface.co/datasets/yoonholee/terminalbench-trajectories`
- The dataset card identifies the staged files as `data/*.parquet`.

Expected raw inputs:

```text
data/external/terminalbench/terminalbench-trajectories/README.md
data/external/terminalbench/terminalbench-trajectories/data/*.parquet
```

Download command in the full tree:

```bash
python scripts/import_terminalbench_trajectories.py \
  --repo-id yoonholee/terminalbench-trajectories \
  --input-root data/external/terminalbench/terminalbench-trajectories \
  --output-jsonl data/interim/terminalbench/terminalbench_trajectories_full.jsonl \
  --output-summary data/interim/terminalbench/terminalbench_trajectories_full_summary.json
```

Source-raw manifest command in the full tree:

```bash
python scripts/prepare_source_raw_baseline_datasets.py --only terminalbench
```

The source-raw pipeline retains a split manifest instead of committing the full
materialized JSONL. It materializes trajectories from the local parquet mirror
at load time and writes a trajectory-stratified fit/calibration/validation/test
manifest with seed `1` and ratios `0.7/0.1/0.1/0.1`.

Expected artifact:

```text
data/interim/terminalbench/source_raw/terminalbench_trajectories_source_raw_traj_split_manifest.jsonl
rows: 34397
sha256: bdb787c4daff4f76719d509da59865170940e98f027c0fa450c0a6f00aab0058
```

## SkillsBench Traces

Source:

- The full experiments used a locally mirrored SkillsBench trajectory corpus
  staged under `data/external/skillsbench/skillsbench-trajectories/`.
- The local mirror contains trial directories with `result.json`,
  `config.json`, and trace payloads. A clearly redistributable public upstream
  was not verified from the repository metadata, so the raw traces are not
  included in this anonymous branch.

Expected raw inputs:

```text
data/external/skillsbench/skillsbench-trajectories/**/result.json
data/external/skillsbench/skillsbench-trajectories/**/config.json
```

Import command in the full tree:

```bash
python scripts/import_skillsbench_traces.py \
  --input-root data/external/skillsbench/skillsbench-trajectories \
  --output-jsonl data/interim/skillsbench/full_repo_main_traces.jsonl \
  --output-summary data/interim/skillsbench/full_repo_main_traces_summary.json
```

Clean-monitor and source-raw manifest commands in the full tree:

```bash
python scripts/build_skillsbench_clean_monitor.py \
  --input-jsonl data/interim/skillsbench/full_repo_main_traces.jsonl \
  --output-jsonl data/interim/skillsbench/full_repo_main_traces_clean_monitor.jsonl \
  --summary-json data/interim/skillsbench/full_repo_main_traces_clean_monitor_summary.json

python scripts/prepare_source_raw_baseline_datasets.py --only skillsbench
```

The source-raw command consumes the canonical SkillsBench split JSONL in the
full tree and writes a manifest that points back to raw trial directories.

Expected artifact:

```text
data/interim/skillsbench/source_raw/full_repo_main_traces_source_raw_split_manifest.jsonl
rows: 10951
sha256: a08b8dccded2fb65f0bb822fad0c24c360fdc0bc60358f3367a66be1cfdd5547
```

## Explicitly Excluded Adapters

The anonymous review package does not require the extra M6/native benchmark
adapter artifacts or task-only adapter datasets. They are excluded from the
public reproduction contract and should not be treated as required reviewer
inputs for this branch.

## Integrity Notes

- The checksums above are SHA-256 values of the canonical local artifacts used
  by the authors' full research tree.
- If a data provider updates an upstream archive or parquet shard, checksums can
  differ even when the preprocessing command is unchanged. In that case, record
  the provider version, file list, and new SHA-256 before comparing results.
- Dataset licenses and redistribution terms remain governed by the upstream
  providers. This branch does not grant additional redistribution rights.
