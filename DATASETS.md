# Dataset Reconstruction Notes

This anonymous review branch intentionally does not redistribute real benchmark
traces. The excluded datasets are large, externally licensed, or locally
mirrored research artifacts. This file records the reconstruction contract for
the real datasets used by the public reproduction branch.

The public branch includes only:

```text
data/toy/trajectories.jsonl
sha256: adc23ed75d19c037c8ca7149ee861a4f49ff2457cfc47846991855f09f488f41
```

Use the commands in `README.md` and `REPRODUCIBILITY.md` to exercise the public
toy-data path.

## General Layout

The public reproduction tree uses this data convention:

```text
data/external/<dataset>/...   # downloaded or locally staged raw data
data/interim/<dataset>/...    # imported JSONL files and split manifests
```

The anonymous branch preserves the package code, toy data, core import scripts,
canonical preprocessing scripts, and the real-data experiment manifest. The raw
real datasets themselves are not redistributed.

Download tooling is part of the project dependencies:

```bash
uv sync --locked
python -m gdown --help
hf download --help
```

Direct download verification was last checked on 2026-05-06:

- WebArena execution trace folders are linked from the official WebArena
  resources file and can be downloaded with `gdown --folder`.
- `HuggingFaceH4/tau2-bench-data` can be downloaded from Hugging Face at
  revision `60e37c7a19672769a6034c45a5c8b36e7cd3768b`, but it contains
  benchmark/domain data only. It does not contain the run-result trajectory
  JSON files consumed by this repository's tau2 importer.
- `benchflow/skillsbench-trajectories-apr2026` can be downloaded from Hugging
  Face at revision `841dfc7d248bb0b1cd35fa65bb993a1eba0d1d2f`. Its public
  ACP trajectory format is supported by this repository's SkillsBench importer.

To verify a reconstructed artifact:

```bash
python scripts/verify_dataset_artifacts.py
```

The main monitor experiment manifest is `configs/main_experiments.json`. After
raw data are staged, the end-to-end public driver is:

```bash
python scripts/reproduce_main_experiments.py --stage prepare
python scripts/verify_dataset_artifacts.py
python scripts/reproduce_main_experiments.py --stage train --device cuda
python scripts/reproduce_main_experiments.py --stage eval --device cuda
python scripts/reproduce_main_experiments.py --stage summarize
```

## WebArena

Source:

- Project page: `https://webarena.dev/`
- Code and benchmark metadata: `https://github.com/web-arena-x/webarena`
- Official resource manifest:
  `https://github.com/web-arena-x/webarena/blob/main/resources/README.md`
- Execution trace releases used by the full pipeline are the WebArena v1/v2
  trace archives staged under `data/external/webarena/`.

Download and staging:

The official WebArena resources file links the v1 and v2 execution trace
Google Drive folders. Download them directly with `gdown`:

```bash
mkdir -p data/external/webarena

curl -L \
  https://raw.githubusercontent.com/web-arena-x/webarena/dce04686a56253aefba7b18a4fa0937cf1dc987b/config_files/test.raw.json \
  -o data/external/webarena/test.raw.json

python -m gdown --folder --remaining-ok --continue \
  "https://drive.google.com/drive/folders/18Oww0fAgwhuSjSzxUNgzBUlC6M9IZZB2?usp=sharing" \
  -O data/external/webarena/execution_v1/072023_release_v1/

python -m gdown --folder --remaining-ok --continue \
  "https://drive.google.com/drive/folders/1H4wkzDkY2ufiC63DISMXllri0j-ipWcs?usp=sharing" \
  -O data/external/webarena/execution_v2/112023_release_v2/
```

The importer expects the downloaded zip filenames below and does not contact
Google Drive itself.

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

Preprocessing command:

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
- Direct-download benchmark/domain data:
  `https://huggingface.co/datasets/HuggingFaceH4/tau2-bench-data`
- The reproduction tree stages tau2-bench run result JSON files under
  `data/external/tau2_bench/results/final/`.

Download and staging:

The official benchmark/domain data can be downloaded directly:

```bash
hf download HuggingFaceH4/tau2-bench-data \
  --repo-type dataset \
  --revision 60e37c7a19672769a6034c45a5c8b36e7cd3768b \
  --local-dir data/external/tau2_bench/tau2_data
```

This is not sufficient for the main monitor experiment. The tau2 importer
consumes completed tau2 evaluation outputs, not only benchmark definitions. The
official tau2 documentation states that text evaluations write monolithic
`results.json` files under `data/simulations/<run_name>/`. Generate or obtain
those run-result JSON files for the evaluated agents, then place or symlink
them under the path below:

```bash
mkdir -p data/external/tau2_bench/results/final
cp /path/to/tau2-bench/data/simulations/<run_name>/results.json \
  data/external/tau2_bench/results/final/<agent>_<domain>_<policy>_<user>.json
```

No public static download was verified for the exact
`data/external/tau2_bench/results/final/*.json` bundle that produces the
checksum below. To make the tau2 main experiment directly reproducible from
downloads alone, this bundle must be published as an anonymous artifact, or the
anonymous package must include the exact tau2 run commands, model/API settings,
seeds, and generated `results.json` files. Substituting third-party tau2
trajectory datasets changes the scientific input and must be treated as a new
dataset/checksum.

Expected raw inputs:

```text
data/external/tau2_bench/results/final/*.json
```

Preprocessing command:

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

Download/import command:

```bash
python scripts/import_terminalbench_trajectories.py \
  --repo-id yoonholee/terminalbench-trajectories \
  --input-root data/external/terminalbench/terminalbench-trajectories \
  --output-jsonl data/interim/terminalbench/terminalbench_trajectories_full.jsonl \
  --output-summary data/interim/terminalbench/terminalbench_trajectories_full_summary.json
```

Source-raw manifest command:

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

- Public trajectory snapshot:
  `https://huggingface.co/datasets/benchflow/skillsbench-trajectories-apr2026`
- Code and task definitions: `https://github.com/benchflow-ai/skillsbench`
- The full experiments used a SkillsBench trajectory corpus staged under
  `data/external/skillsbench/skillsbench-trajectories/`.
- Trial directories contain `result.json`, `config.json`, and trace payloads.
  The public Hugging Face snapshot uses `trajectory/acp_trajectory.jsonl`, which
  is supported by `scripts/import_skillsbench_traces.py`.

Download and staging:

```bash
hf download benchflow/skillsbench-trajectories-apr2026 \
  --repo-type dataset \
  --revision 841dfc7d248bb0b1cd35fa65bb993a1eba0d1d2f \
  --local-dir data/external/skillsbench/skillsbench-trajectories \
  --max-workers 2
```

The dataset contains many small files; using a low `--max-workers` value avoids
Hugging Face HEAD-request rate limiting observed during full dry-run checks.
The public dataset card reports the layout below and a total size of about
181 MB. A real downloaded sample with `trajectory/acp_trajectory.jsonl` was
successfully imported by this repository's importer.

Expected raw inputs:

```text
data/external/skillsbench/skillsbench-trajectories/**/result.json
data/external/skillsbench/skillsbench-trajectories/**/config.json
data/external/skillsbench/skillsbench-trajectories/**/trajectory/acp_trajectory.jsonl
```

Import command:

```bash
python scripts/import_skillsbench_traces.py \
  --input-root data/external/skillsbench/skillsbench-trajectories \
  --output-jsonl data/interim/skillsbench/full_repo_main_traces.jsonl \
  --output-summary data/interim/skillsbench/full_repo_main_traces_summary.json
```

Canonical split, clean-monitor, and source-raw manifest commands:

```bash
python scripts/create_task_grouped_split.py \
  --input data/interim/skillsbench/full_repo_main_traces.jsonl \
  --output data/interim/skillsbench/full_repo_main_traces_split.jsonl \
  --summary-output data/interim/skillsbench/full_repo_main_traces_split_summary.json \
  --protocol-mode outer-train-val-test \
  --seed 13 \
  --val-ratio 0.1 \
  --test-ratio 0.1

python scripts/build_skillsbench_clean_monitor.py \
  --input-jsonl data/interim/skillsbench/full_repo_main_traces_split.jsonl \
  --output-jsonl data/interim/skillsbench/full_repo_main_traces_split_clean_monitor.jsonl \
  --summary-json data/interim/skillsbench/full_repo_main_traces_split_clean_monitor_summary.json

python scripts/prepare_source_raw_baseline_datasets.py --only skillsbench
```

The source-raw command consumes the canonical SkillsBench split JSONL and writes
a manifest that points back to raw trial directories. If the split JSONL is
missing but `full_repo_main_traces.jsonl` exists, the prepare script rebuilds
the split using the command above.

Expected artifact:

```text
data/interim/skillsbench/source_raw/full_repo_main_traces_source_raw_split_manifest.jsonl
rows: 10951
sha256: a08b8dccded2fb65f0bb822fad0c24c360fdc0bc60358f3367a66be1cfdd5547
```

The checksum above is for the full corpus used by the authors' experiments. If
the public Hugging Face snapshot is used and this checksum does not match,
record the Hugging Face revision, row count, and new checksum before comparing
results; do not silently treat a different SkillsBench snapshot as the same
main-experiment input.

## Explicitly Excluded Adapters

The anonymous review package does not require the extra M6/native benchmark
adapter artifacts or task-only adapter datasets. They are excluded from the
public reproduction contract and should not be treated as required reviewer
inputs for this branch.

The package also does not reproduce API-based LLM judge baselines. Those require
external model APIs and cost-limited sampled prefix sets, so they are outside
the anonymous code/data reproduction contract.

## Integrity Notes

- The checksums above are SHA-256 values of the canonical artifacts used by the
  authors' full research tree and expected from this public reconstruction
  pipeline.
- If a data provider updates an upstream archive or parquet shard, checksums can
  differ even when the preprocessing command is unchanged. In that case, record
  the provider version, file list, and new SHA-256 before comparing results.
- Dataset licenses and redistribution terms remain governed by the upstream
  providers. This branch does not grant additional redistribution rights.
