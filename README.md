# Kurve-RSC Benchmark

This repository contains the Kurve-RSC reproduction harness for the RelBench
v1 benchmark tasks. The execution flow is:

```text
RelBench -> Kurve-RSC adapter -> GraphReduce SQL feature frames
         -> incremental CatBoost -> RelBench evaluator
```

The task implementations in `kurve_rsc/` are mirrored from the working
GraphReduce RelBench examples. GraphReduce remains the general-purpose
relational compute engine; this repository owns the benchmark adapters,
task runners, metrics, reports, and reproducibility checks.

## Environment

GraphReduce is installed from the published `graphreduce==1.9.17` package.
No neighboring GraphReduce source checkout is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run Tasks

Run one task by its mirrored script name:

```bash
python scripts/run_task.py relbench_event_user_ignore.py
```

Run all v1 classification tasks:

```bash
python scripts/run_all.py --task-type classification
```

Run all v1 regression tasks:

```bash
python scripts/run_all.py --task-type regression
```

Run the complete v1 benchmark:

```bash
python scripts/run_all.py --task-type v1
```

## Leaderboard submissions

Add `--submission-dir` to generate one official-format prediction CSV for
every task in either the classification or regression leaderboard family:

```bash
python scripts/run_all.py \
  --task-type classification \
  --submission-dir results/submissions/kurve-rsc-classification
```

Use `--task-type regression` for the regression family. Submission mode
requires the complete family, so it cannot be combined with `--task` or
`--match`. Each file is named `<dataset>__<task>.csv`; classification values
are probabilities and regression values remain on the original target scale.
The run fails if a freshly generated file does not cover every official test
key. Normal benchmark runs do not write prediction tables.

Add the required `metadata.yaml` to that directory, following the
[RelBench submission documentation](https://github.com/stanford-star/relbench/blob/relbench-hf/README.md#submitting-to-the-leaderboard),
then validate and create the clean upload zip with the official tooling:

```yaml
name: Kurve-RSC
type: fine-tuned
email: maintainer@example.com
# url and note are optional
```

```bash
python -m relbench.leaderboard \
  results/submissions/kurve-rsc-classification \
  --package
```

This benchmark stays on `relbench==2.1.1` because its task adapters use the
legacy dataset API. The leaderboard module belongs to the newer
`relbench-hf`/3.x code line, so run the validation command from a separate
environment that provides `relbench.leaderboard`; do not replace the benchmark
environment's RelBench dependency.

Training cutoff frames are sequential by default. Bound concurrency explicitly
on a large machine:

```bash
python scripts/run_all.py --task relbench_avito_user_clicks.py \
  --training-frame-workers 50
```

Use `--training-frame-workers all` to submit one worker per training frame.
Each worker receives its own DuckDB cursor and source tables are materialized
when needed to avoid temporary-table collisions.

Long-running `rel-stack` tasks emit `feature_frame_progress` lines for every
cutoff, including elapsed time when a frame finishes. The three Stack runners
use 15 training cutoffs instead of all 46: 14 are selected reproducibly from
chronological strata and the most recent training cutoff is always retained.
Validation and test schedules are not sampled.

`rel-event/user-ignore` plans its GraphReduce operations on the first training
cutoff and replays that frozen plan for all later training, validation, and test
graphs. Before fitting, it selects only feature columns present in every
training, validation, and test cutoff, so cutoff-specific columns cannot enter
the model.

CatBoost continues training across cutoff frames by default to bound peak
memory. To materialize all selected training frames and fit each candidate
configuration jointly with validation-based early stopping, pass:

```bash
python scripts/run_task.py relbench_event_user_ignore.py \
  --train-all-at-once
```

The flag is also supported by `scripts/run_all.py` and applies to every task
that uses the incremental CatBoost helpers. Joint fitting can consume much more
memory on the largest tasks. Use `--no-train-all-at-once` or omit the flag to
retain incremental training.

Apart from the bounded Stack schedule above, tasks retain their complete
historical training schedule by default. To use only one training label period
for a task:

```bash
python scripts/run_task.py relbench_event_user_ignore.py \
  --single-train-period
```

The option is also available on the top-level runner and applies independently
to each selected task:

```bash
python scripts/run_all.py --task-type v1 --single-train-period
```

Each task's singular cutoff is computed as
`dataset.val_timestamp - task.timedelta`, so its own label horizon determines
the date. This mirrors the training schedule in RelBench's
[`BaseTask._get_table`](https://github.com/snap-stanford/relbench/blob/main/relbench/base/task_base.py#L102-L110).
The selected date is emitted as `single_train_cut_date` in task logs and run
reports. Omit the flag (or use `--no-single-train-period`) to retain all
historical training periods.

## TabPFN

CatBoost remains the default. Pass `--tabpfn` to use `TabPFNClassifier` for a
classification task or `TabPFNRegressor` for a regression task:

```bash
python scripts/run_task.py relbench_event_user_ignore.py \
  --tabpfn
```

The option is also available on the top-level runner and applies to every
selected task:

```bash
python scripts/run_all.py --task-type all \
  --single-train-period \
  --tabpfn
```

`--single-train-period` is not required, but it is recommended for TabPFN on
large tasks because TabPFN materializes the selected training frames into one
in-memory matrix. Omitting it retains each task's normal multi-cutoff training
schedule.

This uses `TabPFNClassifier` or `TabPFNRegressor` from `tabpfn==8.1.0`. On
first use, TabPFN downloads its model checkpoint and may require accepting the
model license. For headless runs, set `TABPFN_TOKEN` as described in the
[official TabPFN documentation](https://github.com/PriorLabs/TabPFN#basic-usage).

Reports, logs, and machine-readable results are written to `results/`.
Individual frames spill to Parquet during multi-cutoff feature construction.
CatBoost trains incrementally from those frames; TabPFN combines them before
fitting.

## Metrics

Classification uses AUROC on the official RelBench test split. Regression
reports NMAE, defined as:

```text
NMAE = MAE / standard deviation of the training targets
```

The denominator is computed from all training frames and validation/test
frames are never used to fit it.

## Scope

The initial public harness includes only the RelBench v1 benchmark tasks:
Amazon, Avito, Event, F1, H&M, Stack, and Trial. RelBench v2 tasks remain out
of scope until their adapters are added deliberately.
