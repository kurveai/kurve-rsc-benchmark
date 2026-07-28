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

Training cutoff frames are sequential by default. Bound concurrency explicitly
on a large machine:

```bash
python scripts/run_all.py --task relbench_avito_user_clicks.py \
  --training-frame-workers 50
```

Use `--training-frame-workers all` to submit one worker per training frame.
Each worker receives its own DuckDB cursor and source tables are materialized
when needed to avoid temporary-table collisions.

Reports, logs, and machine-readable results are written to `results/`.
Individual frames spill to Parquet during multi-cutoff training so CatBoost
can train incrementally without retaining every frame in RAM.

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
