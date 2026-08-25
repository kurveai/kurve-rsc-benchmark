#!/usr/bin/env python
"""Generic GraphReduce implementation of rel-trial/study-adverse."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

try:
    from .trial_builder import run_generic_rel_trial_task
    from .relbench_feature_policy import configure_task_cli
except ImportError:  # Direct script execution.
    from relbench_trial_task_utils import run_generic_rel_trial_task
    from relbench_feature_policy import configure_task_cli


def run_rel_trial_study_adverse(
    data_dir: Path | None = None,
) -> tuple[
    object,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float] | None,
    dict[str, float] | None,
    int,
    list[str],
    str,
]:
    return run_generic_rel_trial_task("study-adverse", data_dir=data_dir)


def main() -> None:
    train, validation, test, val_metrics, test_metrics, count, files, target = (
        run_rel_trial_study_adverse()
    )
    print("implementation:", "generic", flush=True)
    print("materialized_files:", files, flush=True)
    print("task:", "study-adverse", flush=True)
    print("target:", target, flush=True)
    print("train_rows:", len(train), flush=True)
    print("validation_rows:", len(validation), flush=True)
    print("test_rows:", len(test), flush=True)
    print("train_timestamps:", train.column_nunique("timestamp"), flush=True)
    print("validation_timestamps:", validation["timestamp"].nunique(), flush=True)
    print("test_timestamps:", test["timestamp"].nunique(), flush=True)
    print("feature_count:", count, flush=True)
    print(
        "validation_nmae:",
        val_metrics["nmae"] if val_metrics is not None else "skipped",
        flush=True,
    )
    print(
        "test_nmae:",
        test_metrics["nmae"] if test_metrics is not None else "skipped",
        flush=True,
    )
    print(
        "validation_metrics:",
        val_metrics if val_metrics is not None else "skipped",
        flush=True,
    )
    print(
        "test_metrics:",
        test_metrics if test_metrics is not None else "skipped",
        flush=True,
    )


if __name__ == "__main__":
    configure_task_cli(description=__doc__)
    main()
