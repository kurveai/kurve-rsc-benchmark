#!/usr/bin/env python
"""RelBench rel-trial: site success example using official task tables."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from relbench_trial_task_utils import build_site_features, run_rel_trial_regression_task
from relbench_feature_manifest import feature_manifest_enabled

MODEL_BACKEND_ENV = "KURVE_RSC_MODEL_BACKEND"


def run_rel_trial_site_success(
    data_dir: Path | None = None,
    use_tabpfn: bool | None = None,
    *,
    use_feature_manifest: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    model_backend = os.environ.get(MODEL_BACKEND_ENV, "catboost")
    if use_tabpfn is not None:
        model_backend = "tabpfn" if use_tabpfn else "catboost"
    return run_rel_trial_regression_task(
        task_name="site-success",
        feature_builder=build_site_features,
        feature_entity_col="fac_facility_id",
        data_dir=data_dir,
        max_train_frames=5,
        model_backend=model_backend,
        use_feature_manifest=use_feature_manifest,
        bounded_probability_target=True,
    )


def main() -> None:
    use_feature_manifest = feature_manifest_enabled("rel-trial/site-success")
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = (
        run_rel_trial_site_success(use_feature_manifest=use_feature_manifest)
    )
    print("feature_manifest_enabled:", use_feature_manifest, flush=True)
    print("materialized_files:", materialized, flush=True)
    print("task:", "site-success", flush=True)
    print("target:", target, flush=True)
    print("train_rows:", len(df_train), flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("train_timestamps:", df_train.column_nunique("timestamp"), flush=True)
    print("validation_timestamps:", df_val["timestamp"].nunique(), flush=True)
    print("test_timestamps:", df_test["timestamp"].nunique(), flush=True)
    print("columns:", len(df_train.columns), flush=True)
    print("feature_count:", n_features, flush=True)
    print("validation_nmae:", val_metrics["nmae"] if val_metrics is not None else "skipped", flush=True)
    print("test_nmae:", test_metrics["nmae"] if test_metrics is not None else "skipped", flush=True)
    print("validation_metrics:", val_metrics if val_metrics is not None else "skipped", flush=True)
    print("test_metrics:", test_metrics if test_metrics is not None else "skipped", flush=True)


if __name__ == "__main__":
    main()
