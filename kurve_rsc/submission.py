"""RelBench leaderboard prediction-table export helpers.

The benchmark currently uses the RelBench 2.x task API, while the official
leaderboard tooling is distributed with the newer ``relbench-hf`` branch.
This module writes the same entity-task CSV schema without changing the
benchmark's data dependency.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


SUBMISSION_DIR_ENV = "KURVE_RSC_SUBMISSION_DIR"
_INSTRUMENTED_ATTRIBUTE = "_kurve_rsc_submission_instrumented"


def submission_dir() -> Path | None:
    """Return the configured prediction-table directory, if export is enabled."""

    raw_value = os.environ.get(SUBMISSION_DIR_ENV, "").strip()
    return Path(raw_value) if raw_value else None


def prediction_filename(dataset_name: str, task_name: str) -> str:
    """Return the filename required by the official RelBench validator."""

    if not dataset_name or not task_name:
        raise ValueError("dataset_name and task_name must be non-empty")
    if any(
        separator in dataset_name or separator in task_name
        for separator in ("/", "\\")
    ):
        raise ValueError("dataset_name and task_name cannot contain path separators")
    return f"{dataset_name}__{task_name}.csv"


def _key_tuples(frame: pd.DataFrame, key_columns: list[str]) -> set[tuple[str, ...]]:
    values: list[list[str]] = []
    for column in key_columns:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            series = pd.to_datetime(series)
        values.append(series.astype(str).tolist())
    return set(zip(*values))


def _validated_test_keys(task: Any, target_frame: pd.DataFrame) -> pd.DataFrame:
    key_columns = [task.entity_col, task.time_col]
    missing_columns = [column for column in key_columns if column not in target_frame]
    if missing_columns:
        raise ValueError(f"prediction frame is missing key columns: {missing_columns}")

    official_test = task.get_table("test", mask_input_cols=True).df
    normalized_keys = target_frame[key_columns].copy()
    for column in key_columns:
        reference_dtype = official_test[column].dtype
        if pd.api.types.is_datetime64_any_dtype(reference_dtype):
            normalized_keys[column] = pd.to_datetime(normalized_keys[column])
        else:
            try:
                normalized_keys[column] = normalized_keys[column].astype(reference_dtype)
            except (TypeError, ValueError):
                pass

    if normalized_keys.duplicated(subset=key_columns).any():
        duplicate_count = int(normalized_keys.duplicated(subset=key_columns).sum())
        raise ValueError(
            f"prediction frame has {duplicate_count} duplicate test key row(s)"
        )
    if official_test.duplicated(subset=key_columns).any():
        raise ValueError("official RelBench test table contains duplicate keys")
    expected_keys = _key_tuples(official_test, key_columns)
    prediction_keys = _key_tuples(normalized_keys, key_columns)
    missing = expected_keys - prediction_keys
    extra = prediction_keys - expected_keys
    if missing or extra or len(target_frame) != len(official_test):
        raise ValueError(
            "prediction keys do not cover the complete official test table: "
            f"{len(missing)} missing, {len(extra)} extra "
            f"(expected {len(official_test)} rows, got {len(target_frame)})"
        )
    return normalized_keys


def write_prediction_table(
    task: Any,
    predictions: Any,
    target_table: Any,
    path: Path,
) -> Path:
    """Write one official-format entity-task prediction CSV.

    ``predictions`` are aligned to ``target_table`` rather than re-assumed to
    follow a separately loaded test table. This is important because the SQL
    feature joins are free to return the official test keys in another order.
    """

    target_frame = target_table.df
    values = np.asarray(predictions).reshape(-1)
    if len(values) != len(target_frame):
        raise ValueError(
            f"pred has {len(values)} values but the target table has "
            f"{len(target_frame)} rows"
        )
    if not np.issubdtype(values.dtype, np.number):
        try:
            values = values.astype("float64")
        except (TypeError, ValueError) as exc:
            raise ValueError("predictions must be numeric") from exc
    numeric_values = values.astype("float64", copy=False)
    if not np.isfinite(numeric_values).all():
        raise ValueError("predictions contain non-finite values")

    task_type = getattr(task.task_type, "value", task.task_type)
    if task_type == "binary_classification" and (
        numeric_values.min(initial=0.0) < 0.0
        or numeric_values.max(initial=1.0) > 1.0
    ):
        raise ValueError("classification predictions must lie in [0, 1]")
    if task_type not in {"binary_classification", "regression"}:
        raise ValueError(f"unsupported submission task type: {task_type}")

    output = _validated_test_keys(task, target_frame)
    output[task.target_col] = numeric_values
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        output.to_csv(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def _is_test_evaluation(task: Any, target_table: Any | None) -> bool:
    if target_table is None:
        return True
    frame = target_table.df
    if frame.empty or task.time_col not in frame:
        return False
    timestamps = pd.to_datetime(frame[task.time_col], errors="coerce")
    if timestamps.isna().any():
        return False
    return bool(timestamps.min() >= pd.Timestamp(task.dataset.test_timestamp))


def instrument_task_for_submission(
    task: Any,
    dataset_name: str,
    task_name: str,
) -> Any:
    """Export successful test predictions when submission mode is enabled."""

    output_dir = submission_dir()
    if output_dir is None or getattr(task, _INSTRUMENTED_ATTRIBUTE, False):
        return task

    original_evaluate: Callable[..., dict[str, float]] = task.evaluate

    def evaluate_with_export(
        predictions: Any,
        target_table: Any | None = None,
        metrics: Any | None = None,
    ) -> dict[str, float]:
        result = original_evaluate(
            predictions,
            target_table=target_table,
            metrics=metrics,
        )
        if _is_test_evaluation(task, target_table):
            table = (
                task.get_table("test", mask_input_cols=False)
                if target_table is None
                else target_table
            )
            path = write_prediction_table(
                task,
                predictions,
                table,
                output_dir / prediction_filename(dataset_name, task_name),
            )
            print(f"submission_prediction: {path.resolve()}", flush=True)
        return result

    task.evaluate = evaluate_with_export
    setattr(task, _INSTRUMENTED_ATTRIBUTE, True)
    return task
