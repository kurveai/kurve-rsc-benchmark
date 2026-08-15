from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kurve_rsc.submission import (
    SUBMISSION_DIR_ENV,
    instrument_task_for_submission,
    prediction_filename,
    write_prediction_table,
)


RUN_ALL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_all.py"
RUN_ALL_SPEC = importlib.util.spec_from_file_location("run_all", RUN_ALL_PATH)
assert RUN_ALL_SPEC is not None and RUN_ALL_SPEC.loader is not None
run_all = importlib.util.module_from_spec(RUN_ALL_SPEC)
RUN_ALL_SPEC.loader.exec_module(run_all)


class FakeTask:
    entity_col = "entity_id"
    time_col = "timestamp"
    target_col = "target"
    entity_table = "entity"

    def __init__(self, official_test: pd.DataFrame, task_type: str) -> None:
        self._official_test = official_test
        self.task_type = SimpleNamespace(value=task_type)
        self.dataset = SimpleNamespace(test_timestamp=pd.Timestamp("2020-02-01"))
        self.evaluation_count = 0

    def get_table(self, split: str, mask_input_cols: bool = True) -> SimpleNamespace:
        assert split == "test"
        return SimpleNamespace(df=self._official_test.copy())

    def evaluate(
        self,
        predictions,
        target_table=None,
        metrics=None,
    ) -> dict[str, float]:
        self.evaluation_count += 1
        return {"metric": float(np.mean(predictions))}


def table(frame: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(df=frame)


def test_prediction_filename_uses_official_double_underscore() -> None:
    assert prediction_filename("rel-f1", "driver-position") == (
        "rel-f1__driver-position.csv"
    )


def test_write_prediction_table_preserves_modeled_frame_alignment(tmp_path) -> None:
    official = pd.DataFrame(
        {
            "entity_id": [1, 2],
            "timestamp": pd.to_datetime(["2020-02-01", "2020-02-01"]),
            "target": [10.0, 20.0],
        }
    )
    modeled = official.iloc[::-1].reset_index(drop=True)
    task = FakeTask(official, "regression")
    path = write_prediction_table(
        task,
        np.array([0.2, 0.1]),
        table(modeled),
        tmp_path / "rel-fake__value.csv",
    )

    prediction_table = pd.read_csv(path)
    assert prediction_table.columns.tolist() == ["entity_id", "timestamp", "target"]
    assert prediction_table["entity_id"].tolist() == [2, 1]
    assert prediction_table["target"].tolist() == [0.2, 0.1]


def test_write_prediction_table_rejects_incomplete_test_coverage(tmp_path) -> None:
    official = pd.DataFrame(
        {
            "entity_id": [1, 2],
            "timestamp": pd.to_datetime(["2020-02-01", "2020-02-01"]),
            "target": [0.0, 0.0],
        }
    )
    task = FakeTask(official, "regression")

    with pytest.raises(ValueError, match="complete official test table"):
        write_prediction_table(
            task,
            np.array([0.1]),
            table(official.iloc[:1]),
            tmp_path / "rel-fake__value.csv",
        )


def test_instrumented_task_exports_only_test_evaluation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(SUBMISSION_DIR_ENV, str(tmp_path))
    official = pd.DataFrame(
        {
            "entity_id": [1],
            "timestamp": pd.to_datetime(["2020-02-01"]),
            "target": [1],
        }
    )
    task = instrument_task_for_submission(
        FakeTask(official, "binary_classification"),
        "rel-fake",
        "is-positive",
    )
    validation = official.assign(timestamp=pd.Timestamp("2020-01-01"))

    task.evaluate(np.array([0.4]), target_table=table(validation))
    assert not (tmp_path / "rel-fake__is-positive.csv").exists()

    task.evaluate(np.array([0.8]), target_table=table(official))
    assert (tmp_path / "rel-fake__is-positive.csv").exists()
    assert task.evaluation_count == 2


def test_classification_submission_rejects_non_probability(tmp_path) -> None:
    official = pd.DataFrame(
        {
            "entity_id": [1],
            "timestamp": pd.to_datetime(["2020-02-01"]),
            "target": [1],
        }
    )

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        write_prediction_table(
            FakeTask(official, "binary_classification"),
            np.array([1.1]),
            table(official),
            tmp_path / "rel-fake__is-positive.csv",
        )


def test_submission_families_match_official_task_counts() -> None:
    classification = run_all.expected_submission_filenames("classification")
    regression = run_all.expected_submission_filenames("regression")

    assert len(classification) == 12
    assert len(regression) == 9
    assert "rel-stack__user-badge.csv" in classification
    assert "rel-f1__driver-position.csv" in regression


def test_submission_mode_rejects_partial_task_selection(tmp_path) -> None:
    args = run_all.parse_args(
        [
            "--task-type",
            "classification",
            "--task",
            "relbench_f1_driver_top3.py",
            "--submission-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(SystemExit, match="requires every task"):
        run_all.submission_path(args)
