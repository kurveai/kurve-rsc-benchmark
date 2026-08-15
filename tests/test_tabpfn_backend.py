from __future__ import annotations

import numpy as np
import pandas as pd

from kurve_rsc.feature_pipeline import (
    fit_tabpfn_classifier,
    fit_tabpfn_regressor,
)
from scripts import run_all, run_task


class FakeTabPFNRegressor:
    def __init__(self, *, random_state: int, ignore_pretraining_limits: bool):
        self.random_state = random_state
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.target_mean = 0.0

    def fit(self, inputs: pd.DataFrame, target: pd.Series):
        assert inputs.dtypes.eq("float32").all()
        assert inputs["feature"].isna().sum() == 1
        self.target_mean = float(target.mean())
        return self

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        return np.full(len(inputs), self.target_mean)


class FakeTabPFNClassifier:
    def __init__(
        self,
        *,
        random_state: int,
        ignore_pretraining_limits: bool,
        categorical_features_indices: list[int],
    ):
        self.random_state = random_state
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.categorical_features_indices = categorical_features_indices
        self.last_inputs = pd.DataFrame()

    def fit(self, inputs: pd.DataFrame, target: pd.Series):
        assert inputs.dtypes.eq("float32").all()
        assert target.tolist() == [0, 0, 1, 1]
        return self

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        self.last_inputs = inputs
        positive = np.clip(inputs["numeric"].fillna(0).to_numpy() / 3, 0, 1)
        return np.column_stack([1 - positive, positive])


def test_tabpfn_regressor_materializes_batches_and_scores_validation_mae():
    batches = [
        pd.DataFrame(
            {
                "feature": [1.0, np.inf],
                "target": [1.0, 3.0],
            }
        )
    ]
    model, mae, predictions = fit_tabpfn_regressor(
        lambda: iter(batches),
        ["feature"],
        "target",
        pd.DataFrame({"feature": [2.0, 4.0]}),
        pd.Series([2.0, 4.0]),
        regressor_factory=FakeTabPFNRegressor,
    )

    assert model.random_state == 42
    assert model.ignore_pretraining_limits is True
    assert mae == 1.0
    assert predictions.tolist() == [2.0, 2.0]


def test_tabpfn_classifier_encodes_categories_and_scores_validation_auc():
    batches = [
        pd.DataFrame(
            {
                "numeric": [0.0, 1.0, 2.0, 3.0],
                "category": ["a", "a", "b", "b"],
                "target": [0, 0, 1, 1],
            }
        )
    ]
    model, auc, predictions = fit_tabpfn_classifier(
        lambda: iter(batches),
        ["numeric", "category"],
        "target",
        pd.DataFrame(
            {
                "numeric": [0.5, 2.5],
                "category": ["a", "unknown"],
            }
        ),
        pd.Series([0, 1]),
        classifier_factory=FakeTabPFNClassifier,
    )

    assert model.random_state == 42
    assert model.ignore_pretraining_limits is True
    assert model.categorical_features_indices == [1]
    assert model.estimator.last_inputs["category"].isna().tolist() == [False, True]
    assert auc == 1.0
    assert predictions[0] < predictions[1]


def test_run_task_cli_accepts_tabpfn_for_any_task_without_single_period():
    args = run_task.parse_args(
        [
            "relbench_event_user_ignore.py",
            "--tabpfn",
        ]
    )

    assert args.single_train_period is False
    assert args.tabpfn is True


def test_run_task_cli_propagates_tabpfn_backend(monkeypatch):
    invoked: dict[str, object] = {}

    def fake_run_path(path: str, *, run_name: str):
        invoked.update(path=path, run_name=run_name)

    monkeypatch.setattr(run_task.runpy, "run_path", fake_run_path)
    monkeypatch.setenv(run_task.MODEL_BACKEND_ENV, "catboost")
    monkeypatch.setenv(run_task.SINGLE_TRAIN_PERIOD_ENV, "0")
    monkeypatch.setenv("RELBench_TRAINING_FRAME_WORKERS", "1")
    monkeypatch.setattr(
        run_task.sys,
        "argv",
        [
            "run_task.py",
            "relbench_event_user_ignore.py",
            "--tabpfn",
        ],
    )

    assert run_task.main() == 0
    assert invoked["run_name"] == "__main__"
    assert run_task.os.environ[run_task.MODEL_BACKEND_ENV] == "tabpfn"


def test_run_all_cli_propagates_tabpfn_backend(monkeypatch, tmp_path):
    invoked: dict[str, object] = {}

    class FakeProcess:
        stdout = iter(())
        returncode = 0

        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        invoked.update(command=command, **kwargs)
        return FakeProcess()

    monkeypatch.setattr(run_all.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_all, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_all, "TASK_DIR", tmp_path / "kurve_rsc")
    args = run_all.parse_args(["--tabpfn"])

    result = run_all.run_task(
        "relbench_event_user_ignore.py",
        args,
        tmp_path / "task.log",
    )

    assert result["status"] == "passed"
    assert invoked["env"][run_all.MODEL_BACKEND_ENV] == "tabpfn"
