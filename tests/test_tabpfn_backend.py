from __future__ import annotations

import numpy as np
import pandas as pd

from kurve_rsc.feature_pipeline import fit_tabpfn_regressor
from scripts import run_task


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


def test_run_task_cli_accepts_tabpfn_for_site_success_command():
    args = run_task.parse_args(
        [
            "relbench_trial_site_success.py",
            "--single-train-period",
            "--tabpfn",
        ]
    )

    assert args.single_train_period is True
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
            "relbench_trial_site_success.py",
            "--single-train-period",
            "--tabpfn",
        ],
    )

    assert run_task.main() == 0
    assert invoked["run_name"] == "__main__"
    assert run_task.os.environ[run_task.MODEL_BACKEND_ENV] == "tabpfn"
