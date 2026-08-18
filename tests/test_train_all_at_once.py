from __future__ import annotations

import pandas as pd

from kurve_rsc import feature_pipeline
from scripts import run_all, run_task


def test_joint_classifier_fit_materializes_every_training_frame(monkeypatch) -> None:
    captured: dict[str, object] = {}
    expected_model = object()
    expected_config = {"name": "joint"}

    def fake_fit(
        train_inputs,
        train_target,
        val_inputs,
        val_target,
        **kwargs,
    ):
        captured["train_inputs"] = train_inputs
        captured["train_target"] = train_target
        captured["kwargs"] = kwargs
        return expected_model, expected_config, 0.75

    monkeypatch.setattr(feature_pipeline, "fit_tuned_classifier", fake_fit)
    batches = [
        pd.DataFrame({"feature": [1.0, 2.0], "target": [0, 1]}),
        pd.DataFrame({"feature": [3.0, 4.0], "target": [1, 0]}),
    ]

    model, config, auc = feature_pipeline.fit_tuned_classifier_incremental(
        lambda: iter(batches),
        ["feature"],
        "target",
        pd.DataFrame({"feature": [1.5, 3.5]}),
        pd.Series([0, 1]),
        batch_count=2,
        train_all_at_once=True,
    )

    assert model is expected_model
    assert config == expected_config
    assert auc == 0.75
    assert captured["train_inputs"]["feature"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert captured["train_target"].tolist() == [0, 1, 1, 0]


def test_single_config_classifier_honors_joint_training(monkeypatch) -> None:
    captured: dict[str, object] = {}
    expected_model = object()
    config = {
        "iterations": 20,
        "depth": 4,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
    }

    def fake_fit(train_inputs, train_target, val_inputs, val_target, **kwargs):
        captured["train_inputs"] = train_inputs
        captured["train_target"] = train_target
        captured["kwargs"] = kwargs
        return expected_model, config, 0.8

    monkeypatch.setattr(feature_pipeline, "fit_tuned_classifier", fake_fit)
    batches = [
        pd.DataFrame({"feature": [1.0, 2.0], "target": [0, 1]}),
        pd.DataFrame({"feature": [3.0], "target": [1]}),
    ]

    model, auc = feature_pipeline.fit_incremental_classifier(
        lambda: iter(batches),
        ["feature"],
        "target",
        pd.DataFrame({"feature": [1.5, 3.5]}),
        pd.Series([0, 1]),
        batch_count=2,
        config=config,
        train_all_at_once=True,
    )

    assert model is expected_model
    assert auc == 0.8
    assert captured["train_inputs"]["feature"].tolist() == [1.0, 2.0, 3.0]
    assert captured["train_target"].tolist() == [0, 1, 1]
    assert captured["kwargs"]["configs"] == (config,)


def test_single_config_regressor_honors_joint_training(monkeypatch) -> None:
    captured: dict[str, object] = {}
    expected_model = object()
    config = {
        "iterations": 20,
        "depth": 4,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
    }

    def fake_fit(train_inputs, train_target, val_inputs, val_target, **kwargs):
        captured["train_inputs"] = train_inputs
        captured["train_target"] = train_target
        captured["kwargs"] = kwargs
        return expected_model, config, 1.25

    monkeypatch.setattr(feature_pipeline, "fit_tuned_regressor", fake_fit)
    batches = [
        pd.DataFrame({"feature": [1.0], "target": [2.0]}),
        pd.DataFrame({"feature": [3.0], "target": [4.0]}),
    ]

    model, mae = feature_pipeline.fit_incremental_regressor(
        lambda: iter(batches),
        ["feature"],
        "target",
        pd.DataFrame({"feature": [1.5]}),
        pd.Series([2.5]),
        batch_count=2,
        config=config,
        cat_features=[0],
        train_all_at_once=True,
    )

    assert model is expected_model
    assert mae == 1.25
    assert captured["train_inputs"]["feature"].tolist() == [1.0, 3.0]
    assert captured["train_target"].tolist() == [2.0, 4.0]
    assert captured["kwargs"]["configs"] == (config,)
    assert captured["kwargs"]["cat_features"] == [0]


def test_cross_entropy_fit_materializes_fractional_training_targets(monkeypatch) -> None:
    captured: dict[str, object] = {}
    expected_model = object()
    expected_config = {"name": "probability"}

    def fake_fit(train_inputs, train_target, val_inputs, val_target, **kwargs):
        captured["train_inputs"] = train_inputs
        captured["train_target"] = train_target
        captured["kwargs"] = kwargs
        return expected_model, expected_config, 0.2

    monkeypatch.setattr(
        feature_pipeline,
        "fit_tuned_cross_entropy_model",
        fake_fit,
    )
    batches = [
        pd.DataFrame({"feature": [1.0, 2.0], "target": [0.0, 0.25]}),
        pd.DataFrame({"feature": [3.0, 4.0], "target": [0.75, 1.0]}),
    ]

    model, config, mae = (
        feature_pipeline.fit_tuned_cross_entropy_model_incremental(
            lambda: iter(batches),
            ["feature"],
            "target",
            pd.DataFrame({"feature": [1.5, 3.5]}),
            pd.Series([0.2, 0.8]),
            batch_count=2,
            cat_features=[0],
            train_all_at_once=True,
        )
    )

    assert model is expected_model
    assert config == expected_config
    assert mae == 0.2
    assert captured["train_inputs"]["feature"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert captured["train_target"].tolist() == [0.0, 0.25, 0.75, 1.0]
    assert captured["kwargs"]["cat_features"] == [0]


def test_cli_flags_enable_joint_training() -> None:
    task_args = run_task.parse_args(
        ["relbench_event_user_ignore.py", "--train-all-at-once"]
    )
    all_args = run_all.parse_args(["--train-all-at-once"])

    assert task_args.train_all_at_once is True
    assert all_args.train_all_at_once is True


def test_cli_flags_enable_scoped_feature_manifests() -> None:
    task_args = run_task.parse_args(
        ["relbench_f1_driver_position.py", "--feature-manifest"]
    )
    all_args = run_all.parse_args(["--feature-manifest"])

    assert task_args.feature_manifest is True
    assert all_args.feature_manifest is True
    assert run_all.parse_args(["--no-feature-manifest"]).feature_manifest is False


def test_run_all_disables_feature_manifest_for_f1_driver_top3() -> None:
    assert run_all.feature_manifest_enabled_for_task(
        "relbench_f1_driver_top3.py",
        requested=True,
    ) is False
    assert run_all.feature_manifest_enabled_for_task(
        "relbench_trial_study_outcome.py",
        requested=True,
    ) is True
    assert run_all.feature_manifest_enabled_for_task(
        "relbench_trial_site_success.py",
        requested=True,
    ) is True
