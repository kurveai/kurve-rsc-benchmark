from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kurve_rsc"))

from kurve_rsc import relbench_trial_site_success
from kurve_rsc.trial_builder import (
    SITE_SUCCESS_FEATURE_FAMILIES,
    _select_evenly_spaced_timestamps,
    bounded_probability_candidates,
    select_bounded_probability_candidate,
)


def test_site_success_uses_compact_feature_families():
    assert SITE_SUCCESS_FEATURE_FAMILIES == ("base", "semantic", "context")


def test_trial_training_timestamp_sampling_keeps_full_range():
    timestamps = list(pd.date_range("2001-01-01", periods=19, freq="YS"))

    selected = _select_evenly_spaced_timestamps(timestamps, 5)

    assert len(selected) == 5
    assert selected[0] == timestamps[0]
    assert selected[-1] == timestamps[-1]
    assert selected == [timestamps[index] for index in (0, 4, 9, 14, 18)]


def test_trial_training_timestamp_sampling_does_not_oversample_short_ranges():
    timestamps = list(pd.date_range("2020-01-01", periods=3, freq="D"))

    assert _select_evenly_spaced_timestamps(timestamps, 5) == timestamps


def test_bounded_probability_candidates_clip_regression_predictions():
    candidates = bounded_probability_candidates(
        regression_predictions=pd.Series([-0.2, 0.4, 1.2]).to_numpy(),
        probability_predictions=pd.Series([0.1, 0.6, 0.9]).to_numpy(),
    )

    assert candidates["mae_regression"].tolist() == [0.0, 0.4, 1.0]
    assert candidates["cross_entropy_probability"].tolist() == [0.1, 0.6, 0.9]
    assert candidates["equal_weight_blend"].tolist() == [0.05, 0.5, 0.95]


def test_probability_candidate_selection_uses_validation_mae():
    selected_name, predictions, scores = select_bounded_probability_candidate(
        pd.Series([0.0, 1.0, 0.5]),
        regression_predictions=pd.Series([0.4, 0.6, 0.5]).to_numpy(),
        probability_predictions=pd.Series([0.1, 0.9, 0.5]).to_numpy(),
    )

    assert selected_name == "cross_entropy_probability"
    assert predictions.tolist() == [0.1, 0.9, 0.5]
    assert scores["cross_entropy_probability"] < scores["equal_weight_blend"]
    assert scores["equal_weight_blend"] < scores["mae_regression"]


def test_site_success_enables_bounded_probability_model_selection(monkeypatch):
    captured: dict[str, object] = {}
    expected_result = object()

    def fake_run(**kwargs):
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr(
        relbench_trial_site_success,
        "run_rel_trial_regression_task",
        fake_run,
    )

    result = relbench_trial_site_success.run_rel_trial_site_success()

    assert result is expected_result
    assert captured["task_name"] == "site-success"
    assert captured["bounded_probability_target"] is True
