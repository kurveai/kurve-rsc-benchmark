from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from relbench.base import Database, Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kurve_rsc"))

from kurve_rsc import (
    relbench_trial_site_success,
    relbench_trial_study_adverse,
    relbench_trial_study_outcome,
)
from kurve_rsc.trial_builder import (
    _feature_cutoff,
    _schema_tree_edges,
    prepare_generic_model_inputs,
    select_generic_model_features,
)


def _table(
    columns: dict[str, list[object]],
    *,
    primary_key: str,
    foreign_keys: dict[str, str] | None = None,
    time_column: str | None = None,
) -> Table:
    return Table(
        df=pd.DataFrame(columns),
        fkey_col_to_pkey_table=foreign_keys or {},
        pkey_col=primary_key,
        time_col=time_column,
    )


def test_schema_tree_is_derived_from_metadata_and_drops_cycles() -> None:
    db = Database(
        {
            "entities": _table({"id": [1]}, primary_key="id"),
            "events": _table(
                {"event_id": [1], "entity_id": [1]},
                primary_key="event_id",
                foreign_keys={"entity_id": "entities"},
            ),
            "details": _table(
                {"detail_id": [1], "entity_id": [1], "event_id": [1]},
                primary_key="detail_id",
                foreign_keys={"entity_id": "entities", "event_id": "events"},
            ),
        }
    )

    edges = _schema_tree_edges(db, "entities")

    assert edges == [
        ("entities", "details", "entity_id", "entities"),
        ("entities", "events", "entity_id", "entities"),
    ]


def test_feature_cutoff_only_includes_the_exact_task_timestamp() -> None:
    timestamp = pd.Timestamp("2020-01-01")

    assert _feature_cutoff(timestamp) == timestamp + pd.Timedelta(microseconds=1)


def test_generic_model_features_and_categorical_layout_are_shared() -> None:
    train = pd.DataFrame(
        {
            "entity_id": [1, 2],
            "country": ["US", None],
            "count": [2, 1],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "target": [0.5, 0.7],
        }
    )
    validation = train.assign(country=["CA", "US"])
    test = train.assign(country=[None, "CA"])

    features = select_generic_model_features(
        train,
        validation,
        test,
        target_column="target",
        excluded_columns={"entity_id"},
    )
    train_inputs, categorical_indices = prepare_generic_model_inputs(train, features)
    test_inputs, replayed_indices = prepare_generic_model_inputs(
        test, features, categorical_indices
    )

    assert features == ["country", "count"]
    assert categorical_indices == [0]
    assert replayed_indices == [0]
    assert train_inputs["country"].tolist() == ["US", "__missing__"]
    assert test_inputs["country"].tolist() == ["__missing__", "CA"]


def test_all_trial_entry_points_use_the_same_generic_runner(monkeypatch) -> None:
    calls: list[str] = []
    expected = object()

    def fake_run(task_name: str, data_dir=None):
        calls.append(task_name)
        return expected

    modules_and_functions = [
        (relbench_trial_study_outcome, "run_rel_trial_study_outcome"),
        (relbench_trial_study_adverse, "run_rel_trial_study_adverse"),
        (relbench_trial_site_success, "run_rel_trial_site_success"),
    ]
    for module, function_name in modules_and_functions:
        monkeypatch.setattr(module, "run_generic_rel_trial_task", fake_run)
        assert getattr(module, function_name)() is expected

    assert calls == ["study-outcome", "study-adverse", "site-success"]
