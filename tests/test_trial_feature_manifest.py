from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kurve_rsc"))

from graphreduce.node import DuckdbNode
from relbench_feature_manifest import (
    FEATURE_MANIFEST_ENV,
    LEGACY_TRIAL_FEATURE_MANIFEST_ENV,
    FeatureManifestSource,
    TRIAL_FEATURE_MANIFEST_SOURCES,
    apply_feature_manifests,
    feature_manifest_enabled,
    load_feature_manifest_samples,
)


def test_manifest_samples_are_frozen_before_validation() -> None:
    con = duckdb.connect()
    try:
        con.register(
            "events_src",
            pd.DataFrame(
                {
                    "event_id": [1, 2, 3],
                    "event_time": pd.to_datetime(
                        ["2019-01-01", "2020-01-01", "2021-01-01"]
                    ),
                    "value": [10, 20, 30],
                }
            ),
        )
        samples = load_feature_manifest_samples(
            con,
            {
                "events": FeatureManifestSource(
                    "events_src", "event_time"
                )
            },
            pd.Timestamp("2020-01-01"),
        )
    finally:
        con.close()

    assert samples["events"]["event_id"].tolist() == [1]


def test_shared_manifest_environment_takes_precedence(monkeypatch) -> None:
    monkeypatch.delenv(FEATURE_MANIFEST_ENV, raising=False)
    monkeypatch.setenv(LEGACY_TRIAL_FEATURE_MANIFEST_ENV, "1")
    assert feature_manifest_enabled() is True

    monkeypatch.setenv(FEATURE_MANIFEST_ENV, "0")
    assert feature_manifest_enabled() is False


def test_trial_manifest_applies_safe_source_columns_and_annotations() -> None:
    row_count = 1_002
    sample = pd.DataFrame(
        {
            "id": range(row_count),
            "nct_id": range(row_count),
            "date": pd.date_range("2015-01-01", periods=row_count, freq="h"),
            "outcome_type": ["Primary", "Secondary"] * (row_count // 2),
            "constant": 1,
            "high_cardinality": [f"value_{index}" for index in range(row_count)],
        }
    )
    node = DuckdbNode(
        fpath="outcomes_src",
        prefix="out",
        pk="id",
        date_key="date",
        columns=["id", "nct_id", "date"],
    )

    summary = apply_feature_manifests(
        {"outcomes": node},
        TRIAL_FEATURE_MANIFEST_SOURCES,
        {"outcomes": sample},
    )

    assert node.columns == ["id", "nct_id", "date", "outcome_type"]
    assert node.feature_manifest is not None
    assert node.feature_manifest.categorical_columns == ("outcome_type",)
    assert node.auto_annotate_features is True
    assert node.auto_text_features is True
    assert summary == {
        "outcomes": {"source": 6, "graph": 4, "features": 1}
    }
