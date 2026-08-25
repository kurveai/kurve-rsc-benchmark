from __future__ import annotations

import datetime
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kurve_rsc"))

from relbench_user_badge_signal_experiments import (
    build_explicit_user_badge_signals,
    experiment_feature_sets,
)


def test_explicit_user_badge_signals_are_point_in_time_and_all_history() -> None:
    con = duckdb.connect()
    con.register(
        "users_src",
        pd.DataFrame(
            {
                "Id": [1],
                "CreationDate": pd.to_datetime(["2000-01-01"]),
            }
        ),
    )
    con.register(
        "badges_src",
        pd.DataFrame(
            {
                "Id": [1, 2, 3],
                "UserId": [1, 1, 1],
                "Class": [3, 2, 1],
                "Name": ["Old", "Recent", "Future"],
                "TagBased": [False, False, False],
                "Date": pd.to_datetime(
                    ["2001-01-01", "2020-06-01", "2021-01-01"]
                ),
            }
        ).astype({"Name": "object"}),
    )
    con.register(
        "post_history_src",
        pd.DataFrame(
            {
                "Id": [1, 2],
                "PostId": [10, 11],
                "UserId": [1, 1],
                "PostHistoryTypeId": [4, 5],
                "CreationDate": pd.to_datetime(["2002-01-01", "2021-01-01"]),
            }
        ),
    )
    con.register(
        "posts_src",
        pd.DataFrame(
            {
                "Id": [10, 11, 12],
                "OwnerUserId": [1, 1, 1],
                "PostTypeId": [1, 2, 2],
                "CreationDate": pd.to_datetime(
                    ["2019-01-01", "2020-06-01", "2021-01-01"]
                ),
            }
        ),
    )

    frame = build_explicit_user_badge_signals(
        con, datetime.datetime(2020, 10, 1)
    )
    row = frame.iloc[0]

    assert row["exp_badge_lifetime_count"] == 2
    assert row["exp_badge_distinct_name_count"] == 2
    assert row["exp_edit_lifetime_count"] == 1
    assert row["exp_post_question_lifetime_count"] == 1
    assert row["exp_post_answer_lifetime_count"] == 1
    assert row["exp_age_days"] > 7_000
    con.close()


def test_experiment_feature_sets_keep_baseline_and_add_only_named_groups() -> None:
    sets = experiment_feature_sets(
        [
            "current_a",
            "exp_age_days",
            "exp_badge_lifetime_count",
            "exp_edit_lifetime_count",
            "exp_post_question_lifetime_count",
        ]
    )

    assert sets["baseline"] == ["current_a"]
    assert sets["baseline_plus_account_badges"] == [
        "current_a",
        "exp_age_days",
        "exp_badge_lifetime_count",
    ]
    assert sets["baseline_plus_direct_edits"] == [
        "current_a",
        "exp_edit_lifetime_count",
    ]
    assert sets["baseline_plus_post_types"] == [
        "current_a",
        "exp_post_question_lifetime_count",
    ]
