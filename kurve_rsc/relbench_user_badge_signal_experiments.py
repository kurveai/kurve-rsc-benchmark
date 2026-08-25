#!/usr/bin/env python
"""One-off rel-stack/user-badge signal experiments.

This runner leaves the production user-badge builder unchanged. It builds the
current GraphReduce frame, appends a small set of explicit point-in-time
features, and compares fixed CatBoost models on identical split frames.
"""

from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from relbench_stack_task_utils import (  # noqa: E402
    STACK_TRAIN_FRAME_LIMIT,
    build_task_split_frame,
    build_user_badge_features,
    prepare_stack_views,
    select_shared_numeric_features,
    target_table_from_frame,
)
from relbench_catboost_utils import fit_incremental_classifier  # noqa: E402


EXPERIMENT_PREFIXES = {
    "account_badges": ("exp_age_", "exp_badge_"),
    "direct_edits": ("exp_edit_",),
    "post_types": ("exp_post_",),
}


def _timestamp_literal(value: datetime.datetime) -> str:
    timestamp = pd.Timestamp(value)
    return f"TIMESTAMP '{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')}'"


def build_explicit_user_badge_signals(
    con: duckdb.DuckDBPyConnection,
    cut_date: datetime.datetime,
) -> pd.DataFrame:
    """Build targeted all-history and type-aware features as of ``cut_date``."""

    cutoff = _timestamp_literal(cut_date)
    return con.sql(
        f"""
        WITH badge_history AS (
            SELECT
                UserId AS user_id,
                COUNT(*) AS exp_badge_lifetime_count,
                COUNT(DISTINCT Name) AS exp_badge_distinct_name_count,
                SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS exp_badge_class1_count,
                SUM(CASE WHEN Class = 2 THEN 1 ELSE 0 END) AS exp_badge_class2_count,
                SUM(CASE WHEN Class = 3 THEN 1 ELSE 0 END) AS exp_badge_class3_count,
                COUNT(*) FILTER (
                    WHERE Date >= {cutoff} - INTERVAL 90 DAY
                ) AS exp_badge_count_90d,
                COUNT(*) FILTER (
                    WHERE Date >= {cutoff} - INTERVAL 365 DAY
                ) AS exp_badge_count_365d,
                DATE_DIFF('day', MAX(Date), {cutoff}) AS exp_badge_days_since_last
            FROM badges_src
            WHERE Date < {cutoff}
              AND UserId IS NOT NULL
            GROUP BY UserId
        ),
        direct_edit_history AS (
            SELECT
                UserId AS user_id,
                COUNT(*) AS exp_edit_lifetime_count,
                COUNT(DISTINCT PostId) AS exp_edit_distinct_post_count,
                COUNT(DISTINCT PostHistoryTypeId) AS exp_edit_distinct_type_count,
                COUNT(*) FILTER (
                    WHERE CreationDate >= {cutoff} - INTERVAL 30 DAY
                ) AS exp_edit_count_30d,
                COUNT(*) FILTER (
                    WHERE CreationDate >= {cutoff} - INTERVAL 90 DAY
                ) AS exp_edit_count_90d,
                COUNT(*) FILTER (
                    WHERE CreationDate >= {cutoff} - INTERVAL 365 DAY
                ) AS exp_edit_count_365d,
                DATE_DIFF('day', MAX(CreationDate), {cutoff}) AS exp_edit_days_since_last
            FROM post_history_src
            WHERE CreationDate < {cutoff}
              AND UserId IS NOT NULL
            GROUP BY UserId
        ),
        post_type_history AS (
            SELECT
                OwnerUserId AS user_id,
                SUM(CASE WHEN PostTypeId = 1 THEN 1 ELSE 0 END)
                    AS exp_post_question_lifetime_count,
                SUM(CASE WHEN PostTypeId = 2 THEN 1 ELSE 0 END)
                    AS exp_post_answer_lifetime_count,
                SUM(CASE
                    WHEN PostTypeId = 1
                     AND CreationDate >= {cutoff} - INTERVAL 30 DAY
                    THEN 1 ELSE 0 END) AS exp_post_question_count_30d,
                SUM(CASE
                    WHEN PostTypeId = 2
                     AND CreationDate >= {cutoff} - INTERVAL 30 DAY
                    THEN 1 ELSE 0 END) AS exp_post_answer_count_30d,
                SUM(CASE
                    WHEN PostTypeId = 1
                     AND CreationDate >= {cutoff} - INTERVAL 90 DAY
                    THEN 1 ELSE 0 END) AS exp_post_question_count_90d,
                SUM(CASE
                    WHEN PostTypeId = 2
                     AND CreationDate >= {cutoff} - INTERVAL 90 DAY
                    THEN 1 ELSE 0 END) AS exp_post_answer_count_90d,
                SUM(CASE
                    WHEN PostTypeId = 1
                     AND CreationDate >= {cutoff} - INTERVAL 365 DAY
                    THEN 1 ELSE 0 END) AS exp_post_question_count_365d,
                SUM(CASE
                    WHEN PostTypeId = 2
                     AND CreationDate >= {cutoff} - INTERVAL 365 DAY
                    THEN 1 ELSE 0 END) AS exp_post_answer_count_365d
            FROM posts_src
            WHERE CreationDate < {cutoff}
              AND OwnerUserId IS NOT NULL
              AND OwnerUserId != -1
            GROUP BY OwnerUserId
        )
        SELECT
            users.Id AS user_id,
            DATE_DIFF('day', users.CreationDate, {cutoff}) AS exp_age_days,
            COALESCE(badges.exp_badge_lifetime_count, 0)
                AS exp_badge_lifetime_count,
            COALESCE(badges.exp_badge_distinct_name_count, 0)
                AS exp_badge_distinct_name_count,
            COALESCE(badges.exp_badge_class1_count, 0)
                AS exp_badge_class1_count,
            COALESCE(badges.exp_badge_class2_count, 0)
                AS exp_badge_class2_count,
            COALESCE(badges.exp_badge_class3_count, 0)
                AS exp_badge_class3_count,
            COALESCE(badges.exp_badge_count_90d, 0) AS exp_badge_count_90d,
            COALESCE(badges.exp_badge_count_365d, 0) AS exp_badge_count_365d,
            badges.exp_badge_days_since_last,
            COALESCE(badges.exp_badge_lifetime_count, 0) * 365.25
                / GREATEST(DATE_DIFF('day', users.CreationDate, {cutoff}), 30)
                AS exp_badge_count_per_account_year,
            COALESCE(badges.exp_badge_count_365d, 0) * 1.0
                / GREATEST(COALESCE(badges.exp_badge_lifetime_count, 0), 1)
                AS exp_badge_recent_share_365d,
            COALESCE(edits.exp_edit_lifetime_count, 0)
                AS exp_edit_lifetime_count,
            COALESCE(edits.exp_edit_distinct_post_count, 0)
                AS exp_edit_distinct_post_count,
            COALESCE(edits.exp_edit_distinct_type_count, 0)
                AS exp_edit_distinct_type_count,
            COALESCE(edits.exp_edit_count_30d, 0) AS exp_edit_count_30d,
            COALESCE(edits.exp_edit_count_90d, 0) AS exp_edit_count_90d,
            COALESCE(edits.exp_edit_count_365d, 0) AS exp_edit_count_365d,
            edits.exp_edit_days_since_last,
            COALESCE(edits.exp_edit_lifetime_count, 0) * 365.25
                / GREATEST(DATE_DIFF('day', users.CreationDate, {cutoff}), 30)
                AS exp_edit_count_per_account_year,
            COALESCE(posts.exp_post_question_lifetime_count, 0)
                AS exp_post_question_lifetime_count,
            COALESCE(posts.exp_post_answer_lifetime_count, 0)
                AS exp_post_answer_lifetime_count,
            COALESCE(posts.exp_post_question_count_30d, 0)
                AS exp_post_question_count_30d,
            COALESCE(posts.exp_post_answer_count_30d, 0)
                AS exp_post_answer_count_30d,
            COALESCE(posts.exp_post_question_count_90d, 0)
                AS exp_post_question_count_90d,
            COALESCE(posts.exp_post_answer_count_90d, 0)
                AS exp_post_answer_count_90d,
            COALESCE(posts.exp_post_question_count_365d, 0)
                AS exp_post_question_count_365d,
            COALESCE(posts.exp_post_answer_count_365d, 0)
                AS exp_post_answer_count_365d,
            COALESCE(posts.exp_post_question_lifetime_count, 0) * 1.0
                / GREATEST(
                    COALESCE(posts.exp_post_question_lifetime_count, 0)
                    + COALESCE(posts.exp_post_answer_lifetime_count, 0),
                    1
                ) AS exp_post_question_share
        FROM users_src AS users
        LEFT JOIN badge_history AS badges ON users.Id = badges.user_id
        LEFT JOIN direct_edit_history AS edits ON users.Id = edits.user_id
        LEFT JOIN post_type_history AS posts ON users.Id = posts.user_id
        WHERE users.CreationDate <= {cutoff}
          AND users.Id IS NOT NULL
        """
    ).to_df()


def build_augmented_user_badge_features(
    con: duckdb.DuckDBPyConnection,
    cut_date: datetime.datetime,
) -> pd.DataFrame:
    """Append explicit experiment features to the unchanged GraphReduce frame."""

    baseline = build_user_badge_features(con, cut_date)
    explicit = build_explicit_user_badge_signals(con, cut_date)
    baseline["user_Id"] = pd.to_numeric(baseline["user_Id"], errors="coerce")
    explicit["user_id"] = pd.to_numeric(explicit["user_id"], errors="coerce")
    return baseline.merge(
        explicit,
        left_on="user_Id",
        right_on="user_id",
        how="left",
        validate="one_to_one",
    ).drop(columns=["user_id"])


def experiment_feature_sets(
    all_features: Sequence[str],
) -> dict[str, list[str]]:
    """Return baseline and additive feature sets in deterministic order."""

    baseline = [column for column in all_features if not column.startswith("exp_")]
    groups = {
        name: [
            column
            for column in all_features
            if any(column.startswith(prefix) for prefix in prefixes)
        ]
        for name, prefixes in EXPERIMENT_PREFIXES.items()
    }
    return {
        "baseline": baseline,
        "baseline_plus_account_badges": baseline + groups["account_badges"],
        "baseline_plus_direct_edits": baseline + groups["direct_edits"],
        "baseline_plus_post_types": baseline + groups["post_types"],
        "baseline_plus_all_explicit": baseline
        + groups["account_badges"]
        + groups["direct_edits"]
        + groups["post_types"],
    }


def _configure_quiet_graphreduce_logging() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    try:
        import structlog

        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
        )
    except ImportError:
        pass


def _feature_diagnostics(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, int]:
    selected = frame[list(columns)]
    return {
        "columns": len(selected.columns),
        "constant": int(selected.nunique(dropna=False).le(1).sum()),
        "missing_ge_90pct": int(selected.isna().mean().ge(0.9).sum()),
    }


def main() -> None:
    _configure_quiet_graphreduce_logging()
    con = duckdb.connect()
    prepare_stack_views(con)

    train_store = None
    val_store = None
    test_store = None
    try:
        task, _, train_store, train_cut_date = build_task_split_frame(
            con,
            task_name="user-badge",
            split="train",
            feature_builder=build_augmented_user_badge_features,
            feature_entity_col="user_Id",
            use_all_timestamps=True,
            max_timestamps=STACK_TRAIN_FRAME_LIMIT,
        )
        _, _, val_store, val_cut_date = build_task_split_frame(
            con,
            task_name="user-badge",
            split="val",
            feature_builder=build_augmented_user_badge_features,
            feature_entity_col="user_Id",
        )
        _, _, test_store, test_cut_date = build_task_split_frame(
            con,
            task_name="user-badge",
            split="test",
            feature_builder=build_augmented_user_badge_features,
            feature_entity_col="user_Id",
        )

        df_val = val_store.to_dataframe()
        df_test = test_store.to_dataframe()
        target = task.target_col
        train_sample = train_store.sample_frame()
        all_features = select_shared_numeric_features(
            train_sample,
            df_val,
            df_test,
            target_col=target,
            excluded_cols={"user_Id", "user_AccountId", task.entity_col},
        )
        feature_sets = experiment_feature_sets(all_features)

        print("experiment: rel-stack/user-badge explicit signals", flush=True)
        print("train_cut_date:", train_cut_date.date(), flush=True)
        print("validation_cut_date:", val_cut_date.date(), flush=True)
        print("test_cut_date:", test_cut_date.date(), flush=True)
        print("train_frames:", len(train_store.part_paths), flush=True)
        print("train_rows:", train_store.row_count, flush=True)
        print("validation_rows:", len(df_val), flush=True)
        print("test_rows:", len(df_test), flush=True)
        print("validation_prevalence:", float(df_val[target].mean()), flush=True)
        print("test_prevalence:", float(df_test[target].mean()), flush=True)

        results: dict[str, dict[str, float]] = {}
        for experiment_name, features in feature_sets.items():
            print(
                "experiment_started:",
                experiment_name,
                _feature_diagnostics(df_val, features),
                flush=True,
            )
            model, validation_auc = fit_incremental_classifier(
                lambda: train_store.iter_batches(),
                features,
                target,
                df_val[features].fillna(0),
                df_val[target],
                batch_count=len(train_store.part_paths),
                config={
                    "iterations": 300,
                    "learning_rate": 0.05,
                    "depth": 4,
                    "l2_leaf_reg": 8.0,
                },
                auto_class_weights="Balanced",
            )
            validation_predictions = model.predict_proba(
                df_val[features].fillna(0)
            )[:, 1]
            test_predictions = model.predict_proba(
                df_test[features].fillna(0)
            )[:, 1]
            validation_metrics = task.evaluate(
                validation_predictions,
                target_table_from_frame(task, df_val),
            )
            test_metrics = task.evaluate(
                test_predictions,
                target_table_from_frame(task, df_test),
            )
            results[experiment_name] = {
                "validation_auc_from_fit": float(validation_auc),
                "validation_auroc": float(validation_metrics["roc_auc"]),
                "test_auroc": float(test_metrics["roc_auc"]),
                "validation_average_precision": float(
                    validation_metrics["average_precision"]
                ),
                "test_average_precision": float(
                    test_metrics["average_precision"]
                ),
                "feature_count": len(features),
            }
            print(
                "experiment_finished:", experiment_name, results[experiment_name],
                flush=True,
            )

        baseline_validation = results["baseline"]["validation_auroc"]
        baseline_test = results["baseline"]["test_auroc"]
        print("experiment_summary:", flush=True)
        for experiment_name, metrics in results.items():
            print(
                experiment_name,
                {
                    **metrics,
                    "validation_auroc_delta": (
                        metrics["validation_auroc"] - baseline_validation
                    ),
                    "test_auroc_delta": metrics["test_auroc"] - baseline_test,
                },
                flush=True,
            )
    finally:
        if train_store is not None:
            train_store.close()
        if val_store is not None:
            val_store.close()
        if test_store is not None:
            test_store.close()
        con.close()


if __name__ == "__main__":
    main()
