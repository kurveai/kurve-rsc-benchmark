#!/usr/bin/env python
"""RelBench rel-event: user ignore example aligned to the official task definition."""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
from relbench_dataset_utils import (
    RelBenchFrameStore,
    get_relbench_dataset_db,
    get_relbench_split_task_table,
    iter_training_frames,
    register_relbench_db_views,
    target_table_from_frame,
)

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode
from relbench_catboost_utils import TEMPORAL_FEATURE_FAMILIES, fit_tuned_classifier_incremental, set_feature_families
from relbench_feature_policy import apply_feature_family_policy, configure_task_cli
from relbench_feature_manifest import (
    FeatureManifestSource,
    apply_feature_manifests,
    feature_manifest_enabled,
    load_feature_manifest_samples,
)

DATASET_NAME = "rel-event"
TABLES = ["users", "events", "event_attendees", "event_interest", "user_friends"]
TABLE_TO_VIEW = {
    "users": "users_src",
    "events": "events_src",
    "event_attendees": "event_attendees_src",
    "event_interest": "event_interest_src",
    "user_friends": "user_friends_src",
}
ROW_NUMBER_IDS = {
    "event_attendees": "attendee_id",
    "event_interest": "interest_id",
    "user_friends": "friendship_id",
}
DROP_COLUMNS = {
    "event_attendees": ["Unnamed: 0"],
    "user_friends": ["Unnamed: 0"],
}
VALIDATION_CUT_DATE = pd.Timestamp("2012-11-21")
TEST_CUT_DATE = pd.Timestamp("2012-11-29")
LABEL_TIMEDELTA = pd.Timedelta(days=7)
TARGET_COLUMN = "target"


def _catboost_inputs(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_indices: list[int] | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """Apply the training frame's categorical layout to every model split."""

    inputs = frame[feature_columns].copy()
    inferred_categorical_indices: list[int] = []
    frozen_categorical_indices = (
        None if categorical_indices is None else set(categorical_indices)
    )
    for index, column in enumerate(feature_columns):
        series = inputs[column]
        is_categorical = (
            index in frozen_categorical_indices
            if frozen_categorical_indices is not None
            else (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
            )
        )
        if is_categorical:
            inputs[column] = series.fillna("__missing__").astype(str)
            inferred_categorical_indices.append(index)
        else:
            inputs[column] = pd.to_numeric(series, errors="coerce").fillna(0)
    return inputs, inferred_categorical_indices


def _columns_present_in_all_frames(frames: Iterable[pd.DataFrame]) -> list[str]:
    """Return columns shared by every frame, ordered like the first frame."""

    frame_iterator = iter(frames)
    first_frame = next(frame_iterator, None)
    if first_frame is None:
        return []

    ordered_columns = first_frame.columns.tolist()
    common_columns = set(ordered_columns)
    for frame in frame_iterator:
        common_columns.intersection_update(frame.columns)
    return [column for column in ordered_columns if column in common_columns]


class _FrozenGraphOperations:
    """Capture one GraphReduce operation plan and replay it on later cutoffs."""

    def __init__(self) -> None:
        self.plan: dict[str, object] | None = None
        self.source_split: str | None = None
        self.source_cut_date: pd.Timestamp | None = None

    @property
    def is_frozen(self) -> bool:
        return self.plan is not None

    def apply(self, graph: GraphReduce) -> None:
        if self.plan is not None:
            graph.apply_execution_plan(self.plan)

    def capture(
        self,
        graph: GraphReduce,
        features: pd.DataFrame,
        *,
        split_name: str,
        cut_date: pd.Timestamp,
    ) -> None:
        if self.plan is not None:
            return

        plan = graph.freeze_execution_plan()
        records = plan.get("records", [])
        if not records:
            raise RuntimeError("GraphReduce produced an empty execution plan")
        self.plan = plan
        self.source_split = split_name
        self.source_cut_date = pd.Timestamp(cut_date)
        print(
            "frozen_execution_plan: "
            f"split={split_name} cutoff={pd.Timestamp(cut_date).isoformat()} "
            f"records={len(records)} features={len(features.columns)}",
            flush=True,
        )


def run_rel_event_user_ignore(
    data_dir: Path | None = None,
    *,
    use_feature_manifest: bool = False,
) -> tuple[RelBenchFrameStore, pd.DataFrame, pd.DataFrame, dict[str, float] | None, dict[str, float] | None, int, list[str], str]:
    _, db = get_relbench_dataset_db(
        DATASET_NAME, download=True, upto_test_timestamp=False
    )
    materialized: list[str] = []

    con = duckdb.connect()
    split_frames: dict[str, RelBenchFrameStore] = {}
    split_tasks = {}
    frozen_graph_operations = _FrozenGraphOperations()
    feature_manifest_summary: dict[str, dict[str, int]] = {}

    try:
        register_relbench_db_views(con, db, TABLE_TO_VIEW, ROW_NUMBER_IDS, DROP_COLUMNS)

        bounds = con.sql(
            """
            SELECT
                MIN(ts) AS min_timestamp
            FROM (
                SELECT joinedAt AS ts FROM users_src WHERE joinedAt IS NOT NULL
                UNION ALL
                SELECT start_time AS ts FROM event_attendees_src WHERE start_time IS NOT NULL
                UNION ALL
                SELECT timestamp AS ts FROM event_interest_src WHERE timestamp IS NOT NULL
            )
            """
        ).to_df()
        lookback_start = pd.Timestamp(bounds.loc[0, "min_timestamp"])
        user_columns = con.sql("SELECT * FROM users_src LIMIT 0").to_df().columns.tolist()
        event_columns = con.sql("SELECT * FROM events_src LIMIT 0").to_df().columns.tolist()
        event_excluded_columns = tuple(
            column for column in event_columns if column.lower() == "zip"
        )
        # RelBench stores ZIP codes as strings. GraphReduce 1.9.1 can infer
        # numeric-looking strings as aggregate inputs and then emit SUM(zip),
        # which DuckDB rejects for this VARCHAR column.
        event_columns = [column for column in event_columns if column.lower() != "zip"]
        attendee_columns = con.sql("SELECT * FROM event_attendees_src LIMIT 0").to_df().columns.tolist()
        interest_columns = con.sql("SELECT * FROM event_interest_src LIMIT 0").to_df().columns.tolist()
        friend_columns = con.sql("SELECT * FROM user_friends_src LIMIT 0").to_df().columns.tolist()

        user_id_col = {column.lower(): column for column in user_columns}["user_id"]
        user_date_col = {column.lower(): column for column in user_columns}["joinedat"]
        event_id_col = {column.lower(): column for column in event_columns}["event_id"]
        event_user_col = {column.lower(): column for column in event_columns}["user_id"]
        event_date_col = {column.lower(): column for column in event_columns}["start_time"]
        attendee_id_col = {column.lower(): column for column in attendee_columns}["attendee_id"]
        attendee_event_col = {column.lower(): column for column in attendee_columns}["event"]
        attendee_user_col = {column.lower(): column for column in attendee_columns}["user_id"]
        attendee_status_col = {column.lower(): column for column in attendee_columns}["status"]
        attendee_date_col = {column.lower(): column for column in attendee_columns}["start_time"]
        interest_id_col = {column.lower(): column for column in interest_columns}["interest_id"]
        interest_event_col = {column.lower(): column for column in interest_columns}["event"]
        interest_user_col = {column.lower(): column for column in interest_columns}["user"]
        interest_date_col = {column.lower(): column for column in interest_columns}["timestamp"]
        friend_id_col = {column.lower(): column for column in friend_columns}["friendship_id"]
        friend_user_col = {column.lower(): column for column in friend_columns}["user"]

        feature_manifest_sources = {
            "users": FeatureManifestSource("users_src", user_date_col),
            "events": FeatureManifestSource(
                "events_src",
                event_date_col,
                (event_user_col,),
                event_excluded_columns,
            ),
            "attendees": FeatureManifestSource(
                "event_attendees_src",
                attendee_date_col,
                (attendee_event_col, attendee_user_col),
            ),
            "interest": FeatureManifestSource(
                "event_interest_src",
                interest_date_col,
                (interest_event_col, interest_user_col),
            ),
            "friends": FeatureManifestSource(
                "user_friends_src", foreign_keys=(friend_user_col,)
            ),
        }
        feature_manifest_samples = (
            load_feature_manifest_samples(
                con,
                feature_manifest_sources,
                VALIDATION_CUT_DATE,
            )
            if use_feature_manifest
            else {}
        )

        official_tables = {}
        split_cut_dates = {}
        for split_name in ("train", "val", "test"):
            task, task_table, cut_timestamps = get_relbench_split_task_table(
                DATASET_NAME, "user-ignore", split_name, download=True, db=db
            )
            split_tasks[split_name] = task
            official_tables[split_name] = task_table.df.copy()
            split_cut_dates[split_name] = cut_timestamps

        for split_name, cut_dates in split_cut_dates.items():
            frame_store = RelBenchFrameStore(
                f"rel-event-user-ignore-{split_name}", persist_each_frame=True
            )

            def build_frame(frame_con, cut_date):
                con = frame_con
                feature_cut_date = pd.Timestamp(cut_date) + pd.Timedelta(seconds=1)
                feature_cut_timestamp = feature_cut_date.strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )

                users_node = DuckdbNode(
                    fpath="users_src",
                    prefix="usr",
                    pk=user_id_col,
                    date_key=user_date_col,
                    columns=user_columns,
                )
                events_node = DuckdbNode(
                    fpath="events_src",
                    prefix="evt",
                    pk=event_id_col,
                    date_key=event_date_col,
                    columns=event_columns,
                )
                attendees_node = DuckdbNode(
                    fpath="event_attendees_src",
                    prefix="att",
                    pk=attendee_id_col,
                    date_key=attendee_date_col,
                    columns=attendee_columns,
                    feature_family_max_columns=4,
                    categorical_top_k=5,
                    context_keys=(attendee_event_col,),
                    annotation_expressions={
                        "is_attending": f"{{{attendee_status_col}}} IN ('yes', 'maybe')",
                        "is_declined": f"{{{attendee_status_col}}} = 'no'",
                    },
                )
                interest_node = DuckdbNode(
                    fpath="event_interest_src",
                    prefix="int",
                    pk=interest_id_col,
                    date_key=interest_date_col,
                    columns=interest_columns,
                    feature_family_max_columns=4,
                    categorical_top_k=5,
                    context_keys=(interest_event_col,),
                )
                friends_node = DuckdbNode(
                    fpath="user_friends_src",
                    prefix="frd",
                    pk=friend_id_col,
                    date_key=None,
                    columns=friend_columns,
                )

                graph = GraphReduce(
                    name=f"rel_event_user_ignore_{pd.Timestamp(cut_date).date()}",
                    parent_node=users_node,
                    compute_layer=ComputeLayerEnum.duckdb,
                    sql_client=con,
                    cut_date=feature_cut_date.to_pydatetime(),
                    compute_period_val=max(1, int((feature_cut_date - lookback_start).days + 1)),
                    compute_period_unit=PeriodUnit.day,
                    auto_features=True,
                    auto_labels=False,
                    date_filters_on_agg=True,
                    auto_feature_hops_back=3,
                    auto_feature_hops_front=0,
                    use_temp_tables=True,
                )

                nodes = [users_node, events_node, attendees_node, interest_node, friends_node]
                set_feature_families(
                    [attendees_node, interest_node], TEMPORAL_FEATURE_FAMILIES
                )
                if use_feature_manifest:
                    current_summary = apply_feature_manifests(
                        {
                            "users": users_node,
                            "events": events_node,
                            "attendees": attendees_node,
                            "interest": interest_node,
                            "friends": friends_node,
                        },
                        feature_manifest_sources,
                        feature_manifest_samples,
                    )
                    if not feature_manifest_summary:
                        feature_manifest_summary.update(current_summary)
                apply_feature_family_policy(nodes)
                for node in nodes:
                    graph.add_node(node)

                graph.add_entity_edge(users_node, attendees_node, user_id_col, attendee_user_col, reduce=True)
                graph.add_entity_edge(users_node, interest_node, user_id_col, interest_user_col, reduce=True)
                graph.add_entity_edge(users_node, friends_node, user_id_col, friend_user_col, reduce=True)
                graph.add_entity_edge(users_node, events_node, user_id_col, event_user_col, reduce=True)
                graph.add_entity_edge(attendees_node, events_node, attendee_event_col, event_id_col, reduce=False)
                graph.add_entity_edge(interest_node, events_node, interest_event_col, event_id_col, reduce=False)

                frozen_graph_operations.apply(graph)
                graph.do_transformations_sql()
                features = con.sql(f"SELECT * FROM {graph.parent_node._cur_data_ref}").to_df().copy()
                frozen_graph_operations.capture(
                    graph,
                    features,
                    split_name=split_name,
                    cut_date=pd.Timestamp(cut_date),
                )
                graph._clean_refs()
                features["timestamp"] = pd.Timestamp(cut_date)

                labels = official_tables[split_name].copy()
                labels["timestamp"] = pd.to_datetime(labels["timestamp"])
                labels = labels[labels["timestamp"] == pd.Timestamp(cut_date)].copy()
                labels["user"] = labels["user"].astype("int64")

                frame = features.merge(
                    labels,
                    left_on=["timestamp", f"usr_{user_id_col}"],
                    right_on=["timestamp", "user"],
                    how="right",
                    validate="one_to_one",
                )
                frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("int8")
                return frame

            pending_cut_dates = list(cut_dates)
            if pending_cut_dates and not frozen_graph_operations.is_frozen:
                # The planning frame must finish before any parallel workers
                # can replay the immutable plan.
                first_cut_date = pending_cut_dates.pop(0)
                frame_store.append(build_frame(con, first_cut_date))

            frame_workers = None if split_name == "train" else 1
            for frame in iter_training_frames(
                con,
                pending_cut_dates,
                build_frame,
                workers=frame_workers,
            ):
                frame_store.append(frame)

            split_frames[split_name] = frame_store
    finally:
        con.close()

    if use_feature_manifest:
        print("feature_manifest_profile:", feature_manifest_summary, flush=True)

    common_columns = set(
        _columns_present_in_all_frames(
            chain.from_iterable(
                split_frames[split_name].iter_batches()
                for split_name in ("train", "val", "test")
            )
        )
    )
    train_store = split_frames["train"]
    df_val = split_frames["val"].to_dataframe()
    df_test = split_frames["test"].to_dataframe()
    split_frames["val"].close()
    split_frames["test"].close()
    train_sample = train_store.sample_frame()
    feature_columns = [
        column
        for column in train_sample.columns
        if column in common_columns
        and column != TARGET_COLUMN
        and "label" not in column.lower()
        and "user_id" not in column.lower()
        and not pd.api.types.is_datetime64_any_dtype(train_sample[column])
        and (
            pd.api.types.is_numeric_dtype(train_sample[column])
            or pd.api.types.is_bool_dtype(train_sample[column])
            or pd.api.types.is_object_dtype(train_sample[column])
            or pd.api.types.is_string_dtype(train_sample[column])
            or pd.api.types.is_categorical_dtype(train_sample[column])
        )
    ]
    if not feature_columns or train_store.target_nunique(TARGET_COLUMN) < 2:
        return train_store, df_val, df_test, None, None, len(feature_columns), materialized, TARGET_COLUMN

    _, categorical_indices = _catboost_inputs(train_sample, feature_columns)
    val_inputs, _ = _catboost_inputs(
        df_val,
        feature_columns,
        categorical_indices,
    )
    test_inputs, _ = _catboost_inputs(
        df_test,
        feature_columns,
        categorical_indices,
    )

    def train_batches():
        for batch in train_store.iter_batches():
            inputs, _ = _catboost_inputs(
                batch,
                feature_columns,
                categorical_indices,
            )
            inputs[TARGET_COLUMN] = batch[TARGET_COLUMN].to_numpy()
            yield inputs

    model, best_config, best_val_auc = fit_tuned_classifier_incremental(
        train_batches,
        feature_columns,
        TARGET_COLUMN,
        val_inputs,
        df_val[TARGET_COLUMN],
        batch_count=len(train_store.part_paths),
        cat_features=categorical_indices,
        auto_class_weights="Balanced",
    )
    print("catboost_config:", best_config, flush=True)
    print("catboost_validation_auc:", best_val_auc, flush=True)
    print("catboost_best_iteration:", model.get_best_iteration(), flush=True)

    val_predictions = np.asarray(model.predict_proba(val_inputs)[:, 1], dtype="float64")
    test_predictions = np.asarray(model.predict_proba(test_inputs)[:, 1], dtype="float64")
    val_metrics = split_tasks["val"].evaluate(
        val_predictions,
        target_table=target_table_from_frame(split_tasks["val"], df_val),
    )
    test_metrics = split_tasks["test"].evaluate(
        test_predictions,
        target_table=target_table_from_frame(split_tasks["test"], df_test),
    )
    return train_store, df_val, df_test, val_metrics, test_metrics, len(feature_columns), materialized, TARGET_COLUMN


def main() -> None:
    use_feature_manifest = feature_manifest_enabled("rel-event/user-ignore")
    df_train, df_val, df_test, val_metrics, test_metrics, n_features, materialized, target = run_rel_event_user_ignore(
        use_feature_manifest=use_feature_manifest,
    )
    print("feature_manifest_enabled:", use_feature_manifest, flush=True)
    print("materialized_files:", materialized, flush=True)
    print("validation_timestamp:", VALIDATION_CUT_DATE.date(), flush=True)
    print("test_timestamp:", TEST_CUT_DATE.date(), flush=True)
    print("target:", target, flush=True)
    print("train_rows:", df_train.row_count, flush=True)
    print("validation_rows:", len(df_val), flush=True)
    print("test_rows:", len(df_test), flush=True)
    print("feature_count:", n_features, flush=True)
    print("validation_metrics:", val_metrics if val_metrics is not None else "skipped", flush=True)
    print("test_metrics:", test_metrics if test_metrics is not None else "skipped", flush=True)
    df_train.close()


if __name__ == "__main__":
    configure_task_cli(description=__doc__)
    main()
