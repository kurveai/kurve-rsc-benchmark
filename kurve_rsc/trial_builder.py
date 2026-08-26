#!/usr/bin/env python
"""Schema-driven GraphReduce runner for every RelBench rel-trial entity task.

The adapter deliberately contains no task-specific feature definitions.  Nodes,
columns, timestamps, and edges are derived from the RelBench database metadata;
the official task object supplies the entity table, labels, splits, task type,
and evaluator.
"""

from __future__ import annotations

import os
import re
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Sequence

import duckdb
import numpy as np
import pandas as pd

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode

try:
    from .relbench_catboost_utils import (
        enable_all_feature_families,
        fit_tuned_classifier_incremental,
        fit_tuned_regressor_incremental,
        selected_model_backend,
    )
    from .relbench_dataset_utils import (
        RelBenchFrameStore,
        get_relbench_dataset_db,
        get_relbench_split_task_table,
        iter_training_frames,
        register_relbench_db_views,
        target_table_from_frame,
    )
    from .relbench_regression_metrics import add_nmae
    from .relbench_feature_policy import (
        apply_feature_family_policy,
        baseline_feature_family_enabled,
    )
except ImportError:  # Direct execution with ``kurve_rsc`` on sys.path.
    from relbench_catboost_utils import (
        enable_all_feature_families,
        fit_tuned_classifier_incremental,
        fit_tuned_regressor_incremental,
        selected_model_backend,
    )
    from relbench_dataset_utils import (
        RelBenchFrameStore,
        get_relbench_dataset_db,
        get_relbench_split_task_table,
        iter_training_frames,
        register_relbench_db_views,
        target_table_from_frame,
    )
    from relbench_regression_metrics import add_nmae
    from relbench_feature_policy import (
        apply_feature_family_policy,
        baseline_feature_family_enabled,
    )


DATASET_NAME = "rel-trial"
TRIAL_FEATURE_HOPS_ENV = "KURVE_RSC_TRIAL_FEATURE_HOPS"
TRIAL_AUTO_ANNOTATE_ENV = "KURVE_RSC_TRIAL_AUTO_ANNOTATE"
_IDENTIFIER_PATTERN = re.compile(r"[^0-9A-Za-z_]+")


def _sql_name(value: str) -> str:
    """Return a stable SQL/GraphReduce identifier derived only from metadata."""

    normalized = _IDENTIFIER_PATTERN.sub("_", value).strip("_").lower()
    if not normalized:
        raise ValueError(f"Cannot derive an identifier from {value!r}")
    if normalized[0].isdigit():
        normalized = f"t_{normalized}"
    return normalized


def _view_name(table_name: str) -> str:
    return f"{_sql_name(table_name)}_src"


def _feature_cutoff(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Make GraphReduce's strict upper bound inclusive at the task timestamp."""

    return pd.Timestamp(timestamp) + pd.Timedelta(microseconds=1)


def _database_lookback_days(db: object, cutoff: pd.Timestamp) -> int:
    starts = [
        pd.Timestamp(table.df[table.time_col].min())
        for table in db.table_dict.values()
        if table.time_col is not None and not table.df.empty
    ]
    if not starts:
        return 1
    return max(1, int((pd.Timestamp(cutoff) - min(starts)).days) + 1)


def configure_generic_trial_feature_families(
    nodes: Sequence[DuckdbNode],
) -> None:
    """Enable every feature family, then apply the optional baseline override."""

    enable_all_feature_families(nodes)
    apply_feature_family_policy(nodes)


def generic_trial_feature_hops(default: int = 3) -> int:
    """Return the schema traversal depth used by generic Trial graphs."""

    raw_value = os.environ.get(TRIAL_FEATURE_HOPS_ENV, str(default)).strip()
    try:
        hops = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{TRIAL_FEATURE_HOPS_ENV} must be a positive integer") from exc
    if hops < 1:
        raise ValueError(f"{TRIAL_FEATURE_HOPS_ENV} must be a positive integer")
    return hops


def _trial_env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name, "1" if default else "0").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def trial_auto_annotate_enabled(default: bool = True) -> bool:
    return _trial_env_flag(TRIAL_AUTO_ANNOTATE_ENV, default)


def trial_auto_annotate_active() -> bool:
    """Return whether annotations belong in the active feature-family mode."""

    return trial_auto_annotate_enabled() and not baseline_feature_family_enabled()


def _schema_tree_edges(
    db: object,
    root_table: str,
) -> list[tuple[str, str, str, str]]:
    """Return a deterministic, cycle-free traversal of the RelBench FK graph.

    Each tuple is ``(current_table, next_table, child_fk, parent_table)``.
    Keeping only the first edge that discovers a table prevents schema cycles
    from expanding the same relation more than once.
    """

    if root_table not in db.table_dict:
        raise ValueError(
            f"Task entity table {root_table!r} is absent from the database"
        )

    adjacency: dict[str, list[tuple[str, str, str]]] = {
        name: [] for name in db.table_dict
    }
    for child_name, child in db.table_dict.items():
        for foreign_key, parent_name in child.fkey_col_to_pkey_table.items():
            if parent_name not in db.table_dict:
                raise ValueError(
                    f"Foreign key {child_name}.{foreign_key} references missing "
                    f"table {parent_name!r}"
                )
            adjacency[child_name].append((parent_name, foreign_key, parent_name))
            adjacency[parent_name].append((child_name, foreign_key, parent_name))

    visited = {root_table}
    queue = deque([root_table])
    edges: list[tuple[str, str, str, str]] = []
    while queue:
        current = queue.popleft()
        for neighbor, foreign_key, parent_name in sorted(adjacency[current]):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
            edges.append((current, neighbor, foreign_key, parent_name))
    return edges


def build_generic_trial_features(
    con: duckdb.DuckDBPyConnection,
    db: object,
    entity_table: str,
    task_timestamp: pd.Timestamp,
    entity_keys: pd.Series | None = None,
) -> tuple[pd.DataFrame, str]:
    """Build one feature frame entirely from the RelBench schema metadata."""

    if entity_table not in db.table_dict:
        raise ValueError(f"Unknown entity table: {entity_table}")

    feature_cutoff = _feature_cutoff(task_timestamp)
    root_source = _view_name(entity_table)
    registered_keys_ref: str | None = None
    if entity_keys is not None:
        root_pk = db.table_dict[entity_table].pkey_col
        timestamp_suffix = pd.Timestamp(task_timestamp).strftime("%Y%m%d%H%M%S%f")
        registered_keys_ref = f"_generic_task_entity_keys_{timestamp_suffix}"
        filtered_root_source = f"_generic_task_entities_src_{timestamp_suffix}"
        con.register(
            registered_keys_ref,
            pd.DataFrame({root_pk: entity_keys.drop_duplicates()}),
        )
        con.sql(
            f"""
            CREATE OR REPLACE TEMP VIEW {filtered_root_source} AS
            SELECT source.*
            FROM {root_source} AS source
            INNER JOIN {registered_keys_ref} AS task_entities
                USING ({root_pk})
            """
        )
        root_source = filtered_root_source
    nodes: dict[str, DuckdbNode] = {}
    for table_name in sorted(db.table_dict):
        table = db.table_dict[table_name]
        if table.pkey_col is None:
            raise ValueError(
                f"Generic GraphReduce requires a primary key for table {table_name!r}"
            )
        filters = None
        if table_name == entity_table and table.time_col is not None:
            prefix = _sql_name(table_name)
            filters = [
                sqlop(
                    optype=SQLOpType.where,
                    opval=(
                        f"{prefix}_{table.time_col} < "
                        f"TIMESTAMP '{feature_cutoff.isoformat(sep=' ')}'"
                    ),
                )
            ]
        nodes[table_name] = DuckdbNode(
            fpath=(
                root_source
                if table_name == entity_table
                else _view_name(table_name)
            ),
            prefix=_sql_name(table_name),
            pk=table.pkey_col,
            date_key=table.time_col,
            columns=table.df.columns.tolist(),
            do_filters_ops=filters,
            auto_annotate_features=trial_auto_annotate_active(),
            # Trial contains many unique names and long descriptions. Generic
            # text-shape annotations add width without domain meaning, while
            # bounded categorical indicators complement the predicates above.
            auto_text_features=False,
            auto_annotate_max_categorical_columns=8,
            auto_annotate_max_gated_numeric_cols=0,
            auto_annotate_gated_numeric_top_k=0,
            categorical_cardinality_threshold=12,
            categorical_top_k=5,
        )

    root = nodes[entity_table]
    graph = GraphReduce(
        name=(
            f"generic_{_sql_name(DATASET_NAME)}_{_sql_name(entity_table)}_"
            f"{pd.Timestamp(task_timestamp).date()}"
        ),
        parent_node=root,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        cut_date=feature_cutoff.to_pydatetime(),
        compute_period_val=_database_lookback_days(db, feature_cutoff),
        compute_period_unit=PeriodUnit.day,
        auto_features=True,
        auto_labels=False,
        date_filters_on_agg=True,
        auto_feature_hops_back=generic_trial_feature_hops(),
        auto_feature_hops_front=0,
        use_temp_tables=True,
    )
    configure_generic_trial_feature_families(list(nodes.values()))
    for node in nodes.values():
        graph.add_node(node)

    for current_name, neighbor_name, foreign_key, parent_name in _schema_tree_edges(
        db, entity_table
    ):
        current = nodes[current_name]
        neighbor = nodes[neighbor_name]
        if current_name == parent_name:
            graph.add_entity_edge(
                current,
                neighbor,
                parent_key=db.table_dict[current_name].pkey_col,
                relation_key=foreign_key,
                reduce=True,
            )
        else:
            graph.add_entity_edge(
                current,
                neighbor,
                parent_key=foreign_key,
                relation_key=db.table_dict[neighbor_name].pkey_col,
                reduce=True,
            )

    graph.do_transformations_sql()
    features = con.sql(
        f"SELECT * FROM {graph.parent_node._cur_data_ref}"
    ).to_df().copy()
    graph._clean_refs()
    if registered_keys_ref is not None:
        con.unregister(registered_keys_ref)
    features["timestamp"] = pd.Timestamp(task_timestamp)
    entity_feature_column = (
        f"{_sql_name(entity_table)}_{db.table_dict[entity_table].pkey_col}"
    )
    return features, entity_feature_column


def _build_split(
    con: duckdb.DuckDBPyConnection,
    db: object,
    task_name: str,
    split: str,
) -> tuple[object, RelBenchFrameStore]:
    task, task_table, timestamps = get_relbench_split_task_table(
        DATASET_NAME,
        task_name,
        split,
        download=True,
        db=db,
    )
    store = RelBenchFrameStore(
        f"generic-{DATASET_NAME}-{task_name}-{split}",
        persist_each_frame=True,
    )
    labels = task_table.df.copy()
    labels[task.time_col] = pd.to_datetime(labels[task.time_col])
    labels["_entity_key"] = labels[task.entity_col].astype(str)

    def build_frame(
        frame_con: duckdb.DuckDBPyConnection,
        timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        timestamp_labels = labels[
            labels[task.time_col] == pd.Timestamp(timestamp)
        ]
        features, entity_feature_column = build_generic_trial_features(
            frame_con,
            db,
            task.entity_table,
            pd.Timestamp(timestamp),
            entity_keys=timestamp_labels[task.entity_col],
        )
        features["_entity_key"] = features[entity_feature_column].astype(str)
        return features.merge(
            timestamp_labels[
                ["_entity_key", task.time_col, task.entity_col, task.target_col]
            ],
            left_on=["timestamp", "_entity_key"],
            right_on=[task.time_col, "_entity_key"],
            how="right",
            validate="one_to_one",
        ).drop(columns=["_entity_key"])

    workers = None if split == "train" else 1
    for frame in iter_training_frames(con, timestamps, build_frame, workers=workers):
        store.append(frame)
    return task, store


def select_generic_model_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
    excluded_columns: set[str],
) -> list[str]:
    """Select one task-independent set of CatBoost-compatible columns."""

    common = set(train.columns) & set(validation.columns) & set(test.columns)
    return [
        column
        for column in train.columns
        if column in common
        and column != target_column
        and column not in excluded_columns
        and not pd.api.types.is_datetime64_any_dtype(train[column])
        and (
            pd.api.types.is_numeric_dtype(train[column])
            or pd.api.types.is_bool_dtype(train[column])
            or pd.api.types.is_object_dtype(train[column])
            or pd.api.types.is_string_dtype(train[column])
            or isinstance(train[column].dtype, pd.CategoricalDtype)
        )
    ]


def prepare_generic_model_inputs(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_indices: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """Normalize model inputs and replay the training categorical layout."""

    inputs = frame.reindex(columns=list(feature_columns)).copy()
    frozen = None if categorical_indices is None else set(categorical_indices)
    inferred: list[int] = []
    for index, column in enumerate(feature_columns):
        series = inputs[column]
        categorical = (
            index in frozen
            if frozen is not None
            else (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
            )
        )
        if categorical:
            inputs[column] = series.fillna("__missing__").astype(str)
            inferred.append(index)
        else:
            inputs[column] = pd.to_numeric(series, errors="coerce").fillna(0)
    return inputs, inferred


def _task_type_name(task: object) -> str:
    task_type = getattr(task.task_type, "value", task.task_type)
    if task_type not in {"binary_classification", "regression"}:
        raise ValueError(f"Unsupported rel-trial task type: {task_type}")
    return str(task_type)


def run_generic_rel_trial_task(
    task_name: str,
    data_dir: Path | None = None,
) -> tuple[
    RelBenchFrameStore,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float] | None,
    dict[str, float] | None,
    int,
    list[str],
    str,
]:
    """Run a RelBench Trial task without any task-specific feature behavior."""

    del data_dir  # RelBench owns dataset retrieval and caching.
    print("trial_feature_hops:", generic_trial_feature_hops(), flush=True)
    print("trial_auto_annotate:", trial_auto_annotate_active(), flush=True)
    _, db = get_relbench_dataset_db(
        DATASET_NAME,
        download=True,
        upto_test_timestamp=False,
    )
    split_tasks: dict[str, object] = {}
    split_stores: dict[str, RelBenchFrameStore] = {}
    with TemporaryDirectory(prefix=f"kurve-{_sql_name(task_name)}-duckdb-") as temp_dir:
        con = duckdb.connect()
        try:
            # DuckDB otherwise spills every concurrently running task into the
            # same relative ``.tmp`` directory.  Deep Trial graphs can spill
            # heavily, so shared files make parallel tasks corrupt one another.
            con.execute("SET temp_directory = ?", [temp_dir])
            register_relbench_db_views(
                con,
                db,
                {
                    table_name: _view_name(table_name)
                    for table_name in db.table_dict
                },
            )
            for split in ("train", "val", "test"):
                task, store = _build_split(con, db, task_name, split)
                split_tasks[split] = task
                split_stores[split] = store
        finally:
            con.close()

    train_store = split_stores["train"]
    validation = split_stores["val"].to_dataframe()
    test = split_stores["test"].to_dataframe()
    split_stores["val"].close()
    split_stores["test"].close()

    task = split_tasks["train"]
    target = task.target_col
    train_sample = train_store.sample_frame()
    root_pk = db.table_dict[task.entity_table].pkey_col
    root_feature_key = f"{_sql_name(task.entity_table)}_{root_pk}"
    features = select_generic_model_features(
        train_sample,
        validation,
        test,
        target_column=target,
        excluded_columns={task.entity_col, root_feature_key},
    )
    if not features:
        return train_store, validation, test, None, None, 0, [], target

    _, categorical_indices = prepare_generic_model_inputs(train_sample, features)
    validation_inputs, _ = prepare_generic_model_inputs(
        validation, features, categorical_indices
    )
    test_inputs, _ = prepare_generic_model_inputs(test, features, categorical_indices)

    task_type = _task_type_name(task)

    def train_batches() -> Iterator[pd.DataFrame]:
        for batch in train_store.iter_batches():
            inputs, _ = prepare_generic_model_inputs(
                batch, features, categorical_indices
            )
            if task_type == "binary_classification":
                inputs[target] = batch[target].astype("int8").to_numpy()
            else:
                inputs[target] = (
                    pd.to_numeric(batch[target], errors="coerce")
                    .fillna(0)
                    .astype("float64")
                    .to_numpy()
                )
            yield inputs

    backend = selected_model_backend()
    print("model_backend:", backend, flush=True)
    if task_type == "binary_classification":
        if train_store.target_nunique(target) < 2 or validation[target].nunique() < 2:
            return train_store, validation, test, None, None, len(features), [], target
        model, best_config, best_score = fit_tuned_classifier_incremental(
            train_batches,
            features,
            target,
            validation_inputs,
            validation[target].astype("int8"),
            batch_count=len(train_store.part_paths),
            cat_features=categorical_indices,
            model_backend=backend,
        )
        print("model_config:", best_config, flush=True)
        print("validation_model_score:", best_score, flush=True)
        validation_predictions = np.asarray(
            model.predict_proba(validation_inputs)[:, 1], dtype="float64"
        )
        test_predictions = np.asarray(
            model.predict_proba(test_inputs)[:, 1], dtype="float64"
        )
        validation_metrics = split_tasks["val"].evaluate(
            validation_predictions,
            target_table=target_table_from_frame(split_tasks["val"], validation),
        )
        test_metrics = split_tasks["test"].evaluate(
            test_predictions,
            target_table=target_table_from_frame(split_tasks["test"], test),
        )
    else:
        validation_target = (
            pd.to_numeric(validation[target], errors="coerce")
            .fillna(0)
            .astype("float64")
        )
        model, best_config, best_score = fit_tuned_regressor_incremental(
            train_batches,
            features,
            target,
            validation_inputs,
            validation_target,
            batch_count=len(train_store.part_paths),
            cat_features=categorical_indices,
            model_backend=backend,
        )
        print("model_config:", best_config, flush=True)
        print("validation_model_score:", best_score, flush=True)
        validation_predictions = np.asarray(
            model.predict(validation_inputs), dtype="float64"
        )
        test_predictions = np.asarray(model.predict(test_inputs), dtype="float64")
        validation_metrics = add_nmae(
            split_tasks["val"].evaluate(
                validation_predictions,
                target_table=target_table_from_frame(split_tasks["val"], validation),
            ),
            validation_target,
            validation_predictions,
            train_store.target_std(target),
        )
        test_metrics = add_nmae(
            split_tasks["test"].evaluate(
                test_predictions,
                target_table=target_table_from_frame(split_tasks["test"], test),
            ),
            test[target],
            test_predictions,
            train_store.target_std(target),
        )

    return (
        train_store,
        validation,
        test,
        validation_metrics,
        test_metrics,
        len(features),
        [],
        target,
    )
