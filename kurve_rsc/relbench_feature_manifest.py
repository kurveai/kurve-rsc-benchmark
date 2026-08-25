"""Shared opt-in GraphReduce feature-manifest support for RelBench tasks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

import duckdb
import pandas as pd

from graphreduce.node import DuckdbNode


FEATURE_MANIFEST_ENV = "KURVE_RSC_FEATURE_MANIFEST"
FEATURE_MANIFEST_SAMPLE_ROWS = 10_000
FEATURE_MANIFEST_TASKS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class FeatureManifestSource:
    """Training-sample metadata needed to profile one relational source."""

    view: str
    date_column: str | None = None
    foreign_keys: tuple[str, ...] = ()
    excluded_columns: tuple[str, ...] = ()
    unsafe_columns: tuple[str, ...] = ()


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name, "1" if default else "0").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def feature_manifest_enabled(task_name: str | None = None) -> bool:
    """Return whether the opt-in is requested and enabled for this task."""

    requested = _env_flag(FEATURE_MANIFEST_ENV)
    if task_name is None:
        return requested
    return requested and task_name in FEATURE_MANIFEST_TASKS


def _quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def load_feature_manifest_samples(
    con: duckdb.DuckDBPyConnection,
    sources: Mapping[str, FeatureManifestSource],
    validation_cutoff: pd.Timestamp,
    *,
    sample_rows: int = FEATURE_MANIFEST_SAMPLE_ROWS,
) -> dict[str, pd.DataFrame]:
    """Load one frozen pre-validation profiling sample per source."""

    if sample_rows < 1:
        raise ValueError("sample_rows must be positive")

    samples: dict[str, pd.DataFrame] = {}
    for source_name, source in sources.items():
        view = _quote_identifier(source.view)
        if source.date_column is None:
            query = f"SELECT * FROM {view} LIMIT {sample_rows}"
            samples[source_name] = con.sql(query).to_df()
            continue

        date_column = _quote_identifier(source.date_column)
        query = (
            f"SELECT * FROM {view} "
            f"WHERE {date_column} < ? LIMIT {sample_rows}"
        )
        samples[source_name] = con.execute(
            query,
            [pd.Timestamp(validation_cutoff).to_pydatetime()],
        ).fetchdf()
    return samples


def apply_feature_manifests(
    nodes: Mapping[str, DuckdbNode],
    sources: Mapping[str, FeatureManifestSource],
    samples: Mapping[str, pd.DataFrame],
    *,
    node_sources: Mapping[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    """Apply frozen source schemas and bounded automatic annotations."""

    summary: dict[str, dict[str, int]] = {}
    for node_name, node in nodes.items():
        source_name = (
            node_name if node_sources is None else node_sources[node_name]
        )
        source = sources[source_name]
        manifest = node.infer_feature_manifest(
            samples[source_name],
            foreign_keys=source.foreign_keys,
            excluded_columns=source.excluded_columns,
            unsafe_columns=source.unsafe_columns,
            apply_columns=True,
        )
        node.auto_annotate_features = True
        node.auto_text_features = True
        node.auto_annotate_max_categorical_columns = 8
        node.auto_annotate_max_gated_numeric_cols = 4
        node.auto_annotate_gated_numeric_top_k = 5
        summary[node_name] = {
            "source": len(manifest.columns),
            "graph": len(manifest.graph_columns),
            "features": len(manifest.feature_columns),
        }
    return summary
