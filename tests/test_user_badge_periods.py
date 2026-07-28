from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kurve_rsc"))

import relbench_stack_task_utils as stack_utils


class _FakeConnection:
    def sql(self, _query: str):
        return self

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame()


class _FakeGraphReduce:
    instances = []

    def __init__(self, *, parent_node, **_kwargs):
        self.parent_node = parent_node
        self.nodes = []
        self.instances.append(self)

    def add_node(self, node) -> None:
        self.nodes.append(node)

    def add_entity_edge(self, *_args, **_kwargs) -> None:
        pass

    def do_transformations_sql(self) -> None:
        self.parent_node._cur_data_ref = "user_badge_features"

    def _clean_refs(self) -> None:
        pass


def test_user_badge_temporal_periods_extend_to_ten_years(monkeypatch):
    _FakeGraphReduce.instances.clear()
    monkeypatch.setattr(stack_utils, "GraphReduce", _FakeGraphReduce)

    stack_utils.build_user_badge_features(
        _FakeConnection(),
        datetime.datetime(2020, 1, 1),
    )

    nodes = _FakeGraphReduce.instances[0].nodes
    expected = [7, 30, 90, 180, 365, 730, 1825, 3650]
    assert len(nodes) == 9
    assert all(node.ts_periods == expected for node in nodes)
    assert len({id(node.ts_periods) for node in nodes}) == len(nodes)
