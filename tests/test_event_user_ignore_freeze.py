from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


TASK_DIR = Path(__file__).resolve().parents[1] / "kurve_rsc"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from relbench_event_user_ignore import (  # noqa: E402
    _FrozenGraphOperations,
    _catboost_inputs,
    _columns_present_in_all_frames,
)


class FakeGraph:
    def __init__(self) -> None:
        self.frozen_plan = {"records": [{"method_name": "do_reduce"}]}
        self.applied_plans: list[dict[str, object]] = []

    def freeze_execution_plan(self) -> dict[str, object]:
        return self.frozen_plan

    def apply_execution_plan(self, plan: dict[str, object]) -> None:
        self.applied_plans.append(plan)


def test_first_graph_plan_is_replayed_without_freezing_feature_schema() -> None:
    operations = _FrozenGraphOperations()
    first_graph = FakeGraph()
    first_features = pd.DataFrame({"entity": [1], "feature": [2.0]})

    operations.apply(first_graph)
    assert first_graph.applied_plans == []
    operations.capture(
        first_graph,
        first_features,
        split_name="train",
        cut_date=pd.Timestamp("2020-01-01"),
    )

    later_graph = FakeGraph()
    operations.apply(later_graph)
    assert later_graph.applied_plans == [first_graph.frozen_plan]
    operations.capture(
        later_graph,
        pd.DataFrame({"entity": [1], "other_feature": [3.0]}),
        split_name="test",
        cut_date=pd.Timestamp("2020-03-01"),
    )


def test_model_columns_are_present_in_every_dataframe() -> None:
    frames = [
        pd.DataFrame({"first": [1], "shared": [2], "train_only": [3]}),
        pd.DataFrame({"first": [4], "shared": [5], "later_train_only": [6]}),
        pd.DataFrame({"first": [7], "shared": [8], "validation_only": [9]}),
        pd.DataFrame({"first": [10], "shared": [11], "test_only": [12]}),
    ]

    assert _columns_present_in_all_frames(frames) == ["first", "shared"]


def test_training_categorical_layout_is_frozen_for_later_splits() -> None:
    features = ["category", "amount"]
    training = pd.DataFrame(
        {
            "category": pd.Series(["a", "b"], dtype="object"),
            "amount": [1.0, 2.0],
        }
    )
    _, categorical_indices = _catboost_inputs(training, features)
    assert categorical_indices == [0]

    later = pd.DataFrame(
        {
            "category": [1, 2],
            "amount": pd.Series(["3.0", "not-numeric"], dtype="object"),
        }
    )
    transformed, replayed_indices = _catboost_inputs(
        later,
        features,
        categorical_indices,
    )

    assert replayed_indices == categorical_indices
    assert transformed["category"].tolist() == ["1", "2"]
    assert transformed["amount"].tolist() == [3.0, 0.0]
