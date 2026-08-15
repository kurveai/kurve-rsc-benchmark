from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
from relbench.base import Table

from kurve_rsc.relbench_adapter import (
    SINGLE_TRAIN_PERIOD_ENV,
    _select_single_timestamp_table,
    _task_split_timestamp,
    single_train_period_enabled,
)
from scripts.run_all import parse_args as parse_all_args, write_report
from scripts.run_task import parse_args as parse_task_args


def _task_table(timestamps: list[str]) -> Table:
    return Table(
        df=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps),
                "entity": range(len(timestamps)),
                "target": range(len(timestamps)),
            }
        ),
        fkey_col_to_pkey_table={},
        pkey_col=None,
        time_col="timestamp",
    )


def test_single_train_cutoff_is_one_task_label_period_before_validation():
    task = SimpleNamespace(
        dataset=SimpleNamespace(val_timestamp=pd.Timestamp("2020-04-01")),
        timedelta=pd.Timedelta(days=30),
    )

    cutoff = _task_split_timestamp(task, "train")
    selected, selected_timestamp = _select_single_timestamp_table(
        _task_table(["2020-01-02", "2020-02-01", "2020-03-02"]),
        "timestamp",
        "train",
        cutoff,
    )

    assert cutoff == pd.Timestamp("2020-03-02")
    assert selected_timestamp == cutoff
    assert selected.df["timestamp"].nunique() == 1
    assert selected.df["timestamp"].iloc[0] == cutoff


def test_single_train_period_can_be_configured_by_environment(monkeypatch):
    monkeypatch.setenv(SINGLE_TRAIN_PERIOD_ENV, "true")
    assert single_train_period_enabled() is True

    monkeypatch.setenv(SINGLE_TRAIN_PERIOD_ENV, "off")
    assert single_train_period_enabled() is False


def test_single_train_period_is_available_on_both_clis():
    assert parse_task_args(
        ["relbench_event_user_ignore.py", "--single-train-period"]
    ).single_train_period
    assert parse_all_args(["--single-train-period"]).single_train_period
    assert not parse_all_args(
        ["--single-train-period", "--no-single-train-period"]
    ).single_train_period


def test_top_level_report_records_single_train_period(tmp_path):
    write_report(
        [],
        tmp_path,
        "v1",
        single_train_period=True,
        model_backend="tabpfn",
    )

    payload = json.loads((tmp_path / "relbench_results.json").read_text())
    assert payload["single_train_period"] is True
    assert payload["model_backend"] == "tabpfn"
