from __future__ import annotations

import json
import os
from types import SimpleNamespace

from kurve_rsc.relbench_feature_policy import (
    BASELINE_FEATURE_FAMILY_ENV,
    apply_feature_family_policy,
    configure_task_cli,
)
from scripts.run_all import parse_args as parse_all_args, write_report
from scripts.run_task import parse_args as parse_task_args


def test_default_policy_preserves_configured_feature_families(monkeypatch):
    monkeypatch.delenv(BASELINE_FEATURE_FAMILY_ENV, raising=False)
    node = SimpleNamespace(feature_families=("base", "temporal", "context"))

    apply_feature_family_policy([node])

    assert node.feature_families == ("base", "temporal", "context")


def test_baseline_policy_forces_base_family_on_every_node(monkeypatch):
    monkeypatch.setenv(BASELINE_FEATURE_FAMILY_ENV, "1")
    nodes = [
        SimpleNamespace(feature_families=("base", "temporal")),
        SimpleNamespace(feature_families=("semantic", "context")),
    ]

    apply_feature_family_policy(nodes)

    assert [node.feature_families for node in nodes] == [("base",), ("base",)]


def test_baseline_is_available_on_orchestrator_and_direct_task_clis(monkeypatch):
    monkeypatch.delenv(BASELINE_FEATURE_FAMILY_ENV, raising=False)

    assert parse_all_args(["--baseline"]).baseline is True
    assert parse_task_args(["relbench_event_user_repeat.py", "--baseline"]).baseline is True
    assert parse_all_args(["--baseline", "--no-baseline"]).baseline is False


def test_direct_task_cli_publishes_feature_family_mode(monkeypatch, capsys):
    monkeypatch.delenv(BASELINE_FEATURE_FAMILY_ENV, raising=False)

    args = configure_task_cli(argv=["--baseline"])

    assert args.baseline is True
    assert os.environ[BASELINE_FEATURE_FAMILY_ENV] == "1"
    assert "feature_family_mode: baseline" in capsys.readouterr().out


def test_top_level_report_records_baseline_mode(tmp_path):
    write_report([], tmp_path, "v1", baseline=True)

    payload = json.loads((tmp_path / "relbench_results.json").read_text())
    assert payload["baseline"] is True
    assert "Baseline feature family only: `True`" in (
        tmp_path / "relbench_results.md"
    ).read_text()
