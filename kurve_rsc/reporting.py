"""Benchmark result persistence and compact summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_result(result_root: Path, result: dict[str, Any]) -> Path:
    result_root.mkdir(parents=True, exist_ok=True)
    result = dict(result)
    result.setdefault("status", "completed")
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = result_root / f"{result['task_id'].replace('/', '-')}.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return path


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the aggregate report small while retaining task-level outcomes."""

    return {
        key: result[key]
        for key in (
            "task_id",
            "status",
            "task_type",
            "metric",
            "feature_count",
            "selected_config",
            "validation_metrics",
            "test_metrics",
            "error",
            "finished_at_utc",
        )
        if key in result
    }


def write_run_report(report_path: Path, results: list[dict[str, Any]]) -> Path:
    """Write a resumable aggregate report for completed and failed tasks."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    completed = sum(result.get("status") == "completed" for result in results)
    failed = sum(result.get("status") == "failed" for result in results)
    report = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_count": len(results),
        "completed_count": completed,
        "failed_count": failed,
        "tasks": [summarize_result(result) for result in results],
    }
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report_path


def load_results(result_root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(result_root.glob("*.json")):
        if path.name in {"official_results.json", "environment_manifest.json"}:
            continue
        results.append(json.loads(path.read_text()))
    return results
