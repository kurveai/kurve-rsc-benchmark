#!/usr/bin/env python3
"""Run the mirrored RelBench v1 task set and write progress reports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = PROJECT_ROOT / "kurve_rsc"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "run_reports"
SINGLE_TRAIN_PERIOD_ENV = "RELBENCH_SINGLE_TRAIN_PERIOD"

CLASSIFICATION_TASKS = (
    "relbench_amazon_user_churn.py",
    "relbench_amazon_item_churn.py",
    "relbench_avito_user_visits.py",
    "relbench_avito_user_clicks.py",
    "relbench_event_user_repeat.py",
    "relbench_event_user_ignore.py",
    "relbench_f1_driver_dnf.py",
    "relbench_f1_driver_top3.py",
    "relbench_hm_user_churn.py",
    "relbench_user_engagement_local_runner.py",
    "relbench_user_badges_local_runner.py",
    "relbench_trial_study_outcome.py",
)
REGRESSION_TASKS = (
    "relbench_amazon_user_ltv.py",
    "relbench_amazon_item_ltv.py",
    "relbench_avito_ad_ctr.py",
    "relbench_event_user_attendance.py",
    "relbench_f1_driver_position.py",
    "relbench_hm_item_sales.py",
    "relbench_post_votes_local_runner.py",
    "relbench_trial_study_adverse.py",
    "relbench_trial_site_success.py",
)
TASK_GROUPS = {
    "classification": CLASSIFICATION_TASKS,
    "regression": REGRESSION_TASKS,
    "v1": CLASSIFICATION_TASKS + REGRESSION_TASKS,
    "all": CLASSIFICATION_TASKS + REGRESSION_TASKS,
}


def worker_value(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"all", "max"}:
        return "0"
    try:
        workers = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workers must be an integer or 'all'") from exc
    if workers < 0:
        raise argparse.ArgumentTypeError("workers must be non-negative")
    return str(workers)


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name, "1" if default else "0").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-type", choices=sorted(TASK_GROUPS), default="v1")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--match", action="append", default=[])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--training-frame-workers",
        type=worker_value,
        default=worker_value(os.environ.get("RELBench_TRAINING_FRAME_WORKERS", "1")),
    )
    parser.add_argument(
        "--single-train-period",
        action=argparse.BooleanOptionalAction,
        default=_env_flag(SINGLE_TRAIN_PERIOD_ENV),
        help=(
            "Use only the latest official training cutoff for each task "
            "(validation timestamp minus that task's label period)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def selected_tasks(args: argparse.Namespace) -> list[str]:
    allowed = set(TASK_GROUPS[args.task_type])
    tasks = list(args.task) if args.task else list(TASK_GROUPS[args.task_type])
    unknown = set(tasks) - allowed
    if unknown:
        raise SystemExit(f"tasks outside {args.task_type}: {sorted(unknown)}")
    if args.match:
        tasks = [task for task in tasks if any(fragment.lower() in task.lower() for fragment in args.match)]
    return tasks


def run_task(task_name: str, args: argparse.Namespace, log_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    env = os.environ.copy()
    env["RELBench_TRAINING_FRAME_WORKERS"] = args.training_frame_workers
    env[SINGLE_TRAIN_PERIOD_ENV] = "1" if args.single_train_period else "0"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(TASK_DIR), str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    process = subprocess.Popen(
        [args.python, str(TASK_DIR / task_name)],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    highlights: list[str] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            stripped = line.strip()
            if args.stream_output:
                print(line, end="", flush=True)
            elif any(
                stripped.startswith(prefix)
                for prefix in (
                    "single_train_cut_date:",
                    "validation_metrics:",
                    "test_metrics:",
                    "validation_nmae:",
                    "test_nmae:",
                    "feature_count:",
                    "num_features:",
                )
            ):
                highlights.append(stripped)
    return {
        "task_id": task_name.removesuffix(".py"),
        "status": "passed" if process.wait() == 0 else "failed",
        "return_code": process.returncode,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "started_at_utc": started_at,
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "highlights": highlights,
    }


def write_report(
    results: list[dict[str, object]],
    output_dir: Path,
    task_type: str,
    single_train_period: bool = False,
) -> None:
    payload = {
        "task_type": task_type,
        "single_train_period": single_train_period,
        "task_count": len(results),
        "passed_count": sum(result["status"] == "passed" for result in results),
        "failed_count": sum(result["status"] == "failed" for result in results),
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "relbench_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Kurve-RSC RelBench Results",
        "",
        f"- Task type: `{task_type}`",
        f"- Single train period: `{single_train_period}`",
        f"- Passed: `{payload['passed_count']}`",
        f"- Failed: `{payload['failed_count']}`",
        "",
        "| Task | Status | Duration (s) | Highlights | Log |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| `{result['task_id']}` | `{result['status']}` | {result['duration_seconds']} | "
            f"{'<br>'.join(result['highlights']) or 'n/a'} | `{result['log_path']}` |"
        )
    (output_dir / "relbench_results.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    tasks = selected_tasks(args)
    if not tasks:
        raise SystemExit("no tasks matched")
    if args.dry_run:
        print("\n".join(tasks))
        return 0

    print(f"Running {len(tasks)} Kurve-RSC RelBench v1 task(s)", flush=True)
    print(f"Training frame workers: {args.training_frame_workers}", flush=True)
    print(f"Single train period: {args.single_train_period}", flush=True)
    results: list[dict[str, object]] = []
    for index, task_name in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] starting {task_name}", flush=True)
        result = run_task(task_name, args, args.output_dir / "logs" / f"{Path(task_name).stem}.log")
        results.append(result)
        print(
            f"[{index}/{len(tasks)}] finished {task_name} | status={result['status']} "
            f"duration={result['duration_seconds']}s | {'; '.join(result['highlights'])}",
            flush=True,
        )
        if args.stop_on_error and result["status"] == "failed":
            break
    write_report(results, args.output_dir, args.task_type, args.single_train_period)
    return 0 if all(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
