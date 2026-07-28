#!/usr/bin/env python3
"""Run one mirrored Kurve-RSC RelBench task."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = PROJECT_ROOT / "kurve_rsc"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Task script name, with or without .py")
    parser.add_argument(
        "--training-frame-workers",
        default=os.environ.get("RELBench_TRAINING_FRAME_WORKERS", "1"),
        help="Concurrent training-frame workers, or 'all'.",
    )
    args = parser.parse_args()

    task_name = args.task if args.task.endswith(".py") else f"{args.task}.py"
    task_path = TASK_DIR / task_name
    if not task_path.is_file():
        parser.error(f"unknown task script: {task_name}")

    os.environ["RELBench_TRAINING_FRAME_WORKERS"] = str(args.training_frame_workers)
    sys.path.insert(0, str(TASK_DIR))
    sys.argv = [str(task_path)]
    runpy.run_path(str(task_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
