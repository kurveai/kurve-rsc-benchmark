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
SINGLE_TRAIN_PERIOD_ENV = "RELBENCH_SINGLE_TRAIN_PERIOD"
MODEL_BACKEND_ENV = "KURVE_RSC_MODEL_BACKEND"
TRAIN_ALL_AT_ONCE_ENV = "KURVE_RSC_TRAIN_ALL_AT_ONCE"
FEATURE_MANIFEST_ENV = "KURVE_RSC_FEATURE_MANIFEST"


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name, "1" if default else "0").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Task script name, with or without .py")
    parser.add_argument(
        "--training-frame-workers",
        default=os.environ.get("RELBench_TRAINING_FRAME_WORKERS", "1"),
        help="Concurrent training-frame workers, or 'all'.",
    )
    parser.add_argument(
        "--single-train-period",
        action=argparse.BooleanOptionalAction,
        default=_env_flag(SINGLE_TRAIN_PERIOD_ENV),
        help=(
            "Use only the latest official training cutoff: the validation "
            "timestamp minus this task's label period."
        ),
    )
    parser.add_argument(
        "--tabpfn",
        action="store_true",
        help=(
            "Use TabPFNClassifier or TabPFNRegressor instead of CatBoost, "
            "according to the selected task type."
        ),
    )
    parser.add_argument(
        "--train-all-at-once",
        action=argparse.BooleanOptionalAction,
        default=_env_flag(TRAIN_ALL_AT_ONCE_ENV),
        help=(
            "Materialize all training frames and fit CatBoost jointly instead "
            "of continuing the model one frame at a time."
        ),
    )
    parser.add_argument(
        "--feature-manifest",
        action=argparse.BooleanOptionalAction,
        default=_env_flag(FEATURE_MANIFEST_ENV),
        help=(
            "Fit and apply GraphReduce feature manifests for rel-f1, "
            "rel-event, and rel-trial tasks. Other datasets are unchanged."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    task_name = args.task if args.task.endswith(".py") else f"{args.task}.py"
    task_path = TASK_DIR / task_name
    if not task_path.is_file():
        parser.error(f"unknown task script: {task_name}")
    os.environ["RELBench_TRAINING_FRAME_WORKERS"] = str(args.training_frame_workers)
    os.environ[SINGLE_TRAIN_PERIOD_ENV] = "1" if args.single_train_period else "0"
    os.environ[MODEL_BACKEND_ENV] = "tabpfn" if args.tabpfn else "catboost"
    os.environ[TRAIN_ALL_AT_ONCE_ENV] = "1" if args.train_all_at_once else "0"
    os.environ[FEATURE_MANIFEST_ENV] = "1" if args.feature_manifest else "0"
    sys.path.insert(0, str(TASK_DIR))
    sys.argv = [str(task_path)]
    runpy.run_path(str(task_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
