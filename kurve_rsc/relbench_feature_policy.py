"""Shared CLI and graph policy for Kurve RSC feature-family modes."""

from __future__ import annotations

import argparse
import os
from typing import Any, Iterable, Sequence


BASELINE_FEATURE_FAMILY_ENV = "KURVE_RSC_BASELINE_FEATURE_FAMILY"
BASELINE_FEATURE_FAMILIES = ("base",)


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name, "1" if default else "0").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def baseline_feature_family_enabled(override: bool | None = None) -> bool:
    """Return whether every graph node must use only the base family."""

    if override is not None:
        return bool(override)
    return _env_flag(BASELINE_FEATURE_FAMILY_ENV)


def feature_family_mode() -> str:
    """Return the user-facing name of the active feature-family mode."""

    return "baseline" if baseline_feature_family_enabled() else "configured"


def apply_feature_family_policy(
    nodes: Iterable[Any],
    *,
    baseline: bool | None = None,
) -> None:
    """Apply a top-level feature-family override without changing defaults."""

    if not baseline_feature_family_enabled(baseline):
        return
    for node in nodes:
        node.feature_families = BASELINE_FEATURE_FAMILIES


def configure_task_cli(
    *,
    description: str | None = None,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the common direct-task CLI and publish its process-wide policy."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--baseline",
        action=argparse.BooleanOptionalAction,
        default=baseline_feature_family_enabled(),
        help=(
            "Use only the baseline 'base' feature family on every graph node. "
            "The default preserves each task's configured feature families."
        ),
    )
    args = parser.parse_args(argv)
    os.environ[BASELINE_FEATURE_FAMILY_ENV] = "1" if args.baseline else "0"
    print("feature_family_mode:", feature_family_mode(), flush=True)
    return args
