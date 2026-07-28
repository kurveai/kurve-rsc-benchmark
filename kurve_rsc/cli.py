"""Package entry points delegating to the benchmark scripts."""

from __future__ import annotations

from scripts.run_all import main as run_all_cli
from scripts.run_task import main as run_task_cli

__all__ = ["run_all_cli", "run_task_cli"]
