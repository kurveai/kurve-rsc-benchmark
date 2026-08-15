from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kurve_rsc"))

from relbench_stack_task_utils import _select_randomly_spaced_timestamps


def test_stack_timestamp_sample_is_reproducible_and_spans_schedule() -> None:
    timestamps = list(range(46))

    selected = _select_randomly_spaced_timestamps(timestamps, 15)

    assert selected == _select_randomly_spaced_timestamps(timestamps, 15)
    assert len(selected) == 15
    assert len(set(selected)) == 15
    assert selected == sorted(selected)
    assert selected[-1] == timestamps[-1]

    historical_count = len(timestamps) - 1
    sample_count = 15 - 1
    for sample_index, timestamp in enumerate(selected[:-1]):
        start = sample_index * historical_count // sample_count
        stop = (sample_index + 1) * historical_count // sample_count
        assert start <= timestamp < stop


def test_stack_timestamp_sample_handles_short_schedules_and_one_frame() -> None:
    assert _select_randomly_spaced_timestamps([1, 2, 3], 15) == [1, 2, 3]
    assert _select_randomly_spaced_timestamps([1, 2, 3], 1) == [3]

    with pytest.raises(ValueError, match="positive"):
        _select_randomly_spaced_timestamps([1, 2, 3], 0)
