import pandas as pd
from relbench.base import Table

from kurve_rsc.relbench_adapter import _select_single_timestamp_table


def test_single_timestamp_frames_are_filtered_to_the_requested_cutoff():
    table = Table(
        df=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2020-01-01", "2020-01-02", "2020-01-02"]
                ),
                "entity": [1, 1, 2],
                "target": [0, 1, 0],
            }
        ),
        fkey_col_to_pkey_table={},
        pkey_col=None,
        time_col="timestamp",
    )

    selected, timestamp = _select_single_timestamp_table(
        table, "timestamp", "train", pd.Timestamp("2020-01-02")
    )

    assert timestamp == pd.Timestamp("2020-01-02")
    assert selected.df["timestamp"].nunique() == 1
    assert len(selected.df) == 2
