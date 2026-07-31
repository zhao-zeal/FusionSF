import pandas as pd

from src.datasets.split_utils import build_target_time_windows


def test_target_timestamps_are_disjoint_and_history_can_cross_boundary():
    timestamps = pd.date_range("2024-01-01", periods=100, freq="h")
    records = build_target_time_windows(timestamps, 24, 6, 0.6, 0.2, 0.2)
    by_split = {name: [r for r in records if r.split == name] for name in ("train", "validation", "test")}
    assert all(by_split.values())
    assert by_split["validation"][0].input_start < timestamps[60]
    target_sets = {}
    for name, rows in by_split.items():
        target_sets[name] = {
            value for row in rows
            for value in timestamps[row.start_index + 24 : row.start_index + 30]
        }
    assert target_sets["train"].isdisjoint(target_sets["validation"])
    assert target_sets["train"].isdisjoint(target_sets["test"])
    assert target_sets["validation"].isdisjoint(target_sets["test"])
