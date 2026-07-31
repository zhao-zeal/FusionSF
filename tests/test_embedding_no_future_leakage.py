import pandas as pd

from src.datasets.split_utils import build_target_time_windows


def test_ts_input_ends_before_forecast_for_every_split():
    timestamps = pd.date_range("2024-01-01", periods=200, freq="h")
    records = build_target_time_windows(timestamps, 24, 24, 0.6, 0.2, 0.2)
    assert records
    assert all(record.input_end < record.forecast_start for record in records)
