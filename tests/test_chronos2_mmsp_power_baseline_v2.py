from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_chronos2_mmsp_power_baseline_v2 as baseline  # noqa: E402


def write_source(directory: Path, *, break_boundary: bool = False) -> None:
    count, context, horizon = 4, 24, 24
    input_end = np.arange(count, dtype=np.int64) * 48 * baseline.ONE_HOUR_NS
    forecast_start = input_end + baseline.ONE_HOUR_NS
    if break_boundary:
        forecast_start[0] += baseline.ONE_HOUR_NS
    forecast_ts = forecast_start[:, None] + np.arange(horizon) * baseline.ONE_HOUR_NS
    arrays = {
        "inputs.npy": np.ones((count, context, 1), dtype=np.float32),
        "targets.npy": np.ones((count, horizon, 1), dtype=np.float32),
        "site_ids.npy": np.array([0, 0, 1, 1], dtype=np.int64),
        "input_end_timestamps.npy": input_end,
        "forecast_start_timestamps.npy": forecast_start,
        "forecast_end_timestamps.npy": forecast_ts[:, -1],
        "timestamps.npy": forecast_ts,
    }
    for name, value in arrays.items():
        np.save(directory / name, value)


def test_load_aligned_windows_accepts_complete_hourly_manifest(tmp_path):
    write_source(tmp_path)
    arrays, paths = baseline.load_aligned_windows(tmp_path)
    summary = baseline.summarize_source(arrays, paths)
    assert arrays["inputs"].shape == (4, 24, 1)
    assert arrays["forecast_timestamps"].shape == (4, 24)
    assert summary["site_ids"] == [0, 1]
    assert set(summary["source_sha256"]) == set(paths)


def test_load_aligned_windows_rejects_target_boundary_gap(tmp_path):
    write_source(tmp_path, break_boundary=True)
    with pytest.raises(ValueError, match="exactly one hour"):
        baseline.load_aligned_windows(tmp_path)


def test_load_aligned_windows_requires_site_and_timing_metadata(tmp_path):
    np.save(tmp_path / "inputs.npy", np.ones((2, 24, 1), dtype=np.float32))
    np.save(tmp_path / "targets.npy", np.ones((2, 24, 1), dtype=np.float32))
    with pytest.raises(FileNotFoundError):
        baseline.load_aligned_windows(tmp_path)
