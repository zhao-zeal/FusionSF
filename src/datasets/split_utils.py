"""Leakage-safe chronological split utilities for pipeline_v1_fixed."""

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowRecord:
    start_index: int
    split: str
    input_start: pd.Timestamp
    input_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp


def chronological_boundaries(
    timestamps: Iterable, train_ratio: float, valid_ratio: float, test_ratio: float
) -> Dict[str, tuple]:
    """Return half-open target-time ranges; ratios must cover the full timeline."""
    ts = pd.DatetimeIndex(timestamps)
    if len(ts) < 3 or not ts.is_monotonic_increasing or ts.has_duplicates:
        raise ValueError("timestamps must be unique, increasing, and contain at least 3 values")
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1")
    train_end = int(len(ts) * train_ratio)
    valid_end = int(len(ts) * (train_ratio + valid_ratio))
    if not (0 < train_end < valid_end < len(ts)):
        raise ValueError("split ratios produce an empty partition")
    terminal = ts[-1] + (ts[-1] - ts[-2])
    return {
        "train": (ts[0], ts[train_end]),
        "validation": (ts[train_end], ts[valid_end]),
        "test": (ts[valid_end], terminal),
    }


def build_target_time_windows(
    timestamps: Iterable,
    seq_len: int,
    pred_len: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    stride: int = 1,
) -> List[WindowRecord]:
    """Assign a window only when its complete forecast target lies in one split."""
    ts = pd.DatetimeIndex(timestamps)
    if seq_len <= 0 or pred_len <= 0 or stride <= 0:
        raise ValueError("seq_len, pred_len, and stride must be positive")
    bounds = chronological_boundaries(ts, train_ratio, valid_ratio, test_ratio)
    records: List[WindowRecord] = []
    for start in range(0, len(ts) - seq_len - pred_len + 1, stride):
        forecast_start_idx = start + seq_len
        forecast_end_idx = forecast_start_idx + pred_len - 1
        forecast_start, forecast_end = ts[forecast_start_idx], ts[forecast_end_idx]
        split = None
        for name, (lower, upper) in bounds.items():
            if forecast_start >= lower and forecast_end < upper:
                split = name
                break
        if split is not None:
            records.append(
                WindowRecord(start, split, ts[start], ts[start + seq_len - 1], forecast_start, forecast_end)
            )
    assert_disjoint_target_times(records, ts, seq_len, pred_len)
    return records


def assert_disjoint_target_times(
    records: Iterable[WindowRecord], timestamps: Iterable, seq_len: int, pred_len: int
) -> None:
    """Assert that no target timestamp is owned by more than one split."""
    ts = pd.DatetimeIndex(timestamps)
    owned = {"train": set(), "validation": set(), "test": set()}
    for record in records:
        begin = record.start_index + seq_len
        owned[record.split].update(ts[begin : begin + pred_len].asi8.tolist())
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for left, right in pairs:
        overlap = owned[left].intersection(owned[right])
        if overlap:
            raise AssertionError(f"target timestamp overlap between {left} and {right}: {len(overlap)}")
