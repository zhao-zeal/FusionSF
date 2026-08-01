from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SUPPORTED_HORIZONS = {1, 4, 16, 72, 288}


@dataclass(frozen=True)
class SolarDatasetSpec:
    dataset: str
    site_id: str
    csv_path: Path
    timestamp_col: str
    power_col: str
    nwp_cols: tuple[str, ...]
    frequency: str
    capacity: Optional[float] = None


@dataclass(frozen=True)
class TrainScaler:
    power_mean: float
    power_std: float
    nwp_mean: np.ndarray
    nwp_std: np.ndarray


def load_solar_frame(spec: SolarDatasetSpec) -> pd.DataFrame:
    frame = pd.read_csv(spec.csv_path)
    required = [spec.timestamp_col, spec.power_col, *spec.nwp_cols]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns in {spec.csv_path}: {missing}")
    frame = frame[required].copy()
    frame[spec.timestamp_col] = pd.to_datetime(frame[spec.timestamp_col], errors="raise")
    frame = frame.sort_values(spec.timestamp_col).reset_index(drop=True)
    if frame[spec.timestamp_col].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {spec.csv_path}")
    expected = pd.Timedelta(spec.frequency)
    deltas = frame[spec.timestamp_col].diff().dropna()
    if not deltas.eq(expected).all():
        raise ValueError(f"Irregular timestamps in {spec.csv_path}: {deltas.value_counts().to_dict()}")
    frame[spec.power_col] = frame[spec.power_col].clip(lower=0)
    if spec.capacity is not None:
        frame[spec.power_col] = frame[spec.power_col].clip(upper=spec.capacity)
    return frame


def fit_train_scaler(frame: pd.DataFrame, spec: SolarDatasetSpec, train_end: int) -> TrainScaler:
    """Fit exactly once on the training-time slice; callers only transform later splits."""
    train = frame.iloc[:train_end]
    power = train[spec.power_col].to_numpy(np.float32)
    nwp = train[list(spec.nwp_cols)].to_numpy(np.float32) if spec.nwp_cols else np.empty((len(train), 0), np.float32)
    power_mean = float(np.nanmean(power))
    power_std = float(np.nanstd(power))
    if not np.isfinite(power_mean) or not np.isfinite(power_std):
        raise ValueError("Training power is entirely missing")
    power_std = max(power_std, 1e-6)
    if nwp.shape[1]:
        nwp_mean = np.nanmean(nwp, axis=0).astype(np.float32)
        nwp_std = np.nanstd(nwp, axis=0).astype(np.float32)
        if not np.isfinite(nwp_mean).all() or not np.isfinite(nwp_std).all():
            raise ValueError("A training NWP feature is entirely missing")
        nwp_std[nwp_std < 1e-6] = 1.0
    else:
        nwp_mean = np.empty(0, np.float32)
        nwp_std = np.empty(0, np.float32)
    return TrainScaler(power_mean, power_std, nwp_mean, nwp_std)


def build_sample_manifest(
    frame: pd.DataFrame,
    spec: SolarDatasetSpec,
    seq_len: int,
    pred_len: int,
    reference_test_origins: Optional[np.ndarray] = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.1,
) -> pd.DataFrame:
    if seq_len <= 0 or pred_len not in SUPPORTED_HORIZONS:
        raise ValueError(f"Unsupported seq_len/pred_len: {seq_len}/{pred_len}")
    n = len(frame)
    bounds = {"train": (0, int(n * train_ratio)), "validation": (int(n * train_ratio), int(n * (train_ratio + val_ratio))), "test": (int(n * (train_ratio + val_ratio)), n)}
    timestamps = frame[spec.timestamp_col]
    reference = None
    if reference_test_origins is not None:
        reference = set(pd.to_datetime(reference_test_origins).to_numpy(dtype="datetime64[ns]"))
    rows = []
    values = frame[[spec.power_col, *spec.nwp_cols]].to_numpy(np.float32)
    for split, (left, right) in bounds.items():
        for start in range(left, right - seq_len - pred_len + 1):
            input_end = start + seq_len - 1
            forecast_start = start + seq_len
            forecast_end = forecast_start + pred_len - 1
            if reference is not None and split == "test" and timestamps.iloc[input_end].to_datetime64() not in reference:
                continue
            power_ok = np.isfinite(values[start : forecast_end + 1, 0]).all()
            nwp_ok = not spec.nwp_cols or np.isfinite(values[forecast_start : forecast_end + 1, 1:]).all()
            if not (power_ok and nwp_ok):
                continue
            rows.append({
                "dataset": spec.dataset, "site_id": spec.site_id, "split": split,
                "input_start_timestamp": timestamps.iloc[start], "input_end_timestamp": timestamps.iloc[input_end],
                "forecast_start_timestamp": timestamps.iloc[forecast_start], "forecast_end_timestamp": timestamps.iloc[forecast_end],
                "seq_len": seq_len, "pred_len": pred_len, "start_index": start,
            })
    manifest = pd.DataFrame(rows)
    for split in ("train", "validation", "test"):
        target_ranges = manifest.loc[manifest.split == split, ["forecast_start_timestamp", "forecast_end_timestamp"]]
        if len(target_ranges) and target_ranges.forecast_end_timestamp.max() < target_ranges.forecast_start_timestamp.min():
            raise AssertionError("Invalid target range")
    return manifest


class SolarWindowDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, manifest: pd.DataFrame, spec: SolarDatasetSpec, scaler: TrainScaler, mode: str):
        if mode not in {"power", "power_nwp"}:
            raise ValueError(f"Unknown mode: {mode}")
        if mode == "power_nwp" and not spec.nwp_cols:
            raise ValueError("Power+NWP requires NWP columns")
        self.frame, self.manifest, self.spec, self.scaler, self.mode = frame, manifest.reset_index(drop=True), spec, scaler, mode

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.manifest.iloc[index]
        start, seq_len, pred_len = int(row.start_index), int(row.seq_len), int(row.pred_len)
        future_start = start + seq_len
        power = self.frame[self.spec.power_col].to_numpy(np.float32)
        history = (power[start:future_start] - self.scaler.power_mean) / self.scaler.power_std
        target = (power[future_start:future_start + pred_len] - self.scaler.power_mean) / self.scaler.power_std
        item: dict[str, torch.Tensor | str] = {
            "history_power": torch.from_numpy(history[:, None].copy()),
            "target_power": torch.from_numpy(target[:, None].copy()),
            "target_power_raw": torch.from_numpy(power[future_start:future_start + pred_len, None].copy()),
            "forecast_start_timestamp": str(row.forecast_start_timestamp),
        }
        if self.mode == "power_nwp":
            nwp = self.frame[list(self.spec.nwp_cols)].to_numpy(np.float32)[future_start:future_start + pred_len]
            item["future_nwp"] = torch.from_numpy(((nwp - self.scaler.nwp_mean) / self.scaler.nwp_std).copy())
        return item
