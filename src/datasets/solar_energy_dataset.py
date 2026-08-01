from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SUPPORTED_HORIZONS = {1, 4, 16, 72, 288}
ERA5_COLUMNS = (
    "temperature_2m", "dew_point_2m", "surface_pressure", "shortwave_radiation",
    "cloud_cover", "wind_speed_10m", "precipitation", "relative_humidity_2m",
)


@dataclass(frozen=True)
class CSGSchema:
    site_id: str
    columns: tuple[str, ...]
    timestamp_position: int = 0
    power_position: int = -1
    status: str = "confirmed_from_repository_header_and_loader"


CSG_SCHEMAS = {
    "site1": CSGSchema("site1", ("date", "total_irr", "dni", "ghi", "temperature", "pressure", "power")),
    "site2": CSGSchema("site2", ("date", "total_irr", "dni", "ghi", "temperature", "pressure", "power")),
    "site3": CSGSchema("site3", ("date", "total_irr", "dni", "ghi", "pressure", "humidity", "power")),
    "site4": CSGSchema("site4", ("date", "total_irr", "dni", "ghi", "temperature", "pressure", "humidity", "power")),
    "site5": CSGSchema("site5", ("date", "total_irr", "dni", "ghi", "temperature", "pressure", "humidity", "power")),
    "site6": CSGSchema("site6", ("date", "total_irr", "dni", "ghi", "temperature", "pressure", "humidity", "power")),
    "site7": CSGSchema("site7", ("date", "total_irr", "dni", "ghi", "temperature", "pressure", "humidity", "power")),
    "site8": CSGSchema("site8", ("date", "total_irr", "dni", "ghi", "temperature", "pressure", "humidity", "power")),
}


@dataclass(frozen=True)
class SolarDatasetSpec:
    dataset: str
    site_id: str
    csv_path: Path
    timestamp_col: str
    power_col: str
    weather_cols: tuple[str, ...]
    frequency: str
    capacity: Optional[float] = None
    covariate_semantics: str = "none"
    weather_path: Optional[Path] = None
    source_timezone: Optional[str] = None
    loader_kind: str = "standard"
    allow_irregular_windows: bool = False

    @property
    def nwp_cols(self) -> tuple[str, ...]:
        """Compatibility alias; callers should prefer weather_cols."""
        return self.weather_cols


@dataclass(frozen=True)
class TrainScaler:
    power_mean: float
    power_std: float
    weather_mean: np.ndarray
    weather_std: np.ndarray

    @property
    def nwp_mean(self):
        return self.weather_mean

    @property
    def nwp_std(self):
        return self.weather_std


def deterministic_time_features(timestamps: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Timestamp-only periodic coordinates shared by every modality mode."""
    index = pd.DatetimeIndex(pd.to_datetime(timestamps))
    hour = (index.hour + index.minute / 60.0).to_numpy()
    day = (index.dayofyear - 1 + hour / 24.0).to_numpy()
    minute_day = (index.hour * 60 + index.minute).to_numpy()
    quarter = (index.minute // 15).to_numpy()
    cycles = ((hour, 24.0), (day, 365.2425), (minute_day, 1440.0), (quarter, 4.0))
    return np.stack([component for value, period in cycles for component in (
        np.sin(2 * np.pi * value / period), np.cos(2 * np.pi * value / period)
    )], axis=-1).astype(np.float32)


def _canonicalize_timestamp(series: pd.Series, timezone: Optional[str]) -> pd.Series:
    index = pd.DatetimeIndex(pd.to_datetime(series, errors="raise"))
    if timezone:
        if index.tz is None:
            index = index.tz_localize(timezone, ambiguous=False, nonexistent="shift_forward")
        index = index.tz_convert("UTC").tz_localize(None)
    elif index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return pd.Series(index)


def load_csg_schema_frame(path: Path, site_id: str) -> pd.DataFrame:
    if site_id not in CSG_SCHEMAS:
        raise ValueError(f"No explicit CSG schema for {site_id}")
    schema = CSG_SCHEMAS[site_id]
    raw = pd.read_csv(path, header=None, skiprows=1, dtype=str)
    if raw.shape[1] != len(schema.columns):
        raise ValueError(
            f"CSG {site_id} schema mismatch: expected {len(schema.columns)} fields "
            f"{schema.columns}, found {raw.shape[1]}; refusing silent parsing"
        )
    raw.columns = schema.columns
    if schema.columns[schema.timestamp_position] != "date" or schema.columns[schema.power_position] != "power":
        raise AssertionError("CSG schema timestamp/power positions are inconsistent")
    for column in schema.columns[1:]:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    return raw


def _load_era5(path: Path) -> pd.DataFrame:
    weather = pd.read_csv(path)
    timestamp = weather.columns[0]
    missing = [column for column in ERA5_COLUMNS if column not in weather]
    if missing:
        raise KeyError(f"ERA5 file missing columns: {missing}")
    weather = weather.rename(columns={timestamp: "timestamp"})[["timestamp", *ERA5_COLUMNS]]
    weather["timestamp"] = _canonicalize_timestamp(weather["timestamp"], "UTC")
    return weather.set_index("timestamp").sort_index()


def load_solar_frame(spec: SolarDatasetSpec) -> pd.DataFrame:
    if spec.loader_kind == "csg_schema":
        frame = load_csg_schema_frame(spec.csv_path, spec.site_id)
    else:
        frame = pd.read_csv(spec.csv_path)
    required = [spec.timestamp_col, spec.power_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns in {spec.csv_path}: {missing}")
    frame = frame[required].copy()
    frame[spec.timestamp_col] = pd.to_datetime(frame[spec.timestamp_col], errors="raise")
    frame = frame.sort_values(spec.timestamp_col).reset_index(drop=True)
    if spec.weather_path is not None:
        weather = _load_era5(spec.weather_path)
        local_index = pd.DatetimeIndex(frame[spec.timestamp_col])
        if not spec.source_timezone:
            raise ValueError("A source timezone is required to align ERA5")
        utc_lookup = local_index.tz_localize(
            spec.source_timezone, ambiguous=False, nonexistent="shift_forward"
        ).tz_convert("UTC").tz_localize(None)
        aligned = weather.reindex(weather.index.union(utc_lookup)).sort_index().ffill().reindex(utc_lookup)
        aligned.index = frame.index
        for column in spec.weather_cols:
            frame[column] = aligned[column]
    else:
        source = pd.read_csv(spec.csv_path) if spec.loader_kind == "standard" else load_csg_schema_frame(spec.csv_path, spec.site_id)
        for column in spec.weather_cols:
            if column not in source:
                raise KeyError(f"Missing required weather column {column!r} in {spec.csv_path}")
            frame[column] = source[column].to_numpy()
    if frame[spec.timestamp_col].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {spec.csv_path}")
    expected = pd.Timedelta(spec.frequency)
    deltas = frame[spec.timestamp_col].diff().dropna()
    if not spec.allow_irregular_windows and not deltas.eq(expected).all():
        raise ValueError(f"Irregular timestamps in {spec.csv_path}: {deltas.value_counts().to_dict()}")
    frame[spec.power_col] = frame[spec.power_col].clip(lower=0)
    if spec.capacity is not None:
        frame[spec.power_col] = frame[spec.power_col].clip(upper=spec.capacity)
    return frame[[spec.timestamp_col, spec.power_col, *spec.weather_cols]]


def fit_train_scaler(frame: pd.DataFrame, spec: SolarDatasetSpec, train_end: int) -> TrainScaler:
    train = frame.iloc[:train_end]
    power = train[spec.power_col].to_numpy(np.float32)
    weather = train[list(spec.weather_cols)].to_numpy(np.float32) if spec.weather_cols else np.empty((len(train), 0), np.float32)
    power_mean, power_std = float(np.nanmean(power)), float(np.nanstd(power))
    if not np.isfinite(power_mean) or not np.isfinite(power_std):
        raise ValueError("Training power is entirely missing")
    power_std = max(power_std, 1e-6)
    if weather.shape[1]:
        weather_mean, weather_std = np.nanmean(weather, axis=0).astype(np.float32), np.nanstd(weather, axis=0).astype(np.float32)
        if not np.isfinite(weather_mean).all() or not np.isfinite(weather_std).all():
            raise ValueError("A training weather feature is entirely missing")
        weather_std[weather_std < 1e-6] = 1.0
    else:
        weather_mean, weather_std = np.empty(0, np.float32), np.empty(0, np.float32)
    return TrainScaler(power_mean, power_std, weather_mean, weather_std)


def _target_timestamp_set(part: pd.DataFrame, frequency: str) -> set[np.datetime64]:
    delta = pd.Timedelta(frequency)
    result: set[np.datetime64] = set()
    for row in part.itertuples():
        result.update(pd.date_range(row.forecast_start_timestamp, periods=row.pred_len, freq=delta).to_numpy())
    return result


def validate_reference_alignment(manifest: pd.DataFrame, origins: np.ndarray, spec: SolarDatasetSpec) -> None:
    test = manifest.loc[manifest.split == "test"].sort_values("input_end_timestamp")
    origin_index = pd.DatetimeIndex(pd.to_datetime(origins))
    if origin_index.has_duplicates:
        raise ValueError("Reference origins contain duplicates")
    actual = pd.DatetimeIndex(test.input_end_timestamp)
    missing = origin_index.difference(actual)
    extra = actual.difference(origin_index)
    if len(missing) or len(extra):
        raise ValueError(f"Reference origin mismatch: missing={len(missing)}, extra={len(extra)}")
    delta = pd.Timedelta(spec.frequency)
    mapped = test.set_index("input_end_timestamp")
    for origin in origin_index:
        row = mapped.loc[origin]
        if row.forecast_start_timestamp != origin + delta:
            raise ValueError(f"Forecast start mismatch at {origin}")
        expected_end = origin + int(row.pred_len) * delta
        if row.forecast_end_timestamp != expected_end:
            raise ValueError(f"Forecast target interval mismatch at {origin}")


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
    n, delta = len(frame), pd.Timedelta(spec.frequency)
    bounds = {"train": (0, int(n * train_ratio)), "validation": (int(n * train_ratio), int(n * (train_ratio + val_ratio))), "test": (int(n * (train_ratio + val_ratio)), n)}
    timestamps = frame[spec.timestamp_col]
    reference = set(pd.DatetimeIndex(pd.to_datetime(reference_test_origins))) if reference_test_origins is not None else None
    rows, values = [], frame[[spec.power_col, *spec.weather_cols]].to_numpy(np.float32)
    for split, (left, right) in bounds.items():
        for start in range(left, right - seq_len - pred_len + 1):
            input_end, forecast_start = start + seq_len - 1, start + seq_len
            forecast_end = forecast_start + pred_len - 1
            if reference is not None and split == "test" and timestamps.iloc[input_end] not in reference:
                continue
            window_times = timestamps.iloc[start : forecast_end + 1]
            if not window_times.diff().dropna().eq(delta).all():
                if spec.allow_irregular_windows:
                    continue
                raise AssertionError("window timestamps are not continuous")
            if timestamps.iloc[input_end] + delta != timestamps.iloc[forecast_start]:
                raise AssertionError("input_end must precede forecast_start by exactly one interval")
            if timestamps.iloc[forecast_end] - timestamps.iloc[forecast_start] != (pred_len - 1) * delta:
                raise AssertionError("forecast interval does not match pred_len/frequency")
            power_ok = np.isfinite(values[start : forecast_end + 1, 0]).all()
            weather_ok = not spec.weather_cols or np.isfinite(values[forecast_start : forecast_end + 1, 1:]).all()
            if not (power_ok and weather_ok):
                continue
            rows.append({
                "dataset": spec.dataset, "site_id": spec.site_id, "split": split,
                "input_start_timestamp": timestamps.iloc[start], "input_end_timestamp": timestamps.iloc[input_end],
                "forecast_start_timestamp": timestamps.iloc[forecast_start], "forecast_end_timestamp": timestamps.iloc[forecast_end],
                "seq_len": seq_len, "pred_len": pred_len, "start_index": start,
                "covariate_semantics": spec.covariate_semantics,
            })
    manifest = pd.DataFrame(rows)
    target_sets = {split: _target_timestamp_set(manifest[manifest.split == split], spec.frequency) for split in bounds}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = target_sets[left] & target_sets[right]
        if overlap:
            raise AssertionError(f"{left}/{right} target timestamps overlap: {len(overlap)}")
    if reference_test_origins is not None:
        validate_reference_alignment(manifest, reference_test_origins, spec)
    return manifest


class SolarWindowDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, manifest: pd.DataFrame, spec: SolarDatasetSpec, scaler: TrainScaler, mode: str):
        if mode not in {"power", "power_weather", "power_nwp"}:
            raise ValueError(f"Unknown mode: {mode}")
        if mode != "power" and not spec.weather_cols:
            raise ValueError("Power+Weather requires weather columns")
        self.frame, self.manifest, self.spec, self.scaler = frame, manifest.reset_index(drop=True), spec, scaler
        self.mode = "power_weather" if mode == "power_nwp" else mode
        self._time_features = deterministic_time_features(frame[spec.timestamp_col])

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.manifest.iloc[index]
        start, seq_len, pred_len = int(row.start_index), int(row.seq_len), int(row.pred_len)
        future_start = start + seq_len
        power = self.frame[self.spec.power_col].to_numpy(np.float32)
        item: dict[str, torch.Tensor | str] = {
            "history_power": torch.from_numpy((((power[start:future_start] - self.scaler.power_mean) / self.scaler.power_std)[:, None]).copy()),
            "history_time_features": torch.from_numpy(self._time_features[start:future_start].copy()),
            "future_time_features": torch.from_numpy(self._time_features[future_start:future_start + pred_len].copy()),
            "target_power": torch.from_numpy((((power[future_start:future_start + pred_len] - self.scaler.power_mean) / self.scaler.power_std)[:, None]).copy()),
            "target_power_raw": torch.from_numpy(power[future_start:future_start + pred_len, None].copy()),
            "forecast_start_timestamp": str(row.forecast_start_timestamp),
        }
        if self.mode == "power_weather":
            weather = self.frame[list(self.spec.weather_cols)].to_numpy(np.float32)[future_start:future_start + pred_len]
            item["future_weather"] = torch.from_numpy(((weather - self.scaler.weather_mean) / self.scaler.weather_std).copy())
        return item
