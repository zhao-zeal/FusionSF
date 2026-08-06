#!/usr/bin/env python3
"""Auditable Chronos-2 baselines on aligned MMSP test windows.

The source must be one explicit FusionSF test artifact directory containing
inputs, targets, site IDs, and forecast timing metadata. The default mode uses
historical power only. The optional future-NWP mode supplies exactly the 15 NWP
values aligned to target timestamps, with missing placeholders for past NWP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.forecast_metrics import compute_forecast_metrics, metrics_by_horizon  # noqa: E402


DEFAULT_MODEL = (
    Path("/home/zhaopp/.cache/huggingface/hub/models--amazon--chronos-2/snapshots")
    / "29ec3766d36d6f73f0696f85560a422f50e8498c"
)
ONE_HOUR_NS = 3_600_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quantile-levels", default="0.1,0.5,0.9")
    parser.add_argument("--point-quantile", type=float, default=0.5)
    parser.add_argument("--use-future-nwp", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/MMSP/data")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_quantile_levels(value: str, point_quantile: float) -> list[float]:
    levels = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not levels or levels != sorted(set(levels)):
        raise ValueError("quantile levels must be a non-empty, sorted, unique list")
    if any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError("every quantile level must lie strictly between 0 and 1")
    if point_quantile not in levels:
        raise ValueError("point quantile must be included in quantile levels")
    return levels


def resolve_array(directory: Path, *names: str) -> Path:
    matches = [directory / name for name in names if (directory / name).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one of {list(names)} under {directory.resolve()}, got {matches}"
        )
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_aligned_windows(source_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    source_dir = source_dir.resolve()
    paths = {
        "inputs": resolve_array(source_dir, "inputs.npy"),
        "targets": resolve_array(source_dir, "targets.npy"),
        "site_ids": resolve_array(source_dir, "site_ids.npy"),
        "input_end_timestamps": resolve_array(source_dir, "input_end_timestamps.npy"),
        "forecast_start_timestamps": resolve_array(source_dir, "forecast_start_timestamps.npy"),
        "forecast_end_timestamps": resolve_array(source_dir, "forecast_end_timestamps.npy"),
        "forecast_timestamps": resolve_array(
            source_dir, "forecast_timestamps.npy", "timestamps.npy"
        ),
    }
    arrays = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
    inputs, targets = arrays["inputs"], arrays["targets"]
    if inputs.ndim != 3 or inputs.shape[-1] != 1:
        raise ValueError(f"inputs must have shape (N, context, 1), got {inputs.shape}")
    if targets.ndim != 3 or targets.shape[-1] != 1:
        raise ValueError(f"targets must have shape (N, horizon, 1), got {targets.shape}")
    if len(inputs) != len(targets) or not len(inputs):
        raise ValueError("inputs and targets must have the same positive window count")
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError("inputs and targets must be finite")

    count, horizon = len(inputs), targets.shape[1]
    site_ids = np.asarray(arrays["site_ids"]).reshape(-1)
    input_end = np.asarray(arrays["input_end_timestamps"]).reshape(-1)
    forecast_start = np.asarray(arrays["forecast_start_timestamps"]).reshape(-1)
    forecast_end = np.asarray(arrays["forecast_end_timestamps"]).reshape(-1)
    forecast_ts = np.asarray(arrays["forecast_timestamps"])
    for name, value in {
        "site_ids": site_ids,
        "input_end_timestamps": input_end,
        "forecast_start_timestamps": forecast_start,
        "forecast_end_timestamps": forecast_end,
    }.items():
        if value.shape != (count,):
            raise ValueError(f"{name} must align one-to-one with windows, got {value.shape}")
    if forecast_ts.shape != (count, horizon):
        raise ValueError(
            f"forecast_timestamps must have shape {(count, horizon)}, got {forecast_ts.shape}"
        )
    if not np.issubdtype(site_ids.dtype, np.integer):
        raise ValueError("site_ids must be integers")
    for name, value in {
        "input_end_timestamps": input_end,
        "forecast_start_timestamps": forecast_start,
        "forecast_end_timestamps": forecast_end,
        "forecast_timestamps": forecast_ts,
    }.items():
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{name} must contain integer nanosecond timestamps")
    if not np.all(forecast_start - input_end == ONE_HOUR_NS):
        raise ValueError("every forecast must start exactly one hour after its input window")
    if not np.array_equal(forecast_ts[:, 0], forecast_start):
        raise ValueError("forecast timestamp column 1 does not match forecast_start_timestamps")
    if not np.array_equal(forecast_ts[:, -1], forecast_end):
        raise ValueError("last forecast timestamp does not match forecast_end_timestamps")
    if horizon > 1 and not np.all(np.diff(forecast_ts, axis=1) == ONE_HOUR_NS):
        raise ValueError("forecast timestamps must be hourly and contiguous")

    arrays.update(
        {
            "site_ids": site_ids,
            "input_end_timestamps": input_end,
            "forecast_start_timestamps": forecast_start,
            "forecast_end_timestamps": forecast_end,
            "forecast_timestamps": forecast_ts,
        }
    )
    return arrays, paths


def summarize_source(arrays: dict[str, np.ndarray], paths: dict[str, Path]) -> dict:
    return {
        "num_windows": int(len(arrays["inputs"])),
        "site_ids": [int(value) for value in np.unique(arrays["site_ids"])],
        "context_length": int(arrays["inputs"].shape[1]),
        "prediction_length": int(arrays["targets"].shape[1]),
        "forecast_start_min": str(
            pd.Timestamp(int(arrays["forecast_start_timestamps"].min()), unit="ns")
        ),
        "forecast_end_max": str(
            pd.Timestamp(int(arrays["forecast_end_timestamps"].max()), unit="ns")
        ),
        "source_files": {name: str(path.resolve()) for name, path in paths.items()},
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
    }


class MMSPFutureNWP:
    """Exact valid-time lookup for the 15 MMSP future NWP variables."""

    def __init__(self, data_dir: Path, arrays: dict[str, np.ndarray]) -> None:
        self.data_dir = data_dir.resolve()
        power_path = self.data_dir / "solar_power/solar_power.csv"
        nwp_path = self.data_dir / "nwp/nwp.csv"
        if not power_path.is_file() or not nwp_path.is_file():
            raise FileNotFoundError(f"MMSP power or NWP CSV is missing under {self.data_dir}")
        power = pd.read_csv(power_path, usecols=["site", "lat", "lon"])
        coordinate_rows = power.drop_duplicates()
        counts = coordinate_rows.groupby("site").size()
        if not (counts == 1).all():
            raise ValueError("each MMSP site must map to exactly one coordinate")
        self.site_coordinates = {
            int(row.site): (round(float(row.lat), 1), round(float(row.lon), 1))
            for row in coordinate_rows.itertuples(index=False)
        }
        missing_sites = sorted(set(map(int, np.unique(arrays["site_ids"]))) - self.site_coordinates.keys())
        if missing_sites:
            raise ValueError(f"source windows contain sites absent from MMSP power CSV: {missing_sites}")

        nwp = pd.read_csv(nwp_path, parse_dates=["fcst_date"])
        nwp["lat"] = nwp["lat"].round(1)
        nwp["lon"] = nwp["lon"].round(1)
        self.feature_names = [
            column for column in nwp.columns if column not in {"fcst_date", "lat", "lon"}
        ]
        if len(self.feature_names) != 15:
            raise ValueError(f"expected 15 MMSP NWP variables, got {len(self.feature_names)}")
        if nwp.duplicated(["lat", "lon", "fcst_date"]).any():
            raise ValueError("MMSP NWP contains duplicate coordinate/valid-time rows")
        self.missing_values_before_interpolation = int(
            nwp[self.feature_names].isna().sum().sum()
        )
        nwp = nwp.sort_values(["lat", "lon", "fcst_date"])
        nwp[self.feature_names] = nwp.groupby(
            ["lat", "lon"], sort=False
        )[self.feature_names].transform(
            lambda group: group.interpolate(limit_direction="both")
        )
        if not np.isfinite(nwp[self.feature_names].to_numpy()).all():
            raise ValueError("MMSP NWP remains non-finite after per-grid temporal interpolation")
        self.groups = {}
        for coordinate, frame in nwp.groupby(["lat", "lon"], sort=False):
            # Pandas may parse CSV timestamps as datetime64[us]. Normalize to
            # nanoseconds before comparing with FusionSF's saved epoch-ns arrays.
            timestamps = frame["fcst_date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            order = np.argsort(timestamps)
            self.groups[coordinate] = (
                timestamps[order], frame[self.feature_names].to_numpy(dtype=np.float32)[order]
            )
        missing_coordinates = {
            site: coordinate
            for site, coordinate in self.site_coordinates.items()
            if site in set(map(int, np.unique(arrays["site_ids"]))) and coordinate not in self.groups
        }
        if missing_coordinates:
            raise ValueError(f"no NWP grid exists for source sites: {missing_coordinates}")
        self._validate_coverage(arrays)
        self.nwp_path = nwp_path

    def _validate_coverage(self, arrays: dict[str, np.ndarray]) -> None:
        for site_id in np.unique(arrays["site_ids"]):
            selected = arrays["site_ids"] == site_id
            required = np.unique(arrays["forecast_timestamps"][selected])
            available, _ = self.groups[self.site_coordinates[int(site_id)]]
            missing = required[~np.isin(required, available)]
            if len(missing):
                first = pd.Timestamp(int(missing[0]), unit="ns")
                raise ValueError(f"future NWP is missing for site {int(site_id)} at {first}")

    def chronos_input(
        self, index: int, arrays: dict[str, np.ndarray]
    ) -> dict[str, object]:
        site_id = int(arrays["site_ids"][index])
        required = arrays["forecast_timestamps"][index]
        available, values = self.groups[self.site_coordinates[site_id]]
        positions = np.searchsorted(available, required)
        if np.any(positions >= len(available)) or not np.array_equal(available[positions], required):
            raise RuntimeError(f"validated future NWP lookup failed for window {index}")
        future = values[positions]
        missing_past = np.full(arrays["inputs"].shape[1], np.nan, dtype=np.float32)
        return {
            "target": arrays["inputs"][index, :, 0],
            "past_covariates": {
                name: missing_past.copy() for name in self.feature_names
            },
            "future_covariates": {
                name: future[:, feature_index]
                for feature_index, name in enumerate(self.feature_names)
            },
        }


def save_results(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    predictions: np.ndarray,
    quantile_predictions: np.ndarray,
    quantile_levels: list[float],
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    clipped = np.clip(predictions, 0.0, None)
    np.save(output_dir / "predictions_raw.npy", predictions)
    np.save(output_dir / "predictions_clipped.npy", clipped)
    np.save(output_dir / "predictions_quantiles.npy", quantile_predictions)
    np.save(output_dir / "targets.npy", arrays["targets"])
    np.save(output_dir / "site_ids.npy", arrays["site_ids"])
    np.save(output_dir / "timestamps.npy", arrays["forecast_timestamps"])

    raw_metrics = compute_forecast_metrics(predictions, arrays["targets"])
    clipped_metrics = compute_forecast_metrics(clipped, arrays["targets"])
    metrics = dict(raw_metrics)
    metrics.update({f"clipped_{key}": value for key, value in clipped_metrics.items()})
    metrics.update(
        {
            "prediction_mean": float(predictions.mean()),
            "prediction_std": float(predictions.std()),
            "negative_prediction_fraction": float(np.mean(predictions < 0)),
            "zero_prediction_fraction": float(np.mean(predictions == 0)),
        }
    )
    target_2d = arrays["targets"][..., 0]
    for index, level in enumerate(quantile_levels):
        error = target_2d - quantile_predictions[..., index]
        metrics[f"pinball_q{level:g}"] = float(
            np.mean(np.maximum(level * error, (level - 1.0) * error))
        )
    if len(quantile_levels) >= 2:
        lower, upper = quantile_predictions[..., 0], quantile_predictions[..., -1]
        metrics["central_interval_lower_quantile"] = quantile_levels[0]
        metrics["central_interval_upper_quantile"] = quantile_levels[-1]
        metrics["central_interval_coverage"] = float(
            np.mean((target_2d >= lower) & (target_2d <= upper))
        )
        metrics["central_interval_mean_width"] = float(np.mean(upper - lower))
    metrics["quantile_crossing_fraction"] = float(
        np.mean(np.diff(quantile_predictions, axis=-1) < 0)
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    horizon_frames = []
    for variant, prediction in (("raw", predictions), ("clipped", clipped)):
        frame = pd.DataFrame(metrics_by_horizon(prediction, arrays["targets"]))
        frame.insert(0, "prediction_variant", variant)
        horizon_frames.append(frame)
    pd.concat(horizon_frames, ignore_index=True).to_csv(
        output_dir / "metrics_by_horizon.csv", index=False
    )
    site_rows = []
    for site_id in np.unique(arrays["site_ids"]):
        selected = arrays["site_ids"] == site_id
        site_rows.append(
            {
                "site_id": int(site_id),
                **compute_forecast_metrics(predictions[selected], arrays["targets"][selected]),
            }
        )
    pd.DataFrame(site_rows).to_csv(output_dir / "metrics_by_site.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    source_dir, output_dir, model_path = (
        args.source_dir.resolve(), args.output_dir.resolve(), args.model.resolve()
    )
    quantile_levels = parse_quantile_levels(args.quantile_levels, args.point_quantile)
    point_index = quantile_levels.index(args.point_quantile)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Chronos-2 model directory is unavailable: {model_path}")

    arrays, paths = load_aligned_windows(source_dir)
    nwp = MMSPFutureNWP(args.data_dir, arrays) if args.use_future_nwp else None
    source_summary = summarize_source(arrays, paths)
    preflight = {
        "protocol": (
            "chronos2_mmsp_power_future_nwp_v2"
            if args.use_future_nwp else "chronos2_mmsp_power_only_v2"
        ),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "power_context_only": not args.use_future_nwp,
        "nwp_used": args.use_future_nwp,
        "nwp_context_policy": "future_only" if args.use_future_nwp else "none",
        "past_nwp_values_provided": False,
        "nwp_features": nwp.feature_names if nwp else [],
        "nwp_source": str(nwp.nwp_path.resolve()) if nwp else None,
        "nwp_source_sha256": sha256(nwp.nwp_path) if nwp else None,
        "nwp_scaling": "raw_units_chronos_internal_scaling" if nwp else None,
        "nwp_interpolation": (
            "per_grid_time_sorted_linear_limit_direction_both" if nwp else None
        ),
        "nwp_missing_values_before_interpolation": (
            nwp.missing_values_before_interpolation if nwp else None
        ),
        "nwp_availability_assumption": (
            "future NWP is available at forecast origin per user confirmation; "
            "the MMSP file has valid time but no issue/publication time"
            if nwp else None
        ),
        "satellite_used": False,
        "fusionsf_embedding_used": False,
        "quantile_levels": quantile_levels,
        "point_quantile": args.point_quantile,
        **source_summary,
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if args.preflight_only:
        return 0

    # Lazy import keeps the preflight usable without initializing model/GPU state.
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(
        str(model_path),
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )
    inputs, targets = arrays["inputs"], arrays["targets"]
    predictions = np.empty_like(targets, dtype=np.float32)
    quantile_predictions = np.empty(
        (len(targets), targets.shape[1], len(quantile_levels)), dtype=np.float32
    )
    started = time.perf_counter()
    for start in range(0, len(inputs), args.batch_size):
        stop = min(start + args.batch_size, len(inputs))
        context = (
            [nwp.chronos_input(index, arrays) for index in range(start, stop)]
            if nwp else torch.from_numpy(inputs[start:stop, :, 0]).unsqueeze(1)
        )
        quantiles, _ = pipeline.predict_quantiles(
            context,
            prediction_length=targets.shape[1],
            quantile_levels=quantile_levels,
            batch_size=args.batch_size,
        )
        if len(quantiles) != stop - start:
            raise RuntimeError(
                f"Chronos returned {len(quantiles)} forecasts for {stop - start} windows"
            )
        for offset, forecast in enumerate(quantiles):
            values = torch.as_tensor(forecast).float().cpu().numpy()
            if values.shape != (1, targets.shape[1], len(quantile_levels)):
                raise RuntimeError(
                    "unexpected forecast shape "
                    f"{values.shape}; expected {(1, targets.shape[1], len(quantile_levels))}"
                )
            quantile_predictions[start + offset] = values[0]
            predictions[start + offset, :, 0] = values[0, :, point_index]
        print(f"completed={stop}/{len(inputs)}", flush=True)

    elapsed = time.perf_counter() - started
    metadata = {
        **preflight,
        "model": "amazon/chronos-2",
        "point_forecast": f"{args.point_quantile:g}_quantile",
        "device_map": args.device_map,
        "batch_size": args.batch_size,
        "inference_seconds": elapsed,
        "inference_ms_per_window": elapsed * 1000 / len(inputs),
    }
    save_results(
        output_dir, arrays, predictions, quantile_predictions, quantile_levels, metadata
    )
    print(f"saved={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
