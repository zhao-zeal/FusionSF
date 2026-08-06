#!/usr/bin/env python3
"""Run Chronos-2 zero-shot on the exact test windows saved by FusionSF."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUSIONSF_DIR = (
    ROOT
    / "logs/fusionSF_train10_test10/runs/1"
    / "src.models.fusionSF_3modal.FusionSF3M"
)
DEFAULT_OUTPUT_DIR = ROOT / "logs/chronos2_mmsp_site0_9"
DEFAULT_NWP_OUTPUT_DIR = ROOT / "logs/chronos2_nwp_mmsp_site0_9"
DEFAULT_DATA_DIR = ROOT / "data/MMSP/data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chronos-2 zero-shot forecast using FusionSF's saved MMSP test windows."
    )
    parser.add_argument("--fusionsf-dir", type=Path, default=DEFAULT_FUSIONSF_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-sites", type=int, default=10)
    parser.add_argument(
        "--use-nwp",
        action="store_true",
        help="Provide the same 15 normalized MMSP NWP variables as known future covariates.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_windows(directory: Path, num_sites: int) -> tuple[np.ndarray, np.ndarray]:
    inputs_path = directory / "inputs.npy"
    targets_path = directory / "targets.npy"
    if not inputs_path.is_file() or not targets_path.is_file():
        raise FileNotFoundError(
            f"Expected inputs.npy and targets.npy under {directory.resolve()}"
        )

    inputs = np.load(inputs_path).astype(np.float32, copy=False)
    targets = np.load(targets_path).astype(np.float32, copy=False)
    if inputs.ndim != 3 or inputs.shape[-1] != 1:
        raise ValueError(f"Expected inputs shape (N, context, 1), got {inputs.shape}")
    if targets.ndim != 3 or targets.shape[-1] != 1:
        raise ValueError(f"Expected targets shape (N, horizon, 1), got {targets.shape}")
    if len(inputs) != len(targets):
        raise ValueError(f"Input/target window count differs: {len(inputs)} != {len(targets)}")
    if len(inputs) % num_sites:
        raise ValueError(f"{len(inputs)} windows cannot be evenly assigned to {num_sites} sites")
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError("Inputs or targets contain non-finite values")
    return inputs, targets


class MMSPNWPCovariates:
    """Reconstruct NWP slices using the same indexing and scaling as FusionSF."""

    def __init__(
        self,
        data_dir: Path,
        num_sites: int,
        windows_per_site: int,
        context_length: int,
        prediction_length: int,
    ) -> None:
        power = pd.read_csv(
            data_dir / "solar_power/solar_power.csv",
            usecols=["datetime", "lat", "lon", "site"],
            parse_dates=["datetime"],
        )
        selected_sites = sorted(power["site"].unique())[:num_sites]
        if selected_sites != list(range(num_sites)):
            raise ValueError(f"Expected MMSP sites 0..{num_sites - 1}, got {selected_sites}")
        power = power[power["site"].isin(selected_sites)]
        site_frames = {int(site): frame.reset_index(drop=True) for site, frame in power.groupby("site")}
        lengths = {site: len(frame) for site, frame in site_frames.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"MMSP station lengths differ: {lengths}")
        power_length = next(iter(lengths.values()))
        total_windows = power_length - context_length - prediction_length + 1
        expected_test_windows = int(total_windows * 0.2)
        if windows_per_site != expected_test_windows:
            raise ValueError(
                f"Saved windows/site={windows_per_site}, but FusionSF split implies "
                f"{expected_test_windows}"
            )
        self.test_start = total_windows - windows_per_site
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.site_coords = {
            site: (round(float(frame.loc[0, "lat"]), 1), round(float(frame.loc[0, "lon"]), 1))
            for site, frame in site_frames.items()
        }

        nwp = pd.read_csv(data_dir / "nwp/nwp.csv", parse_dates=["fcst_date"]).interpolate()
        nwp["lat"] = nwp["lat"].round(1)
        nwp["lon"] = nwp["lon"].round(1)
        self.feature_names = [c for c in nwp.columns if c not in {"fcst_date", "lat", "lon"}]
        # StandardScaler in FusionSF fits these columns globally before the temporal split.
        values = nwp[self.feature_names].to_numpy(dtype=np.float64)
        means = values.mean(axis=0)
        scales = values.std(axis=0)
        scales[scales == 0] = 1.0
        nwp.loc[:, self.feature_names] = (values - means) / scales
        self.groups = {
            key: frame[self.feature_names].to_numpy(dtype=np.float32)
            for key, frame in nwp.groupby(["lat", "lon"], sort=False)
        }
        missing = {site: coord for site, coord in self.site_coords.items() if coord not in self.groups}
        if missing:
            raise KeyError(f"No NWP grid point for MMSP sites: {missing}")

    def chronos_input(self, global_window: int, windows_per_site: int, target: np.ndarray) -> dict:
        site = global_window // windows_per_site
        local_window = global_window % windows_per_site
        start = self.test_start + local_window
        nwp = self.groups[self.site_coords[site]]
        middle = start + self.context_length
        stop = middle + self.prediction_length
        if stop > len(nwp):
            raise IndexError(f"NWP slice exceeds available data for site {site}: {stop}>{len(nwp)}")
        return {
            "target": target,
            "past_covariates": {
                name: nwp[start:middle, index]
                for index, name in enumerate(self.feature_names)
            },
            "future_covariates": {
                name: nwp[middle:stop, index]
                for index, name in enumerate(self.feature_names)
            },
        }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.num_sites <= 0:
        raise ValueError("batch-size and num-sites must be positive")

    # Import lazily so --help and input validation remain usable in the FusionSF env.
    from chronos import Chronos2Pipeline

    inputs, targets = load_windows(args.fusionsf_dir, args.num_sites)
    context_length = inputs.shape[1]
    prediction_length = targets.shape[1]
    windows_per_site = len(inputs) // args.num_sites
    output_dir = args.output_dir or (
        DEFAULT_NWP_OUTPUT_DIR if args.use_nwp else DEFAULT_OUTPUT_DIR
    )
    nwp_covariates = (
        MMSPNWPCovariates(
            args.data_dir,
            args.num_sites,
            windows_per_site,
            context_length,
            prediction_length,
        )
        if args.use_nwp
        else None
    )
    print(
        f"windows={len(inputs)} sites={args.num_sites} windows_per_site={windows_per_site} "
        f"context_length={context_length} prediction_length={prediction_length}",
        flush=True,
    )

    pipeline = Chronos2Pipeline.from_pretrained(
        args.model,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )

    output = np.empty_like(targets, dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, len(inputs), args.batch_size):
        stop = min(start + args.batch_size, len(inputs))
        # Chronos-2 expects (batch, n_variates, history); MMSP is univariate.
        if nwp_covariates is None:
            context = torch.from_numpy(inputs[start:stop, :, 0]).unsqueeze(1)
        else:
            context = [
                nwp_covariates.chronos_input(
                    index, windows_per_site, inputs[index, :, 0]
                )
                for index in range(start, stop)
            ]
        quantiles, _ = pipeline.predict_quantiles(
            context,
            prediction_length=prediction_length,
            quantile_levels=[0.5],
            batch_size=args.batch_size,
        )
        if len(quantiles) != stop - start:
            raise RuntimeError(
                f"Chronos returned {len(quantiles)} forecasts for {stop - start} windows"
            )
        for offset, forecast in enumerate(quantiles):
            values = torch.as_tensor(forecast).float().cpu().numpy().squeeze()
            if values.shape != (prediction_length,):
                raise RuntimeError(
                    f"Unexpected forecast shape {values.shape}; expected {(prediction_length,)}"
                )
            output[start + offset, :, 0] = values
        print(f"completed={stop}/{len(inputs)}", flush=True)

    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "inputs.npy", inputs)
    np.save(output_dir / "outputs.npy", output)
    np.save(output_dir / "targets.npy", targets)
    metadata = {
        "model": args.model,
        "mode": "zero_shot",
        "point_forecast": "0.5_quantile",
        "use_nwp": args.use_nwp,
        "nwp_features": nwp_covariates.feature_names if nwp_covariates else [],
        "source_fusionsf_dir": str(args.fusionsf_dir.resolve()),
        "num_windows": len(inputs),
        "num_sites": args.num_sites,
        "site_ids": list(range(args.num_sites)),
        "windows_per_site": windows_per_site,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "batch_size": args.batch_size,
        "device_map": args.device_map,
        "inference_seconds": elapsed,
        "inference_ms_per_window": elapsed * 1000 / len(inputs),
        "output_shape": list(output.shape),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved={output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
