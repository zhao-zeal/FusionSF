#!/usr/bin/env python3
"""Compare power, power+NWP, and full FusionSF runs on identical MMSP windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = "src.models.fusionSF_3modal.FusionSF3M"


def default_run(task_name: str) -> Path:
    return ROOT / "logs" / task_name / "runs" / "1" / MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--power-dir", type=Path, default=default_run("fusionSF_power_optimized_train10_test10")
    )
    parser.add_argument(
        "--power-nwp-dir",
        type=Path,
        default=default_run("fusionSF_power_nwp_optimized_train10_test10"),
    )
    parser.add_argument("--all-dir", type=Path, default=default_run("fusionSF_train10_test10"))
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "logs" / "fusionsf_modality_comparison"
    )
    parser.add_argument("--num-sites", type=int, default=10)
    parser.add_argument("--target-atol", type=float, default=0.0)
    return parser.parse_args()


def load_pair(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    prediction_path = directory / "outputs.npy"
    target_path = directory / "targets.npy"
    if not prediction_path.is_file() or not target_path.is_file():
        raise FileNotFoundError(f"Missing outputs.npy or targets.npy under {directory}")
    prediction = np.load(prediction_path).astype(np.float64, copy=False)
    target = np.load(target_path).astype(np.float64, copy=False)
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch under {directory}: {prediction.shape} != {target.shape}")
    if prediction.ndim != 3 or prediction.shape[-1] != 1:
        raise ValueError(f"Expected (N, horizon, 1) under {directory}, got {prediction.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError(f"Non-finite predictions or targets under {directory}")
    return prediction[..., 0], target[..., 0]


def metric_row(model: str, prediction: np.ndarray, target: np.ndarray) -> dict:
    error = prediction - target
    return {
        "model": model,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mse": float(np.mean(np.square(error))),
        "windows": int(prediction.shape[0]),
        "prediction_length": int(prediction.shape[1]),
        "values": int(prediction.size),
    }


def main() -> int:
    args = parse_args()
    if args.num_sites <= 0:
        raise ValueError("num-sites must be positive")

    directories = {
        "Power": args.power_dir,
        "Power + NWP": args.power_nwp_dir,
        "All modalities": args.all_dir,
    }
    pairs = {name: load_pair(path) for name, path in directories.items()}
    reference_target = pairs["All modalities"][1]
    predictions = {}
    for name, (prediction, target) in pairs.items():
        if prediction.shape != reference_target.shape:
            raise ValueError(
                f"Shape mismatch: {name} {prediction.shape} != all modalities {reference_target.shape}"
            )
        if not np.allclose(target, reference_target, rtol=0.0, atol=args.target_atol):
            difference = float(np.max(np.abs(target - reference_target)))
            raise ValueError(f"Targets for {name} differ from all modalities; max diff={difference}")
        predictions[name] = prediction

    if len(reference_target) % args.num_sites:
        raise ValueError(f"{len(reference_target)} windows cannot be split over {args.num_sites} sites")

    global_rows = [metric_row(name, prediction, reference_target) for name, prediction in predictions.items()]
    all_metrics = next(row for row in global_rows if row["model"] == "All modalities")
    for row in global_rows:
        for metric in ("mae", "rmse", "mse"):
            row[f"{metric}_change_vs_all_pct"] = (
                (row[metric] - all_metrics[metric]) / all_metrics[metric] * 100.0
            )

    windows_per_site = len(reference_target) // args.num_sites
    target_by_site = reference_target.reshape(args.num_sites, windows_per_site, -1)
    site_rows = []
    horizon_rows = []
    for name, prediction in predictions.items():
        prediction_by_site = prediction.reshape(args.num_sites, windows_per_site, -1)
        for site in range(args.num_sites):
            row = metric_row(name, prediction_by_site[site], target_by_site[site])
            row["site"] = site
            site_rows.append(row)
        for horizon in range(prediction.shape[1]):
            row = metric_row(name, prediction[:, horizon : horizon + 1], reference_target[:, horizon : horizon + 1])
            row["horizon"] = horizon + 1
            horizon_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_frame = pd.DataFrame(global_rows)
    global_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(site_rows).sort_values(["site", "model"]).to_csv(
        args.output_dir / "metrics_by_site.csv", index=False
    )
    pd.DataFrame(horizon_rows).sort_values(["horizon", "model"]).to_csv(
        args.output_dir / "metrics_by_horizon.csv", index=False
    )
    summary = {
        "targets_identical": True,
        "num_sites": args.num_sites,
        "windows_per_site": windows_per_site,
        "directories": {name: str(path.resolve()) for name, path in directories.items()},
        "global_metrics": global_rows,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(global_frame.to_string(index=False))
    print(f"saved={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
