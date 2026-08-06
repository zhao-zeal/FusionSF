#!/usr/bin/env python3
"""Compare FusionSF and Chronos-2 predictions on identical MMSP windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUSIONSF_DIR = (
    ROOT
    / "logs/fusionSF_train10_test10/runs/1"
    / "src.models.fusionSF_3modal.FusionSF3M"
)
DEFAULT_CHRONOS_DIR = ROOT / "logs/chronos2_mmsp_site0_9"
DEFAULT_OUTPUT_DIR = ROOT / "logs/fusionsf_vs_chronos2_mmsp_site0_9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusionsf-dir", type=Path, default=DEFAULT_FUSIONSF_DIR)
    parser.add_argument("--chronos-dir", type=Path, default=DEFAULT_CHRONOS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-sites", type=int, default=10)
    parser.add_argument("--target-atol", type=float, default=0.0)
    return parser.parse_args()


def load_pair(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.load(directory / "outputs.npy").astype(np.float64, copy=False)
    target = np.load(directory / "targets.npy").astype(np.float64, copy=False)
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch under {directory}: {prediction.shape} != {target.shape}")
    if prediction.ndim != 3 or prediction.shape[-1] != 1:
        raise ValueError(f"Expected (N, horizon, 1) under {directory}, got {prediction.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError(f"Non-finite prediction or target under {directory}")
    return prediction[..., 0], target[..., 0]


def metric_row(model: str, prediction: np.ndarray, target: np.ndarray) -> dict:
    error = prediction - target
    return {
        "model": model,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "windows": int(prediction.shape[0]),
        "prediction_length": int(prediction.shape[1]),
        "values": int(prediction.size),
    }


def main() -> int:
    args = parse_args()
    if args.num_sites <= 0:
        raise ValueError("num-sites must be positive")
    fusion_pred, fusion_target = load_pair(args.fusionsf_dir)
    chronos_pred, chronos_target = load_pair(args.chronos_dir)
    if fusion_pred.shape != chronos_pred.shape:
        raise ValueError(
            f"Model output shapes differ: FusionSF {fusion_pred.shape}, Chronos-2 {chronos_pred.shape}"
        )
    if not np.allclose(fusion_target, chronos_target, rtol=0.0, atol=args.target_atol):
        difference = float(np.max(np.abs(fusion_target - chronos_target)))
        raise ValueError(f"Targets are not identical; maximum absolute difference={difference}")
    if len(fusion_pred) % args.num_sites:
        raise ValueError(f"{len(fusion_pred)} windows cannot be split across {args.num_sites} sites")

    target = fusion_target
    models = {"FusionSF": fusion_pred, "Chronos-2": chronos_pred}
    global_rows = [metric_row(name, pred, target) for name, pred in models.items()]

    windows_per_site = len(target) // args.num_sites
    target_by_site = target.reshape(args.num_sites, windows_per_site, target.shape[1])
    site_rows = []
    horizon_rows = []
    for name, prediction in models.items():
        pred_by_site = prediction.reshape(args.num_sites, windows_per_site, prediction.shape[1])
        for site_id in range(args.num_sites):
            row = metric_row(name, pred_by_site[site_id], target_by_site[site_id])
            row["site"] = site_id
            site_rows.append(row)
        for horizon in range(prediction.shape[1]):
            error = prediction[:, horizon] - target[:, horizon]
            horizon_rows.append(
                {
                    "model": name,
                    "horizon": horizon + 1,
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "values": int(len(error)),
                }
            )

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
        "site_ids": list(range(args.num_sites)),
        "windows_per_site": windows_per_site,
        "global_metrics": global_rows,
        "fusionsf_dir": str(args.fusionsf_dir.resolve()),
        "chronos_dir": str(args.chronos_dir.resolve()),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(global_frame.to_string(index=False))
    print(f"saved={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
