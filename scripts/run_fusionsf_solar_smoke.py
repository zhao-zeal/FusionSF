#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.solar_energy_dataset import (
    ERA5_COLUMNS, SolarDatasetSpec, SolarWindowDataset, build_sample_manifest,
    fit_train_scaler, load_solar_frame,
)
from src.models.fusionsf_solar import FusionSFSolar


GEFCOM_WEATHER = ("VAR78", "VAR79", "VAR157", "VAR164", "VAR169", "VAR178")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gefcom", "skippd", "csg"), required=True)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--weather-path", type=Path)
    parser.add_argument("--chronos-origin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("power", "power_weather"), required=True)
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--limit-train-batches", type=int, default=2)
    parser.add_argument("--limit-val-batches", type=int, default=2)
    parser.add_argument("--limit-test-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dim", type=int, default=32)
    return parser.parse_args()


def make_spec(args) -> SolarDatasetSpec:
    if args.dataset == "gefcom":
        return SolarDatasetSpec("GEFCom", args.site_id, args.csv, "TIMESTAMP", "POWER", GEFCOM_WEATHER, "1h", 1.0, "future_nwp")
    if args.dataset == "skippd":
        return SolarDatasetSpec("SKIPPD", args.site_id, args.csv, "date", "OT", ERA5_COLUMNS, "15min", 30.0, "future_reanalysis", args.weather_path, "America/Los_Angeles", "standard", True)
    capacity = float(args.csv.stem.split("capacity-")[1].split("MW")[0])
    return SolarDatasetSpec("CSG", args.site_id, args.csv, "date", "power", ERA5_COLUMNS, "15min", capacity, "future_reanalysis", args.weather_path, "Asia/Shanghai", "csg_schema")


def canonicalize_origins(origins: np.ndarray, spec: SolarDatasetSpec) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(origins)).to_numpy(dtype="datetime64[ns]")


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    spec = make_spec(args)
    frame = load_solar_frame(spec)
    scaler = fit_train_scaler(frame, spec, int(len(frame) * 0.6))
    origins = canonicalize_origins(np.load(args.chronos_origin), spec)
    manifest = build_sample_manifest(frame, spec, args.seq_len, args.pred_len, origins)
    active_covariate_semantics = "none" if args.mode == "power" else spec.covariate_semantics
    manifest["covariate_semantics"] = active_covariate_semantics
    args.output_dir.mkdir(parents=True, exist_ok=True)
    published_manifest = manifest.drop(columns="start_index")
    published_manifest.to_csv(args.output_dir / "sample_manifest.csv", index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionSFSolar(len(spec.weather_cols), seq_len=args.seq_len, dim=args.dim, depth=1, heads=4, dim_head=args.dim // 4).to(device)
    optimizer, criterion = torch.optim.Adam(model.parameters(), lr=1e-3), nn.MSELoss()
    shapes, losses = {}, {}
    for _ in range(args.max_epochs):
        model.train()
        dataset = SolarWindowDataset(frame, manifest[manifest.split == "train"], spec, scaler, args.mode)
        for batch_index, batch in enumerate(DataLoader(dataset, batch_size=args.batch_size, shuffle=True)):
            if batch_index >= args.limit_train_batches: break
            tensors = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
            weather = tensors.get("future_weather")
            pred, embeddings = model(tensors["history_power"], tensors["history_time_features"], tensors["future_time_features"], weather, True)
            loss = criterion(pred, tensors["target_power"])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            shapes = {
                "history_power": list(tensors["history_power"].shape),
                "history_time_features": list(tensors["history_time_features"].shape),
                "future_time_features": list(tensors["future_time_features"].shape),
                "future_weather": list(weather.shape) if weather is not None else None,
                "prediction": list(pred.shape),
                "ts_embedding": list(embeddings["ts_embedding"].shape),
                "fusion_embedding": list(embeddings["fusion_embedding"].shape),
            }
            losses["train_loss_last"] = float(loss.detach())

    predictions, targets, metadata = [], [], []
    model.eval()
    with torch.no_grad():
        for split, limit in (("validation", args.limit_val_batches), ("test", args.limit_test_batches)):
            subset = manifest[manifest.split == split].reset_index(drop=True)
            dataset = SolarWindowDataset(frame, subset, spec, scaler, args.mode)
            split_losses = []
            for batch_index, batch in enumerate(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)):
                if batch_index >= limit: break
                tensors = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
                pred_scaled = model(tensors["history_power"], tensors["history_time_features"], tensors["future_time_features"], tensors.get("future_weather"))
                split_losses.append(float(criterion(pred_scaled, tensors["target_power"])))
                if split == "test":
                    pred_raw = pred_scaled.cpu().numpy() * scaler.power_std + scaler.power_mean
                    predictions.append(pred_raw); targets.append(tensors["target_power_raw"].cpu().numpy())
                    rows = subset.iloc[batch_index * args.batch_size : batch_index * args.batch_size + len(pred_raw)].copy()
                    rows["model_family"] = model.model_family
                    rows["model_class"] = type(model).__name__
                    rows["architecture_version"] = model.architecture_version
                    rows["covariate_semantics"] = active_covariate_semantics
                    metadata.extend(rows.to_dict("records"))
            losses[f"{split}_loss"] = float(np.mean(split_losses))
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    pred = np.clip(pred, 0.0, spec.capacity) if spec.capacity is not None else np.clip(pred, 0.0, None)
    if not np.isfinite(pred).all(): raise AssertionError("Non-finite predictions")
    meta = pd.DataFrame(metadata)
    if len(meta) != len(pred): raise AssertionError("Metadata/prediction row mismatch")
    metrics = {"mae": float(np.mean(np.abs(pred - target))), "rmse": float(np.sqrt(np.mean((pred - target) ** 2))), "samples": len(pred), "full_test_manifest_aligned": True}
    np.save(args.output_dir / "predictions.npy", pred); np.save(args.output_dir / "targets.npy", target)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    meta.drop(columns="start_index").to_csv(args.output_dir / "prediction_metadata.csv", index=False)
    for name, payload in (("metrics", metrics), ("shapes", shapes), ("losses", losses)):
        (args.output_dir / f"{name}.json").write_text(json.dumps(payload, indent=2))
    run_config = {key.replace("_", "-"): str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update({"model_family": model.model_family, "model_class": type(model).__name__, "architecture_version": model.architecture_version, "covariate_semantics": active_covariate_semantics})
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(json.dumps({"dataset": args.dataset, "mode": args.mode, "shapes": shapes, "metrics": metrics, "losses": losses}, indent=2))


if __name__ == "__main__": main()
