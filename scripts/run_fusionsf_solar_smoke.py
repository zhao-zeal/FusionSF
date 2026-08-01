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

from src.datasets.solar_energy_dataset import SolarDatasetSpec, SolarWindowDataset, build_sample_manifest, fit_train_scaler, load_solar_frame
from src.models.fusionsf_solar import FusionSFSolar


NWP = ("VAR78", "VAR79", "VAR157", "VAR164", "VAR169", "VAR178")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--chronos-origin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("power", "power_nwp"), required=True)
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--limit-train-batches", type=int, default=2)
    parser.add_argument("--limit-val-batches", type=int, default=2)
    parser.add_argument("--limit-test-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    spec = SolarDatasetSpec("GEFCom", "zone1", args.csv, "TIMESTAMP", "POWER", NWP, "1h", 1.0)
    frame = load_solar_frame(spec)
    train_end = int(len(frame) * 0.6)
    scaler = fit_train_scaler(frame, spec, train_end)
    origins = np.load(args.chronos_origin)
    manifest = build_sample_manifest(frame, spec, args.seq_len, args.pred_len, origins)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.drop(columns="start_index").to_csv(args.output_dir / "sample_manifest.csv", index=False)
    model = FusionSFSolar(nwp_dim=len(NWP))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    shapes = {}
    losses = {}
    for epoch in range(args.max_epochs):
        model.train()
        dataset = SolarWindowDataset(frame, manifest[manifest.split == "train"], spec, scaler, args.mode)
        for batch_index, batch in enumerate(DataLoader(dataset, batch_size=args.batch_size, shuffle=True)):
            if batch_index >= args.limit_train_batches: break
            nwp = batch.get("future_nwp")
            pred = model(batch["history_power"], nwp, pred_len=args.pred_len)
            loss = criterion(pred, batch["target_power"])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            shapes["history_power"] = list(batch["history_power"].shape)
            shapes["future_nwp"] = list(nwp.shape) if nwp is not None else None
            shapes["prediction"] = list(pred.shape)
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
                pred_scaled = model(batch["history_power"], batch.get("future_nwp"), pred_len=args.pred_len)
                split_losses.append(float(criterion(pred_scaled, batch["target_power"])))
                if split == "test":
                    pred_raw = pred_scaled.numpy() * scaler.power_std + scaler.power_mean
                    predictions.append(pred_raw); targets.append(batch["target_power_raw"].numpy())
                    metadata.extend(subset.iloc[batch_index * args.batch_size : batch_index * args.batch_size + len(pred_raw)].to_dict("records"))
            losses[f"{split}_loss"] = float(np.mean(split_losses))
    pred = np.concatenate(predictions); target = np.concatenate(targets)
    if not np.isfinite(pred).all(): raise AssertionError("Non-finite predictions")
    meta = pd.DataFrame(metadata)
    if len(meta) != len(pred): raise AssertionError("Metadata/prediction row mismatch")
    chronos_forecast = pd.to_datetime(origins[:len(meta)]) + pd.Timedelta("1h")
    aligned = np.array_equal(pd.to_datetime(meta.forecast_start_timestamp).to_numpy(), chronos_forecast.to_numpy())
    if not aligned: raise AssertionError("FusionSF/Chronos forecast starts do not align")
    metrics = {"mae": float(np.mean(np.abs(pred - target))), "rmse": float(np.sqrt(np.mean((pred - target) ** 2))), "samples": len(pred), "chronos_aligned": aligned}
    np.save(args.output_dir / "predictions.npy", pred); np.save(args.output_dir / "targets.npy", target)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    meta.drop(columns="start_index").to_csv(args.output_dir / "prediction_metadata.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output_dir / "shapes.json").write_text(json.dumps(shapes, indent=2))
    (args.output_dir / "losses.json").write_text(json.dumps(losses, indent=2))
    run_config = {key.replace("_", "-"): str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(json.dumps({"mode": args.mode, "shapes": shapes, "metrics": metrics, "losses": losses}, indent=2))


if __name__ == "__main__": main()
