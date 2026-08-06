#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fusionsf_solar_smoke import canonicalize_origins, make_spec
from src.datasets.solar_energy_dataset import (
    SolarWindowDataset, build_sample_manifest, fit_train_scaler, load_solar_frame,
    validate_reference_alignment,
)
from src.models.fusionsf_solar import FusionSFSolar


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad_epochs = float("inf"), 0

    def update(self, value: float) -> tuple[bool, bool]:
        improved = value < self.best - self.min_delta
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gefcom", "skippd", "csg"), required=True)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--mode", choices=("power", "power_weather"), required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--weather-path", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--chronos-origin", type=Path)
    source.add_argument("--shared-manifest", type=Path)
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim-head", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def manifest_hash(test_manifest: pd.DataFrame) -> str:
    columns = ["dataset", "site_id", "split", "input_start_timestamp", "input_end_timestamp", "forecast_start_timestamp", "forecast_end_timestamp", "seq_len", "pred_len"]
    payload = test_manifest[columns].sort_values("input_end_timestamp").to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def build_alignment_report(manifest, origins, spec):
    validate_reference_alignment(manifest, origins, spec)
    test = manifest[manifest.split == "test"].sort_values("input_end_timestamp")
    delta = pd.Timedelta(spec.frequency)
    target_continuous = all(
        row.forecast_start_timestamp == row.input_end_timestamp + delta
        and row.forecast_end_timestamp == row.forecast_start_timestamp + (row.pred_len - 1) * delta
        for row in test.itertuples()
    )
    if not target_continuous:
        raise ValueError("Full target timestamp continuity check failed")
    report = {
        "passed": True,
        "reference_origin_count": int(len(origins)),
        "actual_origin_count": int(len(test)),
        "unique_reference_origins": int(pd.DatetimeIndex(pd.to_datetime(origins)).nunique()),
        "unique_actual_origins": int(pd.DatetimeIndex(test.input_end_timestamp).nunique()),
        "missing_origins": 0,
        "extra_origins": 0,
        "duplicate_origins": 0,
        "frequency": spec.frequency,
        "pred_len": int(test.pred_len.iloc[0]),
        "forecast_start_rule_passed": True,
        "forecast_end_rule_passed": True,
        "complete_target_intervals_continuous": True,
        "test_manifest_sha256": manifest_hash(test),
    }
    return report


def _manifest_from_file(path: Path, frame, spec):
    manifest = pd.read_csv(path, parse_dates=["input_start_timestamp", "input_end_timestamp", "forecast_start_timestamp", "forecast_end_timestamp"])
    timestamp_to_index = {timestamp: index for index, timestamp in enumerate(frame[spec.timestamp_col])}
    missing = set(manifest.input_start_timestamp) - set(timestamp_to_index)
    if missing:
        raise ValueError(f"Shared manifest contains {len(missing)} unknown input timestamps")
    manifest["start_index"] = manifest.input_start_timestamp.map(timestamp_to_index).astype(int)
    origins = manifest.loc[manifest.split == "test", "input_end_timestamp"].to_numpy(dtype="datetime64[ns]")
    validate_reference_alignment(manifest, origins, spec)
    return manifest, origins


def make_loader(frame, manifest, spec, scaler, mode, args, shuffle):
    return DataLoader(
        SolarWindowDataset(frame, manifest, spec, scaler, mode),
        batch_size=args.batch_size, shuffle=shuffle, drop_last=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if torch.is_tensor(value)}


def evaluate_loader(model, loader, scaler, device):
    losses, absolute_errors, squared_errors = [], [], []
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensors = move_batch(batch, device)
            pred = model(tensors["history_power"], tensors["history_time_features"], tensors["future_time_features"], tensors.get("future_weather"))
            losses.append(float(criterion(pred, tensors["target_power"])))
            pred_raw = pred * scaler.power_std + scaler.power_mean
            error = pred_raw - tensors["target_power_raw"]
            absolute_errors.append(error.abs().cpu().numpy())
            squared_errors.append(error.square().cpu().numpy())
    ae, se = np.concatenate(absolute_errors), np.concatenate(squared_errors)
    return float(np.mean(losses)), float(ae.mean()), float(np.sqrt(se.mean()))


def save_checkpoint(path, model, optimizer, epoch, args, optimizer_steps):
    torch.save({
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "optimizer_steps": optimizer_steps, "args": vars(args),
    }, path)


def load_best_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def prediction_metrics(prediction, target, capacity, daylight_mask):
    finite = np.isfinite(prediction)
    error = prediction - target
    ae, se = np.abs(error), error ** 2
    day = np.broadcast_to(daylight_mask[..., None], prediction.shape)
    return {
        "mae": float(np.mean(ae)), "rmse": float(np.sqrt(np.mean(se))),
        "nmae": float(np.mean(ae) / capacity), "nrmse": float(np.sqrt(np.mean(se)) / capacity),
        "daylight_mae": float(np.mean(ae[day])) if day.any() else None,
        "daylight_rmse": float(np.sqrt(np.mean(se[day]))) if day.any() else None,
        "prediction_mean": float(np.mean(prediction)), "prediction_std": float(np.std(prediction)),
        "prediction_min": float(np.min(prediction)), "prediction_max": float(np.max(prediction)),
        "negative_prediction_ratio": float(np.mean(prediction < 0)),
        "all_zero_forecast_ratio": float(np.mean(np.all(np.isclose(prediction, 0.0), axis=(1, 2)))),
        "nan_count": int(np.isnan(prediction).sum()), "inf_count": int(np.isinf(prediction).sum()),
        "finite_count": int(finite.sum()),
    }


def collect_daylight_mask(frame, test_manifest, spec):
    weather_column = "VAR169" if spec.dataset == "GEFCom" else "shortwave_radiation"
    values = frame[weather_column].to_numpy()
    masks = []
    for row in test_manifest.itertuples():
        start = int(row.start_index) + int(row.seq_len)
        masks.append(values[start : start + int(row.pred_len)] > 0)
    return np.stack(masks)


def horizon_metrics(raw, clipped, target):
    rows = []
    for horizon in range(raw.shape[1]):
        row = {"horizon": horizon + 1}
        for name, prediction in (("raw", raw), ("clipped", clipped)):
            error = prediction[:, horizon] - target[:, horizon]
            row[f"{name}_mae"] = float(np.mean(np.abs(error)))
            row[f"{name}_rmse"] = float(np.sqrt(np.mean(error ** 2)))
        rows.append(row)
    return pd.DataFrame(rows)


def naive_baselines(frame, test_manifest, spec, daylight_mask):
    power = frame[spec.power_col].to_numpy(np.float32)
    period = 24 if spec.frequency == "1h" else 96
    persistence, seasonal, targets = [], [], []
    for row in test_manifest.itertuples():
        future = int(row.start_index) + int(row.seq_len)
        pred_len = int(row.pred_len)
        persistence.append(np.repeat(power[future - 1], pred_len))
        last_day = power[future - period : future]
        seasonal.append(np.resize(last_day, pred_len))
        targets.append(power[future : future + pred_len])
    target = np.stack(targets)[..., None]
    result = {}
    for name, values in (("persistence", persistence), ("daily_seasonal_naive", seasonal)):
        prediction = np.clip(np.stack(values)[..., None], 0.0, spec.capacity)
        result[name] = prediction_metrics(prediction, target, spec.capacity, daylight_mask)
    return result


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main():
    args = parse_args()
    if args.batch_size * args.gradient_accumulation != 32:
        raise ValueError("effective_batch_size must equal 32")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    spec = make_spec(args)
    frame = load_solar_frame(spec)
    scaler = fit_train_scaler(frame, spec, int(len(frame) * 0.6))
    if args.chronos_origin:
        origins = canonicalize_origins(np.load(args.chronos_origin), spec)
        manifest = build_sample_manifest(frame, spec, args.seq_len, args.pred_len, origins)
    else:
        manifest, origins = _manifest_from_file(args.shared_manifest, frame, spec)
    active_semantics = "none" if args.mode == "power" else spec.covariate_semantics
    manifest["covariate_semantics"] = active_semantics
    alignment = build_alignment_report(manifest, origins, spec)
    (args.output_dir / "alignment_report.json").write_text(json.dumps(alignment, indent=2))
    manifest.drop(columns="start_index").to_csv(args.output_dir / "sample_manifest.csv", index=False)

    loaders = {split: make_loader(frame, manifest[manifest.split == split].reset_index(drop=True), spec, scaler, args.mode, args, split == "train") for split in ("train", "validation", "test")}
    model = FusionSFSolar(len(spec.weather_cols), time_dim=8, seq_len=args.seq_len, dim=args.dim, depth=args.depth, heads=args.heads, dim_head=args.dim_head, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion, stopper = nn.MSELoss(), EarlyStopping(args.early_stopping_patience, args.early_stopping_min_delta)
    history, optimizer_steps, best_epoch, best_val_rmse = [], 0, None, None
    stop_reason, stopped_epoch = "max_epochs_reached", args.max_epochs
    for epoch in range(1, args.max_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        train_losses, train_ae = [], []
        for batch_index, batch in enumerate(loaders["train"]):
            tensors = move_batch(batch, device)
            prediction = model(tensors["history_power"], tensors["history_time_features"], tensors["future_time_features"], tensors.get("future_weather"))
            loss = criterion(prediction, tensors["target_power"])
            (loss / args.gradient_accumulation).backward()
            train_losses.append(float(loss.detach()))
            prediction_raw = prediction.detach() * scaler.power_std + scaler.power_mean
            train_ae.append(float((prediction_raw - tensors["target_power_raw"]).abs().mean()))
            is_step = (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(loaders["train"])
            if is_step:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
        val_loss, val_mae, val_rmse = evaluate_loader(model, loaders["validation"], scaler, device)
        improved, should_stop = stopper.update(val_mae)
        row = {"epoch": epoch, "optimizer_steps": optimizer_steps, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": float(np.mean(train_losses)), "train_mae": float(np.mean(train_ae)), "val_loss": val_loss, "val_mae": val_mae, "val_rmse": val_rmse}
        history.append(row); pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
        save_checkpoint(args.output_dir / "last.ckpt", model, optimizer, epoch, args, optimizer_steps)
        if improved:
            best_epoch, best_val_rmse = epoch, val_rmse
            save_checkpoint(args.output_dir / "best.ckpt", model, optimizer, epoch, args, optimizer_steps)
        print(json.dumps(row), flush=True)
        if should_stop:
            stop_reason, stopped_epoch = "early_stopping", epoch
            break

    best_checkpoint = load_best_checkpoint(args.output_dir / "best.ckpt", model, device)
    test_manifest = manifest[manifest.split == "test"].sort_values("input_end_timestamp").reset_index(drop=True)
    raw_predictions, targets, metadata = [], [], []
    offset = 0; model.eval()
    with torch.no_grad():
        for batch in loaders["test"]:
            tensors = move_batch(batch, device)
            prediction = model(tensors["history_power"], tensors["history_time_features"], tensors["future_time_features"], tensors.get("future_weather"))
            raw = prediction.cpu().numpy() * scaler.power_std + scaler.power_mean
            raw_predictions.append(raw); targets.append(tensors["target_power_raw"].cpu().numpy())
            rows = test_manifest.iloc[offset : offset + len(raw)].copy(); offset += len(raw)
            rows["model_family"] = model.model_family; rows["model_class"] = type(model).__name__
            rows["architecture_version"] = model.architecture_version; rows["covariate_semantics"] = active_semantics
            rows["oracle_covariates"] = bool(args.mode == "power_weather" and spec.covariate_semantics == "future_reanalysis")
            metadata.append(rows)
    raw, target = np.concatenate(raw_predictions), np.concatenate(targets)
    clipped = np.clip(raw, 0.0, spec.capacity)
    metadata = pd.concat(metadata, ignore_index=True)
    if len(raw) != len(target) or len(raw) != len(metadata) or len(raw) != len(test_manifest):
        raise ValueError("Prediction/target/metadata/full-test-manifest alignment failed")
    np.save(args.output_dir / "predictions_raw.npy", raw); np.save(args.output_dir / "predictions_clipped.npy", clipped); np.save(args.output_dir / "targets.npy", target)
    metadata.drop(columns="start_index").to_csv(args.output_dir / "prediction_metadata.csv", index=False)
    daylight = collect_daylight_mask(frame, test_manifest, spec)
    metrics = {"raw": prediction_metrics(raw, target, spec.capacity, daylight), "clipped": prediction_metrics(clipped, target, spec.capacity, daylight)}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    horizon_metrics(raw, clipped, target).to_csv(args.output_dir / "metrics_by_horizon.csv", index=False)
    baselines = naive_baselines(frame, test_manifest, spec, daylight)
    (args.output_dir / "baseline_metrics.json").write_text(json.dumps(baselines, indent=2))

    summary = {
        "best_epoch": best_epoch, "stopped_epoch": stopped_epoch, "stop_reason": stop_reason,
        "best_val_mae": stopper.best, "best_val_rmse": best_val_rmse,
        "epochs_completed": len(history), "optimizer_steps": optimizer_steps,
        "best_checkpoint_epoch_reloaded": int(best_checkpoint["epoch"]),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "training_budget_may_be_insufficient": bool(best_epoch is not None and best_epoch > 90),
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update({"model_family": model.model_family, "model_class": type(model).__name__, "architecture_version": model.architecture_version, "covariate_semantics": active_semantics, "oracle_covariates": bool(args.mode == "power_weather" and spec.covariate_semantics == "future_reanalysis"), "optimizer": "AdamW", "monitor": "val/mae", "monitor_mode": "min", "scheduler": None, "code_commit": git_commit()})
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    run_manifest = {"experiment_id": args.output_dir.name, "status": "preliminary_seed42", **run_config, **summary, "raw_mae": metrics["raw"]["mae"], "raw_rmse": metrics["raw"]["rmse"], "clipped_mae": metrics["clipped"]["mae"], "clipped_rmse": metrics["clipped"]["rmse"], "manifest_test_rows": len(test_manifest), "test_manifest_sha256": alignment["test_manifest_sha256"], "output_dir": str(args.output_dir)}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
    print(json.dumps({"run_summary": summary, "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__": main()
