#!/usr/bin/env python3
"""Evaluate one explicit fixed_v1 checkpoint without invoking training."""

import argparse
import json
import os
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PROJECT_ROOT", str(ROOT))
user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)
try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    from pip._vendor import pkg_resources
    sys.modules["pkg_resources"] = pkg_resources

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.utils.experiment_records import (
    append_checkpoint_evaluation, record_run_context, save_scaler_state, save_test_outputs,
)


MODES = (
    "full_modalities", "missing_satellite", "missing_nwp", "missing_satellite_and_nwp",
    "random_satellite_missing", "random_nwp_missing",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES[:4]))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def empty_output_dict():
    return {key: [] for key in (
        "inputs", "outputs", "targets", "site_ids", "input_start_timestamps",
        "input_end_timestamps", "forecast_start_timestamps", "forecast_end_timestamps",
        "forecast_timestamps", "forecast_horizons",
    )}


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {args.output_dir}")
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(args.checkpoint_path)
    cfg = OmegaConf.load(args.config)
    if cfg.datamodule.dataset.data_pipeline.version != "fixed_v1":
        raise ValueError("checkpoint evaluation requires fixed_v1")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    module = hydra.utils.instantiate(cfg.pl_module)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    model = module.model.to(args.device).eval()
    args.output_dir.mkdir(parents=True)
    record_run_context(args.output_dir, cfg)
    save_scaler_state(args.output_dir, datamodule.data_all.scaler_state)
    loader = DataLoader(
        datamodule.data_test, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    results = {}
    for mode in args.modes:
        output = empty_output_dict()
        with torch.no_grad():
            for batch in loader:
                prediction = model(
                    batch["stl_input"].float().to(args.device),
                    batch["stl_coords"].float().to(args.device),
                    batch["ts_input"].float().to(args.device),
                    batch["ts_coords"].float().to(args.device),
                    batch["ts_time"].float().to(args.device),
                    batch["ec_input"].float().to(args.device), mask=False,
                    modality_availability=batch["modality_availability"].to(args.device),
                    evaluation_mode=mode,
                ).mean(dim=2).cpu().numpy()
                target = batch["ts_target"].unsqueeze(-1).numpy()
                output["inputs"].append(batch["ts_input"].numpy())
                output["outputs"].append(prediction)
                output["targets"].append(target)
                for output_key, batch_key in {
                    "site_ids": "site_id", "input_start_timestamps": "input_start_timestamp",
                    "input_end_timestamps": "input_end_timestamp",
                    "forecast_start_timestamps": "forecast_start_timestamp",
                    "forecast_end_timestamps": "forecast_end_timestamp",
                    "forecast_timestamps": "forecast_timestamps",
                }.items():
                    output[output_key].append(batch[batch_key].numpy())
                output["forecast_horizons"].append(
                    np.tile(np.arange(1, target.shape[1] + 1), (len(target), 1))
                )
        mode_dir = args.output_dir / mode
        mode_dir.mkdir()
        metrics = save_test_outputs(mode_dir, output)
        results[mode] = metrics
        append_checkpoint_evaluation(
            ROOT / "experiments/experiment_registry.csv", cfg,
            f"{cfg.experiment_id}__eval_{mode}", mode, mode_dir, metrics,
            str(args.checkpoint_path), datamodule,
        )
    manifest = {
        "source_experiment_id": cfg.experiment_id,
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metric": str(cfg.callbacks.model_checkpoint.monitor),
        "modes": args.modes,
        "metrics": results,
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
