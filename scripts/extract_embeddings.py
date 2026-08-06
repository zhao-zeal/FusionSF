#!/usr/bin/env python3
"""Extract aligned FusionSF representations from an explicit checkpoint; never trains."""

import argparse
import csv
import json
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Resolved YAML config")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--embedding-type", choices=["ts", "fusion", "both"], default="both")
    parser.add_argument("--pooling", nargs="+", choices=["none", "mean", "last"], default=["none", "mean"])
    parser.add_argument("--evaluation-mode", default="full_modalities")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--site-seen-during-training", action="store_true")
    return parser.parse_args()


def git_commit():
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()


def update_registry(cfg, args, embedding_dim):
    registry = ROOT / "experiments/experiment_registry.csv"
    with registry.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames, rows = reader.fieldnames, list(reader)
    matched = False
    for row in rows:
        if row["experiment_id"] == cfg.experiment_id:
            row.update({
                "embedding_extracted": "true", "embedding_types": args.embedding_type,
                "embedding_pooling": "|".join(args.pooling), "embedding_dim": embedding_dim,
                "embedding_output_dir": str(args.output_dir.resolve()),
                "embedding_checkpoint": str(args.checkpoint_path.resolve()),
            })
            matched = True
    if not matched:
        raise KeyError(f"experiment_id is not registered: {cfg.experiment_id}")
    with registry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"embedding output already exists: {args.output_dir}")
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(args.checkpoint_path)
    cfg = OmegaConf.load(args.config)
    if cfg.datamodule.dataset.data_pipeline.version != "fixed_v1":
        raise ValueError("fixed_v1 is required by default; legacy embeddings need a separate marked output")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    training_site_ids = sorted(int(site["site"]) for site in datamodule.data_all.data_sp)
    extraction_dataset = getattr(datamodule, "data_test_all", None) or datamodule.data_all
    extraction_site_ids = sorted(int(site["site"]) for site in extraction_dataset.data_sp)
    pl_module = hydra.utils.instantiate(cfg.pl_module)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    pl_module.load_state_dict(checkpoint["state_dict"], strict=True)
    model = pl_module.model.eval()
    args.output_dir.mkdir(parents=True)
    split_datasets = {
        "train": datamodule.data_train,
        "validation": datamodule.data_val,
        "test": datamodule.data_test,
    }
    common_metadata = {
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metric": str(cfg.callbacks.model_checkpoint.monitor),
        "git_commit": git_commit(),
        "data_pipeline": "fixed_v1",
        "evaluation_mode": args.evaluation_mode,
        "training_site_ids": training_site_ids,
        "extraction_site_ids": extraction_site_ids,
        "site_seen_during_training": args.site_seen_during_training,
    }
    all_metadata = []
    embedding_dim = None
    for split in args.splits:
        dataset = split_datasets[split]
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        collected = {(kind, pool): [] for kind in (("ts", "fusion") if args.embedding_type == "both" else (args.embedding_type,)) for pool in args.pooling}
        metadata_rows = []
        with torch.no_grad():
            for batch in loader:
                for pooling in args.pooling:
                    values = model.extract_embeddings(
                        batch, args.embedding_type, pooling, args.evaluation_mode
                    )
                    for kind, value in values.items():
                        collected[(kind, pooling)].append(value.cpu().numpy())
                count = len(batch["site_id"])
                for i in range(count):
                    metadata_rows.append({
                        "site_id": int(batch["site_id"][i]),
                        "input_start_timestamp": pd.Timestamp(int(batch["input_start_timestamp"][i]), unit="ns"),
                        "input_end_timestamp": pd.Timestamp(int(batch["input_end_timestamp"][i]), unit="ns"),
                        "forecast_start_timestamp": pd.Timestamp(int(batch["forecast_start_timestamp"][i]), unit="ns"),
                        "forecast_end_timestamp": pd.Timestamp(int(batch["forecast_end_timestamp"][i]), unit="ns"),
                        "split": split,
                        "forecast_horizon": int(cfg.datamodule.dataset.pred_len),
                        "embedding_type": args.embedding_type,
                        "pooling_type": "|".join(args.pooling),
                        "checkpoint_path": str(args.checkpoint_path.resolve()),
                        "git_commit": common_metadata["git_commit"],
                        "training_site_ids": "|".join(map(str, training_site_ids)),
                        "extraction_site_ids": "|".join(map(str, extraction_site_ids)),
                        "site_seen_during_training": args.site_seen_during_training,
                    })
        split_dir = args.output_dir / "embeddings" / split
        split_dir.mkdir(parents=True)
        arrays = {}
        for (kind, pooling), chunks in collected.items():
            value = np.concatenate(chunks, axis=0)
            if len(value) != len(metadata_rows) or not np.isfinite(value).all():
                raise AssertionError(f"invalid or misaligned {kind}/{pooling} embedding")
            suffix = "" if pooling == "none" else f"_{pooling}"
            np.save(split_dir / f"{kind}_embedding{suffix}.npy", value)
            arrays[(kind, pooling)] = value
            embedding_dim = value.shape[-1]
        pd.DataFrame(metadata_rows).to_csv(split_dir / "metadata.csv", index=False)
        config_record = dict(common_metadata, split=split, shapes={f"{k[0]}_{k[1]}": list(v.shape) for k, v in arrays.items()})
        (split_dir / "embedding_config.yaml").write_text(OmegaConf.to_yaml(config_record), encoding="utf-8")

        preferred_kind = "fusion" if args.embedding_type in {"fusion", "both"} else "ts"
        preferred_pool = "mean" if "mean" in args.pooling else args.pooling[0]
        frame = pd.DataFrame(metadata_rows)
        all_metadata.append(frame)
        covariate_dir = args.output_dir / "chronos_covariates"
        covariate_dir.mkdir(exist_ok=True)
        np.savez(
            covariate_dir / f"{split}_embeddings.npz",
            embedding=arrays[(preferred_kind, preferred_pool)],
            site_id=frame.site_id.to_numpy(),
            forecast_start_timestamp=np.asarray(frame.forecast_start_timestamp.astype(str), dtype="U32"),
            input_end_timestamp=np.asarray(frame.input_end_timestamp.astype(str), dtype="U32"),
        )
        frame.to_csv(covariate_dir / f"{split}_metadata.csv", index=False)
    pd.concat(all_metadata, ignore_index=True).to_csv(
        args.output_dir / "chronos_covariates/metadata.csv", index=False
    )
    (args.output_dir / "extraction_manifest.json").write_text(
        json.dumps(common_metadata, indent=2) + "\n", encoding="utf-8"
    )
    update_registry(cfg, args, embedding_dim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
