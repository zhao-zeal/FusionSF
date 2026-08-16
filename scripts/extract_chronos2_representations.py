#!/usr/bin/env python3
"""Extract frozen Chronos-2 encoder representations from history-only power windows."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    "/home/zhaopp/.cache/huggingface/hub/models--amazon--chronos-2/"
    "snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-windows", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    window_dir, output_dir = args.window_dir.resolve(), args.output_dir.resolve()
    source = json.loads((window_dir / "window_manifest.json").read_text(encoding="utf-8"))
    if source.get("contains_future_target") is not False:
        raise ValueError("source manifest must prove that no future target is present")
    output_dir.mkdir(parents=True, exist_ok=False)

    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(
        str(args.model.resolve()), device_map=args.device_map, local_files_only=True
    )
    pipeline.model.eval()
    pipeline.model.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in pipeline.model.parameters())
    if any(parameter.requires_grad for parameter in pipeline.model.parameters()):
        raise RuntimeError("Chronos-2 parameters must be frozen")

    split_records = {}
    raw_shapes = set()
    started = time.perf_counter()
    for split in ("train", "validation", "test"):
        power_path = window_dir / f"{split}_power.npy"
        power = np.load(power_path, mmap_mode="r")
        row_count = len(power) if args.max_windows is None else min(len(power), args.max_windows)
        pooled_chunks = []
        for start in range(0, row_count, args.batch_size):
            stop = min(start + args.batch_size, row_count)
            context = torch.from_numpy(
                np.array(power[start:stop, :, 0], dtype=np.float32, copy=True)
            ).unsqueeze(1)
            embeddings, _ = pipeline.embed(context, batch_size=args.batch_size, context_length=24)
            raw = torch.stack([embedding.squeeze(0) for embedding in embeddings])
            raw_shapes.add(tuple(raw.shape[1:]))
            pooled_chunks.append(raw.mean(dim=1).float().numpy())
        pooled = np.concatenate(pooled_chunks, axis=0)
        if pooled.shape != (row_count, 768) or not np.isfinite(pooled).all():
            raise ValueError(f"invalid {split} pooled representation: {pooled.shape}")
        output_path = output_dir / f"{split}_mean.npy"
        np.save(output_path, pooled)
        split_records[split] = {
            "rows": int(row_count),
            "source_power": str(power_path.resolve()),
            "source_power_sha256": sha256(power_path),
            "output": str(output_path.resolve()),
            "output_sha256": sha256(output_path),
            "nan_count": int(np.isnan(pooled).sum()),
            "inf_count": int(np.isinf(pooled).sum()),
        }
    if len(raw_shapes) != 1:
        raise ValueError(f"Chronos raw embedding shapes differ across batches: {raw_shapes}")
    raw_shape = list(next(iter(raw_shapes)))
    manifest = {
        "protocol": "frozen_chronos2_history_power_mean_v1",
        "source_manifest": str((window_dir / "window_manifest.json").resolve()),
        "model": str(args.model.resolve()),
        "chronos_version": "2.2.2",
        "model_eval": not pipeline.model.training,
        "parameters_frozen": True,
        "parameter_count": int(parameter_count),
        "input": "history_power_only",
        "future_target_used": False,
        "hidden_state_source": "Chronos2Model encoder final last_hidden_state",
        "raw_shape_per_window": [1, *raw_shape],
        "raw_tokens": "context patches + REG token + masked output patch token",
        "pooling": "mean",
        "final_shape": [None, 768],
        "max_windows": args.max_windows,
        "elapsed_seconds": time.perf_counter() - started,
        "splits": split_records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
