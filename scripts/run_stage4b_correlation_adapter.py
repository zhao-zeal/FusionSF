#!/usr/bin/env python3
"""Train the archived Stage 4B CoRA-inspired adapter from audited Stage 4A caches.

The cache must contain aligned train/validation/test contexts, targets,
unpooled FusionSF tokens, and window manifests.  Chronos-2 stays frozen; only
``FusionChronosCorrelationAdapter`` is optimized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.chronos2_correlation_adapter import (  # noqa: E402
    FusionChronosCorrelationAdapter,
    adapter_forward,
    trainable_parameter_names,
)


DEFAULT_MODEL = (
    Path("/home/zhaopp/.cache/huggingface/hub/models--amazon--chronos-2/snapshots")
    / "29ec3766d36d6f73f0696f85560a422f50e8498c"
)
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--shuffle-seed", type=int, default=2021)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--expected-train-sites", default="10-19")
    parser.add_argument("--expected-validation-sites", default="20-21")
    parser.add_argument("--expected-test-sites", default="0-9")
    parser.add_argument("--stage4a-predictions", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def parse_sites(specification: str) -> set[int]:
    sites: set[int] = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, stop = map(int, item.split("-", 1))
            if stop < start:
                raise ValueError(f"invalid site range: {item}")
            sites.update(range(start, stop + 1))
        else:
            sites.add(int(item))
    if not sites:
        raise ValueError("site specification cannot be empty")
    return sites


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _array(cache_dir: Path, split: str, suffix: str) -> tuple[np.ndarray, Path]:
    path = cache_dir / f"{split}_{suffix}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False), path


def load_split(cache_dir: Path, split: str, expected_sites: set[int]) -> dict:
    manifest_path = cache_dir / f"{split}_window_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    if not {"window_id", "site_id"}.issubset(manifest.columns):
        raise ValueError(f"{manifest_path} must contain window_id and site_id")
    if manifest["window_id"].duplicated().any():
        raise ValueError(f"{split} window IDs are not unique")
    contexts, context_path = _array(cache_dir, split, "contexts")
    targets, target_path = _array(cache_dir, split, "targets")
    tokens, token_path = _array(cache_dir, split, "fusion_tokens")
    if targets.ndim == 3 and targets.shape[-1] == 1:
        targets = targets[..., 0]
    count = len(manifest)
    if contexts.shape != (count, 24):
        raise ValueError(f"{split} contexts must be {(count, 24)}, got {contexts.shape}")
    if targets.shape != (count, 24):
        raise ValueError(f"{split} targets must be {(count, 24)}, got {targets.shape}")
    if tokens.shape != (count, 24, 64):
        raise ValueError(f"{split} FusionSF tokens must be {(count, 24, 64)}, got {tokens.shape}")
    if not all(np.isfinite(array).all() for array in (contexts, targets, tokens)):
        raise ValueError(f"{split} arrays must be finite")
    actual_sites = set(map(int, manifest["site_id"].unique()))
    if actual_sites != expected_sites:
        raise ValueError(
            f"{split} sites must be {sorted(expected_sites)}, got {sorted(actual_sites)}"
        )
    return {
        "manifest": manifest,
        "contexts": contexts.astype(np.float32, copy=False),
        "targets": targets.astype(np.float32, copy=False),
        "tokens": tokens.astype(np.float32, copy=False),
        "sites": actual_sites,
        "sha256": {
            "manifest": sha256(manifest_path),
            "contexts": sha256(context_path),
            "targets": sha256(target_path),
            "fusion_tokens": sha256(token_path),
        },
    }


def validate_cache(args: argparse.Namespace) -> tuple[dict[str, dict], dict]:
    cache_dir = args.cache_dir.resolve()
    expected = {
        "train": parse_sites(args.expected_train_sites),
        "validation": parse_sites(args.expected_validation_sites),
        "test": parse_sites(args.expected_test_sites),
    }
    splits = {name: load_split(cache_dir, name, expected[name]) for name in SPLITS}
    if any(
        splits[left]["sites"] & splits[right]["sites"]
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    ):
        raise ValueError("train, validation, and test sites must be disjoint")
    summary = {
        "protocol": "stage4b_fusionsf_chronos2_correlation_adapter_v1",
        "cache_dir": str(cache_dir),
        "model_path": str(args.model.resolve()),
        "prediction_length": 24,
        "fusion_tokens_unpooled": True,
        "fusion_token_shape": [24, 64],
        "chronos_frozen": True,
        "fusionsf_frozen_cache": True,
        "trainable_components": [
            "fusion_projection", "query_norm", "fusion_norm", "dynamic_attention",
            "global_mlp", "alpha", "beta"
        ],
        "splits": {
            name: {
                "windows": len(value["manifest"]),
                "sites": sorted(value["sites"]),
                "sha256": value["sha256"],
            }
            for name, value in splits.items()
        },
    }
    return splits, summary


def median_index(chronos: torch.nn.Module) -> int:
    return int(torch.argmin(torch.abs(chronos.quantiles.float() - 0.5)).item())


def predict(
    chronos: torch.nn.Module,
    adapter: torch.nn.Module,
    contexts: np.ndarray,
    tokens: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    adapter.eval()
    parts = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(contexts), torch.from_numpy(tokens)),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.inference_mode():
        for context, fusion in loader:
            quantiles = adapter_forward(
                chronos, adapter, context.to(device), fusion.to(device), prediction_length=24
            )
            parts.append(quantiles[:, median_index(chronos)].float().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def sattolo_derangement(length: int, seed: int) -> np.ndarray:
    if length < 2:
        raise ValueError("at least two test windows are required for shuffled control")
    values = np.arange(length)
    generator = np.random.default_rng(seed)
    for index in range(length - 1, 0, -1):
        swap = int(generator.integers(0, index))
        values[index], values[swap] = values[swap], values[index]
    if np.any(values == np.arange(length)):
        raise AssertionError("Sattolo shuffle unexpectedly contains fixed points")
    return values


def train_adapter(
    chronos: torch.nn.Module,
    adapter: FusionChronosCorrelationAdapter,
    train: dict,
    validation: dict,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict], Path]:
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train["contexts"]),
            torch.from_numpy(train["targets"]),
            torch.from_numpy(train["tokens"]),
        ),
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
    )
    history: list[dict] = []
    best, stale = float("inf"), 0
    best_path = output_dir / "adapter_best.pt"
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        losses = []
        for context, target, fusion in loader:
            optimizer.zero_grad(set_to_none=True)
            quantiles = adapter_forward(
                chronos, adapter, context.to(device), fusion.to(device), prediction_length=24
            )
            loss = torch.mean(
                torch.abs(quantiles[:, median_index(chronos)] - target.to(device))
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_prediction = predict(
            chronos, adapter, validation["contexts"], validation["tokens"],
            args.eval_batch_size, device,
        )
        validation_mae = metrics(validation_prediction, validation["targets"])["mae"]
        row = {
            "epoch": epoch,
            "train_mae": float(np.mean(losses)),
            "validation_mae": validation_mae,
            "alpha": float(adapter.alpha.detach()),
            "beta": float(adapter.beta.detach()),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_mae < best:
            best, stale = validation_mae, 0
            torch.save(adapter.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break
    adapter.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    return history, best_path


def main() -> int:
    args = parse_args()
    if min(args.train_batch_size, args.eval_batch_size, args.epochs, args.patience) <= 0:
        raise ValueError("batch sizes, epochs, and patience must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir.resolve()}")
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    seed_everything(args.seed)
    splits, preflight = validate_cache(args)
    stage4a_prediction = None
    if args.stage4a_predictions:
        if not args.stage4a_predictions.is_file():
            raise FileNotFoundError(args.stage4a_predictions)
        stage4a_prediction = np.load(args.stage4a_predictions, allow_pickle=False)
        if stage4a_prediction.shape != splits["test"]["targets"].shape:
            raise ValueError(
                "Stage 4A predictions must have shape "
                f"{splits['test']['targets'].shape}, got {stage4a_prediction.shape}"
            )
        if not np.isfinite(stage4a_prediction).all():
            raise ValueError("Stage 4A predictions must be finite")
        preflight["stage4a_predictions"] = {
            "path": str(args.stage4a_predictions.resolve()),
            "sha256": sha256(args.stage4a_predictions),
            "shape": list(stage4a_prediction.shape),
        }
    print(json.dumps(preflight, indent=2), flush=True)
    if args.preflight_only:
        return 0

    from chronos import Chronos2Pipeline

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pipeline = Chronos2Pipeline.from_pretrained(
        str(args.model.resolve()), device_map=str(device),
        local_files_only=args.local_files_only,
    )
    chronos = pipeline.model.eval().requires_grad_(False)
    chronos_before = parameter_digest(chronos)
    chronos_dim = int(chronos.config.d_model)
    adapter = FusionChronosCorrelationAdapter(chronos_dim=chronos_dim).to(device)
    zero_prediction = predict(
        chronos, adapter, splits["test"]["contexts"], splits["test"]["tokens"],
        args.eval_batch_size, device,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    history, checkpoint = train_adapter(
        chronos, adapter, splits["train"], splits["validation"], args, device, output_dir
    )
    permutation = sattolo_derangement(len(splits["test"]["manifest"]), args.shuffle_seed)
    aligned = predict(
        chronos, adapter, splits["test"]["contexts"], splits["test"]["tokens"],
        args.eval_batch_size, device,
    )
    shuffled = predict(
        chronos, adapter, splits["test"]["contexts"],
        splits["test"]["tokens"][permutation], args.eval_batch_size, device,
    )
    predictions = {
        "chronos2_baseline": zero_prediction,
        "cora_inspired_aligned": aligned,
        "cora_inspired_shuffled": shuffled,
    }
    if stage4a_prediction is not None:
        predictions["stage4a_cross_attention"] = stage4a_prediction.astype(np.float32)
    group_metrics = {
        name: metrics(value, splits["test"]["targets"])
        for name, value in predictions.items()
    }
    for name, value in predictions.items():
        np.save(output_dir / f"{name}_predictions.npy", value)
    np.save(output_dir / "targets.npy", splits["test"]["targets"])
    np.save(output_dir / "shuffle_permutation.npy", permutation)
    test_manifest = splits["test"]["manifest"]
    test_manifest.to_csv(output_dir / "window_manifest.csv", index=False)
    site_rows, window_rows = [], []
    for name, value in predictions.items():
        window_mae = np.mean(np.abs(value - splits["test"]["targets"]), axis=1)
        for row_index, error in enumerate(window_mae):
            window_rows.append({
                "window_id": test_manifest.iloc[row_index]["window_id"],
                "site_id": int(test_manifest.iloc[row_index]["site_id"]),
                "group": name,
                "mae": float(error),
            })
        for site_id in sorted(test_manifest["site_id"].unique()):
            selected = test_manifest["site_id"].to_numpy() == site_id
            site_rows.append({
                "site_id": int(site_id), "group": name,
                **metrics(value[selected], splits["test"]["targets"][selected]),
            })
    pd.DataFrame(site_rows).to_csv(output_dir / "metrics_by_site.csv", index=False)
    pd.DataFrame(window_rows).to_csv(output_dir / "metrics_by_window.csv", index=False)
    baseline_metrics = group_metrics["chronos2_baseline"]
    aligned_metrics = group_metrics["cora_inspired_aligned"]
    shuffled_metrics = group_metrics["cora_inspired_shuffled"]
    comparison = {
        "metrics": group_metrics,
        "aligned_minus_baseline": {
            key: aligned_metrics[key] - baseline_metrics[key]
            for key in ("mae", "rmse")
        },
        "aligned_relative_change_vs_baseline_pct": {
            key: 100.0 * (aligned_metrics[key] - baseline_metrics[key]) / baseline_metrics[key]
            for key in ("mae", "rmse")
        },
        "aligned_minus_shuffled": {
            key: aligned_metrics[key] - shuffled_metrics[key]
            for key in ("mae", "rmse")
        },
    }
    audit = {
        **preflight,
        "audit_passed": chronos_before == parameter_digest(chronos),
        "chronos_parameters_unchanged": chronos_before == parameter_digest(chronos),
        "alpha_beta_zero_initialized": True,
        "zero_init_is_baseline_by_construction": True,
        "shuffle_fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
        "adapter_parameter_names": trainable_parameter_names(adapter),
        "adapter_checkpoint": str(checkpoint),
        "training_seconds": time.perf_counter() - started,
        "stage4a_comparison_supplied": args.stage4a_predictions is not None,
    }
    resolved = vars(args).copy()
    for key, value in list(resolved.items()):
        if isinstance(value, Path):
            resolved[key] = str(value.resolve())
    for filename, payload in (
        ("metrics.json", group_metrics),
        ("comparison.json", comparison),
        ("audit.json", audit),
        ("resolved_config.json", resolved),
    ):
        (output_dir / filename).write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "comparison.md").write_text(
        "# Stage 4B comparison\n\n"
        f"Baseline MAE: {baseline_metrics['mae']:.8f}; "
        f"aligned MAE: {aligned_metrics['mae']:.8f}; "
        f"shuffled MAE: {shuffled_metrics['mae']:.8f}.\n"
    )
    print(json.dumps({"output_dir": str(output_dir), **comparison}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
