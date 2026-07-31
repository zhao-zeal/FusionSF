"""Non-overwriting, traceable fixed_v1 experiment output helpers."""

import json
import csv
import platform
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.metrics.forecast_metrics import compute_forecast_metrics, metrics_by_horizon


def _command(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"


def record_run_context(output_dir: Path, cfg) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config_resolved.yaml"
    if config_path.exists():
        raise FileExistsError(f"experiment output already contains a resolved config: {output_dir}")
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    (output_dir / "command.txt").write_text(
        " ".join(shlex.quote(value) for value in sys.argv) + "\n", encoding="utf-8"
    )
    status = _command(["git", "status", "--short"])
    git_info = {
        "commit": _command(["git", "rev-parse", "HEAD"]),
        "branch": _command(["git", "branch", "--show-current"]),
        "dirty_worktree": bool(status),
        "status": status.splitlines(),
    }
    (output_dir / "git_info.json").write_text(json.dumps(git_info, indent=2) + "\n", encoding="utf-8")
    environment = [
        f"python={sys.version}", f"platform={platform.platform()}",
        "\n[pip freeze]\n" + _command([sys.executable, "-m", "pip", "freeze"]),
        "\n[nvidia-smi]\n" + _command(["nvidia-smi"]),
    ]
    (output_dir / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")
    notes = [f"dirty_worktree: {str(git_info['dirty_worktree']).lower()}"]
    if git_info["dirty_worktree"]:
        notes.append("Run used uncommitted code; see git_info.json for exact status.")
    (output_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def save_scaler_state(output_dir: Path, scaler_state: dict) -> None:
    serializable, arrays = {}, {}
    for modality, state in scaler_state.items():
        if isinstance(state, dict):
            serializable[modality] = {k: v for k, v in state.items() if k not in {"mean", "scale"}}
            for name in ("mean", "scale"):
                if name in state:
                    arrays[f"{modality}_{name}"] = np.asarray(state[name])
        else:
            serializable[modality] = state
    np.savez(Path(output_dir) / "scalers.npz", **arrays)
    (Path(output_dir) / "scaler_metadata.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )


def save_test_outputs(output_dir: Path, out_dict: dict) -> dict:
    output_dir = Path(output_dir)
    arrays = {key: np.concatenate(value, axis=0) for key, value in out_dict.items() if value}
    required = {"outputs", "targets", "site_ids", "forecast_start_timestamps"}
    if not required.issubset(arrays):
        raise ValueError(f"missing aligned test arrays: {sorted(required - arrays.keys())}")
    count = len(arrays["outputs"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("saved test arrays are not row-aligned")
    filename_map = {"outputs": "predictions", "forecast_timestamps": "timestamps"}
    for name, value in arrays.items():
        np.save(output_dir / f"{filename_map.get(name, name)}.npy", value)
    raw_prediction = arrays["outputs"]
    clipped_prediction = np.clip(raw_prediction, 0.0, None)
    np.save(output_dir / "predictions_raw.npy", raw_prediction)
    np.save(output_dir / "predictions_clipped.npy", clipped_prediction)
    raw_metrics = compute_forecast_metrics(raw_prediction, arrays["targets"])
    clipped_metrics = compute_forecast_metrics(clipped_prediction, arrays["targets"])
    metrics = dict(raw_metrics)
    metrics.update({f"clipped_{key}": value for key, value in clipped_metrics.items()})
    metrics.update({
        "negative_prediction_fraction": float(np.mean(raw_prediction < 0)),
        "clipped_value_fraction": float(np.mean(raw_prediction != clipped_prediction)),
    })
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    horizon_frames = []
    for variant, prediction in (("raw", raw_prediction), ("clipped", clipped_prediction)):
        frame = pd.DataFrame(metrics_by_horizon(prediction, arrays["targets"]))
        frame.insert(0, "prediction_variant", variant)
        horizon_frames.append(frame)
    pd.concat(horizon_frames, ignore_index=True).to_csv(output_dir / "metrics_by_horizon.csv", index=False)
    return metrics


def append_experiment_registry(registry_path: Path, cfg, metrics: dict, checkpoint_path: str) -> None:
    """Append one fixed_v1 run after successful evaluation; reject duplicate experiment IDs."""
    registry_path = Path(registry_path)
    with registry_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames, rows = reader.fieldnames, list(reader)
    if any(row["experiment_id"] == cfg.experiment_id for row in rows):
        raise FileExistsError(f"experiment_id already registered: {cfg.experiment_id}")
    dataset, model = cfg.datamodule.dataset, cfg.pl_module.model
    modalities = ["power"]
    if model.modality_mode in {"power_nwp", "all"}:
        modalities.append("future_nwp")
    if model.modality_mode == "all":
        modalities.append("satellite")
    row = {name: "" for name in fieldnames}
    row.update({
        "experiment_id": cfg.experiment_id,
        "experiment_version": "pipeline_v1_fixed",
        "status": "valid_or_debug",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "dataset": "MMSP", "train_sites": f"0-{dataset.num_sites - 1}",
        "test_sites": f"0-{dataset.num_sites - 1}", "seq_len": dataset.seq_len,
        "pred_len": dataset.pred_len, "train_ratio": cfg.datamodule.train_ratio,
        "valid_ratio": cfg.datamodule.valid_ratio, "test_ratio": cfg.datamodule.test_ratio,
        "seed": cfg.seed, "batch_size": cfg.datamodule.batch_size,
        "learning_rate": cfg.pl_module.optimizer.lr, "max_epochs": cfg.trainer.max_epochs,
        "checkpoint_metric": cfg.callbacks.model_checkpoint.monitor,
        "modalities": "|".join(modalities),
        "missing_modality_setting": cfg.pl_module.evaluation_mode,
        "ctx_masking_ratio": model.ctx_masking_ratio, "ts_masking_ratio": model.ts_masking_ratio,
        "vq_in_ts": model.vq_in_ts, "vq_in_ctx": model.vq_in_ctx, "vq_in_guide": model.vq_in_guide,
        "data_split_version": "chronological_target_split_v1",
        "scaler_version": dataset.data_pipeline.get("scaler_version", "train_only_fit_v1"),
        "output_dir": cfg.paths.output_dir,
        "checkpoint_path": checkpoint_path,
        "prediction_path": str(Path(cfg.paths.output_dir) / "predictions.npy"),
        "target_path": str(Path(cfg.paths.output_dir) / "targets.npy"),
        "mae": metrics.get("mae"), "rmse": metrics.get("rmse"), "mape": metrics.get("mape"),
        "nmae": "" if np.isnan(metrics.get("nmae", np.nan)) else metrics.get("nmae"),
        "nrmse": "" if np.isnan(metrics.get("nrmse", np.nan)) else metrics.get("nrmse"),
        "notes": "Capacity-normalized metrics are empty until station capacity metadata is supplied.",
        "embedding_extracted": "false",
    })
    with registry_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def missing_modality_interpretation(model, evaluation_mode: str) -> str:
    """Describe missing-modality evidence according to how the source model was trained."""
    if evaluation_mode == "full_modalities":
        return ""
    trained_with_modality_dropout = any(
        float(model.get(name, 0.0)) > 0
        for name in ("satellite_modality_dropout", "nwp_modality_dropout")
    )
    if trained_with_modality_dropout:
        return "Source run trained with whole-modality dropout; this is a robustness evaluation. "
    return (
        "Source run did not train whole-modality missing tokens; "
        "diagnostic only, not robustness evidence. "
    )


def append_checkpoint_evaluation(
    registry_path: Path, cfg, evaluation_id: str, evaluation_mode: str,
    output_dir: Path, metrics: dict, checkpoint_path: str,
) -> None:
    """Register one checkpoint-only evaluation without pretending it is a training run."""
    registry_path = Path(registry_path)
    with registry_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames, rows = reader.fieldnames, list(reader)
    if any(row["experiment_id"] == evaluation_id for row in rows):
        raise FileExistsError(f"evaluation already registered: {evaluation_id}")
    dataset, model = cfg.datamodule.dataset, cfg.pl_module.model
    modalities = "power|future_nwp|satellite"
    row = {name: "" for name in fieldnames}
    row.update({
        "experiment_id": evaluation_id, "experiment_version": "pipeline_v1_fixed",
        "status": "valid_or_debug", "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "git_commit": _command(["git", "rev-parse", "HEAD"]), "dataset": "MMSP",
        "train_sites": f"0-{dataset.num_sites - 1}", "test_sites": f"0-{dataset.num_sites - 1}",
        "seq_len": dataset.seq_len, "pred_len": dataset.pred_len,
        "train_ratio": cfg.datamodule.train_ratio, "valid_ratio": cfg.datamodule.valid_ratio,
        "test_ratio": cfg.datamodule.test_ratio, "seed": cfg.seed,
        "batch_size": cfg.datamodule.test_batch_size, "learning_rate": cfg.pl_module.optimizer.lr,
        "max_epochs": cfg.trainer.max_epochs, "checkpoint_metric": cfg.callbacks.model_checkpoint.monitor,
        "modalities": modalities, "missing_modality_setting": evaluation_mode,
        "ctx_masking_ratio": model.ctx_masking_ratio, "ts_masking_ratio": model.ts_masking_ratio,
        "vq_in_ts": model.vq_in_ts, "vq_in_ctx": model.vq_in_ctx, "vq_in_guide": model.vq_in_guide,
        "data_split_version": "chronological_target_split_v1",
        "scaler_version": dataset.data_pipeline.get("scaler_version", "train_only_fit_v1"),
        "output_dir": str(Path(output_dir).resolve()), "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "prediction_path": str(Path(output_dir).resolve() / "predictions_raw.npy"),
        "target_path": str(Path(output_dir).resolve() / "targets.npy"),
        "mae": metrics["mae"], "rmse": metrics["rmse"], "mape": metrics["mape"],
        "notes": (
            "Checkpoint-only sensitivity evaluation; raw registry metrics. "
            + missing_modality_interpretation(model, evaluation_mode)
            + f"clipped_mae={metrics['clipped_mae']}; clipped_rmse={metrics['clipped_rmse']}"
        ),
        "embedding_extracted": "false",
    })
    with registry_path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
