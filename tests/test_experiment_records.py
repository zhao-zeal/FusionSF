import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.utils.experiment_records import (
    actual_site_ids,
    append_checkpoint_evaluation,
    append_experiment_registry,
    format_site_ids,
    infer_modalities,
    missing_modality_interpretation,
)


ROOT = Path(__file__).resolve().parents[1]


def dummy_datamodule(train_ids, test_ids=None):
    data_all = SimpleNamespace(data_sp=[{"site": site_id} for site_id in train_ids])
    data_test_all = None if test_ids is None else SimpleNamespace(
        data_sp=[{"site": site_id} for site_id in test_ids]
    )
    return SimpleNamespace(data_all=data_all, data_test_all=data_test_all)


def registry_copy(tmp_path):
    header = (ROOT / "experiments/experiment_registry.csv").read_text(encoding="utf-8").splitlines()[0]
    path = tmp_path / "registry.csv"
    path.write_text(header + "\n", encoding="utf-8")
    return path


def registry_cfg(mode="all", experiment_id="zero_shot_test"):
    return OmegaConf.create({
        "experiment_id": experiment_id,
        "seed": 42,
        "datamodule": {
            "batch_size": 16, "test_batch_size": 64,
            "train_ratio": 0.6, "valid_ratio": 0.2, "test_ratio": 0.2,
            "dataset": {
                "seq_len": 24, "pred_len": 24,
                "data_pipeline": {"scaler_version": "train_sites_and_time_only_fit_v1"},
            },
        },
        "pl_module": {
            "evaluation_mode": "full_modalities",
            "optimizer": {"lr": 0.0016},
            "model": {
                "modality_mode": mode, "ctx_masking_ratio": 0.85,
                "ts_masking_ratio": 0.15, "vq_in_ts": False,
                "vq_in_ctx": False, "vq_in_guide": False,
                "satellite_modality_dropout": 0.0, "nwp_modality_dropout": 0.0,
            },
        },
        "trainer": {"max_epochs": 1},
        "callbacks": {"model_checkpoint": {"monitor": "val/mae"}},
        "paths": {"output_dir": str(ROOT / "outputs/test")},
    })


def test_missing_modality_interpretation_tracks_training_dropout():
    standard = OmegaConf.create({
        "satellite_modality_dropout": 0.0,
        "nwp_modality_dropout": 0.0,
    })
    robust = OmegaConf.create({
        "satellite_modality_dropout": 0.2,
        "nwp_modality_dropout": 0.2,
    })
    assert "diagnostic only" in missing_modality_interpretation(standard, "missing_nwp")
    assert "robustness evaluation" in missing_modality_interpretation(robust, "missing_nwp")
    assert missing_modality_interpretation(robust, "full_modalities") == ""


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("power", "power"),
        ("power_nwp", "power|future_nwp"),
        ("all", "power|future_nwp|satellite"),
    ],
)
def test_infer_modalities(mode, expected):
    assert infer_modalities(OmegaConf.create({"modality_mode": mode})) == expected


def test_site_formatting_preserves_non_contiguous_sets():
    assert format_site_ids([8, 1, 4, 8]) == "1|4|8"
    assert actual_site_ids(dummy_datamodule([1, 4, 8], [2, 7])) == ("1|4|8", "2|7")


def test_zero_shot_training_registry_uses_actual_loaded_sites(tmp_path):
    registry = registry_copy(tmp_path)
    datamodule = dummy_datamodule(range(10, 20), range(10))
    append_experiment_registry(
        registry, registry_cfg(),
        {"mae": 0.1, "rmse": 0.2, "mape": 1.0, "nmae": np.nan, "nrmse": np.nan},
        "checkpoint.ckpt", datamodule,
    )
    with registry.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["train_sites"] == "10|11|12|13|14|15|16|17|18|19"
    assert row["test_sites"] == "0|1|2|3|4|5|6|7|8|9"
    assert row["train_sites"] != row["test_sites"]
    assert row["modalities"] == "power|future_nwp|satellite"
    assert row["scaler_version"] == "train_sites_and_time_only_fit_v1"


def test_checkpoint_registry_uses_actual_sites_and_model_modalities(tmp_path):
    registry = registry_copy(tmp_path)
    datamodule = dummy_datamodule([1, 4, 8], [2, 7])
    append_checkpoint_evaluation(
        registry, registry_cfg(mode="power", experiment_id="source"), "source__eval_full",
        "full_modalities", tmp_path / "evaluation", {
            "mae": 0.1, "rmse": 0.2, "mape": 1.0,
            "clipped_mae": 0.1, "clipped_rmse": 0.2,
        }, "checkpoint.ckpt", datamodule,
    )
    with registry.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["train_sites"] == "1|4|8"
    assert row["test_sites"] == "2|7"
    assert row["modalities"] == "power"
    assert row["status"] == "checkpoint_evaluation"
    assert "no retraining was performed" in row["notes"]
