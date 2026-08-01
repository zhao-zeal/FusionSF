import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.merge_solar_run_manifests import merge_run_manifests
from scripts.run_fusionsf_solar_train import (
    EarlyStopping, build_alignment_report, load_best_checkpoint, naive_baselines,
    prediction_metrics, save_checkpoint,
)
from src.datasets.solar_energy_dataset import SolarDatasetSpec, build_sample_manifest
from src.models.fusionsf_solar import FusionSFSolar


def protocol_frame(frequency="1h", rows=5000):
    index = pd.date_range("2020-01-01", periods=rows, freq=frequency)
    frame = pd.DataFrame({"date": index, "power": np.arange(rows, dtype=np.float32) % 10, "VAR169": np.ones(rows)})
    spec = SolarDatasetSpec("GEFCom", "zone1", None, "date", "power", ("VAR169",), frequency, 10.0, "future_nwp")
    return frame, spec


def test_full_reference_alignment_report_and_missing_extra_duplicate_errors():
    frame, spec = protocol_frame()
    base = build_sample_manifest(frame, spec, 336, 72)
    origins = base.loc[base.split == "test", "input_end_timestamp"].iloc[100:130].to_numpy()
    manifest = build_sample_manifest(frame, spec, 336, 72, origins)
    report = build_alignment_report(manifest, origins, spec)
    assert report["passed"] and report["actual_origin_count"] == 30
    with pytest.raises(ValueError, match="duplicates"):
        build_sample_manifest(frame, spec, 336, 72, np.append(origins, origins[0]))
    with pytest.raises(ValueError, match="missing=1"):
        build_sample_manifest(frame, spec, 336, 72, np.append(origins, np.datetime64("1990-01-01")))
    with pytest.raises(ValueError, match="extra=1"):
        build_alignment_report(manifest, origins[:-1], spec)


def test_power_and_weather_use_identical_test_manifest():
    frame, spec = protocol_frame()
    base = build_sample_manifest(frame, spec, 336, 72)
    origins = base.loc[base.split == "test", "input_end_timestamp"].iloc[:20].to_numpy()
    power = build_sample_manifest(frame, spec, 336, 72, origins)
    weather = build_sample_manifest(frame, spec, 336, 72, origins)
    assert build_alignment_report(power, origins, spec)["test_manifest_sha256"] == build_alignment_report(weather, origins, spec)["test_manifest_sha256"]


def test_early_stopping_min_delta_and_patience():
    stopper = EarlyStopping(patience=2, min_delta=0.1)
    assert stopper.update(1.0) == (True, False)
    assert stopper.update(0.95) == (False, False)
    assert stopper.update(0.94) == (False, True)
    assert stopper.update(0.8) == (True, False)


def test_best_checkpoint_is_reloaded(tmp_path):
    model = FusionSFSolar(1, seq_len=4, dim=8, depth=1, heads=2, dim_head=4)
    optimizer = torch.optim.AdamW(model.parameters())
    args = SimpleNamespace(seed=42)
    path = tmp_path / "best.ckpt"
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    save_checkpoint(path, model, optimizer, 3, args, 9)
    for parameter in model.parameters():
        parameter.data.zero_()
    checkpoint = load_best_checkpoint(path, model, torch.device("cpu"))
    assert checkpoint["epoch"] == 3
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_raw_and_clipped_metrics_are_separate():
    raw = np.array([[[-2.0], [12.0]]])
    clipped = np.clip(raw, 0, 10)
    target = np.array([[[1.0], [9.0]]])
    daylight = np.ones((1, 2), dtype=bool)
    raw_metrics = prediction_metrics(raw, target, 10, daylight)
    clipped_metrics = prediction_metrics(clipped, target, 10, daylight)
    assert raw_metrics["mae"] == 3.0
    assert clipped_metrics["mae"] == 1.0
    assert raw_metrics["negative_prediction_ratio"] == 0.5


def test_baselines_use_every_row_of_same_test_manifest():
    frame, spec = protocol_frame(rows=1000)
    manifest = build_sample_manifest(frame, spec, 24, 4)
    test = manifest[manifest.split == "test"].iloc[:5]
    daylight = np.ones((5, 4), dtype=bool)
    baselines = naive_baselines(frame, test, spec, daylight)
    assert set(baselines) == {"persistence", "daily_seasonal_naive"}
    assert baselines["persistence"]["finite_count"] == 20


def test_parallel_run_manifests_do_not_modify_registry_until_merge(tmp_path):
    registry = tmp_path / "registry.csv"
    run_dirs = []
    for mode in ("power", "power_weather"):
        directory = tmp_path / mode; directory.mkdir(); run_dirs.append(directory)
        (directory / "run_manifest.json").write_text(json.dumps({
            "experiment_id": f"run_{mode}", "dataset": "gefcom", "site_id": "zone1",
            "mode": mode, "test_manifest_sha256": "same", "status": "preliminary_seed42",
        }))
    assert not registry.exists()
    merged = merge_run_manifests(run_dirs, registry)
    assert registry.exists() and len(merged) == 2
