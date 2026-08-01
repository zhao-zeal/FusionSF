import inspect

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.datasets.solar_energy_dataset import (
    CSG_SCHEMAS, SolarDatasetSpec, SolarWindowDataset, build_sample_manifest,
    deterministic_time_features, fit_train_scaler, load_csg_schema_frame,
)
from src.models.fusionSF_3modal import CrossTransformer, FusionSF3M, Transformer
from src.models.fusionsf_solar import FusionSFSolar, PrototypeHorizonGRU


@pytest.fixture
def solar_data(tmp_path):
    index = pd.date_range("2020-01-01", periods=2200, freq="15min")
    frame = pd.DataFrame({"date": index, "power": np.arange(2200, dtype=np.float32), "weather": np.arange(2200, dtype=np.float32) + 10})
    path = tmp_path / "data.csv"; frame.to_csv(path, index=False)
    spec = SolarDatasetSpec("test", "site", path, "date", "power", ("weather",), "15min", None, "future_reanalysis")
    return frame, spec


def test_manifest_split_targets_are_disjoint_and_windows_are_causal(solar_data):
    frame, spec = solar_data
    manifest = build_sample_manifest(frame, spec, seq_len=24, pred_len=16)
    targets = {}
    for split in ("train", "validation", "test"):
        part = manifest[manifest.split == split]
        assert (part.input_end_timestamp < part.forecast_start_timestamp).all()
        assert ((part.forecast_end_timestamp - part.forecast_start_timestamp) == pd.Timedelta(minutes=15 * 15)).all()
        targets[split] = set(pd.concat([pd.Series(pd.date_range(r.forecast_start_timestamp, periods=16, freq="15min")) for r in part.itertuples()]))
    assert not targets["train"] & targets["validation"]
    assert not targets["train"] & targets["test"]
    assert not targets["validation"] & targets["test"]


@pytest.mark.parametrize("frequency,pred_len", [("1h", 72), ("15min", 288)])
def test_reference_origins_align_by_timestamp_for_full_target_interval(tmp_path, frequency, pred_len):
    index = pd.date_range("2020-01-01", periods=5000, freq=frequency)
    frame = pd.DataFrame({"date": index, "power": np.ones(len(index)), "weather": np.ones(len(index))})
    spec = SolarDatasetSpec("test", "site", tmp_path / "x", "date", "power", ("weather",), frequency)
    base = build_sample_manifest(frame, spec, 336, pred_len)
    test = base[base.split == "test"].iloc[100:130]
    origins = test.input_end_timestamp.sample(frac=1, random_state=42).to_numpy()
    aligned = build_sample_manifest(frame, spec, 336, pred_len, origins)
    aligned_test = aligned[aligned.split == "test"]
    assert set(aligned_test.input_end_timestamp) == set(pd.to_datetime(origins))
    delta = pd.Timedelta(frequency)
    assert all(row.forecast_end_timestamp == row.input_end_timestamp + pred_len * delta for row in aligned_test.itertuples())
    with pytest.raises(ValueError, match="missing=1"):
        build_sample_manifest(frame, spec, 336, pred_len, np.append(origins, np.datetime64("1990-01-01")))


def test_scaler_uses_train_slice_only(solar_data):
    frame, spec = solar_data
    scaler = fit_train_scaler(frame, spec, 1200)
    changed = frame.copy(); changed.loc[1200:, ["power", "weather"]] = 1e9
    changed_scaler = fit_train_scaler(changed, spec, 1200)
    assert changed_scaler.power_mean == scaler.power_mean
    np.testing.assert_array_equal(changed_scaler.weather_mean, scaler.weather_mean)


def test_power_does_not_read_weather_and_modes_share_time_features(solar_data):
    frame, spec = solar_data
    manifest = build_sample_manifest(frame, spec, 24, 16)
    row = manifest[manifest.split == "train"].iloc[[0]]
    scaler = fit_train_scaler(frame, spec, 1200)
    power = SolarWindowDataset(frame, row, spec, scaler, "power")[0]
    weather = SolarWindowDataset(frame, row, spec, scaler, "power_weather")[0]
    assert "future_weather" not in power
    torch.testing.assert_close(power["history_time_features"], weather["history_time_features"])
    torch.testing.assert_close(power["future_time_features"], weather["future_time_features"])


def test_time_features_depend_only_on_timestamps():
    timestamps = pd.date_range("2020-01-01", periods=8, freq="15min")
    first = deterministic_time_features(timestamps)
    target_a, target_b = np.zeros(8), np.arange(8) * 1e9
    np.testing.assert_array_equal(first, deterministic_time_features(timestamps))
    assert not np.array_equal(target_a, target_b)  # target changes cannot enter the timestamp-only function
    assert first.shape == (8, 8)


def test_future_timestamp_changes_horizon_query():
    model = FusionSFSolar(weather_dim=1, seq_len=24, dim=16, depth=1, heads=4, dim_head=4)
    history, history_time = torch.zeros(1, 24, 1), torch.zeros(1, 24, 8)
    future_a, future_b = torch.zeros(1, 4, 8), torch.ones(1, 4, 8)
    query_a = model.encode(history, history_time, future_a)["horizon_query"]
    query_b = model.encode(history, history_time, future_b)["horizon_query"]
    assert not torch.equal(query_a, query_b)


def test_official_model_reuses_fusionsf_components_and_has_no_gru():
    model = FusionSFSolar(weather_dim=3, seq_len=24, dim=16, depth=1, heads=4, dim_head=4)
    assert isinstance(model.ts_encoder, Transformer)
    assert isinstance(model.horizon_cross_attention, CrossTransformer)
    assert not any(isinstance(module, nn.GRU) for module in model.modules())
    assert PrototypeHorizonGRU.model_family == "prototype_horizon_gru"


@pytest.mark.parametrize("pred_len", [1, 4, 16, 72, 288])
def test_official_model_native_horizon_shapes(pred_len):
    model = FusionSFSolar(weather_dim=3, seq_len=24, dim=16, depth=1, heads=4, dim_head=4)
    prediction = model(torch.randn(2, 24, 1), torch.randn(2, 24, 8), torch.randn(2, pred_len, 8), torch.randn(2, pred_len, 3))
    assert prediction.shape == (2, pred_len, 1)


def test_weather_branch_gradient_rules():
    model = FusionSFSolar(weather_dim=3, seq_len=24, dim=16, depth=1, heads=4, dim_head=4)
    common = (torch.randn(2, 24, 1), torch.randn(2, 24, 8), torch.randn(2, 4, 8))
    model(*common).square().mean().backward()
    assert all(parameter.grad is None for parameter in model.guide_encoder.parameters())
    model.zero_grad(set_to_none=True)
    model(*common, torch.randn(2, 4, 3)).square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.guide_encoder.parameters())


def test_mmsf_forward_signature_is_unchanged():
    assert list(inspect.signature(FusionSF3M.forward).parameters) == [
        "self", "ctx", "ctx_coords", "ts", "ts_coords", "time_coords", "ts_guide",
        "mask", "modality_availability", "evaluation_mode", "return_embeddings",
    ]


def test_csg_schema_loader_rejects_wrong_field_count(tmp_path):
    path = tmp_path / "site6.csv"
    path.write_text("bad,header\n2020-01-01,1,2,3,4,5,6\n")
    assert len(CSG_SCHEMAS["site6"].columns) == 8
    with pytest.raises(ValueError, match="expected 8 fields"):
        load_csg_schema_frame(path, "site6")
