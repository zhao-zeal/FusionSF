import numpy as np
import pandas as pd
import pytest
import torch

from src.datasets.solar_energy_dataset import SolarDatasetSpec, SolarWindowDataset, build_sample_manifest, fit_train_scaler
from src.models.fusionsf_solar import FusionSFSolar


@pytest.fixture
def solar_data(tmp_path):
    index = pd.date_range("2020-01-01", periods=1000, freq="15min")
    frame = pd.DataFrame({"date": index, "power": np.arange(1000, dtype=np.float32), "nwp_a": np.arange(1000, dtype=np.float32) + 10})
    path = tmp_path / "data.csv"; frame.to_csv(path, index=False)
    spec = SolarDatasetSpec("test", "site", path, "date", "power", ("nwp_a",), "15min", None)
    return frame, spec


def test_manifest_has_disjoint_targets_and_causal_windows(solar_data):
    frame, spec = solar_data
    manifest = build_sample_manifest(frame, spec, seq_len=24, pred_len=16)
    for split in ("train", "validation", "test"):
        part = manifest[manifest.split == split]
        assert (part.input_end_timestamp < part.forecast_start_timestamp).all()
        assert ((part.forecast_end_timestamp - part.forecast_start_timestamp) == pd.Timedelta(minutes=15 * 15)).all()
    ranges = {s: manifest[manifest.split == s] for s in ("train", "validation", "test")}
    assert ranges["train"].forecast_end_timestamp.max() < ranges["validation"].forecast_start_timestamp.min()
    assert ranges["validation"].forecast_end_timestamp.max() < ranges["test"].forecast_start_timestamp.min()


def test_scaler_uses_train_slice_only(solar_data):
    frame, spec = solar_data
    scaler = fit_train_scaler(frame, spec, train_end=600)
    assert scaler.power_mean == pytest.approx(frame.power.iloc[:600].mean())
    changed = frame.copy(); changed.loc[600:, ["power", "nwp_a"]] = 1e9
    changed_scaler = fit_train_scaler(changed, spec, 600)
    assert changed_scaler.power_mean == scaler.power_mean
    np.testing.assert_array_equal(changed_scaler.nwp_mean, scaler.nwp_mean)


def test_power_mode_does_not_read_nwp_and_power_nwp_reads_future_interval(solar_data):
    frame, spec = solar_data
    manifest = build_sample_manifest(frame, spec, 24, 16)
    scaler = fit_train_scaler(frame, spec, 600)
    row = manifest[manifest.split == "train"].iloc[[0]]
    power = SolarWindowDataset(frame, row, spec, scaler, "power")[0]
    assert "future_nwp" not in power
    combined = SolarWindowDataset(frame, row, spec, scaler, "power_nwp")[0]
    expected = (frame.nwp_a.iloc[24:40].to_numpy() - scaler.nwp_mean[0]) / scaler.nwp_std[0]
    np.testing.assert_allclose(combined["future_nwp"].numpy()[:, 0], expected)


@pytest.mark.parametrize("pred_len", [1, 4, 16, 72, 288])
def test_model_supports_native_horizons(pred_len):
    model = FusionSFSolar(nwp_dim=3, hidden_dim=16)
    history = torch.randn(2, 336, 1)
    nwp = torch.randn(2, pred_len, 3)
    pred = model(history, nwp)
    assert pred.shape == (2, pred_len, 1)
    pred.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_power_shape_without_nwp():
    model = FusionSFSolar(nwp_dim=3, hidden_dim=16)
    assert model(torch.randn(2, 336, 1), pred_len=4).shape == (2, 4, 1)
