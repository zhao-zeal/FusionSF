import numpy as np
import pandas as pd

from src.datasets.tscontext_3modal_dataset import Ts3MDataset


def test_fixed_dataset_returns_aligned_shapes_and_metadata(tmp_path):
    (tmp_path / "solar_power").mkdir()
    (tmp_path / "satellite").mkdir()
    (tmp_path / "nwp").mkdir()
    times = pd.date_range("2024-01-01", periods=80, freq="h")
    power_rows, nwp_rows = [], []
    sites = [(0, 1.0, 3.0), (1, 2.0, 3.0)]
    for site, lat, lon in sites:
        for index, timestamp in enumerate(times):
            power_rows.append({"datetime": timestamp, "lat": lat, "lon": lon, "power": index / 80, "site": site})
            nwp_rows.append({"fcst_date": timestamp, "lat": lat, "lon": lon, **{f"f{i}": index + i for i in range(17)}})
    pd.DataFrame(power_rows).to_csv(tmp_path / "solar_power/solar_power.csv", index=False)
    pd.DataFrame(nwp_rows).to_csv(tmp_path / "nwp/nwp.csv", index=False)
    np.save(tmp_path / "satellite/satellite.npy", np.random.default_rng(42).normal(size=(80, 2, 2, 1)))
    np.save(tmp_path / "satellite/satellite_times.npy", times.to_numpy())
    np.save(tmp_path / "satellite/satellite_coords.npy", np.zeros((2, 2, 2)))
    dataset = Ts3MDataset(
        str(tmp_path), satellite_dir="satellite", seq_len=24, pred_len=6, num_sites=2,
        modality_mode="all", data_pipeline={"version": "fixed_v1"},
        train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2,
    )
    sample = dataset[0]
    assert sample["ts_input"].shape == (24, 1)
    assert sample["ts_target"].shape == (6,)
    assert sample["ec_input"].shape == (6, 17)
    assert sample["forecast_start_timestamp"] > sample["input_end_timestamp"]
