import numpy as np
import pandas as pd

from src.datasets.tscontext_3modal_dataset import get_data_nwp, get_data_satellite


def test_nwp_scaler_fits_training_times_only_and_interpolates_per_grid(tmp_path):
    nwp_dir = tmp_path / "nwp"
    nwp_dir.mkdir()
    times = pd.date_range("2024-01-01", periods=4, freq="h")
    rows = []
    for lat, values in [(1.0, [0.0, np.nan, 2.0, 100.0]), (2.0, [10.0, 11.0, 12.0, 200.0])]:
        for timestamp, value in zip(times, values):
            rows.append({"fcst_date": timestamp, "lat": lat, "lon": 3.0, "feature": value})
    pd.DataFrame(rows).to_csv(nwp_dir / "nwp.csv", index=False)
    grouped, _, state = get_data_nwp(
        tmp_path, "nwp/nwp.csv", fit_end_time=times[3], return_scaler=True,
        fit_coordinates=[(1.0, 3.0)],
    )
    # The scaler sees only the selected training grid and training times: [0, 1, 2].
    assert np.isclose(state["mean"][0], 1.0)
    assert state["fit_coordinates"] == [(1.0, 3.0)]
    first_grid = grouped.get_group((1.0, 3.0))["feature"].to_numpy()
    assert np.isfinite(first_grid).all()

    _, _, injected = get_data_nwp(
        tmp_path, "nwp/nwp.csv", return_scaler=True, external_scaler_state=state
    )
    assert np.array_equal(injected["mean"], state["mean"])
    assert injected["source"] == "external_training_dataset"


def test_satellite_scaler_excludes_held_out_time(tmp_path):
    sat_dir = tmp_path / "satellite"
    sat_dir.mkdir()
    values = np.array([0.0, 1.0, 2.0, 100.0]).reshape(4, 1, 1, 1)
    times = pd.date_range("2024-01-01", periods=4, freq="h").to_numpy()
    np.save(sat_dir / "satellite.npy", values)
    np.save(sat_dir / "satellite_times.npy", times)
    np.save(sat_dir / "satellite_coords.npy", np.zeros((1, 1, 2)))
    _, _, _, state = get_data_satellite(
        tmp_path, "satellite", fit_end_time=times[3], return_scaler=True
    )
    assert np.isclose(state["mean"][0], 1.0)

    transformed, _, _, injected = get_data_satellite(
        tmp_path, "satellite", return_scaler=True, external_scaler_state=state
    )
    assert np.isclose(transformed[0, 0, 0, 0], -1.224744871391589)
    assert injected["source"] == "external_training_dataset"
