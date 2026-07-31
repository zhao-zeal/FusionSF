import numpy as np

from src.utils.experiment_records import save_test_outputs


def test_saved_outputs_and_metadata_remain_row_aligned(tmp_path):
    chunks = {
        "inputs": [np.zeros((3, 24, 1))],
        "outputs": [np.concatenate([
            -np.ones((1, 24, 1)), np.zeros((1, 24, 1)), np.ones((1, 24, 1))
        ], axis=0)],
        "targets": [np.ones((3, 24, 1))],
        "site_ids": [np.array([2, 3, 4])],
        "input_start_timestamps": [np.array([1, 2, 3])],
        "input_end_timestamps": [np.array([4, 5, 6])],
        "forecast_start_timestamps": [np.array([7, 8, 9])],
        "forecast_end_timestamps": [np.array([10, 11, 12])],
        "forecast_timestamps": [np.array([[7, 8], [8, 9], [9, 10]])],
        "forecast_horizons": [np.array([[1, 2], [1, 2], [1, 2]])],
    }
    metrics = save_test_outputs(tmp_path, chunks)
    assert np.load(tmp_path / "predictions.npy").shape[0] == 3
    assert np.array_equal(np.load(tmp_path / "site_ids.npy"), [2, 3, 4])
    assert np.array_equal(np.load(tmp_path / "timestamps.npy"), [[7, 8], [8, 9], [9, 10]])
    assert np.load(tmp_path / "predictions_raw.npy").min() == -1
    assert np.load(tmp_path / "predictions_clipped.npy").min() == 0
    assert metrics["negative_prediction_fraction"] == 1 / 3
    assert metrics["clipped_mae"] < metrics["mae"]
