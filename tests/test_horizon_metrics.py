import numpy as np

from src.metrics.forecast_metrics import compute_forecast_metrics, metrics_by_horizon


def test_overall_and_selected_horizon_metrics():
    target = np.zeros((2, 24, 1))
    prediction = np.ones_like(target)
    metrics = compute_forecast_metrics(prediction, target)
    assert metrics["mae"] == 1.0
    assert metrics["rmse@1"] == 1.0
    assert metrics["mae@24"] == 1.0
    assert np.isnan(metrics["nmae"])
    assert len(metrics_by_horizon(prediction, target)) == 24
