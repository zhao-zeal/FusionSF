"""NumPy metrics used for reproducible saved-output evaluation."""

from typing import Dict, Iterable, Optional

import numpy as np


def compute_forecast_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    horizons: Iterable[int] = (1, 6, 12, 24),
    capacity: Optional[np.ndarray] = None,
    daylight_threshold: float = 1e-6,
) -> Dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim not in (2, 3):
        raise ValueError("prediction and target must have matching (N,H) or (N,H,1) shapes")
    if prediction.ndim == 3:
        prediction, target = prediction[..., 0], target[..., 0]
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction and target must be finite")
    error = prediction - target
    result = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mape": float(np.mean(np.abs(error) / np.maximum(np.abs(target), 1e-6))),
    }
    daylight = target > daylight_threshold
    result["daylight_mae"] = float(np.mean(np.abs(error[daylight]))) if daylight.any() else float("nan")
    result["daylight_rmse"] = (
        float(np.sqrt(np.mean(np.square(error[daylight])))) if daylight.any() else float("nan")
    )
    for horizon in horizons:
        if 1 <= horizon <= prediction.shape[1]:
            e = error[:, horizon - 1]
            result[f"mae@{horizon}"] = float(np.mean(np.abs(e)))
            result[f"rmse@{horizon}"] = float(np.sqrt(np.mean(np.square(e))))
    if capacity is None:
        result["nmae"] = float("nan")
        result["nrmse"] = float("nan")
    else:
        scale = np.asarray(capacity, dtype=np.float64)
        result["nmae"] = float(np.mean(np.abs(error) / scale))
        result["nrmse"] = float(np.sqrt(np.mean(np.square(error / scale))))
    return result


def metrics_by_horizon(prediction: np.ndarray, target: np.ndarray) -> list:
    prediction = np.asarray(prediction)[..., 0] if np.asarray(prediction).ndim == 3 else np.asarray(prediction)
    target = np.asarray(target)[..., 0] if np.asarray(target).ndim == 3 else np.asarray(target)
    rows = []
    for index in range(prediction.shape[1]):
        error = prediction[:, index] - target[:, index]
        rows.append({
            "horizon": index + 1,
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
        })
    return rows
