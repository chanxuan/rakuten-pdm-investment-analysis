"""Additive quantile calibration on separate households."""

from __future__ import annotations

import numpy as np


def calibration_delta(prediction, target, alpha=0.9):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    errors = (target - prediction).ravel()
    if not errors.size or not np.isfinite(errors).all():
        raise ValueError("Calibration requires finite, nonempty residuals")
    rank = min(len(errors), int(np.ceil((len(errors) + 1) * alpha)))
    return float(np.partition(errors, rank - 1)[rank - 1])


def calibrated(prediction, delta):
    return np.maximum(0, np.asarray(prediction) + delta)
