"""Validation-only model selection; independent calibration-only correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

from ..calibration import calibration_delta
from ..data import Panel
from ..features import build_features, historical_predictions


@dataclass
class HGBForecaster:
    model: HistGradientBoostingRegressor
    correction: float
    metadata: dict

    def predictions(
        self, panel: Panel, indices: np.ndarray, months: np.ndarray
    ) -> dict[str, np.ndarray]:
        feat, _ = build_features(panel, indices, months)
        raw = np.maximum(0, self.model.predict(feat)).reshape(len(indices), len(months))
        return {
            **historical_predictions(panel, indices, months),
            "hgb_raw": raw,
            "hgb_calibrated": np.maximum(0, raw + self.correction),
        }


def fit_hgb(panel: Panel, config: dict, seed: int) -> HGBForecaster:
    train = panel.indices("train")
    val = panel.indices("validation")
    cal = panel.indices("calibration")
    xtr, ytr = build_features(panel, train, np.arange(12, 36))
    vm = config["validation_months"]
    cm = config["calibration_months"]
    xv, yv = build_features(panel, val, np.arange(vm[0] - 1, vm[1]))
    xc, yc = build_features(panel, cal, np.arange(cm[0] - 1, cm[1]))
    records, models = [], []
    for k, candidate in enumerate(config["candidate_models"]):
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=config["quantile"],
            early_stopping=False,
            random_state=seed,
            **candidate,
        )
        model.fit(xtr, ytr)
        pv = np.maximum(0, model.predict(xv))
        records.append(
            {
                "candidate": k,
                "validation_pinball": float(mean_pinball_loss(yv, pv, alpha=0.9)),
                "validation_coverage": float(np.mean(yv <= pv)),
                **candidate,
            }
        )
        models.append(model)
    best = int(np.argmin([r["validation_pinball"] for r in records]))
    model = models[best]
    rank = min(len(yc), int(np.ceil((len(yc) + 1) * config["quantile"])))
    correction = calibration_delta(np.maximum(0, model.predict(xc)), yc, config["quantile"])
    meta = {
        "seed": seed,
        "selected_candidate": best,
        "candidates": records,
        "calibration_correction_yen": correction,
        "calibration_rank": rank,
        "train_rows": len(ytr),
        "validation_rows": len(yv),
        "calibration_rows": len(yc),
        "trained_on": "train households, months 13-36 only",
        "selected_on": f"validation households, months {vm[0]}-{vm[1]} only",
        "corrected_on": f"calibration households, months {cm[0]}-{cm[1]} only",
    }
    return HGBForecaster(model, correction, meta)
