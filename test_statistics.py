import numpy as np
import pandas as pd
import pytest

from household_cashflow.calibration import calibrated, calibration_delta
from household_cashflow.experiment import forecast_households
from household_cashflow.metrics import bootstrap_tables


def test_calibration_uses_the_documented_order_statistic():
    rng = np.random.default_rng(1)
    actual = rng.normal(100, 10, 3600)
    prediction = np.full(3600, 200.0)
    expected = np.sort(actual - prediction)[3240]
    assert calibration_delta(prediction, actual) == expected < 0
    np.testing.assert_array_equal(calibrated([5, 100], -10), [0, 90])


@pytest.mark.parametrize("prediction,target", [([], []), ([1], [np.nan]), ([1, 2], [1])])
def test_invalid_calibration_data_is_rejected(prediction, target):
    with pytest.raises(ValueError):
        calibration_delta(prediction, target)


def test_bootstrap_keeps_methods_paired(config):
    rng = np.random.default_rng(3)
    rows = []
    for seed in [10, 11]:
        values = rng.normal(1000 * seed, 1000, 12)
        for i, value in enumerate(values):
            for method, offset in [("a", 100.0), ("b", 0.0)]:
                rows.append(
                    {
                        "scenario": "baseline",
                        "seed": seed,
                        "household_id": i,
                        "method": method,
                        "value": value + offset,
                    }
                )
    _, pairs = bootstrap_tables(pd.DataFrame(rows), "method", ["value"], config)
    row = pairs.iloc[0]
    np.testing.assert_allclose(
        [row.difference_a_minus_b, row.ci95_low, row.ci95_high], [100, 100, 100], atol=1e-10
    )


def test_forecast_evaluation_is_independent_of_investment(panel):
    idx = panel.indices("test")
    prediction = {"budget": np.full((len(idx), 12), 50000.0)}
    frame = forecast_households(panel, idx, prediction, 12, "baseline")
    assert frame.coverage.between(0, 1).all() and frame.pinball_loss.ge(0).all()
    assert "investment" not in frame
