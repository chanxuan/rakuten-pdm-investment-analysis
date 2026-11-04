from dataclasses import replace

import numpy as np

from household_cashflow.backtest import match_fixed
from household_cashflow.data import apply_scenario, generate, subgroup_labels
from household_cashflow.features import FEATURE_NAMES, build_features, sequence_inputs
from household_cashflow.models.hgb import fit_hgb


def test_household_partitions_are_disjoint(panel):
    ids = [
        set(panel.household_id[panel.indices(name)])
        for name in ["train", "validation", "calibration", "test"]
    ]
    assert sum(map(len, ids)) == len(set.union(*ids))
    assert len(FEATURE_NAMES) == 16 and "household_id" not in FEATURE_NAMES


def test_inputs_exclude_current_and_future_outcomes(panel):
    indices = panel.indices("test")
    x = panel.variable_expense.copy()
    x[:, 36:] *= 1000
    changed = replace(panel, variable_expense=x)
    original = sequence_inputs(panel, indices, np.array([36]))
    other = sequence_inputs(changed, indices, np.array([36]))
    np.testing.assert_array_equal(original[0], other[0])
    np.testing.assert_array_equal(original[1], other[1])
    np.testing.assert_array_equal(original[0][0, :, 0], panel.variable_expense[indices[0], 24:36])


def test_all_post36_values_are_excluded_from_tuning(panel, config):
    x, income = panel.variable_expense.copy(), panel.income.copy()
    x[:, 36:] *= 1000
    income[:, 36:] *= 0.01
    changed = replace(panel, variable_expense=x, income=income)
    a, b = fit_hgb(panel, config, 133), fit_hgb(changed, config, 133)
    assert a.metadata == b.metadata
    val = panel.indices("validation")
    prediction = a.predictions(panel, val, np.arange(24, 36))["hgb_calibrated"]
    target_a, details_a = match_fixed(panel, prediction, config, 40000, 0.15)
    target_b, details_b = match_fixed(changed, prediction, config, 40000, 0.15)
    assert target_a == target_b and details_a == details_b
    assert details_a["snapshot_month"] == 24 and details_a["selection_split"] == "validation"
    assert a.metadata["train_rows"] == 40 * 24


def test_generator_is_repeatable_and_scenarios_preserve_history(config):
    a, b = generate(config, 128), generate(config, 128)
    for key in ["income", "fixed", "planned", "variable_expense", "initial_cash"]:
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    for name in config["scenarios"]:
        changed = apply_scenario(a, config, name)
        np.testing.assert_array_equal(a.variable_expense[:, :36], changed.variable_expense[:, :36])
        np.testing.assert_array_equal(a.income[:, :36], changed.income[:, :36])


def test_subgroups_do_not_use_future_outcomes(panel):
    x = panel.variable_expense.copy()
    x[:, 36:] *= 1000
    changed = replace(panel, variable_expense=x)
    a, thresholds_a = subgroup_labels(panel)
    b, thresholds_b = subgroup_labels(changed)
    assert thresholds_a == thresholds_b
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_first_twelve_months_cannot_form_a_complete_lag_window(panel):
    import pytest

    with pytest.raises(ValueError, match="12 complete"):
        build_features(panel, panel.indices("test"), np.array([11]))
