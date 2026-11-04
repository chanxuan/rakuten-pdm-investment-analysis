from dataclasses import replace

import numpy as np
import pytest

from household_cashflow.backtest import POLICIES, action_agreement, available_policies, rollout


def predictions(panel):
    n = len(panel.indices("test"))
    return {name: np.full((n, 12), 45000.0) for name in POLICIES[2:]}


def test_each_cap_setting_restarts_its_cash_path(panel):
    n = len(panel.household_id)
    y, f, p, x = [
        a.copy() for a in [panel.income, panel.fixed, panel.planned, panel.variable_expense]
    ]
    y[:, 36:], f[:, 36:], p[:, 36:], x[:, 36:] = 220000, 120000, 20000, 55000
    panel = replace(
        panel,
        income=y,
        fixed=f,
        planned=p,
        variable_expense=x,
        reserve=np.full(n, 300000.0),
        initial_cash=np.full(n, 300000.0),
    )
    idx, pred = panel.indices("test"), predictions(panel)
    a, end_a, _ = rollout(panel, idx, pred, 20000, 0.15, 20000)
    b, end_b, _ = rollout(panel, idx, pred, 60000, 0.20, 40000)
    k = POLICIES.index("hgb_calibrated")
    np.testing.assert_array_equal(a[k, 0, :2], [20000, 20000])
    np.testing.assert_array_equal(end_a[k, 0, :2], [305000, 310000])
    np.testing.assert_array_equal(b[k, 0, :2], [35000, 25000])
    np.testing.assert_array_equal(end_b[k, 0, :2], [290000, 290000])


def test_current_actions_cannot_use_current_actual_expense(panel):
    idx, pred = panel.indices("test"), predictions(panel)
    x = panel.variable_expense.copy()
    x[:, 36] *= 1000
    a, _, _ = rollout(panel, idx, pred, 40000, 0.15, 38000)
    b, _, _ = rollout(replace(panel, variable_expense=x), idx, pred, 40000, 0.15, 38000)
    np.testing.assert_array_equal(a[:, :, 0], b[:, :, 0])


def test_cash_identity_debt_caps_and_alignment(panel):
    idx, pred = panel.indices("test"), predictions(panel)
    a, end, households = rollout(panel, idx, pred, 60000, 0.2, 50000, 0.01)
    start = np.concatenate(
        [
            np.broadcast_to(panel.initial_cash[idx], (len(POLICIES), len(idx)))[:, :, None],
            end[:, :, :-1],
        ],
        axis=2,
    )
    actual = (
        start
        - np.maximum(-start, 0) * 0.01
        + panel.income[idx, 36:48]
        - panel.fixed[idx, 36:48]
        - panel.planned[idx, 36:48]
        - panel.variable_expense[idx, 36:48]
        - a
    )
    np.testing.assert_allclose(end, actual, atol=1e-7)
    assert (a >= 0).all() and (a % 1000 == 0).all() and (a[0] == 0).all()
    assert (a <= np.minimum(60000, 0.2 * panel.income[idx, 36:48])).all()
    assert len(households) == len(POLICIES) * len(idx)


def test_true_cap_binding_and_inclusive_ties(panel):
    n = len(panel.household_id)
    income = panel.income.copy()
    income[:, 36:] = 200000
    panel = replace(panel, income=income, initial_cash=np.full(n, 2e6), reserve=np.full(n, 1e5))
    _, _, hh = rollout(panel, panel.indices("test"), predictions(panel), 20000, 0.1, 40000)
    assert hh[hh.policy == "no_investment"].cap_any_binding_rate.eq(0).all()
    other = hh[hh.policy != "no_investment"]
    assert other.cap_any_binding_rate.eq(1).all() and other.cap_both_binding_rate.eq(1).all()


def test_unavailable_models_are_not_silently_treated_as_zero_budgets(panel):
    pred = {
        "history_mean": np.full((len(panel.indices("test")), 12), 45000.0),
        "hgb_calibrated": np.full((len(panel.indices("test")), 12), 60000.0),
    }
    assert "lightgbm_calibrated" not in available_policies(pred)
    with pytest.raises(ValueError, match="Missing prediction"):
        rollout(
            panel,
            panel.indices("test"),
            pred,
            40000,
            0.15,
            38000,
            policies=["no_investment", "fixed_matched", "lightgbm_calibrated"],
        )


def test_action_agreement_is_a_partition_of_equal_actions(panel):
    pred, idx = predictions(panel), panel.indices("test")
    pred["hgb_calibrated"] += 1000
    before = {key: value.copy() for key, value in pred.items()}
    a, end, _ = rollout(panel, idx, pred, 40000, 0.15, 38000)
    agreement = action_agreement(panel, idx, pred, a, end, 40000, 0.15)
    np.testing.assert_allclose(
        agreement.prediction_diff_same_zero_rate
        + agreement.prediction_diff_same_positive_at_cap_rate
        + agreement.prediction_diff_same_positive_below_cap_rate,
        agreement.prediction_diff_but_action_equal_rate,
    )
    for key in pred:
        np.testing.assert_array_equal(pred[key], before[key])
