"""Vectorized ledger, strict cap attribution, and validation-only matching."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .allocation import actions

POLICIES = [
    "no_investment",
    "fixed_matched",
    "history_mean",
    "history_q90",
    "history_q90_calibrated",
    "hgb_calibrated",
    "lightgbm_calibrated",
    "lstm_calibrated",
    "hgb_raw",
]
PAIRS = [
    ("hgb_calibrated", "history_q90_calibrated"),
    ("lightgbm_calibrated", "hgb_calibrated"),
    ("lstm_calibrated", "hgb_calibrated"),
    ("lightgbm_calibrated", "history_q90_calibrated"),
    ("lstm_calibrated", "history_q90_calibrated"),
]


def match_fixed(panel, prediction, config, amount_cap, income_cap, search_max=100000):
    """Enumerate 101 fixed targets at once, using validation months 25--36."""
    idx = panel.indices("validation")
    r = (
        config["reserve_months"]
        * (panel.fixed[:, 12:24] + panel.variable_expense[:, 12:24]).mean(axis=1)[idx]
    )
    initial = r * panel.hidden["initial_ratio"][idx]
    targets = np.arange(0, search_max + 1, 1000)
    balance = np.broadcast_to(initial, (len(targets), len(idx))).copy()
    reference = initial.copy()
    total, reference_total = np.zeros(len(targets)), 0.0
    for j, t in enumerate(range(24, 36)):
        y, f, p, x = [
            a[idx, t] for a in [panel.income, panel.fixed, panel.planned, panel.variable_expense]
        ]
        maximum, _ = actions(balance, y, f, p, r, 0.0, amount_cap, income_cap)
        action = np.minimum(maximum, targets[:, None])
        ref, _ = actions(reference, y, f, p, r, prediction[:, j], amount_cap, income_cap)
        total += action.sum(axis=1)
        reference_total += ref.sum()
        balance += y - f - p - x - action
        reference += y - f - p - x - ref
    means = total / (len(idx) * 12)
    targetmean = reference_total / (len(idx) * 12)
    best = int(np.argmin(np.abs(means - targetmean)))
    return int(targets[best]), {
        "selection_split": "validation",
        "months": [25, 36],
        "snapshot_month": 24,
        "target": int(targets[best]),
        "validation_reference_mean": targetmean,
        "validation_fixed_mean": float(means[best]),
        "match_error": float(abs(means[best] - targetmean)),
        "search_boundary_reached": bool(best == len(targets) - 1),
        "candidates": [
            {"target": int(t), "mean": float(m), "error": float(abs(m - targetmean))}
            for t, m in zip(targets, means)
        ],
    }


def rollout(
    panel, idx, predictions, amount_cap, income_cap, fixed_target, interest_rate=0, policies=None
):
    """Roll each policy forward on the same external cash-flow path."""
    policies = available_policies(predictions) if policies is None else list(policies)
    if policies[:2] != ["no_investment", "fixed_matched"]:
        raise ValueError("The first two policies must be no_investment and fixed_matched")
    missing = set(policies[2:]) - set(predictions)
    if missing:
        raise ValueError(f"Missing prediction budgets: {sorted(missing)}")
    npol, n, nt = len(policies), len(idx), 12
    y, f, p, x = [
        a[idx, 36:48] for a in [panel.income, panel.fixed, panel.planned, panel.variable_expense]
    ]
    r, initial = panel.reserve[idx], panel.initial_cash[idx]
    balance = np.broadcast_to(initial, (npol, n)).copy()
    ends = np.empty((npol, n, nt))
    amounts = np.empty((npol, n, nt), np.int32)
    flags = {
        k: np.zeros((npol, n, nt), bool)
        for k in [
            "amount_binding",
            "income_binding",
            "both_binding",
            "at_cap",
            "infeasible_common",
            "infeasible_own",
            "incremental_breach",
        ]
    }
    q = np.stack([predictions.get(policy, np.zeros((n, nt))) for policy in policies])
    for j in range(nt):
        interest = np.maximum(-balance, 0) * interest_rate
        usable = balance - interest
        maximum, own_infeasible = actions(
            usable, y[:, j], f[:, j], p[:, j], r, q[:, :, j], amount_cap, income_cap
        )
        action = maximum
        action[0] = 0
        action[1] = np.minimum(action[1], fixed_target)
        margin = usable + y[:, j] - f[:, j] - p[:, j] - r - q[:, :, j]
        desired = np.floor(np.maximum(0, margin) / 1000) * 1000
        desired[0] = 0
        desired[1] = np.minimum(desired[1], fixed_target)
        cap_a = np.floor(amount_cap / 1000) * 1000
        cap_i = np.floor(income_cap * y[:, j] / 1000) * 1000
        true_bind = desired > action
        ties = cap_a == cap_i
        flags["amount_binding"][:, :, j] = true_bind & (cap_a <= cap_i)
        flags["income_binding"][:, :, j] = true_bind & (cap_i <= cap_a)
        flags["both_binding"][:, :, j] = true_bind & ties
        flags["at_cap"][:, :, j] = action == np.minimum(cap_a, cap_i)
        commonmargin = (
            usable + y[:, j] - f[:, j] - p[:, j] - r - predictions["hgb_calibrated"][:, j]
        )
        flags["infeasible_common"][:, :, j] = commonmargin < 0
        flags["infeasible_own"][:, :, j] = own_infeasible
        noaction = usable + y[:, j] - f[:, j] - p[:, j] - x[:, j]
        end = noaction - action
        flags["incremental_breach"][:, :, j] = (end < r) & (noaction >= r)
        amounts[:, :, j], ends[:, :, j] = action, end
        balance = end
    breach = ends < r[None, :, None]
    shortfall = np.maximum(r[None, :, None] - ends, 0)
    records = []
    for k, policy in enumerate(policies):
        row = pd.DataFrame(
            {
                "policy": policy,
                "household_id": panel.household_id[idx],
                "mean_monthly_investment": amounts[k].mean(axis=1),
                "cumulative_investment_12m": amounts[k].sum(axis=1),
                "reserve_breach_rate": breach[k].mean(axis=1),
                "mean_reserve_shortfall": shortfall[k].mean(axis=1),
                "mean_financing_gap": np.maximum(-ends[k], 0).mean(axis=1),
                "negative_cash_rate": (ends[k] < 0).mean(axis=1),
                "zero_investment_rate": (amounts[k] == 0).mean(axis=1),
                "household_any_breach": breach[k].any(axis=1),
                "cap_amount_binding_rate": flags["amount_binding"][k].mean(axis=1),
                "cap_income_binding_rate": flags["income_binding"][k].mean(axis=1),
                "cap_both_binding_rate": flags["both_binding"][k].mean(axis=1),
                "cap_any_binding_rate": (
                    flags["amount_binding"][k] | flags["income_binding"][k]
                ).mean(axis=1),
                "at_cap_rate": flags["at_cap"][k].mean(axis=1),
                "infeasible_rate": flags["infeasible_common"][k].mean(axis=1),
                "own_budget_infeasible_rate": flags["infeasible_own"][k].mean(axis=1),
                "incremental_breach_rate": flags["incremental_breach"][k].mean(axis=1),
            }
        )
        records.append(row)
    return amounts, ends, pd.concat(records, ignore_index=True)


def action_agreement(
    panel,
    idx,
    pred,
    amounts,
    ends,
    amount_cap,
    income_cap,
    interest_rate=0,
    tolerance=1,
    policies=None,
):
    policies = available_policies(pred) if policies is None else list(policies)
    initial = panel.initial_cash[idx]
    hgbindex = policies.index("hgb_calibrated")
    commonstart = np.concatenate([initial[:, None], ends[hgbindex, :, :-1]], axis=1)
    commonstart -= np.maximum(-commonstart, 0) * interest_rate
    y, f, p = [a[idx, 36:48] for a in [panel.income, panel.fixed, panel.planned]]
    r = panel.reserve[idx, None]
    out = []
    for a, b in PAIRS[:3]:
        if a not in pred or b not in pred:
            continue
        qa, qb = pred[a], pred[b]
        different = np.abs(qa - qb) > tolerance
        same = amounts[policies.index(a)] == amounts[policies.index(b)]
        aa = amounts[policies.index(a)]
        capfloor = np.floor(np.minimum(amount_cap, income_cap * y) / 1000) * 1000
        common = different & same
        ca, _ = actions(commonstart, y, f, p, r, qa, amount_cap, income_cap)
        cb, _ = actions(commonstart, y, f, p, r, qb, amount_cap, income_cap)
        for i, hid in enumerate(panel.household_id[idx]):
            out.append(
                {
                    "method_a": a,
                    "method_b": b,
                    "household_id": hid,
                    "prediction_different_rate": different[i].mean(),
                    "actual_action_equal_rate": same[i].mean(),
                    "prediction_diff_but_action_equal_rate": (different[i] & same[i]).mean(),
                    "prediction_diff_same_zero_rate": (common[i] & (aa[i] == 0)).mean(),
                    "prediction_diff_same_positive_at_cap_rate": (
                        common[i] & (aa[i] > 0) & (aa[i] == capfloor[i])
                    ).mean(),
                    "prediction_diff_same_positive_below_cap_rate": (
                        common[i] & (aa[i] > 0) & (aa[i] < capfloor[i])
                    ).mean(),
                    "common_state_prediction_diff_but_action_equal_rate": (
                        different[i] & (ca[i] == cb[i])
                    ).mean(),
                    "different_count": different[i].sum(),
                    "different_and_equal_count": (different[i] & same[i]).sum(),
                }
            )
    return pd.DataFrame(out)


def available_policies(predictions):
    return [
        name
        for name in POLICIES
        if name in ("no_investment", "fixed_matched") or name in predictions
    ]
