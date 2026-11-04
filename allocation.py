"""Single-period allocation under a forecast cash-reserve constraint."""

from __future__ import annotations

import numpy as np


def actions(balance, income, fixed, planned, reserve, budget, amount_cap, income_cap, step=1000):
    """Return the largest grid-aligned allocation and zero-action infeasibility.

    ``balance`` is measured before the current income is credited. A zero
    action does not satisfy the forecast constraint when the margin is negative.
    """
    margin = np.asarray(balance) + income - fixed - planned - reserve - budget
    cap = np.minimum(amount_cap, income_cap * np.asarray(income))
    amount = np.floor(np.maximum(0, np.minimum(margin, cap)) / step) * step
    return amount, margin < 0


def enumerate_action(
    balance, income, fixed, planned, reserve, budget, amount_cap, income_cap, step=1000
):
    """Independently enumerate feasible amounts for a scalar decision."""
    upper = int(np.floor(min(amount_cap, income_cap * income) / step)) * step
    feasible = [
        amount
        for amount in range(0, upper + 1, step)
        if balance + income - fixed - planned - reserve - budget - amount >= 0
    ]
    return (float(max(feasible)), False) if feasible else (0.0, True)
