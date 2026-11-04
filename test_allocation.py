import numpy as np

from household_cashflow.allocation import actions, enumerate_action


def test_analytic_solution_matches_enumeration():
    rng = np.random.default_rng(99001)
    for amount_cap in [20000, 40000, 60000]:
        for income_cap in [0.10, 0.15, 0.20]:
            for _ in range(150):
                values = rng.uniform([-2e5, 0, 0, 0, 0, 0], [8e5, 7e5, 25e4, 2e5, 7e5, 25e4])
                actual = actions(*values, amount_cap, income_cap)
                expected = enumerate_action(*values, amount_cap, income_cap)
                assert actual == expected


def test_cash_budget_and_realized_cash_are_distinct():
    amount, infeasible = actions(300000, 220000, 120000, 20000, 300000, 45000, 40000, 0.15)
    assert amount == 33000 and not infeasible
    actual_variable = 55000
    assert 300000 + 220000 - 120000 - 20000 - actual_variable - amount == 292000


def test_zero_action_can_be_infeasible():
    amount, infeasible = actions(100000, 180000, 120000, 20000, 200000, 50000, 40000, 0.15)
    assert amount == 0 and infeasible


def test_equal_margin_is_feasible_and_rounding_is_downward():
    amount, infeasible = actions(100000, 200000, 100000, 0, 150000, 50000, 40000, 0.15)
    assert amount == 0 and not infeasible
    amount, _ = actions(120999, 200000, 100000, 0, 150000, 50000, 40000, 0.15)
    assert amount == 20000
