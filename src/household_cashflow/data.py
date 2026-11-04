"""External cash flows. Hidden random streams never enter forecast features."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import stdtrit


@dataclass
class Panel:
    household_id: np.ndarray
    split: np.ndarray
    income: np.ndarray
    fixed: np.ndarray
    planned: np.ndarray
    variable_expense: np.ndarray
    reserve: np.ndarray
    initial_cash: np.ndarray
    hidden: dict[str, np.ndarray]

    @property
    def total_expense(self) -> np.ndarray:
        return self.fixed + self.planned + self.variable_expense

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)

    def save(self, path) -> None:
        """Store the complete generated base panel and audit-only random streams."""
        values = {k: v for k, v in vars(self).items() if k != "hidden"}
        values.update({"hidden_" + k: v for k, v in self.hidden.items()})
        np.savez_compressed(path, **values)


def generate(config: dict, seed: int) -> Panel:
    """Generate household cash flows using population-scale anchors.

    Income and consumption anchors refer to the mean over 48 months. Finite
    samples need not match exactly. Reserve uses only months 25--36 observations.
    """
    rng = np.random.default_rng(seed)
    g = config["generator"]
    sizes = config["households"]
    n, m = sum(sizes.values()), config["months"]
    splits = np.concatenate([np.repeat(k, v) for k, v in sizes.items()])
    ids = np.arange(n)
    zi, ze = rng.normal(size=(2, n))
    rho = g["income_spend_latent_correlation"]
    zs = rho * zi + np.sqrt(1 - rho**2) * ze
    income_scale = np.exp(g["income_log_sigma"] * zi - 0.5 * g["income_log_sigma"] ** 2)
    spend_scale = np.exp(g["spend_log_sigma"] * zs - 0.5 * g["spend_log_sigma"] ** 2)
    t = np.arange(m)
    inc_trend = np.exp(g["monthly_income_growth"] * (t - (m - 1) / 2))
    inc_trend /= inc_trend.mean()
    exp_trend = np.exp(g["monthly_expense_growth"] * (t - (m - 1) / 2))
    exp_trend /= exp_trend.mean()
    # Planned expenses are known before the decision; random variable expenses
    # consist of ordinary spending and unknown bursts.
    fixed_share = rng.uniform(*g["fixed_share_range"], n)
    planned_expected = g["planned_probability"] * np.mean(g["planned_size_range"])
    burst_expected = g["shock_probability"] * np.mean(g["shock_size_range"])
    ordinary_share = 1 - fixed_share - planned_expected - burst_expected
    level = config["consumption_anchor"] * spend_scale[:, None] * exp_trend
    fixed = level * fixed_share[:, None]
    up, ps = rng.random((n, m)), rng.uniform(*g["planned_size_range"], (n, m))
    planned = level * ps * (up < g["planned_probability"])
    interruption = rng.random((n, m)) < g["income_interruption_probability"]
    income_noise = np.exp(
        g["income_noise_sigma"] * rng.normal(size=(n, m)) - 0.5 * g["income_noise_sigma"] ** 2
    )
    income_factor = np.where(interruption, g["interrupted_income_fraction"], 1.0)
    expected_factor = 1 - g["income_interruption_probability"] * (
        1 - g["interrupted_income_fraction"]
    )
    income = (
        config["income_anchor"]
        * income_scale[:, None]
        * inc_trend
        * income_noise
        * income_factor
        / expected_factor
    )
    sigma = rng.uniform(*g["variable_log_sigma_range"], n)
    phi = rng.uniform(0, 2 * np.pi, n)
    season = (
        1
        + g["common_season_amplitude"] * np.sin(2 * np.pi * t / 12)[None, :]
        + g["private_season_amplitude"] * np.sin(2 * np.pi * t / 12 + phi[:, None])
    )
    innovations = rng.normal(size=(n, m))
    ar = innovations.copy()
    coeff = g["variable_ar_coefficient"]
    for j in range(1, m):
        ar[:, j] = coeff * ar[:, j - 1] + np.sqrt(1 - coeff**2) * innovations[:, j]
    ordinary = (
        level
        * ordinary_share[:, None]
        * season
        * np.exp(sigma[:, None] * ar - 0.5 * sigma[:, None] ** 2)
    )
    u, severity_u = rng.random((n, m)), rng.random((n, m))
    severity = g["shock_size_range"][0] + np.ptp(g["shock_size_range"]) * severity_u
    variable_expense = ordinary + level * severity * (u < g["shock_probability"])
    income, fixed, planned, variable_expense = [
        np.round(a, 2) for a in (income, fixed, planned, variable_expense)
    ]
    reserve = config["reserve_months"] * (fixed[:, 24:36] + variable_expense[:, 24:36]).mean(axis=1)
    low = rng.random(n) < g["low_initial_reserve_probability"]
    ratio = np.where(
        low,
        rng.uniform(*g["low_initial_reserve_ratio_range"], n),
        rng.uniform(*g["normal_initial_reserve_ratio_range"], n),
    )
    # No future realized flows are used in the initial snapshot.
    initial = np.round(reserve * ratio, 2)
    return Panel(
        ids,
        splits,
        income,
        fixed,
        planned,
        variable_expense,
        reserve,
        initial,
        {
            "ordinary": ordinary,
            "level": level,
            "shock_u": u,
            "severity_u": severity_u,
            "tail_u": rng.random((n, m)),
            "tail_z": rng.normal(size=(n, m)),
            "interruption": interruption,
            "sigma": sigma,
            "initial_ratio": ratio,
        },
    )


def apply_scenario(panel: Panel, config: dict, scenario: str) -> Panel:
    """Freeze training/history; change only months 37--48 using common streams."""
    s = config["scenarios"][scenario]
    if not s:
        return panel
    y, x = panel.income.copy(), panel.variable_expense.copy()
    h = panel.hidden
    start, end = 36, min(48, 36 + s.get("duration", 12))
    span = slice(start, end)
    if "shock_probability" in s:
        lo, hi = config["generator"]["shock_size_range"]
        sev = lo + (hi - lo) * h["severity_u"]
        x[:, 36:] = h["ordinary"][:, 36:] + h["level"][:, 36:] * sev[:, 36:] * (
            h["shock_u"][:, 36:] < s["shock_probability"]
        )
    if s.get("tail") == "pareto":
        a = s["pareto_shape"]
        # Mean burst severity matched to the base mechanism, variance infinite.
        mean_sev = np.mean(config["generator"]["shock_size_range"])
        sev = mean_sev * (a - 1) / a * np.power(1 - h["severity_u"][:, 36:], -1 / a)
        x[:, 36:] = h["ordinary"][:, 36:] + h["level"][:, 36:] * sev * (h["shock_u"][:, 36:] < 0.08)
    if s.get("tail") == "student_t":
        df = s["student_df"]
        shock = np.zeros_like(h["shock_u"], dtype=bool)
        shock[:, 35] = h["shock_u"][:, 35] < 0.08
        for j in range(36, 48):
            prob = np.where(shock[:, j - 1], s["persistence_probability"], 0.08)
            shock[:, j] = h["shock_u"][:, j] < prob
        noise = np.maximum(
            0.05, 1 + 0.3 * stdtrit(df, h["tail_u"][:, 36:]) / np.sqrt(df / (df - 2))
        )
        sev = 0.475 * np.exp(1.1 * h["tail_z"][:, 36:] - 0.5 * 1.1**2)
        x[:, 36:] = h["ordinary"][:, 36:] * noise + h["level"][:, 36:] * sev * shock[:, 36:]
        y[:, 36:] *= np.where(shock[:, 36:], 0.8, 1.0)
    x[:, span] *= s.get("expense_multiplier", 1.0)
    y[:, span] *= s.get("income_multiplier", 1.0)
    return replace(
        panel,
        income=np.round(y, 2),
        variable_expense=np.round(x, 2),
        initial_cash=panel.initial_cash * s.get("initial_cash_multiplier", 1.0),
        reserve=panel.reserve * s.get("reserve_multiplier", 1.0),
    )


def subgroup_labels(panel: Panel) -> tuple[dict[str, np.ndarray], dict]:
    """All thresholds use training households and observed pre-test history."""
    train = panel.indices("train")
    income = panel.income[:, 24:36].mean(axis=1)
    history = panel.variable_expense[:, 24:36]
    cv = history.std(axis=1) / np.maximum(1, history.mean(axis=1))
    cuts = np.quantile(income[train], [1 / 3, 2 / 3])
    vol_cut = np.quantile(cv[train], 0.75)
    labels = {
        "overall": np.ones(len(income), bool),
        "income_low": income <= cuts[0],
        "income_middle": (income > cuts[0]) & (income <= cuts[1]),
        "income_high": income > cuts[1],
        "low_reserve": panel.initial_cash < panel.reserve,
        "high_volatility": cv >= vol_cut,
    }
    return labels, {
        "income_cutoffs": cuts.tolist(),
        "high_cv_cutoff": float(vol_cut),
        "low_reserve_rule": "initial_cash < R (before any future observation)",
    }
