"""Figures built from saved experiment tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

LABELS = {
    "history_mean": "Historical mean",
    "history_q90_calibrated": "Historical Q90",
    "hgb_calibrated": "HGB",
    "lightgbm_calibrated": "LightGBM",
    "lstm_calibrated": "LSTM",
}
COLORS = {
    "history_mean": "#9a9a9a",
    "history_q90_calibrated": "#606060",
    "hgb_calibrated": "#2d628e",
    "lightgbm_calibrated": "#7ca6c6",
    "lstm_calibrated": "#b5c9d8",
}


def _save(fig, output, name):
    fig.savefig(output / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_results(tables, output):
    tables, output = Path(tables), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    forecasts = pd.read_csv(tables / "forecast_models_raw_calibrated.csv")
    policies = pd.read_csv(tables / "all_policy_metrics.csv")
    baseline = forecasts[(forecasts.scenario == "baseline") & (forecasts.subgroup == "overall")]
    baseline = baseline.groupby("method")[["pinball_loss", "coverage"]].mean()
    methods = [name for name in LABELS if name in baseline.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for ax, metric, title in zip(
        axes, ["pinball_loss", "coverage"], ["Mean Q90 pinball loss (JPY)", "Observed coverage"]
    ):
        ax.barh(
            [LABELS[m] for m in methods],
            baseline.loc[methods, metric],
            color=[COLORS[m] for m in methods],
        )
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    axes[0].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[1].set_xlim(0, 1)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].axvline(0.9, color="#b34a38", linestyle="--", linewidth=1.2)
    fig.suptitle("Expense forecasting on independent test households")
    _save(fig, output, "forecast-comparison.png")

    cap = policies[(policies.scenario == "baseline") & (policies.policy == "hgb_calibrated")]
    cap = (
        cap.groupby(["amount_cap", "income_cap"])[
            ["mean_monthly_investment", "reserve_breach_rate"]
        ]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.0), layout="constrained")
    for amount, group in cap.groupby("amount_cap"):
        group = group.sort_values("income_cap")
        ax.plot(
            group.mean_monthly_investment,
            group.reserve_breach_rate,
            "o-",
            label=f"Amount cap: {amount:,.0f} JPY",
        )
        for r in group.itertuples():
            ax.annotate(
                f"{r.income_cap:.0%}",
                (r.mean_monthly_investment, r.reserve_breach_rate),
                xytext=(4, 7),
                textcoords="offset points",
                fontsize=9,
            )
    ax.set(
        xlabel="Mean monthly investment (JPY)",
        ylabel="Reserve shortfall rate",
        title="Allocation under different investment caps",
    )
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="best")
    _save(fig, output, "risk-investment.png")

    scenarios = ["baseline", "shock_probability_18", "joint_shock"]
    labels = ["Baseline", "More expense shocks", "Joint shock"]
    f = (
        forecasts[(forecasts.method == "hgb_calibrated") & (forecasts.subgroup == "overall")]
        .groupby("scenario")
        .coverage.mean()
    )
    p = (
        policies[
            (policies.policy == "hgb_calibrated")
            & (policies.amount_cap == 40000)
            & (policies.income_cap == 0.15)
        ]
        .groupby("scenario")
        .reserve_breach_rate.mean()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, values, title in zip(
        axes, [f.loc[scenarios], p.loc[scenarios]], ["Expense coverage", "Reserve shortfall rate"]
    ):
        ax.bar(np.arange(3), values, color=["#2d628e", "#cc9a46", "#b34a38"])
        ax.set_xticks(np.arange(3), labels, rotation=12)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.9, color="#555555", linestyle="--", linewidth=1)
    fig.suptitle("Frozen HGB model and calibration under distribution shift")
    _save(fig, output, "stress-tests.png")

    cases = pd.read_csv(tables / "failure_case_monthly.csv")
    names = [
        name
        for name in ["low_reserve", "high_volatility", "typical"]
        if name in set(cases.case_group)
    ]
    fig, axes = plt.subplots(
        len(names), 1, figsize=(10, 3.1 * len(names)), squeeze=False, layout="constrained"
    )
    for ax, name in zip(axes[:, 0], names):
        group = cases[cases.case_group == name]
        for policy, label in [
            ("no_investment", "No investment"),
            ("history_q90_calibrated", "Historical Q90"),
            ("hgb_calibrated", "HGB"),
        ]:
            g = group[group.policy == policy].sort_values("month")
            if not g.empty:
                ax.plot(g.month, g.balance_end, marker=".", label=label)
        ax.axhline(group.reserve.iloc[0], color="#b34a38", linestyle="--", label="Minimum reserve")
        ax.set(
            title=f"{name.replace('_', ' ').title()} example: household {group.household_id.iloc[0]}",
            xlabel="Month",
            ylabel="End-of-month net cash (JPY)",
        )
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, ncol=2, fontsize=9)
    _save(fig, output, "household-cases.png")
