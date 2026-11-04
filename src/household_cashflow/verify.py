"""Check saved predictions, decision bounds and cash-flow identities."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json, sha256


def verify_run(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    config = read_json(root / "config.json")
    if manifest["status"] != "complete":
        raise ValueError("Run is incomplete")
    for name, expected in manifest["artifact_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise AssertionError(f"Artifact hash mismatch: {name}")
    metrics = pd.read_csv(root / "tables/all_policy_metrics.csv")
    forecasts = pd.read_csv(root / "tables/forecast_models_raw_calibrated.csv")
    agreement = pd.read_csv(root / "tables/cap_binding_and_action_agreement.csv")
    targets = pd.read_csv(root / "tables/fixed_target_selection.csv")
    if (
        not targets.selection_split.eq("validation").all()
        or not targets.snapshot_month.eq(24).all()
    ):
        raise AssertionError("Fixed targets must use the validation month-24 snapshot")
    keys = ["seed", "scenario", "amount_cap", "income_cap", "policy"]
    if metrics.duplicated(keys).any():
        raise AssertionError("Duplicate policy summaries")
    lookup = metrics.set_index(keys)
    total = 0
    ledger_count = 0
    for seed in config["seeds"]:
        for scenario in config["scenarios"]:
            with np.load(root / f"predictions/{seed}_{scenario}.npz", allow_pickle=False) as z:
                p = {key: z[key] for key in z.files}
            n = config["households"]["test"]
            assert p["actual_variable"].shape == (n, 12)
            fc = forecasts[
                (forecasts.seed == seed)
                & (forecasts.scenario == scenario)
                & (forecasts.subgroup == "overall")
            ].set_index("method")
            for method, row in fc.iterrows():
                predicted = p[method]
                assert (
                    predicted.shape == (n, 12)
                    and np.isfinite(predicted).all()
                    and (predicted >= 0).all()
                )
                error = p["actual_variable"] - predicted
                np.testing.assert_allclose(
                    row.pinball_loss, np.maximum(0.9 * error, -0.1 * error).mean(), atol=1e-7
                )
                np.testing.assert_allclose(
                    row.coverage, np.mean(p["actual_variable"] <= predicted), atol=1e-12
                )
                if method.endswith("_calibrated"):
                    raw = {
                        "history_q90_calibrated": "history_q90",
                        "hgb_calibrated": "hgb_raw",
                        "lightgbm_calibrated": "lightgbm_raw",
                        "lstm_calibrated": "lstm_raw",
                    }[method]
                    np.testing.assert_allclose(
                        predicted, np.maximum(0, p[raw] + row.calibration_delta), atol=1e-7
                    )
            caps = (
                list(product(config["fixed_caps"], config["income_caps"]))
                if scenario in config["grid_scenarios"]
                else [(40000, 0.15)]
            )
            for amount_cap, income_cap in caps:
                filename = f"{seed}_{scenario}_{amount_cap}_{income_cap:.2f}.npz"
                with np.load(root / "ledgers" / filename, allow_pickle=False) as z:
                    a, end, policies = z["investment"], z["balance_end"], list(z["policies"])
                    fixed_target, rate = float(z["fixed_target"]), float(z["monthly_debt_interest"])
                assert a.shape == end.shape == (len(policies), n, 12)
                assert np.isfinite(end).all() and (a >= 0).all() and (a % 1000 == 0).all()
                cap = np.minimum(amount_cap, income_cap * p["known_income"])
                assert (a <= cap + 1e-7).all()
                starts = np.concatenate(
                    [
                        np.broadcast_to(p["initial_cash"], (len(policies), n))[:, :, None],
                        end[:, :, :-1],
                    ],
                    axis=2,
                )
                net = (
                    starts
                    - np.maximum(-starts, 0) * rate
                    + p["known_income"]
                    - p["known_fixed"]
                    - p["known_planned"]
                )
                np.testing.assert_allclose(end, net - p["actual_variable"] - a, atol=1e-7)
                budgets = np.stack([p.get(policy, np.zeros((n, 12))) for policy in policies])
                margins = net - p["reserve"][None, :, None] - budgets
                expected = np.floor(np.maximum(0, np.minimum(margins, cap)) / 1000) * 1000
                expected[0] = 0
                expected[1] = np.minimum(expected[1], fixed_target)
                np.testing.assert_array_equal(a, expected)
                desired = np.floor(np.maximum(0, margins) / 1000) * 1000
                desired[0] = 0
                desired[1] = np.minimum(desired[1], fixed_target)
                binding = desired > a
                for k, policy in enumerate(policies):
                    row = lookup.loc[(seed, scenario, amount_cap, income_cap, policy)]
                    np.testing.assert_allclose(row.mean_monthly_investment, a[k].mean(), atol=1e-7)
                    np.testing.assert_allclose(
                        row.cumulative_investment_12m, a[k].sum(axis=1).mean(), atol=1e-7
                    )
                    np.testing.assert_allclose(
                        row.reserve_breach_rate, np.mean(end[k] < p["reserve"][:, None]), atol=1e-12
                    )
                    np.testing.assert_allclose(
                        row.mean_reserve_shortfall,
                        np.maximum(p["reserve"][:, None] - end[k], 0).mean(),
                        atol=1e-7,
                    )
                    np.testing.assert_allclose(
                        row.cap_any_binding_rate, binding[k].mean(), atol=1e-12
                    )
                    common_margin = net[k] - p["reserve"][:, None] - p["hgb_calibrated"]
                    np.testing.assert_allclose(
                        row.infeasible_rate, np.mean(common_margin < 0), atol=1e-12
                    )
                ag = agreement[
                    (agreement.seed == seed)
                    & (agreement.scenario == scenario)
                    & (agreement.amount_cap == amount_cap)
                    & (agreement.income_cap == income_cap)
                ]
                for row in ag.itertuples():
                    different = (
                        np.abs(p[row.method_a] - p[row.method_b])
                        > config["prediction_difference_tolerance_yen"]
                    )
                    same = a[policies.index(row.method_a)] == a[policies.index(row.method_b)]
                    np.testing.assert_allclose(
                        row.conditional_equal_rate_given_prediction_diff,
                        np.sum(different & same) / max(1, np.sum(different)),
                        atol=1e-12,
                    )
                total += a.size
                ledger_count += 1
    for name in [
        "forecast_summary_ci.csv",
        "paired_forecast_effects.csv",
        "policy_summary_ci.csv",
        "paired_policy_effects.csv",
    ]:
        frame = pd.read_csv(root / "tables" / name)
        assert np.isfinite(frame[["ci95_low", "ci95_high"]]).all().all()
        assert frame.ci95_low.le(frame.ci95_high).all()
    return {
        "status": "passed",
        "seeds": len(config["seeds"]),
        "ledger_files": ledger_count,
        "household_month_policy_rows": total,
    }


def verify_reference(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    for filename, expected in manifest["files"].items():
        if sha256(root / filename) != expected:
            raise AssertionError(f"Reference file changed: {filename}")
    fc = pd.read_csv(root / "forecast_models_raw_calibrated.csv")
    base = fc[(fc.scenario == "baseline") & (fc.subgroup == "overall")]
    assert base.seed.nunique() == 10 and base.method.nunique() == 9
    points = base.groupby("method")[["pinball_loss", "coverage"]].mean()
    improvement = (
        1
        - points.loc["hgb_calibrated", "pinball_loss"]
        / points.loc["history_q90_calibrated", "pinball_loss"]
    )
    np.testing.assert_allclose(improvement, 0.06656001454930327, atol=1e-12)
    np.testing.assert_allclose(
        points.loc["history_q90_calibrated", "coverage"], 54027 / 60000, atol=1e-12
    )
    policy = pd.read_csv(root / "all_policy_metrics.csv")
    assert len(policy) == 3060
    return {
        "status": "passed",
        "files": len(manifest["files"]),
        "seeds": 10,
        "hgb_relative_pinball_improvement": float(improvement),
    }
