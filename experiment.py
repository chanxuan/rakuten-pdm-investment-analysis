"""Fit, calibrate and compare policies on independent household panels."""

from __future__ import annotations

import logging
import time
import warnings
from datetime import UTC, datetime
from importlib import import_module, metadata
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .backtest import action_agreement, available_policies, match_fixed, rollout
from .calibration import calibrated, calibration_delta
from .data import apply_scenario, generate, subgroup_labels
from .features import sequence_inputs
from .io import sha256, write_json
from .metrics import bootstrap_tables
from .models.hgb import fit_hgb

LOG = logging.getLogger(__name__)
CALIBRATED_METHODS = {
    "history_q90": "history_q90",
    "hgb": "hgb_raw",
    "lightgbm": "lightgbm_raw",
    "lstm": "lstm_raw",
}
POLICY_CI_METRICS = [
    "mean_monthly_investment",
    "reserve_breach_rate",
    "mean_reserve_shortfall",
    "infeasible_rate",
]


def raw_predictions(panel, indices, months, hgb, extended=None):
    baseline = hgb.predictions(panel, indices, months)
    raw = {key: baseline[key] for key in ["history_mean", "history_q90", "hgb_raw"]}
    clipped = {"hgb": 0.0}
    if extended is not None:
        sequence, features, _ = sequence_inputs(panel, indices, months)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            light = extended["lightgbm"].predict(features)
        lstm = extended["lstm"].predict_unclipped(sequence, features)
        for name, values in [("lightgbm", light), ("lstm", lstm)]:
            clipped[name] = float(np.mean(values < 0))
            raw[name + "_raw"] = np.maximum(0, values).reshape(len(indices), len(months))
    return raw, clipped


def correct_predictions(raw, deltas):
    result = dict(raw)
    for name, raw_key in CALIBRATED_METHODS.items():
        if raw_key in raw:
            result[name + "_calibrated"] = calibrated(raw[raw_key], deltas[name])
    return result


def forecast_households(panel, indices, predictions, seed, scenario):
    actual = panel.variable_expense[indices, 36:48]
    rows = []
    for method, prediction in predictions.items():
        error = actual - prediction
        rows.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "method": method,
                    "household_id": panel.household_id[indices],
                    "pinball_loss": np.maximum(0.9 * error, -0.1 * error).mean(axis=1),
                    "coverage": (actual <= prediction).mean(axis=1),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def subgroup_summary(frame, labels, method_column, metrics):
    rows = []
    for name, mask in labels.items():
        selected = frame[mask[frame.household_id.to_numpy()]]
        for method, group in selected.groupby(method_column):
            rows.append(
                {
                    "subgroup": name,
                    method_column: method,
                    "households": len(group),
                    **group[metrics].mean().to_dict(),
                }
            )
    return pd.DataFrame(rows)


def _case_records(panel, indices, predictions, investment, balance, policies, labels, seed):
    k = policies.index("hgb_calibrated")
    shortfall = np.maximum(panel.reserve[indices, None] - balance[k], 0)
    candidates = {}
    for name in ["low_reserve", "high_volatility"]:
        mask = labels[name][indices]
        if mask.any():
            candidates[name] = int(np.argmax(np.where(mask, shortfall.sum(axis=1), -1)))
    candidates["typical"] = int(
        np.argmin(np.abs(investment[k].mean(axis=1) - np.median(investment[k].mean(axis=1))))
    )
    rows = []
    for name, i in candidates.items():
        for method_index, policy in enumerate(policies):
            rows.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "case_group": name,
                        "household_id": panel.household_id[indices[i]],
                        "month": np.arange(37, 49),
                        "policy": policy,
                        "investment": investment[method_index, i],
                        "balance_end": balance[method_index, i],
                        "reserve": panel.reserve[indices[i]],
                        "actual_variable": panel.variable_expense[indices[i], 36:48],
                        "known_income": panel.income[indices[i], 36:48],
                        "budget": predictions.get(policy, np.zeros((len(indices), 12)))[i],
                        "reserve_breach": balance[method_index, i] < panel.reserve[indices[i]],
                        "shortfall": np.maximum(
                            panel.reserve[indices[i]] - balance[method_index, i], 0
                        ),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def run_experiment(config, output, include_extended=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new run directory")
    if include_extended:
        try:
            import torch

            import_module(".models.extended", __package__)
        except ImportError as exc:
            raise ImportError(
                'Install the extended models with: pip install -e ".[extended]"'
            ) from exc
        torch.set_num_threads(config["threads"])
        torch.set_num_interop_threads(1)
    for folder in ["tables", "models", "panels", "predictions", "ledgers", "training"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]
    if include_extended:
        packages += ["torch", "lightgbm"]
    package_root = Path(__file__).parent
    manifest = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "model_set": "extended" if include_extended else "hgb",
        "seeds": config["seeds"],
        "threads": config["threads"],
        "versions": {name: metadata.version(name) for name in packages},
        "source_sha256": {
            path.relative_to(package_root).as_posix(): sha256(path)
            for path in sorted(package_root.rglob("*.py"))
        },
    }
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        _evaluate(config, output, include_extended)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        write_json(output / "manifest.json", manifest)
        raise
    manifest.update(
        status="complete",
        elapsed_seconds=time.perf_counter() - started,
        completed_utc=datetime.now(UTC).isoformat(),
    )
    manifest["artifact_sha256"] = {
        p.relative_to(output).as_posix(): sha256(p)
        for folder in ["models", "panels", "predictions", "ledgers", "tables", "training"]
        for p in sorted((output / folder).glob("*"))
        if p.is_file()
    }
    manifest["artifact_sha256"]["config.json"] = sha256(output / "config.json")
    write_json(output / "manifest.json", manifest)
    LOG.info("Complete: %s", output)


def _evaluate(config, output, include_extended):
    forecasts, forecast_groups, households, summaries = [], [], [], []
    agreements, policy_groups, selections, calibrations, training, epochs, population = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    caps = list(product(config["fixed_caps"], config["income_caps"]))
    for seed in config["seeds"]:
        LOG.info("Seed %s: generate and fit", seed)
        panel = generate(config, seed)
        panel.save(output / f"panels/{seed}.npz")
        population.append(
            {
                "seed": seed,
                "income_mean": panel.income.mean(),
                "total_expense_mean": (panel.fixed + panel.planned + panel.variable_expense).mean(),
            }
        )
        hgb = fit_hgb(panel, config, seed)
        joblib.dump(hgb, output / f"models/hgb_{seed}.joblib")
        hgb_records = [
            {
                "seed": seed,
                "model": "hgb",
                **r,
                "selected": r["candidate"] == hgb.metadata["selected_candidate"],
            }
            for r in hgb.metadata["candidates"]
        ]
        training.extend(hgb_records)
        extended = None
        if include_extended:
            from .models.extended import fit_extended_models

            light, lstm, records, epoch_records = fit_extended_models(panel, config, seed, LOG.info)
            extended = {"lightgbm": light, "lstm": lstm}
            joblib.dump(extended, output / f"models/extended_{seed}.joblib", compress=3)
            training.extend(records)
            epochs.extend(epoch_records)
            write_json(output / f"training/normalization_{seed}.json", lstm.provenance)
        write_json(output / f"training/hgb_{seed}.json", hgb.metadata)
        labels, thresholds = subgroup_labels(panel)
        labels["low_volatility"] = ~labels["high_volatility"]
        write_json(output / f"training/subgroups_{seed}.json", thresholds)
        calibration_ids = panel.indices("calibration")
        cal_raw, _ = raw_predictions(panel, calibration_ids, np.arange(24, 36), hgb, extended)
        actual = panel.variable_expense[calibration_ids, 24:36]
        deltas = {
            name: calibration_delta(cal_raw[key], actual)
            for name, key in CALIBRATED_METHODS.items()
            if key in cal_raw
        }
        np.testing.assert_allclose(deltas["hgb"], hgb.correction, atol=1e-8)
        calibrations.append(
            {"seed": seed, **deltas, "split": "calibration", "months": "25-36", "rows": actual.size}
        )
        validation_ids = panel.indices("validation")
        val_raw, _ = raw_predictions(panel, validation_ids, np.arange(24, 36), hgb, extended)
        val_predictions = correct_predictions(val_raw, deltas)
        targets = {}
        for amount_cap, income_cap in caps:
            target, details = match_fixed(
                panel,
                val_predictions["hgb_calibrated"],
                config,
                amount_cap,
                income_cap,
                config["fixed_search_max"],
            )
            targets[(amount_cap, income_cap)] = target
            selections.append(
                {
                    "seed": seed,
                    "amount_cap": amount_cap,
                    "income_cap": income_cap,
                    **{k: v for k, v in details.items() if k != "candidates"},
                }
            )
            write_json(
                output / f"training/fixed_{seed}_{amount_cap}_{income_cap:.2f}.json", details
            )
        for scenario in config["scenarios"]:
            scenario_panel = apply_scenario(panel, config, scenario)
            indices = scenario_panel.indices("test")
            raw, clipped = raw_predictions(
                scenario_panel, indices, np.arange(36, 48), hgb, extended
            )
            prediction = correct_predictions(raw, deltas)
            policies = available_policies(prediction)
            np.savez_compressed(
                output / f"predictions/{seed}_{scenario}.npz",
                household_id=scenario_panel.household_id[indices],
                known_income=scenario_panel.income[indices, 36:48],
                known_fixed=scenario_panel.fixed[indices, 36:48],
                known_planned=scenario_panel.planned[indices, 36:48],
                actual_variable=scenario_panel.variable_expense[indices, 36:48],
                reserve=scenario_panel.reserve[indices],
                initial_cash=scenario_panel.initial_cash[indices],
                **prediction,
            )
            forecast = forecast_households(scenario_panel, indices, prediction, seed, scenario)
            forecasts.append(forecast)
            groups = subgroup_summary(forecast, labels, "method", ["pinball_loss", "coverage"])
            groups["seed"], groups["scenario"] = seed, scenario
            delta_lookup = {name + "_calibrated": value for name, value in deltas.items()}
            groups["calibration_delta"] = groups.method.map(delta_lookup).fillna(0)
            groups["calibration_applied"] = groups.method.str.endswith("_calibrated")
            groups["coverage_minus_90"] = groups.coverage - 0.9
            groups["negative_raw_prediction_rate"] = groups.method.map(
                {method: clipped.get(method.split("_")[0], 0.0) for method in prediction}
            ).fillna(0)
            forecast_groups.append(groups)
            scenario_caps = caps if scenario in config["grid_scenarios"] else [(40000, 0.15)]
            interest_rate = config["scenarios"][scenario].get("monthly_debt_interest", 0.0)
            for amount_cap, income_cap in scenario_caps:
                investment, balance, hh = rollout(
                    scenario_panel,
                    indices,
                    prediction,
                    amount_cap,
                    income_cap,
                    targets[(amount_cap, income_cap)],
                    interest_rate,
                    policies=policies,
                )
                metrics = [
                    c for c in hh.select_dtypes(include=["number", "bool"]) if c != "household_id"
                ]
                summary = hh.groupby("policy")[metrics].mean().reset_index()
                for i, policy in enumerate(policies):
                    shortfall = np.maximum(scenario_panel.reserve[indices, None] - balance[i], 0)
                    summary.loc[summary.policy == policy, "shortfall_p95"] = np.quantile(
                        shortfall, 0.95
                    )
                    count = max(1, int(np.ceil(shortfall.size * 0.05)))
                    summary.loc[summary.policy == policy, "shortfall_cvar95"] = np.sort(
                        shortfall.ravel()
                    )[-count:].mean()
                group_key = f"{scenario}|{amount_cap}|{income_cap:.2f}"
                context = {
                    "seed": seed,
                    "scenario": scenario,
                    "amount_cap": amount_cap,
                    "income_cap": income_cap,
                }
                group = subgroup_summary(hh, labels, "policy", POLICY_CI_METRICS)
                for key, value in context.items():
                    hh[key], summary[key], group[key] = value, value, value
                hh["groupkey"] = group_key
                households.append(hh)
                policy_groups.append(group)
                low = group[group.subgroup == "low_reserve"].set_index("policy").reserve_breach_rate
                summary["low_reserve_breach_rate"] = summary.policy.map(low)
                summary["fixed_target"] = targets[(amount_cap, income_cap)]
                summaries.append(summary)
                agreement = action_agreement(
                    scenario_panel,
                    indices,
                    prediction,
                    investment,
                    balance,
                    amount_cap,
                    income_cap,
                    interest_rate,
                    config["prediction_difference_tolerance_yen"],
                    policies,
                )
                for key, value in context.items():
                    agreement[key] = value
                agreements.append(agreement)
                np.savez_compressed(
                    output / f"ledgers/{seed}_{scenario}_{amount_cap}_{income_cap:.2f}.npz",
                    investment=investment,
                    balance_end=balance,
                    policies=np.array(policies),
                    household_id=scenario_panel.household_id[indices],
                    amount_cap=amount_cap,
                    income_cap=income_cap,
                    fixed_target=targets[(amount_cap, income_cap)],
                    monthly_debt_interest=interest_rate,
                )
                if (
                    scenario == "baseline"
                    and amount_cap == 40000
                    and income_cap == 0.15
                    and seed == config["seeds"][0]
                ):
                    cases = _case_records(
                        scenario_panel,
                        indices,
                        prediction,
                        investment,
                        balance,
                        policies,
                        labels,
                        seed,
                    )
                    cases.to_csv(output / "tables/failure_case_monthly.csv", index=False)
            LOG.info("Seed %s: %s evaluated", seed, scenario)
    forecast_hh = pd.concat(forecasts, ignore_index=True)
    policy_hh = pd.concat(households, ignore_index=True)
    policy_summary = pd.concat(summaries, ignore_index=True)
    tables = {
        "forecast_models_raw_calibrated.csv": pd.concat(forecast_groups, ignore_index=True),
        "all_policy_metrics.csv": policy_summary,
        "policy_subgroups.csv": pd.concat(policy_groups, ignore_index=True),
        "fixed_target_selection.csv": pd.DataFrame(selections),
        "calibration_deltas.csv": pd.DataFrame(calibrations),
        "hyperparameter_selection.csv": pd.DataFrame(training),
        "population_calibration.csv": pd.DataFrame(population),
    }
    if epochs:
        tables["lstm_epoch_history.csv"] = pd.DataFrame(epochs)
    agreement_hh = pd.concat(agreements, ignore_index=True)
    keys = ["seed", "scenario", "amount_cap", "income_cap", "method_a", "method_b"]
    rates = [c for c in agreement_hh if c.endswith("_rate")]
    aggregation = {c: "mean" for c in rates}
    aggregation.update(different_count="sum", different_and_equal_count="sum")
    agreement = agreement_hh.groupby(keys).agg(aggregation).reset_index()
    agreement["conditional_equal_rate_given_prediction_diff"] = (
        agreement.different_and_equal_count / np.maximum(1, agreement.different_count)
    )
    for side in ["a", "b"]:
        binding = [
            "cap_amount_binding_rate",
            "cap_income_binding_rate",
            "cap_both_binding_rate",
            "cap_any_binding_rate",
            "zero_investment_rate",
        ]
        lookup = policy_summary[
            ["seed", "scenario", "amount_cap", "income_cap", "policy", *binding]
        ].rename(columns={"policy": "method_" + side, **{c: side + "_" + c for c in binding}})
        agreement = agreement.merge(
            lookup,
            on=["seed", "scenario", "amount_cap", "income_cap", "method_" + side],
            validate="many_to_one",
        )
    tables["cap_binding_and_action_agreement.csv"] = agreement
    LOG.info("Computing paired seed/household intervals")
    fci, fpairs = bootstrap_tables(forecast_hh, "method", ["pinball_loss", "coverage"], config)
    tables["forecast_summary_ci.csv"], tables["paired_forecast_effects.csv"] = fci, fpairs
    policy_hh = policy_hh.drop(columns="scenario").rename(columns={"groupkey": "scenario"})
    pci, ppairs = bootstrap_tables(policy_hh, "policy", POLICY_CI_METRICS, config)
    tables["policy_summary_ci.csv"], tables["paired_policy_effects.csv"] = pci, ppairs
    for name, frame in tables.items():
        frame.to_csv(output / "tables" / name, index=False, encoding="utf-8")
