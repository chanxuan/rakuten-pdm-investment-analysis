# Reference tables

This directory contains the saved ten-seed experiment results. All household
records are synthetic. These tables were produced before the code was reorganized
into the standalone package.

`manifest.json` identifies the experiment, records file hashes and retains the
original CSV hashes. Published CSV line endings are normalized to LF; numeric
values are unchanged. `environment.json` records the observed reference environment.

| File | Content |
| --- | --- |
| `forecast_models_raw_calibrated.csv` | Per-seed forecast loss and coverage, before and after calibration, by scenario and subgroup |
| `all_policy_metrics.csv` | Per-seed policy outcomes for all 34 scenario/cap combinations |
| `forecast_summary_ci.csv` | Forecast estimates and paired seed/household intervals |
| `paired_forecast_effects.csv` | Differences between forecast methods |
| `policy_summary_ci.csv` | Policy estimates and intervals |
| `paired_policy_effects.csv` | Differences between decision rules |
| `cap_binding_and_action_agreement.csv` | Strict cap attribution and prediction/action agreement |
| `fixed_target_selection.csv` | Validation-only fixed-target selection |
| `calibration_deltas.csv` | Each predictor's independently estimated correction |
| `hyperparameter_selection.csv` | Validation losses and selected candidates |
| `model_costs.csv` | Recorded CPU inference measurements and model-size definitions |
| `policy_subgroups.csv` | Income, reserve and expense-volatility subgroup outcomes |
| `failure_case_monthly.csv` | Selected household trajectories for interpretation |

Rates are stored as fractions. Amounts and losses are in JPY. `infeasible_rate`
uses the common calibrated-HGB forecast on each strategy's own cash path;
`own_budget_infeasible_rate` uses that strategy's own spending budget.

The main comparison is calibrated HGB minus independently calibrated historical
Q90. Additional model and cap comparisons are exploratory. Forecast pinball loss
is evaluated on variable expense X, while cash settlement deducts total expense
F + P + X.

The package checks in `package_checks.json` cover all ten generated panels,
features, 100 scenario cash-flow paths and 340 investment ledgers. Model retraining
was additionally repeated for the complete first seed, including all three HGB,
LightGBM and LSTM candidates. That seed's predictions and decisions match the saved
arrays exactly in the observed environment. The complete ten-seed training can be
repeated with the full configuration described in the root README.

Run the saved-table check from the repository root:

```sh
household-cashflow check-reference
```
