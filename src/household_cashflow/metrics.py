"""Household-level estimates and paired two-stage bootstrap intervals."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def bootstrap_tables(household: pd.DataFrame, method_col: str, metrics: list[str], config: dict):
    """Paired two-stage bootstrap: seeds, then entire households within seed.

    The same household weights apply to every method and metric. All 12 months
    of a household stay together. Also emit intervals conditional on fitted seeds.
    Two-stage intervals account for shared fitted-model variation across seeds.
    """
    rng = np.random.default_rng(config["bootstrap_seed"])
    repetitions = config["bootstrap_repetitions"]
    summaries, pairs = [], []
    for scenario, group in household.groupby("scenario", sort=True):
        seeds = sorted(group.seed.unique())
        methods = sorted(group[method_col].unique())
        seed_arrays = []
        for seed in seeds:
            gs = group.loc[group.seed == seed]
            ids = sorted(gs.household_id.unique())
            matrix = np.stack(
                [
                    gs.loc[gs[method_col] == method]
                    .set_index("household_id")
                    .loc[ids, metrics]
                    .to_numpy(float)
                    for method in methods
                ],
                axis=1,
            )
            if not np.all(np.isfinite(matrix)):
                raise AssertionError("Missing or nonfinite paired household metrics")
            seed_arrays.append(matrix)
        if len({len(a) for a in seed_arrays}) != 1:
            raise AssertionError("Equal household count per seed expected")
        arr = np.stack(seed_arrays)  # seed, household, method, metric
        ns, nh, nm, nk = arr.shape
        boot = np.empty((repetitions, ns, nm, nk))
        for s in range(ns):
            weights = rng.multinomial(nh, np.full(nh, 1 / nh), size=repetitions).astype(float) / nh
            boot[:, s] = (weights @ arr[s].reshape(nh, -1)).reshape(repetitions, nm, nk)
        seed_draws = rng.integers(0, ns, size=(repetitions, ns))
        hierarchical = boot[np.arange(repetitions)[:, None], seed_draws].mean(axis=1)
        conditional = boot.mean(axis=1)
        point = arr.mean(axis=(0, 1))
        seed_mean = arr.mean(axis=1)
        for mi, method in enumerate(methods):
            for ki, metric in enumerate(metrics):
                lo, hi = np.quantile(hierarchical[:, mi, ki], [0.025, 0.975])
                clo, chi = np.quantile(conditional[:, mi, ki], [0.025, 0.975])
                summaries.append(
                    {
                        "scenario": scenario,
                        method_col: method,
                        "metric": metric,
                        "estimate": point[mi, ki],
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "household_only_ci95_low": clo,
                        "household_only_ci95_high": chi,
                        "seed_min": seed_mean[:, mi, ki].min(),
                        "seed_max": seed_mean[:, mi, ki].max(),
                        "seeds": ns,
                        "households_per_seed": nh,
                        "bootstrap_repetitions": repetitions,
                        "interval": "paired hierarchical seed/household percentile bootstrap",
                    }
                )
        for ia, ib in combinations(range(nm), 2):
            for ki, metric in enumerate(metrics):
                delta = hierarchical[:, ia, ki] - hierarchical[:, ib, ki]
                lo, hi = np.quantile(delta, [0.025, 0.975])
                pairs.append(
                    {
                        "scenario": scenario,
                        "method_a": methods[ia],
                        "method_b": methods[ib],
                        "metric": metric,
                        "difference_a_minus_b": point[ia, ki] - point[ib, ki],
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "seed_min_difference": (seed_mean[:, ia, ki] - seed_mean[:, ib, ki]).min(),
                        "seed_max_difference": (seed_mean[:, ia, ki] - seed_mean[:, ib, ki]).max(),
                        "interval": "paired hierarchical seed/household percentile bootstrap",
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(pairs)
