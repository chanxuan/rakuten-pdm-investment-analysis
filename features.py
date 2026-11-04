"""Explicit lag slices exclude the current and every future variable_expense."""

from __future__ import annotations

import numpy as np

from .data import Panel

FEATURE_NAMES = [
    "lag1",
    "lag3",
    "lag6",
    "lag12",
    "mean3",
    "mean6",
    "mean12",
    "std12",
    "q90_12",
    "cv12",
    "recent_ratio",
    "known_income",
    "known_fixed",
    "known_planned",
    "month_sin",
    "month_cos",
]


def build_features(
    panel: Panel, indices: np.ndarray, months: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """months are zero-based; each feature row at t sees variable_expense[:, :t] only."""
    values = []
    for t in months:
        if t < 12:
            raise ValueError("12 complete historical months required")
        hist = panel.variable_expense[indices, t - 12 : t]
        mean = hist.mean(axis=1)
        std = hist.std(axis=1)
        columns = [
            hist[:, -1],
            hist[:, -3],
            hist[:, -6],
            hist[:, -12],
            hist[:, -3:].mean(axis=1),
            hist[:, -6:].mean(axis=1),
            mean,
            std,
            np.quantile(hist, 0.9, axis=1),
            std / np.maximum(1, mean),
            hist[:, -3:].mean(axis=1) / np.maximum(1, mean),
            panel.income[indices, t],
            panel.fixed[indices, t],
            panel.planned[indices, t],
            np.full(len(indices), np.sin(2 * np.pi * (t % 12) / 12)),
            np.full(len(indices), np.cos(2 * np.pi * (t % 12) / 12)),
        ]
        values.append(np.column_stack(columns))
    features = np.stack(values, axis=1).reshape(-1, len(FEATURE_NAMES))
    target = panel.variable_expense[np.ix_(indices, months)].reshape(-1)
    return features, target


def historical_predictions(
    panel: Panel, indices: np.ndarray, months: np.ndarray
) -> dict[str, np.ndarray]:
    x, _ = build_features(panel, indices, months)
    shape = (len(indices), len(months))
    return {"history_mean": x[:, 6].reshape(shape), "history_q90": x[:, 8].reshape(shape)}


def sequence_inputs(panel, indices, months):
    """Return the same information as the table model in sequence form."""
    sequence = np.stack(
        [panel.variable_expense[indices, t - 12 : t] for t in months], axis=1
    ).reshape(-1, 12, 1)
    table, target = build_features(panel, indices, months)
    return sequence, table, target
