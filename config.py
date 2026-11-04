"""Load a study configuration, including relative profile inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .io import read_json


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError("Cyclic configuration inheritance")
    seen.add(path)
    config = read_json(path)
    parent = config.pop("extends", None)
    if parent:
        config = _merge(load_config(path.parent / parent, seen), config)
    validate_config(config)
    return config


def validate_config(config):
    expected = {
        "train_months": [13, 36],
        "validation_months": [25, 36],
        "calibration_months": [25, 36],
        "evaluation_months": [37, 48],
    }
    if config["months"] != 48 or config["quantile"] != 0.9:
        raise ValueError("This study requires 48 months and the 0.9 quantile")
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f"{key} must be {value} for this study")
    if list(config["households"]) != ["train", "validation", "calibration", "test"]:
        raise ValueError(
            "Household splits must be train, validation, calibration, test in that order"
        )
    if any(not isinstance(n, int) or n <= 0 for n in config["households"].values()):
        raise ValueError("Household counts must be positive integers")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    if config["step"] != 1000 or config["investment_step"] != 1000:
        raise ValueError("The decision grid is fixed at 1,000 yen")
    if any(cap <= 0 or cap % 1000 for cap in config["fixed_caps"]):
        raise ValueError("Amount caps must be positive multiples of 1,000 yen")
    if any(not 0 < cap <= 1 for cap in config["income_caps"]):
        raise ValueError("Income fractions must be in (0, 1]")
    if 40000 not in config["fixed_caps"] or 0.15 not in config["income_caps"]:
        raise ValueError("The main 40,000 yen / 15% condition must be included")
    if config["threads"] < 1 or config["bootstrap_repetitions"] < 2:
        raise ValueError("Invalid thread count or bootstrap repetitions")
