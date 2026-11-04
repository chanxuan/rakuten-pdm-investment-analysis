from pathlib import Path

import pytest
from threadpoolctl import threadpool_limits

from household_cashflow.config import load_config
from household_cashflow.data import generate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(2):
        yield


@pytest.fixture
def config():
    config = load_config(ROOT / "configs/smoke.json")
    config["households"] = {"train": 40, "validation": 10, "calibration": 10, "test": 12}
    config["candidate_models"] = [
        {**config["candidate_models"][0], "max_iter": 10, "min_samples_leaf": 15}
    ]
    config["lightgbm_candidates"] = [
        {**config["lightgbm_candidates"][0], "n_estimators": 5, "min_child_samples": 10}
    ]
    config["bootstrap_repetitions"] = 30
    config["threads"] = 2
    return config


@pytest.fixture
def panel(config):
    return generate(config, 1833)
