import json
from pathlib import Path

import pytest

from household_cashflow.config import load_config, validate_config
from household_cashflow.experiment import run_experiment
from household_cashflow.verify import verify_run


def test_inherited_smoke_configuration_is_complete():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/smoke.json")
    assert sum(cfg["households"].values()) == 250
    assert cfg["lstm"]["max_epochs"] == 2 and cfg["lstm"]["batch_size"] == 256
    assert len(cfg["seeds"]) == 1


def test_cyclic_profile_is_rejected(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"extends": "a.json"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(path)


def test_configuration_cannot_move_tuning_into_test_period(config):
    config["calibration_months"] = [37, 48]
    with pytest.raises(ValueError, match="calibration_months"):
        validate_config(config)


def test_complete_small_run_and_saved_artifact_verification(config, tmp_path):
    config["scenarios"] = {"baseline": {}}
    config["grid_scenarios"] = ["baseline"]
    config["fixed_caps"] = [40000]
    config["income_caps"] = [0.15]
    output = tmp_path / "run"
    run_experiment(config, output)
    check = verify_run(output)
    assert check["status"] == "passed" and check["ledger_files"] == 1
    with pytest.raises(FileExistsError):
        run_experiment(config, output)
    saved_config = output / "config.json"
    original_config = saved_config.read_bytes()
    saved_config.write_bytes(original_config + b" ")
    with pytest.raises(AssertionError, match="hash mismatch"):
        verify_run(output)
    saved_config.write_bytes(original_config)
    path = next((output / "ledgers").glob("*.npz"))
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(AssertionError, match="hash mismatch"):
        verify_run(output)
