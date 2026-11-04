"""Command-line entry points for experiments and saved-result checks."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="household-cashflow")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Generate, fit, calibrate and backtest")
    run.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--extended", action="store_true", help="Include LightGBM and LSTM")
    run.add_argument("--seeds", type=int, nargs="+", help="Override the configured seed list")
    verify = commands.add_parser("verify", help="Check a completed run")
    verify.add_argument("--run", type=Path, required=True)
    reference = commands.add_parser("check-reference", help="Check the committed result snapshot")
    reference.add_argument("--path", type=Path, default=Path("results/reference"))
    plots = commands.add_parser("plot", help="Plot result tables")
    plots.add_argument("--tables", type=Path, required=True)
    plots.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "run":
            from .config import load_config, validate_config

            config = load_config(args.config)
            if args.seeds:
                config["seeds"] = args.seeds
                validate_config(config)
            for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
                os.environ.setdefault(name, str(config["threads"]))
            from threadpoolctl import threadpool_limits

            from .experiment import run_experiment

            with threadpool_limits(limits=config["threads"]):
                run_experiment(config, args.output, args.extended)
        elif args.command in ["verify", "check-reference"]:
            from .verify import verify_reference, verify_run

            result = (
                verify_run(args.run) if args.command == "verify" else verify_reference(args.path)
            )
            print(json.dumps(result, indent=2))
        elif args.command == "plot":
            from .plotting import plot_results

            plot_results(args.tables, args.output)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")
