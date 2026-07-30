#!/usr/bin/env python3
"""Manage the complete G1 semantic-map module experiment lifecycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.experiments import ExperimentConfig, ExperimentManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze provenance, prepare G1 splits, materialize the E0-E18/Q1 "
            "matrix, execute selected runs, and aggregate results."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config/g1_semantic_map_experiment.yaml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    subparsers.add_parser("splits")
    subparsers.add_parser("matrix")
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--experiment")
    materialize.add_argument("--variant")
    materialize.add_argument("--repeat", type=int)
    execute = subparsers.add_parser("run")
    execute.add_argument("--experiment", required=True)
    execute.add_argument("--variant")
    execute.add_argument("--repeat", type=int)
    execute.add_argument(
        "--execute",
        action="store_true",
        help="Execute commands. Without this flag only immutable run specs are written.",
    )
    execute.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "Allow execution after a failed upstream geometry gate. Such runs "
            "remain non-formal and cannot support module-effect claims."
        ),
    )
    subparsers.add_parser("collect")
    subparsers.add_parser("preflight")
    subparsers.add_parser("inventory")
    subparsers.add_parser("validate")
    subparsers.add_parser("all")
    return parser.parse_args()


def select_runs(manager: ExperimentManager, args: argparse.Namespace):
    runs = manager.runs()
    experiment = getattr(args, "experiment", None)
    variant = getattr(args, "variant", None)
    repeat = getattr(args, "repeat", None)
    if experiment:
        runs = [run for run in runs if run.experiment_id == experiment]
    if variant:
        runs = [run for run in runs if run.variant == variant]
    if repeat is not None:
        runs = [run for run in runs if run.repeat == repeat]
    if not runs:
        raise ValueError("no experiment runs matched the requested filters")
    return runs


def main() -> int:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    manager = ExperimentManager(config)
    if args.command == "init":
        result = manager.initialize()
    elif args.command == "splits":
        result = manager.build_splits_and_annotation_tasks()
    elif args.command == "matrix":
        runs = manager.generate_registry()
        result = {"status": "complete", "runs": len(runs)}
    elif args.command == "materialize":
        records = [manager.materialize_run(run) for run in select_runs(manager, args)]
        result = {"status": "complete", "runs": records}
    elif args.command == "run":
        statuses = [
            manager.execute(
                run,
                dry_run=not args.execute,
                diagnostic=args.diagnostic,
            )
            for run in select_runs(manager, args)
        ]
        result = {"status": "complete", "runs": statuses}
    elif args.command == "collect":
        result = manager.collect()
    elif args.command == "preflight":
        result = manager.preflight()
    elif args.command == "inventory":
        result = manager.inventory_workspace()
    elif args.command == "validate":
        result = manager.validate()
    elif args.command == "all":
        manifest = manager.initialize()
        splits = manager.build_splits_and_annotation_tasks()
        runs = manager.generate_registry()
        specifications = [manager.materialize_run(run) for run in runs]
        preflight = manager.preflight()
        result = {
            "status": "complete",
            "manifest": manifest,
            "splits": splits,
            "runs": len(specifications),
            "preflight": preflight,
        }
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
