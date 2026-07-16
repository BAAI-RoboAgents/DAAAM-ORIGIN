#!/usr/bin/env python3
"""Validate a clean-HEAD 1 Hz run or an optional legacy 5/10 Hz stress pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.quality.benchmark import (  # noqa: E402
    validate_benchmark_pair,
    validate_realtime_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=Path,
        help="Authoritative realtime run directory (default target: 1 Hz).",
    )
    parser.add_argument(
        "--expected-rate-hz",
        type=float,
        default=1.0,
        help="Configured replay/map target for --run (default: 1.0).",
    )
    parser.add_argument(
        "--run-5hz",
        type=Path,
        help="Optional legacy stress-pair 5 Hz run.",
    )
    parser.add_argument(
        "--run-10hz",
        type=Path,
        help="Optional legacy stress-pair 10 Hz run.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Development-only: return a non-authoritative verdict for dirty runs.",
    )
    parser.add_argument(
        "--allow-no-dam",
        action="store_true",
        help="Development-only: skip real DAM correction and Hydra ACK requirements.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    using_single = args.run is not None
    using_pair = args.run_5hz is not None or args.run_10hz is not None
    if using_single == using_pair:
        raise SystemExit("Specify either --run, or both --run-5hz and --run-10hz")
    if using_pair and (args.run_5hz is None or args.run_10hz is None):
        raise SystemExit("The stress mode requires both --run-5hz and --run-10hz")
    if using_single:
        verdict = validate_realtime_run(
            args.run,
            expected_rate_hz=args.expected_rate_hz,
            require_dam=not args.allow_no_dam,
            allow_dirty=args.allow_dirty,
        )
    else:
        verdict = validate_benchmark_pair(
            args.run_5hz,
            args.run_10hz,
            require_dam=not args.allow_no_dam,
            allow_dirty=args.allow_dirty,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(verdict, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(verdict, indent=2, allow_nan=False))
    if not verdict["passed"]:
        raise SystemExit(2)
    if not verdict["authoritative"] and not (args.allow_dirty or args.allow_no_dam):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
