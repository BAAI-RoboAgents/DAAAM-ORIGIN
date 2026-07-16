#!/usr/bin/env python3
"""Validate paired clean-HEAD 5/10 Hz realtime benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.quality.benchmark import validate_benchmark_pair  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-5hz", required=True, type=Path)
    parser.add_argument("--run-10hz", required=True, type=Path)
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
    if not verdict["authoritative"] and not (
        args.allow_dirty or args.allow_no_dam
    ):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
