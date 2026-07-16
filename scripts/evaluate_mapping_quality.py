#!/usr/bin/env python3
"""Evaluate a mapping run's machine-readable stage quality evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.quality import QualityGateConfig, QualityGateRunner  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "realtime_quality_gates.yaml",
    )
    parser.add_argument(
        "--context",
        type=Path,
        help="Quality context JSON; defaults to RUN_DIR/quality_context.json.",
    )
    parser.add_argument(
        "--required-stages",
        default="time,depth,pose,dynamic,runtime,map,semantic",
    )
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context_path = args.context or args.run_dir / "quality_context.json"
    if not context_path.is_file():
        raise FileNotFoundError(f"Quality context not found: {context_path}")
    context = json.loads(context_path.read_text())
    required = [value.strip() for value in args.required_stages.split(",") if value.strip()]
    runner = QualityGateRunner(QualityGateConfig.from_yaml(args.config))
    report = runner.evaluate(context, required_stages=required)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = args.run_dir / "quality_report.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["passed"] and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
