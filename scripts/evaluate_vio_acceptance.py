#!/usr/bin/env python3
"""Write a no-GT G1/OpenVINS trajectory acceptance report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.slam import (  # noqa: E402
    VioAcceptanceConfig,
    evaluate_vio_acceptance,
    load_odometry_jsonl,
    load_sensor_jsonl,
    write_vio_acceptance_report,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate exact-stamp OpenVINS odometry, loop closure, and "
            "reverse/repeated path consistency without pose ground truth."
        )
    )
    parser.add_argument("--trajectory-jsonl", required=True, type=Path)
    parser.add_argument(
        "--sensor-jsonl",
        type=Path,
        help=(
            "JSONL sensor inventory with sensor_time_ns and stream fields. "
            "Omitting it is a hard failure unless explicitly allowed."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-traversal")
    parser.add_argument("--minimum-duration-s", type=float, default=1800.0)
    parser.add_argument("--minimum-odometry-rate-hz", type=float, default=20.0)
    parser.add_argument("--maximum-loop-translation-m", type=float, default=0.50)
    parser.add_argument("--maximum-loop-rotation-deg", type=float, default=10.0)
    parser.add_argument("--maximum-aligned-p95-m", type=float, default=0.30)
    parser.add_argument("--alignment-sample-count", type=int, default=128)
    parser.add_argument(
        "--required-sensor-streams",
        default="imu,stereo_left,stereo_right",
        help="Comma-separated stream names expected in --sensor-jsonl.",
    )
    parser.add_argument("--allow-missing-sensor-evidence", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_streams = tuple(
        value.strip()
        for value in args.required_sensor_streams.split(",")
        if value.strip()
    )
    config = VioAcceptanceConfig(
        maximum_loop_translation_m=args.maximum_loop_translation_m,
        maximum_loop_rotation_deg=args.maximum_loop_rotation_deg,
        maximum_aligned_p95_m=args.maximum_aligned_p95_m,
        minimum_duration_s=args.minimum_duration_s,
        minimum_odometry_rate_hz=args.minimum_odometry_rate_hz,
        alignment_sample_count=args.alignment_sample_count,
        require_sensor_evidence=not args.allow_missing_sensor_evidence,
        required_sensor_streams=required_streams,
    )
    trajectory = load_odometry_jsonl(args.trajectory_jsonl)
    sensor_events = load_sensor_jsonl(args.sensor_jsonl) if args.sensor_jsonl else None
    report = evaluate_vio_acceptance(
        trajectory,
        sensor_events=sensor_events,
        reference_traversal_id=args.reference_traversal,
        config=config,
    )
    report["provenance"] = {
        "trajectory_jsonl": str(args.trajectory_jsonl.resolve()),
        "trajectory_sha256": _sha256(args.trajectory_jsonl),
        "sensor_jsonl": str(args.sensor_jsonl.resolve()) if args.sensor_jsonl else None,
        "sensor_sha256": _sha256(args.sensor_jsonl) if args.sensor_jsonl else None,
    }
    write_vio_acceptance_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["passed"] and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
