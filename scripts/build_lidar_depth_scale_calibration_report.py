#!/usr/bin/env python3
"""Build a fixed depth-scale report from disjoint LiDAR train/holdout audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "daaam.g1_lidar_depth_scale_calibration.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-report", required=True, type=Path)
    parser.add_argument("--holdout-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-baseline-m", required=True, type=float)
    parser.add_argument("--maximum-train-relative-span", type=float, default=0.25)
    parser.add_argument(
        "--maximum-holdout-median-absolute-relative-error",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--maximum-holdout-absolute-median-signed-error-m",
        type=float,
        default=0.10,
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lidar_reference_policy(report: dict[str, Any], *, scaled: bool) -> dict[str, Any]:
    base = f"{report['filtered_policy_name']}_lidar_reference_0_25_to_5m"
    name = f"candidate_scaled_{base}" if scaled else base
    return report["aggregate"]["policies"][name]


def main() -> None:
    args = parse_args()
    train_path = args.train_report.expanduser().resolve()
    holdout_path = args.holdout_report.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing report: {output}")
    train = load(train_path)
    holdout = load(holdout_path)

    train_scale = float(
        train["aggregate"]["candidate_scale_stability"]["median"]
    )
    requested_scale = float(holdout["candidate_depth_scale"])
    if not math.isclose(train_scale, requested_scale, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "Holdout candidate scale does not equal the train-only estimate: "
            f"{requested_scale} vs {train_scale}"
        )
    train_sources = {int(value) for value in train["source_indices"]}
    holdout_sources = {int(value) for value in holdout["source_indices"]}
    overlap = sorted(train_sources & holdout_sources)
    if overlap:
        raise ValueError(f"Train and holdout source frames overlap: {overlap[:10]}")
    if train["reference_camera"] != "cam0" or holdout["reference_camera"] != "cam0":
        raise ValueError("Depth scale calibration requires cam0 as the reference camera")
    for report in (train, holdout):
        if report["camera_pose_source"] != "depth-dataset":
            raise ValueError("LiDAR projection must use the rectified dataset pose")
        if report["lidar_projection_model"] != "pinhole":
            raise ValueError("Rectified LiDAR validation must use pinhole projection")

    train_stability = train["aggregate"]["candidate_scale_stability"]
    raw_holdout = lidar_reference_policy(holdout, scaled=False)
    scaled_holdout = lidar_reference_policy(holdout, scaled=True)
    thresholds = {
        "maximum_train_relative_span": args.maximum_train_relative_span,
        "maximum_holdout_median_absolute_relative_error": (
            args.maximum_holdout_median_absolute_relative_error
        ),
        "maximum_holdout_absolute_median_signed_error_m": (
            args.maximum_holdout_absolute_median_signed_error_m
        ),
        "require_holdout_median_absolute_relative_error_improvement": True,
        "require_disjoint_source_frames": True,
    }
    checks = {
        "train_scale_stable": (
            float(train_stability["relative_span"])
            <= args.maximum_train_relative_span
        ),
        "holdout_median_absolute_relative_error": (
            float(scaled_holdout["median_absolute_relative_error"])
            <= args.maximum_holdout_median_absolute_relative_error
        ),
        "holdout_median_signed_error": (
            abs(float(scaled_holdout["median_signed_stereo_minus_lidar_m"]))
            <= args.maximum_holdout_absolute_median_signed_error_m
        ),
        "holdout_improved": (
            float(scaled_holdout["median_absolute_relative_error"])
            < float(raw_holdout["median_absolute_relative_error"])
        ),
        "source_frames_disjoint": not overlap,
    }
    passed = all(checks.values())
    source_baseline = float(args.source_baseline_m)
    if source_baseline <= 0.0 or not 0.0 < train_scale <= 2.0:
        raise ValueError("Baseline or fitted depth scale is invalid")

    report = {
        "schema": SCHEMA,
        "method": "interleaved_train_holdout_lidar_depth_scale_identity_rotation",
        "status": "accepted" if passed else "rejected",
        "depth_scale": train_scale,
        "source_baseline_m": source_baseline,
        "effective_baseline_m": source_baseline * train_scale,
        "tf_camera_R_image_camera": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "rotation_policy": "identity",
        "source_camera_rotations_preserved": True,
        "train": {
            "report": str(train_path),
            "sha256": sha256(train_path),
            "selection_fold": train["selection_fold"],
            "frame_count": int(train["evaluated_frame_count"]),
            "source_indices": sorted(train_sources),
            "candidate_scale_stability": train_stability,
        },
        "holdout": {
            "report": str(holdout_path),
            "sha256": sha256(holdout_path),
            "selection_fold": holdout["selection_fold"],
            "frame_count": int(holdout["evaluated_frame_count"]),
            "source_indices": sorted(holdout_sources),
            "raw_lidar_reference_0_25_to_5m": raw_holdout,
            "scaled_lidar_reference_0_25_to_5m": scaled_holdout,
        },
        "acceptance": {
            "passed": passed,
            "checks": checks,
            "thresholds": thresholds,
        },
        "interpretation_guard": (
            "The scale was estimated only on train-fold LiDAR and accepted only "
            "after improvement on disjoint holdout frames. The report applies no "
            "additional image-frame rotation."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("LiDAR depth-scale acceptance failed")


if __name__ == "__main__":
    main()
