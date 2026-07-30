#!/usr/bin/env python3
"""Build an applicable stereo depth-scale report from split LiDAR evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


RAW_POLICY = "default_adaptive_left_right"
SCALED_POLICY = f"candidate_scaled_{RAW_POLICY}"
RAW_NEAR_POLICY = f"{RAW_POLICY}_lidar_reference_0_25_to_5m"
SCALED_NEAR_POLICY = (
    f"candidate_scaled_{RAW_POLICY}_lidar_reference_0_25_to_5m"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Turn disjoint train/holdout LiDAR depth evaluations into a fixed "
            "scale report consumable by apply_g1_floor_calibration.py."
        )
    )
    parser.add_argument("--train-report", required=True, type=Path)
    parser.add_argument("--holdout-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy(report: dict[str, Any], name: str) -> dict[str, float]:
    policies = report["aggregate"]["policies"]
    if name not in policies:
        raise KeyError(f"Required evaluation policy is missing: {name}")
    return policies[name]


def improvements(
    raw: dict[str, float], scaled: dict[str, float]
) -> dict[str, Any]:
    checks = {
        "median_absolute_error_reduced": (
            scaled["median_absolute_error_m"] < raw["median_absolute_error_m"]
        ),
        "p90_absolute_error_reduced": (
            scaled["p90_absolute_error_m"] < raw["p90_absolute_error_m"]
        ),
        "median_absolute_relative_error_reduced": (
            scaled["median_absolute_relative_error"]
            < raw["median_absolute_relative_error"]
        ),
        "within_0_50_m_ratio_increased": (
            scaled["within_0_50_m_ratio"] > raw["within_0_50_m_ratio"]
        ),
        "absolute_median_signed_error_reduced": (
            abs(scaled["median_signed_stereo_minus_lidar_m"])
            < abs(raw["median_signed_stereo_minus_lidar_m"])
        ),
    }
    return {
        "raw": raw,
        "scaled": scaled,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    train_path = args.train_report.expanduser().resolve()
    holdout_path = args.holdout_report.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output_path}")

    train = load_json(train_path)
    holdout = load_json(holdout_path)
    if train["depth_dataset"] != holdout["depth_dataset"]:
        raise ValueError("Train and holdout reports use different depth datasets")
    if train.get("camera_pose_source") != "depth-dataset" or (
        holdout.get("camera_pose_source") != "depth-dataset"
    ):
        raise ValueError("Both evaluations must use rectified depth-dataset poses")
    train_sources = set(map(int, train["source_indices"]))
    holdout_sources = set(map(int, holdout["source_indices"]))
    overlap = sorted(train_sources & holdout_sources)
    if overlap:
        raise ValueError(f"Train/holdout source frames overlap: {overlap}")

    scale = float(
        train["aggregate"]["candidate_scale_stability"]["median"]
    )
    holdout_scale = float(holdout["candidate_depth_scale"])
    if not math.isclose(scale, holdout_scale, rel_tol=1.0e-12):
        raise ValueError(
            f"Holdout evaluated scale {holdout_scale} instead of train scale {scale}"
        )
    source_baseline = float(train["camera"]["baseline"])
    effective_baseline = source_baseline * scale

    full_range = improvements(
        policy(holdout, RAW_POLICY),
        policy(holdout, SCALED_POLICY),
    )
    near_range = improvements(
        policy(holdout, RAW_NEAR_POLICY),
        policy(holdout, SCALED_NEAR_POLICY),
    )
    fold_median = float(
        holdout["aggregate"]["candidate_scale_stability"]["median"]
    )
    fold_relative_difference = abs(fold_median - scale) / scale
    checks = {
        "train_holdout_disjoint": True,
        "minimum_train_frames": len(train_sources) >= 20,
        "minimum_holdout_frames": len(holdout_sources) >= 20,
        "fold_median_relative_difference_le_1pct": (
            fold_relative_difference <= 0.01
        ),
        "holdout_full_range_improved": full_range["passed"],
        "holdout_near_range_improved": near_range["passed"],
    }
    accepted = all(checks.values())
    report = {
        "schema": "daaam.g1_lidar_validated_depth_scale.v1",
        "method": "interleaved_train_holdout_lidar_projection",
        "accepted": accepted,
        "depth_scale": scale,
        "source_baseline_m": source_baseline,
        "effective_baseline_m": effective_baseline,
        "tf_camera_R_image_camera": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "rotation_policy": "identity",
        "depth_dataset": train["depth_dataset"],
        "raw_dataset": train["raw_dataset"],
        "camera_pose_source": "depth-dataset",
        "train": {
            "report": str(train_path),
            "report_sha256": sha256_file(train_path),
            "source_indices": sorted(train_sources),
            "frame_count": len(train_sources),
            "candidate_scale_stability": train["aggregate"][
                "candidate_scale_stability"
            ],
        },
        "holdout": {
            "report": str(holdout_path),
            "report_sha256": sha256_file(holdout_path),
            "source_indices": sorted(holdout_sources),
            "frame_count": len(holdout_sources),
            "independent_candidate_scale_median": fold_median,
            "fold_median_relative_difference": fold_relative_difference,
            "full_range": full_range,
            "lidar_reference_0_25_to_5m": near_range,
        },
        "acceptance_checks": checks,
        "interpretation": (
            "LiDAR was used after stereo inference only. The fixed scale was "
            "estimated on one interleaved frame fold and accepted only after "
            "improving disjoint held-out LiDAR errors."
        ),
    }
    if not accepted:
        raise RuntimeError(
            "LiDAR depth scale failed acceptance checks:\n"
            + json.dumps(checks, indent=2)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
