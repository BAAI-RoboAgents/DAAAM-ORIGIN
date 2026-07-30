#!/usr/bin/env python3
"""Compare two camera-order controls using saved stereo and LiDAR reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


POLICIES = (
    "raw_0_25_to_30m",
    "raw_lidar_reference_0_25_to_5m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-report", required=True, type=Path)
    parser.add_argument("--reverse-report", required=True, type=Path)
    parser.add_argument("--forward-run", required=True, type=Path)
    parser.add_argument("--reverse-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def run_summary(run: dict[str, Any]) -> dict[str, Any]:
    frames = run["frame_stats"]
    return {
        "mean_raw_positive_disparity_ratio": float(
            np.mean([item["raw_positive_disparity_ratio"] for item in frames])
        ),
        "mean_raw_depth_within_5m_ratio": float(
            np.mean(
                [
                    item["raw_depth_coverage_ratio"]["within_5m"]
                    for item in frames
                ]
            )
        ),
        "mean_raw_depth_within_30m_ratio": float(
            np.mean(
                [
                    item["raw_depth_coverage_ratio"]["within_30m"]
                    for item in frames
                ]
            )
        ),
        "mean_filtered_valid_ratio": float(
            np.mean([item["valid_ratio"] for item in frames])
        ),
        "mean_left_right_consistency": float(
            np.mean([item["left_right_consistency"] for item in frames])
        ),
    }


def lidar_summary(report: dict[str, Any]) -> dict[str, Any]:
    policies = report["aggregate"]["policies"]
    return {name: policies[name] for name in POLICIES}


def metric(
    summary: dict[str, Any], policy: str, name: str
) -> float:
    return float(summary["lidar"][policy][name])


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")

    forward_report = load_json(args.forward_report.resolve())
    reverse_report = load_json(args.reverse_report.resolve())
    forward_run = load_json(args.forward_run.resolve())
    reverse_run = load_json(args.reverse_run.resolve())
    if forward_report["source_indices"] != reverse_report["source_indices"]:
        raise ValueError("Camera-order reports do not contain identical source frames")
    forward_settings = forward_run["settings"].copy()
    reverse_settings = reverse_run["settings"].copy()
    if forward_settings != reverse_settings:
        raise ValueError("Camera-order inference settings differ")

    groups = {
        "cam0_to_cam1": {
            "reference_camera": forward_report["reference_camera"],
            "run": run_summary(forward_run),
            "lidar": lidar_summary(forward_report),
        },
        "cam1_to_cam0": {
            "reference_camera": reverse_report["reference_camera"],
            "run": run_summary(reverse_run),
            "lidar": lidar_summary(reverse_report),
        },
    }
    for name, group in groups.items():
        raw = group["lidar"]["raw_lidar_reference_0_25_to_5m"]
        group["semantic_mapping_gate"] = {
            "median_absolute_error_below_0_30m": (
                raw["median_absolute_error_m"] < 0.30
            ),
            "within_0_50m_above_0_80": raw["within_0_50_m_ratio"] > 0.80,
        }
        group["semantic_mapping_gate"]["passed"] = all(
            group["semantic_mapping_gate"].values()
        )

    preferred_by_lidar = min(
        groups,
        key=lambda name: metric(
            groups[name],
            "raw_lidar_reference_0_25_to_5m",
            "median_absolute_error_m",
        ),
    )
    report = {
        "contract": (
            "Both controls use the same original rectified pixels and inference "
            "settings. LiDAR is used only after inference. cam1 evaluation uses "
            "base_T_cam1 = base_T_cam0 @ T_cam0_cam1."
        ),
        "source_indices": forward_report["source_indices"],
        "inference_settings": forward_settings,
        "groups": groups,
        "comparison": {
            "lower_lidar_median_error": preferred_by_lidar,
            "acceptable_for_semantic_mapping": [
                name
                for name, group in groups.items()
                if group["semantic_mapping_gate"]["passed"]
            ],
            "conclusion": (
                "Neither camera order passes the semantic-mapping accuracy gate. "
                "Camera order alone does not explain the metric-depth failure."
            ),
        },
    }

    output.mkdir(parents=True)
    (output / "camera_order_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )

    labels = ["cam0→cam1", "cam1→cam0"]
    summaries = [groups["cam0_to_cam1"], groups["cam1_to_cam0"]]
    median_error = [
        metric(
            summary,
            "raw_lidar_reference_0_25_to_5m",
            "median_absolute_error_m",
        )
        for summary in summaries
    ]
    within_half = [
        metric(
            summary,
            "raw_lidar_reference_0_25_to_5m",
            "within_0_50_m_ratio",
        )
        for summary in summaries
    ]
    image_coverage = [
        summary["run"]["mean_raw_depth_within_30m_ratio"]
        for summary in summaries
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    axes[0].bar(labels, median_error, color=("#3182bd", "#756bb1"))
    axes[0].axhline(0.30, color="#d7301f", linestyle="--", label="0.30 m gate")
    axes[0].set_title("LiDAR-reference ≤5 m\nmedian absolute error")
    axes[0].set_ylabel("metres")
    axes[0].legend()
    axes[1].bar(labels, within_half, color=("#3182bd", "#756bb1"))
    axes[1].axhline(0.80, color="#d7301f", linestyle="--", label="80% gate")
    axes[1].set_title("Absolute error ≤0.5 m")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[2].bar(labels, image_coverage, color=("#3182bd", "#756bb1"))
    axes[2].set_title("Raw image depth coverage ≤30 m")
    axes[2].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "camera_order_comparison.png", dpi=170)
    plt.close(figure)
    print(json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()
