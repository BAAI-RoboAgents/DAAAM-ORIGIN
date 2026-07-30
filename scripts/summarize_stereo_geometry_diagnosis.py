#!/usr/bin/env python3
"""Summarize camera-order, projection, and depth-reconstruction controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--geometry-audit", required=True, type=Path)
    parser.add_argument("--pinhole-report", required=True, type=Path)
    parser.add_argument("--kb-projection-report", required=True, type=Path)
    parser.add_argument("--kb-ray-report", required=True, type=Path)
    parser.add_argument("--reverse-pinhole-report", required=True, type=Path)
    parser.add_argument("--reverse-kb-ray-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def raw_lidar_5m(report: dict[str, Any]) -> dict[str, Any]:
    return report["aggregate"]["policies"][
        "raw_lidar_reference_0_25_to_5m"
    ]


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    dataset = args.dataset.resolve()
    manifest = load_json(dataset / "manifest.json")
    manifest_records = [
        json.loads(line)
        for line in (dataset / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    camera_times = [
        int(image["sensor_time_ns"])
        for record in manifest_records
        for image in record["images"]
    ]
    import yaml

    calibration = yaml.safe_load(
        (
            dataset
            / "calibrations"
            / "000000"
            / "calib_cam0_intrinsics.yaml"
        ).read_text()
    )["intrinsics"]
    calibration_time = int(calibration["header"]["timestamp_ns"])
    geometry = load_json(args.geometry_audit.resolve())
    pinhole = load_json(args.pinhole_report.resolve())
    kb_projection = load_json(args.kb_projection_report.resolve())
    kb_ray = load_json(args.kb_ray_report.resolve())
    reverse_pinhole = load_json(args.reverse_pinhole_report.resolve())
    reverse_kb_ray = load_json(args.reverse_kb_ray_report.resolve())

    controls = {
        "cam0_cam1_pinhole_depth_pinhole_lidar_projection": raw_lidar_5m(
            pinhole
        ),
        "cam0_cam1_pinhole_depth_kb_lidar_projection": raw_lidar_5m(
            kb_projection
        ),
        "cam0_cam1_kb_ray_depth_kb_lidar_projection": raw_lidar_5m(kb_ray),
        "cam1_cam0_pinhole_depth_pinhole_lidar_projection": raw_lidar_5m(
            reverse_pinhole
        ),
        "cam1_cam0_kb_ray_depth_kb_lidar_projection": raw_lidar_5m(
            reverse_kb_ray
        ),
    }
    best_valid_name = min(
        (
            name
            for name, metrics in controls.items()
            if metrics.get("count", 0) > 0
        ),
        key=lambda name: controls[name]["median_absolute_error_m"],
    )
    for metrics in controls.values():
        metrics["semantic_mapping_gate"] = {
            "median_absolute_error_below_0_30m": (
                metrics.get("median_absolute_error_m", float("inf")) < 0.30
            ),
            "within_0_50m_above_0_80": (
                metrics.get("within_0_50_m_ratio", 0.0) > 0.80
            ),
        }
        metrics["semantic_mapping_gate"]["passed"] = all(
            metrics["semantic_mapping_gate"].values()
        )

    report = {
        "dataset": str(dataset),
        "input_contract_evidence": {
            "capture_manifest_calibration_source": manifest.get(
                "calibration_source"
            ),
            "capture_first_camera_timestamp_ns": min(camera_times),
            "capture_last_camera_timestamp_ns": max(camera_times),
            "supplemental_calibration_timestamp_ns": calibration_time,
            "supplemental_calibration_after_capture_end_hours": (
                calibration_time - max(camera_times)
            )
            / 3.6e12,
            "declared_projection_model": calibration["distortion_model"],
            "camera_type": calibration.get("camera_type"),
            "distortion": calibration["D"],
        },
        "camera_only_geometry_audit": geometry["aggregate"],
        "controls": controls,
        "conclusion": {
            "best_valid_control": best_valid_name,
            "best_control_passes_semantic_mapping_gate": controls[
                best_valid_name
            ]["semantic_mapping_gate"]["passed"],
            "reverse_positive_disparity_geometrically_valid_count": controls[
                "cam1_cam0_kb_ray_depth_kb_lidar_projection"
            ].get("count", 0),
            "diagnosis": (
                "The recorded Kannala-Brandt images do not satisfy the pinhole "
                "horizontal-disparity contract. Correct LiDAR projection and "
                "ray triangulation improve the metric but remain far below the "
                "semantic-mapping gate. Reversing the cameras produces no "
                "forward ray intersections under the supplied extrinsic."
            ),
            "required_next_input_or_transform": (
                "Generate a camera-calibration-based virtual pinhole stereo "
                "pair, or provide the exact upstream rectification maps/raw "
                "factory stereo calibration. Do not continue full semantic "
                "mapping from direct unwarped 2d_rect frames."
            ),
        },
    }

    output.mkdir(parents=True)
    (output / "stereo_geometry_diagnosis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    chart_names = [
        "pinhole depth\npinhole LiDAR",
        "pinhole depth\nKB LiDAR",
        "KB-ray depth\nKB LiDAR",
        "reversed\npinhole",
    ]
    chart_keys = list(controls)[:4]
    median_errors = [
        controls[name]["median_absolute_error_m"] for name in chart_keys
    ]
    within_half = [
        controls[name]["within_0_50_m_ratio"] for name in chart_keys
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].bar(chart_names, median_errors, color="#3182bd")
    axes[0].axhline(0.30, color="#d7301f", linestyle="--", label="0.30 m gate")
    axes[0].set_ylabel("metres")
    axes[0].set_title("LiDAR-reference ≤5 m median absolute error")
    axes[0].legend()
    axes[1].bar(chart_names, within_half, color="#756bb1")
    axes[1].axhline(0.80, color="#d7301f", linestyle="--", label="80% gate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Absolute error ≤0.5 m")
    axes[1].legend()
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "stereo_geometry_diagnosis.png", dpi=170)
    plt.close(figure)
    print(json.dumps(report["conclusion"], indent=2))


if __name__ == "__main__":
    main()
