#!/usr/bin/env python3
"""Summarize currently available G1 input/keyframe/calibration evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sync_paths = {
        "2 ms": workspace
        / "shared_inputs/input_sync_audits/sync_2ms/prepared_stereo_geometry_audit.json",
        "5 ms": workspace
        / "shared_inputs/input_sync_audits/sync_5ms/prepared_stereo_geometry_audit.json",
        "10 ms": workspace
        / "shared_inputs/prepared_stereo_geometry_audit_visual/prepared_stereo_geometry_audit.json",
    }
    sync = {
        name: load_json(path)["aggregate"] for name, path in sync_paths.items()
    }
    keyframe_paths = {
        "all": workspace / "shared_inputs/offline_prepared/01_pinhole/tick_index.json",
        "dense": workspace
        / "shared_inputs/keyframe_ablations/dense/keyframe_selection_report.json",
        "default": workspace
        / "shared_inputs/offline_prepared/02_selected/keyframe_selection_report.json",
        "sparse": workspace
        / "shared_inputs/keyframe_ablations/sparse/keyframe_selection_report.json",
    }
    keyframes = {}
    for name, path in keyframe_paths.items():
        data = load_json(path)
        keyframes[name] = (
            len(data["frames"])
            if name == "all"
            else int(data["selected_frame_count"])
        )
    camera_order = load_json(
        workspace
        / "shared_inputs/camera_order_feature_audit/camera_order_feature_audit.json"
    )
    rectification = load_json(
        workspace
        / (
            "shared_inputs/right_rectification_lidar_815_915/"
            "lidar_guided_rectification_report.json"
        )
    )
    lidar_manifest = load_json(
        workspace / "ground_truth/lidar_camera_815_915/manifest.json"
    )
    depth_report_path = (
        workspace
        / "shared_inputs/depth_lidar_diagnostic/lidar_batch_evaluation.json"
    )
    depth_report = (
        load_json(depth_report_path) if depth_report_path.is_file() else None
    )
    floor_report_path = (
        workspace / "shared_inputs/floor_calibration_diagnostic.json"
    )
    floor_report = (
        load_json(floor_report_path) if floor_report_path.is_file() else None
    )
    projection_audit_paths = {
        "LiDAR right homography": workspace
        / "shared_inputs/prepared_stereo_geometry_audit_visual/prepared_stereo_geometry_audit.json",
        "visual right-only": workspace
        / "shared_inputs/projection_model_diagnostics/pinhole_visual_only_audit/prepared_stereo_geometry_audit.json",
        "full pinhole stereo": workspace
        / "shared_inputs/projection_model_diagnostics/pinhole_full_stereo_rectify_audit/prepared_stereo_geometry_audit.json",
        "KB virtual pinhole": workspace
        / "shared_inputs/projection_model_diagnostics/kannala_brandt_virtual_audit/prepared_stereo_geometry_audit.json",
    }
    projection_audits = {
        name: load_json(path)["aggregate"]
        for name, path in projection_audit_paths.items()
        if path.is_file()
    }
    visual_depth_report_path = (
        workspace
        / (
            "shared_inputs/projection_model_diagnostics/"
            "pinhole_visual_only_depth_lidar/lidar_batch_evaluation.json"
        )
    )
    visual_depth_report = (
        load_json(visual_depth_report_path)
        if visual_depth_report_path.is_file()
        else None
    )
    summary = {
        "schema": "daaam.g1_available_experiment_summary.v1",
        "sync_audits": sync,
        "keyframe_selected_counts": keyframes,
        "camera_order": {
            "forward_positive_disparity_median": camera_order[
                "forward_positive_disparity_ratio"
            ]["median"],
            "reverse_positive_disparity_median": camera_order[
                "reverse_positive_disparity_ratio"
            ]["median"],
            "forward_preferred_frames": camera_order["forward_preferred_frames"],
            "frames": camera_order["frames"],
        },
        "right_rectification_holdout": {
            "source_index": rectification["holdout_source_index"],
            "identity_error": rectification["holdout"]["identity_target_error"],
            "corrected_error": rectification["holdout"]["corrected_target_error"],
        },
        "lidar_ground_truth_frames": lidar_manifest["frame_count"],
        "depth_lidar_diagnostic": (
            depth_report["aggregate"] if depth_report is not None else None
        ),
        "floor_calibration_diagnostic": floor_report,
        "projection_model_diagnostics": projection_audits,
        "visual_right_only_depth_lidar_diagnostic": (
            visual_depth_report["aggregate"]
            if visual_depth_report is not None
            else None
        ),
    }
    (output / "available_experiment_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    figure, axes = plt.subplots(4, 2, figsize=(15, 20), constrained_layout=True)
    sync_names = list(sync)
    evaluated = [sync[name]["evaluated_frame_count"] for name in sync_names]
    axes[0, 0].bar(sync_names, evaluated, color="#3182bd")
    axes[0, 0].set_title("Frames admitted by stereo synchronization gate")
    axes[0, 0].set_ylabel("frames")
    keyframe_names = list(keyframes)
    axes[0, 1].bar(
        keyframe_names,
        [keyframes[name] for name in keyframe_names],
        color="#31a354",
    )
    axes[0, 1].set_title("Frames retained by keyframe policy")
    axes[0, 1].set_ylabel("frames")
    axes[1, 0].bar(
        ["cam0->cam1", "cam1->cam0"],
        [
            summary["camera_order"]["forward_positive_disparity_median"],
            summary["camera_order"]["reverse_positive_disparity_median"],
        ],
        color=["#31a354", "#de2d26"],
    )
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_title("Median positive-disparity ratio")
    holdout = rectification["holdout"]
    axes[1, 1].bar(
        ["identity", "LiDAR-corrected"],
        [
            holdout["identity_target_error"]["euclidean_error_px"]["p50"],
            holdout["corrected_target_error"]["euclidean_error_px"]["p50"],
        ],
        color=["#de2d26", "#31a354"],
    )
    axes[1, 1].set_title("Held-out LiDAR target median error")
    axes[1, 1].set_ylabel("pixels")
    if depth_report is not None:
        policies = depth_report["aggregate"]["policies"]
        policy_names = [
            "raw_0_25_to_30m",
            "default_adaptive_left_right",
            "lr_error_le_0.5px",
        ]
        axes[2, 0].bar(
            ["raw", "adaptive LR", "LR <= 0.5px"],
            [
                policies[name]["median_absolute_error_m"]
                for name in policy_names
            ],
            color=["#969696", "#3182bd", "#756bb1"],
        )
        axes[2, 0].set_title("Stereo depth median absolute error vs LiDAR")
        axes[2, 0].set_ylabel("meters")
        scales = depth_report["aggregate"][
            "candidate_scale_reference_over_stereo_per_frame"
        ]
        axes[2, 1].plot(scales, color="#de2d26", linewidth=1.5)
        axes[2, 1].axhline(1.0, color="#31a354", linestyle="--")
        axes[2, 1].set_title("Per-frame LiDAR/stereo scale candidate")
        axes[2, 1].set_xlabel("selected frame order")
        if floor_report is not None:
            axes[2, 1].text(
                0.02,
                0.03,
                (
                    "floor calibration: "
                    f"{floor_report.get('status', 'unknown')} / "
                    f"{floor_report.get('reason', 'no reason')}"
                ),
                transform=axes[2, 1].transAxes,
                fontsize=9,
                color="#de2d26",
            )
    elif floor_report is not None:
        axes[2, 0].bar(
            ["accepted", "rejected"],
            [
                floor_report.get("accepted_frame_count", 0),
                floor_report.get("rejected_frame_count", 0),
            ],
            color=["#31a354", "#de2d26"],
        )
        axes[2, 0].set_title("Floor calibration diagnostic")
        axes[2, 1].text(
            0.05,
            0.5,
            floor_report.get("reason", "no depth/LiDAR report"),
            transform=axes[2, 1].transAxes,
        )
    if projection_audits:
        names = list(projection_audits)
        x = np.arange(len(names))
        axes[3, 0].bar(
            x - 0.2,
            [
                projection_audits[name][
                    "median_of_frame_median_absolute_vertical_error_px"
                ]
                for name in names
            ],
            width=0.4,
            label="median frame p50",
            color="#3182bd",
        )
        axes[3, 0].bar(
            x + 0.2,
            [
                projection_audits[name][
                    "median_of_frame_p95_absolute_vertical_error_px"
                ]
                for name in names
            ],
            width=0.4,
            label="median frame p95",
            color="#756bb1",
        )
        axes[3, 0].set_xticks(x, names, rotation=18, ha="right")
        axes[3, 0].set_ylabel("vertical error (pixels)")
        axes[3, 0].set_title("Projection/rectification geometry diagnostics")
        axes[3, 0].legend()
    if depth_report is not None and visual_depth_report is not None:
        axes[3, 1].bar(
            ["LiDAR right homography", "visual right-only"],
            [
                depth_report["aggregate"]["policies"][
                    "default_adaptive_left_right"
                ]["median_absolute_error_m"],
                visual_depth_report["aggregate"]["policies"][
                    "default_adaptive_left_right"
                ]["median_absolute_error_m"],
            ],
            color=["#31a354", "#3182bd"],
        )
        axes[3, 1].set_ylabel("median absolute error (m)")
        axes[3, 1].set_title("Depth accuracy trade-off vs LiDAR")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "available_experiment_summary.png", dpi=180)
    plt.close(figure)
    print(
        json.dumps(
            {
                "summary": str(output / "available_experiment_summary.json"),
                "visualization": str(output / "available_experiment_summary.png"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
