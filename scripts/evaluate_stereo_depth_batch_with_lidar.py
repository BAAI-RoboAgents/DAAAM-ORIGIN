#!/usr/bin/env python3
"""Independently evaluate saved raw stereo depth against recorded LiDAR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from fuse_lidar_camera_frame import (
    auxiliary_pose_samples,
    colorize_depth,
    draw_projected_points,
    error_summary,
    interpolate_pose,
    load_json,
    load_jsonl,
    map_pose_samples,
    transform_points,
    z_buffer_projection,
)


RANGE_BINS_M = ((0.25, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 30.0))
CONSISTENCY_THRESHOLDS_PX = (0.5, 1.0, 2.0, 3.0, 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", required=True, type=Path)
    parser.add_argument("--depth-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--lidar-projection-model",
        choices=("pinhole", "kannala_brandt"),
        default="pinhole",
        help="Projection used only to place independent LiDAR samples on RGB.",
    )
    parser.add_argument(
        "--camera-pose-source",
        choices=("auto", "raw", "depth-dataset"),
        default="auto",
        help=(
            "Pose used for LiDAR projection. auto prefers the depth dataset's "
            "world_T_camera poses, which is required when stereo rectification "
            "rotates the reference camera frame."
        ),
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=None,
        help="Uniformly evaluate this many depth frames instead of every frame.",
    )
    parser.add_argument(
        "--source-frame-start",
        type=int,
        default=None,
        help="Optional inclusive raw source tick lower bound.",
    )
    parser.add_argument(
        "--source-frame-end",
        type=int,
        default=None,
        help="Optional inclusive raw source tick upper bound.",
    )
    parser.add_argument(
        "--selection-fold-count",
        type=int,
        default=1,
        help="Split the time-ordered frames into interleaved folds.",
    )
    parser.add_argument(
        "--selection-fold-index",
        type=int,
        default=0,
        help="Zero-based interleaved fold to evaluate.",
    )
    parser.add_argument(
        "--candidate-depth-scale",
        type=float,
        default=None,
        help="Also evaluate a fixed metric depth scale without modifying input.",
    )
    parser.add_argument("--minimum-depth-m", type=float, default=0.25)
    parser.add_argument("--maximum-depth-m", type=float, default=30.0)
    return parser.parse_args()


def append_policy(
    storage: dict[str, dict[str, list[np.ndarray]]],
    name: str,
    prediction: np.ndarray,
    reference: np.ndarray,
    selected: np.ndarray,
) -> None:
    policy = storage.setdefault(name, {"prediction": [], "reference": []})
    policy["prediction"].append(prediction[selected].astype(np.float64))
    policy["reference"].append(reference[selected].astype(np.float64))


def concatenate(values: list[np.ndarray]) -> np.ndarray:
    nonempty = [value for value in values if value.size]
    return np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float64)


def z_buffer_projection_kannala_brandt(
    camera_points: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: np.ndarray,
    width: int,
    height: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> dict[str, np.ndarray]:
    coefficients = np.zeros(4, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if distortion.size > 4:
        raise ValueError("Kannala-Brandt projection supports at most 4 coefficients")
    coefficients[: distortion.size] = distortion
    x = camera_points[:, 0]
    y = camera_points[:, 1]
    z = camera_points[:, 2]
    radius = np.hypot(x, y)
    theta = np.arctan2(radius, z)
    theta2 = theta * theta
    theta_distorted = theta * (
        1.0
        + coefficients[0] * theta2
        + coefficients[1] * theta2**2
        + coefficients[2] * theta2**3
        + coefficients[3] * theta2**4
    )
    scale = np.divide(
        theta_distorted,
        radius,
        out=np.ones_like(theta_distorted),
        where=radius > 1.0e-12,
    )
    u = fx * x * scale + cx
    v = fy * y * scale + cy
    inside = (
        np.isfinite(camera_points).all(axis=1)
        & (z >= minimum_depth_m)
        & (z <= maximum_depth_m)
        & (u >= 0.0)
        & (u <= width - 1)
        & (v >= 0.0)
        & (v <= height - 1)
    )
    source_indices = np.flatnonzero(inside)
    u_pixel = np.rint(u[inside]).astype(np.int32)
    v_pixel = np.rint(v[inside]).astype(np.int32)
    depth = z[inside]
    flat = v_pixel.astype(np.int64) * width + u_pixel
    depth_order = np.argsort(depth)
    _, first = np.unique(flat[depth_order], return_index=True)
    selected = depth_order[first]
    return {
        "inside_source_indices": source_indices,
        "source_indices": source_indices[selected],
        "u": u_pixel[selected],
        "v": v_pixel[selected],
        "depth": depth[selected],
    }


def summarize_policy(policy: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    prediction = concatenate(policy["prediction"])
    reference = concatenate(policy["reference"])
    return error_summary(prediction, reference)


def save_frame_overlay(
    path: Path,
    rgb: np.ndarray,
    projection: dict[str, np.ndarray],
    prediction: np.ndarray,
    valid: np.ndarray,
    source_index: int,
    reference_camera: str,
) -> None:
    u = projection["u"][valid]
    v = projection["v"][valid]
    reference = projection["depth"][valid]
    residual = prediction[valid] - reference
    if residual.size:
        residual_encoded = np.rint(
            np.clip((residual + 2.0) / 4.0, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        residual_colors = cv2.applyColorMap(
            residual_encoded[:, None], cv2.COLORMAP_COOL
        )[:, 0, :]
        residual_overlay = draw_projected_points(
            rgb, u, v, residual_colors, radius=2
        )
    else:
        residual_overlay = rgb.copy()
    lidar_colors = colorize_depth(
        projection["depth"], 0.25, 10.0
    )
    lidar_overlay = draw_projected_points(
        rgb, projection["u"], projection["v"], lidar_colors, radius=2
    )
    figure, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    panels = (
        (
            rgb,
            f"Original {reference_camera} reference RGB — source {source_index:06d}",
        ),
        (lidar_overlay, "Independent LiDAR projection"),
        (
            residual_overlay,
            f"Raw stereo − LiDAR residual (±2 m)\n{int(valid.sum())} common pixels",
        ),
    )
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_summary_chart(path: Path, frames: list[dict[str, Any]]) -> None:
    source_indices = [item["source_index"] for item in frames]
    coverage = [item["raw_0_25_to_30m"]["coverage_ratio"] for item in frames]
    median_error = [
        item["raw_0_25_to_30m"]["depth_error"].get(
            "median_absolute_error_m", np.nan
        )
        for item in frames
    ]
    within_half = [
        item["raw_0_25_to_30m"]["depth_error"].get(
            "within_0_50_m_ratio", np.nan
        )
        for item in frames
    ]
    x = np.arange(len(frames))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].bar(x, coverage, color="#3182bd")
    axes[0].set_title("LiDAR pixels with raw stereo depth")
    axes[0].set_ylabel("coverage ratio")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].bar(x, median_error, color="#e6550d")
    axes[1].set_title("Median absolute depth error")
    axes[1].set_ylabel("metres")
    axes[2].bar(x, within_half, color="#31a354")
    axes[2].set_title("Absolute error ≤ 0.5 m")
    axes[2].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.set_xticks(x, [str(index) for index in source_indices], rotation=30)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_depth_gallery(
    path: Path,
    depth_dataset: Path,
    raw_product_root: Path,
    frames: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(
        len(frames),
        1,
        figsize=(14, 4.8 * len(frames)),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    for axis, frame in zip(axes_array, frames):
        output_index = int(frame["idx"])
        source_index = int(frame["source_idx"])
        overlay = cv2.imread(
            str(
                raw_product_root
                / "raw_depth_overlay_5m"
                / f"{output_index:08d}.png"
            ),
            cv2.IMREAD_COLOR,
        )
        if overlay is None:
            raise FileNotFoundError(
                f"Missing saved depth visualization for source {source_index}"
            )
        axis.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axis.set_title(
            f"source {source_index:06d} — raw Fast-FoundationStereo depth "
            "overlay, 0.25–5.0 m"
        )
        axis.axis("off")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    raw_dataset = args.raw_dataset.resolve()
    depth_dataset = args.depth_dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if not 0.0 < args.minimum_depth_m < args.maximum_depth_m:
        raise ValueError("Depth evaluation range is invalid")
    output.mkdir(parents=True)
    overlay_dir = output / "frame_overlays"
    overlay_dir.mkdir()
    correspondence_dir = output / "correspondences"
    correspondence_dir.mkdir()

    manifest = {
        int(record["tick"]): record
        for record in load_jsonl(raw_dataset / "manifest.jsonl")
    }
    tick_index = load_json(depth_dataset / "tick_index.json")
    tick_frames = list(tick_index["frames"])
    if not tick_frames:
        raise ValueError("Depth dataset tick_index contains no frames")
    if (
        args.source_frame_start is not None
        and args.source_frame_end is not None
        and args.source_frame_start > args.source_frame_end
    ):
        raise ValueError("--source-frame-start must be <= --source-frame-end")
    if args.source_frame_start is not None or args.source_frame_end is not None:
        lower = (
            -np.inf
            if args.source_frame_start is None
            else args.source_frame_start
        )
        upper = (
            np.inf if args.source_frame_end is None else args.source_frame_end
        )
        tick_frames = [
            frame
            for frame in tick_frames
            if lower <= int(frame["source_idx"]) <= upper
        ]
        if not tick_frames:
            raise ValueError(
                "No depth frames overlap the requested raw source range"
            )
    if args.selection_fold_count <= 0:
        raise ValueError("--selection-fold-count must be positive")
    if not 0 <= args.selection_fold_index < args.selection_fold_count:
        raise ValueError(
            "--selection-fold-index must be in [0, --selection-fold-count)"
        )
    tick_frames = tick_frames[
        args.selection_fold_index :: args.selection_fold_count
    ]
    if not tick_frames:
        raise ValueError("Selected evaluation fold contains no frames")
    if args.frame_count is not None:
        if args.frame_count <= 0:
            raise ValueError("--frame-count must be positive")
        selected_indices = np.linspace(
            0,
            len(tick_frames) - 1,
            num=min(args.frame_count, len(tick_frames)),
            dtype=np.int64,
        )
        tick_frames = [tick_frames[int(index)] for index in np.unique(selected_indices)]
    camera = load_json(depth_dataset / "camera_info.json")
    if args.candidate_depth_scale is not None and args.candidate_depth_scale <= 0.0:
        raise ValueError("--candidate-depth-scale must be positive")
    reference_camera = str(
        tick_index.get(
            "reference_camera", camera.get("reference_camera", "cam0")
        )
    )
    if reference_camera not in {"cam0", "cam1"}:
        raise ValueError(f"Unsupported reference camera: {reference_camera}")
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    width = int(camera["width"])
    height = int(camera["height"])

    map_samples = map_pose_samples(
        load_jsonl(raw_dataset / "state" / "000000" / "map_pose.jsonl")
    )
    auxiliary_records = load_jsonl(
        raw_dataset / "poses" / "dense_global" / "000000" / "aux_poses.jsonl"
    )
    camera_samples = auxiliary_pose_samples(auxiliary_records, "head_camera")
    lidar_samples = auxiliary_pose_samples(auxiliary_records, "lidar")
    depth_pose_path = depth_dataset / "pose" / "poses.txt"
    use_depth_dataset_pose = args.camera_pose_source == "depth-dataset" or (
        args.camera_pose_source == "auto" and depth_pose_path.is_file()
    )
    depth_poses = None
    if use_depth_dataset_pose:
        if not depth_pose_path.is_file():
            raise FileNotFoundError(
                f"Depth-dataset camera poses not found: {depth_pose_path}"
            )
        depth_poses = np.loadtxt(depth_pose_path, dtype=np.float64).reshape(-1, 4, 4)
        maximum_pose_row = max(int(frame["pose_row"]) for frame in tick_frames)
        if maximum_pose_row >= len(depth_poses):
            raise ValueError(
                "Depth-dataset pose file does not cover every selected tick: "
                f"max pose_row={maximum_pose_row}, poses={len(depth_poses)}"
            )
    resolved_camera_pose_source = (
        "depth-dataset" if use_depth_dataset_pose else "raw"
    )
    cam0_to_cam1_record = yaml.safe_load(
        (
            raw_dataset
            / "calibrations"
            / "000000"
            / "calib_cam0_to_cam1.yaml"
        ).read_text()
    )
    cam0_t_cam1 = np.asarray(
        cam0_to_cam1_record["transform"]["matrix_4x4"],
        dtype=np.float64,
    )
    if cam0_t_cam1.shape != (4, 4):
        raise ValueError("T_cam0_cam1 must be a 4x4 matrix")
    reference_calibration_record = yaml.safe_load(
        (
            raw_dataset
            / "calibrations"
            / "000000"
            / f"calib_{reference_camera}_intrinsics.yaml"
        ).read_text()
    )["intrinsics"]
    source_projection_model = str(
        reference_calibration_record["distortion_model"]
    )
    source_distortion = np.asarray(
        reference_calibration_record["D"], dtype=np.float64
    )
    if (
        args.lidar_projection_model == "kannala_brandt"
        and source_projection_model != "kannala_brandt"
    ):
        raise ValueError(
            "Requested Kannala-Brandt projection but source calibration "
            f"declares {source_projection_model!r}"
        )
    depth_run_path = depth_dataset / "fast_foundation_stereo_run.json"
    depth_run = load_json(depth_run_path) if depth_run_path.is_file() else {}
    raw_product_root = depth_dataset
    if not (raw_product_root / "raw_depth_meter").is_dir():
        recorded_output = depth_run.get("output")
        if recorded_output is None:
            raise FileNotFoundError(
                f"No raw_depth_meter in {depth_dataset} and depth run has no output"
            )
        raw_product_root = Path(str(recorded_output)).resolve()
        if not (raw_product_root / "raw_depth_meter").is_dir():
            raise FileNotFoundError(
                "Recorded FoundationStereo raw product root is unavailable: "
                f"{raw_product_root}"
            )
    confidence_mode = str(
        (depth_run.get("settings") or {}).get("confidence_mode", "unknown")
    )
    has_left_right_consistency = (
        depth_dataset / "lr_consistency_error"
    ).is_dir()
    filtered_policy_name = (
        "default_adaptive_left_right"
        if has_left_right_consistency
        else f"default_{confidence_mode.replace('-', '_')}"
    )

    policy_values: dict[str, dict[str, list[np.ndarray]]] = {}
    ratio_by_frame = []
    frame_reports = []
    for frame in tick_frames:
        output_index = int(frame["idx"])
        source_index = int(frame["source_idx"])
        record = manifest[source_index]
        reference_descriptor = next(
            image
            for image in record["images"]
            if image["camera"] == reference_camera
        )
        lidar_descriptor = next(
            lidar for lidar in record["lidar"] if lidar["lidar"] == "lidar0"
        )
        camera_time = int(reference_descriptor["sensor_time_ns"])
        lidar_time = int(lidar_descriptor["sensor_time_ns"])
        map_t_base_lidar, map_lidar_clamped = interpolate_pose(
            *map_samples, lidar_time
        )
        base_t_lidar, lidar_clamped = interpolate_pose(*lidar_samples, lidar_time)
        if use_depth_dataset_pose:
            assert depth_poses is not None
            map_t_camera = depth_poses[int(frame["pose_row"])]
            camera_pose_clamped = False
        else:
            map_t_base_camera, map_camera_clamped = interpolate_pose(
                *map_samples, camera_time
            )
            base_t_cam0, camera_clamped = interpolate_pose(
                *camera_samples, camera_time
            )
            base_t_camera = (
                base_t_cam0
                if reference_camera == "cam0"
                else base_t_cam0 @ cam0_t_cam1
            )
            map_t_camera = map_t_base_camera @ base_t_camera
            camera_pose_clamped = map_camera_clamped or camera_clamped
        if any((camera_pose_clamped, map_lidar_clamped, lidar_clamped)):
            raise ValueError(f"Pose interpolation clamped for source {source_index}")
        camera_t_lidar = np.linalg.inv(map_t_camera) @ (
            map_t_base_lidar @ base_t_lidar
        )

        lidar_points = np.load(
            raw_dataset / lidar_descriptor["path"], allow_pickle=False
        ).astype(np.float64)
        lidar_points = lidar_points[
            np.isfinite(lidar_points).all(axis=1)
            & (np.linalg.norm(lidar_points, axis=1) > 0.01)
        ]
        points_camera = transform_points(camera_t_lidar, lidar_points)
        if args.lidar_projection_model == "pinhole":
            projection = z_buffer_projection(
                points_camera,
                fx,
                fy,
                cx,
                cy,
                width,
                height,
                args.minimum_depth_m,
                args.maximum_depth_m,
            )
        else:
            projection = z_buffer_projection_kannala_brandt(
                points_camera,
                fx,
                fy,
                cx,
                cy,
                source_distortion,
                width,
                height,
                args.minimum_depth_m,
                args.maximum_depth_m,
            )

        raw_depth = np.load(
            raw_product_root / "raw_depth_meter" / f"{output_index:08d}.npy",
            allow_pickle=False,
        )
        disparity = np.load(
            raw_product_root / "raw_disparity" / f"{output_index:08d}.npy",
            allow_pickle=False,
        )
        consistency_path = (
            raw_product_root
            / "lr_consistency_error"
            / f"{output_index:08d}.npy"
        )
        consistency_error = (
            np.load(consistency_path, allow_pickle=False)
            if consistency_path.is_file()
            else None
        )
        filtered_depth_mm = cv2.imread(
            str(depth_dataset / "depth" / f"{output_index:08d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        rgb = cv2.imread(str(frame["cam0"]), cv2.IMREAD_COLOR)
        if (
            raw_depth.shape != (height, width)
            or disparity.shape != raw_depth.shape
            or (
                consistency_error is not None
                and consistency_error.shape != raw_depth.shape
            )
            or filtered_depth_mm is None
            or filtered_depth_mm.shape != raw_depth.shape
            or rgb is None
        ):
            raise ValueError(f"Depth product shape mismatch for {source_index}")

        u = projection["u"]
        v = projection["v"]
        reference = projection["depth"].astype(np.float64)
        prediction = raw_depth[v, u].astype(np.float64)
        projected_disparity = disparity[v, u].astype(np.float64)
        projected_consistency_error = (
            consistency_error[v, u].astype(np.float64)
            if consistency_error is not None
            else None
        )
        filtered_prediction = (
            filtered_depth_mm[v, u].astype(np.float64) / 1000.0
        )
        raw_valid = (
            np.isfinite(prediction)
            & (prediction >= args.minimum_depth_m)
            & (prediction <= args.maximum_depth_m)
        )
        filtered_valid = (
            np.isfinite(filtered_prediction)
            & (filtered_prediction >= args.minimum_depth_m)
            & (filtered_prediction <= args.maximum_depth_m)
        )
        append_policy(
            policy_values, "raw_0_25_to_30m", prediction, reference, raw_valid
        )
        lidar_reference_5m = (
            reference >= args.minimum_depth_m
        ) & (reference <= 5.0)
        append_policy(
            policy_values,
            "raw_lidar_reference_0_25_to_5m",
            prediction,
            reference,
            raw_valid & lidar_reference_5m,
        )
        append_policy(
            policy_values,
            filtered_policy_name,
            filtered_prediction,
            reference,
            filtered_valid,
        )
        append_policy(
            policy_values,
            f"{filtered_policy_name}_lidar_reference_0_25_to_5m",
            filtered_prediction,
            reference,
            filtered_valid & lidar_reference_5m,
        )
        if args.candidate_depth_scale is not None:
            scaled_prediction = prediction * args.candidate_depth_scale
            scaled_filtered_prediction = (
                filtered_prediction * args.candidate_depth_scale
            )
            scaled_raw_valid = raw_valid & (
                (scaled_prediction >= args.minimum_depth_m)
                & (scaled_prediction <= args.maximum_depth_m)
            )
            scaled_filtered_valid = filtered_valid & (
                (scaled_filtered_prediction >= args.minimum_depth_m)
                & (scaled_filtered_prediction <= args.maximum_depth_m)
            )
            append_policy(
                policy_values,
                "candidate_scaled_raw_0_25_to_30m",
                scaled_prediction,
                reference,
                scaled_raw_valid,
            )
            append_policy(
                policy_values,
                "candidate_scaled_raw_lidar_reference_0_25_to_5m",
                scaled_prediction,
                reference,
                scaled_raw_valid & lidar_reference_5m,
            )
            append_policy(
                policy_values,
                f"candidate_scaled_{filtered_policy_name}",
                scaled_filtered_prediction,
                reference,
                scaled_filtered_valid,
            )
            append_policy(
                policy_values,
                (
                    f"candidate_scaled_{filtered_policy_name}"
                    "_lidar_reference_0_25_to_5m"
                ),
                scaled_filtered_prediction,
                reference,
                scaled_filtered_valid & lidar_reference_5m,
            )
        threshold_reports = {}
        if projected_consistency_error is not None:
            for threshold in CONSISTENCY_THRESHOLDS_PX:
                selected = raw_valid & np.isfinite(
                    projected_consistency_error
                ) & (projected_consistency_error <= threshold)
                name = f"lr_error_le_{threshold:g}px"
                append_policy(policy_values, name, prediction, reference, selected)
                threshold_reports[name] = {
                    "selected_count": int(selected.sum()),
                    "coverage_ratio": float(selected.mean()),
                    "depth_error": error_summary(
                        prediction[selected], reference[selected]
                    ),
                }

        bin_reports = []
        for lower, upper in RANGE_BINS_M:
            reference_bin = (reference >= lower) & (reference < upper)
            selected = reference_bin & raw_valid
            bin_reports.append(
                {
                    "lidar_range_m": [lower, upper],
                    "lidar_pixel_count": int(reference_bin.sum()),
                    "raw_depth_count": int(selected.sum()),
                    "coverage_ratio": float(
                        selected.sum() / max(1, int(reference_bin.sum()))
                    ),
                    "depth_error": error_summary(
                        prediction[selected], reference[selected]
                    ),
                }
            )

        scale_sample = (
            raw_valid
            & (reference >= 0.5)
            & (reference <= 5.0)
            & (prediction >= 0.5)
            & (prediction <= 5.0)
        )
        scale_ratio = reference[scale_sample] / prediction[scale_sample]
        frame_scale = (
            float(np.median(scale_ratio)) if scale_ratio.size else None
        )
        if frame_scale is not None:
            ratio_by_frame.append(frame_scale)
        frame_report = {
            "output_index": output_index,
            "source_index": source_index,
            "reference_camera": reference_camera,
            "stereo_delta_ms": float(frame["stereo_delta_ms"]),
            "camera_minus_lidar_ms": (camera_time - lidar_time) / 1.0e6,
            "lidar_unique_projected_pixels": int(reference.size),
            "raw_0_25_to_30m": {
                "selected_count": int(raw_valid.sum()),
                "coverage_ratio": float(raw_valid.mean()),
                "depth_error": error_summary(
                    prediction[raw_valid], reference[raw_valid]
                ),
            },
            "raw_lidar_reference_0_25_to_5m": {
                "selected_count": int((raw_valid & lidar_reference_5m).sum()),
                "coverage_ratio": float(
                    (raw_valid & lidar_reference_5m).sum()
                    / max(1, int(lidar_reference_5m.sum()))
                ),
                "depth_error": error_summary(
                    prediction[raw_valid & lidar_reference_5m],
                    reference[raw_valid & lidar_reference_5m],
                ),
            },
            filtered_policy_name: {
                "selected_count": int(filtered_valid.sum()),
                "coverage_ratio": float(filtered_valid.mean()),
                "depth_error": error_summary(
                    filtered_prediction[filtered_valid],
                    reference[filtered_valid],
                ),
            },
            "fixed_left_right_error_thresholds": threshold_reports,
            "lidar_range_bins": bin_reports,
            "candidate_scale_reference_over_stereo_0_5_to_5m": {
                "sample_count": int(scale_ratio.size),
                "median": frame_scale,
                "p25": (
                    float(np.percentile(scale_ratio, 25))
                    if scale_ratio.size
                    else None
                ),
                "p75": (
                    float(np.percentile(scale_ratio, 75))
                    if scale_ratio.size
                    else None
                ),
            },
            "projected_disparity_px": {
                "p50": (
                    float(np.median(projected_disparity[raw_valid]))
                    if np.any(raw_valid)
                    else None
                )
            },
        }
        correspondence_path = (
            correspondence_dir / f"{source_index:06d}.npz"
        )
        np.savez_compressed(
            correspondence_path,
            u=u.astype(np.int32),
            v=v.astype(np.int32),
            lidar_depth_m=reference.astype(np.float32),
            raw_prediction_m=prediction.astype(np.float32),
            filtered_prediction_m=filtered_prediction.astype(np.float32),
            raw_valid=raw_valid.astype(np.bool_),
            filtered_valid=filtered_valid.astype(np.bool_),
            disparity_px=projected_disparity.astype(np.float32),
            lr_consistency_error_px=(
                projected_consistency_error.astype(np.float32)
                if projected_consistency_error is not None
                else np.full(reference.shape, np.nan, dtype=np.float32)
            ),
        )
        frame_report["correspondence_array"] = str(correspondence_path)
        frame_reports.append(frame_report)
        save_frame_overlay(
            overlay_dir / f"{source_index:06d}.png",
            rgb,
            projection,
            prediction,
            raw_valid,
            source_index,
            reference_camera,
        )

    aggregate_policies = {
        name: summarize_policy(values) for name, values in policy_values.items()
    }
    aggregate_counts = {
        name: int(
            sum(array.size for array in values["prediction"])
        )
        for name, values in policy_values.items()
    }
    scale_array = np.asarray(ratio_by_frame, dtype=np.float64)
    report = {
        "contract": (
            "LiDAR is used only after stereo inference for independent validation; "
            "no image warp, homography, or depth generation uses LiDAR"
        ),
        "raw_dataset": str(raw_dataset),
        "depth_dataset": str(depth_dataset),
        "raw_product_root": str(raw_product_root),
        "confidence_mode": confidence_mode,
        "has_left_right_consistency": has_left_right_consistency,
        "filtered_policy_name": filtered_policy_name,
        "source_indices": [item["source_index"] for item in frame_reports],
        "requested_source_frame_range_inclusive": [
            args.source_frame_start,
            args.source_frame_end,
        ],
        "evaluated_frame_count": len(frame_reports),
        "available_depth_frame_count": len(tick_index["frames"]),
        "selection_fold": {
            "count": args.selection_fold_count,
            "index": args.selection_fold_index,
        },
        "candidate_depth_scale": args.candidate_depth_scale,
        "reference_camera": reference_camera,
        "camera_pose_source": resolved_camera_pose_source,
        "reference_pose_semantics": (
            "world_T_camera from depth-dataset pose/poses.txt"
            if use_depth_dataset_pose
            else (
                "base_T_cam0 from auxiliary head_camera poses"
                if reference_camera == "cam0"
                else "base_T_cam1 = base_T_cam0 @ T_cam0_cam1"
            )
        ),
        "source_projection_model": source_projection_model,
        "lidar_projection_model": args.lidar_projection_model,
        "source_distortion": source_distortion.tolist(),
        "camera": camera,
        "evaluation_range_m": [
            args.minimum_depth_m,
            args.maximum_depth_m,
        ],
        "aggregate": {
            "selected_counts": aggregate_counts,
            "policies": aggregate_policies,
            "candidate_scale_reference_over_stereo_per_frame": (
                ratio_by_frame
            ),
            "candidate_scale_stability": {
                "median": (
                    float(np.median(scale_array)) if scale_array.size else None
                ),
                "minimum": (
                    float(scale_array.min()) if scale_array.size else None
                ),
                "maximum": (
                    float(scale_array.max()) if scale_array.size else None
                ),
                "relative_span": (
                    float(
                        (scale_array.max() - scale_array.min())
                        / np.median(scale_array)
                    )
                    if scale_array.size
                    else None
                ),
            },
        },
        "frames": frame_reports,
        "interpretation_guard": (
            "Coverage measures availability, not metric accuracy. A global depth "
            "scale should only be applied if the per-frame candidate ratios are "
            "stable and held-out validation improves."
        ),
    }
    (output / "lidar_batch_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    save_summary_chart(output / "lidar_batch_summary.png", frame_reports)
    gallery_indices = np.linspace(
        0,
        len(tick_frames) - 1,
        num=min(12, len(tick_frames)),
        dtype=np.int64,
    )
    save_depth_gallery(
        output / "raw_depth_overlay_5m_gallery.png",
        depth_dataset,
        raw_product_root,
        [tick_frames[int(index)] for index in np.unique(gallery_indices)],
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
