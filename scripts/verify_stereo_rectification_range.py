#!/usr/bin/env python3
"""Verify undistortion evidence and horizontal stereo rectification over a range.

The experiment deliberately evaluates the stored PNG pixels.  Feature matches are
selected with a general fundamental-matrix RANSAC, then scored against the
calibration's *fixed* horizontal-stereo model.  This avoids declaring a pair
rectified merely because an arbitrary fundamental matrix can be fitted to it.

Two projection hypotheses are also compared on the same inlier correspondences:

* pinhole: the stored pixels are already undistorted to a virtual pinhole image;
* Kannala-Brandt: the stored pixels still use the declared fisheye projection.

Without raw images or the producer's factory rectification maps, the projection
comparison is evidence rather than proof of which upstream pixel operation ran.
The horizontal epipolar residual, however, directly tests whether the delivered
pair is ready for conventional rectified stereo matching.
"""

from __future__ import annotations

import argparse
import csv
import json
import tomllib
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from audit_rectified_stereo_frame import match_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--calibration",
        type=Path,
        help=(
            "Optional Kalibr-style camchain YAML. When supplied, the report "
            "uses its equidistant intrinsics and T_cn_cnm1 instead of the "
            "dataset-native CameraInfo export."
        ),
    )
    parser.add_argument(
        "--visualization-frames",
        nargs="*",
        type=int,
        help="Defaults to the start, midpoint, and end frames.",
    )
    parser.add_argument(
        "--sync-threshold-ms",
        type=float,
        default=5.0,
        help="Threshold used for the low-time-delta aggregate.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def percentile_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0}
    levels = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    result = np.percentile(values, levels)
    return {
        "count": int(values.size),
        "minimum": float(result[0]),
        "p10": float(result[1]),
        "p25": float(result[2]),
        "p50": float(result[3]),
        "p75": float(result[4]),
        "p90": float(result[5]),
        "p95": float(result[6]),
        "p99": float(result[7]),
        "maximum": float(result[8]),
    }


def normalized_rays(
    points: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | None]]:
    normalized = np.column_stack(
        (
            (points[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0],
            (points[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1],
        )
    )
    if model == "pinhole":
        rays = np.column_stack((normalized, np.ones(len(normalized))))
        valid = np.all(np.isfinite(rays), axis=1)
        model_diagnostics: dict[str, float | None] = {
            "maximum_monotonic_theta_rad": None,
            "maximum_monotonic_distorted_radius": None,
        }
    elif model == "kannala_brandt":
        theta_distorted = np.linalg.norm(normalized, axis=1)
        k = np.pad(np.asarray(distortion, dtype=np.float64), (0, 4))[:4]
        derivative_roots = np.roots(
            [9.0 * k[3], 7.0 * k[2], 5.0 * k[1], 3.0 * k[0], 1.0]
        )
        positive_real_roots = sorted(
            float(root.real)
            for root in derivative_roots
            if abs(float(root.imag)) < 1.0e-9 and float(root.real) > 0.0
        )
        maximum_theta = (
            float(np.sqrt(positive_real_roots[0]))
            if positive_real_roots
            else float(np.pi)
        )

        def distorted_radius(theta: np.ndarray | float) -> np.ndarray:
            theta_array = np.asarray(theta, dtype=np.float64)
            theta2 = theta_array * theta_array
            return theta_array * (
                1.0
                + k[0] * theta2
                + k[1] * theta2**2
                + k[2] * theta2**3
                + k[3] * theta2**4
            )

        maximum_radius = float(distorted_radius(maximum_theta))
        valid = (
            np.isfinite(theta_distorted)
            & (theta_distorted >= 0.0)
            & (theta_distorted <= maximum_radius + 1.0e-12)
        )
        lower = np.zeros(len(theta_distorted), dtype=np.float64)
        upper = np.full(len(theta_distorted), maximum_theta, dtype=np.float64)
        for _ in range(56):
            middle = (lower + upper) * 0.5
            below = distorted_radius(middle) < theta_distorted
            lower = np.where(valid & below, middle, lower)
            upper = np.where(valid & ~below, middle, upper)
        theta = (lower + upper) * 0.5
        radial_direction = np.divide(
            normalized,
            theta_distorted[:, None],
            out=np.zeros_like(normalized),
            where=theta_distorted[:, None] > 1.0e-12,
        )
        rays = np.column_stack(
            (
                radial_direction[:, 0] * np.sin(theta),
                radial_direction[:, 1] * np.sin(theta),
                np.cos(theta),
            )
        )
        model_diagnostics = {
            "maximum_monotonic_theta_rad": maximum_theta,
            "maximum_monotonic_distorted_radius": maximum_radius,
        }
    else:
        raise ValueError(f"Unknown projection model: {model}")
    ray_norm = np.linalg.norm(rays, axis=1, keepdims=True)
    valid &= np.isfinite(ray_norm[:, 0]) & (ray_norm[:, 0] > 1.0e-12)
    rays = np.divide(
        rays,
        ray_norm,
        out=np.full_like(rays, np.nan),
        where=ray_norm > 1.0e-12,
    )
    return rays, valid, model_diagnostics


def intrinsics_matrix(values: list[float]) -> np.ndarray:
    fx, fy, cx, cy = (float(value) for value in values)
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def adapt_projection_for_center_crop(
    matrix: np.ndarray,
    source_resolution: list[int],
    target_resolution: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    source_width, source_height = (
        float(value) for value in source_resolution
    )
    target_width, target_height = (
        float(value) for value in target_resolution
    )
    scale = max(
        target_width / source_width,
        target_height / source_height,
    )
    crop_x = (source_width * scale - target_width) * 0.5
    crop_y = (source_height * scale - target_height) * 0.5
    pixel_transform = np.asarray(
        [
            [scale, 0.0, -crop_x],
            [0.0, scale, -crop_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    adapted = pixel_transform @ np.asarray(matrix, dtype=np.float64)
    return adapted, {
        "policy": "center_crop_then_uniform_resize",
        "source_resolution": [
            int(source_width),
            int(source_height),
        ],
        "target_resolution": [
            int(target_width),
            int(target_height),
        ],
        "uniform_scale": scale,
        "crop_offset_target_px": [crop_x, crop_y],
        "crop_offset_source_px": [crop_x / scale, crop_y / scale],
        "pixel_transform_3x3": pixel_transform.tolist(),
    }


def load_calibration(
    dataset: Path,
    manifest: dict[str, Any],
    calibration_path: Path | None,
    target_resolution: list[int],
) -> dict[str, Any]:
    calibration_directory = dataset / "calibrations" / manifest["sequence"]
    if calibration_path is None:
        intrinsics_by_camera = {}
        for camera in ("cam0", "cam1"):
            payload = yaml.safe_load(
                (
                    calibration_directory / f"calib_{camera}_intrinsics.yaml"
                ).read_text()
            )["intrinsics"]
            intrinsics_by_camera[camera] = payload
        stereo_calibration = yaml.safe_load(
            (calibration_directory / "calib_cam0_to_cam1.yaml").read_text()
        )
        transform_cam0_from_cam1 = np.asarray(
            stereo_calibration["transform"]["matrix_4x4"], dtype=np.float64
        )
        return {
            "source_format": "dataset_native_camera_info",
            "source_path": str(calibration_directory),
            "intrinsics_by_camera": intrinsics_by_camera,
            "cam0_K": np.asarray(
                intrinsics_by_camera["cam0"]["K"], dtype=np.float64
            ).reshape(3, 3),
            "cam1_K": np.asarray(
                intrinsics_by_camera["cam1"]["K"], dtype=np.float64
            ).reshape(3, 3),
            "cam0_D": np.asarray(
                intrinsics_by_camera["cam0"]["D"], dtype=np.float64
            ),
            "cam1_D": np.asarray(
                intrinsics_by_camera["cam1"]["D"], dtype=np.float64
            ),
            "cam0_R": np.asarray(
                intrinsics_by_camera["cam0"]["R"], dtype=np.float64
            ).reshape(3, 3),
            "cam1_R": np.asarray(
                intrinsics_by_camera["cam1"]["R"], dtype=np.float64
            ).reshape(3, 3),
            "cam0_P": np.asarray(
                intrinsics_by_camera["cam0"]["P"], dtype=np.float64
            ).reshape(3, 4),
            "cam1_P": np.asarray(
                intrinsics_by_camera["cam1"]["P"], dtype=np.float64
            ).reshape(3, 4),
            "transform_cam0_from_cam1": transform_cam0_from_cam1,
            "transform_cam1_from_cam0": np.linalg.inv(
                transform_cam0_from_cam1
            ),
            "resolution": [
                int(intrinsics_by_camera["cam0"]["width"]),
                int(intrinsics_by_camera["cam0"]["height"]),
            ],
            "supports_raw_to_rectified_hypothesis": False,
        }

    resolved = calibration_path.resolve()
    calibration_text = resolved.read_text()
    payload = yaml.safe_load(calibration_text)
    if not isinstance(payload, dict):
        payload = tomllib.loads(calibration_text)
    if {"left", "right", "stereo", "rectification"}.issubset(payload):
        left = payload["left"]
        right = payload["right"]
        stereo = payload["stereo"]
        rectification = payload["rectification"]
        source_resolution = [int(value) for value in left["resolution"]]
        if source_resolution != [
            int(value) for value in right["resolution"]
        ]:
            raise ValueError("V2 left/right resolutions disagree")
        cam0_K, resolution_adaptation = adapt_projection_for_center_crop(
            np.asarray(left["intrinsic"], dtype=np.float64),
            source_resolution,
            target_resolution,
        )
        cam1_K, right_resolution_adaptation = (
            adapt_projection_for_center_crop(
                np.asarray(right["intrinsic"], dtype=np.float64),
                source_resolution,
                target_resolution,
            )
        )
        if resolution_adaptation != right_resolution_adaptation:
            raise RuntimeError("Left/right resolution adaptations disagree")
        projection0, projection_adaptation = (
            adapt_projection_for_center_crop(
                np.asarray(rectification["P1"], dtype=np.float64),
                source_resolution,
                target_resolution,
            )
        )
        projection1, right_projection_adaptation = (
            adapt_projection_for_center_crop(
                np.asarray(rectification["P2"], dtype=np.float64),
                source_resolution,
                target_resolution,
            )
        )
        if projection_adaptation != right_projection_adaptation:
            raise RuntimeError("Left/right projection adaptations disagree")
        transform_cam1_from_cam0 = np.eye(4, dtype=np.float64)
        transform_cam1_from_cam0[:3, :3] = np.asarray(
            stereo["rotation"], dtype=np.float64
        )
        transform_cam1_from_cam0[:3, 3] = np.asarray(
            stereo["translation"], dtype=np.float64
        )
        return {
            "source_format": "shitong_fisheye_precomputed_rectification",
            "source_path": str(resolved),
            "raw_payload": payload,
            "intrinsics_by_camera": {
                "cam0": {
                    "distortion_model": "equidistant",
                    "roi": {"do_rectify": False},
                },
                "cam1": {
                    "distortion_model": "equidistant",
                    "roi": {"do_rectify": False},
                },
            },
            "cam0_K": cam0_K,
            "cam1_K": cam1_K,
            "cam0_D": np.asarray(left["distortion"], dtype=np.float64),
            "cam1_D": np.asarray(right["distortion"], dtype=np.float64),
            "cam0_R": np.asarray(
                rectification["R1"], dtype=np.float64
            ),
            "cam1_R": np.asarray(
                rectification["R2"], dtype=np.float64
            ),
            "cam0_P": projection0,
            "cam1_P": projection1,
            "transform_cam0_from_cam1": np.linalg.inv(
                transform_cam1_from_cam0
            ),
            "transform_cam1_from_cam0": transform_cam1_from_cam0,
            "resolution": [int(value) for value in target_resolution],
            "source_resolution": source_resolution,
            "resolution_adaptation": resolution_adaptation,
            "provided_reprojection_error_px": {
                "cam0": float(left["reprojection_error"]),
                "cam1": float(right["reprojection_error"]),
                "stereo": float(stereo["reprojection_error"]),
            },
            "supports_raw_to_rectified_hypothesis": True,
        }
    if not {"cam0", "cam1"}.issubset(payload):
        raise ValueError(
            "External calibration must contain either cam0/cam1 or "
            "left/right/stereo/rectification tables"
        )
    cam0 = payload["cam0"]
    cam1 = payload["cam1"]
    if "T_cn_cnm1" not in cam1:
        raise ValueError("Kalibr cam1 must contain T_cn_cnm1")
    for name, camera in (("cam0", cam0), ("cam1", cam1)):
        if camera.get("camera_model") != "pinhole":
            raise ValueError(f"{name} camera_model must be pinhole")
        if camera.get("distortion_model") != "equidistant":
            raise ValueError(f"{name} distortion_model must be equidistant")
        if len(camera.get("intrinsics", [])) != 4:
            raise ValueError(f"{name} must provide four intrinsics")
        if len(camera.get("distortion_coeffs", [])) != 4:
            raise ValueError(f"{name} must provide four distortion coefficients")
    if cam0.get("resolution") != cam1.get("resolution"):
        raise ValueError("External stereo resolutions disagree")
    transform_cam1_from_cam0 = np.asarray(
        cam1["T_cn_cnm1"], dtype=np.float64
    )
    if transform_cam1_from_cam0.shape != (4, 4):
        raise ValueError("T_cn_cnm1 must be 4x4")
    cam0_K = intrinsics_matrix(cam0["intrinsics"])
    cam1_K = intrinsics_matrix(cam1["intrinsics"])
    cam0_D = np.asarray(cam0["distortion_coeffs"], dtype=np.float64)
    cam1_D = np.asarray(cam1["distortion_coeffs"], dtype=np.float64)
    resolution = [int(value) for value in cam0["resolution"]]
    rectification0, rectification1, projection0, projection1, _ = (
        cv2.fisheye.stereoRectify(
            cam0_K,
            cam0_D,
            cam1_K,
            cam1_D,
            tuple(resolution),
            transform_cam1_from_cam0[:3, :3],
            transform_cam1_from_cam0[:3, 3],
            flags=cv2.CALIB_ZERO_DISPARITY,
            newImageSize=tuple(resolution),
            balance=0.0,
            fov_scale=1.0,
        )
    )
    return {
        "source_format": "kalibr_camchain_equidistant",
        "source_path": str(resolved),
        "raw_payload": payload,
        "intrinsics_by_camera": {
            "cam0": {
                "distortion_model": "equidistant",
                "roi": {"do_rectify": False},
            },
            "cam1": {
                "distortion_model": "equidistant",
                "roi": {"do_rectify": False},
            },
        },
        "cam0_K": cam0_K,
        "cam1_K": cam1_K,
        "cam0_D": cam0_D,
        "cam1_D": cam1_D,
        "cam0_R": rectification0,
        "cam1_R": rectification1,
        "cam0_P": projection0,
        "cam1_P": projection1,
        "transform_cam0_from_cam1": np.linalg.inv(
            transform_cam1_from_cam0
        ),
        "transform_cam1_from_cam0": transform_cam1_from_cam0,
        "resolution": resolution,
        "supports_raw_to_rectified_hypothesis": True,
    }


def rectified_normalized_points(
    rays: np.ndarray,
    rectification: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rectified_rays = (rectification @ rays.T).T
    valid = (
        np.all(np.isfinite(rectified_rays), axis=1)
        & (rectified_rays[:, 2] > 1.0e-9)
    )
    points = np.divide(
        rectified_rays[:, :2],
        rectified_rays[:, 2:3],
        out=np.full((len(rays), 2), np.nan, dtype=np.float64),
        where=np.abs(rectified_rays[:, 2:3]) > 1.0e-9,
    )
    return points, valid


def angular_epipolar_error_deg(
    cam0_rays: np.ndarray,
    cam1_rays: np.ndarray,
    rotation_cam0_from_cam1: np.ndarray,
    translation_cam0_from_cam1: np.ndarray,
) -> np.ndarray:
    rotated_cam1_rays = (rotation_cam0_from_cam1 @ cam1_rays.T).T
    plane_normals = np.cross(
        np.broadcast_to(translation_cam0_from_cam1, rotated_cam1_rays.shape),
        rotated_cam1_rays,
    )
    sine_error = np.abs(np.sum(cam0_rays * plane_normals, axis=1)) / np.maximum(
        np.linalg.norm(plane_normals, axis=1), 1.0e-12
    )
    return np.degrees(np.arcsin(np.clip(sine_error, 0.0, 1.0)))


def summarize_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        return {"frame_count": 0}
    median_vertical = np.asarray(
        [item["stereo"]["absolute_vertical_error_px"]["p50"] for item in frames]
    )
    p95_vertical = np.asarray(
        [item["stereo"]["absolute_vertical_error_px"]["p95"] for item in frames]
    )
    positive = np.asarray(
        [item["stereo"]["positive_disparity_ratio"] for item in frames]
    )
    pinhole = np.asarray(
        [item["projection_test"]["pinhole_error_deg"]["p50"] for item in frames]
    )
    fisheye = np.asarray(
        [
            item["projection_test"]["kannala_brandt_error_deg"]["p50"]
            for item in frames
        ]
    )
    stereo_delta = np.asarray([item["stereo_delta_ms"] for item in frames])
    summary = {
        "frame_count": len(frames),
        "feature_inlier_count": int(
            sum(item["stereo"]["feature_inlier_count"] for item in frames)
        ),
        "passed_horizontal_rectification_frame_count": int(
            sum(item["rectification_gate"]["passed"] for item in frames)
        ),
        "passed_horizontal_rectification_frame_ratio": float(
            np.mean([item["rectification_gate"]["passed"] for item in frames])
        ),
        "frame_median_absolute_vertical_error_px": percentile_summary(
            median_vertical
        ),
        "frame_p95_absolute_vertical_error_px": percentile_summary(p95_vertical),
        "frame_positive_disparity_ratio": percentile_summary(positive),
        "stereo_delta_ms": percentile_summary(stereo_delta),
        "pinhole_frame_median_angular_epipolar_error_deg": percentile_summary(
            pinhole
        ),
        "kannala_brandt_frame_median_angular_epipolar_error_deg": (
            percentile_summary(fisheye)
        ),
        "pinhole_better_frame_count": int(np.sum(pinhole < fisheye)),
        "pinhole_better_frame_ratio": float(np.mean(pinhole < fisheye)),
        "median_pinhole_to_kannala_brandt_error_ratio": float(
            np.median(pinhole / np.maximum(fisheye, 1.0e-12))
        ),
    }
    rectified_hypothesis_frames = [
        item
        for item in frames
        if item.get("new_calibration_raw_input_hypothesis") is not None
    ]
    if rectified_hypothesis_frames:
        valid_ratio = np.asarray(
            [
                item["new_calibration_raw_input_hypothesis"][
                    "valid_feature_ratio"
                ]
                for item in rectified_hypothesis_frames
            ]
        )
        rectified_median_vertical = np.asarray(
            [
                item["new_calibration_raw_input_hypothesis"][
                    "absolute_vertical_error_equivalent_px"
                ]["p50"]
                for item in rectified_hypothesis_frames
            ]
        )
        rectified_p95_vertical = np.asarray(
            [
                item["new_calibration_raw_input_hypothesis"][
                    "absolute_vertical_error_equivalent_px"
                ]["p95"]
                for item in rectified_hypothesis_frames
            ]
        )
        rectified_positive = np.asarray(
            [
                item["new_calibration_raw_input_hypothesis"][
                    "positive_disparity_ratio"
                ]
                for item in rectified_hypothesis_frames
            ]
        )
        stored_same_subset_median = np.asarray(
            [
                item["new_calibration_raw_input_hypothesis"][
                    "stored_absolute_vertical_error_same_valid_subset_px"
                ]["p50"]
                for item in rectified_hypothesis_frames
            ]
        )
        stored_same_subset_p95 = np.asarray(
            [
                item["new_calibration_raw_input_hypothesis"][
                    "stored_absolute_vertical_error_same_valid_subset_px"
                ]["p95"]
                for item in rectified_hypothesis_frames
            ]
        )
        summary["new_calibration_raw_input_hypothesis"] = {
            "frame_count": len(rectified_hypothesis_frames),
            "passed_frame_count": int(
                sum(
                    item["new_calibration_raw_input_hypothesis"]["gate"][
                        "passed"
                    ]
                    for item in rectified_hypothesis_frames
                )
            ),
            "passed_frame_ratio": float(
                np.mean(
                    [
                        item["new_calibration_raw_input_hypothesis"]["gate"][
                            "passed"
                        ]
                        for item in rectified_hypothesis_frames
                    ]
                )
            ),
            "frame_valid_feature_ratio": percentile_summary(valid_ratio),
            "frame_median_absolute_vertical_error_equivalent_px": (
                percentile_summary(rectified_median_vertical)
            ),
            "frame_p95_absolute_vertical_error_equivalent_px": (
                percentile_summary(rectified_p95_vertical)
            ),
            "frame_positive_disparity_ratio": percentile_summary(
                rectified_positive
            ),
            "stored_frame_median_absolute_vertical_error_same_valid_subset_px": (
                percentile_summary(stored_same_subset_median)
            ),
            "stored_frame_p95_absolute_vertical_error_same_valid_subset_px": (
                percentile_summary(stored_same_subset_p95)
            ),
        }
    return summary


def draw_visualization(
    path: Path,
    cam0: np.ndarray,
    cam1: np.ndarray,
    cam0_points: np.ndarray,
    cam1_points: np.ndarray,
    frame: dict[str, Any],
) -> None:
    scale = 0.5
    height = int(round(cam0.shape[0] * scale))
    width = int(round(cam0.shape[1] * scale))
    left = cv2.resize(cam0, (width, height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(cam1, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.hstack((left, right))
    for y in range(40, height, 40):
        cv2.line(canvas, (0, y), (2 * width - 1, y), (0, 190, 0), 1)
    if len(cam0_points):
        selected = np.linspace(
            0, len(cam0_points) - 1, min(120, len(cam0_points))
        ).astype(int)
        for index in selected:
            point0 = np.rint(cam0_points[index] * scale).astype(int)
            point1 = np.rint(cam1_points[index] * scale).astype(int)
            destination = (int(point1[0] + width), int(point1[1]))
            vertical_error = abs(float(point0[1] - point1[1]))
            color = (30, 210, 30) if vertical_error < 0.5 else (30, 50, 230)
            cv2.line(
                canvas,
                tuple(point0),
                destination,
                color,
                1,
                cv2.LINE_AA,
            )
    stereo = frame["stereo"]
    text = (
        f"frame={frame['frame']} dt={frame['stereo_delta_ms']:.3f} ms "
        f"|dy| p50={stereo['absolute_vertical_error_px']['p50']:.2f} px "
        f"p95={stereo['absolute_vertical_error_px']['p95']:.2f} px"
    )
    cv2.rectangle(canvas, (0, 0), (2 * width, 38), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write visualization: {path}")


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if args.start_frame < 0 or args.end_frame < args.start_frame:
        raise ValueError("Frame range must satisfy 0 <= start <= end")
    if args.sync_threshold_ms <= 0.0:
        raise ValueError("sync-threshold-ms must be positive")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")

    manifest = load_json(dataset / "manifest.json")
    records = {
        int(record["tick"]): record
        for record in load_jsonl(dataset / "manifest.jsonl")
    }
    expected_frames = list(range(args.start_frame, args.end_frame + 1))
    missing = [frame for frame in expected_frames if frame not in records]
    if missing:
        raise ValueError(f"Frames absent from manifest: {missing}")

    first_descriptors = {
        item["camera"]: item
        for item in records[expected_frames[0]]["images"]
    }
    first_image = cv2.imread(
        str((dataset / first_descriptors["cam0"]["path"]).resolve()),
        cv2.IMREAD_COLOR,
    )
    if first_image is None:
        raise RuntimeError("Could not decode the first cam0 image")
    target_resolution = [int(first_image.shape[1]), int(first_image.shape[0])]
    calibration = load_calibration(
        dataset,
        manifest,
        args.calibration,
        target_resolution,
    )
    intrinsics_by_camera = calibration["intrinsics_by_camera"]
    transform = calibration["transform_cam0_from_cam1"]
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    cam0_intrinsics = calibration["cam0_K"]
    cam1_intrinsics = calibration["cam1_K"]
    cam0_distortion = calibration["cam0_D"]
    cam1_distortion = calibration["cam1_D"]
    cam0_rectification = calibration["cam0_R"]
    cam1_rectification = calibration["cam1_R"]
    cam0_projection = calibration["cam0_P"]
    cam1_projection = calibration["cam1_P"]
    baseline_m = float(np.linalg.norm(translation))
    expected_right_projection_tx = -float(cam1_projection[0, 0]) * baseline_m
    projection_encodes_baseline = bool(
        np.isclose(
            cam1_projection[0, 3],
            expected_right_projection_tx,
            atol=max(1.0e-6, abs(expected_right_projection_tx) * 1.0e-6),
        )
    )
    equivalent_focal_length_px = float(
        np.mean(
            [
                cam0_intrinsics[0, 0],
                cam0_intrinsics[1, 1],
                cam1_intrinsics[0, 0],
                cam1_intrinsics[1, 1],
            ]
        )
    )

    output.mkdir(parents=True)
    visualization_directory = output / "visualizations"
    visualization_directory.mkdir()
    requested_visualizations = set(
        args.visualization_frames
        if args.visualization_frames is not None
        else [
            args.start_frame,
            (args.start_frame + args.end_frame) // 2,
            args.end_frame,
        ]
    )
    if not requested_visualizations.issubset(expected_frames):
        raise ValueError("visualization-frames must be inside the requested range")

    frame_reports = []
    pooled_vertical = []
    pooled_pinhole_angular = []
    pooled_fisheye_angular = []
    image_shapes: set[tuple[int, ...]] = set()
    for frame_index in expected_frames:
        descriptors = {
            item["camera"]: item for item in records[frame_index]["images"]
        }
        paths = {
            camera: (dataset / descriptors[camera]["path"]).resolve()
            for camera in ("cam0", "cam1")
        }
        images = {
            camera: cv2.imread(str(paths[camera]), cv2.IMREAD_COLOR)
            for camera in ("cam0", "cam1")
        }
        if images["cam0"] is None or images["cam1"] is None:
            raise RuntimeError(f"Could not decode frame {frame_index}")
        if images["cam0"].shape != images["cam1"].shape:
            raise RuntimeError(f"Stereo shape mismatch at frame {frame_index}")
        image_shapes.add(tuple(images["cam0"].shape))
        stereo_delta_ms = abs(
            int(descriptors["cam0"]["sensor_time_ns"])
            - int(descriptors["cam1"]["sensor_time_ns"])
        ) / 1.0e6

        geometry = match_features(images["cam0"], images["cam1"])
        cam0_points = geometry.pop("_left_inliers")
        cam1_points = geometry.pop("_right_inliers")
        vertical_error = np.abs(cam0_points[:, 1] - cam1_points[:, 1])
        pinhole_cam0_rays, pinhole_cam0_valid, _ = normalized_rays(
            cam0_points, cam0_intrinsics, cam0_distortion, "pinhole"
        )
        pinhole_cam1_rays, pinhole_cam1_valid, _ = normalized_rays(
            cam1_points, cam1_intrinsics, cam1_distortion, "pinhole"
        )
        fisheye_cam0_rays, fisheye_cam0_valid, cam0_model_diagnostics = (
            normalized_rays(
            cam0_points,
            cam0_intrinsics,
            cam0_distortion,
            "kannala_brandt",
        )
        )
        fisheye_cam1_rays, fisheye_cam1_valid, cam1_model_diagnostics = (
            normalized_rays(
            cam1_points,
            cam1_intrinsics,
            cam1_distortion,
            "kannala_brandt",
        )
        )
        pinhole_valid = pinhole_cam0_valid & pinhole_cam1_valid
        fisheye_valid = fisheye_cam0_valid & fisheye_cam1_valid
        common_projection_valid = pinhole_valid & fisheye_valid
        pinhole_angular = angular_epipolar_error_deg(
            pinhole_cam0_rays[common_projection_valid],
            pinhole_cam1_rays[common_projection_valid],
            rotation,
            translation,
        )
        fisheye_angular = angular_epipolar_error_deg(
            fisheye_cam0_rays[common_projection_valid],
            fisheye_cam1_rays[common_projection_valid],
            rotation,
            translation,
        )
        pooled_vertical.append(vertical_error)
        pooled_pinhole_angular.append(pinhole_angular)
        pooled_fisheye_angular.append(fisheye_angular)

        median_vertical = float(np.median(vertical_error))
        p95_vertical = float(np.percentile(vertical_error, 95))
        gate = {
            "median_absolute_vertical_error_below_0_5px": median_vertical < 0.5,
            "p95_absolute_vertical_error_below_1_0px": p95_vertical < 1.0,
            "positive_disparity_ratio_above_0_95": (
                geometry["positive_disparity_ratio"] > 0.95
            ),
        }
        gate["passed"] = all(gate.values())
        raw_input_hypothesis = None
        if calibration["supports_raw_to_rectified_hypothesis"]:
            rectified_cam0, rectified_cam0_valid = rectified_normalized_points(
                fisheye_cam0_rays, cam0_rectification
            )
            rectified_cam1, rectified_cam1_valid = rectified_normalized_points(
                fisheye_cam1_rays, cam1_rectification
            )
            rectified_valid = (
                fisheye_valid & rectified_cam0_valid & rectified_cam1_valid
            )
            rectified_vertical = (
                np.abs(
                    rectified_cam0[rectified_valid, 1]
                    - rectified_cam1[rectified_valid, 1]
                )
                * equivalent_focal_length_px
            )
            stored_vertical_same_valid_subset = vertical_error[
                rectified_valid
            ]
            rectified_disparity = (
                rectified_cam0[rectified_valid, 0]
                - rectified_cam1[rectified_valid, 0]
            ) * equivalent_focal_length_px
            rectified_valid_ratio = float(
                np.mean(rectified_valid)
            )
            rectified_median = float(np.median(rectified_vertical))
            rectified_p95 = float(np.percentile(rectified_vertical, 95))
            rectified_positive_ratio = float(
                np.mean(rectified_disparity > 0.0)
            )
            rectified_gate = {
                "valid_feature_ratio_above_0_95": (
                    rectified_valid_ratio > 0.95
                ),
                "median_absolute_vertical_error_below_0_5px": (
                    rectified_median < 0.5
                ),
                "p95_absolute_vertical_error_below_1_0px": (
                    rectified_p95 < 1.0
                ),
                "positive_disparity_ratio_above_0_95": (
                    rectified_positive_ratio > 0.95
                ),
            }
            rectified_gate["passed"] = all(rectified_gate.values())
            raw_input_hypothesis = {
                "interpretation": (
                    "Stored feature coordinates are treated as raw equidistant "
                    "pixels, unprojected with the external calibration, rotated "
                    "by the stereo-rectification rotations, and scored in a "
                    "shared normalized plane. Pixel values are equivalents at "
                    "the mean supplied focal length; no image remap is applied."
                ),
                "equivalent_focal_length_px": equivalent_focal_length_px,
                "valid_feature_count": int(np.sum(rectified_valid)),
                "valid_feature_ratio": rectified_valid_ratio,
                "absolute_vertical_error_equivalent_px": percentile_summary(
                    rectified_vertical
                ),
                "stored_absolute_vertical_error_same_valid_subset_px": (
                    percentile_summary(stored_vertical_same_valid_subset)
                ),
                "horizontal_disparity_equivalent_px": percentile_summary(
                    rectified_disparity
                ),
                "positive_disparity_ratio": rectified_positive_ratio,
                "gate": rectified_gate,
            }

        frame_report = {
            "frame": frame_index,
            "cam0_path": str(paths["cam0"]),
            "cam1_path": str(paths["cam1"]),
            "stereo_delta_ms": stereo_delta_ms,
            "stereo": {
                "feature_inlier_count": len(cam0_points),
                "absolute_vertical_error_px": geometry[
                    "absolute_vertical_error_px"
                ],
                "signed_vertical_error_px": geometry["vertical_error_px"],
                "horizontal_disparity_px": geometry[
                    "left_minus_right_disparity_px"
                ],
                "positive_disparity_ratio": geometry[
                    "positive_disparity_ratio"
                ],
                "x_bins": geometry["x_bins"],
                "estimated_vertical_affine_model": geometry[
                    "apparent_global_image_alignment_diagnostic"
                ]["robust_vertical_model_y_left_from_x_right_y_right"],
                "post_affine_absolute_vertical_residual_px": geometry[
                    "apparent_global_image_alignment_diagnostic"
                ]["robust_vertical_model_absolute_residual_px"],
            },
            "projection_test": {
                "pinhole_error_deg": percentile_summary(pinhole_angular),
                "kannala_brandt_error_deg": percentile_summary(fisheye_angular),
                "pinhole_valid_feature_ratio": float(
                    np.mean(pinhole_valid)
                ),
                "kannala_brandt_valid_feature_ratio": float(
                    np.mean(fisheye_valid)
                ),
                "common_projection_model_feature_ratio": float(
                    np.mean(common_projection_valid)
                ),
                "lower_median_error_model": (
                    "pinhole"
                    if np.median(pinhole_angular) < np.median(fisheye_angular)
                    else "kannala_brandt"
                ),
                "cam0_kannala_brandt_domain": cam0_model_diagnostics,
                "cam1_kannala_brandt_domain": cam1_model_diagnostics,
            },
            "new_calibration_raw_input_hypothesis": raw_input_hypothesis,
            "rectification_gate": gate,
        }
        frame_reports.append(frame_report)
        if frame_index in requested_visualizations:
            draw_visualization(
                visualization_directory / f"{frame_index:06d}.png",
                images["cam0"],
                images["cam1"],
                cam0_points,
                cam1_points,
                frame_report,
            )
        if (
            frame_index == args.start_frame
            or frame_index == args.end_frame
            or (frame_index - args.start_frame) % 25 == 0
        ):
            print(
                f"{frame_index:06d}: inliers={len(cam0_points)} "
                f"dt={stereo_delta_ms:.3f}ms "
                f"|dy| p50={median_vertical:.3f}px p95={p95_vertical:.3f}px "
                f"positive={geometry['positive_disparity_ratio']:.3f}",
                flush=True,
            )

    aggregate = summarize_frames(frame_reports)
    low_delta_frames = [
        item
        for item in frame_reports
        if item["stereo_delta_ms"] <= args.sync_threshold_ms
    ]
    high_delta_frames = [
        item
        for item in frame_reports
        if item["stereo_delta_ms"] > args.sync_threshold_ms
    ]
    projection_supports_pinhole = (
        aggregate["pinhole_better_frame_ratio"] >= 0.8
        and aggregate["median_pinhole_to_kannala_brandt_error_ratio"] < 0.9
    )
    horizontal_rectification_passed = (
        aggregate["passed_horizontal_rectification_frame_ratio"] >= 0.95
    )
    raw_input_hypothesis_aggregate = aggregate.get(
        "new_calibration_raw_input_hypothesis"
    )
    new_calibration_raw_input_hypothesis_passed = bool(
        raw_input_hypothesis_aggregate is not None
        and raw_input_hypothesis_aggregate["passed_frame_ratio"] >= 0.95
    )
    report = {
        "schema": "daaam.stereo_rectification_range_verification.v1",
        "contract": (
            "Stored PNG pixels are decoded without resize, crop, rotation, or "
            "remap. A general fundamental matrix selects geometric inliers; the "
            "reported residuals are then evaluated against the fixed supplied "
            "calibration."
        ),
        "dataset": str(dataset),
        "requested_frame_range_inclusive": [
            args.start_frame,
            args.end_frame,
        ],
        "input_audit": {
            "frame_count": len(frame_reports),
            "image_shapes_hwc": [list(shape) for shape in sorted(image_shapes)],
            "manifest_layout": manifest.get("layout"),
            "manifest_calibration_source": manifest.get("calibration_source"),
            "camera_mapping": manifest.get("cameras"),
            "calibration": {
                "source_format": calibration["source_format"],
                "source_path": calibration["source_path"],
                "cam0_distortion_model": intrinsics_by_camera["cam0"][
                    "distortion_model"
                ],
                "cam1_distortion_model": intrinsics_by_camera["cam1"][
                    "distortion_model"
                ],
                "cam0_D": cam0_distortion.tolist(),
                "cam1_D": cam1_distortion.tolist(),
                "same_intrinsics": bool(
                    np.allclose(cam0_intrinsics, cam1_intrinsics)
                ),
                "cam0_K": cam0_intrinsics.tolist(),
                "cam1_K": cam1_intrinsics.tolist(),
                "cam0_R": cam0_rectification.tolist(),
                "cam1_R": cam1_rectification.tolist(),
                "cam0_P": cam0_projection.tolist(),
                "cam1_P": cam1_projection.tolist(),
                "stereo_rotation_cam0_from_cam1": rotation.tolist(),
                "stereo_translation_cam0_from_cam1_m": translation.tolist(),
                "input_transform_cam1_from_cam0": calibration[
                    "transform_cam1_from_cam0"
                ].tolist(),
                "input_transform_semantics": (
                    (
                        "Kalibr T_cn_cnm1 maps cam0 points into cam1"
                        if calibration["source_format"]
                        == "kalibr_camchain_equidistant"
                        else (
                            "V2 stereo rotation/translation maps left/cam0 "
                            "points into right/cam1"
                        )
                    )
                    if calibration["source_format"]
                    in {
                        "kalibr_camchain_equidistant",
                        "shitong_fisheye_precomputed_rectification",
                    }
                    else (
                        "Dataset T_cam0_cam1 maps cam1 points into cam0; its "
                        "inverse is also reported as cam1_from_cam0"
                    )
                ),
                "baseline_m": baseline_m,
                "expected_right_projection_tx": (
                    expected_right_projection_tx
                ),
                "observed_right_projection_tx": float(
                    cam1_projection[0, 3]
                ),
                "right_projection_encodes_stereo_baseline": (
                    projection_encodes_baseline
                ),
                "cam0_roi_do_rectify": bool(
                    intrinsics_by_camera["cam0"]["roi"]["do_rectify"]
                ),
                "cam1_roi_do_rectify": bool(
                    intrinsics_by_camera["cam1"]["roi"]["do_rectify"]
                ),
                "raw_to_rectified_hypothesis_available": calibration[
                    "supports_raw_to_rectified_hypothesis"
                ],
                "equivalent_focal_length_px_for_normalized_rectification_test": (
                    equivalent_focal_length_px
                ),
                "source_resolution": calibration.get("source_resolution"),
                "resolution_adaptation": calibration.get(
                    "resolution_adaptation"
                ),
                "provided_reprojection_error_px": calibration.get(
                    "provided_reprojection_error_px"
                ),
            },
        },
        "gates": {
            "per_frame": {
                "median_absolute_vertical_error_below_px": 0.5,
                "p95_absolute_vertical_error_below_px": 1.0,
                "positive_disparity_ratio_above": 0.95,
            },
            "range_pass_rule": "at least 95% of frames pass every per-frame gate",
            "low_time_delta_threshold_ms": args.sync_threshold_ms,
        },
        "aggregate": aggregate,
        "low_time_delta_aggregate": summarize_frames(low_delta_frames),
        "high_time_delta_aggregate": summarize_frames(high_delta_frames),
        "pooled_match_statistics": {
            "absolute_vertical_error_px": percentile_summary(
                np.concatenate(pooled_vertical)
            ),
            "pinhole_angular_epipolar_error_deg": percentile_summary(
                np.concatenate(pooled_pinhole_angular)
            ),
            "kannala_brandt_angular_epipolar_error_deg": percentile_summary(
                np.concatenate(pooled_fisheye_angular)
            ),
        },
        "conclusion": {
            "stored_pixels_are_pinhole_like_and_likely_already_undistorted": (
                projection_supports_pinhole
            ),
            "horizontal_stereo_rectification_passed": (
                horizontal_rectification_passed
            ),
            "camera_order_matches_positive_disparity_contract": (
                aggregate["frame_positive_disparity_ratio"]["p50"] > 0.95
            ),
            "camera_info_encodes_standard_rectified_stereo_projection": (
                projection_encodes_baseline
            ),
            "new_calibration_rectifies_stored_pixels_if_treated_as_raw": (
                new_calibration_raw_input_hypothesis_passed
            ),
            "interpretation": (
                "The projection-model comparison can support, but cannot prove, "
                "which upstream undistortion operation ran without raw pixels or "
                "factory maps. Horizontal rectification is tested directly on "
                "stored PNG pixels. For an external calibration file, a second "
                "test "
                "treats those same pixels as raw equidistant observations and "
                "applies the supplied model to matched coordinates; its validity "
                "coverage is part of the pass gate."
            ),
        },
        "frames": frame_reports,
    }
    (output / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )

    with (output / "per_frame_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame",
                "stereo_delta_ms",
                "feature_inlier_count",
                "median_abs_vertical_error_px",
                "p95_abs_vertical_error_px",
                "positive_disparity_ratio",
                "pinhole_median_angular_error_deg",
                "kb_median_angular_error_deg",
                "new_calibration_valid_feature_ratio",
                "stored_same_valid_subset_median_abs_vertical_error_px",
                "stored_same_valid_subset_p95_abs_vertical_error_px",
                "new_calibration_median_abs_vertical_error_equivalent_px",
                "new_calibration_p95_abs_vertical_error_equivalent_px",
                "new_calibration_positive_disparity_ratio",
                "new_calibration_gate_passed",
                "rectification_gate_passed",
            ]
        )
        for item in frame_reports:
            rectified_hypothesis = item[
                "new_calibration_raw_input_hypothesis"
            ]
            writer.writerow(
                [
                    item["frame"],
                    item["stereo_delta_ms"],
                    item["stereo"]["feature_inlier_count"],
                    item["stereo"]["absolute_vertical_error_px"]["p50"],
                    item["stereo"]["absolute_vertical_error_px"]["p95"],
                    item["stereo"]["positive_disparity_ratio"],
                    item["projection_test"]["pinhole_error_deg"]["p50"],
                    item["projection_test"]["kannala_brandt_error_deg"]["p50"],
                    (
                        rectified_hypothesis["valid_feature_ratio"]
                        if rectified_hypothesis is not None
                        else ""
                    ),
                    (
                        rectified_hypothesis[
                            "stored_absolute_vertical_error_same_valid_subset_px"
                        ]["p50"]
                        if rectified_hypothesis is not None
                        else ""
                    ),
                    (
                        rectified_hypothesis[
                            "stored_absolute_vertical_error_same_valid_subset_px"
                        ]["p95"]
                        if rectified_hypothesis is not None
                        else ""
                    ),
                    (
                        rectified_hypothesis[
                            "absolute_vertical_error_equivalent_px"
                        ]["p50"]
                        if rectified_hypothesis is not None
                        else ""
                    ),
                    (
                        rectified_hypothesis[
                            "absolute_vertical_error_equivalent_px"
                        ]["p95"]
                        if rectified_hypothesis is not None
                        else ""
                    ),
                    (
                        rectified_hypothesis["positive_disparity_ratio"]
                        if rectified_hypothesis is not None
                        else ""
                    ),
                    (
                        rectified_hypothesis["gate"]["passed"]
                        if rectified_hypothesis is not None
                        else ""
                    ),
                    item["rectification_gate"]["passed"],
                ]
            )

    indices = np.asarray([item["frame"] for item in frame_reports])
    delta = np.asarray([item["stereo_delta_ms"] for item in frame_reports])
    median_vertical = np.asarray(
        [
            item["stereo"]["absolute_vertical_error_px"]["p50"]
            for item in frame_reports
        ]
    )
    p95_vertical = np.asarray(
        [
            item["stereo"]["absolute_vertical_error_px"]["p95"]
            for item in frame_reports
        ]
    )
    positive = np.asarray(
        [item["stereo"]["positive_disparity_ratio"] for item in frame_reports]
    )
    pinhole = np.asarray(
        [
            item["projection_test"]["pinhole_error_deg"]["p50"]
            for item in frame_reports
        ]
    )
    fisheye = np.asarray(
        [
            item["projection_test"]["kannala_brandt_error_deg"]["p50"]
            for item in frame_reports
        ]
    )
    figure, axes = plt.subplots(4, 1, figsize=(15, 13), constrained_layout=True)
    axes[0].plot(indices, median_vertical, label="median |dy|", linewidth=1)
    axes[0].plot(indices, p95_vertical, label="p95 |dy|", linewidth=1)
    if calibration["supports_raw_to_rectified_hypothesis"]:
        rectified_median = np.asarray(
            [
                item["new_calibration_raw_input_hypothesis"][
                    "absolute_vertical_error_equivalent_px"
                ]["p50"]
                for item in frame_reports
            ]
        )
        rectified_p95 = np.asarray(
            [
                item["new_calibration_raw_input_hypothesis"][
                    "absolute_vertical_error_equivalent_px"
                ]["p95"]
                for item in frame_reports
            ]
        )
        axes[0].plot(
            indices,
            rectified_median,
            label="new calib median |dy| equivalent",
            linewidth=1,
        )
        axes[0].plot(
            indices,
            rectified_p95,
            label="new calib p95 |dy| equivalent",
            linewidth=1,
        )
    axes[0].axhline(0.5, color="#31a354", linestyle="--", label="median gate")
    axes[0].axhline(1.0, color="#d7301f", linestyle="--", label="p95 gate")
    axes[0].set_ylabel("pixels")
    axes[0].set_title("Horizontal rectification residual")
    axes[0].legend(ncol=4)
    axes[1].plot(indices, positive, color="#756bb1", linewidth=1)
    if calibration["supports_raw_to_rectified_hypothesis"]:
        new_calibration_positive = np.asarray(
            [
                item["new_calibration_raw_input_hypothesis"][
                    "positive_disparity_ratio"
                ]
                for item in frame_reports
            ]
        )
        axes[1].plot(
            indices,
            new_calibration_positive,
            color="#238b45",
            linewidth=1,
            label="new calibration raw-input hypothesis",
        )
    axes[1].axhline(0.95, color="#d7301f", linestyle="--")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("positive disparity ratio")
    axes[1].set_title("Supplied cam0/cam1 order")
    if calibration["supports_raw_to_rectified_hypothesis"]:
        axes[1].legend()
    axes[2].plot(indices, pinhole, label="pinhole hypothesis", linewidth=1)
    axes[2].plot(indices, fisheye, label="Kannala-Brandt hypothesis", linewidth=1)
    axes[2].set_ylabel("degrees")
    axes[2].set_title("Median angular epipolar error: projection-model control")
    axes[2].legend()
    axes[3].scatter(delta, median_vertical, s=10, alpha=0.6)
    axes[3].axvline(args.sync_threshold_ms, color="#d7301f", linestyle="--")
    axes[3].set_xlabel("cam0/cam1 timestamp delta (ms)")
    axes[3].set_ylabel("median |dy| (pixels)")
    axes[3].set_title("Synchronization control")
    for axis in axes[:3]:
        if args.start_frame == args.end_frame:
            axis.set_xlim(args.start_frame - 0.5, args.end_frame + 0.5)
        else:
            axis.set_xlim(args.start_frame, args.end_frame)
        axis.set_xlabel("frame")
        axis.grid(alpha=0.2)
    axes[3].grid(alpha=0.2)
    figure.savefig(output / "verification_summary.png", dpi=180)
    plt.close(figure)

    print(json.dumps(report["conclusion"], ensure_ascii=False, indent=2))
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
