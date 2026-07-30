#!/usr/bin/env python3
"""Audit one stored rectified stereo pair without changing either image.

The script creates two minimal Fast-FoundationStereo datasets:

* ``provided_order`` keeps cam0/cam1 exactly as recorded.
* ``swapped_order_control`` swaps only the model inputs to diagnose whether
  the stored camera labels agree with the positive-disparity convention.

Both datasets reference the original PNG files directly.  No image is copied,
decoded/re-encoded, resized, cropped, rotated, or remapped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    percentiles = np.percentile(values, [0, 10, 25, 50, 75, 90, 95, 99, 100])
    return {
        "count": int(values.size),
        "minimum": float(percentiles[0]),
        "p10": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p90": float(percentiles[5]),
        "p95": float(percentiles[6]),
        "p99": float(percentiles[7]),
        "maximum": float(percentiles[8]),
    }


def match_features(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.02)
    left_keypoints, left_descriptors = sift.detectAndCompute(
        cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None
    )
    right_keypoints, right_descriptors = sift.detectAndCompute(
        cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None
    )
    if left_descriptors is None or right_descriptors is None:
        raise RuntimeError("SIFT could not find descriptors in the stereo pair")
    nearest = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        left_descriptors, right_descriptors, k=2
    )
    ratio_matches = [
        first for first, second in nearest if first.distance < 0.72 * second.distance
    ]
    if len(ratio_matches) < 100:
        raise RuntimeError("Too few stereo feature correspondences")
    left_points = np.float32(
        [left_keypoints[match.queryIdx].pt for match in ratio_matches]
    )
    right_points = np.float32(
        [right_keypoints[match.trainIdx].pt for match in ratio_matches]
    )
    fundamental, mask = cv2.findFundamentalMat(
        left_points,
        right_points,
        cv2.FM_RANSAC,
        1.0,
        0.999,
    )
    if fundamental is None or fundamental.shape != (3, 3) or mask is None:
        raise RuntimeError("Fundamental-matrix estimation failed")
    keep = mask.reshape(-1).astype(bool)
    left_inliers = left_points[keep]
    right_inliers = right_points[keep]
    delta = left_inliers - right_inliers
    partial_affine, partial_mask = cv2.estimateAffinePartial2D(
        right_inliers,
        left_inliers,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=10000,
        confidence=0.999,
        refineIters=20,
    )
    full_affine, full_mask = cv2.estimateAffine2D(
        right_inliers,
        left_inliers,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=10000,
        confidence=0.999,
        refineIters=20,
    )
    if (
        partial_affine is None
        or partial_mask is None
        or full_affine is None
        or full_mask is None
    ):
        raise RuntimeError("Image-scale diagnostic affine estimation failed")
    apparent_scale = float(
        np.hypot(partial_affine[0, 0], partial_affine[0, 1])
    )
    apparent_rotation_deg = float(
        np.degrees(np.arctan2(partial_affine[1, 0], partial_affine[0, 0]))
    )
    affine_singular_values = np.linalg.svd(full_affine[:, :2], compute_uv=False)

    vertical_design = np.column_stack(
        (right_inliers[:, 0], right_inliers[:, 1], np.ones(len(right_inliers)))
    )
    target_y = left_inliers[:, 1]
    vertical_keep = np.ones(len(target_y), dtype=bool)
    for _ in range(8):
        vertical_coefficients = np.linalg.lstsq(
            vertical_design[vertical_keep],
            target_y[vertical_keep],
            rcond=None,
        )[0]
        vertical_residual = vertical_design @ vertical_coefficients - target_y
        median = np.median(vertical_residual[vertical_keep])
        mad = (
            np.median(np.abs(vertical_residual[vertical_keep] - median)) * 1.4826
        )
        vertical_keep = np.abs(vertical_residual - median) < max(0.5, 2.5 * mad)
    vertical_absolute_residual = np.abs(
        (vertical_design @ vertical_coefficients - target_y)[vertical_keep]
    )
    x_bins = []
    width = left.shape[1]
    for lower in range(0, width, width // 4):
        upper = min(width, lower + width // 4)
        in_bin = (left_inliers[:, 0] >= lower) & (left_inliers[:, 0] < upper)
        if not np.any(in_bin):
            continue
        x_bins.append(
            {
                "left_x_range_px": [lower, upper],
                "count": int(in_bin.sum()),
                "median_left_minus_right_x_px": float(
                    np.median(delta[in_bin, 0])
                ),
                "median_absolute_vertical_error_px": float(
                    np.median(np.abs(delta[in_bin, 1]))
                ),
                "positive_disparity_ratio": float(
                    np.mean(delta[in_bin, 0] > 0.0)
                ),
            }
        )
    return {
        "left_keypoint_count": len(left_keypoints),
        "right_keypoint_count": len(right_keypoints),
        "ratio_match_count": len(ratio_matches),
        "fundamental_ransac_inlier_count": int(keep.sum()),
        "fundamental_ransac_inlier_ratio": float(keep.mean()),
        "fundamental_matrix": fundamental.tolist(),
        "left_minus_right_disparity_px": percentile_summary(delta[:, 0]),
        "vertical_error_px": percentile_summary(delta[:, 1]),
        "absolute_vertical_error_px": percentile_summary(np.abs(delta[:, 1])),
        "positive_disparity_ratio": float(np.mean(delta[:, 0] > 0.0)),
        "positive_disparity_within_model_range_ratio": float(
            np.mean((delta[:, 0] > 0.0) & (delta[:, 0] < 416.0))
        ),
        "apparent_global_image_alignment_diagnostic": {
            "warning": (
                "This natural-scene affine fit is diagnostic only; horizontal "
                "translation also contains real scene disparity."
            ),
            "partial_affine_right_to_left": partial_affine.tolist(),
            "partial_affine_inlier_ratio": float(partial_mask.mean()),
            "apparent_uniform_scale": apparent_scale,
            "apparent_rotation_deg": apparent_rotation_deg,
            "full_affine_right_to_left": full_affine.tolist(),
            "full_affine_inlier_ratio": float(full_mask.mean()),
            "full_affine_linear_singular_values": affine_singular_values.tolist(),
            "robust_vertical_model_y_left_from_x_right_y_right": (
                vertical_coefficients.tolist()
            ),
            "robust_vertical_model_inlier_ratio": float(vertical_keep.mean()),
            "robust_vertical_model_absolute_residual_px": percentile_summary(
                vertical_absolute_residual
            ),
        },
        "x_bins": x_bins,
        "_left_inliers": left_inliers,
        "_right_inliers": right_inliers,
    }


def draw_epipolar_audit(
    path: Path,
    left: np.ndarray,
    right: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> None:
    scale = 0.5
    height = int(round(left.shape[0] * scale))
    width = int(round(left.shape[1] * scale))
    left_small = cv2.resize(left, (width, height), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.hstack((left_small, right_small))
    for y in range(40, height, 40):
        cv2.line(canvas, (0, y), (2 * width - 1, y), (0, 255, 0), 1)
    order = np.argsort(left_points[:, 0])
    selected = order[np.linspace(0, max(0, len(order) - 1), 80).astype(np.int32)]
    random = np.random.default_rng(831)
    colors = random.integers(32, 255, size=(len(selected), 3), dtype=np.uint8)
    for index, color in zip(selected, colors):
        left_xy = tuple(np.rint(left_points[index] * scale).astype(int))
        right_xy_array = np.rint(right_points[index] * scale).astype(int)
        right_xy = (int(right_xy_array[0] + width), int(right_xy_array[1]))
        bgr = tuple(int(value) for value in color)
        cv2.circle(canvas, left_xy, 2, bgr, -1, cv2.LINE_AA)
        cv2.circle(canvas, right_xy, 2, bgr, -1, cv2.LINE_AA)
        cv2.line(canvas, left_xy, right_xy, bgr, 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "recorded cam0 / HEAD_LEFT_CAMERA",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "recorded cam1 / HEAD_RIGHT_CAMERA",
        (width + 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Failed to write {path}")


def write_fast_dataset(
    directory: Path,
    *,
    source_dataset: Path,
    source_index: int,
    left_path: Path,
    right_path: Path,
    left_timestamp_ns: int,
    right_timestamp_ns: int,
    intrinsics: np.ndarray,
    baseline: float,
    order: str,
) -> None:
    directory.mkdir()
    height, width = cv2.imread(str(left_path), cv2.IMREAD_COLOR).shape[:2]
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    frame = {
        "idx": 0,
        "source_idx": source_index,
        "cam0_source_idx": source_index,
        "cam1_source_idx": source_index,
        "pose_row": 0,
        "cam0": str(left_path.resolve()),
        "cam1": str(right_path.resolve()),
        "timestamp": 0.0,
        "cam0_sensor_time_ns": left_timestamp_ns,
        "cam1_sensor_time_ns": right_timestamp_ns,
        "sensor_time_ns": left_timestamp_ns,
        "pose_sensor_time_ns": left_timestamp_ns,
        "stereo_delta_ms": abs(left_timestamp_ns - right_timestamp_ns) / 1.0e6,
        "validation_input_order": order,
        "image_geometry_operation": "none",
    }
    tick_index = {
        "source": str(source_dataset.resolve()),
        "projection_model": "pinhole",
        "pose_frame": "map",
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "baseline": baseline,
        "width": width,
        "height": height,
        "recommended_max_depth_m": 5.0,
        "image_policy": "absolute references to original PNGs; no pixel changes",
        "frames": [frame],
    }
    camera_info = {
        "width": width,
        "height": height,
        "model": "pinhole",
        "intrinsics": intrinsics.tolist(),
        "distortion": [0.0, 0.0, 0.0, 0.0],
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "baseline": baseline,
    }
    (directory / "tick_index.json").write_text(
        json.dumps(tick_index, indent=2) + "\n"
    )
    (directory / "camera_info.json").write_text(
        json.dumps(camera_info, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    if args.source_index < 0:
        raise ValueError("source-index must be non-negative")
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {args.output}")
    args.output.mkdir(parents=True)

    records = load_jsonl(dataset / "manifest.jsonl")
    record = next(
        (item for item in records if int(item["tick"]) == args.source_index), None
    )
    if record is None:
        raise IndexError(f"Source index {args.source_index} is absent from the manifest")
    left_descriptor = next(
        item for item in record["images"] if item["camera"] == "cam0"
    )
    right_descriptor = next(
        item for item in record["images"] if item["camera"] == "cam1"
    )
    left_path = dataset / left_descriptor["path"]
    right_path = dataset / right_descriptor["path"]
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None or right is None or left.shape != right.shape:
        raise RuntimeError("Recorded stereo images are missing or have different dimensions")

    calibration_dir = dataset / "calibrations" / "000000"
    left_calibration = yaml.safe_load(
        (calibration_dir / "calib_cam0_intrinsics.yaml").read_text()
    )["intrinsics"]
    right_calibration = yaml.safe_load(
        (calibration_dir / "calib_cam1_intrinsics.yaml").read_text()
    )["intrinsics"]
    extrinsic = yaml.safe_load(
        (calibration_dir / "calib_cam0_to_cam1.yaml").read_text()
    )
    left_k = np.asarray(left_calibration["K"], dtype=np.float64).reshape(3, 3)
    right_k = np.asarray(right_calibration["K"], dtype=np.float64).reshape(3, 3)
    left_r = np.asarray(left_calibration["R"], dtype=np.float64).reshape(3, 3)
    right_r = np.asarray(right_calibration["R"], dtype=np.float64).reshape(3, 3)
    left_p = np.asarray(left_calibration["P"], dtype=np.float64).reshape(3, 4)
    right_p = np.asarray(right_calibration["P"], dtype=np.float64).reshape(3, 4)
    left_t_right = np.asarray(
        extrinsic["transform"]["matrix_4x4"], dtype=np.float64
    )
    baseline = float(np.linalg.norm(left_t_right[:3, 3]))
    left_timestamp_ns = int(left_descriptor["sensor_time_ns"])
    right_timestamp_ns = int(right_descriptor["sensor_time_ns"])

    features = match_features(left, right)
    left_inliers = features.pop("_left_inliers")
    right_inliers = features.pop("_right_inliers")
    draw_epipolar_audit(
        args.output / "epipolar_matches.png",
        left,
        right,
        left_inliers,
        right_inliers,
    )

    expected_right_projection_tx = -float(left_k[0, 0]) * baseline
    report = {
        "contract": "camera-only input audit; no LiDAR used",
        "source_dataset": str(dataset),
        "source_index": args.source_index,
        "recorded_camera_order": {
            "cam0": {
                "sensor": left_descriptor["sensor"],
                "path": str(left_path.resolve()),
                "sensor_time_ns": left_timestamp_ns,
                "sha256": sha256(left_path),
            },
            "cam1": {
                "sensor": right_descriptor["sensor"],
                "path": str(right_path.resolve()),
                "sensor_time_ns": right_timestamp_ns,
                "sha256": sha256(right_path),
            },
        },
        "image_integrity": {
            "shape_hwc": list(left.shape),
            "same_dimensions": left.shape == right.shape,
            "no_resize": True,
            "no_crop": True,
            "no_rotation": True,
            "no_remap": True,
            "no_decode_reencode_for_model_input": True,
        },
        "timing": {
            "left_minus_right_ms": (left_timestamp_ns - right_timestamp_ns) / 1.0e6,
            "absolute_stereo_delta_ms": abs(left_timestamp_ns - right_timestamp_ns)
            / 1.0e6,
        },
        "post_stereo_rectification_calibration": {
            "left_K": left_k.tolist(),
            "right_K": right_k.tolist(),
            "K_matrices_identical": bool(np.allclose(left_k, right_k)),
            "left_D": left_calibration["D"],
            "right_D": right_calibration["D"],
            "left_R": left_r.tolist(),
            "right_R": right_r.tolist(),
            "R_matrices_identity": bool(
                np.allclose(left_r, np.eye(3)) and np.allclose(right_r, np.eye(3))
            ),
            "left_P": left_p.tolist(),
            "right_P": right_p.tolist(),
            "baseline_m": baseline,
            "declared_left_T_right": left_t_right.tolist(),
            "expected_standard_right_P_0_3_for_negative_tx": (
                expected_right_projection_tx
            ),
            "stored_right_P_0_3": float(right_p[0, 3]),
            "right_projection_contains_baseline": bool(
                np.isclose(
                    abs(float(right_p[0, 3])),
                    abs(expected_right_projection_tx),
                    atol=1.0e-6,
                )
            ),
        },
        "provided_order_feature_geometry": features,
        "interpretation": {
            "expected_positive_disparity_definition": "x_cam0 - x_cam1 > 0",
            "feature_geometry_supports_recorded_order": (
                features["positive_disparity_ratio"] > 0.9
            ),
            "warning": (
                None
                if features["positive_disparity_ratio"] > 0.9
                else "Most robust matches have non-positive x_cam0-x_cam1 disparity; "
                "verify physical camera labels, transform sign, and stored image order."
            ),
        },
    }
    (args.output / "input_and_epipolar_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )

    common = {
        "source_dataset": dataset,
        "source_index": args.source_index,
        "left_timestamp_ns": left_timestamp_ns,
        "right_timestamp_ns": right_timestamp_ns,
        "intrinsics": left_k,
        "baseline": baseline,
    }
    write_fast_dataset(
        args.output / "provided_order",
        left_path=left_path,
        right_path=right_path,
        order="recorded cam0 then cam1",
        **common,
    )
    write_fast_dataset(
        args.output / "swapped_order_control",
        left_path=right_path,
        right_path=left_path,
        order="diagnostic control: recorded cam1 then cam0",
        **common,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
