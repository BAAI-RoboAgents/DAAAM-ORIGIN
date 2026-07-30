#!/usr/bin/env python3
"""Find a held-out-validated V1/V2 geometry for already-undistorted G1 stereo.

The stored images are treated as pinhole images with zero distortion.  V1 is
used as the initial relative-pose hypothesis; V2 supplies the metric baseline
and the target rectified projection.  Geometry is fitted only on training
frames and evaluated on a deterministic held-out subset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from audit_rectified_stereo_frame import match_features
from prepare_g1_pinhole_stereo_dataset import normalized_sampson_residuals
from verify_stereo_rectification_range import load_calibration, load_json, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--v1", required=True, type=Path)
    parser.add_argument("--v2", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=653)
    parser.add_argument("--end-frame", type=int, default=953)
    parser.add_argument("--holdout-stride", type=int, default=5)
    parser.add_argument("--max-pose-points", type=int, default=60000)
    parser.add_argument("--max-warp-points", type=int, default=100000)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
        np.asarray(homography, dtype=np.float64),
    ).reshape(-1, 2)


def normalize_points(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    return cv2.undistortPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
        np.asarray(intrinsics, dtype=np.float64),
        None,
    ).reshape(-1, 2)


def deterministic_sample(
    left: np.ndarray,
    right: np.ndarray,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(left) <= maximum:
        return left, right
    indices = np.linspace(0, len(left) - 1, maximum, dtype=np.int64)
    return left[indices], right[indices]


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def essential_from_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    unit_translation = translation / np.linalg.norm(translation)
    return skew(unit_translation) @ rotation


def recover_pose(
    normalized_left: np.ndarray,
    normalized_right: np.ndarray,
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    focal_scale_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate and robustly refine camera1-from-camera0 pose.

    Essential-matrix sign is ambiguous.  The V1 translation direction resolves
    that ambiguity after the data-driven rotation/translation estimate.
    """

    threshold_normalized = 1.5 / focal_scale_px
    essential, mask = cv2.findEssentialMat(
        normalized_left,
        normalized_right,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=threshold_normalized,
        maxIters=10000,
    )
    if essential is None or mask is None:
        raise RuntimeError("Essential-matrix estimation failed")
    essential = essential[:3]
    ransac_count = int(np.count_nonzero(mask))
    _, recovered_rotation, recovered_translation, _ = cv2.recoverPose(
        essential,
        normalized_left,
        normalized_right,
        np.eye(3),
        mask=mask.copy(),
    )
    recovered_translation = recovered_translation.reshape(3)
    initial_unit_translation = initial_translation / np.linalg.norm(
        initial_translation
    )
    if np.dot(recovered_translation, initial_unit_translation) < 0.0:
        recovered_translation *= -1.0

    initial_vector = np.concatenate(
        (
            Rotation.from_matrix(recovered_rotation).as_rotvec(),
            recovered_translation,
        )
    )

    def residuals(values: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_rotvec(values[:3]).as_matrix()
        translation = values[3:]
        if np.linalg.norm(translation) < 1.0e-9:
            translation = initial_unit_translation
        return focal_scale_px * normalized_sampson_residuals(
            rotation,
            translation,
            normalized_left,
            normalized_right,
        )

    result = least_squares(
        residuals,
        initial_vector,
        loss="huber",
        f_scale=1.0,
        max_nfev=80,
    )
    refined_rotation = Rotation.from_rotvec(result.x[:3]).as_matrix()
    refined_translation = result.x[3:]
    refined_translation /= np.linalg.norm(refined_translation)
    if np.dot(refined_translation, initial_unit_translation) < 0.0:
        refined_translation *= -1.0
    residual = np.abs(
        focal_scale_px
        * normalized_sampson_residuals(
            refined_rotation,
            refined_translation,
            normalized_left,
            normalized_right,
        )
    )
    return refined_rotation, refined_translation, {
        "essential_ransac_inliers": ransac_count,
        "essential_ransac_inlier_ratio": ransac_count / len(normalized_left),
        "least_squares_success": bool(result.success),
        "least_squares_cost": float(result.cost),
        "sampson_residual_px_p50": float(np.median(residual)),
        "sampson_residual_px_p95": float(np.percentile(residual, 95.0)),
        "rotation_euler_xyz_deg": Rotation.from_matrix(
            refined_rotation
        ).as_euler("xyz", degrees=True).tolist(),
        "translation_direction": refined_translation.tolist(),
        "v1_initial_rotation_difference_deg": float(
            np.degrees(
                Rotation.from_matrix(
                    refined_rotation @ initial_rotation.T
                ).magnitude()
            )
        ),
    }


def rectification_homographies(
    source_left_K: np.ndarray,
    source_right_K: np.ndarray,
    target_K: np.ndarray,
    rotation_cam1_from_cam0: np.ndarray,
    translation_cam1_from_cam0: np.ndarray,
    resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rectification_result = cv2.stereoRectify(
        source_left_K,
        None,
        source_right_K,
        None,
        resolution,
        rotation_cam1_from_cam0,
        translation_cam1_from_cam0,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
        newImageSize=resolution,
    )
    rectification_left, rectification_right = rectification_result[:2]
    left_homography = (
        target_K @ rectification_left @ np.linalg.inv(source_left_K)
    )
    right_homography = (
        target_K @ rectification_right @ np.linalg.inv(source_right_K)
    )
    left_homography /= left_homography[2, 2]
    right_homography /= right_homography[2, 2]
    return (
        left_homography,
        right_homography,
        rectification_left,
        rectification_right,
    )


def fit_vertical_models(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, dict[str, Any]]:
    design = np.column_stack((right[:, 0], right[:, 1], np.ones(len(right))))
    affine_initial = np.linalg.lstsq(design, left[:, 1], rcond=None)[0]
    affine_result = least_squares(
        lambda values: design @ values - left[:, 1],
        affine_initial,
        loss="huber",
        f_scale=1.0,
        max_nfev=80,
    )
    a, b, c = affine_result.x
    projective_initial = np.asarray([a, b, c, 0.0, 0.0])

    def projective_residual(values: np.ndarray) -> np.ndarray:
        denominator = values[3] * right[:, 0] + values[4] * right[:, 1] + 1.0
        predicted = (
            values[0] * right[:, 0]
            + values[1] * right[:, 1]
            + values[2]
        ) / denominator
        return predicted - left[:, 1]

    projective_result = least_squares(
        projective_residual,
        projective_initial,
        loss="huber",
        f_scale=1.0,
        max_nfev=80,
    )
    return {
        "none": {"model": "none"},
        "affine": {
            "model": "right_y_affine_x_preserving",
            "coefficients_a_b_c": affine_result.x.tolist(),
            "success": bool(affine_result.success),
        },
        "projective": {
            "model": "right_y_projective_x_preserving",
            "coefficients_a_b_c_g_h": projective_result.x.tolist(),
            "success": bool(projective_result.success),
        },
    }


def apply_vertical_model(
    right: np.ndarray,
    model: dict[str, Any],
) -> np.ndarray:
    corrected = np.asarray(right, dtype=np.float64).copy()
    if model["model"] == "none":
        return corrected
    if model["model"] == "right_y_affine_x_preserving":
        a, b, c = model["coefficients_a_b_c"]
        corrected[:, 1] = a * right[:, 0] + b * right[:, 1] + c
        return corrected
    if model["model"] == "right_y_projective_x_preserving":
        a, b, c, g, h = model["coefficients_a_b_c_g_h"]
        denominator = g * right[:, 0] + h * right[:, 1] + 1.0
        corrected[:, 1] = (
            a * right[:, 0] + b * right[:, 1] + c
        ) / denominator
        return corrected
    raise ValueError(f"Unsupported vertical model: {model['model']}")


def evaluate_frames(
    frames: list[dict[str, Any]],
    left_homography: np.ndarray,
    right_homography: np.ndarray,
    vertical_model: dict[str, Any],
    width: int,
    height: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    pooled_vertical: list[np.ndarray] = []
    pooled_disparity: list[np.ndarray] = []
    for frame in frames:
        left = transform_points(frame["left"], left_homography)
        right = transform_points(frame["right"], right_homography)
        right = apply_vertical_model(right, vertical_model)
        finite = np.all(np.isfinite(left), axis=1) & np.all(
            np.isfinite(right), axis=1
        )
        inside = (
            finite
            & (left[:, 0] >= 0.0)
            & (left[:, 0] < width)
            & (left[:, 1] >= 0.0)
            & (left[:, 1] < height)
            & (right[:, 0] >= 0.0)
            & (right[:, 0] < width)
            & (right[:, 1] >= 0.0)
            & (right[:, 1] < height)
        )
        vertical = np.abs(left[inside, 1] - right[inside, 1])
        disparity = left[inside, 0] - right[inside, 0]
        if not len(vertical):
            reports.append(
                {
                    "frame": frame["frame"],
                    "subset": frame["subset"],
                    "matches": len(left),
                    "valid_matches": 0,
                    "valid_ratio": 0.0,
                    "absolute_vertical_p50_px": float("inf"),
                    "absolute_vertical_p95_px": float("inf"),
                    "positive_disparity_ratio": 0.0,
                    "stereo_delta_ms": frame["stereo_delta_ms"],
                    "strict_pass": False,
                }
            )
            continue
        report = {
            "frame": frame["frame"],
            "subset": frame["subset"],
            "matches": len(left),
            "valid_matches": int(np.count_nonzero(inside)),
            "valid_ratio": float(np.mean(inside)),
            "absolute_vertical_p50_px": float(np.median(vertical)),
            "absolute_vertical_p95_px": float(np.percentile(vertical, 95.0)),
            "positive_disparity_ratio": float(np.mean(disparity > 0.0)),
            "stereo_delta_ms": frame["stereo_delta_ms"],
        }
        report["strict_pass"] = bool(
            report["absolute_vertical_p50_px"] <= 1.0
            and report["absolute_vertical_p95_px"] <= 3.0
            and report["positive_disparity_ratio"] >= 0.95
            and report["valid_ratio"] >= 0.90
        )
        reports.append(report)
        pooled_vertical.append(vertical)
        pooled_disparity.append(disparity)

    if not pooled_vertical:
        return (
            {
                "frames": len(reports),
                "feature_matches": int(
                    sum(report["matches"] for report in reports)
                ),
                "valid_feature_matches": 0,
                "strict_pass_frames": 0,
                "strict_pass_ratio": 0.0,
                "frame_median_absolute_vertical_error_px_p50": float("inf"),
                "frame_p95_absolute_vertical_error_px_p50": float("inf"),
                "frame_positive_disparity_ratio_p50": 0.0,
                "frame_valid_match_ratio_p50": 0.0,
                "pooled_absolute_vertical_error_px": {
                    "p50": float("inf"),
                    "p90": float("inf"),
                    "p95": float("inf"),
                    "p99": float("inf"),
                },
                "pooled_positive_disparity_ratio": 0.0,
                "selection_score": float("inf"),
            },
            reports,
        )
    vertical_all = np.concatenate(pooled_vertical)
    disparity_all = np.concatenate(pooled_disparity)
    median_by_frame = np.asarray(
        [report["absolute_vertical_p50_px"] for report in reports]
    )
    p95_by_frame = np.asarray(
        [report["absolute_vertical_p95_px"] for report in reports]
    )
    positive_by_frame = np.asarray(
        [report["positive_disparity_ratio"] for report in reports]
    )
    valid_by_frame = np.asarray([report["valid_ratio"] for report in reports])
    summary = {
        "frames": len(reports),
        "feature_matches": int(sum(report["matches"] for report in reports)),
        "valid_feature_matches": int(
            sum(report["valid_matches"] for report in reports)
        ),
        "strict_pass_frames": int(
            sum(bool(report["strict_pass"]) for report in reports)
        ),
        "strict_pass_ratio": float(
            np.mean([report["strict_pass"] for report in reports])
        ),
        "frame_median_absolute_vertical_error_px_p50": float(
            np.median(median_by_frame)
        ),
        "frame_p95_absolute_vertical_error_px_p50": float(
            np.median(p95_by_frame)
        ),
        "frame_positive_disparity_ratio_p50": float(
            np.median(positive_by_frame)
        ),
        "frame_valid_match_ratio_p50": float(np.median(valid_by_frame)),
        "pooled_absolute_vertical_error_px": {
            "p50": float(np.median(vertical_all)),
            "p90": float(np.percentile(vertical_all, 90.0)),
            "p95": float(np.percentile(vertical_all, 95.0)),
            "p99": float(np.percentile(vertical_all, 99.0)),
        },
        "pooled_positive_disparity_ratio": float(
            np.mean(disparity_all > 0.0)
        ),
    }
    summary["selection_score"] = float(
        summary["frame_median_absolute_vertical_error_px_p50"]
        + summary["frame_p95_absolute_vertical_error_px_p50"]
        + 10.0
        * max(0.0, 0.95 - summary["frame_positive_disparity_ratio_p50"])
        + 10.0 * max(0.0, 0.90 - summary["frame_valid_match_ratio_p50"])
    )
    return summary, reports


def inverse_y_map(
    width: int,
    height: int,
    model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    if model["model"] == "none":
        return x, y
    if model["model"] == "right_y_affine_x_preserving":
        a, b, c = model["coefficients_a_b_c"]
        source_y = (y - a * x - c) / b
        return x, source_y.astype(np.float32)
    a, b, c, g, h = model["coefficients_a_b_c_g_h"]
    denominator = y * h - b
    source_y = (a * x + c - y * (g * x + 1.0)) / denominator
    return x, source_y.astype(np.float32)


def image_valid_area(
    width: int,
    height: int,
    left_homography: np.ndarray,
    right_homography: np.ndarray,
    vertical_model: dict[str, Any],
) -> dict[str, float]:
    source_mask = np.full((height, width), 255, dtype=np.uint8)
    left_mask = cv2.warpPerspective(
        source_mask,
        left_homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    right_mask = cv2.warpPerspective(
        source_mask,
        right_homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    map_x, map_y = inverse_y_map(width, height, vertical_model)
    right_mask = cv2.remap(
        right_mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    left_valid = left_mask > 0
    right_valid = right_mask > 0
    return {
        "left_valid_area_ratio": float(np.mean(left_valid)),
        "right_valid_area_ratio": float(np.mean(right_valid)),
        "joint_valid_area_ratio": float(np.mean(left_valid & right_valid)),
    }


def write_preview(
    path: Path,
    left_path: Path,
    right_path: Path,
    left_homography: np.ndarray,
    right_homography: np.ndarray,
    vertical_model: dict[str, Any],
    width: int,
    height: int,
) -> None:
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise RuntimeError("Could not load preview images")
    left = cv2.warpPerspective(left, left_homography, (width, height))
    right = cv2.warpPerspective(right, right_homography, (width, height))
    map_x, map_y = inverse_y_map(width, height, vertical_model)
    right = cv2.remap(
        right,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    scale = 0.5
    left = cv2.resize(left, None, fx=scale, fy=scale)
    right = cv2.resize(right, None, fx=scale, fy=scale)
    canvas = np.hstack((left, right))
    for y in range(20, canvas.shape[0], 40):
        cv2.line(
            canvas,
            (0, y),
            (canvas.shape[1] - 1, y),
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write preview: {path}")


def make_q(intrinsics: np.ndarray, baseline: float) -> np.ndarray:
    focal = float(intrinsics[0, 0])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    return np.asarray(
        [
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 0.0, focal],
            [0.0, 0.0, 1.0 / baseline, 0.0],
        ]
    )


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    feature_cache_path = output / "feature_matches.npz"
    if output.exists() and any(
        path.name != feature_cache_path.name for path in output.iterdir()
    ):
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if args.holdout_stride < 2:
        raise ValueError("holdout-stride must be at least 2")
    manifest = load_json(dataset / "manifest.json")
    records = {
        int(record["tick"]): record
        for record in load_jsonl(dataset / "manifest.jsonl")
    }
    frame_indices = list(range(args.start_frame, args.end_frame + 1))
    missing = [frame for frame in frame_indices if frame not in records]
    if missing:
        raise ValueError(f"Frames absent from manifest: {missing}")

    first_descriptors = {
        item["camera"]: item for item in records[frame_indices[0]]["images"]
    }
    first_image = cv2.imread(
        str(dataset / first_descriptors["cam0"]["path"]),
        cv2.IMREAD_COLOR,
    )
    if first_image is None:
        raise RuntimeError("Could not load first image")
    height, width = first_image.shape[:2]
    resolution = (width, height)
    v1 = load_calibration(
        dataset, manifest, args.v1.resolve(), [width, height]
    )
    v2 = load_calibration(
        dataset, manifest, args.v2.resolve(), [width, height]
    )
    native = load_calibration(dataset, manifest, None, [width, height])
    baseline = float(
        np.linalg.norm(v2["transform_cam1_from_cam0"][:3, 3])
    )
    target_K = np.asarray(v2["cam0_P"], dtype=np.float64)[:, :3]

    output.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    if feature_cache_path.exists():
        cache = np.load(feature_cache_path, allow_pickle=False)
        cached_frames = cache["frame_indices"].astype(np.int64)
        if not np.array_equal(cached_frames, np.asarray(frame_indices)):
            raise ValueError("Feature cache frame range does not match")
        offsets = cache["offsets"].astype(np.int64)
        cached_left = cache["left"].astype(np.float64)
        cached_right = cache["right"].astype(np.float64)
        print(f"loading feature cache: {feature_cache_path}", flush=True)
    else:
        cached_left_parts: list[np.ndarray] = []
        cached_right_parts: list[np.ndarray] = []
        offsets_list = [0]
        for position, frame_index in enumerate(frame_indices):
            descriptors = {
                item["camera"]: item for item in records[frame_index]["images"]
            }
            left_path = (dataset / descriptors["cam0"]["path"]).resolve()
            right_path = (dataset / descriptors["cam1"]["path"]).resolve()
            left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
            right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
            if left_image is None or right_image is None:
                raise RuntimeError(f"Could not decode frame {frame_index}")
            geometry = match_features(left_image, right_image)
            cached_left_parts.append(
                geometry["_left_inliers"].astype(np.float32)
            )
            cached_right_parts.append(
                geometry["_right_inliers"].astype(np.float32)
            )
            offsets_list.append(
                offsets_list[-1] + len(geometry["_left_inliers"])
            )
            if position % 25 == 0 or position == len(frame_indices) - 1:
                print(
                    f"features {position + 1}/{len(frame_indices)} "
                    f"frame={frame_index} "
                    f"inliers={len(geometry['_left_inliers'])}",
                    flush=True,
                )
        cached_left = np.concatenate(cached_left_parts)
        cached_right = np.concatenate(cached_right_parts)
        offsets = np.asarray(offsets_list, dtype=np.int64)
        np.savez_compressed(
            feature_cache_path,
            frame_indices=np.asarray(frame_indices, dtype=np.int64),
            offsets=offsets,
            left=cached_left,
            right=cached_right,
        )
        print(f"wrote feature cache: {feature_cache_path}", flush=True)

    for position, frame_index in enumerate(frame_indices):
        descriptors = {
            item["camera"]: item for item in records[frame_index]["images"]
        }
        frames.append(
            {
                "frame": frame_index,
                "subset": (
                    "holdout"
                    if position % args.holdout_stride == 0
                    else "train"
                ),
                "left": cached_left[offsets[position] : offsets[position + 1]],
                "right": cached_right[offsets[position] : offsets[position + 1]],
                "left_path": (
                    dataset / descriptors["cam0"]["path"]
                ).resolve(),
                "right_path": (
                    dataset / descriptors["cam1"]["path"]
                ).resolve(),
                "stereo_delta_ms": abs(
                    int(descriptors["cam0"]["sensor_time_ns"])
                    - int(descriptors["cam1"]["sensor_time_ns"])
                )
                / 1.0e6,
            }
        )

    train_frames = [frame for frame in frames if frame["subset"] == "train"]
    holdout_frames = [frame for frame in frames if frame["subset"] == "holdout"]
    train_left = np.concatenate([frame["left"] for frame in train_frames])
    train_right = np.concatenate([frame["right"] for frame in train_frames])
    pose_left, pose_right = deterministic_sample(
        train_left, train_right, args.max_pose_points
    )

    native_left_K = np.asarray(native["cam0_K"], dtype=np.float64)
    native_right_K = np.asarray(native["cam1_K"], dtype=np.float64)
    v1_left_K = np.asarray(v1["cam0_K"], dtype=np.float64)
    v1_right_K = np.asarray(v1["cam1_K"], dtype=np.float64)
    v2_raw_left_K = np.asarray(v2["cam0_K"], dtype=np.float64)
    v2_raw_right_K = np.asarray(v2["cam1_K"], dtype=np.float64)
    source_candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    for coefficient in np.linspace(0.0, 1.0, 9):
        source_candidates.append(
            (
                f"native_to_v1_{coefficient:.3f}",
                (1.0 - coefficient) * native_left_K
                + coefficient * v1_left_K,
                (1.0 - coefficient) * native_right_K
                + coefficient * v1_right_K,
            )
        )
    for coefficient in np.linspace(0.125, 1.0, 8):
        source_candidates.append(
            (
                f"native_to_v2_raw_{coefficient:.3f}",
                (1.0 - coefficient) * native_left_K
                + coefficient * v2_raw_left_K,
                (1.0 - coefficient) * native_right_K
                + coefficient * v2_raw_right_K,
            )
        )

    v1_rotation = np.asarray(
        v1["transform_cam1_from_cam0"][:3, :3], dtype=np.float64
    )
    v1_translation = np.asarray(
        v1["transform_cam1_from_cam0"][:3, 3], dtype=np.float64
    )
    v2_rotation = np.asarray(
        v2["transform_cam1_from_cam0"][:3, :3], dtype=np.float64
    )
    v2_translation = np.asarray(
        v2["transform_cam1_from_cam0"][:3, 3], dtype=np.float64
    )

    candidate_records: list[dict[str, Any]] = []
    candidate_payloads: dict[str, dict[str, Any]] = {}
    for candidate_index, (name, left_K, right_K) in enumerate(source_candidates):
        normalized_left = normalize_points(pose_left, left_K)
        normalized_right = normalize_points(pose_right, right_K)
        focal_scale = float(
            np.mean(
                [left_K[0, 0], left_K[1, 1], right_K[0, 0], right_K[1, 1]]
            )
        )
        rotation, translation_direction, pose_diagnostics = recover_pose(
            normalized_left,
            normalized_right,
            v1_rotation,
            v1_translation,
            focal_scale,
        )
        translation = translation_direction * baseline
        left_H, right_H, rectification_left, rectification_right = (
            rectification_homographies(
                left_K,
                right_K,
                target_K,
                rotation,
                translation,
                resolution,
            )
        )
        transformed_train_left = transform_points(train_left, left_H)
        transformed_train_right = transform_points(train_right, right_H)
        warp_left, warp_right = deterministic_sample(
            transformed_train_left,
            transformed_train_right,
            args.max_warp_points,
        )
        vertical_models = fit_vertical_models(warp_left, warp_right)
        for model_name, vertical_model in vertical_models.items():
            train_summary, _ = evaluate_frames(
                train_frames,
                left_H,
                right_H,
                vertical_model,
                width,
                height,
            )
            record = {
                "candidate": name,
                "vertical_model": model_name,
                **train_summary,
            }
            candidate_records.append(record)
            payload_name = f"{name}__{model_name}"
            candidate_payloads[payload_name] = {
                "name": name,
                "source_left_K": left_K,
                "source_right_K": right_K,
                "rotation": rotation,
                "translation": translation,
                "pose_diagnostics": pose_diagnostics,
                "left_H": left_H,
                "right_H": right_H,
                "rectification_left": rectification_left,
                "rectification_right": rectification_right,
                "vertical_model": vertical_model,
                "train_summary": train_summary,
            }
        print(
            f"candidate {candidate_index + 1}/{len(source_candidates)} "
            f"{name}",
            flush=True,
        )

    # Explicit file-only controls establish what V1 and V2 achieve before the
    # data-driven combination.
    fixed_controls = [
        ("v1_fixed", v1_left_K, v1_right_K, v1_rotation, v1_translation),
        ("v2_fixed", v2_raw_left_K, v2_raw_right_K, v2_rotation, v2_translation),
    ]
    for name, left_K, right_K, rotation, translation in fixed_controls:
        translation = translation / np.linalg.norm(translation) * baseline
        left_H, right_H, rectification_left, rectification_right = (
            rectification_homographies(
                left_K,
                right_K,
                target_K,
                rotation,
                translation,
                resolution,
            )
        )
        transformed_train_left = transform_points(train_left, left_H)
        transformed_train_right = transform_points(train_right, right_H)
        warp_left, warp_right = deterministic_sample(
            transformed_train_left,
            transformed_train_right,
            args.max_warp_points,
        )
        vertical_models = fit_vertical_models(warp_left, warp_right)
        for model_name, vertical_model in vertical_models.items():
            train_summary, _ = evaluate_frames(
                train_frames,
                left_H,
                right_H,
                vertical_model,
                width,
                height,
            )
            record = {
                "candidate": name,
                "vertical_model": model_name,
                **train_summary,
            }
            candidate_records.append(record)
            candidate_payloads[f"{name}__{model_name}"] = {
                "name": name,
                "source_left_K": left_K,
                "source_right_K": right_K,
                "rotation": rotation,
                "translation": translation,
                "pose_diagnostics": {"mode": "fixed_file_control"},
                "left_H": left_H,
                "right_H": right_H,
                "rectification_left": rectification_left,
                "rectification_right": rectification_right,
                "vertical_model": vertical_model,
                "train_summary": train_summary,
            }

    eligible_records = [
        record
        for record in candidate_records
        if record["pooled_positive_disparity_ratio"] >= 0.95
        and record["frame_valid_match_ratio_p50"] >= 0.95
    ]
    if not eligible_records:
        raise RuntimeError("No candidate satisfies positive-disparity/validity gates")
    best_record = min(
        eligible_records,
        key=lambda record: (
            -record["strict_pass_ratio"],
            record["selection_score"],
        ),
    )
    best_key = (
        f"{best_record['candidate']}__{best_record['vertical_model']}"
    )
    best = candidate_payloads[best_key]
    holdout_summary, holdout_reports = evaluate_frames(
        holdout_frames,
        best["left_H"],
        best["right_H"],
        best["vertical_model"],
        width,
        height,
    )
    all_summary, all_reports = evaluate_frames(
        frames,
        best["left_H"],
        best["right_H"],
        best["vertical_model"],
        width,
        height,
    )
    valid_area = image_valid_area(
        width,
        height,
        best["left_H"],
        best["right_H"],
        best["vertical_model"],
    )

    projection_left = np.column_stack((target_K, np.zeros(3)))
    projection_right = projection_left.copy()
    projection_right[0, 3] = -target_K[0, 0] * baseline
    q_matrix = make_q(target_K, baseline)
    result = {
        "experiment": {
            "dataset": str(dataset),
            "frame_range_inclusive": [args.start_frame, args.end_frame],
            "cam0_semantics": "left",
            "input_pixel_model": "already_monocular_undistorted_pinhole_D_zero",
            "training_split": (
                f"(frame_position % {args.holdout_stride}) != 0"
            ),
            "holdout_split": (
                f"(frame_position % {args.holdout_stride}) == 0"
            ),
            "training_frames": len(train_frames),
            "holdout_frames": len(holdout_frames),
            "v1_path": str(args.v1.resolve()),
            "v2_path": str(args.v2.resolve()),
        },
        "combination_semantics": {
            "v1_role": "initial effective camera1-from-camera0 pose",
            "v2_role": (
                "metric baseline, target rectified K, P1/P2/Q depth scale"
            ),
            "image_evidence_role": (
                "refine effective pose and fit optional x-preserving right-y "
                "residual using training frames only"
            ),
            "not_used": "fisheye distortion is not applied again",
        },
        "selected_candidate": {
            "name": best["name"],
            "vertical_model": best["vertical_model"],
            "source_left_K": best["source_left_K"].tolist(),
            "source_right_K": best["source_right_K"].tolist(),
            "rotation_cam1_from_cam0": best["rotation"].tolist(),
            "rotation_euler_xyz_deg": Rotation.from_matrix(
                best["rotation"]
            ).as_euler("xyz", degrees=True).tolist(),
            "translation_cam1_from_cam0_m": best["translation"].tolist(),
            "baseline_m": baseline,
            "pose_diagnostics": best["pose_diagnostics"],
            "source_to_rectified_left_H": best["left_H"].tolist(),
            "source_to_rectified_right_base_H": best["right_H"].tolist(),
            "rectification_left_R": best["rectification_left"].tolist(),
            "rectification_right_R": best["rectification_right"].tolist(),
        },
        "depth_projection": {
            "K": target_K.tolist(),
            "P1": projection_left.tolist(),
            "P2": projection_right.tolist(),
            "Q": q_matrix.tolist(),
            "disparity_convention": "d = x_left - x_right",
            "depth_formula": "Z_m = K[0,0] * baseline_m / disparity_px",
            "fb_px_m": float(target_K[0, 0] * baseline),
        },
        "training_metrics": best["train_summary"],
        "holdout_metrics": holdout_summary,
        "all_653_953_metrics": all_summary,
        "rectified_image_valid_area": valid_area,
        "acceptance": {
            "criterion": (
                "held-out frame p50 median |dy| <= 1 px, p50 P95 |dy| "
                "<= 3 px, pooled positive disparity >= 0.95, strict-pass "
                "frames >= 0.90, joint valid image area >= 0.75"
            ),
            "passed": bool(
                holdout_summary[
                    "frame_median_absolute_vertical_error_px_p50"
                ]
                <= 1.0
                and holdout_summary[
                    "frame_p95_absolute_vertical_error_px_p50"
                ]
                <= 3.0
                and holdout_summary[
                    "pooled_positive_disparity_ratio"
                ]
                >= 0.95
                and holdout_summary["strict_pass_ratio"] >= 0.90
                and valid_area["joint_valid_area_ratio"] >= 0.75
            ),
        },
    }
    (output / "best_combination.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    with (output / "candidate_comparison.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidate_records[0]))
        writer.writeheader()
        writer.writerows(candidate_records)
    with (output / "best_frame_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_reports[0]))
        writer.writeheader()
        writer.writerows(all_reports)

    preview_frames = [
        frames[0],
        frames[len(frames) // 2],
        frames[-1],
    ]
    for frame in preview_frames:
        write_preview(
            output / f"rectified_preview_{frame['frame']:06d}.jpg",
            frame["left_path"],
            frame["right_path"],
            best["left_H"],
            best["right_H"],
            best["vertical_model"],
            width,
            height,
        )
    print(json.dumps(result["selected_candidate"], indent=2))
    print(json.dumps(result["holdout_metrics"], indent=2))
    print(json.dumps(result["acceptance"], indent=2))


if __name__ == "__main__":
    main()
