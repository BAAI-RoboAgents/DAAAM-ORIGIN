#!/usr/bin/env python3
"""Fit a right-only rectification warp using held-out LiDAR geometry.

The original left image is the immutable reference.  Training frames associate
mutual SIFT stereo matches with motion-compensated LiDAR depths in the left
image.  For a horizontal rectified baseline, a left pixel with camera-frame
depth Z must have:

    u_right = u_left - fx * baseline / Z
    v_right = v_left

A robust homography maps stored right-image pixels to those target pixels.
The requested holdout frame is excluded from fitting and emitted as a
single-frame dataset for independent stereo-depth and LiDAR-fusion validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from fuse_lidar_camera_frame import (
    auxiliary_pose_samples,
    interpolate_pose,
    load_json,
    load_jsonl,
    map_pose_samples,
    transform_points,
    z_buffer_projection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", required=True, type=Path)
    parser.add_argument("--camera-info", required=True, type=Path)
    parser.add_argument("--holdout-source-index", required=True, type=int)
    parser.add_argument("--source-start-index", type=int)
    parser.add_argument("--source-end-index", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--training-frames", type=int, default=36)
    parser.add_argument("--maximum-camera-lidar-skew-ms", type=float, default=8.0)
    parser.add_argument("--holdout-exclusion-radius", type=int, default=40)
    parser.add_argument("--lidar-association-radius-px", type=float, default=3.0)
    parser.add_argument("--minimum-lidar-depth-m", type=float, default=0.30)
    parser.add_argument("--maximum-lidar-depth-m", type=float, default=8.0)
    parser.add_argument("--ransac-threshold-px", type=float, default=2.5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_dict(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    return {
        f"p{percentile:02d}": float(np.percentile(values, percentile))
        for percentile in (25, 50, 75, 90, 95)
    }


def stereo_descriptors(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    sift: Any,
    matcher: Any,
) -> tuple[np.ndarray, np.ndarray]:
    left_keypoints, left_descriptors = sift.detectAndCompute(left_gray, None)
    right_keypoints, right_descriptors = sift.detectAndCompute(right_gray, None)
    if left_descriptors is None or right_descriptors is None:
        return np.empty((0, 2)), np.empty((0, 2))
    left_to_right = matcher.knnMatch(
        left_descriptors, right_descriptors, k=2
    )
    right_to_left = matcher.knnMatch(
        right_descriptors, left_descriptors, k=2
    )
    accepted_left = {
        match.queryIdx: match
        for match, second in left_to_right
        if match.distance < 0.72 * second.distance
    }
    accepted_right = {
        match.queryIdx: match
        for match, second in right_to_left
        if match.distance < 0.72 * second.distance
    }
    mutual = [
        match
        for query_index, match in accepted_left.items()
        if match.trainIdx in accepted_right
        and accepted_right[match.trainIdx].trainIdx == query_index
    ]
    left_pixels = np.asarray(
        [left_keypoints[match.queryIdx].pt for match in mutual],
        dtype=np.float64,
    ).reshape(-1, 2)
    right_pixels = np.asarray(
        [right_keypoints[match.trainIdx].pt for match in mutual],
        dtype=np.float64,
    ).reshape(-1, 2)
    return left_pixels, right_pixels


def transform_for_frame(
    record: dict[str, Any],
    map_samples: tuple[np.ndarray, np.ndarray, np.ndarray],
    camera_samples: tuple[np.ndarray, np.ndarray, np.ndarray],
    lidar_samples: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    left_descriptor = next(
        image for image in record["images"] if image["camera"] == "cam0"
    )
    right_descriptor = next(
        image for image in record["images"] if image["camera"] == "cam1"
    )
    lidar_descriptor = next(
        lidar for lidar in record["lidar"] if lidar["lidar"] == "lidar0"
    )
    camera_timestamp = int(left_descriptor["sensor_time_ns"])
    lidar_timestamp = int(lidar_descriptor["sensor_time_ns"])
    map_T_base_camera, camera_map_clamped = interpolate_pose(
        *map_samples, camera_timestamp
    )
    map_T_base_lidar, lidar_map_clamped = interpolate_pose(
        *map_samples, lidar_timestamp
    )
    base_T_camera, camera_aux_clamped = interpolate_pose(
        *camera_samples, camera_timestamp
    )
    base_T_lidar, lidar_aux_clamped = interpolate_pose(
        *lidar_samples, lidar_timestamp
    )
    if any(
        (
            camera_map_clamped,
            lidar_map_clamped,
            camera_aux_clamped,
            lidar_aux_clamped,
        )
    ):
        raise ValueError("Pose interpolation was clamped")
    camera_T_lidar = np.linalg.inv(
        map_T_base_camera @ base_T_camera
    ) @ (map_T_base_lidar @ base_T_lidar)
    return (
        camera_T_lidar,
        left_descriptor,
        right_descriptor,
        lidar_descriptor,
    )


def lidar_projection_for_frame(
    raw_dataset: Path,
    record: dict[str, Any],
    map_samples: tuple[np.ndarray, np.ndarray, np.ndarray],
    camera_samples: tuple[np.ndarray, np.ndarray, np.ndarray],
    lidar_samples: tuple[np.ndarray, np.ndarray, np.ndarray],
    camera: dict[str, float | int],
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> tuple[
    dict[str, np.ndarray], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    (
        camera_T_lidar,
        left_descriptor,
        right_descriptor,
        lidar_descriptor,
    ) = transform_for_frame(record, map_samples, camera_samples, lidar_samples)
    points = np.load(
        raw_dataset / lidar_descriptor["path"], allow_pickle=False
    ).astype(np.float64)
    valid = np.isfinite(points).all(axis=1) & (
        np.linalg.norm(points, axis=1) > 0.01
    )
    camera_points = transform_points(camera_T_lidar, points[valid])
    projection = z_buffer_projection(
        camera_points,
        float(camera["fx"]),
        float(camera["fy"]),
        float(camera["cx"]),
        float(camera["cy"]),
        int(camera["width"]),
        int(camera["height"]),
        minimum_depth_m,
        maximum_depth_m,
    )
    return (
        projection,
        left_descriptor,
        right_descriptor,
        lidar_descriptor,
    )


def collect_lidar_guided_matches(
    raw_dataset: Path,
    records: list[dict[str, Any]],
    frame_indices: list[int],
    map_samples: tuple[np.ndarray, np.ndarray, np.ndarray],
    camera_samples: tuple[np.ndarray, np.ndarray, np.ndarray],
    lidar_samples: tuple[np.ndarray, np.ndarray, np.ndarray],
    camera: dict[str, float | int],
    sift: Any,
    matcher: Any,
    association_radius_px: float,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    source_right_pixels: list[np.ndarray] = []
    target_rectified_pixels: list[np.ndarray] = []
    frame_reports: list[dict[str, Any]] = []
    fx = float(camera["fx"])
    baseline = float(camera["baseline"])
    width = int(camera["width"])
    height = int(camera["height"])
    for position, frame_index in enumerate(frame_indices, start=1):
        record = records[frame_index]
        (
            projection,
            left_descriptor,
            right_descriptor,
            _,
        ) = lidar_projection_for_frame(
            raw_dataset,
            record,
            map_samples,
            camera_samples,
            lidar_samples,
            camera,
            minimum_depth_m,
            maximum_depth_m,
        )
        left_gray = cv2.imread(
            str(raw_dataset / left_descriptor["path"]), cv2.IMREAD_GRAYSCALE
        )
        right_gray = cv2.imread(
            str(raw_dataset / right_descriptor["path"]), cv2.IMREAD_GRAYSCALE
        )
        if left_gray is None or right_gray is None:
            raise FileNotFoundError(f"Could not read training frame {frame_index}")
        left_pixels, right_pixels = stereo_descriptors(
            left_gray, right_gray, sift, matcher
        )
        if projection["depth"].size == 0 or left_pixels.size == 0:
            continue
        interpolated_depth, distances, interpolation_residuals = (
            interpolate_lidar_depth(
                projection,
                left_pixels,
                fx_baseline=fx * baseline,
                nearest_radius_px=association_radius_px,
            )
        )
        plausible = (
            np.isfinite(interpolated_depth)
            & (np.abs(left_pixels[:, 0] - right_pixels[:, 0]) < 250.0)
            & (np.abs(left_pixels[:, 1] - right_pixels[:, 1]) < 50.0)
        )
        selected_left = left_pixels[plausible]
        selected_right = right_pixels[plausible]
        selected_depth = interpolated_depth[plausible]
        expected = np.column_stack(
            (
                selected_left[:, 0] - fx * baseline / selected_depth,
                selected_left[:, 1],
            )
        )
        inside = (
            (expected[:, 0] >= 0.0)
            & (expected[:, 0] <= width - 1)
            & (expected[:, 1] >= 0.0)
            & (expected[:, 1] <= height - 1)
        )
        selected_right = selected_right[inside]
        expected = expected[inside]
        selected_distances = distances[plausible][inside]
        selected_residuals = interpolation_residuals[plausible][inside]
        source_right_pixels.append(selected_right)
        target_rectified_pixels.append(expected)
        frame_reports.append(
            {
                "source_index": frame_index,
                "stereo_mutual_matches": int(left_pixels.shape[0]),
                "lidar_associations": int(selected_right.shape[0]),
                "association_distance_px": percentile_dict(
                    selected_distances
                ),
                "local_inverse_depth_plane_residual_disparity_px": (
                    percentile_dict(selected_residuals)
                ),
                "camera_minus_lidar_ms": float(
                    (
                        int(left_descriptor["sensor_time_ns"])
                        - int(record["lidar"][0]["sensor_time_ns"])
                    )
                    / 1e6
                ),
            }
        )
        print(
            f"training {position}/{len(frame_indices)} source={frame_index} "
            f"matches={left_pixels.shape[0]} associations={selected_right.shape[0]}",
            flush=True,
        )
    if not source_right_pixels:
        raise RuntimeError("No LiDAR-guided stereo associations were found")
    return (
        np.vstack(source_right_pixels),
        np.vstack(target_rectified_pixels),
        frame_reports,
    )


def interpolate_lidar_depth(
    projection: dict[str, np.ndarray],
    query_pixels: np.ndarray,
    *,
    fx_baseline: float,
    nearest_radius_px: float,
    neighborhood_radius_px: float = 14.0,
    neighbor_count: int = 16,
    minimum_neighbors: int = 6,
    maximum_p90_residual_disparity_px: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate depth with a local planar inverse-depth model.

    Inverse depth is affine in image coordinates for a 3-D plane.  This avoids
    assigning a nearby floor scan point's depth directly to a feature several
    pixels away, while rejecting mixed-depth neighborhoods at object edges.
    """

    coordinates = np.column_stack((projection["u"], projection["v"])).astype(
        np.float64
    )
    depths = projection["depth"].astype(np.float64)
    result = np.full(query_pixels.shape[0], np.nan, dtype=np.float64)
    nearest = np.full(query_pixels.shape[0], np.inf, dtype=np.float64)
    plane_residual = np.full(query_pixels.shape[0], np.nan, dtype=np.float64)
    if coordinates.shape[0] < minimum_neighbors:
        return result, nearest, plane_residual
    count = min(neighbor_count, coordinates.shape[0])
    tree = cKDTree(coordinates)
    distances, indices = tree.query(query_pixels, k=count)
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    for query_index, query in enumerate(query_pixels):
        valid = (
            np.isfinite(distances[query_index])
            & (distances[query_index] <= neighborhood_radius_px)
            & (indices[query_index] < coordinates.shape[0])
        )
        if int(np.count_nonzero(valid)) < minimum_neighbors:
            continue
        local_distances = distances[query_index, valid]
        nearest[query_index] = float(local_distances.min())
        if nearest[query_index] > nearest_radius_px:
            continue
        local_indices = indices[query_index, valid]
        offsets = coordinates[local_indices] - query
        singular_values = np.linalg.svd(offsets, compute_uv=False)
        if (
            singular_values.size < 2
            or singular_values[0] <= 1.0e-9
            or singular_values[1] / singular_values[0] < 0.08
        ):
            continue
        design = np.column_stack(
            (offsets[:, 0], offsets[:, 1], np.ones(offsets.shape[0]))
        )
        inverse_depth = 1.0 / depths[local_indices]
        weights = 1.0 / (1.0 + np.square(local_distances / 4.0))
        weighted_design = design * np.sqrt(weights[:, None])
        weighted_target = inverse_depth * np.sqrt(weights)
        coefficients, _, rank, _ = np.linalg.lstsq(
            weighted_design, weighted_target, rcond=None
        )
        if rank < 3:
            continue
        residual_disparity = (
            fx_baseline * np.abs(design @ coefficients - inverse_depth)
        )
        robust_limit = max(
            0.35, 3.0 * float(np.median(residual_disparity))
        )
        inliers = residual_disparity <= robust_limit
        if int(np.count_nonzero(inliers)) >= minimum_neighbors:
            inlier_weights = weights[inliers]
            coefficients, _, rank, _ = np.linalg.lstsq(
                design[inliers] * np.sqrt(inlier_weights[:, None]),
                inverse_depth[inliers] * np.sqrt(inlier_weights),
                rcond=None,
            )
            if rank < 3:
                continue
            residual_disparity = fx_baseline * np.abs(
                design[inliers] @ coefficients - inverse_depth[inliers]
            )
        p90_residual = float(np.percentile(residual_disparity, 90))
        predicted_inverse_depth = float(coefficients[2])
        if (
            p90_residual > maximum_p90_residual_disparity_px
            or not np.isfinite(predicted_inverse_depth)
            or predicted_inverse_depth <= 0.0
        ):
            continue
        result[query_index] = 1.0 / predicted_inverse_depth
        plane_residual[query_index] = p90_residual
    return result, nearest, plane_residual


def target_error(
    homography: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    predicted = cv2.perspectiveTransform(
        source.astype(np.float32).reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    delta = predicted - target
    norm = np.linalg.norm(delta, axis=1)
    return {
        "count": int(source.shape[0]),
        "euclidean_error_px": percentile_dict(norm),
        "absolute_horizontal_error_px": percentile_dict(np.abs(delta[:, 0])),
        "absolute_vertical_error_px": percentile_dict(np.abs(delta[:, 1])),
        "horizontal_signed_median_px": float(np.median(delta[:, 0])),
        "vertical_signed_median_px": float(np.median(delta[:, 1])),
        "within_1px_ratio": float((norm <= 1.0).mean()),
        "within_2px_ratio": float((norm <= 2.0).mean()),
        "within_3px_ratio": float((norm <= 3.0).mean()),
    }


def rectified_feature_geometry(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    sift: Any,
    matcher: Any,
) -> dict[str, Any]:
    left, right = stereo_descriptors(left_gray, right_gray, sift, matcher)
    if left.size == 0:
        return {"matches": 0}
    disparity = left[:, 0] - right[:, 0]
    vertical = left[:, 1] - right[:, 1]
    plausible = (np.abs(disparity) < 250.0) & (np.abs(vertical) < 100.0)
    disparity = disparity[plausible]
    vertical = vertical[plausible]
    return {
        "matches": int(disparity.size),
        "absolute_vertical_error_px": percentile_dict(np.abs(vertical)),
        "vertical_within_1px_ratio": float((np.abs(vertical) <= 1.0).mean()),
        "vertical_within_2px_ratio": float((np.abs(vertical) <= 2.0).mean()),
        "disparity_px": percentile_dict(disparity),
        "positive_disparity_ratio": float((disparity > 0.0).mean()),
    }


def save_epipolar_pair(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    combined = np.hstack((left, right))
    width = left.shape[1]
    for y in range(40, left.shape[0], 80):
        cv2.line(
            combined,
            (0, y),
            (combined.shape[1] - 1, y),
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            combined,
            str(y),
            (8, max(18, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            combined,
            str(y),
            (width + 8, max(18, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), combined)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {args.output}")
    if args.training_frames < 8:
        raise ValueError("At least eight training frames are required")
    if (
        (args.source_start_index is None) != (args.source_end_index is None)
        or (
            args.source_start_index is not None
            and (
                args.source_start_index < 0
                or args.source_end_index < args.source_start_index
            )
        )
    ):
        raise ValueError("source start/end indices must be a valid paired range")
    records = load_jsonl(args.raw_dataset / "manifest.jsonl")
    if not 0 <= args.holdout_source_index < len(records):
        raise IndexError("holdout-source-index is outside the manifest")
    camera_info = load_json(args.camera_info)
    camera = {
        key: camera_info[key]
        for key in ("width", "height", "fx", "fy", "cx", "cy", "baseline")
    }
    map_samples = map_pose_samples(
        load_jsonl(
            args.raw_dataset / "state" / "000000" / "map_pose.jsonl"
        )
    )
    auxiliary_records = load_jsonl(
        args.raw_dataset
        / "poses"
        / "dense_global"
        / "000000"
        / "aux_poses.jsonl"
    )
    camera_samples = auxiliary_pose_samples(auxiliary_records, "head_camera")
    lidar_samples = auxiliary_pose_samples(auxiliary_records, "lidar")

    eligible = []
    for index, record in enumerate(records):
        if (
            args.source_start_index is not None
            and not args.source_start_index <= index <= args.source_end_index
        ):
            continue
        skew = (
            record.get("sync", {})
            .get("relative_skew_ms", {})
            .get("camera:HEAD_LEFT_CAMERA")
        )
        if skew is None or abs(float(skew)) > args.maximum_camera_lidar_skew_ms:
            continue
        if (
            abs(index - args.holdout_source_index)
            <= args.holdout_exclusion_radius
        ):
            continue
        eligible.append(index)
    if len(eligible) < args.training_frames:
        raise RuntimeError(
            f"Only {len(eligible)} eligible training frames are available"
        )
    selection_positions = np.linspace(
        0, len(eligible) - 1, args.training_frames
    ).astype(int)
    training_indices = [eligible[position] for position in selection_positions]

    cv2.setRNGSeed(0)
    sift = cv2.SIFT_create(nfeatures=6500, contrastThreshold=0.015)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    training_source, training_target, training_reports = (
        collect_lidar_guided_matches(
            args.raw_dataset,
            records,
            training_indices,
            map_samples,
            camera_samples,
            lidar_samples,
            camera,
            sift,
            matcher,
            args.lidar_association_radius_px,
            args.minimum_lidar_depth_m,
            args.maximum_lidar_depth_m,
        )
    )
    homography, ransac_mask = cv2.findHomography(
        training_source.astype(np.float32),
        training_target.astype(np.float32),
        cv2.RANSAC,
        args.ransac_threshold_px,
        maxIters=20000,
        confidence=0.999,
    )
    if homography is None or ransac_mask is None:
        raise RuntimeError("LiDAR-guided right-image homography fitting failed")
    homography = homography.astype(np.float64)
    homography /= homography[2, 2]
    determinant = float(np.linalg.det(homography))
    if not np.isfinite(homography).all() or abs(determinant) < 1.0e-6:
        raise RuntimeError("Fitted homography is invalid")

    holdout_record = records[args.holdout_source_index]
    (
        holdout_projection,
        left_descriptor,
        right_descriptor,
        lidar_descriptor,
    ) = lidar_projection_for_frame(
        args.raw_dataset,
        holdout_record,
        map_samples,
        camera_samples,
        lidar_samples,
        camera,
        args.minimum_lidar_depth_m,
        args.maximum_lidar_depth_m,
    )
    left_path = args.raw_dataset / left_descriptor["path"]
    right_path = args.raw_dataset / right_descriptor["path"]
    left_bgr = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left_bgr is None or right_bgr is None:
        raise FileNotFoundError("Could not read holdout stereo pair")
    warped_right = cv2.warpPerspective(
        right_bgr,
        homography,
        (int(camera["width"]), int(camera["height"])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid_warp = cv2.warpPerspective(
        np.full(right_bgr.shape[:2], 255, dtype=np.uint8),
        homography,
        (int(camera["width"]), int(camera["height"])),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )

    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    raw_right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    warped_right_gray = cv2.cvtColor(warped_right, cv2.COLOR_BGR2GRAY)
    holdout_left, holdout_raw_right = stereo_descriptors(
        left_gray, raw_right_gray, sift, matcher
    )
    holdout_depths, distances, holdout_plane_residuals = (
        interpolate_lidar_depth(
            holdout_projection,
            holdout_left,
            fx_baseline=float(camera["fx"]) * float(camera["baseline"]),
            nearest_radius_px=args.lidar_association_radius_px,
        )
    )
    usable = (
        np.isfinite(holdout_depths)
        & (np.abs(holdout_left[:, 0] - holdout_raw_right[:, 0]) < 250.0)
        & (np.abs(holdout_left[:, 1] - holdout_raw_right[:, 1]) < 50.0)
    )
    holdout_source = holdout_raw_right[usable]
    holdout_depth = holdout_depths[usable]
    holdout_target = np.column_stack(
        (
            holdout_left[usable, 0]
            - float(camera["fx"]) * float(camera["baseline"]) / holdout_depth,
            holdout_left[usable, 1],
        )
    )

    args.output.mkdir(parents=True)
    rgb_directory = args.output / "rgb"
    right_directory = args.output / "stereo_right"
    rgb_directory.mkdir()
    right_directory.mkdir()
    output_left = rgb_directory / "00000000.png"
    output_right = right_directory / "00000000.png"
    shutil.copyfile(left_path, output_left)
    if not cv2.imwrite(str(output_right), warped_right):
        raise RuntimeError("Could not write the corrected right image")
    save_epipolar_pair(
        args.output / "raw_epipolar_pair.png", left_bgr, right_bgr
    )
    save_epipolar_pair(
        args.output / "corrected_epipolar_pair.png", left_bgr, warped_right
    )

    output_camera_info = dict(camera_info)
    (args.output / "camera_info.json").write_text(
        json.dumps(output_camera_info, ensure_ascii=False, indent=2) + "\n"
    )
    frame = {
        "idx": 0,
        "source_idx": args.holdout_source_index,
        "cam0_source_idx": args.holdout_source_index,
        "cam1_source_idx": args.holdout_source_index,
        "pose_row": 0,
        "cam0": str(output_left.resolve()),
        "cam1": str(output_right.resolve()),
        "timestamp": 0.0,
        "cam0_sensor_time_ns": int(left_descriptor["sensor_time_ns"]),
        "cam1_sensor_time_ns": int(right_descriptor["sensor_time_ns"]),
        "sensor_time_ns": int(left_descriptor["sensor_time_ns"]),
        "pose_sensor_time_ns": int(left_descriptor["sensor_time_ns"]),
        "stereo_delta_ms": float(
            (
                int(right_descriptor["sensor_time_ns"])
                - int(left_descriptor["sensor_time_ns"])
            )
            / 1e6
        ),
    }
    tick_index = {
        "source": str(args.raw_dataset.resolve()),
        "projection_model": "pinhole",
        "pose_frame": "map",
        "fx": float(camera["fx"]),
        "fy": float(camera["fy"]),
        "cx": float(camera["cx"]),
        "cy": float(camera["cy"]),
        "baseline": float(camera["baseline"]),
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "recommended_max_depth_m": 5.0,
        "frames": [frame],
    }
    (args.output / "tick_index.json").write_text(
        json.dumps(tick_index, ensure_ascii=False, indent=2) + "\n"
    )

    report = {
        "method": "lidar_guided_right_only_projective_rectification",
        "source_dataset": str(args.raw_dataset.resolve()),
        "camera_geometry": {
            "width": int(camera["width"]),
            "height": int(camera["height"]),
            "fx": float(camera["fx"]),
            "fy": float(camera["fy"]),
            "cx": float(camera["cx"]),
            "cy": float(camera["cy"]),
            "baseline": float(camera["baseline"]),
        },
        "left_image_policy": "byte-for-byte original left PNG",
        "extra_left_rotation_applied": False,
        "holdout_source_index": args.holdout_source_index,
        "holdout_exclusion_radius": args.holdout_exclusion_radius,
        "source_index_range": [
            args.source_start_index,
            args.source_end_index,
        ],
        "training_indices": training_indices,
        "training": {
            "eligible_frames": len(eligible),
            "selected_frames": len(training_indices),
            "associations": int(training_source.shape[0]),
            "ransac_inliers": int(np.count_nonzero(ransac_mask)),
            "ransac_inlier_ratio": float(np.mean(ransac_mask != 0)),
            "all_association_error": target_error(
                homography, training_source, training_target
            ),
            "frames": training_reports,
        },
        "right_source_to_rectified_homography": homography.tolist(),
        "homography_determinant": determinant,
        "right_warp_valid_ratio": float(np.mean(valid_warp != 0)),
        "holdout": {
            "raw_left": str(left_path.resolve()),
            "raw_right": str(right_path.resolve()),
            "raw_lidar": str(
                (args.raw_dataset / lidar_descriptor["path"]).resolve()
            ),
            "camera_minus_lidar_ms": float(
                (
                    int(left_descriptor["sensor_time_ns"])
                    - int(lidar_descriptor["sensor_time_ns"])
                )
                / 1e6
            ),
            "lidar_guided_associations": int(holdout_source.shape[0]),
            "lidar_association_distance_px": percentile_dict(
                distances[usable]
            ),
            "local_inverse_depth_plane_residual_disparity_px": (
                percentile_dict(holdout_plane_residuals[usable])
            ),
            "identity_target_error": target_error(
                np.eye(3), holdout_source, holdout_target
            ),
            "corrected_target_error": target_error(
                homography, holdout_source, holdout_target
            ),
            "raw_feature_geometry": rectified_feature_geometry(
                left_gray, raw_right_gray, sift, matcher
            ),
            "corrected_feature_geometry": rectified_feature_geometry(
                left_gray, warped_right_gray, sift, matcher
            ),
        },
        "integrity": {
            "raw_left_sha256": sha256(left_path),
            "output_left_sha256": sha256(output_left),
            "left_byte_identical": sha256(left_path) == sha256(output_left),
        },
        "caveats": [
            "The stored LiDAR NPY contains XYZ only, so per-point deskew is unavailable.",
            "Nearest sparse LiDAR/image associations are noisy near occlusion and depth boundaries.",
            "The homography must be accepted only if held-out stereo depth also improves against LiDAR.",
        ],
    }
    (args.output / "lidar_guided_rectification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
