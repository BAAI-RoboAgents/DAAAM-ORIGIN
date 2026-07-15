#!/usr/bin/env python3
"""Select time-aligned stereo keyframes without dropping visual observations.

The selector only removes frames that are strict duplicates of the last accepted
frame. A visual change is sufficient to retain a frame even when the camera pose
does not change, which preserves stationary observations of new content.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SelectionConfig:
    analysis_width: int = 320
    analysis_height: int = 240
    soft_translation_m: float = 0.06
    soft_rotation_deg: float = 5.0
    hard_translation_m: float = 0.15
    hard_rotation_deg: float = 12.0
    max_gap_s: float = 1.5
    ratio_test: float = 0.75
    min_orb_matches: int = 50
    min_orb_inliers: int = 40
    min_inlier_ratio: float = 0.75
    max_duplicate_flow_px: float = 3.0
    max_duplicate_mean_lab_delta: float = 8.0
    local_change_lab_delta: float = 18.0
    local_change_component_ratio: float = 0.003
    local_change_aggregate_ratio: float = 0.01
    salient_change_lab_delta: float = 35.0
    salient_change_component_ratio: float = 0.0005
    feature_cluster_min_count: int = 5
    max_feature_match_distance: int = 32


@dataclass(frozen=True)
class SourceFrame:
    index: int
    frame: dict[str, Any]
    pose_row: int
    sensor_time_ns: int
    cam0_time_ns: int
    cam1_time_ns: int
    pose_time_ns: int
    cam0_path: Path
    cam1_path: Path


@dataclass
class VisualFrame:
    lab: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select time-aligned mapping keyframes while retaining all visible "
            "content changes at a stationary pose."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--analysis-width", type=int, default=320)
    parser.add_argument("--analysis-height", type=int, default=240)
    parser.add_argument("--soft-translation-m", type=float, default=0.06)
    parser.add_argument("--soft-rotation-deg", type=float, default=5.0)
    parser.add_argument("--hard-translation-m", type=float, default=0.15)
    parser.add_argument("--hard-rotation-deg", type=float, default=12.0)
    parser.add_argument("--max-gap-s", type=float, default=1.5)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--min-orb-matches", type=int, default=50)
    parser.add_argument("--min-orb-inliers", type=int, default=40)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.75)
    parser.add_argument("--max-duplicate-flow-px", type=float, default=3.0)
    parser.add_argument("--max-duplicate-mean-lab-delta", type=float, default=8.0)
    parser.add_argument("--local-change-lab-delta", type=float, default=18.0)
    parser.add_argument("--local-change-component-ratio", type=float, default=0.003)
    parser.add_argument("--local-change-aggregate-ratio", type=float, default=0.01)
    parser.add_argument("--salient-change-lab-delta", type=float, default=35.0)
    parser.add_argument("--salient-change-component-ratio", type=float, default=0.0005)
    parser.add_argument("--feature-cluster-min-count", type=int, default=5)
    parser.add_argument("--max-feature-match-distance", type=int, default=32)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze the dataset and print the selection report without writing output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory created by this selector.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> SelectionConfig:
    config = SelectionConfig(
        analysis_width=args.analysis_width,
        analysis_height=args.analysis_height,
        soft_translation_m=args.soft_translation_m,
        soft_rotation_deg=args.soft_rotation_deg,
        hard_translation_m=args.hard_translation_m,
        hard_rotation_deg=args.hard_rotation_deg,
        max_gap_s=args.max_gap_s,
        ratio_test=args.ratio_test,
        min_orb_matches=args.min_orb_matches,
        min_orb_inliers=args.min_orb_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        max_duplicate_flow_px=args.max_duplicate_flow_px,
        max_duplicate_mean_lab_delta=args.max_duplicate_mean_lab_delta,
        local_change_lab_delta=args.local_change_lab_delta,
        local_change_component_ratio=args.local_change_component_ratio,
        local_change_aggregate_ratio=args.local_change_aggregate_ratio,
        salient_change_lab_delta=args.salient_change_lab_delta,
        salient_change_component_ratio=args.salient_change_component_ratio,
        feature_cluster_min_count=args.feature_cluster_min_count,
        max_feature_match_distance=args.max_feature_match_distance,
    )
    if config.analysis_width < 32 or config.analysis_height < 32:
        raise ValueError("Analysis image dimensions must be at least 32 pixels")
    if not 0.0 < config.ratio_test < 1.0:
        raise ValueError("--ratio-test must be in (0, 1)")
    if (
        config.soft_translation_m <= 0.0
        or config.soft_rotation_deg <= 0.0
        or config.hard_translation_m < config.soft_translation_m
        or config.hard_rotation_deg < config.soft_rotation_deg
        or config.max_gap_s <= 0.0
    ):
        raise ValueError("Pose thresholds and --max-gap-s must be positive and ordered")
    if (
        config.min_orb_matches < 2
        or config.min_orb_inliers < 2
        or not 0.0 < config.min_inlier_ratio <= 1.0
        or config.max_duplicate_flow_px < 0.0
        or config.max_duplicate_mean_lab_delta < 0.0
        or config.local_change_lab_delta <= 0.0
        or not 0.0 < config.local_change_component_ratio <= 1.0
        or not 0.0 < config.local_change_aggregate_ratio <= 1.0
        or config.salient_change_lab_delta < config.local_change_lab_delta
        or not 0.0 < config.salient_change_component_ratio <= 1.0
        or config.feature_cluster_min_count < 1
        or config.max_feature_match_distance < 0
    ):
        raise ValueError("Invalid visual duplicate thresholds")
    return config


def load_pose_timestamps(path: Path) -> np.ndarray:
    if not path.exists():
        raise ValueError(
            f"Missing {path}. Rerun preprocessing to create the absolute pose time index."
        )
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError(f"No pose timestamps in {path}")
    try:
        timestamps = np.asarray([int(value) for value in values], dtype=np.int64)
    except ValueError as error:
        raise ValueError(f"Invalid integer pose timestamp in {path}") from error
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"Pose timestamps must be strictly increasing in {path}")
    return timestamps


def load_poses(path: Path) -> np.ndarray:
    if not path.exists():
        raise ValueError(f"Missing pose file: {path}")
    poses = np.loadtxt(path, dtype=np.float64).reshape(-1, 4, 4)
    if not np.all(np.isfinite(poses)) or not np.allclose(
        poses[:, 3, :], [0.0, 0.0, 0.0, 1.0]
    ):
        raise ValueError(f"Expected finite homogeneous poses in {path}")
    return poses


def resolve_frame_path(dataset: Path, value: Any, field: str, index: int) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Frame {index} has no {field} path")
    path = Path(value)
    path = path if path.is_absolute() else dataset / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Frame {index} {field} image is missing: {path}")
    return path


def require_int(frame: dict[str, Any], key: str, index: int) -> int:
    if key not in frame:
        raise ValueError(f"Frame {index} is missing required time field {key!r}")
    try:
        return int(frame[key])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Frame {index} has invalid {key!r}: {frame[key]!r}") from error


def validate_source_dataset(
    dataset: Path,
) -> tuple[dict[str, Any], list[SourceFrame], np.ndarray, np.ndarray]:
    tick_path = dataset / "tick_index.json"
    if not tick_path.exists():
        raise ValueError(f"Missing frame index: {tick_path}")
    metadata = json.loads(tick_path.read_text())
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("tick_index.json must contain at least one frame")
    if "time_origin_ns" not in metadata:
        raise ValueError(
            "tick_index.json is missing time_origin_ns. Rerun preprocessing before selection."
        )
    try:
        time_origin_ns = int(metadata["time_origin_ns"])
    except (TypeError, ValueError) as error:
        raise ValueError("time_origin_ns must be an integer nanosecond timestamp") from error

    poses = load_poses(dataset / "pose" / "poses.txt")
    pose_timestamps = load_pose_timestamps(dataset / "pose" / "pose_timestamps_ns.txt")
    if len(poses) != len(pose_timestamps):
        raise ValueError("poses.txt and pose_timestamps_ns.txt have different lengths")
    if len(frames) != len(poses):
        raise ValueError("Frame, pose, and pose timestamp counts must agree")

    source_frames: list[SourceFrame] = []
    used_pose_rows: set[int] = set()
    previous_sensor_time: int | None = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Frame {index} metadata must be an object")
        if int(frame.get("idx", -1)) != index:
            raise ValueError(f"Frame index mismatch at position {index}: {frame.get('idx')}")
        if "source_idx" not in frame and "source_frame_idx" not in frame:
            raise ValueError(f"Frame {index} is missing source image provenance")
        pose_row = require_int(frame, "pose_row", index)
        if pose_row < 0 or pose_row >= len(poses) or pose_row in used_pose_rows:
            raise ValueError(f"Frame {index} has invalid or duplicate pose_row {pose_row}")
        used_pose_rows.add(pose_row)

        sensor_time_ns = require_int(frame, "sensor_time_ns", index)
        cam0_time_ns = require_int(frame, "cam0_sensor_time_ns", index)
        cam1_time_ns = require_int(frame, "cam1_sensor_time_ns", index)
        pose_time_ns = require_int(frame, "pose_sensor_time_ns", index)
        if sensor_time_ns != cam0_time_ns or pose_time_ns != cam0_time_ns:
            raise ValueError(
                f"Frame {index} image/pose timestamps do not share cam0 absolute time"
            )
        if int(pose_timestamps[pose_row]) != pose_time_ns:
            raise ValueError(
                f"Frame {index} pose_row {pose_row} does not match pose timestamp index"
            )
        if previous_sensor_time is not None and sensor_time_ns <= previous_sensor_time:
            raise ValueError("Frame sensor_time_ns values must be strictly increasing")
        previous_sensor_time = sensor_time_ns

        expected_timestamp = (sensor_time_ns - time_origin_ns) / 1.0e9
        if "timestamp" not in frame or not math.isclose(
            float(frame["timestamp"]), expected_timestamp, abs_tol=1.0e-6
        ):
            raise ValueError(
                f"Frame {index} timestamp is not derived from its absolute sensor time"
            )
        expected_stereo_delta_ms = abs(cam0_time_ns - cam1_time_ns) / 1.0e6
        if "stereo_delta_ms" in frame and not math.isclose(
            float(frame["stereo_delta_ms"]), expected_stereo_delta_ms, abs_tol=1.0e-6
        ):
            raise ValueError(f"Frame {index} stereo_delta_ms disagrees with absolute times")

        source_frames.append(
            SourceFrame(
                index=index,
                frame=frame,
                pose_row=pose_row,
                sensor_time_ns=sensor_time_ns,
                cam0_time_ns=cam0_time_ns,
                cam1_time_ns=cam1_time_ns,
                pose_time_ns=pose_time_ns,
                cam0_path=resolve_frame_path(dataset, frame.get("cam0"), "cam0", index),
                cam1_path=resolve_frame_path(dataset, frame.get("cam1"), "cam1", index),
            )
        )
    return metadata, source_frames, poses, pose_timestamps


def make_visual_frame(path: Path, config: SelectionConfig, orb: cv2.ORB) -> VisualFrame:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image for visual selection: {path}")
    image = cv2.resize(
        image,
        (config.analysis_width, config.analysis_height),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return VisualFrame(
        lab=cv2.cvtColor(image, cv2.COLOR_BGR2LAB),
        keypoints=keypoints,
        descriptors=descriptors,
    )


def pose_rotation_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def clustered_feature_points(
    points: list[tuple[float, float]],
    width: int,
    height: int,
    min_count: int,
) -> tuple[bool, int]:
    if len(points) < min_count:
        return False, len(points)
    grid_width = 4
    grid_height = 3
    counts = np.zeros((grid_height, grid_width), dtype=np.int32)
    for x, y in points:
        col = min(grid_width - 1, max(0, int(x * grid_width / width)))
        row = min(grid_height - 1, max(0, int(y * grid_height / height)))
        counts[row, col] += 1
    return bool(counts.max() >= min_count), len(points)


def points_inside_mask(
    points: list[tuple[float, float]], mask: np.ndarray
) -> list[tuple[float, float]]:
    height, width = mask.shape
    return [
        (x, y)
        for x, y in points
        if 0 <= int(round(x)) < width
        and 0 <= int(round(y)) < height
        and mask[int(round(y)), int(round(x))]
    ]


def compare_visual_frames(
    reference: VisualFrame,
    candidate: VisualFrame,
    config: SelectionConfig,
    matcher: cv2.BFMatcher,
) -> dict[str, Any]:
    base_metrics: dict[str, Any] = {
        "status": "uncertain",
        "strict_duplicate": False,
        "content_event": True,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "median_inlier_flow_px": None,
        "mean_lab_delta": None,
        "local_change_ratio": None,
        "largest_change_component_ratio": None,
        "largest_salient_change_component_ratio": None,
        "reference_unmatched_feature_count": None,
        "candidate_unmatched_feature_count": None,
        "reference_changed_feature_count": None,
        "candidate_changed_feature_count": None,
        "feature_cluster_event": False,
    }
    if (
        reference.descriptors is None
        or candidate.descriptors is None
        or len(reference.descriptors) < 2
        or len(candidate.descriptors) < 2
    ):
        base_metrics["status"] = "insufficient_features"
        return base_metrics

    pairs = matcher.knnMatch(reference.descriptors, candidate.descriptors, k=2)
    good = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < config.ratio_test * second.distance
    ]
    base_metrics["good_matches"] = len(good)
    if len(good) < config.min_orb_inliers:
        base_metrics["status"] = "insufficient_matches"
        return base_metrics

    reference_points = np.float32(
        [reference.keypoints[match.queryIdx].pt for match in good]
    ).reshape(-1, 1, 2)
    candidate_points = np.float32(
        [candidate.keypoints[match.trainIdx].pt for match in good]
    ).reshape(-1, 1, 2)
    affine, mask = cv2.estimateAffinePartial2D(
        reference_points,
        candidate_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=1000,
        confidence=0.99,
        refineIters=10,
    )
    if affine is None or mask is None:
        base_metrics["status"] = "affine_failed"
        return base_metrics

    inlier_mask = mask.reshape(-1).astype(bool)
    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / len(good)
    base_metrics["inliers"] = inliers
    base_metrics["inlier_ratio"] = float(inlier_ratio)
    if inliers == 0:
        base_metrics["status"] = "affine_no_inliers"
        return base_metrics

    flows = np.linalg.norm(
        reference_points.reshape(-1, 2)[inlier_mask]
        - candidate_points.reshape(-1, 2)[inlier_mask],
        axis=1,
    )
    median_flow = float(np.median(flows))
    base_metrics["median_inlier_flow_px"] = median_flow

    inverse_affine = cv2.invertAffineTransform(affine)
    height, width = reference.lab.shape[:2]
    aligned_candidate = cv2.warpAffine(
        candidate.lab,
        inverse_affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    source_mask = np.full((height, width), 255, dtype=np.uint8)
    valid_mask = cv2.warpAffine(
        source_mask,
        inverse_affine,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    if not valid_mask.any():
        base_metrics["status"] = "alignment_outside_image"
        return base_metrics

    reference_lab = reference.lab.astype(np.float32)
    aligned_lab = aligned_candidate.astype(np.float32)
    luminance_offset = np.median(
        reference_lab[:, :, 0][valid_mask] - aligned_lab[:, :, 0][valid_mask]
    )
    aligned_lab[:, :, 0] = np.clip(aligned_lab[:, :, 0] + luminance_offset, 0, 255)
    lab_delta = np.linalg.norm(reference_lab - aligned_lab, axis=2)
    mean_lab_delta = float(lab_delta[valid_mask].mean())
    base_metrics["mean_lab_delta"] = mean_lab_delta

    changed_mask = (lab_delta >= config.local_change_lab_delta) & valid_mask
    changed_mask = cv2.morphologyEx(
        changed_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    valid_count = int(valid_mask.sum())
    change_ratio = float(changed_mask.sum() / valid_count)
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
        changed_mask.astype(np.uint8), connectivity=8
    )
    largest_component = (
        int(component_stats[1:, cv2.CC_STAT_AREA].max())
        if component_count > 1
        else 0
    )
    largest_component_ratio = largest_component / valid_count
    base_metrics["local_change_ratio"] = change_ratio
    base_metrics["largest_change_component_ratio"] = largest_component_ratio

    salient_mask = (lab_delta >= config.salient_change_lab_delta) & valid_mask
    salient_mask = cv2.morphologyEx(
        salient_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    salient_component_count, _, salient_component_stats, _ = cv2.connectedComponentsWithStats(
        salient_mask.astype(np.uint8), connectivity=8
    )
    largest_salient_component = (
        int(salient_component_stats[1:, cv2.CC_STAT_AREA].max())
        if salient_component_count > 1
        else 0
    )
    largest_salient_component_ratio = largest_salient_component / valid_count
    base_metrics["largest_salient_change_component_ratio"] = (
        largest_salient_component_ratio
    )

    # Ratio-tested matches are suitable for RANSAC, but repeated texture can
    # leave many identical descriptors unmatched by that stricter test. Use a
    # low absolute Hamming distance for the content-change feature mask instead.
    reliable_forward = matcher.match(reference.descriptors, candidate.descriptors)
    reliable_backward = matcher.match(candidate.descriptors, reference.descriptors)
    matched_reference = {
        match.queryIdx
        for match in reliable_forward
        if match.distance <= config.max_feature_match_distance
    }
    matched_candidate = {
        match.queryIdx
        for match in reliable_backward
        if match.distance <= config.max_feature_match_distance
    }
    reference_unmatched_points = [
        keypoint.pt
        for index, keypoint in enumerate(reference.keypoints)
        if index not in matched_reference
    ]
    candidate_unmatched_points = [
        keypoint.pt
        for index, keypoint in enumerate(candidate.keypoints)
        if index not in matched_candidate
    ]
    if candidate_unmatched_points:
        candidate_unmatched_array = np.float32(candidate_unmatched_points).reshape(-1, 1, 2)
        candidate_unmatched_in_reference = cv2.transform(
            candidate_unmatched_array, inverse_affine
        ).reshape(-1, 2)
        candidate_unmatched_points = [
            (float(point[0]), float(point[1])) for point in candidate_unmatched_in_reference
        ]
    reference_unmatched = len(reference_unmatched_points)
    candidate_unmatched = len(candidate_unmatched_points)
    reference_changed_points = points_inside_mask(reference_unmatched_points, changed_mask)
    candidate_changed_points = points_inside_mask(candidate_unmatched_points, changed_mask)
    reference_cluster, reference_changed = clustered_feature_points(
        reference_changed_points,
        width,
        height,
        config.feature_cluster_min_count,
    )
    candidate_cluster, candidate_changed = clustered_feature_points(
        candidate_changed_points,
        width,
        height,
        config.feature_cluster_min_count,
    )
    feature_cluster_event = reference_cluster or candidate_cluster
    base_metrics["reference_unmatched_feature_count"] = reference_unmatched
    base_metrics["candidate_unmatched_feature_count"] = candidate_unmatched
    base_metrics["reference_changed_feature_count"] = reference_changed
    base_metrics["candidate_changed_feature_count"] = candidate_changed
    base_metrics["feature_cluster_event"] = feature_cluster_event

    local_content_change = (
        change_ratio >= config.local_change_aggregate_ratio
        or largest_component_ratio >= config.local_change_component_ratio
        or largest_salient_component_ratio >= config.salient_change_component_ratio
    )
    visual_difference = mean_lab_delta > config.max_duplicate_mean_lab_delta
    content_event = local_content_change or feature_cluster_event or visual_difference
    strict_duplicate = (
        len(good) >= config.min_orb_matches
        and inliers >= config.min_orb_inliers
        and inlier_ratio >= config.min_inlier_ratio
        and median_flow <= config.max_duplicate_flow_px
        and not content_event
    )
    base_metrics["strict_duplicate"] = strict_duplicate
    base_metrics["content_event"] = not strict_duplicate
    if strict_duplicate:
        base_metrics["status"] = "strict_duplicate"
    elif local_content_change:
        base_metrics["status"] = "local_content_change"
    elif feature_cluster_event:
        base_metrics["status"] = "feature_cluster_change"
    else:
        base_metrics["status"] = "visual_difference"
    return base_metrics


def select_frames(
    source_frames: list[SourceFrame], poses: np.ndarray, config: SelectionConfig
) -> tuple[list[int], list[dict[str, Any]]]:
    orb = cv2.ORB_create(nfeatures=700, fastThreshold=12)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    selected_indices = [0]
    decisions: list[dict[str, Any]] = [
        {
            "source_frame_idx": 0,
            "source_pose_row": source_frames[0].pose_row,
            "selected": True,
            "reason": "initial_frame",
            "pose_translation_m": 0.0,
            "pose_rotation_deg": 0.0,
            "elapsed_s": 0.0,
            "visual": {"status": "initial_frame", "strict_duplicate": False},
        }
    ]
    reference_index = 0
    reference_visual = make_visual_frame(source_frames[0].cam0_path, config, orb)

    for index in range(1, len(source_frames)):
        source = source_frames[index]
        reference = source_frames[reference_index]
        translation = float(
            np.linalg.norm(
                poses[source.pose_row, :3, 3] - poses[reference.pose_row, :3, 3]
            )
        )
        rotation = pose_rotation_degrees(
            poses[reference.pose_row], poses[source.pose_row]
        )
        elapsed_s = (source.sensor_time_ns - reference.sensor_time_ns) / 1.0e9
        candidate_visual = make_visual_frame(source.cam0_path, config, orb)
        visual = compare_visual_frames(
            reference_visual, candidate_visual, config, matcher
        )
        hard_pose_motion = (
            translation >= config.hard_translation_m
            or rotation >= config.hard_rotation_deg
        )
        pose_motion = (
            translation >= config.soft_translation_m
            or rotation >= config.soft_rotation_deg
        )
        if hard_pose_motion or pose_motion:
            selected = True
            reason = "pose_motion"
            pose_motion_level = "hard" if hard_pose_motion else "soft"
        elif visual["content_event"]:
            selected = True
            reason = "image_event_at_static_pose"
            pose_motion_level = "below_soft_threshold"
        elif elapsed_s >= config.max_gap_s:
            selected = True
            reason = "watchdog"
            pose_motion_level = "below_soft_threshold"
        elif visual["strict_duplicate"]:
            selected = False
            reason = "strict_duplicate"
            pose_motion_level = "below_soft_threshold"
        else:
            # The visual classifier is deliberately conservative. This branch
            # protects a new image if a future classifier returns an unknown state.
            selected = True
            reason = "image_event_at_static_pose"
            pose_motion_level = "below_soft_threshold"

        decision = {
            "source_frame_idx": index,
            "source_pose_row": source.pose_row,
            "selected": selected,
            "reason": reason,
            "pose_motion_level": pose_motion_level,
            "pose_translation_m": translation,
            "pose_rotation_deg": rotation,
            "elapsed_s": elapsed_s,
            "visual": visual,
        }
        decisions.append(decision)
        if selected:
            selected_indices.append(index)
            reference_index = index
            reference_visual = candidate_visual

    if selected_indices[-1] != len(source_frames) - 1:
        last_index = len(source_frames) - 1
        decisions[last_index]["selected"] = True
        decisions[last_index]["reason"] = "terminal_frame"
        decisions[last_index]["forced_terminal_frame"] = True
        selected_indices.append(last_index)
    return selected_indices, decisions


def prepare_output_directory(output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output already exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    for directory in (output / "rgb", output / "stereo_right", output / "pose"):
        directory.mkdir(parents=True, exist_ok=True)


def link_file(link: Path, target: Path) -> None:
    link.symlink_to(target.resolve())


def write_selected_dataset(
    dataset: Path,
    output: Path,
    metadata: dict[str, Any],
    source_frames: list[SourceFrame],
    poses: np.ndarray,
    pose_timestamps: np.ndarray,
    selected_indices: list[int],
    decisions: list[dict[str, Any]],
    config: SelectionConfig,
    overwrite: bool,
) -> None:
    camera_info = dataset / "camera_info.json"
    if not camera_info.exists():
        raise ValueError(f"Missing camera calibration: {camera_info}")
    prepare_output_directory(output, overwrite)
    decision_by_source = {item["source_frame_idx"]: item for item in decisions}
    output_frames: list[dict[str, Any]] = []
    selected_poses: list[np.ndarray] = []
    selected_timestamps: list[int] = []
    for output_index, source_index in enumerate(selected_indices):
        source = source_frames[source_index]
        left_suffix = source.cam0_path.suffix.lower() or ".png"
        right_suffix = source.cam1_path.suffix.lower() or ".png"
        left_link = output / "rgb" / f"{output_index:08d}{left_suffix}"
        right_link = output / "stereo_right" / f"{output_index:08d}{right_suffix}"
        link_file(left_link, source.cam0_path)
        link_file(right_link, source.cam1_path)

        frame = copy.deepcopy(source.frame)
        frame["idx"] = output_index
        frame["source_frame_idx"] = source_index
        frame["source_pose_row"] = source.pose_row
        frame["pose_row"] = output_index
        frame["cam0"] = str(left_link)
        frame["cam1"] = str(right_link)
        frame["selection_reason"] = decision_by_source[source_index]["reason"]
        output_frames.append(frame)
        selected_poses.append(poses[source.pose_row])
        selected_timestamps.append(int(pose_timestamps[source.pose_row]))

    (output / "pose" / "poses.txt").write_text(
        "".join(
            " ".join(f"{value:.12g}" for value in pose.reshape(-1)) + "\n"
            for pose in selected_poses
        )
    )
    (output / "pose" / "pose_timestamps_ns.txt").write_text(
        "".join(f"{timestamp}\n" for timestamp in selected_timestamps)
    )
    link_file(output / "camera_info.json", camera_info)
    source_manifest = dataset / "source_manifest.json"
    if source_manifest.exists():
        link_file(output / "source_manifest.json", source_manifest)

    output_metadata = copy.deepcopy(metadata)
    output_metadata["frames"] = output_frames
    output_metadata["source_dataset"] = str(dataset)
    output_metadata["keyframe_selection"] = {
        "method": "absolute_time_pose_visual_content_safe",
        "report": "keyframe_selection_report.json",
        "source_frame_count": len(source_frames),
        "selected_frame_count": len(selected_indices),
        "config": asdict(config),
    }
    (output / "tick_index.json").write_text(
        json.dumps(output_metadata, indent=2) + "\n"
    )


def summarize_decisions(decisions: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for decision in decisions:
        reason = str(decision["reason"])
        summary[reason] = summary.get(reason, 0) + 1
    return summary


def run_selection(
    dataset: Path,
    output: Path,
    config: SelectionConfig,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    dataset = dataset.resolve()
    output = output.resolve()
    metadata, source_frames, poses, pose_timestamps = validate_source_dataset(dataset)
    selected_indices, decisions = select_frames(source_frames, poses, config)
    report = {
        "method": "absolute_time_pose_visual_content_safe",
        "source_dataset": str(dataset),
        "output_dataset": str(output),
        "dry_run": dry_run,
        "time_origin_ns": int(metadata["time_origin_ns"]),
        "source_frame_count": len(source_frames),
        "selected_frame_count": len(selected_indices),
        "reduction_ratio": 1.0 - len(selected_indices) / len(source_frames),
        "selection_reasons": summarize_decisions(decisions),
        "config": asdict(config),
        "decisions": decisions,
    }
    if not dry_run:
        write_selected_dataset(
            dataset,
            output,
            metadata,
            source_frames,
            poses,
            pose_timestamps,
            selected_indices,
            decisions,
            config,
            overwrite,
        )
        (output / "keyframe_selection_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    return report


def main() -> None:
    args = parse_args()
    report = run_selection(
        args.dataset,
        args.output,
        config_from_args(args),
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"decisions", "config"}
    }
    summary["config"] = report["config"]
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
