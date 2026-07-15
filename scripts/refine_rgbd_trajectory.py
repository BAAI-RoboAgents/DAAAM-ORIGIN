#!/usr/bin/env python3
"""Refine a planar RGB-D trajectory using depth-backed feature tracks.

This is intended for image sequences whose odometry translation is too
inconsistent for TSDF fusion.  Visual constraints are 3D-2D PnP measurements
from the input depth maps.  The visual-odometry mode jointly refines XY and
yaw while retaining the source camera height and small pitch/roll variations.
"""

import argparse
import copy
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


@dataclass
class VisualConstraint:
    first: int
    second: int
    delta_world_xy: list[float]
    sigma_m: float
    inliers: int
    median_reprojection_error_px: float
    rotation_error_deg: float
    relative_camera_rotation: list[list[float]]
    relative_camera_translation_m: list[float]
    constraint_type: str
    median_3d_error_m: float | None
    inlier_ratio: float | None
    loop: bool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine a planar camera path from RGB-D feature correspondences."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--keyframe-distance-m",
        type=float,
        default=0.10,
        help="Minimum input XY travel between selected keyframes.",
    )
    parser.add_argument(
        "--max-keyframe-gap", type=int, default=40, help="Maximum frame gap."
    )
    parser.add_argument(
        "--local-neighbor-span",
        type=int,
        default=2,
        help="Number of forward keyframe neighbors considered as local RGB-D links.",
    )
    parser.add_argument(
        "--loop-radius-m",
        type=float,
        default=0.60,
        help="Input-path radius used to propose nonlocal loop candidates.",
    )
    parser.add_argument(
        "--loop-min-keyframe-separation",
        type=int,
        default=15,
        help="Minimum keyframe-index separation for a loop candidate.",
    )
    parser.add_argument(
        "--max-loop-candidates-per-keyframe", type=int, default=2
    )
    parser.add_argument("--ratio-test", type=float, default=0.65)
    parser.add_argument("--min-inliers", type=int, default=80)
    parser.add_argument("--max-rotation-error-deg", type=float, default=3.0)
    parser.add_argument("--odom-sigma-m", type=float, default=0.25)
    parser.add_argument("--position-prior-sigma-m", type=float, default=0.75)
    parser.add_argument(
        "--mode",
        choices=(
            "global-scale",
            "free-form",
            "visual-odometry",
            "smooth-3d",
            "smooth-se3",
            "pose-graph-3d",
        ),
        default="global-scale",
        help=(
            "global-scale preserves the odometry path shape and estimates one XY "
            "scale from RGB-D; free-form adjusts XY keyframes; visual-odometry "
            "jointly refines XY and yaw from the metric PnP constraints; smooth-3d "
            "uses two-frame depth consistency with low-frequency XY/yaw corrections; "
            "smooth-se3 extends that model to smooth XYZ and full rotation corrections; "
            "pose-graph-3d integrates local two-depth rigid constraints at every "
            "keyframe."
        ),
    )
    parser.add_argument("--smooth-knot-count", type=int, default=12)
    parser.add_argument("--smooth-xy-curvature-sigma-m", type=float, default=0.16)
    parser.add_argument("--smooth-z-curvature-sigma-m", type=float, default=0.08)
    parser.add_argument("--smooth-yaw-curvature-sigma-deg", type=float, default=3.0)
    parser.add_argument(
        "--smooth-roll-pitch-curvature-sigma-deg", type=float, default=2.0
    )
    parser.add_argument("--smooth-se3-max-nfev", type=int, default=300)
    parser.add_argument(
        "--visual-max-nfev",
        type=int,
        default=300,
        help="Maximum evaluations for the sparse visual pose-graph solver.",
    )
    parser.add_argument("--max-depth-m", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rotation_error_deg(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.rad2deg(Rotation.from_matrix(lhs @ rhs.T).magnitude()))


def load_poses(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64).reshape(-1, 4, 4)
    if not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"Expected homogeneous poses in {path}")
    return poses


def select_keyframes(
    poses: np.ndarray, min_distance: float, max_gap: int
) -> list[int]:
    if min_distance <= 0.0 or max_gap < 1:
        raise ValueError("Keyframe distance and gap must be positive")
    keyframes = [0]
    last = 0
    for index in range(1, len(poses)):
        distance = np.linalg.norm(poses[index, :2, 3] - poses[last, :2, 3])
        if distance >= min_distance or index - last >= max_gap:
            keyframes.append(index)
            last = index
    if keyframes[-1] != len(poses) - 1:
        keyframes.append(len(poses) - 1)
    return keyframes


def make_pairs(
    keyframes: list[int], poses: np.ndarray, loop_radius: float, min_separation: int,
    max_loops_per_keyframe: int, local_neighbor_span: int = 2,
) -> list[tuple[int, int, bool]]:
    if local_neighbor_span < 1:
        raise ValueError("Local neighbor span must be positive")
    pairs: set[tuple[int, int, bool]] = set()
    for first in range(len(keyframes)):
        for offset in range(1, local_neighbor_span + 1):
            second = first + offset
            if second < len(keyframes):
                pairs.add((first, second, False))

    xy = poses[keyframes, :2, 3]
    for first in range(len(keyframes)):
        candidates = []
        for second in range(first + min_separation, len(keyframes)):
            distance = float(np.linalg.norm(xy[second] - xy[first]))
            if distance <= loop_radius:
                candidates.append((distance, second))
        for _, second in sorted(candidates)[:max_loops_per_keyframe]:
            pairs.add((first, second, True))
    return sorted(pairs)


class FrameCache:
    def __init__(self, dataset: Path):
        self.dataset = dataset
        self.sift = cv2.SIFT_create(nfeatures=4000)
        self.matcher = cv2.BFMatcher()
        self.features = {}
        self.depths = {}

    def feature(self, frame_index: int):
        if frame_index not in self.features:
            image = cv2.imread(
                str(self.dataset / "rgb" / f"{frame_index:08d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            if image is None:
                raise FileNotFoundError(f"Missing RGB frame {frame_index}")
            self.features[frame_index] = self.sift.detectAndCompute(image, None)
        return self.features[frame_index]

    def depth(self, frame_index: int) -> np.ndarray:
        if frame_index not in self.depths:
            image = cv2.imread(
                str(self.dataset / "depth" / f"{frame_index:08d}.png"),
                cv2.IMREAD_UNCHANGED,
            )
            if image is None or image.dtype != np.uint16:
                raise ValueError(f"Expected uint16 depth image for frame {frame_index}")
            self.depths[frame_index] = image.astype(np.float64) / 1000.0
        return self.depths[frame_index]


def estimate_constraint(
    cache: FrameCache,
    first: int,
    second: int,
    poses: np.ndarray,
    camera_matrix: np.ndarray,
    max_depth_m: float,
    ratio_test: float,
    min_inliers: int,
    max_rotation_error_deg: float,
    loop: bool,
) -> VisualConstraint | None:
    keypoints_first, descriptors_first = cache.feature(first)
    keypoints_second, descriptors_second = cache.feature(second)
    if descriptors_first is None or descriptors_second is None:
        return None

    candidates = cache.matcher.knnMatch(descriptors_first, descriptors_second, k=2)
    matches = [
        match for match, alternate in candidates if match.distance < ratio_test * alternate.distance
    ]
    if len(matches) < min_inliers:
        return None

    pixels_first = np.float32(
        [keypoints_first[match.queryIdx].pt for match in matches]
    )
    pixels_second = np.float32(
        [keypoints_second[match.trainIdx].pt for match in matches]
    )
    _, fundamental_mask = cv2.findFundamentalMat(
        pixels_first,
        pixels_second,
        cv2.USAC_MAGSAC,
        1.5,
        0.999,
        10000,
    )
    if fundamental_mask is None:
        return None
    fundamental_mask = fundamental_mask.reshape(-1).astype(bool)
    pixels_first = pixels_first[fundamental_mask]
    pixels_second = pixels_second[fundamental_mask]
    if len(pixels_first) < min_inliers:
        return None

    depth = cache.depth(first)
    height, width = depth.shape
    u = np.clip(np.rint(pixels_first[:, 0]).astype(int), 0, width - 1)
    v = np.clip(np.rint(pixels_first[:, 1]).astype(int), 0, height - 1)
    z = depth[v, u]
    valid = (z >= 0.25) & (z <= max_depth_m)
    pixels_first = pixels_first[valid]
    pixels_second = pixels_second[valid]
    z = z[valid]
    if len(z) < min_inliers:
        return None

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    points_first = np.column_stack(
        (
            (pixels_first[:, 0] - cx) * z / fx,
            (pixels_first[:, 1] - cy) * z / fy,
            z,
        )
    ).astype(np.float64)
    success, rotation_vector, translation, inlier_indices = cv2.solvePnPRansac(
        points_first,
        pixels_second,
        camera_matrix,
        None,
        iterationsCount=1000,
        reprojectionError=2.5,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inlier_indices is None or len(inlier_indices) < min_inliers:
        return None
    inlier_indices = inlier_indices.reshape(-1)
    cv2.solvePnPRefineLM(
        points_first[inlier_indices],
        pixels_second[inlier_indices],
        camera_matrix,
        None,
        rotation_vector,
        translation,
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    relative_initial = np.linalg.inv(poses[second]) @ poses[first]
    angle_error = rotation_error_deg(rotation, relative_initial[:3, :3])
    if angle_error > max_rotation_error_deg:
        return None

    projected, _ = cv2.projectPoints(
        points_first[inlier_indices], rotation_vector, translation, camera_matrix, None
    )
    reprojection_error = np.linalg.norm(
        projected.reshape(-1, 2) - pixels_second[inlier_indices], axis=1
    )
    median_error = float(np.median(reprojection_error))
    # X_second = R * X_first + t. Therefore p_second - p_first = -R_W_second * t.
    delta_world = -poses[second, :3, :3] @ translation.reshape(3)
    horizontal_distance = float(np.linalg.norm(delta_world[:2]))
    if not 0.002 <= horizontal_distance <= 1.5:
        return None

    sigma = float(np.clip(0.02 + 0.025 * median_error + 0.2 / np.sqrt(len(inlier_indices)), 0.03, 0.12))
    return VisualConstraint(
        first=first,
        second=second,
        delta_world_xy=delta_world[:2].tolist(),
        sigma_m=sigma,
        inliers=int(len(inlier_indices)),
        median_reprojection_error_px=median_error,
        rotation_error_deg=angle_error,
        relative_camera_rotation=rotation.tolist(),
        relative_camera_translation_m=translation.reshape(3).tolist(),
        constraint_type="pnp_3d2d",
        median_3d_error_m=None,
        inlier_ratio=None,
        loop=loop,
    )


def backproject_pixels(
    pixels: np.ndarray, depth: np.ndarray, camera_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject pixels and return their Z values for validity filtering."""
    height, width = depth.shape
    u = np.clip(np.rint(pixels[:, 0]).astype(int), 0, width - 1)
    v = np.clip(np.rint(pixels[:, 1]).astype(int), 0, height - 1)
    z = depth[v, u]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    points = np.column_stack(
        (
            (pixels[:, 0] - cx) * z / fx,
            (pixels[:, 1] - cy) * z / fy,
            z,
        )
    ).astype(np.float64)
    return points, z


def fit_rigid_transform(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the proper rigid transform mapping source points to target points."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    return rotation, target_center - rotation @ source_center


def estimate_3d_constraint(
    cache: FrameCache,
    first: int,
    second: int,
    poses: np.ndarray,
    camera_matrix: np.ndarray,
    max_depth_m: float,
    ratio_test: float,
    min_inliers: int,
    max_rotation_error_deg: float,
    loop: bool,
) -> VisualConstraint | None:
    """Estimate a rigid camera transform using depth from both matched frames.

    The initial 2D epipolar filter removes gross descriptor mismatches.  A
    3D RANSAC followed by rigid Kabsch refinement then requires the two depth
    maps to agree metrically, which is substantially more selective than a
    source-depth-only PnP fit on this sequence.
    """
    keypoints_first, descriptors_first = cache.feature(first)
    keypoints_second, descriptors_second = cache.feature(second)
    if descriptors_first is None or descriptors_second is None:
        return None

    candidates = cache.matcher.knnMatch(descriptors_first, descriptors_second, k=2)
    matches = [
        match
        for match, alternate in candidates
        if match.distance < ratio_test * alternate.distance
    ]
    if len(matches) < min_inliers:
        return None
    pixels_first = np.float32(
        [keypoints_first[match.queryIdx].pt for match in matches]
    )
    pixels_second = np.float32(
        [keypoints_second[match.trainIdx].pt for match in matches]
    )
    _, fundamental_mask = cv2.findFundamentalMat(
        pixels_first,
        pixels_second,
        cv2.USAC_MAGSAC,
        1.5,
        0.999,
        10000,
    )
    if fundamental_mask is None:
        return None
    fundamental_mask = fundamental_mask.reshape(-1).astype(bool)
    pixels_first = pixels_first[fundamental_mask]
    pixels_second = pixels_second[fundamental_mask]
    if len(pixels_first) < min_inliers:
        return None

    points_first, z_first = backproject_pixels(
        pixels_first, cache.depth(first), camera_matrix
    )
    points_second, z_second = backproject_pixels(
        pixels_second, cache.depth(second), camera_matrix
    )
    valid = (
        (z_first >= 0.25)
        & (z_first <= max_depth_m)
        & (z_second >= 0.25)
        & (z_second <= max_depth_m)
    )
    points_first = points_first[valid]
    points_second = points_second[valid]
    pixels_second = pixels_second[valid]
    if len(points_first) < min_inliers:
        return None

    success, _, inlier_mask = cv2.estimateAffine3D(
        points_first,
        points_second,
        ransacThreshold=0.06,
        confidence=0.999,
    )
    if not success or inlier_mask is None:
        return None
    inlier_mask = inlier_mask.reshape(-1).astype(bool)
    if int(inlier_mask.sum()) < min_inliers:
        return None
    rotation, translation = fit_rigid_transform(
        points_first[inlier_mask], points_second[inlier_mask]
    )
    for _ in range(3):
        errors = np.linalg.norm(
            (rotation @ points_first.T).T + translation - points_second, axis=1
        )
        inlier_mask = errors < 0.05
        if int(inlier_mask.sum()) < min_inliers:
            return None
        rotation, translation = fit_rigid_transform(
            points_first[inlier_mask], points_second[inlier_mask]
        )
    errors = np.linalg.norm(
        (rotation @ points_first.T).T + translation - points_second, axis=1
    )
    median_3d_error = float(np.median(errors[inlier_mask]))
    relative_initial = np.linalg.inv(poses[second]) @ poses[first]
    angle_error = rotation_error_deg(rotation, relative_initial[:3, :3])
    if angle_error > max_rotation_error_deg:
        return None
    # Local links should be short, while a retrieved revisit can legitimately
    # observe the same structure from opposite sides of this small room.
    max_translation_m = 4.5 if loop else 1.5
    if not 0.002 <= float(np.linalg.norm(translation)) <= max_translation_m:
        return None

    rotation_vector, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(
        points_first[inlier_mask], rotation_vector, translation, camera_matrix, None
    )
    reprojection_error = np.linalg.norm(
        projected.reshape(-1, 2) - pixels_second[inlier_mask], axis=1
    )
    median_reprojection_error = float(np.median(reprojection_error))
    delta_world = -poses[second, :3, :3] @ translation
    sigma = float(
        np.clip(median_3d_error + 0.25 / np.sqrt(inlier_mask.sum()), 0.025, 0.08)
    )
    return VisualConstraint(
        first=first,
        second=second,
        delta_world_xy=delta_world[:2].tolist(),
        sigma_m=sigma,
        inliers=int(inlier_mask.sum()),
        median_reprojection_error_px=median_reprojection_error,
        rotation_error_deg=angle_error,
        relative_camera_rotation=rotation.tolist(),
        relative_camera_translation_m=translation.tolist(),
        constraint_type="depth_3d3d",
        median_3d_error_m=median_3d_error,
        inlier_ratio=float(inlier_mask.mean()),
        loop=loop,
    )


def optimize_keyframe_positions(
    keyframes: list[int],
    poses: np.ndarray,
    constraints: list[VisualConstraint],
    odom_sigma_m: float,
    position_prior_sigma_m: float,
) -> tuple[np.ndarray, dict]:
    initial = poses[keyframes, :2, 3].copy()
    if len(initial) < 2:
        raise ValueError("Need at least two keyframes")

    frame_to_key = {frame: index for index, frame in enumerate(keyframes)}
    visual = [
        (frame_to_key[item.first], frame_to_key[item.second], item)
        for item in constraints
    ]
    odometry = [
        (index, index + 1, initial[index + 1] - initial[index])
        for index in range(len(initial) - 1)
    ]

    def unpack(values):
        positions = initial.copy()
        positions[1:] = values.reshape(-1, 2)
        return positions

    def residuals(values):
        positions = unpack(values)
        output = []
        for first, second, constraint in visual:
            delta = np.asarray(constraint.delta_world_xy)
            output.extend((positions[second] - positions[first] - delta) / constraint.sigma_m)
        for first, second, delta in odometry:
            output.extend((positions[second] - positions[first] - delta) / odom_sigma_m)
        # A broad absolute prior prevents an underconstrained loop from translating a
        # large portion of the path while leaving visual constraints dominant.
        output.extend(((positions[1:] - initial[1:]) / position_prior_sigma_m).ravel())
        return np.asarray(output, dtype=np.float64)

    result = least_squares(
        residuals,
        initial[1:].reshape(-1),
        loss="huber",
        f_scale=1.0,
        max_nfev=300,
        verbose=0,
    )
    refined = unpack(result.x)

    def visual_errors(positions):
        return np.asarray(
            [
                np.linalg.norm(
                    positions[second]
                    - positions[first]
                    - np.asarray(constraint.delta_world_xy)
                )
                for first, second, constraint in visual
            ]
        )

    def path_length(positions):
        return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())

    initial_errors = visual_errors(initial)
    refined_errors = visual_errors(refined)
    report = {
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
        "optimizer_cost": float(result.cost),
        "visual_constraints": len(visual),
        "visual_residual_median_before_m": float(np.median(initial_errors)),
        "visual_residual_median_after_m": float(np.median(refined_errors)),
        "keyframe_path_length_before_m": path_length(initial),
        "keyframe_path_length_after_m": path_length(refined),
    }
    return refined, report


def estimate_global_xy_scale(
    keyframes: list[int], poses: np.ndarray, constraints: list[VisualConstraint]
) -> tuple[float, dict]:
    """Fit visual translation along the trusted odometry path direction.

    A single PnP segment has useful metric scale but can have sizeable lateral
    direction error in texture-poor office views.  Projecting it onto the
    gyroscope/odometry direction keeps turns and the global path topology
    stable while correcting the distance-to-depth ratio that breaks TSDF fusion.
    """
    del keyframes  # Constraints already use source frame indices.
    visual = []
    odometry = []
    weights = []
    for constraint in constraints:
        if constraint.loop:
            continue
        first = constraint.first
        second = constraint.second
        delta_odom = poses[second, :2, 3] - poses[first, :2, 3]
        distance = float(np.linalg.norm(delta_odom))
        if distance < 0.04:
            continue
        delta_visual = np.asarray(constraint.delta_world_xy)
        direction_cosine = float(
            np.dot(delta_visual, delta_odom)
            / (np.linalg.norm(delta_visual) * distance + 1.0e-12)
        )
        if direction_cosine < 0.55:
            continue
        visual.append(delta_visual)
        odometry.append(delta_odom)
        weights.append(1.0 / constraint.sigma_m**2)
    if len(visual) < 12:
        raise RuntimeError("Too few direction-consistent RGB-D constraints for scale")

    visual = np.asarray(visual)
    odometry = np.asarray(odometry)
    weights = np.asarray(weights)
    projections = np.sum(visual * odometry, axis=1) / np.sum(odometry**2, axis=1)
    scale = float(np.median(projections))
    # Iteratively discard gross segment-scale outliers while retaining the
    # direction-selected measurements above.
    for _ in range(3):
        residual = np.abs(projections - scale)
        median = float(np.median(residual))
        keep = residual <= max(0.08, 3.0 * median)
        if int(keep.sum()) < 12:
            break
        scale = float(
            np.sum(weights[keep] * np.sum(visual[keep] * odometry[keep], axis=1))
            / np.sum(weights[keep] * np.sum(odometry[keep] ** 2, axis=1))
        )
    if not 0.35 <= scale <= 1.25:
        raise RuntimeError(f"Implausible RGB-D XY scale: {scale:.3f}")
    return scale, {
        "scale_constraints": int(len(projections)),
        "scale_constraints_inlier": int(keep.sum()),
        "xy_scale": scale,
        "segment_scale_median": float(np.median(projections)),
        "segment_scale_p5_p95": np.percentile(projections, [5, 95]).tolist(),
    }


def apply_global_xy_scale(poses: np.ndarray, scale: float) -> np.ndarray:
    refined = poses.copy()
    origin = poses[0, :2, 3]
    refined[:, :2, 3] = origin + scale * (poses[:, :2, 3] - origin)
    return refined


def rotation_z(angle_rad: float) -> np.ndarray:
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def yaw_from_rotation(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def wrap_angle(angle_rad: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))


def optimize_visual_odometry(
    keyframes: list[int],
    poses: np.ndarray,
    constraints: list[VisualConstraint],
    odom_sigma_m: float,
    position_prior_sigma_m: float,
    max_nfev: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Optimize planar camera poses against metric RGB-D PnP constraints.

    PnP measures ``C_second_T_C_first`` directly.  We retain the camera's
    calibrated height and level the trajectory to the first camera's
    pitch/roll, leaving XY and yaw as the only optimized degrees of freedom.
    This keeps the solution physically compatible with the depth maps while
    avoiding accumulated wheel/odometry heading error.
    """
    if len(keyframes) < 2:
        raise ValueError("Need at least two keyframes")

    frame_to_key = {frame: index for index, frame in enumerate(keyframes)}
    visual = [
        (frame_to_key[item.first], frame_to_key[item.second], item)
        for item in constraints
    ]
    reference_rotation = poses[keyframes[0], :3, :3]
    source_xy = poses[keyframes, :2, 3].copy()
    source_z = poses[keyframes, 2, 3].copy()
    source_yaw = np.unwrap(
        np.asarray(
            [
                yaw_from_rotation(
                    poses[frame, :3, :3] @ reference_rotation.T
                )
                for frame in keyframes
            ],
            dtype=np.float64,
        )
    )

    try:
        fallback_scale, _ = estimate_global_xy_scale(keyframes, poses, constraints)
    except RuntimeError:
        fallback_scale = 1.0

    # A sequential visual estimate is a good initialization.  The later pose
    # graph uses both one- and two-keyframe PnP links to remove its local drift.
    direct = {
        (item.first, item.second): item
        for item in constraints
        if not item.loop
    }
    initial_xy = source_xy.copy()
    initial_yaw = source_yaw.copy()
    initial_yaw[0] = 0.0
    visual_links = 0
    fallback_links = 0
    for key_index in range(1, len(keyframes)):
        previous_frame = keyframes[key_index - 1]
        current_frame = keyframes[key_index]
        constraint = direct.get((previous_frame, current_frame))
        if constraint is None:
            fallback_links += 1
            initial_xy[key_index] = (
                initial_xy[key_index - 1]
                + fallback_scale * (source_xy[key_index] - source_xy[key_index - 1])
            )
            initial_yaw[key_index] = (
                initial_yaw[key_index - 1]
                + wrap_angle(source_yaw[key_index] - source_yaw[key_index - 1])
            )
            continue

        visual_links += 1
        relative_rotation = np.asarray(
            constraint.relative_camera_rotation, dtype=np.float64
        )
        relative_yaw = yaw_from_rotation(
            reference_rotation @ relative_rotation @ reference_rotation.T
        )
        initial_yaw[key_index] = initial_yaw[key_index - 1] - relative_yaw
        current_rotation = rotation_z(initial_yaw[key_index]) @ reference_rotation
        relative_translation = np.asarray(
            constraint.relative_camera_translation_m, dtype=np.float64
        )
        initial_xy[key_index] = (
            initial_xy[key_index - 1]
            - (current_rotation @ relative_translation)[:2]
        )

    def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        optimized = values.reshape(-1, 3)
        xy = initial_xy.copy()
        yaw = initial_yaw.copy()
        xy[1:] = optimized[:, :2]
        yaw[1:] = optimized[:, 2]
        return xy, yaw

    visual_first = np.asarray([item[0] for item in visual], dtype=np.int64)
    visual_second = np.asarray([item[1] for item in visual], dtype=np.int64)
    visual_translation = np.asarray(
        [item[2].relative_camera_translation_m for item in visual],
        dtype=np.float64,
    )
    visual_translation_sigma = np.asarray(
        [item[2].sigma_m for item in visual], dtype=np.float64
    )
    visual_measured_yaw = np.asarray(
        [
            yaw_from_rotation(
                reference_rotation
                @ np.asarray(item[2].relative_camera_rotation, dtype=np.float64)
                @ reference_rotation.T
            )
            for item in visual
        ],
        dtype=np.float64,
    )
    visual_yaw_sigma = np.deg2rad(
        np.clip(
            0.40
            + 0.45
            * np.asarray(
                [item[2].median_reprojection_error_px for item in visual],
                dtype=np.float64,
            ),
            0.5,
            2.0,
        )
    )
    scaled_source_xy = source_xy[0] + fallback_scale * (source_xy - source_xy[0])
    odom_target_xy = fallback_scale * np.diff(source_xy, axis=0)
    odom_target_yaw = wrap_angle(np.diff(source_yaw))

    def planar_rotations(yaw: np.ndarray) -> np.ndarray:
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        world_yaw = np.zeros((len(yaw), 3, 3), dtype=np.float64)
        world_yaw[:, 0, 0] = cosine
        world_yaw[:, 0, 1] = -sine
        world_yaw[:, 1, 0] = sine
        world_yaw[:, 1, 1] = cosine
        world_yaw[:, 2, 2] = 1.0
        return world_yaw @ reference_rotation

    def visual_errors(
        xy: np.ndarray, yaw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.column_stack((xy, source_z))
        rotations = planar_rotations(yaw)
        world_delta = positions[visual_first] - positions[visual_second]
        predicted_translation = np.einsum(
            "nij,nj->ni",
            np.transpose(rotations[visual_second], (0, 2, 1)),
            world_delta,
        )
        translation_delta = predicted_translation - visual_translation
        yaw_delta = wrap_angle(
            yaw[visual_first] - yaw[visual_second] - visual_measured_yaw
        )
        return (
            translation_delta,
            yaw_delta,
            np.linalg.norm(translation_delta, axis=1),
        )

    def measure_residuals(
        xy: np.ndarray, yaw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        _, yaw_delta, translation_error = visual_errors(xy, yaw)
        return translation_error, yaw_delta

    def residuals(values: np.ndarray) -> np.ndarray:
        xy, yaw = unpack(values)
        translation_delta, yaw_delta, _ = visual_errors(xy, yaw)
        visual_residual = np.column_stack(
            (
                translation_delta / visual_translation_sigma[:, None],
                yaw_delta / visual_yaw_sigma,
            )
        ).ravel()
        odom_residual = np.column_stack(
            (
                (np.diff(xy, axis=0) - odom_target_xy) / odom_sigma_m,
                wrap_angle(np.diff(yaw) - odom_target_yaw) / np.deg2rad(20.0),
            )
        ).ravel()
        position_prior = (
            (xy[1:] - scaled_source_xy[1:]) / position_prior_sigma_m
        ).ravel()
        yaw_prior = (
            wrap_angle(yaw[1:] - source_yaw[1:]) / np.deg2rad(35.0)
        )
        return np.concatenate(
            (visual_residual, odom_residual, position_prior, yaw_prior)
        )

    initial_translation_errors, initial_yaw_errors = measure_residuals(
        source_xy, source_yaw
    )
    variable_count = (len(keyframes) - 1) * 3
    residual_count = len(visual) * 4 + (len(keyframes) - 1) * 6
    sparsity = lil_matrix((residual_count, variable_count), dtype=np.int8)
    row = 0

    def mark_keyframe(rows: slice, key_index: int) -> None:
        if key_index == 0:
            return
        columns = slice((key_index - 1) * 3, key_index * 3)
        sparsity[rows, columns] = 1

    for first, second, _ in visual:
        rows = slice(row, row + 4)
        mark_keyframe(rows, first)
        mark_keyframe(rows, second)
        row += 4
    for key_index in range(len(keyframes) - 1):
        rows = slice(row, row + 3)
        mark_keyframe(rows, key_index)
        mark_keyframe(rows, key_index + 1)
        row += 3
    for key_index in range(1, len(keyframes)):
        mark_keyframe(slice(row, row + 2), key_index)
        row += 2
    for key_index in range(1, len(keyframes)):
        mark_keyframe(slice(row, row + 1), key_index)
        row += 1
    if row != residual_count:
        raise RuntimeError(
            f"Jacobian sparsity rows disagree: {row} vs {residual_count}"
        )
    result = least_squares(
        residuals,
        np.column_stack((initial_xy[1:], initial_yaw[1:])).reshape(-1),
        jac_sparsity=sparsity.tocsr(),
        method="trf",
        loss="huber",
        f_scale=1.0,
        max_nfev=max_nfev,
        verbose=0,
    )
    optimized_xy, optimized_yaw = unpack(result.x)
    translation_errors, yaw_errors = measure_residuals(optimized_xy, optimized_yaw)
    report = {
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
        "optimizer_cost": float(result.cost),
        "optimizer_nfev": int(result.nfev),
        "visual_constraints": len(visual),
        "sequential_visual_links": visual_links,
        "sequential_fallback_links": fallback_links,
        "fallback_xy_scale": fallback_scale,
        "visual_translation_residual_median_before_m": float(
            np.median(initial_translation_errors)
        ),
        "visual_translation_residual_median_after_m": float(
            np.median(translation_errors)
        ),
        "visual_residual_median_before_m": float(
            np.median(initial_translation_errors)
        ),
        "visual_residual_median_after_m": float(np.median(translation_errors)),
        "visual_yaw_residual_median_before_deg": float(
            np.rad2deg(np.median(np.abs(initial_yaw_errors)))
        ),
        "visual_yaw_residual_median_after_deg": float(
            np.rad2deg(np.median(np.abs(yaw_errors)))
        ),
        "keyframe_path_length_before_m": float(
            np.linalg.norm(np.diff(source_xy, axis=0), axis=1).sum()
        ),
        "keyframe_path_length_after_m": float(
            np.linalg.norm(np.diff(optimized_xy, axis=0), axis=1).sum()
        ),
        "keyframe_yaw_correction_p5_p95_deg": np.rad2deg(
            np.percentile(wrap_angle(optimized_yaw - source_yaw), [5, 95])
        ).tolist(),
    }
    return optimized_xy, optimized_yaw, report


def interpolate_visual_odometry(
    keyframes: list[int],
    keyframe_xy: np.ndarray,
    keyframe_yaw: np.ndarray,
    poses: np.ndarray,
) -> np.ndarray:
    """Apply smoothly interpolated visual XY/yaw corrections to every frame."""
    reference_rotation = poses[keyframes[0], :3, :3]
    source_xy = poses[keyframes, :2, 3]
    source_yaw = np.unwrap(
        np.asarray(
            [
                yaw_from_rotation(
                    poses[frame, :3, :3] @ reference_rotation.T
                )
                for frame in keyframes
            ],
            dtype=np.float64,
        )
    )
    xy_correction = keyframe_xy - source_xy
    yaw_correction = np.unwrap(keyframe_yaw - source_yaw)
    all_indices = np.arange(len(poses))
    refined = poses.copy()
    refined[:, 0, 3] += np.interp(all_indices, keyframes, xy_correction[:, 0])
    refined[:, 1, 3] += np.interp(all_indices, keyframes, xy_correction[:, 1])
    interpolated_yaw = np.interp(all_indices, keyframes, yaw_correction)
    for frame, correction in enumerate(interpolated_yaw):
        refined[frame, :3, :3] = rotation_z(correction) @ poses[frame, :3, :3]

    # Use the optimized level orientation exactly at keyframes, while keeping
    # the original high-frequency pitch/roll variation between them.
    for key_index, frame in enumerate(keyframes):
        refined[frame, :3, :3] = (
            rotation_z(keyframe_yaw[key_index]) @ reference_rotation
        )
    return refined


def optimize_smooth_3d_trajectory(
    poses: np.ndarray,
    constraints: list[VisualConstraint],
    knot_count: int,
    xy_curvature_sigma_m: float,
    yaw_curvature_sigma_deg: float,
) -> tuple[np.ndarray, dict]:
    """Fit a smooth global trajectory to depth-verified 3D-3D constraints.

    Position and yaw corrections are linearly interpolated from a small number
    of temporal knots.  This has enough flexibility to correct accumulated
    odometry drift, while preventing an individual visual match from creating
    a kink in the fusion trajectory.
    """
    if knot_count < 3:
        raise ValueError("Smooth 3D trajectory refinement needs at least three knots")
    if xy_curvature_sigma_m <= 0.0 or yaw_curvature_sigma_deg <= 0.0:
        raise ValueError("Smooth trajectory curvature sigmas must be positive")
    if any(item.constraint_type != "depth_3d3d" for item in constraints):
        raise ValueError("Smooth 3D refinement requires two-frame depth constraints")

    frame_count = len(poses)
    knot_count = min(knot_count, frame_count)
    frame_indices = np.arange(frame_count)
    knot_indices = np.linspace(0, frame_count - 1, knot_count, dtype=int)
    source_xy = poses[:, :2, 3]

    def build_poses(
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        scale = float(np.exp(values[0]))
        knot_values = values[1:].reshape(knot_count - 1, 3)
        knot_xy_correction = np.vstack(([0.0, 0.0], knot_values[:, :2]))
        knot_yaw_correction = np.r_[0.0, knot_values[:, 2]]
        xy_correction = np.column_stack(
            (
                np.interp(frame_indices, knot_indices, knot_xy_correction[:, 0]),
                np.interp(frame_indices, knot_indices, knot_xy_correction[:, 1]),
            )
        )
        yaw_correction = np.interp(
            frame_indices, knot_indices, knot_yaw_correction
        )
        refined = poses.copy()
        refined[:, :2, 3] = (
            source_xy[0] + scale * (source_xy - source_xy[0]) + xy_correction
        )
        for frame, yaw in enumerate(yaw_correction):
            refined[frame, :3, :3] = rotation_z(yaw) @ poses[frame, :3, :3]
        return (
            refined,
            xy_correction,
            yaw_correction,
            scale,
            knot_xy_correction,
            knot_yaw_correction,
        )

    def measure_errors(refined: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        translation_errors = []
        rotation_errors = []
        for constraint in constraints:
            relative = np.linalg.inv(refined[constraint.second]) @ refined[constraint.first]
            measured_translation = np.asarray(
                constraint.relative_camera_translation_m, dtype=np.float64
            )
            measured_rotation = np.asarray(
                constraint.relative_camera_rotation, dtype=np.float64
            )
            translation_errors.append(
                float(np.linalg.norm(relative[:3, 3] - measured_translation))
            )
            rotation_errors.append(
                rotation_error_deg(relative[:3, :3], measured_rotation)
            )
        return np.asarray(translation_errors), np.asarray(rotation_errors)

    def residuals(values: np.ndarray) -> np.ndarray:
        (
            refined,
            _,
            _,
            scale,
            knot_xy_correction,
            knot_yaw_correction,
        ) = build_poses(values)
        output = []
        for constraint in constraints:
            relative = np.linalg.inv(refined[constraint.second]) @ refined[constraint.first]
            measured_translation = np.asarray(
                constraint.relative_camera_translation_m, dtype=np.float64
            )
            measured_rotation = np.asarray(
                constraint.relative_camera_rotation, dtype=np.float64
            )
            output.extend(
                (relative[:3, 3] - measured_translation) / constraint.sigma_m
            )
            output.extend(
                Rotation.from_matrix(
                    relative[:3, :3] @ measured_rotation.T
                ).as_rotvec()
                / np.deg2rad(1.2)
            )

        output.extend(
            np.diff(knot_xy_correction, n=2, axis=0).ravel()
            / xy_curvature_sigma_m
        )
        output.extend(
            np.diff(knot_yaw_correction, n=2)
            / np.deg2rad(yaw_curvature_sigma_deg)
        )
        output.extend(knot_xy_correction[1:].ravel() / 1.0)
        output.extend(knot_yaw_correction[1:] / np.deg2rad(25.0))
        output.append(np.log(scale / 0.58) / 0.2)
        return np.asarray(output, dtype=np.float64)

    initial_values = np.r_[np.log(0.58), np.zeros(3 * (knot_count - 1))]
    initial_poses, *_ = build_poses(initial_values)
    initial_translation_errors, initial_rotation_errors = measure_errors(initial_poses)
    result = least_squares(
        residuals,
        initial_values,
        loss="huber",
        f_scale=1.0,
        max_nfev=600,
        verbose=0,
    )
    (
        refined,
        xy_correction,
        yaw_correction,
        scale,
        knot_xy_correction,
        knot_yaw_correction,
    ) = build_poses(result.x)
    translation_errors, rotation_errors = measure_errors(refined)
    report = {
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
        "optimizer_cost": float(result.cost),
        "visual_constraints": len(constraints),
        "constraint_type": "depth_3d3d",
        "initial_xy_scale": 0.58,
        "xy_scale": scale,
        "smooth_knot_count": knot_count,
        "smooth_knot_indices": knot_indices.tolist(),
        "smooth_xy_curvature_sigma_m": xy_curvature_sigma_m,
        "smooth_yaw_curvature_sigma_deg": yaw_curvature_sigma_deg,
        "visual_translation_residual_median_before_m": float(
            np.median(initial_translation_errors)
        ),
        "visual_translation_residual_median_after_m": float(
            np.median(translation_errors)
        ),
        "visual_rotation_residual_median_before_deg": float(
            np.median(initial_rotation_errors)
        ),
        "visual_rotation_residual_median_after_deg": float(
            np.median(rotation_errors)
        ),
        "visual_residual_median_before_m": float(
            np.median(initial_translation_errors)
        ),
        "visual_residual_median_after_m": float(np.median(translation_errors)),
        "full_path_length_before_m": float(
            np.linalg.norm(np.diff(source_xy, axis=0), axis=1).sum()
        ),
        "full_path_length_after_m": float(
            np.linalg.norm(np.diff(refined[:, :2, 3], axis=0), axis=1).sum()
        ),
        "xy_correction_p5_p95_m": np.percentile(
            xy_correction, [5, 95], axis=0
        ).tolist(),
        "yaw_correction_p5_p95_deg": np.rad2deg(
            np.percentile(yaw_correction, [5, 95])
        ).tolist(),
        "knot_xy_correction_m": knot_xy_correction.tolist(),
        "knot_yaw_correction_deg": np.rad2deg(knot_yaw_correction).tolist(),
    }
    return refined, report


def optimize_smooth_se3_trajectory(
    poses: np.ndarray,
    constraints: list[VisualConstraint],
    knot_count: int,
    xy_curvature_sigma_m: float,
    z_curvature_sigma_m: float,
    yaw_curvature_sigma_deg: float,
    roll_pitch_curvature_sigma_deg: float,
    max_nfev: int,
) -> tuple[np.ndarray, dict]:
    """Fit smooth full-pose corrections to local two-depth RGB-D constraints.

    The prior G1 trajectory has useful high-frequency motion, but over this
    sequence its accumulated orientation and translation drift are large
    enough to make a global mesh fold into itself.  We model the correction as
    time-interpolated world-frame XYZ and rotation-vector knots.  Unlike an
    unconstrained visual odometry chain, this admits only low-frequency drift
    corrections and preserves the metric local 3D-3D measurements.
    """
    if knot_count < 3:
        raise ValueError("Smooth SE(3) refinement needs at least three knots")
    if min(
        xy_curvature_sigma_m,
        z_curvature_sigma_m,
        yaw_curvature_sigma_deg,
        roll_pitch_curvature_sigma_deg,
    ) <= 0.0:
        raise ValueError("Smooth SE(3) curvature sigmas must be positive")
    if max_nfev < 1:
        raise ValueError("Smooth SE(3) max_nfev must be positive")
    if any(item.constraint_type != "depth_3d3d" for item in constraints):
        raise ValueError("Smooth SE(3) refinement requires two-frame depth constraints")

    frame_count = len(poses)
    knot_count = min(knot_count, frame_count)
    frame_indices = np.arange(frame_count)
    knot_indices = np.linspace(0, frame_count - 1, knot_count, dtype=int)
    source_positions = poses[:, :3, 3]
    translation_curvature_sigma = np.array(
        (xy_curvature_sigma_m, xy_curvature_sigma_m, z_curvature_sigma_m),
        dtype=np.float64,
    )
    rotation_curvature_sigma = np.deg2rad(
        np.array(
            (
                roll_pitch_curvature_sigma_deg,
                roll_pitch_curvature_sigma_deg,
                yaw_curvature_sigma_deg,
            ),
            dtype=np.float64,
        )
    )

    def build_poses(
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        scale = float(np.exp(values[0]))
        knot_values = values[1:].reshape(knot_count - 1, 6)
        knot_translation = np.vstack(([0.0, 0.0, 0.0], knot_values[:, :3]))
        knot_rotation = np.vstack(([0.0, 0.0, 0.0], knot_values[:, 3:]))
        translation_correction = np.column_stack(
            [
                np.interp(frame_indices, knot_indices, knot_translation[:, axis])
                for axis in range(3)
            ]
        )
        rotation_correction = np.column_stack(
            [
                np.interp(frame_indices, knot_indices, knot_rotation[:, axis])
                for axis in range(3)
            ]
        )
        refined = poses.copy()
        scaled_positions = source_positions.copy()
        scaled_positions[:, :2] = (
            source_positions[0, :2]
            + scale * (source_positions[:, :2] - source_positions[0, :2])
        )
        refined[:, :3, 3] = scaled_positions + translation_correction
        correction_rotations = Rotation.from_rotvec(rotation_correction).as_matrix()
        refined[:, :3, :3] = correction_rotations @ poses[:, :3, :3]
        return (
            refined,
            translation_correction,
            rotation_correction,
            scale,
            knot_translation,
            knot_rotation,
        )

    def measure_errors(refined: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        translation_errors = []
        rotation_errors = []
        for constraint in constraints:
            relative = np.linalg.inv(refined[constraint.second]) @ refined[constraint.first]
            measured_translation = np.asarray(
                constraint.relative_camera_translation_m, dtype=np.float64
            )
            measured_rotation = np.asarray(
                constraint.relative_camera_rotation, dtype=np.float64
            )
            translation_errors.append(
                float(np.linalg.norm(relative[:3, 3] - measured_translation))
            )
            rotation_errors.append(
                rotation_error_deg(relative[:3, :3], measured_rotation)
            )
        return np.asarray(translation_errors), np.asarray(rotation_errors)

    def residuals(values: np.ndarray) -> np.ndarray:
        (
            refined,
            _,
            _,
            scale,
            knot_translation,
            knot_rotation,
        ) = build_poses(values)
        output = []
        for constraint in constraints:
            relative = np.linalg.inv(refined[constraint.second]) @ refined[constraint.first]
            measured_translation = np.asarray(
                constraint.relative_camera_translation_m, dtype=np.float64
            )
            measured_rotation = np.asarray(
                constraint.relative_camera_rotation, dtype=np.float64
            )
            output.extend(
                (relative[:3, 3] - measured_translation) / constraint.sigma_m
            )
            output.extend(
                Rotation.from_matrix(
                    relative[:3, :3] @ measured_rotation.T
                ).as_rotvec()
                / np.deg2rad(1.2)
            )

        output.extend(
            (
                np.diff(knot_translation, n=2, axis=0)
                / translation_curvature_sigma
            ).ravel()
        )
        output.extend(
            (
                np.diff(knot_rotation, n=2, axis=0) / rotation_curvature_sigma
            ).ravel()
        )
        output.extend((knot_translation[1:] / np.array((1.0, 1.0, 0.20))).ravel())
        output.extend(
            (
                knot_rotation[1:]
                / np.deg2rad(np.array((15.0, 15.0, 25.0)))
            ).ravel()
        )
        output.append(np.log(scale / 0.58) / 0.2)
        return np.asarray(output, dtype=np.float64)

    def knot_dependencies(frame: int) -> list[int]:
        right = int(np.searchsorted(knot_indices, frame, side="right"))
        left = min(max(right - 1, 0), knot_count - 2)
        return [left, left + 1]

    def make_jacobian_sparsity():
        residual_count = (
            6 * len(constraints)
            + 6 * (knot_count - 2)
            + 6 * (knot_count - 1)
            + 1
        )
        variable_count = 1 + 6 * (knot_count - 1)
        sparsity = lil_matrix((residual_count, variable_count), dtype=np.int8)

        def mark(
            row_start: int,
            row_count: int,
            knots: list[int],
            components: range,
            include_scale: bool = False,
        ) -> None:
            if include_scale:
                sparsity[row_start : row_start + row_count, 0] = 1
            for knot in knots:
                if knot == 0:
                    continue
                variable_start = 1 + 6 * (knot - 1)
                for component in components:
                    sparsity[
                        row_start : row_start + row_count,
                        variable_start + component,
                    ] = 1

        row = 0
        for constraint in constraints:
            dependencies = knot_dependencies(constraint.first) + knot_dependencies(
                constraint.second
            )
            mark(row, 6, dependencies, range(6), include_scale=True)
            row += 6
        for knot in range(knot_count - 2):
            mark(row, 3, [knot, knot + 1, knot + 2], range(3))
            row += 3
        for knot in range(knot_count - 2):
            mark(row, 3, [knot, knot + 1, knot + 2], range(3, 6))
            row += 3
        for knot in range(1, knot_count):
            mark(row, 3, [knot], range(3))
            row += 3
        for knot in range(1, knot_count):
            mark(row, 3, [knot], range(3, 6))
            row += 3
        sparsity[row, 0] = 1
        row += 1
        if row != residual_count:
            raise AssertionError("Unexpected SE(3) residual layout")
        return sparsity.tocsr()

    print("Computing smooth planar initialization for SE(3)", flush=True)
    planar_poses, planar_report = optimize_smooth_3d_trajectory(
        poses,
        constraints,
        knot_count,
        xy_curvature_sigma_m,
        yaw_curvature_sigma_deg,
    )
    planar_scale = float(planar_report["xy_scale"])
    planar_scaled_positions = source_positions.copy()
    planar_scaled_positions[:, :2] = (
        source_positions[0, :2]
        + planar_scale * (source_positions[:, :2] - source_positions[0, :2])
    )
    planar_translation_correction = planar_poses[:, :3, 3] - planar_scaled_positions
    planar_rotation_correction = Rotation.from_matrix(
        planar_poses[:, :3, :3] @ np.swapaxes(poses[:, :3, :3], 1, 2)
    ).as_rotvec()
    initial_knot_translation = planar_translation_correction[knot_indices]
    initial_knot_rotation = planar_rotation_correction[knot_indices]
    initial_values = np.r_[
        np.log(planar_scale),
        np.column_stack(
            (initial_knot_translation[1:], initial_knot_rotation[1:])
        ).reshape(-1),
    ]
    initial_poses, *_ = build_poses(initial_values)
    initial_translation_errors, initial_rotation_errors = measure_errors(initial_poses)
    print(
        f"Optimizing smooth SE(3): constraints={len(constraints)} "
        f"knots={knot_count} variables={len(initial_values)}",
        flush=True,
    )
    result = least_squares(
        residuals,
        initial_values,
        jac_sparsity=make_jacobian_sparsity(),
        tr_solver="lsmr",
        loss="huber",
        f_scale=1.0,
        max_nfev=max_nfev,
        verbose=0,
    )
    print(
        f"Finished smooth SE(3) optimization: success={result.success} "
        f"evaluations={result.nfev}",
        flush=True,
    )
    (
        refined,
        translation_correction,
        rotation_correction,
        scale,
        knot_translation,
        knot_rotation,
    ) = build_poses(result.x)
    translation_errors, rotation_errors = measure_errors(refined)
    rotation_correction_euler_deg = Rotation.from_rotvec(rotation_correction).as_euler(
        "xyz", degrees=True
    )
    knot_rotation_euler_deg = Rotation.from_rotvec(knot_rotation).as_euler(
        "xyz", degrees=True
    )
    report = {
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
        "optimizer_cost": float(result.cost),
        "visual_constraints": len(constraints),
        "constraint_type": "depth_3d3d",
        "initial_xy_scale": 0.58,
        "xy_scale": scale,
        "smooth_knot_count": knot_count,
        "smooth_knot_indices": knot_indices.tolist(),
        "smooth_xy_curvature_sigma_m": xy_curvature_sigma_m,
        "smooth_z_curvature_sigma_m": z_curvature_sigma_m,
        "smooth_yaw_curvature_sigma_deg": yaw_curvature_sigma_deg,
        "smooth_roll_pitch_curvature_sigma_deg": roll_pitch_curvature_sigma_deg,
        "smooth_se3_max_nfev": max_nfev,
        "planar_initialization": {
            "xy_scale": planar_scale,
            "visual_translation_residual_median_m": planar_report[
                "visual_translation_residual_median_after_m"
            ],
            "visual_rotation_residual_median_deg": planar_report[
                "visual_rotation_residual_median_after_deg"
            ],
        },
        "visual_translation_residual_median_before_m": float(
            np.median(initial_translation_errors)
        ),
        "visual_translation_residual_median_after_m": float(
            np.median(translation_errors)
        ),
        "visual_rotation_residual_median_before_deg": float(
            np.median(initial_rotation_errors)
        ),
        "visual_rotation_residual_median_after_deg": float(
            np.median(rotation_errors)
        ),
        "visual_residual_median_before_m": float(
            np.median(initial_translation_errors)
        ),
        "visual_residual_median_after_m": float(np.median(translation_errors)),
        "full_path_length_before_m": float(
            np.linalg.norm(np.diff(source_positions, axis=0), axis=1).sum()
        ),
        "full_path_length_after_m": float(
            np.linalg.norm(np.diff(refined[:, :3, 3], axis=0), axis=1).sum()
        ),
        "full_xy_path_length_before_m": float(
            np.linalg.norm(np.diff(source_positions[:, :2], axis=0), axis=1).sum()
        ),
        "full_xy_path_length_after_m": float(
            np.linalg.norm(np.diff(refined[:, :2, 3], axis=0), axis=1).sum()
        ),
        "translation_correction_p5_p95_m": np.percentile(
            translation_correction, [5, 95], axis=0
        ).tolist(),
        "rotation_correction_p5_p95_deg": np.percentile(
            rotation_correction_euler_deg, [5, 95], axis=0
        ).tolist(),
        "knot_translation_correction_m": knot_translation.tolist(),
        "knot_rotation_correction_euler_deg": knot_rotation_euler_deg.tolist(),
    }
    return refined, report


def interpolate_corrections(
    keyframes: list[int], initial_xy: np.ndarray, refined_xy: np.ndarray, poses: np.ndarray
) -> np.ndarray:
    correction = refined_xy - initial_xy
    all_indices = np.arange(len(poses))
    correction_x = np.interp(all_indices, keyframes, correction[:, 0])
    correction_y = np.interp(all_indices, keyframes, correction[:, 1])
    refined_poses = poses.copy()
    refined_poses[:, 0, 3] += correction_x
    refined_poses[:, 1, 3] += correction_y
    return refined_poses


def ensure_output_directory(output: Path, overwrite: bool):
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def write_output_dataset(
    source: Path,
    output: Path,
    refined_poses: np.ndarray,
    report: dict,
):
    for name in (
        "rgb",
        "depth",
        "stereo_right",
        "camera_info.json",
        "source_manifest.json",
        "keyframe_selection_report.json",
        "floor_calibration_application.json",
        "foundation_stereo_nominal_run.json",
    ):
        input_path = source / name
        if input_path.exists():
            (output / name).symlink_to(input_path.resolve(), target_is_directory=input_path.is_dir())

    tick_index = json.loads((source / "tick_index.json").read_text())
    tick_index = copy.deepcopy(tick_index)
    mode = report.get("mode", "trajectory_refinement")
    if mode == "smooth-3d":
        method = "rgbd_depth_3d3d_smooth_trajectory"
    elif mode == "smooth-se3":
        method = "rgbd_depth_3d3d_smooth_se3_trajectory"
    elif mode == "pose-graph-3d":
        method = "rgbd_depth_3d3d_pose_graph"
    else:
        method = f"rgbd_pnp_{mode}"
    tick_index["trajectory_refinement"] = {
        "method": method,
        "report": "trajectory_refinement.json",
    }
    (output / "tick_index.json").write_text(json.dumps(tick_index, indent=2) + "\n")

    pose_directory = output / "pose"
    pose_directory.mkdir()
    pose_timestamps = source / "pose" / "pose_timestamps_ns.txt"
    if not pose_timestamps.is_file():
        raise FileNotFoundError(
            "Absolute pose timestamps are required: "
            f"{pose_timestamps}"
        )
    shutil.copy2(pose_timestamps, pose_directory / pose_timestamps.name)
    pose_directory.joinpath("poses.txt").write_text(
        "".join(
            " ".join(f"{value:.12g}" for value in pose.reshape(-1)) + "\n"
            for pose in refined_poses
        )
    )
    (output / "trajectory_refinement.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    args = parse_args()
    cv2.setRNGSeed(0)
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    poses = load_poses(dataset / "pose" / "poses.txt")
    camera = json.loads((dataset / "camera_info.json").read_text())
    if camera.get("model") != "pinhole":
        raise ValueError("RGB-D trajectory refinement requires pinhole input")
    camera_matrix = np.asarray(camera["intrinsics"], dtype=np.float64)
    tick_index = json.loads((dataset / "tick_index.json").read_text())
    max_depth_m = args.max_depth_m or float(
        tick_index.get("recommended_max_depth_m", 3.0)
    )

    keyframes = select_keyframes(
        poses, args.keyframe_distance_m, args.max_keyframe_gap
    )
    pairs = make_pairs(
        keyframes,
        poses,
        args.loop_radius_m,
        args.loop_min_keyframe_separation,
        args.max_loop_candidates_per_keyframe,
        args.local_neighbor_span,
    )
    if args.mode in ("smooth-3d", "smooth-se3", "pose-graph-3d"):
        # The original odometry proposes no reliable revisit candidates here;
        # retain only local links so repeated office texture cannot form a
        # spurious global constraint.
        pairs = [pair for pair in pairs if not pair[2]]
    print(f"Selected {len(keyframes)} keyframes and {len(pairs)} candidate pairs", flush=True)

    cache = FrameCache(dataset)
    constraint_estimator = (
        estimate_3d_constraint
        if args.mode in ("smooth-3d", "smooth-se3", "pose-graph-3d")
        else estimate_constraint
    )
    constraints = []
    for pair_index, (first_key, second_key, loop) in enumerate(pairs, start=1):
        first = keyframes[first_key]
        second = keyframes[second_key]
        constraint = constraint_estimator(
            cache,
            first,
            second,
            poses,
            camera_matrix,
            max_depth_m,
            args.ratio_test,
            args.min_inliers,
            args.max_rotation_error_deg,
            loop,
        )
        if constraint is not None:
            constraints.append(constraint)
        if pair_index % 25 == 0 or pair_index == len(pairs):
            print(
                f"Measured {pair_index}/{len(pairs)} pairs; accepted {len(constraints)}",
                flush=True,
            )
    if len(constraints) < max(12, len(keyframes) // 4):
        raise RuntimeError(
            f"Too few reliable RGB-D constraints: {len(constraints)} for {len(keyframes)} keyframes"
        )

    initial_xy = poses[keyframes, :2, 3]
    if args.mode == "smooth-3d":
        refined_poses, optimizer_report = optimize_smooth_3d_trajectory(
            poses,
            constraints,
            args.smooth_knot_count,
            args.smooth_xy_curvature_sigma_m,
            args.smooth_yaw_curvature_sigma_deg,
        )
        refinement_report = {"mode": args.mode, **optimizer_report}
    elif args.mode == "smooth-se3":
        refined_poses, optimizer_report = optimize_smooth_se3_trajectory(
            poses,
            constraints,
            args.smooth_knot_count,
            args.smooth_xy_curvature_sigma_m,
            args.smooth_z_curvature_sigma_m,
            args.smooth_yaw_curvature_sigma_deg,
            args.smooth_roll_pitch_curvature_sigma_deg,
            args.smooth_se3_max_nfev,
        )
        refinement_report = {"mode": args.mode, **optimizer_report}
    elif args.mode in ("visual-odometry", "pose-graph-3d"):
        refined_xy, refined_yaw, optimizer_report = optimize_visual_odometry(
            keyframes,
            poses,
            constraints,
            args.odom_sigma_m,
            args.position_prior_sigma_m,
            args.visual_max_nfev,
        )
        refined_poses = interpolate_visual_odometry(
            keyframes, refined_xy, refined_yaw, poses
        )
        refinement_report = {
            "mode": args.mode,
            "constraint_type": (
                "depth_3d3d" if args.mode == "pose-graph-3d" else "pnp_3d2d"
            ),
            **optimizer_report,
        }
    elif args.mode == "free-form":
        refined_xy, optimizer_report = optimize_keyframe_positions(
            keyframes,
            poses,
            constraints,
            args.odom_sigma_m,
            args.position_prior_sigma_m,
        )
        refined_poses = interpolate_corrections(
            keyframes, initial_xy, refined_xy, poses
        )
        refinement_report = {"mode": args.mode, **optimizer_report}
    else:
        scale, scale_report = estimate_global_xy_scale(keyframes, poses, constraints)
        refined_poses = apply_global_xy_scale(poses, scale)
        refined_xy = refined_poses[keyframes, :2, 3]
        residual_before = []
        residual_after = []
        for constraint in constraints:
            delta_visual = np.asarray(constraint.delta_world_xy)
            residual_before.append(
                np.linalg.norm(
                    poses[constraint.second, :2, 3]
                    - poses[constraint.first, :2, 3]
                    - delta_visual
                )
            )
            residual_after.append(
                np.linalg.norm(
                    refined_poses[constraint.second, :2, 3]
                    - refined_poses[constraint.first, :2, 3]
                    - delta_visual
                )
            )
        refinement_report = {
            "mode": args.mode,
            "visual_constraints": len(constraints),
            "visual_residual_median_before_m": float(np.median(residual_before)),
            "visual_residual_median_after_m": float(np.median(residual_after)),
            "keyframe_path_length_before_m": float(
                np.linalg.norm(np.diff(initial_xy, axis=0), axis=1).sum()
            ),
            "keyframe_path_length_after_m": float(
                np.linalg.norm(np.diff(refined_xy, axis=0), axis=1).sum()
            ),
            **scale_report,
        }
    report = {
        "source_dataset": str(dataset),
        "output_dataset": str(output),
        "projection_model": tick_index.get("projection_model"),
        "max_depth_m": max_depth_m,
        "keyframe_count": len(keyframes),
        "keyframes": keyframes,
        "constraints": [asdict(constraint) for constraint in constraints],
        "loop_constraints": sum(constraint.loop for constraint in constraints),
        **refinement_report,
        "full_path_length_before_m": float(
            np.linalg.norm(np.diff(poses[:, :2, 3], axis=0), axis=1).sum()
        ),
        "full_path_length_after_m": float(
            np.linalg.norm(np.diff(refined_poses[:, :2, 3], axis=0), axis=1).sum()
        ),
    }
    ensure_output_directory(output, args.overwrite)
    write_output_dataset(dataset, output, refined_poses, report)
    print(
        "Refined RGB-D trajectory\n"
        f"  visual residual: {report['visual_residual_median_before_m']:.3f}m -> "
        f"{report['visual_residual_median_after_m']:.3f}m\n"
        f"  XY path length: {report['full_path_length_before_m']:.3f}m -> "
        f"{report['full_path_length_after_m']:.3f}m\n"
        f"  constraints: {len(constraints)} ({report['loop_constraints']} loops)\n"
        f"  output: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
