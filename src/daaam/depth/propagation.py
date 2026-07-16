"""Pose-based z-buffer propagation for frames between stereo keyframes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class DepthPropagationConfig:
    maximum_age_s: float = 0.35
    maximum_translation_m: float = 0.20
    maximum_rotation_deg: float = 8.0
    minimum_output_valid_ratio: float = 0.10
    confidence_decay_s: float = 0.50

    def __post_init__(self) -> None:
        if min(
            self.maximum_age_s,
            self.maximum_translation_m,
            self.maximum_rotation_deg,
            self.confidence_decay_s,
        ) <= 0:
            raise ValueError("depth propagation limits must be positive")
        if not 0.0 <= self.minimum_output_valid_ratio <= 1.0:
            raise ValueError("minimum_output_valid_ratio must be in [0, 1]")


@dataclass(frozen=True)
class PropagatedDepth:
    depth_m: np.ndarray
    confidence: np.ndarray
    source_index: np.ndarray
    needs_keyframe: bool
    reason: str
    metrics: dict[str, float | int | bool]


def propagate_depth(
    depth_m: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    world_T_source_camera: np.ndarray,
    world_T_target_camera: np.ndarray,
    *,
    source_time_ns: int,
    target_time_ns: int,
    config: DepthPropagationConfig = DepthPropagationConfig(),
) -> PropagatedDepth:
    depth = np.asarray(depth_m, dtype=np.float32)
    weights = np.asarray(confidence, dtype=np.float32)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    source_pose = np.asarray(world_T_source_camera, dtype=np.float64)
    target_pose = np.asarray(world_T_target_camera, dtype=np.float64)
    if depth.ndim != 2 or weights.shape != depth.shape:
        raise ValueError("depth and confidence must have matching HxW shapes")
    if camera_matrix.shape != (3, 3):
        raise ValueError("intrinsics must be 3x3")
    if source_pose.shape != (4, 4) or target_pose.shape != (4, 4):
        raise ValueError("camera poses must be 4x4")
    if source_time_ns <= 0 or target_time_ns <= source_time_ns:
        raise ValueError("propagation times must be positive and increasing")

    age_s = (target_time_ns - source_time_ns) / 1e9
    target_T_source = np.linalg.inv(target_pose) @ source_pose
    translation_m = float(np.linalg.norm(target_T_source[:3, 3]))
    rotation_deg = float(
        np.rad2deg(Rotation.from_matrix(target_T_source[:3, :3]).magnitude())
    )
    empty_depth = np.zeros_like(depth)
    empty_confidence = np.zeros_like(weights)
    empty_source = np.full(depth.shape, -1, dtype=np.int64)
    base_metrics: dict[str, float | int | bool] = {
        "age_s": age_s,
        "translation_m": translation_m,
        "rotation_deg": rotation_deg,
        "input_valid_ratio": float(np.mean((depth > 0.0) & (weights > 0.0))),
    }
    if age_s > config.maximum_age_s:
        return PropagatedDepth(
            empty_depth, empty_confidence, empty_source, True, "maximum_age", base_metrics
        )
    if translation_m > config.maximum_translation_m:
        return PropagatedDepth(
            empty_depth,
            empty_confidence,
            empty_source,
            True,
            "maximum_translation",
            base_metrics,
        )
    if rotation_deg > config.maximum_rotation_deg:
        return PropagatedDepth(
            empty_depth,
            empty_confidence,
            empty_source,
            True,
            "maximum_rotation",
            base_metrics,
        )

    height, width = depth.shape
    rows, columns = np.indices((height, width), dtype=np.float64)
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    valid = (
        np.isfinite(depth)
        & (depth > 0.0)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    source_indices = np.flatnonzero(valid)
    if not len(source_indices):
        return PropagatedDepth(
            empty_depth,
            empty_confidence,
            empty_source,
            True,
            "no_valid_source_depth",
            base_metrics,
        )
    z = depth.ravel()[source_indices].astype(np.float64)
    u = columns.ravel()[source_indices]
    v = rows.ravel()[source_indices]
    points = np.c_[
        (u - cx) * z / fx,
        (v - cy) * z / fy,
        z,
        np.ones(len(z)),
    ]
    transformed = points @ target_T_source.T
    target_z = transformed[:, 2]
    projected_u = fx * transformed[:, 0] / np.where(target_z > 0, target_z, 1.0) + cx
    projected_v = fy * transformed[:, 1] / np.where(target_z > 0, target_z, 1.0) + cy
    projected_columns = np.rint(projected_u).astype(np.int64)
    projected_rows = np.rint(projected_v).astype(np.int64)
    projects = (
        np.isfinite(transformed).all(axis=1)
        & (target_z > 1e-6)
        & (projected_columns >= 0)
        & (projected_columns < width)
        & (projected_rows >= 0)
        & (projected_rows < height)
    )
    source_indices = source_indices[projects]
    target_z = target_z[projects]
    target_linear = projected_rows[projects] * width + projected_columns[projects]
    if not len(target_linear):
        return PropagatedDepth(
            empty_depth,
            empty_confidence,
            empty_source,
            True,
            "all_points_out_of_view",
            base_metrics,
        )

    # Sort by target pixel and then depth; the first item is the visible surface.
    order = np.lexsort((target_z, target_linear))
    target_sorted = target_linear[order]
    first = np.r_[True, target_sorted[1:] != target_sorted[:-1]]
    selected = order[first]
    target_selected = target_linear[selected]
    source_selected = source_indices[selected]
    depth_output = np.zeros(depth.size, dtype=np.float32)
    confidence_output = np.zeros(depth.size, dtype=np.float32)
    source_output = np.full(depth.size, -1, dtype=np.int64)
    depth_output[target_selected] = target_z[selected].astype(np.float32)
    motion_decay = np.exp(-age_s / config.confidence_decay_s)
    confidence_output[target_selected] = (
        weights.ravel()[source_selected] * motion_decay
    )
    source_output[target_selected] = source_selected
    depth_output = depth_output.reshape(depth.shape)
    confidence_output = confidence_output.reshape(depth.shape)
    source_output = source_output.reshape(depth.shape)
    output_ratio = float(np.mean(depth_output > 0.0))
    needs_keyframe = output_ratio < config.minimum_output_valid_ratio
    reason = "low_propagated_coverage" if needs_keyframe else "propagated"
    metrics = {
        **base_metrics,
        "output_valid_ratio": output_ratio,
        "propagated_pixels": int(np.count_nonzero(depth_output)),
        "confidence_decay": float(motion_decay),
        "needs_keyframe": needs_keyframe,
    }
    return PropagatedDepth(
        depth_output,
        confidence_output,
        source_output,
        needs_keyframe,
        reason,
        metrics,
    )
