"""Category-independent motion masks from camera-flow prediction residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    residual_threshold_px: float = 2.5
    forward_backward_threshold_px: float = 1.5
    minimum_dynamic_pixels: int = 24
    morphology_kernel: int = 3

    def __post_init__(self) -> None:
        if self.residual_threshold_px <= 0:
            raise ValueError("residual threshold must be positive")
        if self.forward_backward_threshold_px <= 0:
            raise ValueError("forward/backward threshold must be positive")
        if self.minimum_dynamic_pixels < 0:
            raise ValueError("minimum_dynamic_pixels must be non-negative")
        if self.morphology_kernel < 1 or self.morphology_kernel % 2 == 0:
            raise ValueError("morphology_kernel must be a positive odd number")


@dataclass(frozen=True)
class MotionClassification:
    static_mask: np.ndarray
    dynamic_mask: np.ndarray
    unknown_mask: np.ndarray
    residual_px: np.ndarray
    expected_flow: np.ndarray
    actual_flow: np.ndarray
    metrics: dict[str, float | int]

    def __post_init__(self) -> None:
        shape = self.static_mask.shape
        if self.dynamic_mask.shape != shape or self.unknown_mask.shape != shape:
            raise ValueError("motion masks must have identical shapes")
        if np.any(self.static_mask & self.dynamic_mask):
            raise ValueError("static and dynamic masks overlap")
        if np.any(self.static_mask & self.unknown_mask):
            raise ValueError("static and unknown masks overlap")
        if np.any(self.dynamic_mask & self.unknown_mask):
            raise ValueError("dynamic and unknown masks overlap")


def predict_background_flow(
    depth_m: np.ndarray,
    world_T_previous_camera: np.ndarray,
    world_T_current_camera: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project previous-frame depth into the current camera."""
    depth = np.asarray(depth_m, dtype=np.float64)
    previous = np.asarray(world_T_previous_camera, dtype=np.float64)
    current = np.asarray(world_T_current_camera, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must be HxW")
    if previous.shape != (4, 4) or current.shape != (4, 4):
        raise ValueError("camera transforms must be 4x4")
    if camera_matrix.shape != (3, 3):
        raise ValueError("intrinsics must be 3x3")

    height, width = depth.shape
    rows, columns = np.indices((height, width), dtype=np.float64)
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    valid = np.isfinite(depth) & (depth > 0.0) & (fx > 0.0) & (fy > 0.0)
    points = np.stack(
        (
            (columns - cx) * depth / fx,
            (rows - cy) * depth / fy,
            depth,
            np.ones_like(depth),
        ),
        axis=-1,
    )
    current_T_previous = np.linalg.inv(current) @ previous
    transformed = points @ current_T_previous.T
    z = transformed[..., 2]
    valid &= np.isfinite(transformed).all(axis=-1) & (z > 1e-6)
    projected_u = fx * transformed[..., 0] / np.where(valid, z, 1.0) + cx
    projected_v = fy * transformed[..., 1] / np.where(valid, z, 1.0) + cy
    boundary_tolerance = 1e-5
    valid &= (
        (projected_u >= -boundary_tolerance)
        & (projected_u <= width - 1 + boundary_tolerance)
        & (projected_v >= -boundary_tolerance)
        & (projected_v <= height - 1 + boundary_tolerance)
    )
    flow = np.stack((projected_u - columns, projected_v - rows), axis=-1)
    flow[~valid] = np.nan
    return flow.astype(np.float32), valid


def classify_flow_residual(
    actual_flow: np.ndarray,
    expected_flow: np.ndarray,
    valid_background: np.ndarray,
    *,
    forward_backward_error: Optional[np.ndarray] = None,
    config: MotionConfig = MotionConfig(),
) -> MotionClassification:
    """Split pixels into static, dynamic, and conservative unknown classes."""
    actual = np.asarray(actual_flow, dtype=np.float32)
    expected = np.asarray(expected_flow, dtype=np.float32)
    valid = np.asarray(valid_background, dtype=bool)
    if actual.shape != expected.shape or actual.ndim != 3 or actual.shape[-1] != 2:
        raise ValueError("flows must have matching HxWx2 shapes")
    if valid.shape != actual.shape[:2]:
        raise ValueError("valid_background shape does not match flow")

    reliable = valid & np.isfinite(actual).all(axis=-1) & np.isfinite(expected).all(axis=-1)
    if forward_backward_error is not None:
        fb_error = np.asarray(forward_backward_error, dtype=np.float32)
        if fb_error.shape != valid.shape:
            raise ValueError("forward_backward_error shape mismatch")
        reliable &= np.isfinite(fb_error) & (
            fb_error <= config.forward_backward_threshold_px
        )

    residual = np.linalg.norm(actual - expected, axis=-1)
    residual[~np.isfinite(residual)] = np.inf
    dynamic = reliable & (residual > config.residual_threshold_px)
    if 0 < int(dynamic.sum()) < config.minimum_dynamic_pixels:
        # Tiny unsupported residual islands are uncertain, not silently static.
        reliable = reliable & ~dynamic
        dynamic[:] = False
    static = reliable & ~dynamic
    unknown = ~reliable
    total = max(1, valid.size)
    metrics: dict[str, float | int] = {
        "pixels": int(valid.size),
        "static_pixels": int(static.sum()),
        "dynamic_pixels": int(dynamic.sum()),
        "unknown_pixels": int(unknown.sum()),
        "static_ratio": float(static.sum() / total),
        "dynamic_ratio": float(dynamic.sum() / total),
        "unknown_ratio": float(unknown.sum() / total),
        "median_reliable_residual_px": (
            float(np.median(residual[reliable])) if np.any(reliable) else 0.0
        ),
    }
    return MotionClassification(
        static,
        dynamic,
        unknown,
        residual.astype(np.float32),
        expected,
        actual,
        metrics,
    )


def estimate_motion_masks(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    previous_depth_m: np.ndarray,
    world_T_previous_camera: np.ndarray,
    world_T_current_camera: np.ndarray,
    intrinsics: np.ndarray,
    *,
    config: MotionConfig = MotionConfig(),
) -> MotionClassification:
    """Compute optical flow and compare it with pose-predicted background flow."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required to estimate optical flow") from error

    previous = np.asarray(previous_rgb)
    current = np.asarray(current_rgb)
    if previous.shape != current.shape or previous.ndim != 3:
        raise ValueError("RGB frames must have matching HxWxC shapes")
    conversion = cv2.COLOR_RGB2GRAY if previous.shape[2] == 3 else None
    if conversion is None:
        raise ValueError("RGB frames must have three channels")
    previous_gray = cv2.cvtColor(previous, conversion)
    current_gray = cv2.cvtColor(current, conversion)
    forward = cv2.calcOpticalFlowFarneback(
        previous_gray, current_gray, None, 0.5, 4, 21, 3, 5, 1.2, 0
    )
    backward = cv2.calcOpticalFlowFarneback(
        current_gray, previous_gray, None, 0.5, 4, 21, 3, 5, 1.2, 0
    )
    rows, columns = np.indices(previous_gray.shape, dtype=np.float32)
    target_x = columns + forward[..., 0]
    target_y = rows + forward[..., 1]
    sampled_backward = cv2.remap(
        backward,
        target_x,
        target_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    fb_error = np.linalg.norm(forward + sampled_backward, axis=-1)
    expected, valid = predict_background_flow(
        previous_depth_m,
        world_T_previous_camera,
        world_T_current_camera,
        intrinsics,
    )
    result = classify_flow_residual(
        forward,
        expected,
        valid,
        forward_backward_error=fb_error,
        config=config,
    )
    if config.morphology_kernel > 1 and np.any(result.dynamic_mask):
        kernel = np.ones(
            (config.morphology_kernel, config.morphology_kernel), np.uint8
        )
        dynamic = cv2.morphologyEx(
            result.dynamic_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
        dynamic &= ~result.unknown_mask
        static = ~result.unknown_mask & ~dynamic
        metrics = dict(result.metrics)
        metrics["dynamic_pixels"] = int(dynamic.sum())
        metrics["static_pixels"] = int(static.sum())
        metrics["dynamic_ratio"] = float(dynamic.mean())
        metrics["static_ratio"] = float(static.mean())
        result = MotionClassification(
            static,
            dynamic,
            result.unknown_mask,
            result.residual_px,
            result.expected_flow,
            result.actual_flow,
            metrics,
        )
    return result
