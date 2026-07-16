"""Metric stereo depth and explicit left/right consistency products."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StereoConfidence:
    confidence: np.ndarray
    valid_mask: np.ndarray
    consistent_mask: np.ndarray
    occlusion_mask: np.ndarray
    consistency_error_px: np.ndarray
    metrics: dict[str, float | int]


def _sample_horizontal(image: np.ndarray, x_coordinates: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x0 = np.floor(x_coordinates).astype(np.int64)
    x1 = x0 + 1
    alpha = x_coordinates - x0
    rows = np.arange(height)[:, None]
    x0_clipped = np.clip(x0, 0, width - 1)
    x1_clipped = np.clip(x1, 0, width - 1)
    return (
        (1.0 - alpha) * image[rows, x0_clipped]
        + alpha * image[rows, x1_clipped]
    )


def compute_left_right_confidence(
    left_disparity_px: np.ndarray,
    right_disparity_px: np.ndarray,
    *,
    absolute_tolerance_px: float = 0.75,
    relative_tolerance: float = 0.03,
) -> StereoConfidence:
    """Check d_left(x) against right disparity magnitude at x-d_left(x)."""
    left = np.asarray(left_disparity_px, dtype=np.float32)
    right = np.asarray(right_disparity_px, dtype=np.float32)
    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("left and right disparities must have matching HxW shapes")
    if absolute_tolerance_px <= 0 or relative_tolerance < 0:
        raise ValueError("left/right consistency tolerances are invalid")

    height, width = left.shape
    columns = np.arange(width, dtype=np.float32)[None, :]
    right_x = columns - left
    left_valid = np.isfinite(left) & (left > 0.0)
    in_bounds = (right_x >= 0.0) & (right_x <= width - 1)
    sampled_right = _sample_horizontal(right, right_x)
    right_valid = np.isfinite(sampled_right) & (sampled_right > 0.0)
    valid = left_valid & in_bounds & right_valid
    error = np.abs(left - sampled_right)
    error[~valid] = np.inf
    tolerance = absolute_tolerance_px + relative_tolerance * np.abs(left)
    consistent = valid & (error <= tolerance)
    occluded = left_valid & (~in_bounds | ~right_valid)
    confidence = np.zeros(left.shape, dtype=np.float32)
    confidence[valid] = np.exp(
        -np.square(error[valid] / np.maximum(tolerance[valid], 1e-6))
    )
    confidence[~consistent] = 0.0
    metrics: dict[str, float | int] = {
        "pixels": int(left.size),
        "left_valid_pixels": int(left_valid.sum()),
        "consistent_pixels": int(consistent.sum()),
        "occlusion_pixels": int(occluded.sum()),
        "left_right_consistency": float(
            consistent.sum() / max(1, int(left_valid.sum()))
        ),
        "occlusion_ratio": float(occluded.mean()),
        "median_consistency_error_px": (
            float(np.median(error[valid])) if np.any(valid) else 0.0
        ),
    }
    return StereoConfidence(
        confidence,
        valid,
        consistent,
        occluded,
        error.astype(np.float32),
        metrics,
    )


def disparity_to_metric_depth(
    disparity_px: np.ndarray,
    *,
    focal_length_px: float,
    baseline_m: float,
    maximum_depth_m: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    disparity = np.asarray(disparity_px, dtype=np.float32)
    if disparity.ndim != 2:
        raise ValueError("disparity must be HxW")
    if focal_length_px <= 0 or baseline_m <= 0 or maximum_depth_m <= 0:
        raise ValueError("stereo geometry and maximum depth must be positive")
    valid = np.isfinite(disparity) & (disparity > 0.0)
    if valid_mask is not None:
        requested = np.asarray(valid_mask, dtype=bool)
        if requested.shape != disparity.shape:
            raise ValueError("valid_mask shape does not match disparity")
        valid &= requested
    depth = np.zeros(disparity.shape, dtype=np.float32)
    depth[valid] = focal_length_px * baseline_m / disparity[valid]
    valid &= np.isfinite(depth) & (depth > 0.0) & (depth <= maximum_depth_m)
    depth[~valid] = 0.0
    return depth
