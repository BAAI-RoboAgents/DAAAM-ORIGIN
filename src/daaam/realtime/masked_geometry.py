"""Joint RGB-D backprojection for one semantic instance mask."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MaskedDepthGeometry:
    """Robust world-frame geometry sampled from matching mask/depth pixels."""

    position_m: np.ndarray
    dimensions_m: np.ndarray
    points_world_m: np.ndarray
    pixel_yx: np.ndarray
    valid_pixel_count: int


def _deterministic_indices(count: int, maximum: Optional[int]) -> np.ndarray:
    if maximum is None or count <= maximum:
        return np.arange(count, dtype=np.int64)
    if maximum <= 0:
        raise ValueError("maximum_points must be positive")
    return np.linspace(0, count - 1, num=maximum, dtype=np.int64)


def backproject_masked_depth(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    world_T_camera: np.ndarray,
    *,
    maximum_points: Optional[int] = 20_000,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    minimum_dimension_m: float = 0.05,
) -> MaskedDepthGeometry:
    """Backproject joint mask/depth pixels and estimate robust world geometry.

    The old path combined a median pixel with an independently computed median
    depth.  On floors, walls, and large objects those values need not belong to
    the same 3D ray.  This routine preserves each pixel-depth correspondence.
    """

    mask_array = np.asarray(mask, dtype=bool)
    depth_array = np.asarray(depth_m, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(world_T_camera, dtype=np.float64)
    if mask_array.ndim != 2 or depth_array.shape != mask_array.shape:
        raise ValueError("mask and depth must be matching 2D arrays")
    if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
        raise ValueError("intrinsics focal lengths must be positive")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("world_T_camera must be a finite 4x4 matrix")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("geometry quantiles are invalid")
    if minimum_dimension_m <= 0.0:
        raise ValueError("minimum_dimension_m must be positive")

    valid = mask_array & np.isfinite(depth_array) & (depth_array > 0.0)
    pixel_y, pixel_x = np.nonzero(valid)
    if not len(pixel_x):
        raise ValueError("semantic mask contains no valid depth pixels")
    valid_pixel_count = int(len(pixel_x))
    selected = _deterministic_indices(valid_pixel_count, maximum_points)
    pixel_x = pixel_x[selected]
    pixel_y = pixel_y[selected]
    z = depth_array[pixel_y, pixel_x]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    points_camera = np.column_stack(
        (
            (pixel_x.astype(np.float64) - cx) * z / fx,
            (pixel_y.astype(np.float64) - cy) * z / fy,
            z,
            np.ones_like(z),
        )
    )
    points_world = (transform @ points_camera.T).T[:, :3]
    if not np.isfinite(points_world).all():
        raise ValueError("backprojected mask geometry is not finite")
    lower = np.quantile(points_world, lower_quantile, axis=0)
    upper = np.quantile(points_world, upper_quantile, axis=0)
    dimensions = np.maximum(upper - lower, float(minimum_dimension_m))
    position = np.median(points_world, axis=0)
    return MaskedDepthGeometry(
        position_m=np.asarray(position, dtype=np.float64),
        dimensions_m=np.asarray(dimensions, dtype=np.float64),
        points_world_m=np.asarray(points_world, dtype=np.float32),
        pixel_yx=np.column_stack((pixel_y, pixel_x)).astype(np.int32, copy=False),
        valid_pixel_count=valid_pixel_count,
    )
