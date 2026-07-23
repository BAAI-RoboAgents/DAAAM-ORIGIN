"""Tests for joint mask/depth world-geometry recovery."""

from __future__ import annotations

import numpy as np

from daaam.realtime.masked_geometry import backproject_masked_depth


def test_backprojection_preserves_pixel_depth_correspondence() -> None:
    depth = np.zeros((4, 5), dtype=np.float64)
    mask = np.zeros_like(depth, dtype=bool)
    mask[1, 1] = True
    mask[2, 3] = True
    depth[1, 1] = 1.0
    depth[2, 3] = 3.0
    intrinsics = np.asarray(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transform = np.eye(4)
    transform[:3, 3] = [10.0, -2.0, 0.5]

    geometry = backproject_masked_depth(
        mask,
        depth,
        intrinsics,
        transform,
        maximum_points=None,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    expected = np.asarray([[10.5, -1.5, 1.5], [14.5, 1.0, 3.5]])
    assert np.allclose(geometry.points_world_m, expected)
    assert np.allclose(geometry.position_m, np.median(expected, axis=0))
    assert np.allclose(geometry.dimensions_m, expected[1] - expected[0])
    assert geometry.valid_pixel_count == 2


def test_backprojection_subsampling_is_deterministic() -> None:
    mask = np.ones((20, 20), dtype=bool)
    depth = np.ones((20, 20), dtype=np.float32)
    intrinsics = np.asarray(
        [[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]]
    )

    first = backproject_masked_depth(
        mask, depth, intrinsics, np.eye(4), maximum_points=37
    )
    second = backproject_masked_depth(
        mask, depth, intrinsics, np.eye(4), maximum_points=37
    )

    assert first.valid_pixel_count == 400
    assert len(first.points_world_m) == 37
    assert np.array_equal(first.pixel_yx, second.pixel_yx)
    assert np.array_equal(first.points_world_m, second.points_world_m)
