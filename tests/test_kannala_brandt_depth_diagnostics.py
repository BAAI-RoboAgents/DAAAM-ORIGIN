"""Tests for the camera-only Kannala-Brandt depth controls."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from reconstruct_kannala_brandt_depth import (  # noqa: E402
    kannala_brandt_rays,
    triangulate_depth,
)


def test_zero_coefficient_kannala_brandt_is_equidistant_not_pinhole():
    fx = fy = 100.0
    cx = cy = 0.0
    theta = 0.8
    u = np.array([fx * theta])
    ray = kannala_brandt_rays(
        u,
        np.zeros_like(u),
        fx,
        fy,
        cx,
        cy,
        np.zeros(4),
    )[0]
    assert ray[0] == pytest.approx(np.sin(theta))
    assert ray[2] == pytest.approx(np.cos(theta))
    assert ray[0] != pytest.approx(u[0] / fx / np.hypot(u[0] / fx, 1.0))


def test_kannala_brandt_ray_triangulation_recovers_metric_z():
    fx = fy = 100.0
    cx = 4.0
    cy = 3.0
    baseline = 0.1
    expected_z = 3.0
    cam0_u = cx
    cam1_u = cx + fx * np.arctan2(-baseline, expected_z)
    disparity = np.full((7, 9), cam0_u - cam1_u, dtype=np.float32)
    cam0_t_cam1 = np.eye(4)
    cam0_t_cam1[0, 3] = baseline
    depth = triangulate_depth(
        disparity,
        fx,
        fy,
        cx,
        cy,
        np.zeros(4),
        cam0_t_cam1,
    )
    assert depth[int(cy), int(cx)] == pytest.approx(expected_z, abs=1.0e-5)


def test_positive_reverse_disparity_has_no_forward_ray_intersection():
    fx = fy = 100.0
    cx = 4.0
    cy = 3.0
    disparity = np.full((7, 9), 3.0, dtype=np.float32)
    cam1_t_cam0 = np.eye(4)
    cam1_t_cam0[0, 3] = -0.1
    depth = triangulate_depth(
        disparity,
        fx,
        fy,
        cx,
        cy,
        np.zeros(4),
        cam1_t_cam0,
    )
    assert not np.isfinite(depth[int(cy), int(cx)])
