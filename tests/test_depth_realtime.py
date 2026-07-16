"""Acceptance tests for stereo confidence and non-keyframe propagation."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.depth import (  # noqa: E402
    DepthPropagationConfig,
    compute_left_right_confidence,
    disparity_to_metric_depth,
    propagate_depth,
)


ORIGIN_NS = 1_783_933_507_759_540_877
K = np.array([[100.0, 0.0, 4.5], [0.0, 100.0, 3.5], [0.0, 0.0, 1.0]])


def test_left_right_confidence_marks_consistency_occlusion_and_conflict():
    left = np.full((6, 10), 2.0, dtype=np.float32)
    right = np.full_like(left, 2.0)
    right[2:5, 3:6] = 5.0
    result = compute_left_right_confidence(left, right)
    assert np.all(result.occlusion_mask[:, :2])
    assert np.all(result.consistent_mask[0:2, 2:])
    # Left x=5..7 samples the conflicting right x=3..5 region.
    assert not np.any(result.consistent_mask[2:5, 5:8])
    assert np.all(result.confidence[2:5, 5:8] == 0.0)
    assert 0.0 < result.metrics["left_right_consistency"] < 1.0


def test_metric_depth_uses_scaled_focal_geometry_and_valid_mask():
    disparity = np.full((2, 3), 5.0, dtype=np.float32)
    valid = np.ones_like(disparity, dtype=bool)
    valid[0, 0] = False
    depth = disparity_to_metric_depth(
        disparity,
        focal_length_px=100.0,
        baseline_m=0.1,
        maximum_depth_m=5.0,
        valid_mask=valid,
    )
    assert depth[0, 0] == 0.0
    assert np.all(depth[valid] == pytest.approx(2.0))


def test_identity_pose_propagates_depth_without_spatial_error():
    depth = np.full((8, 10), 2.0, dtype=np.float32)
    confidence = np.ones_like(depth)
    result = propagate_depth(
        depth,
        confidence,
        K,
        np.eye(4),
        np.eye(4),
        source_time_ns=ORIGIN_NS,
        target_time_ns=ORIGIN_NS + 100_000_000,
        config=DepthPropagationConfig(minimum_output_valid_ratio=0.9),
    )
    assert not result.needs_keyframe
    assert result.reason == "propagated"
    assert np.allclose(result.depth_m, depth)
    assert np.all(result.confidence > 0.0)
    assert result.metrics["output_valid_ratio"] == 1.0


def test_propagation_requests_keyframe_when_time_or_view_change_is_unsafe():
    depth = np.ones((8, 10), dtype=np.float32)
    confidence = np.ones_like(depth)
    old = propagate_depth(
        depth,
        confidence,
        K,
        np.eye(4),
        np.eye(4),
        source_time_ns=ORIGIN_NS,
        target_time_ns=ORIGIN_NS + 1_000_000_000,
    )
    assert old.needs_keyframe
    assert old.reason == "maximum_age"
    moved_pose = np.eye(4)
    moved_pose[0, 3] = 1.0
    moved = propagate_depth(
        depth,
        confidence,
        K,
        np.eye(4),
        moved_pose,
        source_time_ns=ORIGIN_NS,
        target_time_ns=ORIGIN_NS + 100_000_000,
    )
    assert moved.needs_keyframe
    assert moved.reason == "maximum_translation"


def test_propagation_rejects_non_increasing_absolute_time():
    with pytest.raises(ValueError, match="increasing"):
        propagate_depth(
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            K,
            np.eye(4),
            np.eye(4),
            source_time_ns=ORIGIN_NS,
            target_time_ns=ORIGIN_NS,
        )
