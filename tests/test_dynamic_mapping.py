"""Acceptance tests for dynamic/static separation and object lifecycle."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.mapping.dynamic_layer import (  # noqa: E402
    DynamicLayer,
    DynamicLayerConfig,
    ObjectObservation,
    ObjectState,
)
from daaam.mapping.fusion import isolate_static_depth  # noqa: E402
from daaam.mapping.motion import (  # noqa: E402
    MotionConfig,
    classify_flow_residual,
    predict_background_flow,
)


ORIGIN_NS = 1_783_933_507_759_540_877


def observation(
    seconds: float,
    position,
    *,
    track_id: int = 1,
    entity_id: str | None = "snack-cart",
    motion_score: float = 0.0,
) -> ObjectObservation:
    return ObjectObservation(
        track_id=track_id,
        entity_id=entity_id,
        sensor_time_ns=ORIGIN_NS + int(seconds * 1e9),
        position_m=np.asarray(position, dtype=np.float64),
        dimensions_m=np.array([0.5, 0.4, 0.8]),
        position_covariance=np.eye(3) * 0.0025,
        semantic_probabilities={"cart": 0.8, "object": 0.2},
        motion_score=motion_score,
    )


def test_flow_residual_detects_local_motion_and_marks_uncertain_pixels_unknown():
    expected = np.zeros((12, 16, 2), dtype=np.float32)
    actual = expected.copy()
    actual[2:8, 4:10, 0] = 4.0
    valid = np.ones((12, 16), dtype=bool)
    valid[:, :2] = False
    result = classify_flow_residual(
        actual,
        expected,
        valid,
        config=MotionConfig(residual_threshold_px=2.0, minimum_dynamic_pixels=8),
    )
    assert np.all(result.dynamic_mask[2:8, 4:10])
    assert np.all(result.unknown_mask[:, :2])
    assert not np.any(result.static_mask & result.dynamic_mask)
    assert result.metrics["dynamic_pixels"] == 36


def test_pose_predicted_flow_keeps_static_background_static():
    depth = np.full((8, 10), 2.0, dtype=np.float32)
    intrinsics = np.array([[100.0, 0.0, 4.5], [0.0, 100.0, 3.5], [0.0, 0.0, 1.0]])
    previous = np.eye(4)
    current = np.eye(4)
    expected, valid = predict_background_flow(depth, previous, current, intrinsics)
    result = classify_flow_residual(
        expected.copy(),
        expected,
        valid,
        config=MotionConfig(minimum_dynamic_pixels=1),
    )
    assert np.all(result.static_mask)
    assert not np.any(result.dynamic_mask)
    assert not np.any(result.unknown_mask)


def test_dynamic_and_unknown_depth_never_reach_static_fusion():
    depth = np.full((6, 8), 2.0, dtype=np.float32)
    original = depth.copy()
    dynamic = np.zeros_like(depth, dtype=bool)
    unknown = np.zeros_like(depth, dtype=bool)
    dynamic[1:3, 2:5] = True
    unknown[:, 0] = True
    result = isolate_static_depth(depth, dynamic, unknown)
    assert np.all(result.depth_m[dynamic | unknown] == 0.0)
    assert np.all(result.confidence[dynamic | unknown] == 0.0)
    assert np.array_equal(depth, original)
    assert result.metrics["dynamic_contamination_rate"] == 0.0


def test_nonuniform_observation_time_changes_velocity_estimate():
    slow_layer = DynamicLayer()
    fast_layer = DynamicLayer()
    slow_layer.update(observation(0.0, [0.0, 0.0, 0.0], entity_id="slow"))
    slow = slow_layer.update(observation(2.0, [2.0, 0.0, 0.0], entity_id="slow"))
    fast_layer.update(observation(0.0, [0.0, 0.0, 0.0], entity_id="fast"))
    fast = fast_layer.update(observation(0.5, [2.0, 0.0, 0.0], entity_id="fast"))
    assert slow.velocity_mps[0] == pytest.approx(1.0, abs=0.08)
    assert fast.velocity_mps[0] == pytest.approx(4.0, abs=0.25)


def test_static_promotion_is_revoked_by_new_motion_then_object_expires():
    config = DynamicLayerConfig(
        stable_duration_s=1.0,
        stable_observations=3,
        stable_speed_threshold_mps=0.12,
        occluded_after_s=0.5,
        remove_after_s=2.0,
    )
    layer = DynamicLayer(config)
    obj = layer.update(observation(0.0, [0.0, 0.0, 0.0]))
    assert obj.state is ObjectState.TENTATIVE
    layer.update(observation(0.5, [0.0, 0.0, 0.0]))
    layer.update(observation(1.0, [0.0, 0.0, 0.0]))
    obj = layer.update(observation(1.5, [0.0, 0.0, 0.0]))
    assert obj.state is ObjectState.PROMOTED_STATIC

    obj = layer.update(observation(1.8, [0.4, 0.0, 0.0], motion_score=1.0))
    assert obj.state is ObjectState.DYNAMIC
    assert obj.stable_observations == 0
    layer.advance_time(ORIGIN_NS + int(2.5e9))
    assert obj.state is ObjectState.OCCLUDED
    covariance_after_occlusion = np.trace(obj.covariance)
    expired = layer.advance_time(ORIGIN_NS + int(4.0e9))
    assert expired == ["snack-cart"]
    assert "snack-cart" not in layer.active_objects
    assert layer.history["snack-cart"].state is ObjectState.EXPIRED
    assert np.trace(layer.history["snack-cart"].covariance) > covariance_after_occlusion


def test_track_id_change_after_short_occlusion_reassociates_entity():
    layer = DynamicLayer(
        DynamicLayerConfig(occluded_after_s=0.2, remove_after_s=3.0)
    )
    first = layer.update(observation(0.0, [1.0, 2.0, 0.0]))
    layer.advance_time(ORIGIN_NS + int(0.3e9))
    assert first.state is ObjectState.OCCLUDED
    reassociated = layer.update(
        observation(
            0.4,
            [1.02, 2.0, 0.0],
            track_id=99,
            entity_id=None,
        )
    )
    assert reassociated.entity_id == first.entity_id
    assert reassociated.track_ids == {1, 99}
