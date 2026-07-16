"""Acceptance tests for online pose correction, submaps, and path merging."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.mapping.paths import PathObservation, PathRepository  # noqa: E402
from daaam.mapping.submaps import SubmapManager  # noqa: E402
from daaam.realtime.contracts import PoseEstimate  # noqa: E402
from daaam.slam.backend import PoseBackendConfig, PoseInputValidator  # noqa: E402
from daaam.slam.incremental_pose_graph import (  # noqa: E402
    IncrementalPoseGraph,
    PoseConstraint,
    PoseGraphConfig,
)


ORIGIN_NS = 1_783_933_507_759_540_877


def transform(x=0.0, y=0.0, yaw_deg=0.0):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    value[:3, 3] = [x, y, 0.0]
    return value


def estimate(index: int, pose: np.ndarray, covariance_scale: float = 0.01):
    return PoseEstimate(
        ORIGIN_NS + index * 100_000_000,
        pose,
        np.eye(6) * covariance_scale,
        "stereo-vio",
    )


def constraint(source, target, relative, *, kind="odometry", variance=0.02, **kwargs):
    return PoseConstraint(
        ORIGIN_NS + source * 100_000_000,
        ORIGIN_NS + target * 100_000_000,
        relative,
        np.eye(6) * variance,
        kind=kind,
        **kwargs,
    )


def test_pose_backend_rejects_nonmonotonic_time_clock_jump_and_bad_covariance():
    validator = PoseInputValidator(PoseBackendConfig(maximum_clock_jump_s=1.0))
    first = estimate(0, np.eye(4))
    assert validator.validate(first, calibration_revision=3).accepted
    repeated = PoseEstimate(first.sensor_time_ns, np.eye(4), np.eye(6) * 0.01, "vio")
    assert validator.validate(repeated, calibration_revision=3).reason == "non_monotonic_time"
    jump = PoseEstimate(
        first.sensor_time_ns + 2_000_000_000,
        np.eye(4),
        np.eye(6) * 0.01,
        "vio",
    )
    assert validator.validate(jump, calibration_revision=3).reason == "clock_jump"
    uncertain = PoseEstimate(
        first.sensor_time_ns + 100_000_000,
        np.eye(4),
        np.diag([4.0, 4.0, 4.0, 0.01, 0.01, 0.01]),
        "vio",
    )
    assert validator.validate(uncertain, calibration_revision=3).reason == "position_covariance"
    assert (
        validator.validate(estimate(1, np.eye(4)), calibration_revision=4).reason
        == "calibration_revision_changed"
    )


def test_verified_loop_reduces_endpoint_drift_and_increments_revision():
    graph = IncrementalPoseGraph(
        PoseGraphConfig(window_size=8, maximum_iterations=100, robust_loss_scale=5.0)
    )
    initial_poses = [
        transform(0.0, 0.0),
        transform(1.0, 0.0),
        transform(1.0, 1.0),
        transform(0.0, 1.0),
        transform(0.35, 0.15),
    ]
    for index, pose in enumerate(initial_poses):
        graph.add_pose(estimate(index, pose))
    for index in range(len(initial_poses) - 1):
        relative = np.linalg.inv(initial_poses[index]) @ initial_poses[index + 1]
        graph.add_constraint(constraint(index, index + 1, relative, variance=0.08))
    graph.add_constraint(
        constraint(
            0,
            4,
            np.eye(4),
            kind="loop",
            variance=0.001,
            geometrically_verified=True,
            verification_score=0.95,
            gravity_residual_deg=0.5,
        )
    )
    error_before = np.linalg.norm(graph.pose(ORIGIN_NS + 400_000_000)[:3, 3])
    report = graph.optimize()
    error_after = np.linalg.norm(graph.pose(ORIGIN_NS + 400_000_000)[:3, 3])
    assert report.success
    assert report.loop_applied
    assert report.map_revision == 1
    assert report.cost_after < report.cost_before
    assert error_after < error_before * 0.25


def test_unverified_or_gravity_incompatible_loop_is_rejected():
    graph = IncrementalPoseGraph(PoseGraphConfig(maximum_gravity_residual_deg=8.0))
    graph.add_pose(estimate(0, np.eye(4)))
    graph.add_pose(estimate(1, transform(1.0)))
    bad = constraint(
        0,
        1,
        np.eye(4),
        kind="loop",
        geometrically_verified=True,
        verification_score=0.9,
        gravity_residual_deg=20.0,
    )
    assert not graph.add_constraint(bad)
    assert graph.map_revision == 0


def test_pose_graph_only_optimizes_fixed_recent_window():
    graph = IncrementalPoseGraph(PoseGraphConfig(window_size=5))
    poses = [transform(index * 0.1) for index in range(12)]
    for index, pose in enumerate(poses):
        graph.add_pose(estimate(index, pose))
    for index in range(11):
        graph.add_constraint(
            constraint(index, index + 1, np.linalg.inv(poses[index]) @ poses[index + 1])
        )
    report = graph.optimize()
    assert report.success
    assert report.optimized_nodes == 4


def test_submap_revision_reprojects_objects_without_reintegration():
    manager = SubmapManager(maximum_frames=2)
    first_id = manager.add_frame(ORIGIN_NS, transform(0.0))
    manager.add_frame(ORIGIN_NS + 100_000_000, transform(0.1))
    second_id = manager.add_frame(ORIGIN_NS + 200_000_000, transform(2.0))
    assert first_id != second_id
    world_before = manager.world_point(first_id, np.array([1.0, 0.0, 0.0]), revision=0)
    corrected = transform(0.5, 0.0)
    update = manager.apply_global_correction(
        {first_id: corrected},
        sensor_time_ns=ORIGIN_NS + 300_000_000,
        expected_revision=0,
        reason="verified_loop",
    )
    assert update.map_revision == 1
    assert manager.map_revision == 1
    world_after = manager.reproject_world_point(
        first_id, world_before, from_revision=0, to_revision=1
    )
    assert np.allclose(world_after, [1.5, 0.0, 0.0])
    with pytest.raises(ValueError, match="stale"):
        manager.apply_global_correction(
            {},
            sensor_time_ns=ORIGIN_NS + 400_000_000,
            expected_revision=0,
            reason="stale",
        )


def path_observation(session: str, points: np.ndarray, start_offset: int):
    times = ORIGIN_NS + start_offset + np.arange(len(points), dtype=np.int64) * 100_000_000
    return PathObservation(session, times, points, map_revision=0)


def test_forward_and_reverse_repeated_paths_merge_but_parallel_path_does_not():
    repository = PathRepository()
    x = np.linspace(0.0, 5.0, 40)
    base = np.c_[x, np.sin(x) * 0.1, np.zeros_like(x)]
    first, first_result = repository.add(path_observation("morning", base, 0))
    assert not first_result["merged"]
    reverse, reverse_result = repository.add(
        path_observation("afternoon", base[::-1] + [0.02, 0.01, 0.0], 10_000_000_000)
    )
    assert reverse.path_id == first.path_id
    assert reverse_result["merged"]
    assert reverse_result["reversed"]
    assert len(reverse.observations) == 2

    parallel = base + [0.0, 0.8, 0.0]
    distinct, distinct_result = repository.add(
        path_observation("evening", parallel, 20_000_000_000)
    )
    assert not distinct_result["merged"]
    assert distinct.path_id != first.path_id
    assert len(repository.paths) == 2


def test_path_repository_snapshot_restores_observation_history_and_merge_identity():
    x = np.linspace(0.0, 3.0, 24)
    base = np.c_[x, np.zeros_like(x), np.zeros_like(x)]
    repository = PathRepository()
    original, _ = repository.add(path_observation("first", base, 0))

    restored = PathRepository.from_snapshot(repository.snapshot())
    merged, result = restored.add(
        path_observation("return", base[::-1], 10_000_000_000)
    )
    assert result["merged"]
    assert result["reversed"]
    assert merged.path_id == original.path_id
    assert len(merged.observations) == 2
