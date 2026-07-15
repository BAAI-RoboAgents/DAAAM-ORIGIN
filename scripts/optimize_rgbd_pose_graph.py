#!/usr/bin/env python3
"""Optimize a full timestamp-preserving RGB-D pose graph.

The supplied robot trajectory contributes weak consecutive priors. Relative
RGB-D odometry contributes stronger local geometry, and only independently
verified nonlocal links from discover_rgbd_loop_closures.py are admitted as
loop edges. All selected content-event frames remain in the output dataset.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares, minimize_scalar
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


@dataclass
class GraphEdge:
    first: int
    second: int
    transform: np.ndarray
    translation_sigma_m: float
    rotation_sigma_deg: float
    kind: str
    uncertain: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize robot priors, RGB-D odometry, and verified loops."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--rgbd-odometry-dataset", required=True, type=Path)
    parser.add_argument("--temporal-report", required=True, type=Path)
    parser.add_argument("--loop-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--robot-translation-sigma-m", type=float, default=0.08)
    parser.add_argument("--robot-rotation-sigma-deg", type=float, default=4.0)
    parser.add_argument("--rgbd-translation-sigma-m", type=float, default=0.04)
    parser.add_argument("--rgbd-rotation-sigma-deg", type=float, default=2.0)
    parser.add_argument("--loop-translation-sigma-m", type=float, default=0.06)
    parser.add_argument("--loop-rotation-sigma-deg", type=float, default=2.0)
    parser.add_argument("--loop-cluster-radius-frames", type=int, default=30)
    parser.add_argument("--max-loop-edges", type=int, default=8)
    parser.add_argument(
        "--max-loop-gravity-residual-deg",
        type=float,
        default=8.0,
        help="Reject a verified loop incompatible with planar gravity by more than this.",
    )
    parser.add_argument(
        "--loop-uncertain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow Open3D to prune verified loops. Strict verified loops default to certain.",
    )
    parser.add_argument(
        "--preserve-source-gravity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep source camera height, roll, and pitch while applying optimized "
            "XY and world-yaw corrections."
        ),
    )
    parser.add_argument(
        "--optimizer-mode",
        choices=("gravity-se3", "planar-gravity", "se3-projected"),
        default="gravity-se3",
        help=(
            "gravity-se3 optimizes full poses with height/tilt priors; "
            "planar-gravity preserves height/roll/pitch exactly; se3-projected "
            "retains the legacy two-step method."
        ),
    )
    parser.add_argument(
        "--planar-initialization",
        choices=("rgbd-odometry", "source"),
        default="rgbd-odometry",
        help="Initial XY/yaw branch used by the planar optimizer.",
    )
    parser.add_argument("--gravity-z-sigma-m", type=float, default=0.04)
    parser.add_argument(
        "--gravity-roll-pitch-sigma-deg", type=float, default=2.0
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_poses(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64).reshape(-1, 4, 4)
    if not np.isfinite(poses).all():
        raise ValueError(f"Non-finite poses in {path}")
    if not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"Expected homogeneous poses in {path}")
    return poses


def relative_pose(poses: np.ndarray, first: int, second: int) -> np.ndarray:
    """Return the transform mapping points from first camera to second camera."""
    return np.linalg.inv(poses[second]) @ poses[first]


def transform_error(measured: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    translation_m = float(np.linalg.norm(measured[:3, 3] - expected[:3, 3]))
    rotation_deg = float(
        np.rad2deg(
            Rotation.from_matrix(
                measured[:3, :3] @ expected[:3, :3].T
            ).magnitude()
        )
    )
    return translation_m, rotation_deg


def gravity_twist_error(
    rotation_error: np.ndarray, gravity_axis: np.ndarray
) -> np.ndarray:
    """Return the signed twist angle about gravity for one or more rotations."""
    quaternions = Rotation.from_matrix(rotation_error).as_quat()
    scalar = quaternions[..., 3]
    sign = np.where(scalar < 0.0, -1.0, 1.0)
    quaternions = quaternions * sign[..., None]
    axes = gravity_axis / np.linalg.norm(gravity_axis, axis=-1, keepdims=True)
    projected = np.sum(quaternions[..., :3] * axes, axis=-1)
    return 2.0 * np.arctan2(projected, quaternions[..., 3])


def edge_error_summary(poses: np.ndarray, edges: list[GraphEdge]) -> dict:
    summary = {}
    for kind in sorted({edge.kind for edge in edges}):
        selected = [edge for edge in edges if edge.kind == kind]
        translation = []
        rotation = []
        gravity_yaw = []
        for edge in selected:
            measured = relative_pose(poses, edge.first, edge.second)
            translation_error, rotation_error = transform_error(
                measured, edge.transform
            )
            translation.append(translation_error)
            rotation.append(rotation_error)
            rotation_delta = measured[:3, :3] @ edge.transform[:3, :3].T
            gravity_axis = poses[edge.second, :3, :3].T @ np.asarray(
                [0.0, 0.0, 1.0]
            )
            gravity_yaw.append(
                abs(float(gravity_twist_error(rotation_delta, gravity_axis)))
            )
        summary[kind] = {
            "count": len(selected),
            "translation_error_median_m": float(np.median(translation)),
            "translation_error_p95_m": float(np.percentile(translation, 95)),
            "rotation_error_median_deg": float(np.median(rotation)),
            "rotation_error_p95_deg": float(np.percentile(rotation, 95)),
            "gravity_yaw_error_median_deg": float(
                np.rad2deg(np.median(gravity_yaw))
            ),
            "gravity_yaw_error_p95_deg": float(
                np.rad2deg(np.percentile(gravity_yaw, 95))
            ),
        }
    return summary


def loop_quality(candidate: dict) -> tuple[float, float, float, float]:
    dense = candidate.get("dense_verification", {})
    selected = dense.get("selected_hypothesis", {})
    pixel = dense.get("pixel_verification", {})
    return (
        min(
            float(selected.get("forward_fitness_5cm", 0.0)),
            float(selected.get("reverse_fitness_5cm", 0.0)),
        ),
        float(pixel.get("depth_agreement_rate", 0.0)),
        float(pixel.get("color_agreement_rate_on_depth_agreement", 0.0)),
        float(candidate.get("similarity", 0.0)),
    )


def select_loop_candidates(
    report: dict,
    poses: np.ndarray,
    frame_count: int,
    cluster_radius: int,
    max_edges: int,
    max_gravity_residual_deg: float,
) -> list[dict]:
    verified = [
        candidate
        for candidate in report.get("verified_links", [])
        if candidate.get("quality_ok") and candidate.get("constraint")
    ]
    gravity_compatible = []
    for candidate in verified:
        constraint = candidate["constraint"]
        first = int(constraint["first"])
        second = int(constraint["second"])
        if not 0 <= first < second < frame_count:
            raise ValueError(f"Invalid verified loop endpoints: {first}, {second}")
        measured = np.asarray(
            constraint["relative_camera_rotation"], dtype=np.float64
        )

        def rotation_residual(yaw: float) -> float:
            cosine = np.cos(yaw)
            sine = np.sin(yaw)
            world_yaw = np.asarray(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
            )
            predicted = (
                poses[second, :3, :3].T
                @ world_yaw
                @ poses[first, :3, :3]
            )
            return float(
                Rotation.from_matrix(predicted @ measured.T).magnitude()
            )

        projection = minimize_scalar(
            rotation_residual,
            bounds=(-np.pi, np.pi),
            method="bounded",
        )
        gravity_residual_deg = float(np.rad2deg(projection.fun))
        if gravity_residual_deg > max_gravity_residual_deg:
            continue
        accepted = copy.deepcopy(candidate)
        accepted["gravity_compatibility"] = {
            "minimum_full_rotation_residual_deg": gravity_residual_deg,
            "projected_world_yaw_deg": float(np.rad2deg(projection.x)),
        }
        gravity_compatible.append(accepted)
    verified = gravity_compatible
    verified.sort(
        key=lambda candidate: (
            *loop_quality(candidate),
            -candidate["gravity_compatibility"][
                "minimum_full_rotation_residual_deg"
            ],
        ),
        reverse=True,
    )
    selected = []
    for candidate in verified:
        constraint = candidate["constraint"]
        first = int(constraint["first"])
        second = int(constraint["second"])
        duplicate_cluster = any(
            abs(first - int(item["constraint"]["first"])) <= cluster_radius
            and abs(second - int(item["constraint"]["second"])) <= cluster_radius
            for item in selected
        )
        if duplicate_cluster:
            continue
        selected.append(candidate)
        if len(selected) >= max_edges:
            break
    if not selected:
        raise RuntimeError(
            "No independently verified, gravity-compatible loop remains after "
            "filtering and clustering"
        )
    return selected


def load_temporal_agreement(path: Path, frame_count: int) -> dict[int, float]:
    report = json.loads(path.read_text())
    if not report.get("absolute_time_contract", {}).get("validated"):
        raise ValueError("Temporal report did not validate the absolute time contract")
    agreement = {}
    for pair in report.get("pairs", []):
        first = int(pair["reference_frame"])
        second = int(pair["neighbor_frame"])
        if second == first + 1 and pair.get("agreement_rate") is not None:
            agreement[first] = float(pair["agreement_rate"])
    if len(agreement) != frame_count - 1:
        raise ValueError(
            f"Expected {frame_count - 1} adjacent temporal scores, got {len(agreement)}"
        )
    return agreement


def make_information(translation_sigma_m: float, rotation_sigma_deg: float) -> np.ndarray:
    rotation_sigma_rad = np.deg2rad(rotation_sigma_deg)
    return np.diag(
        [1.0 / rotation_sigma_rad**2] * 3
        + [1.0 / translation_sigma_m**2] * 3
    )


def optimize(
    initial: np.ndarray, edges: list[GraphEdge], iterations: int
) -> np.ndarray:
    graph = o3d.pipelines.registration.PoseGraph()
    for pose in initial:
        graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose.copy()))
    for edge in edges:
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                edge.first,
                edge.second,
                edge.transform,
                make_information(
                    edge.translation_sigma_m, edge.rotation_sigma_deg
                ),
                edge.uncertain,
            )
        )
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    criteria.max_iteration = iterations
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.10,
        edge_prune_threshold=0.25,
        preference_loop_closure=2.0,
        reference_node=0,
    )
    previous_level = o3d.utility.get_verbosity_level()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    try:
        o3d.pipelines.registration.global_optimization(
            graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            criteria,
            option,
        )
    finally:
        o3d.utility.set_verbosity_level(previous_level)
    optimized = np.asarray([node.pose for node in graph.nodes])
    if not np.isfinite(optimized).all():
        raise RuntimeError("Pose graph optimization produced non-finite poses")
    return optimized


def preserve_source_gravity(
    optimized: np.ndarray, source: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Project free SE(3) corrections onto XY plus rotation about world Z."""
    projected = optimized.copy()
    yaw_corrections = []
    for index in range(len(projected)):
        correction = optimized[index, :3, :3] @ source[index, :3, :3].T
        yaw = float(
            np.arctan2(
                correction[1, 0] - correction[0, 1],
                correction[0, 0] + correction[1, 1],
            )
        )
        yaw_corrections.append(yaw)
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        world_yaw = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        projected[index, :3, :3] = world_yaw @ source[index, :3, :3]
        projected[index, 2, 3] = source[index, 2, 3]
    yaw_degrees = np.rad2deg(yaw_corrections)
    return projected, {
        "yaw_correction_deg_percentiles": np.percentile(
            yaw_degrees, [0, 5, 50, 95, 100]
        ).tolist(),
        "camera_height_preserved": True,
        "source_roll_pitch_preserved": True,
    }


def optimize_planar_gravity(
    source: np.ndarray,
    edges: list[GraphEdge],
    max_nfev: int,
    initial_poses: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Jointly optimize XY and world-yaw corrections with fixed gravity."""
    frame_count = len(source)
    initial = np.zeros((frame_count - 1, 3), dtype=np.float64)
    initial_poses = source if initial_poses is None else initial_poses
    if initial_poses.shape != source.shape:
        raise ValueError("Planar initializer must match the source pose array")
    initial[:, :2] = initial_poses[1:, :2, 3]
    rotation_correction = (
        initial_poses[:, :3, :3]
        @ np.transpose(source[:, :3, :3], (0, 2, 1))
    )
    initial_yaw = np.arctan2(
        rotation_correction[:, 1, 0] - rotation_correction[:, 0, 1],
        rotation_correction[:, 0, 0] + rotation_correction[:, 1, 1],
    )
    initial[:, 2] = initial_yaw[1:]
    first_indices = np.asarray([edge.first for edge in edges], dtype=np.int64)
    second_indices = np.asarray([edge.second for edge in edges], dtype=np.int64)
    measured_rotation = np.asarray(
        [edge.transform[:3, :3] for edge in edges], dtype=np.float64
    )
    measured_translation = np.asarray(
        [edge.transform[:3, 3] for edge in edges], dtype=np.float64
    )
    translation_sigma = np.asarray(
        [edge.translation_sigma_m for edge in edges], dtype=np.float64
    )
    rotation_sigma = np.deg2rad(
        [edge.rotation_sigma_deg for edge in edges]
    )

    def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = values.reshape(frame_count - 1, 3)
        positions = source[:, :3, 3].copy()
        positions[1:, :2] = values[:, :2]
        yaw = np.zeros(frame_count, dtype=np.float64)
        yaw[1:] = values[:, 2]
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        world_yaw = np.zeros((frame_count, 3, 3), dtype=np.float64)
        world_yaw[:, 0, 0] = cosine
        world_yaw[:, 0, 1] = -sine
        world_yaw[:, 1, 0] = sine
        world_yaw[:, 1, 1] = cosine
        world_yaw[:, 2, 2] = 1.0
        rotations = world_yaw @ source[:, :3, :3]
        return positions, rotations

    def residual(values: np.ndarray) -> np.ndarray:
        positions, rotations = unpack(values)
        first_rotation = rotations[first_indices]
        second_rotation = rotations[second_indices]
        predicted_rotation = (
            np.transpose(second_rotation, (0, 2, 1)) @ first_rotation
        )
        world_delta = positions[first_indices] - positions[second_indices]
        predicted_translation = np.einsum(
            "nij,nj->ni", np.transpose(second_rotation, (0, 2, 1)), world_delta
        )
        translation_error = (
            predicted_translation - measured_translation
        ) / translation_sigma[:, None]
        rotation_delta = (
            predicted_rotation @ np.transpose(measured_rotation, (0, 2, 1))
        )
        gravity_axis = np.einsum(
            "nij,j->ni",
            np.transpose(second_rotation, (0, 2, 1)),
            np.asarray([0.0, 0.0, 1.0]),
        )
        yaw_error = gravity_twist_error(rotation_delta, gravity_axis)
        return np.hstack(
            (yaw_error[:, None] / rotation_sigma[:, None], translation_error)
        ).reshape(-1)

    sparsity = lil_matrix((len(edges) * 4, (frame_count - 1) * 3), dtype=np.int8)
    for edge_index, edge in enumerate(edges):
        rows = slice(edge_index * 4, (edge_index + 1) * 4)
        for frame in (edge.first, edge.second):
            if frame > 0:
                columns = slice((frame - 1) * 3, frame * 3)
                sparsity[rows, columns] = 1
    initial_residual = residual(initial.reshape(-1))
    result = least_squares(
        residual,
        initial.reshape(-1),
        jac_sparsity=sparsity.tocsr(),
        method="trf",
        loss="linear",
        max_nfev=max_nfev,
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
    )
    positions, rotations = unpack(result.x)
    optimized = source.copy()
    optimized[:, :3, :3] = rotations
    optimized[:, :3, 3] = positions
    return optimized, {
        "mode": "planar-gravity",
        "success": bool(result.success),
        "message": result.message,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "initial_residual_rms": float(np.sqrt(np.mean(initial_residual**2))),
        "final_residual_rms": float(np.sqrt(np.mean(result.fun**2))),
        "camera_height_preserved": True,
        "source_roll_pitch_preserved": True,
        "rotation_residual": "gravity_twist_only",
        "initialization": (
            "source" if initial_poses is source else "rgbd-odometry"
        ),
        "yaw_correction_deg_percentiles": np.percentile(
            np.rad2deg(result.x.reshape(frame_count - 1, 3)[:, 2]),
            [0, 5, 50, 95, 100],
        ).tolist(),
    }


def optimize_gravity_se3(
    source: np.ndarray,
    edges: list[GraphEdge],
    max_nfev: int,
    z_sigma_m: float,
    roll_pitch_sigma_deg: float,
    initial_poses: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Optimize full poses while softly retaining gravity and camera height."""
    if z_sigma_m <= 0.0 or roll_pitch_sigma_deg <= 0.0:
        raise ValueError("Gravity prior sigmas must be positive")
    frame_count = len(source)
    initial_poses = source if initial_poses is None else initial_poses
    if initial_poses.shape != source.shape:
        raise ValueError("Gravity-SE3 initializer must match the source poses")

    initial = np.zeros((frame_count - 1, 6), dtype=np.float64)
    initial[:, :3] = initial_poses[1:, :3, 3]
    initial_correction = (
        initial_poses[:, :3, :3]
        @ np.transpose(source[:, :3, :3], (0, 2, 1))
    )
    initial[:, 3:] = Rotation.from_matrix(initial_correction[1:]).as_euler(
        "xyz"
    )
    first_indices = np.asarray([edge.first for edge in edges], dtype=np.int64)
    second_indices = np.asarray([edge.second for edge in edges], dtype=np.int64)
    measured_rotation = np.asarray(
        [edge.transform[:3, :3] for edge in edges], dtype=np.float64
    )
    measured_translation = np.asarray(
        [edge.transform[:3, 3] for edge in edges], dtype=np.float64
    )
    translation_sigma = np.asarray(
        [edge.translation_sigma_m for edge in edges], dtype=np.float64
    )
    rotation_sigma = np.deg2rad(
        [edge.rotation_sigma_deg for edge in edges]
    )
    roll_pitch_sigma = np.deg2rad(roll_pitch_sigma_deg)

    def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = values.reshape(frame_count - 1, 6)
        positions = source[:, :3, 3].copy()
        positions[1:] = values[:, :3]
        corrections = np.zeros((frame_count, 3), dtype=np.float64)
        corrections[1:] = values[:, 3:]
        correction_rotation = Rotation.from_euler(
            "xyz", corrections
        ).as_matrix()
        rotations = correction_rotation @ source[:, :3, :3]
        return positions, rotations, corrections

    def residual(values: np.ndarray) -> np.ndarray:
        positions, rotations, corrections = unpack(values)
        first_rotation = rotations[first_indices]
        second_rotation = rotations[second_indices]
        predicted_rotation = (
            np.transpose(second_rotation, (0, 2, 1)) @ first_rotation
        )
        world_delta = positions[first_indices] - positions[second_indices]
        predicted_translation = np.einsum(
            "nij,nj->ni", np.transpose(second_rotation, (0, 2, 1)), world_delta
        )
        rotation_error = Rotation.from_matrix(
            predicted_rotation @ np.transpose(measured_rotation, (0, 2, 1))
        ).as_rotvec() / rotation_sigma[:, None]
        translation_error = (
            predicted_translation - measured_translation
        ) / translation_sigma[:, None]
        edge_residual = np.hstack((rotation_error, translation_error)).reshape(-1)
        gravity_residual = np.column_stack(
            (
                (positions[1:, 2] - source[1:, 2, 3]) / z_sigma_m,
                corrections[1:, 0] / roll_pitch_sigma,
                corrections[1:, 1] / roll_pitch_sigma,
            )
        ).reshape(-1)
        return np.concatenate((edge_residual, gravity_residual))

    edge_rows = len(edges) * 6
    prior_rows = (frame_count - 1) * 3
    sparsity = lil_matrix(
        (edge_rows + prior_rows, (frame_count - 1) * 6), dtype=np.int8
    )
    for edge_index, edge in enumerate(edges):
        rows = slice(edge_index * 6, (edge_index + 1) * 6)
        for frame in (edge.first, edge.second):
            if frame > 0:
                columns = slice((frame - 1) * 6, frame * 6)
                sparsity[rows, columns] = 1
    for frame in range(1, frame_count):
        row = edge_rows + (frame - 1) * 3
        columns = slice((frame - 1) * 6, frame * 6)
        sparsity[row : row + 3, columns] = 1

    initial_residual = residual(initial.reshape(-1))
    result = least_squares(
        residual,
        initial.reshape(-1),
        jac_sparsity=sparsity.tocsr(),
        method="trf",
        loss="linear",
        max_nfev=max_nfev,
        xtol=1.0e-7,
        ftol=1.0e-7,
        gtol=1.0e-7,
    )
    positions, rotations, corrections = unpack(result.x)
    optimized = source.copy()
    optimized[:, :3, :3] = rotations
    optimized[:, :3, 3] = positions
    correction_degrees = np.rad2deg(corrections)
    height_change = positions[:, 2] - source[:, 2, 3]
    return optimized, {
        "mode": "gravity-se3",
        "success": bool(result.success),
        "message": result.message,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "initial_residual_rms": float(np.sqrt(np.mean(initial_residual**2))),
        "final_residual_rms": float(np.sqrt(np.mean(result.fun**2))),
        "initialization": (
            "source" if initial_poses is source else "rgbd-odometry"
        ),
        "gravity_z_sigma_m": z_sigma_m,
        "gravity_roll_pitch_sigma_deg": roll_pitch_sigma_deg,
        "height_change_m_percentiles": np.percentile(
            height_change, [0, 5, 50, 95, 100]
        ).tolist(),
        "roll_correction_deg_percentiles": np.percentile(
            correction_degrees[:, 0], [0, 5, 50, 95, 100]
        ).tolist(),
        "pitch_correction_deg_percentiles": np.percentile(
            correction_degrees[:, 1], [0, 5, 50, 95, 100]
        ).tolist(),
        "yaw_correction_deg_percentiles": np.percentile(
            correction_degrees[:, 2], [0, 5, 50, 95, 100]
        ).tolist(),
    }


def prepare_output(source: Path, output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("rgb", "depth", "stereo_right"):
        source_directory = source / directory
        if source_directory.exists():
            (output / directory).symlink_to(
                source_directory.resolve(), target_is_directory=True
            )
    (output / "pose").mkdir()
    shutil.copy2(
        source / "pose" / "pose_timestamps_ns.txt",
        output / "pose" / "pose_timestamps_ns.txt",
    )
    for name in (
        "camera_info.json",
        "source_manifest.json",
        "keyframe_selection_report.json",
        "floor_calibration_application.json",
        "foundation_stereo_nominal_run.json",
        "trajectory_refinement.json",
    ):
        source_path = source / name
        if source_path.exists():
            (output / name).symlink_to(source_path.resolve())


def write_output(
    source: Path,
    output: Path,
    poses: np.ndarray,
    report: dict,
    overwrite: bool,
) -> None:
    prepare_output(source, output, overwrite)
    (output / "pose" / "poses.txt").write_text(
        "".join(
            " ".join(f"{value:.12g}" for value in pose.reshape(-1)) + "\n"
            for pose in poses
        )
    )
    tick_index = copy.deepcopy(json.loads((source / "tick_index.json").read_text()))
    tick_index["trajectory_refinement"] = {
        "method": "robot_prior_rgbd_odometry_verified_loop_pose_graph",
        "report": "global_pose_graph_report.json",
        "source_pose_use": "weak_consecutive_prior",
    }
    (output / "tick_index.json").write_text(json.dumps(tick_index, indent=2) + "\n")
    (output / "global_pose_graph_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    if args.loop_cluster_radius_frames < 0 or args.max_loop_edges < 1:
        raise ValueError("Loop clustering parameters must be non-negative")
    source = args.dataset.resolve()
    rgbd_dataset = args.rgbd_odometry_dataset.resolve()
    output = args.output.resolve()
    source_poses = load_poses(source / "pose" / "poses.txt")
    rgbd_poses = load_poses(rgbd_dataset / "pose" / "poses.txt")
    if source_poses.shape != rgbd_poses.shape:
        raise ValueError("Robot and RGB-D trajectories must contain the same frames")
    frame_count = len(source_poses)
    temporal_agreement = load_temporal_agreement(
        args.temporal_report.resolve(), frame_count
    )
    loop_report = json.loads(args.loop_report.resolve().read_text())
    selected_loops = select_loop_candidates(
        loop_report,
        source_poses,
        frame_count,
        args.loop_cluster_radius_frames,
        args.max_loop_edges,
        args.max_loop_gravity_residual_deg,
    )

    edges = []
    for first in range(frame_count - 1):
        second = first + 1
        edges.append(
            GraphEdge(
                first,
                second,
                relative_pose(source_poses, first, second),
                args.robot_translation_sigma_m,
                args.robot_rotation_sigma_deg,
                "robot_odom_prior",
            )
        )
        agreement = temporal_agreement[first]
        uncertainty_scale = 1.0 + 3.0 * max(0.0, 0.85 - agreement)
        edges.append(
            GraphEdge(
                first,
                second,
                relative_pose(rgbd_poses, first, second),
                args.rgbd_translation_sigma_m * uncertainty_scale,
                args.rgbd_rotation_sigma_deg * uncertainty_scale,
                "rgbd_odometry",
            )
        )

    loop_details = []
    for candidate in selected_loops:
        constraint = candidate["constraint"]
        first = int(constraint["first"])
        second = int(constraint["second"])
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(
            constraint["relative_camera_rotation"], dtype=np.float64
        )
        transform[:3, 3] = np.asarray(
            constraint["relative_camera_translation_m"], dtype=np.float64
        )
        translation_sigma = max(
            args.loop_translation_sigma_m, float(constraint.get("sigma_m", 0.0))
        )
        edges.append(
            GraphEdge(
                first,
                second,
                transform,
                translation_sigma,
                args.loop_rotation_sigma_deg,
                "verified_loop",
                args.loop_uncertain,
            )
        )
        loop_details.append(
            {
                "first": first,
                "second": second,
                "method": candidate.get("method"),
                "similarity": candidate.get("similarity"),
                "quality": loop_quality(candidate),
                "translation_sigma_m": translation_sigma,
                "rotation_sigma_deg": args.loop_rotation_sigma_deg,
                "gravity_compatibility": candidate.get("gravity_compatibility"),
                "transform": transform.tolist(),
            }
        )

    before = edge_error_summary(source_poses, edges)
    raw_after = None
    gravity_projection = None
    visual_initial = (
        rgbd_poses
        if args.planar_initialization == "rgbd-odometry"
        else source_poses
    )
    if args.optimizer_mode == "planar-gravity":
        optimized, optimization_report = optimize_planar_gravity(
            source_poses, edges, args.iterations, visual_initial
        )
    elif args.optimizer_mode == "gravity-se3":
        optimized, optimization_report = optimize_gravity_se3(
            source_poses,
            edges,
            args.iterations,
            args.gravity_z_sigma_m,
            args.gravity_roll_pitch_sigma_deg,
            visual_initial,
        )
    else:
        optimized_raw = optimize(source_poses, edges, args.iterations)
        raw_after = edge_error_summary(optimized_raw, edges)
        if args.preserve_source_gravity:
            optimized, gravity_projection = preserve_source_gravity(
                optimized_raw, source_poses
            )
        else:
            optimized = optimized_raw
        optimization_report = {
            "mode": "se3-projected",
            "camera_height_preserved": bool(args.preserve_source_gravity),
            "source_roll_pitch_preserved": bool(args.preserve_source_gravity),
        }
    after = edge_error_summary(optimized, edges)
    source_path_length = float(
        np.linalg.norm(np.diff(source_poses[:, :3, 3], axis=0), axis=1).sum()
    )
    optimized_path_length = float(
        np.linalg.norm(np.diff(optimized[:, :3, 3], axis=0), axis=1).sum()
    )
    position_change = np.linalg.norm(
        optimized[:, :3, 3] - source_poses[:, :3, 3], axis=1
    )
    report = {
        "source_dataset": str(source),
        "rgbd_odometry_dataset": str(rgbd_dataset),
        "temporal_report": str(args.temporal_report.resolve()),
        "loop_report": str(args.loop_report.resolve()),
        "output_dataset": str(output),
        "frame_count": frame_count,
        "edge_counts": {
            kind: sum(edge.kind == kind for edge in edges)
            for kind in sorted({edge.kind for edge in edges})
        },
        "parameters": {
            "robot_translation_sigma_m": args.robot_translation_sigma_m,
            "robot_rotation_sigma_deg": args.robot_rotation_sigma_deg,
            "rgbd_translation_sigma_m": args.rgbd_translation_sigma_m,
            "rgbd_rotation_sigma_deg": args.rgbd_rotation_sigma_deg,
            "loop_translation_sigma_m": args.loop_translation_sigma_m,
            "loop_rotation_sigma_deg": args.loop_rotation_sigma_deg,
            "loop_uncertain": args.loop_uncertain,
            "max_loop_gravity_residual_deg": args.max_loop_gravity_residual_deg,
            "preserve_source_gravity": args.preserve_source_gravity,
            "optimizer_mode": args.optimizer_mode,
            "planar_initialization": args.planar_initialization,
            "gravity_z_sigma_m": args.gravity_z_sigma_m,
            "gravity_roll_pitch_sigma_deg": (
                args.gravity_roll_pitch_sigma_deg
            ),
            "iterations": args.iterations,
        },
        "optimization": optimization_report,
        "selected_verified_loops": loop_details,
        "edge_residuals_before": before,
        "edge_residuals_after_raw_se3": raw_after,
        "edge_residuals_after": after,
        "gravity_projection": gravity_projection,
        "source_path_length_m": source_path_length,
        "rgbd_chain_path_length_m": float(
            np.linalg.norm(np.diff(rgbd_poses[:, :3, 3], axis=0), axis=1).sum()
        ),
        "optimized_path_length_m": optimized_path_length,
        "position_change_from_source_m": {
            "median": float(np.median(position_change)),
            "p95": float(np.percentile(position_change, 95)),
            "max": float(position_change.max()),
        },
    }
    write_output(source, output, optimized, report, args.overwrite)
    trajectory_report = rgbd_dataset / "trajectory_refinement.json"
    output_trajectory_report = output / "trajectory_refinement.json"
    if (
        trajectory_report.is_file()
        and not output_trajectory_report.exists()
        and not output_trajectory_report.is_symlink()
    ):
        output_trajectory_report.symlink_to(trajectory_report.resolve())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
