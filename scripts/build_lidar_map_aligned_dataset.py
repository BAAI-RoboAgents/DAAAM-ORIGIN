#!/usr/bin/env python3
"""Re-express a prepared G1 RGB-D dataset in a LiDAR map frame.

The LiDAR mapper provides sparse ``map_T_base_link`` poses.  This script
interpolates the slowly varying ``map_T_odom`` correction between those
anchors, combines it with dense wheel odometry and the time-varying head
camera extrinsic, and writes an otherwise zero-copy RGB-D dataset whose poses
are ``lidar_map_T_image_camera``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--raw-dataset", required=True, type=Path)
    parser.add_argument("--lidar-map", required=True, type=Path)
    parser.add_argument("--pinhole-report", required=True, type=Path)
    parser.add_argument("--floor-calibration-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--validate-lidar-cloud",
        action="store_true",
        help="Sample raw scans against global_cloud_cleaned.pcd (requires open3d).",
    )
    parser.add_argument("--validation-frames", type=int, default=24)
    parser.add_argument("--validation-points-per-frame", type=int, default=1200)
    parser.add_argument("--validation-threshold-m", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def transform(position: Iterable[float], quaternion_xyzw: Iterable[float]) -> np.ndarray:
    position_array = np.asarray(position, dtype=np.float64)
    quaternion_array = np.asarray(quaternion_xyzw, dtype=np.float64)
    if position_array.shape != (3,) or quaternion_array.shape != (4,):
        raise ValueError("A pose requires three translation and four quaternion values")
    if not np.isfinite(position_array).all() or not np.isfinite(quaternion_array).all():
        raise ValueError("Pose contains non-finite values")
    quaternion_norm = np.linalg.norm(quaternion_array)
    if quaternion_norm < 1.0e-12:
        raise ValueError("Pose quaternion has zero norm")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        quaternion_array / quaternion_norm
    ).as_matrix()
    matrix[:3, 3] = position_array
    return matrix


def validate_transform_series(name: str, matrices: np.ndarray) -> None:
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape (N, 4, 4)")
    if not np.isfinite(matrices).all():
        raise ValueError(f"{name} contains non-finite values")
    bottom_error = np.max(
        np.abs(matrices[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0]))
    )
    orthogonality_error = np.max(
        np.abs(
            np.swapaxes(matrices[:, :3, :3], 1, 2) @ matrices[:, :3, :3]
            - np.eye(3)
        )
    )
    determinants = np.linalg.det(matrices[:, :3, :3])
    if bottom_error > 1.0e-8 or orthogonality_error > 1.0e-6:
        raise ValueError(
            f"{name} is not rigid: bottom={bottom_error}, rotation={orthogonality_error}"
        )
    if np.max(np.abs(determinants - 1.0)) > 1.0e-6:
        raise ValueError(f"{name} contains improper rotations")


def unique_pose_samples(
    timestamps_ns: Iterable[int], matrices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    matrices = np.asarray(matrices, dtype=np.float64)
    if len(timestamps) != len(matrices):
        raise ValueError("Pose timestamps and matrices have different counts")
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    matrices = matrices[order]
    keep = np.r_[np.diff(timestamps) != 0, True]
    timestamps = timestamps[keep]
    matrices = matrices[keep]
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("At least two strictly increasing pose samples are required")
    validate_transform_series("pose samples", matrices)
    return timestamps, matrices


def interpolate_transforms(
    timestamps_ns: Iterable[int],
    matrices: np.ndarray,
    targets_ns: Iterable[int],
) -> tuple[np.ndarray, int]:
    """Interpolate SE(3), clamping targets beyond the supplied time range."""

    timestamps, matrices = unique_pose_samples(timestamps_ns, matrices)
    targets = np.asarray(targets_ns, dtype=np.int64)
    if targets.ndim != 1:
        raise ValueError("Target timestamps must be one-dimensional")
    clipped = np.clip(targets, timestamps[0], timestamps[-1])
    origin_ns = int(timestamps[0])
    times_s = (timestamps - origin_ns).astype(np.float64) / 1.0e9
    targets_s = (clipped - origin_ns).astype(np.float64) / 1.0e9
    translations = np.column_stack(
        [
            np.interp(targets_s, times_s, matrices[:, axis, 3])
            for axis in range(3)
        ]
    )
    rotations = Slerp(
        times_s, Rotation.from_matrix(matrices[:, :3, :3])
    )(targets_s).as_matrix()
    output = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(targets), axis=0)
    output[:, :3, :3] = rotations
    output[:, :3, 3] = translations
    return output, int(np.count_nonzero(clipped != targets))


def odom_pose_sample(record: dict[str, Any]) -> tuple[int, np.ndarray]:
    try:
        odom = record["odom"]
        if "timestamp_ns" in odom:
            timestamp_ns = int(odom["timestamp_ns"])
            position = odom["position"]
            orientation = odom["orientation"]
        else:
            timestamp_ns = int(
                odom.get("header", {}).get("timestamp_ns", record["sensor_time_ns"])
            )
            pose = odom["pose"]["pose"]
            position = pose["position"]
            orientation = pose["orientation"]
        if isinstance(position, dict):
            position = [position[axis] for axis in ("x", "y", "z")]
        if isinstance(orientation, dict):
            orientation = [orientation[axis] for axis in ("x", "y", "z", "w")]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed odometry record at tick {record.get('tick')}") from error
    return timestamp_ns, transform(position, orientation)


def aux_pose_sample(
    record: dict[str, Any], sensor: str, quaternion_order: str
) -> tuple[int, np.ndarray]:
    try:
        sample = record["poses"][sensor]
        if sample["target_frame"] != "base_link":
            raise ValueError(f"Unexpected {sensor} target frame: {sample['target_frame']}")
        timestamp_ns = int(sample["timestamp_ns"])
        position = sample["position"]
        quaternion = list(sample["orientation_xyzw"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Malformed {sensor} auxiliary pose at tick {record.get('tick')}"
        ) from error
    if quaternion_order == "wxyz":
        quaternion = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    elif quaternion_order != "xyzw":
        raise ValueError(f"Unsupported quaternion order: {quaternion_order}")
    return timestamp_ns, transform(position, quaternion)


def load_map_anchors(lidar_map: Path) -> tuple[np.ndarray, np.ndarray]:
    poses_path = lidar_map / "poses.txt"
    times_path = lidar_map / "times.txt"
    rows = np.loadtxt(poses_path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] != 12:
        raise ValueError(f"Expected 12 values per LiDAR pose in {poses_path}")
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(rows), axis=0)
    matrices[:, :3, :] = rows.reshape(-1, 3, 4)
    # The mapper serializes only six decimals, which leaves rotations a few
    # parts per million away from SO(3).  Project them back to proper rotations
    # before inversion and interpolation rather than propagating tiny scale.
    for matrix in matrices:
        left, _, right = np.linalg.svd(matrix[:3, :3])
        projected = left @ right
        if np.linalg.det(projected) < 0.0:
            left[:, -1] *= -1.0
            projected = left @ right
        matrix[:3, :3] = projected
    time_lines = [line.strip() for line in times_path.read_text().splitlines() if line.strip()]
    timestamps_ns = np.asarray(
        [int(Decimal(line) * Decimal(1_000_000_000)) for line in time_lines],
        dtype=np.int64,
    )
    if len(timestamps_ns) != len(matrices):
        raise ValueError("LiDAR map pose and timestamp counts differ")
    if np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError("LiDAR map timestamps are not strictly increasing")
    validate_transform_series("LiDAR map anchors", matrices)
    return timestamps_ns, matrices


def post_camera_transform(pinhole_report: Path, floor_report: Path) -> tuple[np.ndarray, dict]:
    pinhole = json.loads(pinhole_report.read_text())
    floor = json.loads(floor_report.read_text())
    if pinhole.get("camera_quaternion_order") != "wxyz":
        raise ValueError(
            "This capture requires the report-proven camera quaternion order wxyz; "
            f"got {pinhole.get('camera_quaternion_order')!r}"
        )
    original_T_virtual = np.eye(4, dtype=np.float64)
    original_T_virtual[:3, :3] = np.asarray(
        pinhole["original_camera_R_virtual_camera"], dtype=np.float64
    )
    tf_camera_T_image_camera = np.eye(4, dtype=np.float64)
    tf_camera_T_image_camera[:3, :3] = np.asarray(
        floor["tf_camera_R_image_camera"], dtype=np.float64
    )
    post = original_T_virtual @ tf_camera_T_image_camera
    validate_transform_series("camera post-transform", post[None, :, :])
    return post, {
        "camera_quaternion_order": "wxyz",
        "base_T_image_camera": (
            "base_T_head_camera @ original_camera_T_virtual_camera "
            "@ tf_camera_T_image_camera"
        ),
        "original_camera_R_virtual_camera": original_T_virtual[:3, :3].tolist(),
        "tf_camera_R_image_camera": tf_camera_T_image_camera[:3, :3].tolist(),
    }


def dense_map_body_poses(
    odom_timestamps_ns: np.ndarray,
    odom_poses: np.ndarray,
    anchor_timestamps_ns: np.ndarray,
    map_T_body_anchors: np.ndarray,
    target_timestamps_ns: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    odom_at_anchors, odom_anchor_clamped = interpolate_transforms(
        odom_timestamps_ns, odom_poses, anchor_timestamps_ns
    )
    if odom_anchor_clamped:
        raise ValueError(
            f"{odom_anchor_clamped} LiDAR map anchors fall outside odometry coverage"
        )
    map_T_odom_anchors = np.asarray(
        [map_pose @ np.linalg.inv(odom_pose) for map_pose, odom_pose in zip(
            map_T_body_anchors, odom_at_anchors
        )]
    )
    map_T_odom, correction_clamped = interpolate_transforms(
        anchor_timestamps_ns, map_T_odom_anchors, target_timestamps_ns
    )
    odom_T_body, odom_target_clamped = interpolate_transforms(
        odom_timestamps_ns, odom_poses, target_timestamps_ns
    )
    if odom_target_clamped:
        raise ValueError(
            f"{odom_target_clamped} requested poses fall outside odometry coverage"
        )
    map_T_body = map_T_odom @ odom_T_body

    reconstructed_anchors, _ = dense_map_body_poses_without_report(
        odom_timestamps_ns,
        odom_poses,
        anchor_timestamps_ns,
        map_T_odom_anchors,
    )
    anchor_translation_errors = np.linalg.norm(
        reconstructed_anchors[:, :3, 3] - map_T_body_anchors[:, :3, 3], axis=1
    )
    anchor_rotation_errors = Rotation.from_matrix(
        np.swapaxes(map_T_body_anchors[:, :3, :3], 1, 2)
        @ reconstructed_anchors[:, :3, :3]
    ).magnitude()
    report = {
        "method": "interpolate_map_T_odom_se3_then_compose_dense_odom_T_base_link",
        "anchor_count": int(len(anchor_timestamps_ns)),
        "correction_interpolation_clamped": correction_clamped,
        "targets_before_first_anchor": int(
            np.count_nonzero(target_timestamps_ns < anchor_timestamps_ns[0])
        ),
        "targets_after_last_anchor": int(
            np.count_nonzero(target_timestamps_ns > anchor_timestamps_ns[-1])
        ),
        "maximum_anchor_translation_error_m": float(anchor_translation_errors.max()),
        "maximum_anchor_rotation_error_deg": float(
            np.rad2deg(anchor_rotation_errors.max())
        ),
    }
    return map_T_body, report


def dense_map_body_poses_without_report(
    odom_timestamps_ns: np.ndarray,
    odom_poses: np.ndarray,
    target_timestamps_ns: np.ndarray,
    map_T_odom_anchors: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Compose already-time-aligned corrections; used for anchor self-checks."""

    if len(target_timestamps_ns) != len(map_T_odom_anchors):
        raise ValueError("Correction and target counts differ")
    odom_T_body, clamped = interpolate_transforms(
        odom_timestamps_ns, odom_poses, target_timestamps_ns
    )
    return map_T_odom_anchors @ odom_T_body, clamped


def path_length(poses: np.ndarray) -> float:
    if len(poses) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum())


def ensure_output_directory(output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def link_dataset_payload(source: Path, output: Path) -> None:
    excluded = {"pose", "tick_index.json"}
    for path in source.iterdir():
        if path.name in excluded:
            continue
        destination = output / path.name
        destination.symlink_to(path.resolve(), target_is_directory=path.is_dir())


def write_pose_file(path: Path, poses: np.ndarray) -> None:
    path.write_text(
        "".join(
            " ".join(f"{value:.12g}" for value in pose.reshape(-1)) + "\n"
            for pose in poses
        )
    )


def load_lidar_manifest(raw_dataset: Path) -> tuple[np.ndarray, list[Path]]:
    timestamps = []
    paths = []
    for record in load_jsonl(raw_dataset / "manifest.jsonl"):
        lidar_records = record.get("lidar") or []
        if len(lidar_records) != 1:
            raise ValueError(f"Expected one LiDAR sample at tick {record.get('tick')}")
        sample = lidar_records[0]
        timestamps.append(int(sample["sensor_time_ns"]))
        paths.append(raw_dataset / sample["path"])
    return np.asarray(timestamps, dtype=np.int64), paths


def validate_lidar_cloud(
    raw_dataset: Path,
    lidar_map: Path,
    odom_timestamps_ns: np.ndarray,
    odom_poses: np.ndarray,
    anchor_timestamps_ns: np.ndarray,
    map_T_body_anchors: np.ndarray,
    aux_records: list[dict[str, Any]],
    frame_count: int,
    points_per_frame: int,
    threshold_m: float,
) -> dict[str, Any]:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "--validate-lidar-cloud requires open3d; use the repository .repro venv"
        ) from error
    cloud_path = lidar_map / "global_cloud_cleaned.pcd"
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    cloud_points = np.asarray(cloud.points, dtype=np.float64)
    if len(cloud_points) == 0:
        raise ValueError(f"LiDAR map cloud is empty: {cloud_path}")
    tree = cKDTree(cloud_points)
    lidar_timestamps, scan_paths = load_lidar_manifest(raw_dataset)
    eligible = np.flatnonzero(
        (lidar_timestamps >= np.min(odom_timestamps_ns))
        & (lidar_timestamps <= np.max(odom_timestamps_ns))
    )
    if len(eligible) == 0:
        raise ValueError("No LiDAR scans fall within dense odometry coverage")
    chosen_offsets = np.unique(
        np.linspace(0, len(eligible) - 1, min(frame_count, len(eligible)))
        .round()
        .astype(int)
    )
    chosen = eligible[chosen_offsets]
    target_timestamps = lidar_timestamps[chosen]
    map_T_body, propagation = dense_map_body_poses(
        odom_timestamps_ns,
        odom_poses,
        anchor_timestamps_ns,
        map_T_body_anchors,
        target_timestamps,
    )
    lidar_samples = [aux_pose_sample(record, "lidar", "xyzw") for record in aux_records]
    lidar_aux_times = np.asarray([sample[0] for sample in lidar_samples], dtype=np.int64)
    body_T_lidar, lidar_clamped = interpolate_transforms(
        lidar_aux_times,
        np.asarray([sample[1] for sample in lidar_samples]),
        target_timestamps,
    )
    if lidar_clamped:
        raise ValueError("LiDAR extrinsics do not cover validation timestamps")
    per_frame = []
    all_distances = []
    rng = np.random.default_rng(20260722)
    for local_index, source_index in enumerate(chosen):
        points = np.asarray(np.load(scan_paths[source_index]), dtype=np.float64)
        points = points[:, :3]
        finite = np.isfinite(points).all(axis=1)
        ranges = np.linalg.norm(points, axis=1)
        points = points[finite & (ranges > 0.2) & (ranges < 20.0)]
        if len(points) > points_per_frame:
            points = points[rng.choice(len(points), points_per_frame, replace=False)]
        map_T_lidar = map_T_body[local_index] @ body_T_lidar[local_index]
        map_points = points @ map_T_lidar[:3, :3].T + map_T_lidar[:3, 3]
        distances = tree.query(map_points, workers=-1)[0]
        all_distances.append(distances)
        per_frame.append(
            {
                "source_index": int(source_index),
                "sensor_time_ns": int(target_timestamps[local_index]),
                "points": int(len(distances)),
                "median_nearest_neighbor_m": float(np.median(distances)),
                "p95_nearest_neighbor_m": float(np.percentile(distances, 95.0)),
                "within_threshold_ratio": float(np.mean(distances <= threshold_m)),
            }
        )
    distances = np.concatenate(all_distances)
    return {
        "cloud": str(cloud_path.resolve()),
        "cloud_sha256": sha256_file(cloud_path),
        "sampled_frames": int(len(chosen)),
        "sampled_points": int(len(distances)),
        "threshold_m": threshold_m,
        "median_nearest_neighbor_m": float(np.median(distances)),
        "p95_nearest_neighbor_m": float(np.percentile(distances, 95.0)),
        "within_threshold_ratio": float(np.mean(distances <= threshold_m)),
        "propagation": propagation,
        "frames": per_frame,
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.resolve()
    raw_dataset = args.raw_dataset.resolve()
    lidar_map = args.lidar_map.resolve()
    pinhole_report = args.pinhole_report.resolve()
    floor_report = args.floor_calibration_report.resolve()
    output = args.output.resolve()
    for required in (
        dataset / "tick_index.json",
        dataset / "pose/poses.txt",
        dataset / "pose/pose_timestamps_ns.txt",
        raw_dataset / "state/000000/odom.jsonl",
        raw_dataset / "poses/dense_global/000000/aux_poses.jsonl",
        lidar_map / "poses.txt",
        lidar_map / "times.txt",
        pinhole_report,
        floor_report,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    tick_index = json.loads((dataset / "tick_index.json").read_text())
    frames = tick_index.get("frames") or []
    target_timestamps = np.asarray(
        [int(frame["pose_sensor_time_ns"]) for frame in frames], dtype=np.int64
    )
    pose_timestamps = np.loadtxt(
        dataset / "pose/pose_timestamps_ns.txt", dtype=np.int64, ndmin=1
    )
    if len(frames) == 0 or len(frames) != len(pose_timestamps):
        raise ValueError("Prepared frame and pose timestamp counts differ or are empty")
    if not np.array_equal(target_timestamps, pose_timestamps):
        raise ValueError("tick_index pose times disagree with pose_timestamps_ns.txt")
    if np.any(np.diff(target_timestamps) <= 0):
        raise ValueError("Prepared pose timestamps are not strictly increasing")

    odom_records = load_jsonl(raw_dataset / "state/000000/odom.jsonl")
    odom_samples = [odom_pose_sample(record) for record in odom_records]
    odom_timestamps = np.asarray([sample[0] for sample in odom_samples], dtype=np.int64)
    odom_poses = np.asarray([sample[1] for sample in odom_samples])
    aux_records = load_jsonl(
        raw_dataset / "poses/dense_global/000000/aux_poses.jsonl"
    )
    camera_samples = [aux_pose_sample(record, "head_camera", "wxyz") for record in aux_records]
    camera_timestamps = np.asarray([sample[0] for sample in camera_samples], dtype=np.int64)
    body_T_head_camera, camera_clamped = interpolate_transforms(
        camera_timestamps,
        np.asarray([sample[1] for sample in camera_samples]),
        target_timestamps,
    )
    if camera_clamped:
        raise ValueError(f"{camera_clamped} camera poses fall outside auxiliary pose coverage")
    post_transform, camera_contract = post_camera_transform(pinhole_report, floor_report)
    body_T_image_camera = body_T_head_camera @ post_transform

    anchor_timestamps, map_T_body_anchors = load_map_anchors(lidar_map)
    map_T_body, propagation_report = dense_map_body_poses(
        odom_timestamps,
        odom_poses,
        anchor_timestamps,
        map_T_body_anchors,
        target_timestamps,
    )
    map_T_image_camera = map_T_body @ body_T_image_camera
    validate_transform_series("output camera poses", map_T_image_camera)

    ensure_output_directory(output, args.overwrite)
    link_dataset_payload(dataset, output)
    (output / "pose").mkdir()
    write_pose_file(output / "pose/poses.txt", map_T_image_camera)
    shutil.copy2(
        dataset / "pose/pose_timestamps_ns.txt",
        output / "pose/pose_timestamps_ns.txt",
    )

    output_tick = copy.deepcopy(tick_index)
    output_tick["source_dataset"] = str(dataset)
    output_tick["pose_frame"] = "lidar_map"
    output_tick["pose_composition"] = (
        "lidar_map_T_odom(interpolated_from_lidar_map_anchors) "
        "@ odom_T_base_link @ base_link_T_head_camera "
        "@ original_camera_T_virtual_camera @ tf_camera_T_image_camera"
    )
    output_tick["pose_time_alignment"] = {
        "method": "dense_odom_plus_interpolated_lidar_map_T_odom_correction",
        "pose_timestamp_file": "pose/pose_timestamps_ns.txt",
        "pose_row_field": "pose_row",
        "registration_report": "lidar_map_alignment_report.json",
    }
    output_tick["lidar_map_alignment"] = {
        "map_directory": str(lidar_map),
        "map_pose_semantics": "lidar_map_T_base_link",
        "report": "lidar_map_alignment_report.json",
    }
    for frame in output_tick["frames"]:
        index = int(frame["idx"])
        frame["cam0"] = str(output / "rgb" / f"{index:08d}.png")
        frame["cam1"] = str(output / "stereo_right" / f"{index:08d}.png")
    (output / "tick_index.json").write_text(json.dumps(output_tick, indent=2) + "\n")

    translations = map_T_image_camera[:, :3, 3]
    report: dict[str, Any] = {
        "schema": "daaam.lidar_map_alignment.v1",
        "status": "passed",
        "output_dataset": str(output),
        "source_dataset": str(dataset),
        "raw_dataset": str(raw_dataset),
        "coordinate_contract": {
            "output_pose": "lidar_map_T_image_camera",
            "lidar_map_pose_rows": "lidar_map_T_base_link",
            "correction": "lidar_map_T_odom = lidar_map_T_base_link @ inverse(odom_T_base_link)",
            "propagation": "lidar_map_T_base_link(t) = interpolated_lidar_map_T_odom(t) @ odom_T_base_link(t)",
            **camera_contract,
        },
        "counts": {
            "output_frames": len(frames),
            "odometry_samples": len(odom_records),
            "auxiliary_pose_samples": len(aux_records),
            "lidar_map_anchors": len(anchor_timestamps),
        },
        "time_coverage_ns": {
            "output_first": int(target_timestamps[0]),
            "output_last": int(target_timestamps[-1]),
            "odom_first": int(odom_timestamps.min()),
            "odom_last": int(odom_timestamps.max()),
            "lidar_anchor_first": int(anchor_timestamps[0]),
            "lidar_anchor_last": int(anchor_timestamps[-1]),
        },
        "trajectory": {
            "first_translation_m": translations[0].tolist(),
            "last_translation_m": translations[-1].tolist(),
            "translation_path_length_m": path_length(map_T_image_camera),
            "body_translation_path_length_m": path_length(map_T_body),
        },
        "propagation": propagation_report,
        "inputs": {
            "tick_index_sha256": sha256_file(dataset / "tick_index.json"),
            "source_poses_sha256": sha256_file(dataset / "pose/poses.txt"),
            "odom_sha256": sha256_file(raw_dataset / "state/000000/odom.jsonl"),
            "aux_poses_sha256": sha256_file(
                raw_dataset / "poses/dense_global/000000/aux_poses.jsonl"
            ),
            "lidar_poses_sha256": sha256_file(lidar_map / "poses.txt"),
            "lidar_times_sha256": sha256_file(lidar_map / "times.txt"),
            "pinhole_report_sha256": sha256_file(pinhole_report),
            "floor_calibration_report_sha256": sha256_file(floor_report),
        },
    }
    if args.validate_lidar_cloud:
        if args.validation_frames <= 0 or args.validation_points_per_frame <= 0:
            raise ValueError("LiDAR validation sample counts must be positive")
        if args.validation_threshold_m <= 0:
            raise ValueError("LiDAR validation threshold must be positive")
        report["lidar_cloud_validation"] = validate_lidar_cloud(
            raw_dataset,
            lidar_map,
            odom_timestamps,
            odom_poses,
            anchor_timestamps,
            map_T_body_anchors,
            aux_records,
            args.validation_frames,
            args.validation_points_per_frame,
            args.validation_threshold_m,
        )
    report_path = output / "lidar_map_alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    args = parse_args()
    report = build_dataset(args)
    validation = report.get("lidar_cloud_validation")
    validation_text = ""
    if validation:
        validation_text = (
            f", cloud median={validation['median_nearest_neighbor_m']:.3f}m, "
            f"within {validation['threshold_m']:.2f}m="
            f"{100.0 * validation['within_threshold_ratio']:.1f}%"
        )
    print(
        f"Built {report['output_dataset']}: {report['counts']['output_frames']} frames, "
        f"camera path={report['trajectory']['translation_path_length_m']:.3f}m"
        f"{validation_text}"
    )


if __name__ == "__main__":
    main()
