#!/usr/bin/env python3
"""Recover the G1 TF-camera to stored-image rotation from map-frame LiDAR.

The capture records an optical-frame transform for the head camera, but the
stored ``2d_rect`` pixels do not include the fixed rotation between that TF
frame and the emitted image rays.  This program estimates only that missing
fixed rotation.  It deliberately does not use any queried object position:

* the raw camera quaternion is first checked against the recorded 4x4 matrices;
* the map-frame LiDAR floor mode is estimated from the point cloud itself;
* wide bottom-connected FastSAM regions provide independent image-floor masks;
* a fixed rotation about optical X is selected on training frames and checked
  on held-out frames.

Optical X is held fixed because the captured stereo baseline and image
horizontal axis are both declared along camera X.  Consequently floor evidence
does not get used to invent an unobservable yaw correction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


SCHEMA = "daaam.g1_image_lidar_rotation_calibration.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--prepared-dataset", required=True, type=Path)
    parser.add_argument("--label-dir", required=True, type=Path)
    parser.add_argument("--lidar-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=120)
    parser.add_argument("--validation-fold", type=int, default=5)
    parser.add_argument("--projection-scale", type=int, default=4)
    parser.add_argument("--local-radius-m", type=float, default=8.0)
    parser.add_argument("--ground-half-width-m", type=float, default=0.14)
    parser.add_argument("--minimum-floor-frames", type=int, default=30)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Invalid JSONL object at {path}:{line_number}")
        records.append(value)
    return records


def transform_from_pose_xyzw(values: Any) -> np.ndarray:
    pose = np.asarray(values, dtype=np.float64).reshape(-1)
    if pose.size != 7 or not np.isfinite(pose).all():
        raise ValueError("Expected a finite xyz + quaternion-xyzw pose")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def map_T_tf_camera(record: dict[str, Any]) -> np.ndarray:
    map_pose = record.get("map_pose", {}).get("value", {})
    if (
        map_pose.get("target_frame") != "map"
        or map_pose.get("source_frame") != "base_link"
    ):
        raise ValueError("Raw record does not contain map_T_base_link")
    camera = (
        record.get("poses", {})
        .get("values", {})
        .get("head_camera", {})
    )
    if (
        camera.get("target_frame") != "base_link"
        or camera.get("source_frame")
        != "head_left_camera_color_optical_frame"
    ):
        raise ValueError("Raw record does not contain base_link_T_head_camera")
    return transform_from_pose_xyzw(
        map_pose["pose_xyz_quat_xyzw"]
    ) @ transform_from_pose_xyzw(camera["pose_xyz_quat_xyzw"])


def rotation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.degrees(
            Rotation.from_matrix(left.T @ right).magnitude()
        )
    )


def validate_quaternion_contract(
    raw_records: list[dict[str, Any]],
    matrix_path: Path,
    source_indices: list[int],
) -> dict[str, Any]:
    matrices = np.loadtxt(matrix_path, dtype=np.float64)
    if matrices.ndim == 1:
        matrices = matrices.reshape(1, -1)
    if matrices.shape != (len(raw_records), 16):
        raise ValueError(
            "Recorded camera matrices do not match raw manifest: "
            f"{matrices.shape} vs ({len(raw_records)}, 16)"
        )
    xyzw_errors: list[float] = []
    forced_wxyz_errors: list[float] = []
    for source_index in source_indices:
        reference = matrices[source_index].reshape(4, 4)[:3, :3]
        camera = (
            raw_records[source_index]["poses"]["values"]["head_camera"]
        )
        quaternion = np.asarray(camera["orientation_xyzw"], dtype=np.float64)
        xyzw = Rotation.from_quat(quaternion).as_matrix()
        forced_wxyz = Rotation.from_quat(
            [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        ).as_matrix()
        xyzw_errors.append(rotation_error_deg(reference, xyzw))
        forced_wxyz_errors.append(
            rotation_error_deg(reference, forced_wxyz)
        )
    result = {
        "schema_field": "orientation_xyzw",
        "recorded_matrix_file": str(matrix_path.resolve()),
        "recorded_matrix_sha256": sha256_file(matrix_path),
        "samples": len(xyzw_errors),
        "xyzw_error_deg": {
            "median": float(np.median(xyzw_errors)),
            "maximum": float(np.max(xyzw_errors)),
        },
        "forced_wxyz_error_deg": {
            "median": float(np.median(forced_wxyz_errors)),
            "minimum": float(np.min(forced_wxyz_errors)),
        },
        "selected_order": "xyzw",
    }
    if result["xyzw_error_deg"]["maximum"] > 1.0e-6:
        raise ValueError(
            "orientation_xyzw does not reproduce recorded camera matrices"
        )
    if result["forced_wxyz_error_deg"]["minimum"] < 1.0:
        raise ValueError("Quaternion-order validation is not discriminative")
    return result


def estimate_ground_mode(
    points: np.ndarray,
    raw_records: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    base_heights = np.asarray(
        [
            record["map_pose"]["value"]["position"][2]
            for record in raw_records
        ],
        dtype=np.float64,
    )
    center = float(np.median(base_heights))
    lower = center - 1.0
    upper = center + 1.0
    candidates = points[
        np.isfinite(points).all(axis=1)
        & (points[:, 2] >= lower)
        & (points[:, 2] <= upper)
    ][:, 2]
    edges = np.arange(lower, upper + 0.0200001, 0.02)
    counts, edges = np.histogram(candidates, bins=edges)
    mode_index = int(np.argmax(counts))
    mode = float((edges[mode_index] + edges[mode_index + 1]) * 0.5)
    return mode, {
        "method": "dominant_2cm_z_histogram_near_recorded_base_height",
        "base_height_median_m": center,
        "search_range_m": [lower, upper],
        "histogram_bin_m": 0.02,
        "mode_bin_m": [float(edges[mode_index]), float(edges[mode_index + 1])],
        "mode_count": int(counts[mode_index]),
        "mode_center_m": mode,
    }


def select_floor_mask(label_image: np.ndarray) -> tuple[int, np.ndarray] | None:
    height, width = label_image.shape
    y0 = int(round(height * 0.875))
    x0 = int(round(width * 0.10))
    x1 = int(round(width * 0.90))
    band = label_image[y0:height, x0:x1]
    labels, counts = np.unique(band[band > 0], return_counts=True)
    candidates: list[tuple[int, int, np.ndarray]] = []
    for label, bottom_pixels in zip(labels, counts):
        mask = label_image == label
        ys, xs = np.where(mask)
        if (
            xs.size >= int(round(width * height * 0.075))
            and int(xs.max() - xs.min() + 1) >= int(round(width * 0.60))
            and int(ys.max()) >= int(round(height * 0.97))
        ):
            candidates.append((int(bottom_pixels), int(label), mask))
    if not candidates:
        return None
    _, label, mask = max(candidates, key=lambda value: (value[0], value[1]))
    return label, mask


def project_ground(
    local_points: np.ndarray,
    map_T_tf: np.ndarray,
    tf_R_image: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsics
    height, width = shape
    map_R_image = map_T_tf[:3, :3] @ tf_R_image
    image_points = map_R_image.T @ (
        local_points - map_T_tf[:3, 3]
    ).T
    depth = image_points[2]
    valid = depth > 0.20
    image_points = image_points[:, valid]
    depth = depth[valid]
    if depth.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    u = np.rint(fx * image_points[0] / depth + cx).astype(np.int32)
    v = np.rint(fy * image_points[1] / depth + cy).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[inside], v[inside]


def frame_iou(
    frame: dict[str, Any],
    tf_R_image: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    shape: tuple[int, int],
    splat_kernel: np.ndarray,
) -> float:
    u, v = project_ground(
        frame["ground_points"],
        frame["map_T_tf_camera"],
        tf_R_image,
        intrinsics,
        shape,
    )
    if u.size < 20:
        return 0.0
    projected = np.zeros(shape, dtype=np.uint8)
    projected[v, u] = 1
    projected = cv2.dilate(projected, splat_kernel) > 0
    floor_mask = frame["floor_mask"]
    intersection = int(np.count_nonzero(projected & floor_mask))
    union = int(np.count_nonzero(projected | floor_mask))
    return float(intersection / union) if union else 0.0


def score_rotation(
    frames: list[dict[str, Any]],
    angle_deg: float,
    intrinsics: tuple[float, float, float, float],
    shape: tuple[int, int],
    splat_kernel: np.ndarray,
) -> dict[str, Any]:
    correction = Rotation.from_euler("x", angle_deg, degrees=True).as_matrix()
    values = [
        frame_iou(
            frame,
            correction,
            intrinsics,
            shape,
            splat_kernel,
        )
        for frame in frames
    ]
    return {
        "angle_deg": float(angle_deg),
        "median_iou": float(np.median(values)),
        "mean_iou": float(np.mean(values)),
        "frame_count": len(values),
        "per_frame_iou": [float(value) for value in values],
    }


def select_angle(
    frames: list[dict[str, Any]],
    intrinsics: tuple[float, float, float, float],
    shape: tuple[int, int],
    splat_kernel: np.ndarray,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    coarse = [
        score_rotation(
            frames, float(angle), intrinsics, shape, splat_kernel
        )
        for angle in np.arange(-80.0, 80.0001, 2.0)
    ]
    coarse_best = max(
        coarse,
        key=lambda item: (
            item["median_iou"],
            item["mean_iou"],
            -abs(item["angle_deg"]),
        ),
    )
    fine = [
        score_rotation(
            frames, float(angle), intrinsics, shape, splat_kernel
        )
        for angle in np.arange(
            coarse_best["angle_deg"] - 3.0,
            coarse_best["angle_deg"] + 3.0001,
            0.25,
        )
    ]
    best = max(
        fine,
        key=lambda item: (
            item["median_iou"],
            item["mean_iou"],
            -abs(item["angle_deg"]),
        ),
    )
    return float(best["angle_deg"]), coarse, fine


def render_validation(
    output: Path,
    frames: list[dict[str, Any]],
    correction: np.ndarray,
    tick_index: dict[str, Any],
    full_intrinsics: tuple[float, float, float, float],
) -> None:
    selected = frames[:6]
    panels: list[np.ndarray] = []
    for frame in selected:
        rgb_path = Path(
            tick_index["frames"][frame["frame_index"]]["cam0"]
        )
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        mask_full = cv2.resize(
            np.uint8(frame["floor_mask"]),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        contours, _ = cv2.findContours(
            mask_full,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(rgb, contours, -1, (0, 165, 255), 2)
        u, v = project_ground(
            frame["ground_points"],
            frame["map_T_tf_camera"],
            correction,
            full_intrinsics,
            rgb.shape[:2],
        )
        if u.size:
            rgb[v, u] = (40, 220, 40)
        cv2.putText(
            rgb,
            f"frame {frame['frame_index']}  floor label {frame['floor_label']}",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(cv2.resize(rgb, (640, 480)))
    if not panels:
        return
    while len(panels) < 6:
        panels.append(np.zeros_like(panels[0]))
    montage = np.vstack(
        [np.hstack(panels[:3]), np.hstack(panels[3:6])]
    )
    cv2.imwrite(str(output), montage)


def main() -> None:
    args = parse_args()
    if args.sample_count < args.minimum_floor_frames:
        raise ValueError("sample-count must cover the minimum floor frames")
    if args.validation_fold < 2:
        raise ValueError("validation-fold must be at least 2")
    if args.projection_scale < 1:
        raise ValueError("projection-scale must be positive")

    source = args.source_dataset.expanduser().resolve()
    prepared = args.prepared_dataset.expanduser().resolve()
    label_dir = args.label_dir.expanduser().resolve()
    lidar_map = args.lidar_map.expanduser().resolve()
    output = args.output.expanduser().resolve()
    tick_path = prepared / "tick_index.json"
    matrix_path = (
        source
        / "poses"
        / "dense_global"
        / "000000"
        / "poses.txt"
    )
    raw_manifest_path = source / "manifest.jsonl"
    for required in (
        tick_path,
        matrix_path,
        raw_manifest_path,
        lidar_map,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not label_dir.is_dir():
        raise FileNotFoundError(label_dir)

    tick_index = load_json(tick_path)
    if tick_index.get("pose_frame") != "map":
        raise ValueError("Prepared dataset poses are not declared in map frame")
    prepared_frames = tick_index.get("frames")
    if not isinstance(prepared_frames, list) or not prepared_frames:
        raise ValueError("Prepared tick index has no frames")
    raw_records = load_jsonl(raw_manifest_path)
    sample_positions = np.linspace(
        0,
        len(prepared_frames) - 1,
        min(args.sample_count, len(prepared_frames)),
        dtype=np.int64,
    )
    frame_indices = sorted(set(int(value) for value in sample_positions))
    source_indices = [
        int(prepared_frames[index]["source_idx"]) for index in frame_indices
    ]
    quaternion_contract = validate_quaternion_contract(
        raw_records, matrix_path, source_indices
    )

    cloud = o3d.io.read_point_cloud(str(lidar_map))
    points = np.asarray(cloud.points, dtype=np.float64)
    if points.shape[0] < 1000:
        raise ValueError("LiDAR map is unexpectedly small")
    ground_mode, ground_estimate = estimate_ground_mode(points, raw_records)
    ground_lower = ground_mode - args.ground_half_width_m
    ground_upper = ground_mode + args.ground_half_width_m
    ground_points = points[
        np.isfinite(points).all(axis=1)
        & (points[:, 2] >= ground_lower)
        & (points[:, 2] <= ground_upper)
    ]
    if ground_points.shape[0] < 1000:
        raise ValueError("No dense LiDAR ground band was found")

    scale = args.projection_scale
    height = int(tick_index["height"])
    width = int(tick_index["width"])
    scaled_shape = (height // scale, width // scale)
    full_intrinsics = tuple(
        float(tick_index[field]) for field in ("fx", "fy", "cx", "cy")
    )
    scaled_intrinsics = tuple(value / scale for value in full_intrinsics)
    frames: list[dict[str, Any]] = []
    rejected_frames: list[dict[str, Any]] = []
    for frame_index in frame_indices:
        label_path = label_dir / f"{frame_index:08d}.png"
        label_image = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label_image is None or label_image.shape != (height, width):
            rejected_frames.append(
                {"frame_index": frame_index, "reason": "missing_or_wrong_shape"}
            )
            continue
        floor = select_floor_mask(label_image)
        if floor is None:
            rejected_frames.append(
                {"frame_index": frame_index, "reason": "no_wide_bottom_mask"}
            )
            continue
        floor_label, floor_mask = floor
        source_index = int(prepared_frames[frame_index]["source_idx"])
        if source_index < 0 or source_index >= len(raw_records):
            raise ValueError("Prepared source_idx is outside raw manifest")
        map_T_tf = map_T_tf_camera(raw_records[source_index])
        distance = np.linalg.norm(
            ground_points[:, :2] - map_T_tf[:2, 3],
            axis=1,
        )
        local = ground_points[distance <= args.local_radius_m]
        if local.shape[0] < 100:
            rejected_frames.append(
                {"frame_index": frame_index, "reason": "sparse_local_ground"}
            )
            continue
        if local.shape[0] > 15000:
            selection = np.linspace(
                0, local.shape[0] - 1, 15000, dtype=np.int64
            )
            local = local[selection]
        frames.append(
            {
                "frame_index": frame_index,
                "source_index": source_index,
                "floor_label": floor_label,
                "floor_mask": cv2.resize(
                    np.uint8(floor_mask),
                    (scaled_shape[1], scaled_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0,
                "map_T_tf_camera": map_T_tf,
                "ground_points": local,
            }
        )
    if len(frames) < args.minimum_floor_frames:
        raise ValueError(
            f"Only {len(frames)} independent floor frames passed selection"
        )

    training = [
        frame
        for index, frame in enumerate(frames)
        if index % args.validation_fold != 0
    ]
    validation = [
        frame
        for index, frame in enumerate(frames)
        if index % args.validation_fold == 0
    ]
    splat_kernel = np.ones((3, 3), dtype=np.uint8)
    angle_deg, coarse_curve, fine_curve = select_angle(
        training,
        scaled_intrinsics,
        scaled_shape,
        splat_kernel,
    )
    correction = Rotation.from_euler(
        "x", angle_deg, degrees=True
    ).as_matrix()
    train_selected = score_rotation(
        training,
        angle_deg,
        scaled_intrinsics,
        scaled_shape,
        splat_kernel,
    )
    validation_selected = score_rotation(
        validation,
        angle_deg,
        scaled_intrinsics,
        scaled_shape,
        splat_kernel,
    )
    train_identity = score_rotation(
        training,
        0.0,
        scaled_intrinsics,
        scaled_shape,
        splat_kernel,
    )
    validation_identity = score_rotation(
        validation,
        0.0,
        scaled_intrinsics,
        scaled_shape,
        splat_kernel,
    )
    if validation_selected["median_iou"] < 0.15:
        raise ValueError("Held-out LiDAR/floor-mask calibration score is too low")
    if (
        validation_selected["median_iou"]
        <= validation_identity["median_iou"] + 0.10
    ):
        raise ValueError("Held-out frames do not validate the fixed rotation")
    if abs(angle_deg) >= 79.0:
        raise ValueError("Selected image rotation lies on the search boundary")

    validation_image = output.with_name(
        f"{output.stem}_validation.png"
    )
    render_validation(
        validation_image,
        validation,
        correction,
        tick_index,
        full_intrinsics,
    )
    calibration_labels = sorted(
        {int(frame["floor_label"]) for frame in frames}
    )
    payload = {
        "schema": SCHEMA,
        "status": "passed",
        "method": "held_out_lidar_floor_mask_iou_fixed_optical_x",
        "source_dataset": str(source),
        "prepared_dataset": str(prepared),
        "label_dir": str(label_dir),
        "lidar_map": str(lidar_map),
        "coordinate_frame": "map",
        "query_target_object_used": False,
        "quaternion_contract": quaternion_contract,
        "axis_observability": {
            "fixed_axis": "tf_camera optical X",
            "reason": (
                "captured stereo baseline and stored image horizontal axis "
                "both declare camera X"
            ),
            "estimated_parameter": "rotation_about_optical_x_deg",
            "yaw_correction_deg": 0.0,
            "yaw_policy": "preserve_recorded_tf_yaw",
        },
        "ground_estimate": {
            **ground_estimate,
            "selected_band_m": [ground_lower, ground_upper],
            "selected_points": int(ground_points.shape[0]),
        },
        "frame_selection": {
            "uniform_candidates": len(frame_indices),
            "accepted": len(frames),
            "training": len(training),
            "held_out_validation": len(validation),
            "validation_fold": args.validation_fold,
            "accepted_frame_indices": [
                int(frame["frame_index"]) for frame in frames
            ],
            "dynamically_selected_floor_labels": calibration_labels,
            "rejected": rejected_frames,
        },
        "search": {
            "coarse_range_deg": [-80.0, 80.0],
            "coarse_step_deg": 2.0,
            "fine_step_deg": 0.25,
            "projection_scale": scale,
            "splat_kernel_pixels_at_scaled_resolution": [3, 3],
            "coarse_curve": [
                {
                    key: value
                    for key, value in item.items()
                    if key != "per_frame_iou"
                }
                for item in coarse_curve
            ],
            "fine_curve": [
                {
                    key: value
                    for key, value in item.items()
                    if key != "per_frame_iou"
                }
                for item in fine_curve
            ],
        },
        "selected_rotation_about_optical_x_deg": angle_deg,
        "tf_camera_R_image_camera": correction.tolist(),
        "tf_camera_R_image_camera_euler_xyz_deg": [
            float(value)
            for value in Rotation.from_matrix(correction).as_euler(
                "xyz", degrees=True
            )
        ],
        "training_score": train_selected,
        "held_out_validation_score": validation_selected,
        "identity_training_score": train_identity,
        "identity_held_out_validation_score": validation_identity,
        "inputs": {
            "raw_manifest": {
                "path": str(raw_manifest_path),
                "sha256": sha256_file(raw_manifest_path),
            },
            "prepared_tick_index": {
                "path": str(tick_path),
                "sha256": sha256_file(tick_path),
            },
            "lidar_map": {
                "path": str(lidar_map),
                "sha256": sha256_file(lidar_map),
            },
        },
        "validation_image": (
            str(validation_image)
            if validation_image.is_file()
            else None
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "selected_rotation_about_optical_x_deg": angle_deg,
                "training_median_iou": train_selected["median_iou"],
                "held_out_median_iou": validation_selected["median_iou"],
                "accepted_frames": len(frames),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
