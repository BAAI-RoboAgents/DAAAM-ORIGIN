#!/usr/bin/env python3
"""Fuse one recorded LiDAR scan with a scale-corrected stereo depth frame.

The raw manifest supplies the authoritative camera/LiDAR timestamp pairing.
Transforms retain their recorded TF semantics:

    map_T_lidar  = map_T_base(t_lidar) @ base_T_lidar(t_lidar)
    map_T_camera = map_T_base(t_camera) @ base_T_camera(t_camera)

No additional image-frame or floor-alignment rotation is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", required=True, type=Path)
    parser.add_argument("--depth-dataset", required=True, type=Path)
    parser.add_argument("--source-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-depth-m", type=float, default=0.25)
    parser.add_argument("--maximum-depth-m", type=float, default=5.0)
    parser.add_argument("--overlay-maximum-depth-m", type=float, default=12.0)
    parser.add_argument("--camera-point-stride", type=int, default=3)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pose_matrix(position: Any, quaternion_xyzw: Any) -> np.ndarray:
    position_array = np.asarray(position, dtype=np.float64)
    quaternion_array = np.asarray(quaternion_xyzw, dtype=np.float64)
    if position_array.shape != (3,) or quaternion_array.shape != (4,):
        raise ValueError("Pose must contain a 3-vector and an xyzw quaternion")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion_array).as_matrix()
    transform[:3, 3] = position_array
    return transform


def unique_samples(
    timestamps: np.ndarray, positions: np.ndarray, quaternions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    positions = positions[order]
    quaternions = quaternions[order]
    keep = np.r_[np.diff(timestamps) != 0, True]
    timestamps = timestamps[keep]
    positions = positions[keep]
    quaternions = quaternions[keep]
    if timestamps.size < 2 or not np.all(np.diff(timestamps) > 0):
        raise ValueError("Pose timestamps must contain at least two unique samples")
    return timestamps, positions, quaternions


def interpolate_pose(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    target_timestamp_ns: int,
) -> tuple[np.ndarray, bool]:
    timestamps, positions, quaternions = unique_samples(
        timestamps, positions, quaternions
    )
    target = float(target_timestamp_ns)
    clipped = float(np.clip(target, timestamps[0], timestamps[-1]))
    translation = np.array(
        [
            np.interp(clipped, timestamps, positions[:, axis])
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    rotation = Slerp(
        timestamps.astype(np.float64), Rotation.from_quat(quaternions)
    )([clipped])[0]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = translation
    return transform, clipped != target


def map_pose_samples(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = []
    positions = []
    quaternions = []
    for record in records:
        if record.get("target_frame") != "map" or record.get("source_frame") != "base_link":
            raise ValueError("Expected map_T_base_link records")
        pose = record["pose"]
        if pose.get("target_frame") != "map" or pose.get("source_frame") != "base_link":
            raise ValueError("Nested map pose does not describe map_T_base_link")
        timestamps.append(int(pose.get("timestamp_ns", record["sensor_time_ns"])))
        positions.append(pose["position"])
        quaternions.append(pose["orientation_xyzw"])
    return (
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(quaternions, dtype=np.float64),
    )


def auxiliary_pose_samples(
    records: list[dict[str, Any]], name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = []
    positions = []
    quaternions = []
    for record in records:
        pose = record["poses"][name]
        if pose.get("target_frame") != "base_link":
            raise ValueError(f"Expected base_T_{name} records")
        timestamps.append(int(pose["timestamp_ns"]))
        positions.append(pose["position"])
        quaternions.append(pose["orientation_xyzw"])
    return (
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(quaternions, dtype=np.float64),
    )


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def z_buffer_projection(
    camera_points: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> dict[str, np.ndarray]:
    z = camera_points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * camera_points[:, 0] / z + cx
        v = fy * camera_points[:, 1] / z + cy
    inside = (
        np.isfinite(camera_points).all(axis=1)
        & (z >= minimum_depth_m)
        & (z <= maximum_depth_m)
        & (u >= 0.0)
        & (u <= width - 1)
        & (v >= 0.0)
        & (v <= height - 1)
    )
    source_indices = np.flatnonzero(inside)
    u_pixel = np.rint(u[inside]).astype(np.int32)
    v_pixel = np.rint(v[inside]).astype(np.int32)
    depth = z[inside]
    flat = v_pixel.astype(np.int64) * width + u_pixel
    depth_order = np.argsort(depth)
    _, first = np.unique(flat[depth_order], return_index=True)
    selected = depth_order[first]
    return {
        "inside_source_indices": source_indices,
        "source_indices": source_indices[selected],
        "u": u_pixel[selected],
        "v": v_pixel[selected],
        "depth": depth[selected],
    }


def colorize_depth(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    normalized = np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)
    encoded = np.rint(normalized * 255.0).astype(np.uint8)
    return cv2.applyColorMap(encoded[:, None], cv2.COLORMAP_TURBO)[:, 0, :]


def draw_projected_points(
    rgb_bgr: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    colors_bgr: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    output = rgb_bgr.copy()
    for x, y, color in zip(u, v, colors_bgr):
        cv2.circle(
            output,
            (int(x), int(y)),
            radius,
            tuple(int(value) for value in color),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return output


def error_summary(prediction: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if prediction.size == 0:
        return {"count": 0}
    signed = prediction - reference
    absolute = np.abs(signed)
    relative = absolute / reference
    return {
        "count": int(prediction.size),
        "mean_signed_stereo_minus_lidar_m": float(signed.mean()),
        "median_signed_stereo_minus_lidar_m": float(np.median(signed)),
        "mean_absolute_error_m": float(absolute.mean()),
        "median_absolute_error_m": float(np.median(absolute)),
        "p90_absolute_error_m": float(np.percentile(absolute, 90)),
        "median_absolute_relative_error": float(np.median(relative)),
        "within_0_10_m_ratio": float((absolute <= 0.10).mean()),
        "within_0_20_m_ratio": float((absolute <= 0.20).mean()),
        "within_0_50_m_ratio": float((absolute <= 0.50).mean()),
    }


def write_binary_ply(path: Path, points: np.ndarray, colors_rgb: np.ndarray) -> None:
    if points.shape[0] != colors_rgb.shape[0]:
        raise ValueError("PLY point and color counts differ")
    vertices = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors_rgb[:, 0]
    vertices["green"] = colors_rgb[:, 1]
    vertices["blue"] = colors_rgb[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def save_comparison_figure(
    path: Path,
    rgb_bgr: np.ndarray,
    lidar_overlay_bgr: np.ndarray,
    residual_overlay_bgr: np.ndarray,
    skew_ms: float,
    common_count: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(22, 6.2), constrained_layout=True)
    panels = (
        (rgb_bgr, "Original left image"),
        (
            lidar_overlay_bgr,
            f"LiDAR projected into left image\ncamera − LiDAR = {skew_ms:+.3f} ms",
        ),
        (
            residual_overlay_bgr,
            f"Stereo − LiDAR depth residual\n{common_count} common pixels, ±1 m color scale",
        ),
    )
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_map_view(
    path: Path,
    camera_points_map: np.ndarray,
    lidar_points_map: np.ndarray,
    camera_center_map: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    camera_sample = camera_points_map[:: max(1, camera_points_map.shape[0] // 30000)]
    lidar_sample = lidar_points_map[:: max(1, lidar_points_map.shape[0] // 15000)]
    axes[0].scatter(
        camera_sample[:, 0],
        camera_sample[:, 1],
        s=0.35,
        c="#707070",
        alpha=0.25,
        label="stereo depth",
    )
    axes[0].scatter(
        lidar_sample[:, 0],
        lidar_sample[:, 1],
        s=1.0,
        c="#ff2020",
        alpha=0.75,
        label="LiDAR in camera FOV",
    )
    axes[0].scatter(
        [camera_center_map[0]],
        [camera_center_map[1]],
        s=45,
        c="#00b050",
        marker="x",
        label="camera",
    )
    axes[0].set_xlabel("map x (m)")
    axes[0].set_ylabel("map y (m)")
    axes[0].set_title("Top view in map frame")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    axes[0].legend(markerscale=3)

    axes[1].scatter(
        camera_sample[:, 0],
        camera_sample[:, 2],
        s=0.35,
        c="#707070",
        alpha=0.25,
    )
    axes[1].scatter(
        lidar_sample[:, 0],
        lidar_sample[:, 2],
        s=1.0,
        c="#ff2020",
        alpha=0.75,
    )
    axes[1].scatter(
        [camera_center_map[0]],
        [camera_center_map[2]],
        s=45,
        c="#00b050",
        marker="x",
    )
    axes[1].set_xlabel("map x (m)")
    axes[1].set_ylabel("map z (m)")
    axes[1].set_title("Side view in map frame")
    axes[1].axis("equal")
    axes[1].grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.source_index < 0:
        raise ValueError("source-index must be non-negative")
    if args.camera_point_stride <= 0:
        raise ValueError("camera-point-stride must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {args.output}")
    args.output.mkdir(parents=True)

    manifest_records = load_jsonl(args.raw_dataset / "manifest.jsonl")
    if args.source_index >= len(manifest_records):
        raise IndexError("source-index is outside the raw manifest")
    record = manifest_records[args.source_index]
    if int(record["tick"]) != args.source_index:
        raise ValueError("Manifest line and tick do not match")
    left_descriptor = next(
        image for image in record["images"] if image["camera"] == "cam0"
    )
    lidar_descriptor = next(
        lidar for lidar in record["lidar"] if lidar["lidar"] == "lidar0"
    )
    camera_timestamp_ns = int(left_descriptor["sensor_time_ns"])
    lidar_timestamp_ns = int(lidar_descriptor["sensor_time_ns"])

    tick_index = load_json(args.depth_dataset / "tick_index.json")
    matching_frames = [
        frame
        for frame in tick_index["frames"]
        if int(frame["source_idx"]) == args.source_index
    ]
    if len(matching_frames) != 1:
        raise ValueError(
            f"Expected one depth mapping for source {args.source_index}, "
            f"found {len(matching_frames)}"
        )
    depth_frame = matching_frames[0]
    depth_index = int(depth_frame["idx"])
    depth_path = args.depth_dataset / "depth" / f"{depth_index:08d}.png"
    rgb_path = args.raw_dataset / left_descriptor["path"]
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or depth_mm is None:
        raise FileNotFoundError("Could not read the RGB or depth image")
    if depth_mm.dtype != np.uint16 or depth_mm.shape != rgb_bgr.shape[:2]:
        raise ValueError("Depth must be uint16 millimetres with the RGB dimensions")
    depth_m = depth_mm.astype(np.float32) / 1000.0

    camera_info = load_json(args.depth_dataset / "camera_info.json")
    calibration_application_path = (
        args.depth_dataset / "floor_calibration_application.json"
    )
    calibration_application = (
        load_json(calibration_application_path)
        if calibration_application_path.is_file()
        else None
    )
    height, width = rgb_bgr.shape[:2]
    if width != int(camera_info["width"]) or height != int(camera_info["height"]):
        raise ValueError("RGB dimensions and camera calibration differ")
    fx = float(camera_info["fx"])
    fy = float(camera_info["fy"])
    cx = float(camera_info["cx"])
    cy = float(camera_info["cy"])

    map_records = load_jsonl(
        args.raw_dataset / "state" / "000000" / "map_pose.jsonl"
    )
    auxiliary_records = load_jsonl(
        args.raw_dataset
        / "poses"
        / "dense_global"
        / "000000"
        / "aux_poses.jsonl"
    )
    map_samples = map_pose_samples(map_records)
    camera_samples = auxiliary_pose_samples(auxiliary_records, "head_camera")
    lidar_samples = auxiliary_pose_samples(auxiliary_records, "lidar")
    map_T_base_at_camera, map_camera_clamped = interpolate_pose(
        *map_samples, camera_timestamp_ns
    )
    map_T_base_at_lidar, map_lidar_clamped = interpolate_pose(
        *map_samples, lidar_timestamp_ns
    )
    base_T_camera, camera_pose_clamped = interpolate_pose(
        *camera_samples, camera_timestamp_ns
    )
    base_T_lidar, lidar_pose_clamped = interpolate_pose(
        *lidar_samples, lidar_timestamp_ns
    )
    if any(
        (
            map_camera_clamped,
            map_lidar_clamped,
            camera_pose_clamped,
            lidar_pose_clamped,
        )
    ):
        raise ValueError("A requested pose timestamp fell outside the recorded range")
    map_T_camera = map_T_base_at_camera @ base_T_camera
    map_T_lidar = map_T_base_at_lidar @ base_T_lidar
    camera_T_lidar = np.linalg.inv(map_T_camera) @ map_T_lidar

    raw_lidar_points = np.load(
        args.raw_dataset / lidar_descriptor["path"], allow_pickle=False
    )
    if raw_lidar_points.ndim != 2 or raw_lidar_points.shape[1] != 3:
        raise ValueError("LiDAR array must have shape (N, 3)")
    valid_lidar = np.isfinite(raw_lidar_points).all(axis=1) & (
        np.linalg.norm(raw_lidar_points, axis=1) > 0.01
    )
    lidar_points = raw_lidar_points[valid_lidar].astype(np.float64)
    lidar_points_map = transform_points(map_T_lidar, lidar_points)
    lidar_points_camera = transform_points(camera_T_lidar, lidar_points)

    overlay_projection = z_buffer_projection(
        lidar_points_camera,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        args.minimum_depth_m,
        args.overlay_maximum_depth_m,
    )
    comparison_projection = z_buffer_projection(
        lidar_points_camera,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        args.minimum_depth_m,
        args.maximum_depth_m,
    )
    lidar_colors_bgr = colorize_depth(
        overlay_projection["depth"],
        args.minimum_depth_m,
        args.overlay_maximum_depth_m,
    )
    lidar_overlay_bgr = draw_projected_points(
        rgb_bgr,
        overlay_projection["u"],
        overlay_projection["v"],
        lidar_colors_bgr,
    )
    cv2.imwrite(str(args.output / "lidar_on_left_rgb.png"), lidar_overlay_bgr)

    stride = args.camera_point_stride
    grid_v, grid_u = np.mgrid[0:height:stride, 0:width:stride]
    sampled_depth = depth_m[::stride, ::stride]
    valid_camera_depth = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= args.minimum_depth_m)
        & (sampled_depth <= args.maximum_depth_m)
    )
    sampled_u = grid_u[valid_camera_depth].astype(np.float64)
    sampled_v = grid_v[valid_camera_depth].astype(np.float64)
    sampled_z = sampled_depth[valid_camera_depth].astype(np.float64)
    camera_points_camera = np.column_stack(
        (
            (sampled_u - cx) * sampled_z / fx,
            (sampled_v - cy) * sampled_z / fy,
            sampled_z,
        )
    )
    camera_points_map = transform_points(map_T_camera, camera_points_camera)
    camera_colors_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)[
        grid_v[valid_camera_depth], grid_u[valid_camera_depth]
    ]

    comparison_u = comparison_projection["u"]
    comparison_v = comparison_projection["v"]
    lidar_reference = comparison_projection["depth"]
    stereo_prediction = depth_m[comparison_v, comparison_u]
    common = (
        np.isfinite(stereo_prediction)
        & (stereo_prediction >= args.minimum_depth_m)
        & (stereo_prediction <= args.maximum_depth_m)
    )
    common_u = comparison_u[common]
    common_v = comparison_v[common]
    common_reference = lidar_reference[common]
    common_prediction = stereo_prediction[common].astype(np.float64)
    signed_residual = common_prediction - common_reference
    residual_normalized = np.clip(
        (signed_residual + 1.0) / 2.0, 0.0, 1.0
    )
    residual_colors_bgr = cv2.applyColorMap(
        np.rint(residual_normalized * 255.0).astype(np.uint8)[:, None],
        cv2.COLORMAP_COOL,
    )[:, 0, :]
    residual_overlay_bgr = draw_projected_points(
        rgb_bgr, common_u, common_v, residual_colors_bgr, radius=2
    )
    cv2.imwrite(
        str(args.output / "stereo_minus_lidar_residual_on_rgb.png"),
        residual_overlay_bgr,
    )

    depth_colors_bgr = colorize_depth(
        sampled_z, args.minimum_depth_m, args.maximum_depth_m
    )
    depth_overlay_bgr = draw_projected_points(
        rgb_bgr,
        sampled_u.astype(np.int32),
        sampled_v.astype(np.int32),
        depth_colors_bgr,
        radius=1,
    )
    cv2.imwrite(str(args.output / "stereo_depth_on_left_rgb.png"), depth_overlay_bgr)

    lidar_fov_source = comparison_projection["inside_source_indices"]
    lidar_fov_map = lidar_points_map[lidar_fov_source]
    lidar_red = np.tile(
        np.asarray([[255, 32, 32]], dtype=np.uint8),
        (lidar_fov_map.shape[0], 1),
    )
    write_binary_ply(
        args.output / "lidar_camera_fov_map.ply", lidar_fov_map, lidar_red
    )
    write_binary_ply(
        args.output / "stereo_depth_map.ply", camera_points_map, camera_colors_rgb
    )
    combined_points = np.vstack((camera_points_map, lidar_fov_map))
    combined_colors = np.vstack((camera_colors_rgb, lidar_red))
    write_binary_ply(
        args.output / "stereo_lidar_fused_map.ply",
        combined_points,
        combined_colors,
    )

    save_map_view(
        args.output / "map_frame_alignment.png",
        camera_points_map,
        lidar_fov_map,
        map_T_camera[:3, 3],
    )
    skew_ms = (camera_timestamp_ns - lidar_timestamp_ns) / 1e6
    save_comparison_figure(
        args.output / "fusion_comparison.png",
        rgb_bgr,
        lidar_overlay_bgr,
        residual_overlay_bgr,
        skew_ms,
        int(common.sum()),
    )

    relative_base_motion = np.linalg.inv(map_T_base_at_lidar) @ map_T_base_at_camera
    relative_rotation_deg = math.degrees(
        Rotation.from_matrix(relative_base_motion[:3, :3]).magnitude()
    )
    selected_rgb_path = Path(depth_frame["cam0"])
    report = {
        "contract": "no forced rotation; recorded optical-frame TF only",
        "source_index": args.source_index,
        "raw_tick": int(record["tick"]),
        "depth_frame_index": depth_index,
        "paths": {
            "raw_left_rgb": str(rgb_path.resolve()),
            "selected_left_rgb": str(selected_rgb_path.resolve()),
            "scale_only_depth": str(depth_path.resolve()),
            "raw_lidar": str(
                (args.raw_dataset / lidar_descriptor["path"]).resolve()
            ),
            "map_pose": str(
                (
                    args.raw_dataset
                    / "state"
                    / "000000"
                    / "map_pose.jsonl"
                ).resolve()
            ),
            "aux_poses": str(
                (
                    args.raw_dataset
                    / "poses"
                    / "dense_global"
                    / "000000"
                    / "aux_poses.jsonl"
                ).resolve()
            ),
        },
        "timestamps": {
            "camera_sensor_time_ns": camera_timestamp_ns,
            "lidar_sensor_time_ns": lidar_timestamp_ns,
            "camera_minus_lidar_ms": skew_ms,
            "motion_compensated_using_independent_timestamps": True,
            "base_motion_lidar_to_camera_translation_m": float(
                np.linalg.norm(relative_base_motion[:3, 3])
            ),
            "base_motion_lidar_to_camera_rotation_deg": relative_rotation_deg,
        },
        "frame_semantics": {
            "map_pose": "map_T_base_link",
            "camera_aux_pose": "base_link_T_head_left_camera_color_optical_frame",
            "lidar_aux_pose": "base_link_T_lidar_base_link",
            "map_T_camera": "map_T_base(t_camera) @ base_T_camera(t_camera)",
            "map_T_lidar": "map_T_base(t_lidar) @ base_T_lidar(t_lidar)",
            "extra_camera_rotation_applied": False,
        },
        "calibration": {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "depth_scale_already_applied": calibration_application is not None,
            "applied_depth_scale": (
                calibration_application.get("depth_scale")
                if calibration_application is not None
                else 1.0
            ),
            "applied_rotation_policy": (
                calibration_application.get("rotation_policy")
                if calibration_application is not None
                else "none"
            ),
            "effective_baseline_m": camera_info.get("baseline"),
        },
        "integrity": {
            "raw_left_sha256": sha256(rgb_path),
            "selected_left_sha256": sha256(selected_rgb_path),
            "raw_and_selected_left_byte_identical": sha256(rgb_path)
            == sha256(selected_rgb_path),
        },
        "counts": {
            "raw_lidar_points": int(raw_lidar_points.shape[0]),
            "valid_lidar_points": int(lidar_points.shape[0]),
            "lidar_points_inside_camera_fov_to_overlay_max_before_z_buffer": int(
                overlay_projection["inside_source_indices"].size
            ),
            "lidar_unique_projected_pixels_to_overlay_max": int(
                overlay_projection["depth"].size
            ),
            "lidar_points_inside_camera_fov_to_comparison_max_before_z_buffer": int(
                comparison_projection["inside_source_indices"].size
            ),
            "lidar_unique_projected_pixels_to_comparison_max": int(
                comparison_projection["depth"].size
            ),
            "stereo_depth_points_in_fused_ply": int(camera_points_map.shape[0]),
            "common_stereo_lidar_pixels": int(common.sum()),
        },
        "direct_projected_pixel_depth_error": error_summary(
            common_prediction, common_reference
        ),
        "caveats": [
            "LiDAR is sparse and has a different viewpoint from the camera.",
            "The stored NPY contains XYZ only, so per-point LiDAR deskew is unavailable.",
            "Depth residuals at occlusion boundaries do not necessarily indicate calibration error.",
        ],
        "transforms": {
            "map_T_base_at_camera": map_T_base_at_camera.tolist(),
            "map_T_base_at_lidar": map_T_base_at_lidar.tolist(),
            "base_T_camera_at_camera": base_T_camera.tolist(),
            "base_T_lidar_at_lidar": base_T_lidar.tolist(),
            "map_T_camera": map_T_camera.tolist(),
            "map_T_lidar": map_T_lidar.tolist(),
            "camera_T_lidar_motion_compensated": camera_T_lidar.tolist(),
        },
    }
    (args.output / "fusion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
