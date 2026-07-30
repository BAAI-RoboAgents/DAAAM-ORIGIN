#!/usr/bin/env python3
"""Build motion-compensated LiDAR/camera correspondences and visual evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fuse_lidar_camera_frame import (
    auxiliary_pose_samples,
    colorize_depth,
    draw_projected_points,
    interpolate_pose,
    load_json,
    load_jsonl,
    map_pose_samples,
    transform_points,
    z_buffer_projection,
)


SCHEMA = "daaam.g1_lidar_camera_ground_truth.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", required=True, type=Path)
    parser.add_argument("--prepared-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-start-index", required=True, type=int)
    parser.add_argument("--source-end-index", required=True, type=int)
    parser.add_argument("--minimum-depth-m", type=float, default=0.25)
    parser.add_argument("--maximum-depth-m", type=float, default=65.535)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(record: dict[str, Any], key: str, value: str) -> dict[str, Any]:
    items = [item for item in record[key] if item.get(key.rstrip("s")) == value]
    if key == "images":
        items = [item for item in record[key] if item.get("camera") == value]
    if not items:
        raise KeyError(f"{key} descriptor is missing: {value}")
    return items[0]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    raw = args.raw_dataset.expanduser().resolve()
    prepared = args.prepared_dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if (
        args.source_start_index < 0
        or args.source_end_index < args.source_start_index
        or not 0.0 < args.minimum_depth_m < args.maximum_depth_m <= 65.535
    ):
        raise ValueError("source range or depth range is invalid")

    records = {
        int(record["tick"]): record
        for record in load_jsonl(raw / "manifest.jsonl")
    }
    requested = range(args.source_start_index, args.source_end_index + 1)
    missing = set(requested) - set(records)
    if missing:
        raise ValueError(f"Raw source frames are absent: {sorted(missing)}")
    camera = load_json(prepared / "camera_info.json")
    prepared_index_path = prepared / "tick_index.json"
    prepared_index = load_json(prepared_index_path)
    prepared_frames = {
        int(frame["source_idx"]): frame
        for frame in prepared_index.get("frames", [])
    }
    duplicate_prepared_frames = len(prepared_frames) != len(
        prepared_index.get("frames", [])
    )
    if duplicate_prepared_frames:
        raise ValueError("Prepared tick index has duplicate source indices")
    missing_prepared = set(requested) - set(prepared_frames)
    if missing_prepared:
        raise ValueError(
            "Prepared rectified frames are absent: "
            f"{sorted(missing_prepared)}"
        )
    rectification = prepared_index.get("rectification_provenance") or {}
    left_rectification_rotation = np.asarray(
        rectification.get("left_rectification_rotation"),
        dtype=np.float64,
    )
    if left_rectification_rotation.shape != (3, 3):
        raise ValueError(
            "Prepared tick index lacks a 3x3 left rectification rotation"
        )
    if (
        not np.isfinite(left_rectification_rotation).all()
        or np.linalg.norm(
            left_rectification_rotation.T
            @ left_rectification_rotation
            - np.eye(3)
        )
        > 1.0e-8
        or abs(float(np.linalg.det(left_rectification_rotation)) - 1.0)
        > 1.0e-8
    ):
        raise ValueError("Prepared left rectification rotation is invalid")
    source_camera_T_rectified_camera = np.eye(4, dtype=np.float64)
    source_camera_T_rectified_camera[:3, :3] = (
        left_rectification_rotation.T
    )
    width = int(camera["width"])
    height = int(camera["height"])
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    map_samples = map_pose_samples(
        load_jsonl(raw / "state" / "000000" / "map_pose.jsonl")
    )
    auxiliary = load_jsonl(
        raw / "poses" / "dense_global" / "000000" / "aux_poses.jsonl"
    )
    camera_samples = auxiliary_pose_samples(auxiliary, "head_camera")
    lidar_samples = auxiliary_pose_samples(auxiliary, "lidar")

    directories = {
        name: output / name
        for name in (
            "sparse_depth_mm",
            "valid_masks",
            "overlays",
            "correspondences",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=False)
    frame_records = []
    with (output / "frames.jsonl").open("w") as frame_stream:
        for source_index in requested:
            record = records[source_index]
            image = descriptor(record, "images", "cam0")
            lidar = descriptor(record, "lidar", "lidar0")
            prepared_frame = prepared_frames[source_index]
            camera_time_ns = int(image["sensor_time_ns"])
            lidar_time_ns = int(lidar["sensor_time_ns"])
            map_T_base_camera, map_camera_clamped = interpolate_pose(
                *map_samples, camera_time_ns
            )
            map_T_base_lidar, map_lidar_clamped = interpolate_pose(
                *map_samples, lidar_time_ns
            )
            base_T_camera, camera_clamped = interpolate_pose(
                *camera_samples, camera_time_ns
            )
            base_T_lidar, lidar_clamped = interpolate_pose(
                *lidar_samples, lidar_time_ns
            )
            if any(
                (
                    map_camera_clamped,
                    map_lidar_clamped,
                    camera_clamped,
                    lidar_clamped,
                )
            ):
                raise ValueError(
                    f"Pose interpolation clamped for source frame {source_index}"
                )
            map_T_source_camera = map_T_base_camera @ base_T_camera
            map_T_camera = (
                map_T_source_camera @ source_camera_T_rectified_camera
            )
            map_T_lidar = map_T_base_lidar @ base_T_lidar
            camera_T_lidar = np.linalg.inv(map_T_camera) @ map_T_lidar
            raw_points = np.load(raw / lidar["path"], allow_pickle=False)
            if raw_points.ndim != 2 or raw_points.shape[1] != 3:
                raise ValueError(f"LiDAR XYZ has invalid shape at {source_index}")
            finite = np.isfinite(raw_points).all(axis=1)
            finite &= np.linalg.norm(raw_points, axis=1) > 0.01
            lidar_points = raw_points[finite].astype(np.float64)
            camera_points = transform_points(camera_T_lidar, lidar_points)
            map_points = transform_points(map_T_lidar, lidar_points)
            projection = z_buffer_projection(
                camera_points,
                fx,
                fy,
                cx,
                cy,
                width,
                height,
                args.minimum_depth_m,
                args.maximum_depth_m,
            )
            selected = projection["source_indices"]
            name = f"{source_index:06d}"
            depth_mm = np.zeros((height, width), dtype=np.uint16)
            depth_mm[projection["v"], projection["u"]] = np.clip(
                np.rint(projection["depth"] * 1000.0),
                1,
                65535,
            ).astype(np.uint16)
            mask = depth_mm > 0
            if not cv2.imwrite(
                str(directories["sparse_depth_mm"] / f"{name}.png"),
                depth_mm,
            ):
                raise RuntimeError("Could not write sparse LiDAR depth")
            if not cv2.imwrite(
                str(directories["valid_masks"] / f"{name}.png"),
                mask.astype(np.uint8) * 255,
            ):
                raise RuntimeError("Could not write sparse LiDAR mask")
            prepared_rgb_path = Path(prepared_frame["cam0"])
            if not prepared_rgb_path.is_absolute():
                prepared_rgb_path = prepared / prepared_rgb_path
            prepared_rgb_path = prepared_rgb_path.resolve()
            rgb = cv2.imread(str(prepared_rgb_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(prepared_rgb_path)
            colors = colorize_depth(
                projection["depth"],
                args.minimum_depth_m,
                min(args.maximum_depth_m, 12.0),
            )
            overlay = draw_projected_points(
                rgb,
                projection["u"],
                projection["v"],
                colors,
                radius=2,
            )
            if not cv2.imwrite(
                str(directories["overlays"] / f"{name}.png"),
                overlay,
            ):
                raise RuntimeError("Could not write LiDAR overlay")
            correspondence_path = directories["correspondences"] / f"{name}.npz"
            np.savez_compressed(
                correspondence_path,
                source_indices=selected.astype(np.int64),
                u=projection["u"].astype(np.int32),
                v=projection["v"].astype(np.int32),
                depth_m=projection["depth"].astype(np.float32),
                camera_points_m=camera_points[selected].astype(np.float32),
                map_points_m=map_points[selected].astype(np.float32),
                camera_T_lidar=camera_T_lidar,
                map_T_camera=map_T_camera,
                map_T_lidar=map_T_lidar,
            )
            frame_record = {
                "schema": SCHEMA,
                "source_index": source_index,
                "prepared_index": int(prepared_frame["idx"]),
                "camera_sensor_time_ns": camera_time_ns,
                "lidar_sensor_time_ns": lidar_time_ns,
                "camera_minus_lidar_ms": (
                    camera_time_ns - lidar_time_ns
                )
                / 1.0e6,
                "raw_lidar_points": int(raw_points.shape[0]),
                "finite_lidar_points": int(lidar_points.shape[0]),
                "projected_visible_points": int(len(selected)),
                "paths": {
                    "source_rgb": str((raw / image["path"]).resolve()),
                    "rgb": str(prepared_rgb_path),
                    "lidar": str((raw / lidar["path"]).resolve()),
                    "sparse_depth_mm": str(
                        directories["sparse_depth_mm"] / f"{name}.png"
                    ),
                    "valid_mask": str(
                        directories["valid_masks"] / f"{name}.png"
                    ),
                    "overlay": str(directories["overlays"] / f"{name}.png"),
                    "correspondences": str(correspondence_path),
                },
                "sha256": {
                    "source_rgb": sha256(raw / image["path"]),
                    "rgb": sha256(prepared_rgb_path),
                    "lidar": sha256(raw / lidar["path"]),
                    "correspondences": sha256(correspondence_path),
                },
                "transforms": {
                    "map_T_camera": map_T_camera.tolist(),
                    "map_T_source_camera": map_T_source_camera.tolist(),
                    "map_T_lidar": map_T_lidar.tolist(),
                    "camera_T_lidar_motion_compensated": camera_T_lidar.tolist(),
                },
            }
            frame_stream.write(
                json.dumps(frame_record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            frame_records.append(frame_record)
    manifest = {
        "schema": SCHEMA,
        "raw_dataset": str(raw),
        "prepared_dataset": str(prepared),
        "prepared_tick_index_sha256": sha256(prepared_index_path),
        "source_range": [
            args.source_start_index,
            args.source_end_index,
        ],
        "frame_count": len(frame_records),
        "camera_geometry": {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
        "depth_range_m": [
            args.minimum_depth_m,
            args.maximum_depth_m,
        ],
        "frame_semantics": {
            "map_T_camera": (
                "map_T_base(t_camera) @ base_T_head_left_camera(t_camera) "
                "@ source_camera_T_rectified_camera"
            ),
            "source_camera_T_rectified_camera": (
                "homogeneous rotation using "
                "transpose(left_rectification_rotation)"
            ),
            "map_T_lidar": "map_T_base(t_lidar) @ base_T_lidar(t_lidar)",
            "camera_T_lidar": "inv(map_T_camera) @ map_T_lidar",
            "rgb_pixel_space": "prepared rectified cam0",
        },
        "artifacts": {
            "frames_jsonl": str(output / "frames.jsonl"),
            **{name: str(path) for name, path in directories.items()},
        },
        "caveats": [
            "Sparse LiDAR is a reference only at projected visible pixels.",
            "Stored XYZ has no per-point timestamps, so scan-internal deskew is unavailable.",
            "Occlusion boundaries require manual exclusion during depth scoring.",
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
