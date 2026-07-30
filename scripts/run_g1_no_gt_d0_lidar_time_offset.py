#!/usr/bin/env python3
"""Inject camera–LiDAR clock offsets and retain native projection evidence.

The recorded camera image and LiDAR scan remain unchanged.  The injected clock
bias is applied only to the camera pose interpolation time used by the
motion-compensated camera_T_lidar transform.  Each cell stores projected point
correspondences, frozen stereo-depth comparisons, boundary distances, overlays,
per-frame metrics, and content hashes.

This is a no-human-GT D0 observability experiment.  Sparse LiDAR is a geometric
reference, but scan-internal deskew and manually adjudicated occlusion
boundaries remain unavailable.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from fuse_lidar_camera_frame import (
    auxiliary_pose_samples,
    interpolate_pose,
    load_jsonl,
    transform_points,
    z_buffer_projection,
)


SCHEMA = "daaam.g1_no_gt_d0_lidar_time_offset.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", required=True, type=Path)
    parser.add_argument("--prepared-window", required=True, type=Path)
    parser.add_argument("--geometry-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-start", type=int, default=473)
    parser.add_argument("--source-end", type=int, default=573)
    parser.add_argument("--minimum-depth-m", type=float, default=0.25)
    parser.add_argument("--maximum-depth-m", type=float, default=5.0)
    parser.add_argument("--png-compression", type=int, default=2)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(record: dict[str, Any], group: str, name: str) -> dict[str, Any]:
    key = "camera" if group == "images" else group.rstrip("s")
    rows = [row for row in record[group] if row.get(key) == name]
    if len(rows) != 1:
        raise ValueError(
            f"Raw tick {record.get('tick')} has {len(rows)} {group}/{name} rows"
        )
    return rows[0]


def source_camera_transform(prepared_index: dict[str, Any]) -> np.ndarray:
    rectification = prepared_index.get("rectification_provenance") or {}
    rotation = np.asarray(
        rectification.get("left_rectification_rotation"), dtype=np.float64
    )
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 1.0e-8
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-8
    ):
        raise ValueError("Prepared left rectification rotation is invalid")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.T
    return transform


def map_pose_samples_from_auxiliary(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read map_T_base_link samples persisted under auxiliary poses['map']."""
    timestamps = []
    positions = []
    quaternions = []
    for record in records:
        pose = record["poses"]["map"]
        if (
            pose.get("target_frame") != "map"
            or pose.get("source_frame") != "base_link"
        ):
            raise ValueError("Expected auxiliary map_T_base_link pose records")
        timestamps.append(int(pose["timestamp_ns"]))
        positions.append(pose["position"])
        quaternions.append(pose["orientation_xyzw"])
    return (
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(quaternions, dtype=np.float64),
    )


def dense_boundary_distance(depth_m: np.ndarray) -> np.ndarray:
    valid = depth_m > 0.0
    valid_u8 = valid.astype(np.uint8) * 255
    valid_boundary = (
        cv2.morphologyEx(
            valid_u8,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        > 0
    )
    depth_for_gradient = depth_m.copy()
    if valid.any():
        depth_for_gradient[~valid] = float(np.median(depth_m[valid]))
    gx = cv2.Sobel(depth_for_gradient, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_for_gradient, cv2.CV_32F, 0, 1, ksize=3)
    depth_boundary = np.hypot(gx, gy) > 0.15
    boundary = valid_boundary | (depth_boundary & valid)
    not_boundary = (~boundary).astype(np.uint8)
    return cv2.distanceTransform(not_boundary, cv2.DIST_L2, 3)


def frozen_lidar_boundary_sources(
    projection: dict[str, np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    sparse = np.zeros((height, width), dtype=np.float32)
    sparse[projection["v"], projection["u"]] = projection["depth"].astype(
        np.float32
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    maximum = cv2.dilate(sparse, kernel)
    filled = np.where(sparse > 0.0, sparse, np.float32(1.0e6))
    minimum = cv2.erode(filled, kernel)
    local_range = maximum[projection["v"], projection["u"]] - minimum[
        projection["v"], projection["u"]
    ]
    return projection["source_indices"][local_range > 0.25]


def projection_metrics(
    projection: dict[str, np.ndarray],
    stereo_depth_m: np.ndarray,
    boundary_distance_px: np.ndarray,
    frozen_boundary_sources: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    u = projection["u"]
    v = projection["v"]
    lidar_depth = projection["depth"].astype(np.float32)
    stereo = stereo_depth_m[v, u].astype(np.float32)
    valid = stereo > 0.0
    residual = stereo[valid] - lidar_depth[valid]
    if frozen_boundary_sources is None:
        boundary_sources = np.asarray([], dtype=np.int64)
        boundary_variant_indices = np.asarray([], dtype=np.int64)
    else:
        boundary_sources, _, boundary_variant_indices = np.intersect1d(
            frozen_boundary_sources,
            projection["source_indices"],
            assume_unique=True,
            return_indices=True,
        )
    boundary_distance = boundary_distance_px[
        v[boundary_variant_indices], u[boundary_variant_indices]
    ]
    metrics = {
        "projected_visible_points": int(len(u)),
        "stereo_valid_correspondence_count": int(valid.sum()),
        "stereo_valid_correspondence_ratio": float(valid.mean()) if len(valid) else 0.0,
        "mean_signed_stereo_minus_lidar_m": (
            float(residual.mean()) if residual.size else None
        ),
        "mean_absolute_stereo_lidar_error_m": (
            float(np.abs(residual).mean()) if residual.size else None
        ),
        "median_absolute_stereo_lidar_error_m": (
            float(np.median(np.abs(residual))) if residual.size else None
        ),
        "frozen_lidar_boundary_count": int(len(boundary_sources)),
        "mean_frozen_boundary_distance_to_stereo_edge_px": (
            float(boundary_distance.mean()) if boundary_distance.size else None
        ),
        "median_frozen_boundary_distance_to_stereo_edge_px": (
            float(np.median(boundary_distance)) if boundary_distance.size else None
        ),
    }
    arrays = {
        "stereo_depth_m": stereo,
        "stereo_valid": valid,
        "residual_m": residual,
        "frozen_boundary_source_indices": boundary_sources,
        "frozen_boundary_variant_indices": boundary_variant_indices,
        "frozen_boundary_distance_px": boundary_distance,
    }
    return metrics, arrays


def pixel_displacement(
    control: dict[str, np.ndarray], variant: dict[str, np.ndarray]
) -> dict[str, Any]:
    sources, control_indices, variant_indices = np.intersect1d(
        control["source_indices"],
        variant["source_indices"],
        assume_unique=True,
        return_indices=True,
    )
    du = (
        variant["u"][variant_indices].astype(np.float64)
        - control["u"][control_indices].astype(np.float64)
    )
    dv = (
        variant["v"][variant_indices].astype(np.float64)
        - control["v"][control_indices].astype(np.float64)
    )
    distance = np.hypot(du, dv)
    return {
        "common_source_point_count": int(len(sources)),
        "mean_projection_displacement_px": (
            float(distance.mean()) if distance.size else None
        ),
        "median_projection_displacement_px": (
            float(np.median(distance)) if distance.size else None
        ),
        "p90_projection_displacement_px": (
            float(np.percentile(distance, 90)) if distance.size else None
        ),
        "maximum_projection_displacement_px": (
            float(distance.max()) if distance.size else None
        ),
        "common_source_indices": sources,
        "control_common_indices": control_indices,
        "variant_common_indices": variant_indices,
        "du_px": du,
        "dv_px": dv,
        "displacement_px": distance,
    }


def render_overlay(
    path: Path,
    rgb: np.ndarray,
    control: dict[str, np.ndarray],
    variant: dict[str, np.ndarray],
    title: str,
) -> None:
    image = rgb.copy()
    for projection, color in ((control, (30, 220, 220)), (variant, (220, 30, 220))):
        count = len(projection["u"])
        if not count:
            continue
        chosen = np.linspace(0, count - 1, min(1800, count)).astype(np.int64)
        for index in chosen:
            cv2.circle(
                image,
                (int(projection["u"][index]), int(projection["v"][index])),
                1,
                color,
                -1,
                cv2.LINE_AA,
            )
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 38), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not write overlay: {path}")


def inventory(output: Path) -> dict[str, Any]:
    rows = []
    total_bytes = 0
    failures = 0
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        size = path.stat().st_size
        total_bytes += size
        try:
            digest = sha256(path)
            status = "ok"
        except OSError as exc:
            digest = ""
            status = f"failure:{type(exc).__name__}:{exc}"
            failures += 1
        rows.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": size,
                "sha256": digest,
                "status": status,
            }
        )
    write_csv(output / "EVIDENCE_INVENTORY.csv", rows)
    root = hashlib.sha256()
    for row in rows:
        root.update(
            (
                f"{row['path']}\0{row['size_bytes']}\0"
                f"{row['sha256']}\0{row['status']}\n"
            ).encode("utf-8")
        )
    result = {
        "schema": f"{SCHEMA}.evidence_inventory",
        "created_at": utc_now(),
        "object_count_before_inventory_files": len(rows),
        "regular_bytes_before_inventory_files": total_bytes,
        "hash_failures": failures,
        "root_sha256": root.hexdigest(),
        "root_definition": (
            "SHA-256 over sorted path\\0size\\0content_sha256\\0status rows"
        ),
        "self_excluded": [
            "EVIDENCE_INVENTORY.csv",
            "EVIDENCE_INVENTORY.json",
        ],
    }
    write_json(output / "EVIDENCE_INVENTORY.json", result)
    return result


def main() -> None:
    args = parse_args()
    for key in (
        "raw_dataset",
        "prepared_window",
        "geometry_dataset",
        "output",
    ):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {args.output}")
    if (
        args.source_start != 473
        or args.source_end != 573
        or not 0.0 < args.minimum_depth_m < args.maximum_depth_m <= 65.535
    ):
        raise ValueError("Frozen protocol requires source 473–573 and valid depth range")
    args.output.mkdir(parents=True)
    preregistration = {
        "schema": f"{SCHEMA}.preregistration",
        "created_at": utc_now(),
        "created_before_result_inspection": True,
        "doses_ms": [-60, -40, -20, 20, 40, 60],
        "injection": (
            "Add signed clock bias to camera pose interpolation timestamp; retain "
            "recorded camera image, LiDAR scan, and LiDAR timestamp."
        ),
        "primary_collector": "L4/E4 motion-compensated sparse projection",
        "eligible_frame": (
            "at least 100 frozen stereo-valid projected points and at least 20 "
            "frozen LiDAR boundary source points"
        ),
        "prescribed_alarm": (
            "stereo-valid projection ratio decreases by >=0.002 AND mean frozen "
            "boundary distance to a stereo-depth edge increases by >=0.10 px"
        ),
        "secondary_observability_alarm": (
            "p90 same-source projection displacement >=0.25 px"
        ),
        "medium_heavy": "absolute dose >=40 ms",
        "d0_target": (
            "medium/heavy prescribed-alarm detection >=90%, control false alarm <5%, "
            "dose direction monotone"
        ),
        "caveats": [
            "LiDAR XYZ has no per-point time; scan-internal deskew is unavailable.",
            "Stereo-depth edges substitute for manually adjudicated occlusion boundaries.",
            "This is a no-human-GT observability qualification, not dense depth GT.",
        ],
    }
    write_json(args.output / "PRE_REGISTRATION.json", preregistration)
    write_json(
        args.output / "invocation.json",
        {
            "schema": f"{SCHEMA}.invocation",
            "created_at": utc_now(),
            "argv": sys.argv,
            "resolved_inputs": {
                "raw_dataset": str(args.raw_dataset),
                "prepared_window": str(args.prepared_window),
                "geometry_dataset": str(args.geometry_dataset),
            },
            "output": str(args.output),
        },
    )

    raw_records = {
        int(row["tick"]): row
        for row in read_jsonl(args.raw_dataset / "manifest.jsonl")
    }
    prepared_index = read_json(args.prepared_window / "tick_index.json")
    prepared_frames = {
        int(row["source_idx"]): row for row in prepared_index["frames"]
    }
    geometry_index = read_json(args.geometry_dataset / "tick_index.json")
    selected_frames = sorted(
        [
            row
            for row in geometry_index["frames"]
            if args.source_start <= int(row["source_idx"]) <= args.source_end
        ],
        key=lambda row: int(row["source_idx"]),
    )
    if not selected_frames:
        raise ValueError("No selected geometry frames are in source 473–573")
    camera = read_json(args.prepared_window / "camera_info.json")
    width = int(camera["width"])
    height = int(camera["height"])
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    source_camera_T_rectified = source_camera_transform(prepared_index)
    auxiliary = load_jsonl(
        args.raw_dataset
        / "poses"
        / "dense_global"
        / "000000"
        / "aux_poses.jsonl"
    )
    map_samples = map_pose_samples_from_auxiliary(auxiliary)
    camera_samples = auxiliary_pose_samples(auxiliary, "head_camera")
    lidar_samples = auxiliary_pose_samples(auxiliary, "lidar")
    doses = (0, -60, -40, -20, 20, 40, 60)
    rows: list[dict[str, Any]] = []

    for frame in selected_frames:
        source = int(frame["source_idx"])
        selected_index = int(frame["idx"])
        record = raw_records[source]
        image_row = descriptor(record, "images", "cam0")
        lidar_row = descriptor(record, "lidar", "lidar0")
        camera_time_ns = int(image_row["sensor_time_ns"])
        lidar_time_ns = int(lidar_row["sensor_time_ns"])
        raw_points = np.load(
            args.raw_dataset / lidar_row["path"], allow_pickle=False
        )
        finite = np.isfinite(raw_points).all(axis=1)
        finite &= np.linalg.norm(raw_points, axis=1) > 0.01
        lidar_points = raw_points[finite].astype(np.float64)
        map_T_base_lidar, map_lidar_clamped = interpolate_pose(
            *map_samples, lidar_time_ns
        )
        base_T_lidar, lidar_clamped = interpolate_pose(
            *lidar_samples, lidar_time_ns
        )
        if map_lidar_clamped or lidar_clamped:
            raise ValueError(f"LiDAR pose interpolation clamped at source {source}")
        map_T_lidar = map_T_base_lidar @ base_T_lidar
        depth_path = (
            args.geometry_dataset / "depth" / f"{selected_index:08d}.png"
        )
        rgb_path = args.geometry_dataset / "rgb" / f"{selected_index:08d}.png"
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if (
            depth_mm is None
            or rgb is None
            or depth_mm.shape != (height, width)
        ):
            raise RuntimeError(f"Invalid frozen geometry input at source {source}")
        stereo_depth_m = depth_mm.astype(np.float32) / 1000.0
        boundary_distance_px = dense_boundary_distance(stereo_depth_m)
        projections: dict[int, dict[str, np.ndarray]] = {}
        transforms: dict[int, np.ndarray] = {}
        for dose in doses:
            used_camera_time_ns = camera_time_ns + int(dose * 1_000_000)
            map_T_base_camera, map_camera_clamped = interpolate_pose(
                *map_samples, used_camera_time_ns
            )
            base_T_camera, camera_clamped = interpolate_pose(
                *camera_samples, used_camera_time_ns
            )
            if map_camera_clamped or camera_clamped:
                raise ValueError(
                    f"Camera pose interpolation clamped at source {source}, dose {dose}"
                )
            map_T_source_camera = map_T_base_camera @ base_T_camera
            map_T_camera = map_T_source_camera @ source_camera_T_rectified
            camera_T_lidar = np.linalg.inv(map_T_camera) @ map_T_lidar
            camera_points = transform_points(camera_T_lidar, lidar_points)
            projections[dose] = z_buffer_projection(
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
            transforms[dose] = camera_T_lidar
        control = projections[0]
        frozen_boundary = frozen_lidar_boundary_sources(
            control, width, height
        )
        control_metrics, control_arrays = projection_metrics(
            control,
            stereo_depth_m,
            boundary_distance_px,
            frozen_boundary,
        )
        if source == int(selected_frames[0]["source_idx"]):
            write_json(
                args.output / "boundary_definition.json",
                {
                    "stereo_depth_boundary": (
                        "validity morphology gradient OR 3x3 Sobel depth "
                        "gradient magnitude >0.15 m"
                    ),
                    "frozen_lidar_boundary": (
                        "5x5 sparse projected LiDAR local depth range >0.25 m, "
                        "source-point identities frozen from control"
                    ),
                    "distance": "OpenCV L2 distance transform in pixels",
                },
            )
        for dose in doses:
            variant = projections[dose]
            metrics, arrays = projection_metrics(
                variant,
                stereo_depth_m,
                boundary_distance_px,
                frozen_boundary,
            )
            displacement = pixel_displacement(control, variant)
            eligible = bool(
                control_metrics["stereo_valid_correspondence_count"] >= 100
                and control_metrics["frozen_lidar_boundary_count"] >= 20
                and metrics["mean_frozen_boundary_distance_to_stereo_edge_px"]
                is not None
            )
            valid_ratio_drop = float(
                control_metrics["stereo_valid_correspondence_ratio"]
                - metrics["stereo_valid_correspondence_ratio"]
            )
            boundary_increase = float(
                metrics["mean_frozen_boundary_distance_to_stereo_edge_px"]
                - control_metrics[
                    "mean_frozen_boundary_distance_to_stereo_edge_px"
                ]
            )
            prescribed_alarm = bool(
                dose != 0
                and eligible
                and valid_ratio_drop >= 0.002
                and boundary_increase >= 0.10
            )
            displacement_alarm = bool(
                dose != 0
                and eligible
                and displacement["p90_projection_displacement_px"] is not None
                and displacement["p90_projection_displacement_px"] >= 0.25
            )
            name = "control" if dose == 0 else (
                f"{dose:+03d}ms".replace("+", "plus_").replace("-", "minus_")
            )
            cell = args.output / "cells" / name
            correspondence = cell / "correspondences" / f"{source:06d}.npz"
            correspondence.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                correspondence,
                lidar_point_indices=variant["source_indices"].astype(np.int64),
                u=variant["u"].astype(np.int32),
                v=variant["v"].astype(np.int32),
                lidar_depth_m=variant["depth"].astype(np.float32),
                stereo_depth_m=arrays["stereo_depth_m"],
                stereo_valid=arrays["stereo_valid"],
                frozen_boundary_source_indices=arrays[
                    "frozen_boundary_source_indices"
                ].astype(np.int64),
                frozen_boundary_variant_indices=arrays[
                    "frozen_boundary_variant_indices"
                ].astype(np.int64),
                frozen_boundary_distance_px=arrays[
                    "frozen_boundary_distance_px"
                ].astype(np.float32),
                common_control_source_indices=displacement[
                    "common_source_indices"
                ].astype(np.int64),
                control_common_indices=displacement[
                    "control_common_indices"
                ].astype(np.int64),
                variant_common_indices=displacement[
                    "variant_common_indices"
                ].astype(np.int64),
                du_px=displacement["du_px"].astype(np.float32),
                dv_px=displacement["dv_px"].astype(np.float32),
                displacement_px=displacement["displacement_px"].astype(
                    np.float32
                ),
                camera_T_lidar=transforms[dose],
            )
            overlay = cell / "overlays" / f"{source:06d}.jpg"
            render_overlay(
                overlay,
                rgb,
                control,
                variant,
                (
                    f"raw={source} selected={selected_index} dose={dose:+d} ms "
                    f"valid_drop={valid_ratio_drop:+.4f} "
                    f"boundary_delta={boundary_increase:+.2f}px"
                ),
            )
            row = {
                "source_index": source,
                "selected_frame_index": selected_index,
                "variant_id": name,
                "dose_ms": dose,
                "recorded_camera_time_ns": camera_time_ns,
                "used_camera_time_ns": camera_time_ns + dose * 1_000_000,
                "lidar_time_ns": lidar_time_ns,
                "recorded_camera_minus_lidar_ms": (
                    camera_time_ns - lidar_time_ns
                )
                / 1.0e6,
                "finite_lidar_points": int(len(lidar_points)),
                **metrics,
                "control_stereo_valid_correspondence_ratio": control_metrics[
                    "stereo_valid_correspondence_ratio"
                ],
                "stereo_valid_ratio_drop": valid_ratio_drop,
                "control_mean_frozen_boundary_distance_px": control_metrics[
                    "mean_frozen_boundary_distance_to_stereo_edge_px"
                ],
                "frozen_boundary_distance_increase_px": boundary_increase,
                "common_source_point_count": displacement[
                    "common_source_point_count"
                ],
                "mean_projection_displacement_px": displacement[
                    "mean_projection_displacement_px"
                ],
                "median_projection_displacement_px": displacement[
                    "median_projection_displacement_px"
                ],
                "p90_projection_displacement_px": displacement[
                    "p90_projection_displacement_px"
                ],
                "maximum_projection_displacement_px": displacement[
                    "maximum_projection_displacement_px"
                ],
                "eligible": eligible,
                "prescribed_alarm": prescribed_alarm,
                "secondary_displacement_alarm": displacement_alarm,
                "correspondence_artifact": str(correspondence),
                "overlay_artifact": str(overlay),
            }
            rows.append(row)

    write_csv(args.output / "per_frame.csv", rows)
    write_jsonl(args.output / "per_frame.jsonl", rows)
    summary = []
    for dose in doses:
        cell_rows = [
            row
            for row in rows
            if int(row["dose_ms"]) == dose and bool(row["eligible"])
        ]
        prescribed_rate = float(
            np.mean([bool(row["prescribed_alarm"]) for row in cell_rows])
        )
        displacement_rate = float(
            np.mean(
                [bool(row["secondary_displacement_alarm"]) for row in cell_rows]
            )
        )
        summary.append(
            {
                "variant_id": (
                    "control"
                    if dose == 0
                    else f"{dose:+03d}ms".replace("+", "plus_").replace("-", "minus_")
                ),
                "dose_ms": dose,
                "eligible_frame_count": len(cell_rows),
                "prescribed_detection_rate": prescribed_rate,
                "secondary_displacement_detection_rate": displacement_rate,
                "mean_valid_ratio_drop": float(
                    np.mean([float(row["stereo_valid_ratio_drop"]) for row in cell_rows])
                ),
                "mean_boundary_distance_increase_px": float(
                    np.mean(
                        [
                            float(row["frozen_boundary_distance_increase_px"])
                            for row in cell_rows
                        ]
                    )
                ),
                "mean_p90_projection_displacement_px": float(
                    np.mean(
                        [
                            float(row["p90_projection_displacement_px"])
                            for row in cell_rows
                        ]
                    )
                ),
                "medium_or_heavy": abs(dose) >= 40,
                "passed_if_eligible": (
                    dose == 0
                    or abs(dose) < 40
                    or prescribed_rate >= 0.90
                ),
            }
        )
    write_csv(args.output / "dose_summary.csv", summary)
    write_json(args.output / "dose_summary.json", summary)
    branches = {}
    for sign, sign_name in ((-1, "negative"), (1, "positive")):
        branch = sorted(
            [
                row
                for row in summary
                if int(row["dose_ms"]) * sign > 0
            ],
            key=lambda row: abs(int(row["dose_ms"])),
        )
        branches[sign_name] = {
            "absolute_doses_ms": [abs(int(row["dose_ms"])) for row in branch],
            "mean_projection_displacement_px": [
                float(row["mean_p90_projection_displacement_px"]) for row in branch
            ],
            "projection_displacement_monotone": bool(
                np.all(
                    np.diff(
                        [
                            float(row["mean_p90_projection_displacement_px"])
                            for row in branch
                        ]
                    )
                    >= -1.0e-12
                )
            ),
            "validity_drop_monotone": bool(
                np.all(
                    np.diff(
                        [float(row["mean_valid_ratio_drop"]) for row in branch]
                    )
                    >= -1.0e-12
                )
            ),
            "boundary_increase_monotone": bool(
                np.all(
                    np.diff(
                        [
                            float(row["mean_boundary_distance_increase_px"])
                            for row in branch
                        ]
                    )
                    >= -1.0e-12
                )
            ),
        }
    control_row = next(row for row in summary if int(row["dose_ms"]) == 0)
    medium_heavy = [row for row in summary if bool(row["medium_or_heavy"])]
    qualification = {
        "schema": f"{SCHEMA}.qualification",
        "created_at": utc_now(),
        "selected_frame_count": len(selected_frames),
        "selected_raw_source_range": [
            int(selected_frames[0]["source_idx"]),
            int(selected_frames[-1]["source_idx"]),
        ],
        "control_false_alarm_rate": control_row["prescribed_detection_rate"],
        "control_false_alarm_passed": (
            control_row["prescribed_detection_rate"] < 0.05
        ),
        "medium_heavy_detection_target": 0.90,
        "medium_heavy_all_passed": all(
            bool(row["passed_if_eligible"]) for row in medium_heavy
        ),
        "dose_branches": branches,
        "prescribed_response_both_branches_monotone": all(
            branch["validity_drop_monotone"]
            and branch["boundary_increase_monotone"]
            for branch in branches.values()
        ),
        "secondary_projection_displacement_both_branches_monotone": all(
            branch["projection_displacement_monotone"]
            for branch in branches.values()
        ),
        "family_passed": False,
        "ground_truth_status": (
            "sparse LiDAR proxy; no scan-internal deskew or reviewed occlusion boundary"
        ),
    }
    qualification["family_passed"] = bool(
        qualification["control_false_alarm_passed"]
        and qualification["medium_heavy_all_passed"]
        and qualification["prescribed_response_both_branches_monotone"]
    )
    write_json(args.output / "D0_LIDAR_TIME_QUALIFICATION.json", qualification)

    injected_summary = [row for row in summary if int(row["dose_ms"]) != 0]
    x = np.arange(len(injected_summary))
    labels = [row["variant_id"] for row in injected_summary]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.3), constrained_layout=True)
    axes[0].bar(
        x,
        [row["prescribed_detection_rate"] for row in injected_summary],
        color="#d95f02",
    )
    axes[0].axhline(0.90, color="black", linestyle="--")
    axes[0].set(title="Prescribed L4/E4 alarm", ylabel="frame detection rate", ylim=(0, 1.04))
    axes[1].bar(
        x,
        [row["mean_valid_ratio_drop"] for row in injected_summary],
        color="#1b9e77",
    )
    axes[1].axhline(0.002, color="black", linestyle="--")
    axes[1].set(title="Projection validity response", ylabel="mean valid-ratio drop")
    axes[2].bar(
        x,
        [
            row["mean_boundary_distance_increase_px"]
            for row in injected_summary
        ],
        color="#7570b3",
    )
    axes[2].axhline(0.10, color="black", linestyle="--")
    axes[2].set(title="Frozen boundary response", ylabel="mean increase (px)")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(args.output / "dose_response.png", dpi=180)
    plt.close(figure)

    report = [
        "# G1 camera–LiDAR time-offset D0",
        "",
        f"- Selected raw coverage: {qualification['selected_raw_source_range']}.",
        f"- Eligible selected frames: {summary[0]['eligible_frame_count']}.",
        (
            "- Medium/heavy prescribed detection passed: "
            f"`{qualification['medium_heavy_all_passed']}`."
        ),
        (
            "- Prescribed validity+boundary dose direction passed: "
            f"`{qualification['prescribed_response_both_branches_monotone']}`."
        ),
        (
            "- Secondary same-point displacement dose direction passed: "
            f"`{qualification['secondary_projection_displacement_both_branches_monotone']}`."
        ),
        f"- Family qualification: `{qualification['family_passed']}`.",
        "",
        "The strict result uses both responses required by the experiment protocol: "
        "projection-validity decline and boundary-error increase. The same-source "
        "pixel displacement is retained as a secondary observability diagnostic and "
        "does not override a strict failure.",
        "",
        "Sparse LiDAR has no per-point timestamps and the boundary reference is "
        "derived from frozen stereo depth rather than human adjudication; this result "
        "remains proxy/diagnostic.",
    ]
    (args.output / "REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    write_json(
        args.output / "COMPLETION.json",
        {
            "schema": f"{SCHEMA}.completion",
            "status": "complete",
            "completed_at": utc_now(),
            "family_passed": qualification["family_passed"],
            "evidence_inventory": (
                "See EVIDENCE_INVENTORY.json; it hashes this completion file and "
                "self-excludes its inventory files."
            ),
        },
    )
    evidence = inventory(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "qualification": qualification,
                "evidence_inventory": evidence,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            if "--output" in sys.argv:
                output = Path(
                    sys.argv[sys.argv.index("--output") + 1]
                ).expanduser().resolve()
                if output.exists():
                    write_json(
                        output / "terminal_failure.json",
                        {
                            "schema": f"{SCHEMA}.terminal_failure",
                            "failed_at": utc_now(),
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                            "argv": sys.argv,
                        },
                    )
        finally:
            raise
