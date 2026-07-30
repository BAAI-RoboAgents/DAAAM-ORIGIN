#!/usr/bin/env python3
"""Audit and prepare Unitree head-D455 RGB-D captures for semantic mapping.

The adapter is intentionally strict. It never substitutes a model-default camera
calibration, treats ``base_link`` and ``body`` as aliases, or assumes that camera
and odometry timestamps share a clock. Those facts must be supplied in explicit,
validated sidecars before the adapter writes a mapping-ready dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


EXPECTED_LAYOUT = "daaam_g1_hardware_v1"
EXPECTED_CAMERA = "cam0"
EXPECTED_SENSOR = "HEAD_D455"
CAMERA_CALIBRATION_SCHEMA = "unitree_head_rgbd_calibration_v1"
TIME_CONTRACT_SCHEMA = "unitree_rgbd_time_contract_v1"
PREPARED_MODALITY = "aligned_rgbd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Run a read-only source audit.")
    add_source_arguments(audit)
    audit.add_argument("--report", type=Path)

    prepare = subparsers.add_parser(
        "prepare",
        help="Build a mapping-ready aligned RGB-D dataset after all hard gates pass.",
    )
    add_source_arguments(prepare)
    prepare.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--sequence", default="000000")
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        help="Validated Unitree head-D455 calibration sidecar.",
    )
    parser.add_argument(
        "--time-contract",
        type=Path,
        help="Validated camera/odometry timebase and RGB-D capture sidecar.",
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        raise
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        records.append(value)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gate(
    code: str,
    status: str,
    message: str,
    *,
    hard: bool,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "status": status,
        "hard": hard,
        "blocks_preparation": bool(hard and status == "FAIL"),
        "message": message,
        "evidence": evidence or {},
    }


def apple_double_count(root: Path) -> int:
    return sum(
        1
        for path in root.rglob("._*")
        if path.is_file()
    )


def quaternion_transform(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("pose_xyz_quat_xyzw must contain seven finite values")
    quaternion = values[3:]
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, abs_tol=1.0e-5):
        raise ValueError(f"Quaternion is not normalized: norm={norm}")
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    output[:3, 3] = values[:3]
    return output


def validate_transform(values: Any, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size == 16:
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-5):
        raise ValueError(f"{name} rotation determinant is not one")
    return matrix


def validate_camera_calibration(
    path: Path,
    *,
    source_width: int | None,
    source_height: int | None,
) -> tuple[dict[str, Any], np.ndarray]:
    calibration = load_json(path)
    if calibration.get("schema") != CAMERA_CALIBRATION_SCHEMA:
        raise ValueError(
            f"camera calibration schema must be {CAMERA_CALIBRATION_SCHEMA!r}"
        )
    if calibration.get("sensor") != EXPECTED_SENSOR:
        raise ValueError(f"camera calibration sensor must be {EXPECTED_SENSOR!r}")
    if calibration.get("target_frame") != "body":
        raise ValueError(
            "camera calibration must provide body_T_camera; base_link/body aliases "
            "are not inferred"
        )
    if not str(calibration.get("source_frame", "")).strip():
        raise ValueError("camera calibration source_frame is required")
    body_T_camera = validate_transform(
        calibration.get("target_T_camera"),
        "body_T_camera",
    )

    intrinsics = calibration.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise ValueError("camera calibration intrinsics are required")
    if intrinsics.get("model") != "pinhole":
        raise ValueError("prepared Unitree RGB-D requires a pinhole camera model")
    required = ("fx", "fy", "cx", "cy", "width", "height")
    missing = [name for name in required if name not in intrinsics]
    if missing:
        raise ValueError(f"camera calibration intrinsics are missing {missing}")
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    if (
        width <= 0
        or height <= 0
        or fx <= 0.0
        or fy <= 0.0
        or not 0.0 <= cx < width
        or not 0.0 <= cy < height
    ):
        raise ValueError("camera calibration pinhole intrinsics are invalid")
    if source_width is not None and source_height is not None:
        if (width, height) != (source_width, source_height):
            raise ValueError(
                "camera calibration resolution does not match source RGB/depth: "
                f"{width}x{height} vs {source_width}x{source_height}"
            )
    distortion = intrinsics.get("distortion")
    if not isinstance(distortion, dict):
        raise ValueError(
            "camera calibration must explicitly describe the stored image distortion"
        )
    if distortion.get("model") != "none":
        raise ValueError(
            "2d_rect input must have distortion.model='none'; the adapter does not "
            "rectify or remap pixels"
        )
    coefficients = distortion.get("coefficients")
    if not isinstance(coefficients, list) or any(
        abs(float(value)) > 1.0e-12 for value in coefficients
    ):
        raise ValueError("rectified camera distortion coefficients must be explicit zeros")

    depth = calibration.get("depth")
    if not isinstance(depth, dict):
        raise ValueError("camera calibration depth contract is required")
    if depth.get("aligned_to_color") is not True:
        raise ValueError("head-D455 depth must be explicitly aligned to stored RGB")
    if depth.get("unit") != "meter":
        raise ValueError("source depth unit must be explicitly 'meter'")
    minimum_depth_m = float(depth.get("minimum_valid_depth_m", float("nan")))
    maximum_depth_m = float(depth.get("maximum_valid_depth_m", float("nan")))
    if (
        not math.isfinite(minimum_depth_m)
        or not math.isfinite(maximum_depth_m)
        or minimum_depth_m <= 0.0
        or maximum_depth_m <= minimum_depth_m
        or maximum_depth_m > 65.535
    ):
        raise ValueError(
            "depth minimum/maximum valid range must be explicit and fit uint16 mm"
        )
    invalid_values = depth.get("invalid_values_m")
    if not isinstance(invalid_values, list):
        raise ValueError("depth.invalid_values_m must be an explicit list")

    provenance = calibration.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("validated") is not True:
        raise ValueError("camera calibration provenance.validated must be true")
    for name in ("method", "source", "timestamp"):
        if not str(provenance.get(name, "")).strip():
            raise ValueError(f"camera calibration provenance.{name} is required")
    return calibration, body_T_camera


def validate_time_contract(path: Path) -> dict[str, Any]:
    contract = load_json(path)
    if contract.get("schema") != TIME_CONTRACT_SCHEMA:
        raise ValueError(f"time contract schema must be {TIME_CONTRACT_SCHEMA!r}")
    if contract.get("camera") != EXPECTED_CAMERA:
        raise ValueError(f"time contract camera must be {EXPECTED_CAMERA!r}")
    if contract.get("camera_time_source") not in {"sensor", "host"}:
        raise ValueError("camera_time_source must be explicit: 'sensor' or 'host'")
    if contract.get("shared_timebase_verified") is not True:
        raise ValueError("shared_timebase_verified must be true")
    if contract.get("rgb_depth_same_capture_verified") is not True:
        raise ValueError("rgb_depth_same_capture_verified must be true")
    if not str(contract.get("verification_method", "")).strip():
        raise ValueError("time contract verification_method is required")
    maximum_gap_ms = float(
        contract.get("maximum_pose_interpolation_gap_ms", float("nan"))
    )
    if not math.isfinite(maximum_gap_ms) or maximum_gap_ms <= 0.0:
        raise ValueError("maximum_pose_interpolation_gap_ms must be positive")
    allow_drop = contract.get("allow_drop_unbracketed_frames", False)
    if not isinstance(allow_drop, bool):
        raise ValueError("allow_drop_unbracketed_frames must be boolean")
    maximum_dropped = contract.get("maximum_dropped_frames", 0)
    if isinstance(maximum_dropped, bool) or int(maximum_dropped) < 0:
        raise ValueError("maximum_dropped_frames must be non-negative")
    contract["maximum_dropped_frames"] = int(maximum_dropped)
    return contract


def camera_timestamp(record: dict[str, Any], source: str) -> int:
    camera = record["images"][EXPECTED_CAMERA]
    field = "timestamp_ns" if source == "sensor" else "host_ns"
    value = int(camera[field])
    if value <= 0:
        raise ValueError(f"{EXPECTED_CAMERA} {field} must be a positive timestamp")
    return value


def source_payload_audit(
    src: Path,
    sequence: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    gates: list[dict[str, Any]] = []
    facts: dict[str, Any] = {
        "source": str(src),
        "sequence": sequence,
        "primary_camera": EXPECTED_CAMERA,
        "sensor": EXPECTED_SENSOR,
    }

    required_files = (
        src / "manifest.json",
        src / "manifest.jsonl",
        src / "quality_report.json",
        src / "calibrations" / sequence / "camera_info.json",
        src / "state" / sequence / "odom.jsonl",
        src / "poses" / "dense_global" / sequence / "aux_poses.jsonl",
        src / "poses" / "dense_global" / sequence / "poses.txt",
        src / "poses" / "dense_global" / sequence / "poses_7d.txt",
        src / "timestamps" / f"{sequence}.txt",
    )
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        gates.append(
            gate(
                "source.required_files",
                "FAIL",
                "Required Unitree capture metadata is missing",
                hard=True,
                evidence={"missing": missing_files},
            )
        )
        return facts, gates, []

    manifest = load_json(src / "manifest.json")
    records = load_jsonl(src / "manifest.jsonl")
    quality = load_json(src / "quality_report.json")
    calibration = load_json(src / "calibrations" / sequence / "camera_info.json")
    odometry = load_jsonl(src / "state" / sequence / "odom.jsonl")
    auxiliary = load_jsonl(
        src / "poses" / "dense_global" / sequence / "aux_poses.jsonl"
    )
    facts["layout"] = manifest.get("layout")
    if manifest.get("layout") == EXPECTED_LAYOUT:
        gates.append(
            gate(
                "source.layout",
                "PASS",
                "Unitree capture layout is recognized",
                hard=True,
                evidence={"layout": manifest.get("layout")},
            )
        )
    else:
        gates.append(
            gate(
                "source.layout",
                "FAIL",
                "Unexpected capture layout",
                hard=True,
                evidence={
                    "expected": EXPECTED_LAYOUT,
                    "actual": manifest.get("layout"),
                },
            )
        )

    counts = quality.get("counts", {})
    expected_count = len(records)
    count_fields = (
        "frames",
        "timestamps",
        "poses",
        "aux_poses",
        "manifest_records",
        "cam0_images",
        "cam0_depth",
        "lidar",
        "odom",
    )
    reported_counts = {name: counts.get(name) for name in count_fields}
    count_ok = (
        expected_count > 0
        and quality.get("alignment_ok") is True
        and counts.get("ok") is True
        and all(int(counts.get(name, -1)) == expected_count for name in count_fields)
        and len(odometry) == expected_count
        and len(auxiliary) == expected_count
    )
    gates.append(
        gate(
            "source.record_counts",
            "PASS" if count_ok else "FAIL",
            "Accepted records and required stream counts agree"
            if count_ok
            else "Accepted records or required stream counts disagree",
            hard=True,
            evidence={
                "manifest_records": expected_count,
                "reported": reported_counts,
                "odometry_records": len(odometry),
                "auxiliary_pose_records": len(auxiliary),
                "quality_alignment_ok": quality.get("alignment_ok"),
                "quality_counts_ok": counts.get("ok"),
            },
        )
    )

    ticks = [record.get("tick") for record in records]
    sequential = ticks == list(range(expected_count))
    gates.append(
        gate(
            "source.tick_sequence",
            "PASS" if sequential else "FAIL",
            "Frame ticks are contiguous and zero-based"
            if sequential
            else "Frame ticks are not contiguous and zero-based",
            hard=True,
            evidence={
                "first": ticks[0] if ticks else None,
                "last": ticks[-1] if ticks else None,
                "count": len(ticks),
            },
        )
    )

    rgb_shapes: Counter[str] = Counter()
    depth_shapes: Counter[str] = Counter()
    depth_dtypes: Counter[str] = Counter()
    valid_ratios: list[float] = []
    saturated_pixels = 0
    payload_errors: list[str] = []
    rgb_hashes: set[str] = set()
    depth_hashes: set[str] = set()
    for index, record in enumerate(records):
        try:
            rgb_path = src / record["images"][EXPECTED_CAMERA]["path"]
            depth_path = src / record["depth"][EXPECTED_CAMERA]["path"]
            if rgb_path.name.startswith("._") or depth_path.name.startswith("._"):
                raise ValueError("AppleDouble metadata cannot be a payload path")
            if not rgb_path.is_file() or not depth_path.is_file():
                raise FileNotFoundError("RGB or depth payload is missing")
            expected_stem = f"{index:06d}"
            if rgb_path.stem != expected_stem or depth_path.stem != expected_stem:
                raise ValueError("payload filename does not match tick")
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
            depth = np.load(depth_path, allow_pickle=False)
            if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("RGB payload is not a decodable three-channel image")
            if rgb.dtype != np.uint8:
                raise ValueError(f"RGB dtype is {rgb.dtype}, expected uint8")
            if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
                raise ValueError("depth is not pixel-aligned by dimensions")
            if depth.dtype != np.float32:
                raise ValueError(f"depth dtype is {depth.dtype}, expected float32")
            if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
                raise ValueError("depth contains non-finite or negative values")
            valid = depth > 0.0
            valid_ratio = float(np.mean(valid))
            recorded_ratio = float(record["depth"][EXPECTED_CAMERA]["valid_ratio"])
            if not math.isclose(valid_ratio, recorded_ratio, abs_tol=1.0e-12):
                raise ValueError("manifest depth valid ratio disagrees with payload")
            rgb_shapes[str(tuple(rgb.shape))] += 1
            depth_shapes[str(tuple(depth.shape))] += 1
            depth_dtypes[str(depth.dtype)] += 1
            valid_ratios.append(valid_ratio)
            saturated_pixels += int(np.count_nonzero(depth >= 65.535 - 1.0e-4))
            rgb_hashes.add(sha256_file(rgb_path))
            depth_hashes.add(sha256_file(depth_path))
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            payload_errors.append(f"tick {index}: {error}")

    payload_ok = not payload_errors and len(valid_ratios) == expected_count
    facts["rgb"] = {
        "payload_count": len(valid_ratios),
        "shapes": dict(rgb_shapes),
        "dtype": "uint8" if payload_ok else None,
        "unique_payloads": len(rgb_hashes),
    }
    facts["depth"] = {
        "payload_count": len(valid_ratios),
        "shapes": dict(depth_shapes),
        "dtypes": dict(depth_dtypes),
        "valid_ratio_minimum": min(valid_ratios) if valid_ratios else None,
        "valid_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else None,
        "valid_ratio_maximum": max(valid_ratios) if valid_ratios else None,
        "pixels_at_or_above_65_535_m": saturated_pixels,
        "unique_payloads": len(depth_hashes),
    }
    gates.append(
        gate(
            "source.head_rgbd_payload",
            "PASS" if payload_ok else "FAIL",
            "All head-camera RGB and depth payloads are readable and dimension-aligned"
            if payload_ok
            else "Head-camera RGB/depth payload validation failed",
            hard=True,
            evidence={
                **facts["rgb"],
                "depth": facts["depth"],
                "errors": payload_errors[:20],
            },
        )
    )

    secondary_errors: list[str] = []
    secondary_rgb_shapes: Counter[str] = Counter()
    secondary_depth_shapes: Counter[str] = Counter()
    secondary_valid_ratios: list[float] = []
    for index, record in enumerate(records):
        try:
            rgb_path = src / record["images"]["cam1"]["path"]
            depth_path = src / record["depth"]["cam1"]["path"]
            if not rgb_path.is_file() or not depth_path.is_file():
                raise FileNotFoundError("secondary RGB or depth payload is missing")
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
            depth = np.load(depth_path, allow_pickle=False)
            if (
                rgb is None
                or rgb.ndim != 3
                or rgb.shape[2] != 3
                or rgb.dtype != np.uint8
            ):
                raise ValueError("secondary RGB shape/dtype is invalid")
            if (
                depth.shape != rgb.shape[:2]
                or depth.dtype != np.float32
                or not np.all(np.isfinite(depth))
                or np.any(depth < 0.0)
            ):
                raise ValueError("secondary depth shape/dtype/values are invalid")
            valid_ratio = float(np.mean(depth > 0.0))
            if not math.isclose(
                valid_ratio,
                float(record["depth"]["cam1"]["valid_ratio"]),
                abs_tol=1.0e-12,
            ):
                raise ValueError("secondary depth valid ratio disagrees with manifest")
            secondary_rgb_shapes[str(tuple(rgb.shape))] += 1
            secondary_depth_shapes[str(tuple(depth.shape))] += 1
            secondary_valid_ratios.append(valid_ratio)
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            secondary_errors.append(f"tick {index}: {error}")
    secondary_ok = (
        not secondary_errors and len(secondary_valid_ratios) == expected_count
    )
    facts["secondary_camera"] = {
        "camera": "cam1",
        "sensor": manifest.get("cameras", {}).get("cam1"),
        "used_for_mapping": False,
        "rgb_payload_count": len(secondary_valid_ratios),
        "rgb_shapes": dict(secondary_rgb_shapes),
        "depth_shapes": dict(secondary_depth_shapes),
        "depth_valid_ratio_minimum": (
            min(secondary_valid_ratios) if secondary_valid_ratios else None
        ),
        "depth_valid_ratio_mean": (
            float(np.mean(secondary_valid_ratios))
            if secondary_valid_ratios
            else None
        ),
        "depth_valid_ratio_maximum": (
            max(secondary_valid_ratios) if secondary_valid_ratios else None
        ),
        "errors": secondary_errors[:20],
    }
    gates.append(
        gate(
            "source.secondary_rgbd_payload",
            "PASS" if secondary_ok else "WARN",
            "All unused chest-camera RGB-D payloads are readable"
            if secondary_ok
            else "Unused chest-camera RGB-D payload validation found issues",
            hard=False,
            evidence=facts["secondary_camera"],
        )
    )

    lidar_errors: list[str] = []
    lidar_counts: list[int] = []
    lidar_dtypes: Counter[str] = Counter()
    for index, record in enumerate(records):
        try:
            lidar_path = src / record["lidar_path"]
            if not lidar_path.is_file() or lidar_path.stem != f"{index:06d}":
                raise FileNotFoundError("LiDAR payload is missing or misnamed")
            points = np.load(lidar_path, allow_pickle=False)
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or points.dtype != np.float32
                or not np.all(np.isfinite(points))
            ):
                raise ValueError("LiDAR payload must be finite float32 Nx3")
            if int(record.get("lidar_points", -1)) != len(points):
                raise ValueError("LiDAR point count disagrees with manifest")
            lidar_counts.append(len(points))
            lidar_dtypes[str(points.dtype)] += 1
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            lidar_errors.append(f"tick {index}: {error}")
    lidar_ok = not lidar_errors and len(lidar_counts) == expected_count
    facts["lidar"] = {
        "payload_count": len(lidar_counts),
        "dtypes": dict(lidar_dtypes),
        "point_count_minimum": min(lidar_counts) if lidar_counts else None,
        "point_count_median": (
            float(np.median(lidar_counts)) if lidar_counts else None
        ),
        "point_count_maximum": max(lidar_counts) if lidar_counts else None,
        "errors": lidar_errors[:20],
    }
    gates.append(
        gate(
            "source.lidar_payload",
            "PASS" if lidar_ok else "FAIL",
            "All LiDAR payloads are finite float32 Nx3 and match manifest counts"
            if lidar_ok
            else "LiDAR payload validation failed",
            hard=True,
            evidence=facts["lidar"],
        )
    )

    try:
        timestamp_values = [
            int(line)
            for line in (
                src / "timestamps" / f"{sequence}.txt"
            ).read_text().splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as error:
        timestamp_values = []
        timestamp_error = str(error)
    else:
        timestamp_error = None
    anchor_times = [int(record["anchor_timestamp_ns"]) for record in records]
    timestamps_ok = timestamp_values == anchor_times
    facts["timestamp_index"] = {
        "count": len(timestamp_values),
        "matches_manifest_anchor_times": timestamps_ok,
        "error": timestamp_error,
    }
    gates.append(
        gate(
            "source.timestamp_index",
            "PASS" if timestamps_ok else "FAIL",
            "timestamps.txt exactly matches manifest anchor times"
            if timestamps_ok
            else "timestamps.txt does not exactly match manifest anchor times",
            hard=True,
            evidence=facts["timestamp_index"],
        )
    )

    try:
        stored_poses = np.loadtxt(
            src / "poses" / "dense_global" / sequence / "poses.txt"
        ).reshape(-1, 4, 4)
        stored_poses_7d = np.loadtxt(
            src / "poses" / "dense_global" / sequence / "poses_7d.txt"
        ).reshape(-1, 7)
        auxiliary_pose_ok = (
            len(stored_poses) == expected_count
            and len(stored_poses_7d) == expected_count
        )
        unique_matrices = len(np.unique(np.round(stored_poses, 12), axis=0))
        unique_poses_7d = len(np.unique(np.round(stored_poses_7d, 12), axis=0))
        auxiliary_pose_error = None
    except (OSError, ValueError) as error:
        auxiliary_pose_ok = False
        unique_matrices = None
        unique_poses_7d = None
        auxiliary_pose_error = str(error)
    facts["stored_camera_pose_files"] = {
        "count": len(stored_poses) if auxiliary_pose_ok else None,
        "unique_matrices": unique_matrices,
        "unique_poses_7d": unique_poses_7d,
        "declared_meaning": "static chest-D435i-in-body extrinsic",
        "usable_as_head_camera_world_trajectory": False,
        "error": auxiliary_pose_error,
    }
    gates.append(
        gate(
            "source.stored_camera_pose_files",
            "PASS" if auxiliary_pose_ok else "WARN",
            "Stored pose files are complete static chest-camera extrinsics"
            if auxiliary_pose_ok
            else "Stored static chest-camera pose files are invalid",
            hard=False,
            evidence=facts["stored_camera_pose_files"],
        )
    )

    raw_camera = calibration.get(EXPECTED_CAMERA, {})
    notes = calibration.get("notes", {})
    declared_aligned = notes.get("cam0_depth_aligned_to_rgb") is True
    gates.append(
        gate(
            "source.depth_alignment_declaration",
            "PASS" if declared_aligned else "FAIL",
            "Capture metadata declares head depth aligned to RGB"
            if declared_aligned
            else "Capture metadata does not declare head depth aligned to RGB",
            hard=True,
            evidence={
                "declared_aligned": notes.get("cam0_depth_aligned_to_rgb"),
                "independent_registration_validation_present": False,
            },
        )
    )
    embedded_intrinsics = raw_camera.get("intrinsics", {})
    missing_embedded = [
        name
        for name in (
            "distortion_model",
            "distortion_coefficients",
            "extrinsics_to_base_link",
            "serial_number",
            "calibration_timestamp",
            "rgb_depth_registration_transform",
        )
        if name not in raw_camera
    ]
    facts["embedded_head_camera_metadata"] = {
        "intrinsics": embedded_intrinsics,
        "missing_fields": missing_embedded,
    }

    pose_errors: list[str] = []
    odom_times: list[int] = []
    odom_transforms: list[np.ndarray] = []
    for index, (record, aux, odom) in enumerate(
        zip(records, auxiliary, odometry, strict=False)
    ):
        try:
            if (
                int(record["tick"]) != index
                or int(aux["tick"]) != index
                or int(odom["tick"]) != index
            ):
                raise ValueError("tick binding mismatch")
            anchor = int(record["anchor_timestamp_ns"])
            odom_time = int(odom["header"]["timestamp_ns"])
            if anchor != odom_time or anchor != int(record["odom"]["timestamp_ns"]):
                raise ValueError("anchor and odometry timestamps disagree")
            if odom["header"]["frame_id"] != "local" or odom["child_frame_id"] != "body":
                raise ValueError("odometry direction is not local_T_body")
            map_entry = aux["poses"]["map"]
            if (
                map_entry["target_frame"] != "local"
                or map_entry["source_frame"] != "body"
            ):
                raise ValueError("auxiliary trajectory direction is not local_T_body")
            if aux["poses"][EXPECTED_CAMERA].get("pose_xyz_quat_xyzw") is not None:
                raise ValueError("unexpected non-null cam0 pose")
            odom_times.append(odom_time)
            odom_transforms.append(
                quaternion_transform(odom["pose_xyz_quat_xyzw"])
            )
        except (KeyError, TypeError, ValueError) as error:
            pose_errors.append(f"tick {index}: {error}")
    odom_monotonic = bool(
        odom_times
        and all(b > a for a, b in zip(odom_times, odom_times[1:]))
    )
    pose_stream_ok = (
        not pose_errors
        and len(odom_times) == expected_count
        and odom_monotonic
    )
    facts["pose_stream"] = {
        "available_transform": "local_T_body",
        "map_T_body_available": False,
        "head_camera_pose_null_count": sum(
            int(
                aux.get("poses", {})
                .get(EXPECTED_CAMERA, {})
                .get("pose_xyz_quat_xyzw")
                is None
            )
            for aux in auxiliary
        ),
        "timestamps_monotonic": odom_monotonic,
    }
    gates.append(
        gate(
            "source.local_body_trajectory",
            "PASS" if pose_stream_ok else "FAIL",
            "A valid local_T_body trajectory is present"
            if pose_stream_ok
            else "The local_T_body trajectory is incomplete or invalid",
            hard=True,
            evidence={**facts["pose_stream"], "errors": pose_errors[:20]},
        )
    )

    camera_sensor_times: list[int] = []
    camera_host_times: list[int] = []
    time_errors: list[str] = []
    for index, record in enumerate(records):
        try:
            camera_sensor_times.append(camera_timestamp(record, "sensor"))
            camera_host_times.append(camera_timestamp(record, "host"))
        except (KeyError, TypeError, ValueError) as error:
            time_errors.append(f"tick {index}: {error}")
    time_monotonic = {
        "camera_sensor": bool(
            camera_sensor_times
            and all(
                b > a
                for a, b in zip(camera_sensor_times, camera_sensor_times[1:])
            )
        ),
        "camera_host": bool(
            camera_host_times
            and all(b > a for a, b in zip(camera_host_times, camera_host_times[1:]))
        ),
        "odometry": odom_monotonic,
    }
    facts["time_streams"] = {
        "monotonic": time_monotonic,
        "camera_sensor_minus_pose_ms": summarize_deltas(
            camera_sensor_times, odom_times
        ),
        "camera_host_minus_pose_ms": summarize_deltas(
            camera_host_times, odom_times
        ),
        "per_depth_timestamp_present": all(
            "timestamp_ns" in record.get("depth", {}).get(EXPECTED_CAMERA, {})
            for record in records
        ),
        "errors": time_errors,
    }
    gates.append(
        gate(
            "source.timestamp_streams",
            (
                "PASS"
                if not time_errors and all(time_monotonic.values())
                else "FAIL"
            ),
            "Stored camera and odometry timestamps are strictly increasing"
            if not time_errors and all(time_monotonic.values())
            else "Stored camera or odometry timestamps are invalid",
            hard=True,
            evidence=facts["time_streams"],
        )
    )

    artifacts = apple_double_count(src)
    facts["appledouble_metadata_files"] = artifacts
    if artifacts:
        gates.append(
            gate(
                "source.appledouble_metadata",
                "WARN",
                "AppleDouble files are present and will be ignored",
                hard=False,
                evidence={"count": artifacts},
            )
        )

    skipped = int(quality.get("extra", {}).get("skipped", 0))
    attempts = int(quality.get("extra", {}).get("attempts", expected_count + skipped))
    facts["capture_rate"] = {
        "target_hz": quality.get("target_hz"),
        "actual_hz_estimate": quality.get("actual_hz_estimate"),
        "attempts": attempts,
        "accepted_frames": expected_count,
        "skipped_attempts": skipped,
        "skipped_fraction": skipped / max(1, attempts),
        "skip_reasons": quality.get("extra", {}).get("skip_reasons", {}),
    }
    if skipped:
        gates.append(
            gate(
                "source.capture_rate",
                "WARN",
                "The accepted dataset is internally complete, but acquisition missed "
                "its target rate",
                hard=False,
                evidence=facts["capture_rate"],
            )
        )

    facts["_records"] = records
    facts["_odom_times"] = odom_times
    facts["_odom_transforms"] = odom_transforms
    return facts, gates, auxiliary


def summarize_deltas(first: list[int], second: list[int]) -> dict[str, float] | None:
    if not first or len(first) != len(second):
        return None
    deltas = (np.asarray(first, dtype=np.int64) - np.asarray(second, dtype=np.int64))
    deltas_ms = deltas.astype(np.float64) / 1.0e6
    return {
        "minimum": float(np.min(deltas_ms)),
        "median": float(np.median(deltas_ms)),
        "p95": float(np.percentile(deltas_ms, 95)),
        "maximum": float(np.max(deltas_ms)),
        "maximum_absolute": float(np.max(np.abs(deltas_ms))),
    }


def pose_coverage(
    camera_times: list[int],
    pose_times: list[int],
    *,
    maximum_gap_ms: float,
) -> dict[str, Any]:
    pose_array = np.asarray(pose_times, dtype=np.int64)
    outside: list[int] = []
    excessive_gap: list[int] = []
    bracket_gaps_ms: list[float] = []
    for index, timestamp in enumerate(camera_times):
        position = int(np.searchsorted(pose_array, timestamp, side="left"))
        if position < len(pose_array) and int(pose_array[position]) == timestamp:
            bracket_gaps_ms.append(0.0)
            continue
        if position == 0 or position == len(pose_array):
            outside.append(index)
            continue
        gap_ms = float(pose_array[position] - pose_array[position - 1]) / 1.0e6
        bracket_gaps_ms.append(gap_ms)
        if gap_ms > maximum_gap_ms:
            excessive_gap.append(index)
    return {
        "outside_pose_range": outside,
        "excessive_interpolation_gap": excessive_gap,
        "maximum_bracket_gap_ms": max(bracket_gaps_ms, default=0.0),
        "configured_maximum_gap_ms": maximum_gap_ms,
    }


def audit_dataset(
    src: Path,
    sequence: str,
    *,
    camera_calibration_path: Path | None = None,
    time_contract_path: Path | None = None,
) -> dict[str, Any]:
    src = src.resolve()
    facts, gates, _auxiliary = source_payload_audit(src, sequence)
    records = facts.pop("_records", [])
    odom_times = facts.pop("_odom_times", [])
    facts.pop("_odom_transforms", None)

    source_shape = None
    shapes = facts.get("depth", {}).get("shapes", {})
    if len(shapes) == 1:
        text = next(iter(shapes))
        try:
            height, width = (
                int(value.strip()) for value in text.strip("()").split(",")
            )
            source_shape = (width, height)
        except (TypeError, ValueError):
            source_shape = None

    camera_calibration = None
    if camera_calibration_path is None:
        gates.append(
            gate(
                "contract.head_camera_calibration",
                "FAIL",
                "Validated head-D455 calibration sidecar is missing",
                hard=True,
                evidence={
                    "required_schema": CAMERA_CALIBRATION_SCHEMA,
                    "explicitly_missing": [
                        "body_T_head_D455_color_optical_frame",
                        "rectified pinhole/distortion contract",
                        "depth valid/invalid value policy",
                        "calibration identity and validation provenance",
                    ],
                },
            )
        )
    else:
        try:
            camera_calibration, _body_T_camera = validate_camera_calibration(
                camera_calibration_path.resolve(),
                source_width=source_shape[0] if source_shape else None,
                source_height=source_shape[1] if source_shape else None,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            gates.append(
                gate(
                    "contract.head_camera_calibration",
                    "FAIL",
                    f"Head-D455 calibration sidecar is invalid: {error}",
                    hard=True,
                    evidence={"path": str(camera_calibration_path)},
                )
            )
        else:
            gates.append(
                gate(
                    "contract.head_camera_calibration",
                    "PASS",
                    "Validated body_T_head-camera and RGB-D calibration are present",
                    hard=True,
                    evidence={
                        "path": str(camera_calibration_path.resolve()),
                        "sha256": sha256_file(camera_calibration_path.resolve()),
                        "source_frame": camera_calibration["source_frame"],
                    },
                )
            )

    time_contract = None
    if time_contract_path is None:
        gates.append(
            gate(
                "contract.rgbd_time",
                "FAIL",
                "Validated camera/odometry time contract sidecar is missing",
                hard=True,
                evidence={
                    "required_schema": TIME_CONTRACT_SCHEMA,
                    "explicitly_unproven": [
                        "camera and odometry timestamps share a timebase",
                        "RGB and depth belong to the same capture time",
                        "out-of-range pose handling policy",
                    ],
                },
            )
        )
    else:
        try:
            time_contract = validate_time_contract(time_contract_path.resolve())
            camera_times = [
                camera_timestamp(record, time_contract["camera_time_source"])
                for record in records
            ]
            coverage = pose_coverage(
                camera_times,
                odom_times,
                maximum_gap_ms=float(
                    time_contract["maximum_pose_interpolation_gap_ms"]
                ),
            )
            invalid_indices = sorted(
                set(coverage["outside_pose_range"])
                | set(coverage["excessive_interpolation_gap"])
            )
            allowed = (
                not invalid_indices
                or (
                    time_contract["allow_drop_unbracketed_frames"]
                    and len(invalid_indices)
                    <= time_contract["maximum_dropped_frames"]
                )
            )
            if not allowed:
                raise ValueError(
                    f"{len(invalid_indices)} frames lack an allowed pose bracket"
                )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
            gates.append(
                gate(
                    "contract.rgbd_time",
                    "FAIL",
                    f"RGB-D time contract is invalid: {error}",
                    hard=True,
                    evidence={"path": str(time_contract_path)},
                )
            )
        else:
            gates.append(
                gate(
                    "contract.rgbd_time",
                    "PASS",
                    "Validated RGB-D/odometry time contract and pose coverage are present",
                    hard=True,
                    evidence={
                        "path": str(time_contract_path.resolve()),
                        "sha256": sha256_file(time_contract_path.resolve()),
                        "camera_time_source": time_contract["camera_time_source"],
                        "pose_coverage": coverage,
                        "dropped_by_explicit_policy": invalid_indices,
                    },
                )
            )

    blockers = [item for item in gates if item["blocks_preparation"]]
    warnings = [item for item in gates if item["status"] == "WARN"]
    return {
        "schema": "unitree_head_rgbd_audit_v1",
        "status": "ready" if not blockers else "blocked",
        "mapping_ready": not blockers,
        "facts": facts,
        "gates": gates,
        "summary": {
            "passed": sum(item["status"] == "PASS" for item in gates),
            "warnings": len(warnings),
            "failed": sum(item["status"] == "FAIL" for item in gates),
            "hard_blockers": [
                {
                    "code": item["code"],
                    "message": item["message"],
                    "evidence": item["evidence"],
                }
                for item in blockers
            ],
        },
    }


def interpolate_transforms(
    pose_times: np.ndarray,
    poses: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    output = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(target_times), axis=0)
    positions = np.searchsorted(pose_times, target_times, side="left")
    for index, (timestamp, position) in enumerate(zip(target_times, positions)):
        if position < len(pose_times) and pose_times[position] == timestamp:
            output[index] = poses[position]
            continue
        if position == 0 or position == len(pose_times):
            raise ValueError(f"Target timestamp {timestamp} is outside pose coverage")
        first = position - 1
        second = position
        fraction = float(timestamp - pose_times[first]) / float(
            pose_times[second] - pose_times[first]
        )
        output[index, :3, 3] = (
            poses[first, :3, 3] * (1.0 - fraction)
            + poses[second, :3, 3] * fraction
        )
        rotations = Rotation.from_matrix(poses[[first, second], :3, :3])
        output[index, :3, :3] = Slerp(
            [float(pose_times[first]), float(pose_times[second])],
            rotations,
        )([float(timestamp)]).as_matrix()[0]
    return output


def ensure_empty_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)


def write_pose_file(path: Path, poses: np.ndarray) -> None:
    path.write_text(
        "".join(
            " ".join(f"{value:.12g}" for value in pose.reshape(-1)) + "\n"
            for pose in poses
        )
    )


def prepare_dataset(
    src: Path,
    output: Path,
    sequence: str,
    *,
    camera_calibration_path: Path,
    time_contract_path: Path,
) -> dict[str, Any]:
    src = src.resolve()
    output = output.resolve()
    camera_calibration_path = camera_calibration_path.resolve()
    time_contract_path = time_contract_path.resolve()
    audit = audit_dataset(
        src,
        sequence,
        camera_calibration_path=camera_calibration_path,
        time_contract_path=time_contract_path,
    )
    if not audit["mapping_ready"]:
        messages = "; ".join(
            item["message"] for item in audit["summary"]["hard_blockers"]
        )
        raise RuntimeError(f"Unitree source is not mapping-ready: {messages}")

    source_width = int(
        audit["facts"]["embedded_head_camera_metadata"]["intrinsics"]["width"]
    )
    source_height = int(
        audit["facts"]["embedded_head_camera_metadata"]["intrinsics"]["height"]
    )
    calibration, body_T_camera = validate_camera_calibration(
        camera_calibration_path,
        source_width=source_width,
        source_height=source_height,
    )
    contract = validate_time_contract(time_contract_path)
    records = load_jsonl(src / "manifest.jsonl")
    odometry = load_jsonl(src / "state" / sequence / "odom.jsonl")
    pose_times = np.asarray(
        [int(record["header"]["timestamp_ns"]) for record in odometry],
        dtype=np.int64,
    )
    local_T_body = np.asarray(
        [
            quaternion_transform(record["pose_xyz_quat_xyzw"])
            for record in odometry
        ],
        dtype=np.float64,
    )
    source_camera_times = [
        camera_timestamp(record, contract["camera_time_source"])
        for record in records
    ]
    coverage = pose_coverage(
        source_camera_times,
        pose_times.tolist(),
        maximum_gap_ms=float(contract["maximum_pose_interpolation_gap_ms"]),
    )
    dropped = sorted(
        set(coverage["outside_pose_range"])
        | set(coverage["excessive_interpolation_gap"])
    )
    retained_indices = [
        index for index in range(len(records)) if index not in set(dropped)
    ]
    if dropped and not contract["allow_drop_unbracketed_frames"]:
        raise RuntimeError("Time contract does not allow dropping unbracketed frames")
    if len(dropped) > contract["maximum_dropped_frames"]:
        raise RuntimeError("Unbracketed frame count exceeds the explicit time contract")
    retained_times = np.asarray(
        [source_camera_times[index] for index in retained_indices],
        dtype=np.int64,
    )
    interpolated_local_T_body = interpolate_transforms(
        pose_times,
        local_T_body,
        retained_times,
    )
    local_T_camera = interpolated_local_T_body @ body_T_camera

    ensure_empty_output(output)
    for name in ("rgb", "depth", "depth_confidence", "depth_metadata", "pose"):
        (output / name).mkdir()

    intrinsics = calibration["intrinsics"]
    depth_contract = calibration["depth"]
    minimum_depth_m = float(depth_contract["minimum_valid_depth_m"])
    maximum_depth_m = float(depth_contract["maximum_valid_depth_m"])
    invalid_values_m = [float(value) for value in depth_contract["invalid_values_m"]]
    output_frames: list[dict[str, Any]] = []
    valid_ratios: list[float] = []
    source_rgb_hashes: list[str] = []
    output_rgb_hashes: list[str] = []
    invalid_policy_counts: Counter[str] = Counter()
    origin_ns = int(retained_times[0])

    for output_index, (source_index, timestamp_ns) in enumerate(
        zip(retained_indices, retained_times)
    ):
        record = records[source_index]
        source_rgb = src / record["images"][EXPECTED_CAMERA]["path"]
        source_depth = src / record["depth"][EXPECTED_CAMERA]["path"]
        output_rgb = output / "rgb" / f"{output_index:08d}.png"
        output_depth = output / "depth" / f"{output_index:08d}.png"
        output_confidence = output / "depth_confidence" / f"{output_index:08d}.png"
        shutil.copyfile(source_rgb, output_rgb)
        source_hash = sha256_file(source_rgb)
        output_hash = sha256_file(output_rgb)
        if source_hash != output_hash:
            raise RuntimeError(f"RGB byte-copy verification failed: {source_rgb}")
        source_rgb_hashes.append(source_hash)
        output_rgb_hashes.append(output_hash)

        depth_m = np.load(source_depth, allow_pickle=False)
        valid = np.isfinite(depth_m)
        invalid_policy_counts["non_finite"] += int(np.count_nonzero(~valid))
        non_negative = depth_m >= minimum_depth_m
        invalid_policy_counts["below_minimum"] += int(
            np.count_nonzero(valid & ~non_negative)
        )
        valid &= non_negative
        within_maximum = depth_m <= maximum_depth_m
        invalid_policy_counts["above_maximum"] += int(
            np.count_nonzero(valid & ~within_maximum)
        )
        valid &= within_maximum
        for value in invalid_values_m:
            matches = np.isclose(depth_m, value, rtol=0.0, atol=5.0e-7)
            invalid_policy_counts[f"explicit_value_{value:g}"] += int(
                np.count_nonzero(valid & matches)
            )
            valid &= ~matches
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        depth_mm[valid] = np.rint(depth_m[valid] * 1000.0).astype(np.uint16)
        confidence = np.where(valid, 255, 0).astype(np.uint8)
        if not cv2.imwrite(str(output_depth), depth_mm):
            raise RuntimeError(f"Failed to write depth: {output_depth}")
        if not cv2.imwrite(str(output_confidence), confidence):
            raise RuntimeError(f"Failed to write confidence: {output_confidence}")
        valid_ratio = float(np.mean(valid))
        valid_ratios.append(valid_ratio)
        metadata = {
            "frame_index": output_index,
            "source_tick": source_index,
            "sensor_time_ns": int(timestamp_ns),
            "source": "unitree_head_d455_aligned_rgbd",
            "confidence_mode": "sensor-validity",
            "left_right_verified": False,
            "valid_ratio": valid_ratio,
            "minimum_valid_depth_m": minimum_depth_m,
            "maximum_valid_depth_m": maximum_depth_m,
        }
        (
            output
            / "depth_metadata"
            / f"{output_index:08d}.json"
        ).write_text(json.dumps(metadata, indent=2) + "\n")
        output_frames.append(
            {
                "idx": output_index,
                "source_idx": source_index,
                "source_tick": source_index,
                "pose_row": output_index,
                "cam0": str(output_rgb),
                "depth": str(output_depth),
                "timestamp": (int(timestamp_ns) - origin_ns) / 1.0e9,
                "sensor_time_ns": int(timestamp_ns),
                "cam0_sensor_time_ns": int(timestamp_ns),
                "depth_sensor_time_ns": int(timestamp_ns),
                "pose_sensor_time_ns": int(timestamp_ns),
                "selection_reason": (
                    "initial_frame" if output_index == 0 else "routine"
                ),
                "source_camera_sensor_time_ns": int(
                    record["images"][EXPECTED_CAMERA]["timestamp_ns"]
                ),
                "source_camera_host_time_ns": int(
                    record["images"][EXPECTED_CAMERA]["host_ns"]
                ),
                "source_pose_anchor_time_ns": int(record["anchor_timestamp_ns"]),
            }
        )
    if output_frames:
        output_frames[-1]["selection_reason"] = "final_frame"

    write_pose_file(output / "pose" / "poses.txt", local_T_camera)
    (output / "pose" / "pose_timestamps_ns.txt").write_text(
        "".join(f"{int(value)}\n" for value in retained_times)
    )
    camera_matrix = [
        [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
        [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
        [0.0, 0.0, 1.0],
    ]
    camera_info = {
        "width": int(intrinsics["width"]),
        "height": int(intrinsics["height"]),
        "model": "pinhole",
        "intrinsics": camera_matrix,
        "distortion": [float(value) for value in intrinsics["distortion"]["coefficients"]],
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "cx": float(intrinsics["cx"]),
        "cy": float(intrinsics["cy"]),
        "source_frame": calibration["source_frame"],
        "depth_aligned_to_rgb": True,
    }
    (output / "camera_info.json").write_text(
        json.dumps(camera_info, indent=2) + "\n"
    )
    tick_index = {
        "source": str(src),
        "source_layout": EXPECTED_LAYOUT,
        "sequence": sequence,
        "input_modality": PREPARED_MODALITY,
        "projection_model": "pinhole",
        "pose_frame": "local",
        "pose_composition": "local_T_body(interpolated) @ body_T_head_D455_color",
        "camera_frame": calibration["source_frame"],
        "fx": camera_info["fx"],
        "fy": camera_info["fy"],
        "cx": camera_info["cx"],
        "cy": camera_info["cy"],
        "width": camera_info["width"],
        "height": camera_info["height"],
        "recommended_max_depth_m": maximum_depth_m,
        "depth_evidence_type": "aligned_rgbd_sensor",
        "time_origin_ns": origin_ns,
        "timebase": {
            "clock": contract["camera_time_source"],
            "unit": "ns",
            "shared_timebase_verified": True,
            "rgb_depth_same_capture_verified": True,
            "timestamp_definition": "(sensor_time_ns - time_origin_ns) / 1e9",
        },
        "pose_time_alignment": {
            "method": "SE3_interpolation_at_camera_capture_time",
            "source_pose": "state/<sequence>/odom.jsonl local_T_body",
            "pose_timestamp_file": "pose/pose_timestamps_ns.txt",
            "pose_row_field": "pose_row",
            "maximum_pose_interpolation_gap_ms": contract[
                "maximum_pose_interpolation_gap_ms"
            ],
        },
        "frames": output_frames,
    }
    (output / "tick_index.json").write_text(
        json.dumps(tick_index, indent=2) + "\n"
    )
    (output / "source_manifest.json").write_text(
        json.dumps(load_json(src / "manifest.json"), indent=2) + "\n"
    )
    (output / "source_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )

    report = {
        "schema": "unitree_head_rgbd_preparation_v1",
        "status": "complete",
        "source": str(src),
        "output": str(output),
        "input_modality": "unitree_head_d455_rgbd",
        "output_modality": PREPARED_MODALITY,
        "depth_backend": "sensor_rgbd",
        "confidence_mode": "sensor-validity",
        "settings": {
            "camera": EXPECTED_CAMERA,
            "sensor": EXPECTED_SENSOR,
            "camera_time_source": contract["camera_time_source"],
            "minimum_valid_depth_m": minimum_depth_m,
            "maximum_depth_m": maximum_depth_m,
            "depth_unit": "uint16_millimeter",
            "world_frame": "local",
        },
        "counts": {
            "source_frames": len(records),
            "prepared_frames": len(output_frames),
            "dropped_frames": len(dropped),
            "dropped_source_indices": dropped,
        },
        "depth": {
            "mean_valid_ratio": float(np.mean(valid_ratios)),
            "minimum_valid_ratio": min(valid_ratios),
            "maximum_valid_ratio": max(valid_ratios),
            "invalid_policy_counts": dict(invalid_policy_counts),
        },
        "artifacts": {
            "camera_calibration": str(camera_calibration_path),
            "camera_calibration_sha256": sha256_file(camera_calibration_path),
            "time_contract": str(time_contract_path),
            "time_contract_sha256": sha256_file(time_contract_path),
            "source_manifest_sha256": sha256_file(src / "manifest.json"),
            "source_manifest_jsonl_sha256": sha256_file(src / "manifest.jsonl"),
            "rgb_byte_copies_verified": source_rgb_hashes == output_rgb_hashes,
        },
        "pose_coverage": coverage,
    }
    (output / "rgbd_preparation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        report = audit_dataset(
            args.src,
            args.sequence,
            camera_calibration_path=args.camera_calibration,
            time_contract_path=args.time_contract,
        )
        if args.report is not None:
            report_path = args.report.resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    if args.camera_calibration is None or args.time_contract is None:
        raise ValueError(
            "prepare requires --camera-calibration and --time-contract; missing "
            "contracts are never inferred"
        )
    report = prepare_dataset(
        args.src,
        args.output,
        args.sequence,
        camera_calibration_path=args.camera_calibration,
        time_contract_path=args.time_contract,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
