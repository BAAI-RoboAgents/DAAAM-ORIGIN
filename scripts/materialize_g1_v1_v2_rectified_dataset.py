#!/usr/bin/env python3
"""Materialize a held-out-validated G1 V1/V2 rectification as prepared stereo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from build_synchronized_stereo_dataset import (
    camera_timestamps,
    compose_global_camera_poses,
    load_jsonl,
    map_pose_sample,
    monotonic_matches,
)

CONTRACT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--calibration-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequence", default="000000")
    parser.add_argument("--source-start-index", type=int)
    parser.add_argument("--source-end-index", type=int)
    parser.add_argument("--maximum-stereo-delta-ms", type=float, default=10.0)
    parser.add_argument("--recommended-maximum-depth-m", type=float, default=65.535)
    parser.add_argument("--png-compression", type=int, default=1)
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_image_and_hash(path: Path) -> tuple[np.ndarray, str]:
    payload = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode image: {path}")
    return image, sha256_bytes(payload)


def encode_png(image: np.ndarray, compression: int) -> tuple[bytes, str]:
    success, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, compression],
    )
    if not success:
        raise RuntimeError("OpenCV PNG encoding failed")
    payload = encoded.tobytes()
    return payload, sha256_bytes(payload)


def image_descriptor(record: dict[str, Any], camera: str) -> dict[str, Any]:
    descriptors = {
        item["camera"]: item for item in record.get("images", [])
    }
    if camera not in descriptors:
        raise ValueError(f"Missing {camera} at tick {record.get('tick')}")
    return descriptors[camera]


def image_path(source: Path, record: dict[str, Any], camera: str) -> Path:
    path = Path(image_descriptor(record, camera)["path"])
    return path.resolve() if path.is_absolute() else (source / path).resolve()


def validate_rotation(rotation: np.ndarray, name: str) -> None:
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 1.0e-8:
        raise ValueError(f"{name} is not orthonormal")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-8:
        raise ValueError(f"{name} is not a proper rotation")


def load_rectification(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    if not report.get("acceptance", {}).get("passed", False):
        raise ValueError("Combination calibration did not pass held-out acceptance")
    selected = report["selected_candidate"]
    depth = report["depth_projection"]
    vertical = selected["vertical_model"]
    if vertical.get("model") != "right_y_projective_x_preserving":
        raise ValueError("Expected right_y_projective_x_preserving residual")

    left_h = np.asarray(
        selected["source_to_rectified_left_H"], dtype=np.float64
    )
    right_h = np.asarray(
        selected["source_to_rectified_right_base_H"], dtype=np.float64
    )
    left_r = np.asarray(selected["rectification_left_R"], dtype=np.float64)
    right_r = np.asarray(selected["rectification_right_R"], dtype=np.float64)
    stereo_r = np.asarray(
        selected["rotation_cam1_from_cam0"], dtype=np.float64
    )
    stereo_t = np.asarray(
        selected["translation_cam1_from_cam0_m"], dtype=np.float64
    )
    K = np.asarray(depth["K"], dtype=np.float64)
    P1 = np.asarray(depth["P1"], dtype=np.float64)
    P2 = np.asarray(depth["P2"], dtype=np.float64)
    Q = np.asarray(depth["Q"], dtype=np.float64)
    for matrix, name, shape in (
        (left_h, "left homography", (3, 3)),
        (right_h, "right homography", (3, 3)),
        (K, "K", (3, 3)),
        (P1, "P1", (3, 4)),
        (P2, "P2", (3, 4)),
        (Q, "Q", (4, 4)),
    ):
        if matrix.shape != shape or not np.isfinite(matrix).all():
            raise ValueError(f"{name} must be a finite {shape} matrix")
    for rotation, name in (
        (left_r, "left rectification rotation"),
        (right_r, "right rectification rotation"),
        (stereo_r, "stereo rotation"),
    ):
        validate_rotation(rotation, name)
    baseline = float(selected["baseline_m"])
    if (
        stereo_t.shape != (3,)
        or not np.isfinite(stereo_t).all()
        or not np.isclose(np.linalg.norm(stereo_t), baseline, atol=1.0e-12)
    ):
        raise ValueError("Stereo translation does not encode the reported baseline")
    expected_tx = -float(K[0, 0]) * baseline
    if not np.isclose(P2[0, 3], expected_tx, atol=1.0e-10):
        raise ValueError("P2 does not encode the reported metric baseline")
    if not np.isclose(Q[3, 2], 1.0 / baseline, atol=1.0e-10):
        raise ValueError("Q does not encode the reported metric baseline")
    return {
        "document": report,
        "left_h": left_h,
        "right_h": right_h,
        "left_r": left_r,
        "right_r": right_r,
        "stereo_r": stereo_r,
        "stereo_t": stereo_t,
        "K": K,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "baseline": baseline,
        "vertical": vertical,
    }


def inverse_homography_map(
    homography: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(homography)
    denominator = inverse[2, 0] * x + inverse[2, 1] * y + inverse[2, 2]
    source_x = (
        inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]
    ) / denominator
    source_y = (
        inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]
    ) / denominator
    return source_x, source_y


def build_maps(
    width: int,
    height: int,
    left_h: np.ndarray,
    right_h: np.ndarray,
    vertical: dict[str, Any],
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    dict[str, float],
    np.ndarray,
]:
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    left_x, left_y = inverse_homography_map(left_h, x, y)

    a, b, c, g, h = np.asarray(
        vertical["coefficients_a_b_c_g_h"], dtype=np.float64
    )
    vertical_denominator = y * h - b
    if np.any(np.abs(vertical_denominator) < 1.0e-8):
        raise ValueError("Right vertical residual is singular in the output image")
    right_base_y = (
        a * x + c - y * (g * x + 1.0)
    ) / vertical_denominator
    right_x, right_y = inverse_homography_map(
        right_h,
        x,
        right_base_y,
    )
    left_valid = (
        np.isfinite(left_x)
        & np.isfinite(left_y)
        & (left_x >= 0.0)
        & (left_x <= width - 1)
        & (left_y >= 0.0)
        & (left_y <= height - 1)
    )
    right_valid = (
        np.isfinite(right_x)
        & np.isfinite(right_y)
        & (right_x >= 0.0)
        & (right_x <= width - 1)
        & (right_y >= 0.0)
        & (right_y <= height - 1)
    )
    joint_valid = left_valid & right_valid
    valid_area = {
        "left_valid_area_ratio": float(np.mean(left_valid)),
        "right_valid_area_ratio": float(np.mean(right_valid)),
        "joint_valid_area_ratio": float(np.mean(joint_valid)),
    }
    return (
        (left_x.astype(np.float32), left_y.astype(np.float32)),
        (right_x.astype(np.float32), right_y.astype(np.float32)),
        valid_area,
        (joint_valid.astype(np.uint8) * 255),
    )


def pose_coverage(
    source: Path,
    sequence: str,
) -> tuple[int, int]:
    map_records = load_jsonl(source / "state" / sequence / "map_pose.jsonl")
    map_timestamps = [map_pose_sample(record)[0] for record in map_records]
    aux_records = load_jsonl(
        source / "poses" / "dense_global" / sequence / "aux_poses.jsonl"
    )
    camera_timestamps_ns = [
        int(record["poses"]["head_camera"]["timestamp_ns"])
        for record in aux_records
    ]
    return (
        max(map_timestamps[0], camera_timestamps_ns[0]),
        min(map_timestamps[-1], camera_timestamps_ns[-1]),
    )


def write_preview(
    output: Path,
    left: np.ndarray,
    right: np.ndarray,
    source_index: int,
) -> None:
    scale = 0.5
    left_small = cv2.resize(left, None, fx=scale, fy=scale)
    right_small = cv2.resize(right, None, fx=scale, fy=scale)
    canvas = np.hstack((left_small, right_small))
    for row in range(20, canvas.shape[0], 40):
        cv2.line(
            canvas,
            (0, row),
            (canvas.shape[1] - 1, row),
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    path = output / "previews" / f"{source_index:06d}.jpg"
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not write preview: {path}")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    report_path = args.calibration_report.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if args.maximum_stereo_delta_ms <= 0.0:
        raise ValueError("--maximum-stereo-delta-ms must be positive")
    if (args.source_start_index is None) != (args.source_end_index is None):
        raise ValueError(
            "--source-start-index and --source-end-index must be specified together"
        )
    if (
        args.source_start_index is not None
        and args.source_end_index < args.source_start_index
    ):
        raise ValueError("source frame range is invalid")
    if args.recommended_maximum_depth_m <= 0.0:
        raise ValueError("--recommended-maximum-depth-m must be positive")
    if not 0 <= args.png_compression <= 9:
        raise ValueError("--png-compression must be in [0, 9]")

    manifest = json.loads((source / "manifest.json").read_text())
    if (manifest.get("layout_version") or manifest.get("layout")) != (
        "capture4daaam_like"
    ):
        raise ValueError("Expected capture4daaam_like input")
    quality = json.loads((source / "quality_report.json").read_text())
    if not quality.get("alignment", {}).get("ok", False):
        raise ValueError("Source quality report alignment is not OK")
    rectification = load_rectification(report_path)
    records = load_jsonl(source / "manifest.jsonl")
    if not records:
        raise ValueError("Source manifest is empty")
    if args.source_start_index is not None:
        records = [
            record
            for record in records
            if args.source_start_index
            <= int(record["tick"])
            <= args.source_end_index
        ]
        expected_ticks = list(
            range(args.source_start_index, args.source_end_index + 1)
        )
        actual_ticks = [int(record["tick"]) for record in records]
        if actual_ticks != expected_ticks:
            raise ValueError(
                "Selected source records are not one unique contiguous range: "
                f"expected {expected_ticks[0]}-{expected_ticks[-1]}, "
                f"found {actual_ticks[:3]}...{actual_ticks[-3:]}"
            )

    first_image, _ = load_image_and_hash(image_path(source, records[0], "cam0"))
    height, width = first_image.shape[:2]
    left_map, right_map, valid_area, joint_valid_mask = build_maps(
        width,
        height,
        rectification["left_h"],
        rectification["right_h"],
        rectification["vertical"],
    )
    reported_valid_area = rectification["document"].get(
        "rectified_image_valid_area", {}
    )
    if reported_valid_area and any(
        abs(valid_area[key] - float(reported_valid_area[key])) > 2.0e-3
        for key in valid_area
    ):
        raise ValueError(
            "Materialized inverse maps disagree with calibration valid-area report"
        )
    if valid_area["joint_valid_area_ratio"] < 0.75:
        raise ValueError("Joint valid image area is below the accepted gate")

    left_timestamps = camera_timestamps(records, "cam0")
    right_timestamps = camera_timestamps(records, "cam1")
    threshold_ns = int(round(args.maximum_stereo_delta_ms * 1.0e6))
    matches, skipped_left, skipped_right = monotonic_matches(
        left_timestamps,
        right_timestamps,
        threshold_ns,
    )
    coverage_start_ns, coverage_end_ns = pose_coverage(source, args.sequence)
    coverage_dropped = []
    covered_matches = []
    for match in matches:
        timestamp_ns = int(left_timestamps[match[0]])
        if coverage_start_ns <= timestamp_ns <= coverage_end_ns:
            covered_matches.append(match)
        else:
            coverage_dropped.append(match[0])
    matches = covered_matches
    if not matches:
        raise ValueError("No synchronized pairs lie inside pose coverage")
    selected_timestamps = left_timestamps[
        [left_index for left_index, _, _ in matches]
    ]
    global_poses, map_clamped, camera_clamped = compose_global_camera_poses(
        source,
        selected_timestamps,
        camera_quaternion_order="xyzw",
        base_pose_source="map",
        sequence=args.sequence,
    )
    if map_clamped or camera_clamped:
        raise ValueError(
            "Pose interpolation was clamped after explicit coverage filtering"
        )
    for pose in global_poses:
        pose[:3, :3] = pose[:3, :3] @ rectification["left_r"].T

    for directory in ("rgb", "stereo_right", "pose", "previews"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output / "joint_valid_mask.png"), joint_valid_mask):
        raise RuntimeError("Could not write joint valid mask")

    preview_positions = {0, len(matches) // 2, len(matches) - 1}
    frame_records = []
    integrity_records = []
    for output_index, ((left_index, right_index, delta_ns), pose) in enumerate(
        zip(matches, global_poses)
    ):
        left_record = records[left_index]
        right_record = records[right_index]
        left_source_path = image_path(source, left_record, "cam0")
        right_source_path = image_path(source, right_record, "cam1")
        left_source, left_source_sha = load_image_and_hash(left_source_path)
        right_source, right_source_sha = load_image_and_hash(right_source_path)
        if (
            left_source.shape[:2] != (height, width)
            or right_source.shape[:2] != (height, width)
        ):
            raise ValueError(f"Unexpected image size at source tick {left_index}")

        left_rectified = cv2.remap(
            left_source,
            left_map[0],
            left_map[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        right_rectified = cv2.remap(
            right_source,
            right_map[0],
            right_map[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        left_payload, left_output_sha = encode_png(
            left_rectified, args.png_compression
        )
        right_payload, right_output_sha = encode_png(
            right_rectified, args.png_compression
        )
        left_output = output / "rgb" / f"{output_index:08d}.png"
        right_output = output / "stereo_right" / f"{output_index:08d}.png"
        left_output.write_bytes(left_payload)
        right_output.write_bytes(right_payload)

        source_index = int(left_record["tick"])
        if output_index in preview_positions:
            write_preview(
                output,
                left_rectified,
                right_rectified,
                source_index,
            )
        left_time_ns = int(
            image_descriptor(left_record, "cam0")["sensor_time_ns"]
        )
        right_time_ns = int(
            image_descriptor(right_record, "cam1")["sensor_time_ns"]
        )
        frame_records.append(
            {
                "idx": output_index,
                "source_idx": source_index,
                "cam0_source_idx": int(left_record["tick"]),
                "cam1_source_idx": int(right_record["tick"]),
                "pose_row": output_index,
                "cam0": str(left_output),
                "cam1": str(right_output),
                "timestamp": (
                    left_time_ns - int(selected_timestamps[0])
                )
                / 1.0e9,
                "cam0_sensor_time_ns": left_time_ns,
                "cam1_sensor_time_ns": right_time_ns,
                "sensor_time_ns": left_time_ns,
                "pose_sensor_time_ns": left_time_ns,
                "stereo_delta_ms": delta_ns / 1.0e6,
            }
        )
        integrity_records.append(
            {
                "idx": output_index,
                "source_idx": source_index,
                "cam0_source_path": str(left_source_path),
                "cam1_source_path": str(right_source_path),
                "cam0_source_sha256": left_source_sha,
                "cam1_source_sha256": right_source_sha,
                "cam0_output_sha256": left_output_sha,
                "cam1_output_sha256": right_output_sha,
            }
        )
        if output_index % 50 == 0 or output_index == len(matches) - 1:
            print(
                f"materialized {output_index + 1}/{len(matches)} "
                f"source={source_index}",
                flush=True,
            )

    pose_text = "".join(
        " ".join(f"{value:.12g}" for value in pose.reshape(-1)) + "\n"
        for pose in global_poses
    )
    (output / "pose" / "poses.txt").write_text(pose_text)
    (output / "pose" / "pose_timestamps_ns.txt").write_text(
        "".join(f"{int(timestamp)}\n" for timestamp in selected_timestamps)
    )
    K = rectification["K"]
    camera_info = {
        "width": width,
        "height": height,
        "model": "pinhole",
        "intrinsics": K.tolist(),
        "distortion": [0.0, 0.0, 0.0, 0.0],
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "baseline": rectification["baseline"],
        "P1": rectification["P1"].tolist(),
        "P2": rectification["P2"].tolist(),
        "Q": rectification["Q"].tolist(),
        "disparity_convention": "x_left - x_right",
        "rectification_report": str(report_path),
        "rectification_report_sha256": sha256_file(report_path),
    }
    (output / "camera_info.json").write_text(
        json.dumps(camera_info, indent=2) + "\n"
    )

    translations = np.asarray([pose[:3, 3] for pose in global_poses])
    tick_index = {
        "materialization_contract_version": CONTRACT_VERSION,
        "source": str(source),
        "source_layout": "capture4daaam_like",
        "sequence": args.sequence,
        "input_modality": "stereo",
        "projection_model": "pinhole",
        "source_projection_model": "already_monocular_undistorted_pinhole",
        "stereo_rectification": "held_out_validated_v1_v2_combination",
        "pose_frame": "map",
        "base_pose_source": "map",
        "camera_quaternion_order": "xyzw",
        "pose_composition": (
            "map_T_base_link @ base_link_T_head_camera @ "
            "source_camera_T_rectified_left_camera"
        ),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "baseline": rectification["baseline"],
        "width": width,
        "height": height,
        "recommended_max_depth_m": args.recommended_maximum_depth_m,
        "time_origin_ns": int(selected_timestamps[0]),
        "timebase": {
            "clock": "sensor_time_ns",
            "unit": "ns",
            "timestamp_definition": (
                "(sensor_time_ns - time_origin_ns) / 1e9"
            ),
        },
        "pose_time_alignment": {
            "method": (
                "interpolate_map_base_and_head_camera_at_cam0_sensor_time_ns"
            ),
            "pose_timestamp_file": "pose/pose_timestamps_ns.txt",
            "pose_row_field": "pose_row",
        },
        "rectification_provenance": {
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
            "left_homography": rectification["left_h"].tolist(),
            "right_base_homography": rectification["right_h"].tolist(),
            "right_vertical_model": rectification["vertical"],
            "left_rectification_rotation": rectification["left_r"].tolist(),
            "joint_valid_area_ratio": valid_area[
                "joint_valid_area_ratio"
            ],
        },
        "frames": frame_records,
    }
    (output / "tick_index.json").write_text(
        json.dumps(tick_index, indent=2) + "\n"
    )
    (output / "image_integrity.json").write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "rectification_report_sha256": sha256_file(report_path),
                "frames": integrity_records,
            },
            indent=2,
        )
        + "\n"
    )
    source_manifest = {
        "source_capture": str(source),
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "source_manifest_jsonl_sha256": sha256_file(
            source / "manifest.jsonl"
        ),
        "source_quality_report_sha256": sha256_file(
            source / "quality_report.json"
        ),
        "rectification_report": str(report_path),
        "rectification_report_sha256": sha256_file(report_path),
    }
    (output / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n"
    )
    report = {
        "status": "complete",
        "contract_version": CONTRACT_VERSION,
        "source": str(source),
        "output": str(output),
        "sequence": args.sequence,
        "source_frames": len(records),
        "source_frame_range": [
            int(records[0]["tick"]),
            int(records[-1]["tick"]),
        ],
        "matched_frames_before_pose_coverage": len(covered_matches)
        + len(coverage_dropped),
        "materialized_frames": len(matches),
        "skipped_cam0_sync": len(skipped_left),
        "skipped_cam1_sync": len(skipped_right),
        "pose_coverage_dropped_source_indices": coverage_dropped,
        "maximum_matched_stereo_delta_ms": max(
            delta for _, _, delta in matches
        )
        / 1.0e6,
        "map_pose_interpolation_clamped": map_clamped,
        "camera_pose_interpolation_clamped": camera_clamped,
        "valid_area": valid_area,
        "source_to_output": {
            "fisheye_undistortion_applied": False,
            "left_remap": "inverse source_to_rectified_left_H",
            "right_remap": (
                "inverse right_y_projective_x_preserving composed with "
                "inverse source_to_rectified_right_base_H"
            ),
            "resampling_passes_per_image": 1,
            "interpolation": "cv2.INTER_LINEAR",
            "border_mode": "cv2.BORDER_CONSTANT",
            "png_compression": args.png_compression,
        },
        "calibration": camera_info,
        "pose": {
            "frame": "map",
            "camera_quaternion_order": "xyzw",
            "left_rectification_rotation": rectification["left_r"].tolist(),
            "first_translation_m": translations[0].tolist(),
            "last_translation_m": translations[-1].tolist(),
            "path_length_m": float(
                np.linalg.norm(np.diff(translations, axis=0), axis=1).sum()
            ),
        },
        "artifacts": {
            "camera_info": str(output / "camera_info.json"),
            "tick_index": str(output / "tick_index.json"),
            "image_integrity": str(output / "image_integrity.json"),
            "joint_valid_mask": str(output / "joint_valid_mask.png"),
        },
    }
    (output / "rectification_materialization_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
