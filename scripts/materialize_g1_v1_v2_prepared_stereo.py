#!/usr/bin/env python3
"""Materialize a validated V1/V2 rectification as prepared stereo.

The source PNGs are already monocular-undistorted pinhole images.  This script
therefore applies only the fixed stereo homographies and the x-preserving
right-y residual stored in ``best_combination.json``.  Camera poses are rotated
into the rectified left-camera frame while absolute timestamps are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--pose-reference", required=True, type=Path)
    parser.add_argument("--calibration-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--png-compression", type=int, default=1)
    parser.add_argument(
        "--source-indices",
        nargs="+",
        type=int,
        help="Optional source tick subset for a smoke test.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def image_descriptors(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    descriptors = {
        str(image["camera"]): image for image in record.get("images", [])
    }
    if not {"cam0", "cam1"}.issubset(descriptors):
        raise ValueError(f"Stereo images missing at tick {record.get('tick')}")
    return descriptors


def resolve_source_path(source: Path, descriptor: dict[str, Any]) -> Path:
    path = Path(descriptor["path"])
    return path.resolve() if path.is_absolute() else (source / path).resolve()


def inverse_right_y_map(
    width: int,
    height: int,
    model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    if model.get("model") == "right_y_affine_x_preserving":
        a, b, c = np.asarray(
            model["coefficients_a_b_c"], dtype=np.float64
        )
        if abs(float(b)) < 1.0e-9:
            raise ValueError("Right-y affine model is singular")
        source_y = (y - a * x - c) / b
        return x, source_y.astype(np.float32)
    if model.get("model") == "right_y_projective_x_preserving":
        a, b, c, g, h = np.asarray(
            model["coefficients_a_b_c_g_h"], dtype=np.float64
        )
        denominator = y * h - b
        if np.any(np.abs(denominator) < 1.0e-6):
            raise ValueError("Right-y projective model is singular")
        source_y = (a * x + c - y * (g * x + 1.0)) / denominator
        return x, source_y.astype(np.float32)
    raise ValueError(f"Unsupported right-y model: {model.get('model')!r}")


def decode_png(path: Path) -> tuple[np.ndarray, str]:
    payload = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode PNG: {path}")
    return image, sha256_bytes(payload)


def encode_png(image: np.ndarray, compression: int) -> bytes:
    success, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, compression],
    )
    if not success:
        raise RuntimeError("Could not encode rectified PNG")
    return encoded.tobytes()


def transform_pair(
    frame: dict[str, Any],
    source_record: dict[str, Any],
    source: Path,
    output: Path,
    left_homography: np.ndarray,
    right_homography: np.ndarray,
    right_map_x: np.ndarray,
    right_map_y: np.ndarray,
    width: int,
    height: int,
    png_compression: int,
) -> tuple[int, dict[str, Any]]:
    descriptors = image_descriptors(source_record)
    left_source = resolve_source_path(source, descriptors["cam0"])
    right_source = resolve_source_path(source, descriptors["cam1"])
    left, left_source_sha256 = decode_png(left_source)
    right, right_source_sha256 = decode_png(right_source)
    if left.shape[:2] != (height, width) or right.shape[:2] != (height, width):
        raise ValueError(
            f"Unexpected source resolution at tick {frame['source_idx']}: "
            f"{left.shape[:2]} / {right.shape[:2]}"
        )

    rectified_left = cv2.warpPerspective(
        left,
        left_homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    rectified_right = cv2.warpPerspective(
        right,
        right_homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    rectified_right = cv2.remap(
        rectified_right,
        right_map_x,
        right_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    left_payload = encode_png(rectified_left, png_compression)
    right_payload = encode_png(rectified_right, png_compression)
    output_index = int(frame["idx"])
    left_output = output / "rgb" / f"{output_index:08d}.png"
    right_output = output / "stereo_right" / f"{output_index:08d}.png"
    left_output.write_bytes(left_payload)
    right_output.write_bytes(right_payload)
    integrity = {
        "source_cam0": str(left_source),
        "source_cam1": str(right_source),
        "source_cam0_sha256": left_source_sha256,
        "source_cam1_sha256": right_source_sha256,
        "output_cam0_sha256": sha256_bytes(left_payload),
        "output_cam1_sha256": sha256_bytes(right_payload),
    }
    return output_index, integrity


def validate_rotation(rotation: np.ndarray, name: str) -> None:
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    orthogonality = np.linalg.norm(rotation.T @ rotation - np.eye(3))
    determinant = np.linalg.det(rotation)
    if orthogonality > 1.0e-8 or abs(determinant - 1.0) > 1.0e-8:
        raise ValueError(
            f"{name} is not in SO(3): error={orthogonality}, det={determinant}"
        )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    pose_reference = args.pose_reference.resolve()
    calibration_path = args.calibration_report.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not 0 <= args.png_compression <= 9:
        raise ValueError("--png-compression must be in [0, 9]")

    calibration = load_json(calibration_path)
    if not calibration.get("acceptance", {}).get("passed", False):
        raise ValueError("Combination calibration has not passed acceptance")
    experiment = calibration["experiment"]
    if Path(experiment["dataset"]).resolve() != source:
        raise ValueError(
            "Calibration dataset differs from materialization source: "
            f"{experiment['dataset']} vs {source}"
        )
    selected = calibration["selected_candidate"]
    depth_projection = calibration["depth_projection"]
    source_left_K = np.asarray(selected["source_left_K"], dtype=np.float64)
    source_right_K = np.asarray(selected["source_right_K"], dtype=np.float64)
    target_K = np.asarray(depth_projection["K"], dtype=np.float64)
    left_homography = np.asarray(
        selected["source_to_rectified_left_H"], dtype=np.float64
    )
    right_homography = np.asarray(
        selected["source_to_rectified_right_base_H"], dtype=np.float64
    )
    rectification_left = np.asarray(
        selected["rectification_left_R"], dtype=np.float64
    )
    vertical_model = selected["vertical_model"]
    for matrix, name in (
        (source_left_K, "source_left_K"),
        (source_right_K, "source_right_K"),
        (target_K, "target_K"),
        (left_homography, "left_homography"),
        (right_homography, "right_homography"),
    ):
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must be a finite 3x3 matrix")
    validate_rotation(rectification_left, "rectification_left_R")

    reference_index = load_json(pose_reference / "tick_index.json")
    reference_camera = load_json(pose_reference / "camera_info.json")
    reference_preparation = load_json(
        pose_reference / "pinhole_preparation_report.json"
    )
    if reference_index.get("pose_frame") != "map":
        raise ValueError("Pose reference must use map-frame camera poses")
    if reference_index.get("camera_quaternion_order") != "xyzw":
        raise ValueError("Pose reference must use the validated xyzw convention")
    if not reference_preparation.get("left_image_pixels_preserved", False):
        raise ValueError(
            "Pose reference left orientation was already modified; refusing "
            "to apply the rectified-left pose rotation twice"
        )
    reference_rotation = np.asarray(
        reference_preparation["opencv_left_original_to_virtual_R"],
        dtype=np.float64,
    )
    if not np.allclose(reference_rotation, np.eye(3), atol=1.0e-10):
        raise ValueError("Pose reference does not preserve source-left orientation")

    source_calibration_dir = source / "calibrations" / "000000"
    source_intrinsics = {}
    for camera in ("cam0", "cam1"):
        source_intrinsics[camera] = yaml.safe_load(
            (
                source_calibration_dir
                / f"calib_{camera}_intrinsics.yaml"
            ).read_text()
        )["intrinsics"]
    stored_left_K = np.asarray(
        source_intrinsics["cam0"]["K"], dtype=np.float64
    ).reshape(3, 3)
    stored_right_K = np.asarray(
        source_intrinsics["cam1"]["K"], dtype=np.float64
    ).reshape(3, 3)
    if not np.allclose(stored_left_K, source_left_K, atol=1.0e-9):
        raise ValueError("Source cam0 K differs from accepted combination")
    if not np.allclose(stored_right_K, source_right_K, atol=1.0e-9):
        raise ValueError("Source cam1 K differs from accepted combination")
    for camera in ("cam0", "cam1"):
        distortion = np.asarray(
            source_intrinsics[camera]["D"], dtype=np.float64
        )
        if not np.allclose(distortion, 0.0, atol=1.0e-12):
            raise ValueError(f"{camera} stored pixels are not zero-distortion")
    width = int(reference_camera["width"])
    height = int(reference_camera["height"])
    if [width, height] != [1280, 960]:
        raise ValueError(f"Unexpected image size: {width}x{height}")

    source_records = {
        int(record["tick"]): record
        for record in load_jsonl(source / "manifest.jsonl")
    }
    frames = list(reference_index["frames"])
    if args.source_indices is not None:
        requested = set(args.source_indices)
        frames = [
            frame for frame in frames if int(frame["source_idx"]) in requested
        ]
        found = {int(frame["source_idx"]) for frame in frames}
        if found != requested:
            raise ValueError(
                f"Source indices absent from pose reference: {sorted(requested-found)}"
            )
    if not frames:
        raise ValueError("No frames selected for materialization")
    time_origin_ns = int(frames[0]["sensor_time_ns"])
    normalized_frames = []
    for output_index, frame in enumerate(frames):
        normalized = dict(frame)
        normalized["_reference_pose_row"] = int(frame["pose_row"])
        normalized["idx"] = output_index
        normalized["pose_row"] = output_index
        normalized["timestamp"] = (
            int(frame["sensor_time_ns"]) - time_origin_ns
        ) / 1.0e9
        normalized_frames.append(normalized)
    frames = normalized_frames
    missing_records = [
        int(frame["source_idx"])
        for frame in frames
        if int(frame["source_idx"]) not in source_records
    ]
    if missing_records:
        raise ValueError(f"Source manifest ticks missing: {missing_records}")

    source_poses = np.loadtxt(
        pose_reference / "pose" / "poses.txt", dtype=np.float64
    ).reshape(-1, 4, 4)
    source_timestamps = np.loadtxt(
        pose_reference / "pose" / "pose_timestamps_ns.txt",
        dtype=np.int64,
    ).reshape(-1)
    pose_rows = np.asarray(
        [int(frame["_reference_pose_row"]) for frame in frames]
    )
    if np.any(pose_rows < 0) or np.any(pose_rows >= len(source_poses)):
        raise ValueError("Pose rows fall outside the reference pose file")
    rectified_poses = source_poses[pose_rows].copy()
    rectified_poses[:, :3, :3] = (
        rectified_poses[:, :3, :3] @ rectification_left.T
    )
    rectified_timestamps = source_timestamps[pose_rows]
    expected_timestamps = np.asarray(
        [int(frame["pose_sensor_time_ns"]) for frame in frames],
        dtype=np.int64,
    )
    if not np.array_equal(rectified_timestamps, expected_timestamps):
        raise ValueError("Pose timestamp rows disagree with tick_index")

    for directory in (output / "rgb", output / "stereo_right", output / "pose"):
        directory.mkdir(parents=True, exist_ok=False)
    right_map_x, right_map_y = inverse_right_y_map(
        width, height, vertical_model
    )
    integrity_by_index: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for frame in frames:
            source_index = int(frame["source_idx"])
            futures.append(
                executor.submit(
                    transform_pair,
                    frame,
                    source_records[source_index],
                    source,
                    output,
                    left_homography,
                    right_homography,
                    right_map_x,
                    right_map_y,
                    width,
                    height,
                    args.png_compression,
                )
            )
        for completed, future in enumerate(as_completed(futures), start=1):
            output_index, integrity = future.result()
            integrity_by_index[output_index] = integrity
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"materialized {completed}/{len(futures)} stereo pairs",
                    flush=True,
                )

    pose_lines = [
        " ".join(f"{value:.12g}" for value in pose.reshape(-1))
        for pose in rectified_poses
    ]
    (output / "pose" / "poses.txt").write_text("\n".join(pose_lines) + "\n")
    (output / "pose" / "pose_timestamps_ns.txt").write_text(
        "\n".join(str(int(value)) for value in rectified_timestamps) + "\n"
    )

    baseline = float(selected["baseline_m"])
    projection_left = np.asarray(depth_projection["P1"], dtype=np.float64)
    projection_right = np.asarray(depth_projection["P2"], dtype=np.float64)
    q_matrix = np.asarray(depth_projection["Q"], dtype=np.float64)
    if not np.isclose(
        projection_right[0, 3],
        -target_K[0, 0] * baseline,
        atol=1.0e-10,
    ):
        raise ValueError("P2 does not encode the accepted V2 baseline")
    camera_info = {
        "width": width,
        "height": height,
        "model": "pinhole",
        "intrinsics": target_K.tolist(),
        "distortion": [0.0, 0.0, 0.0, 0.0],
        "fx": float(target_K[0, 0]),
        "fy": float(target_K[1, 1]),
        "cx": float(target_K[0, 2]),
        "cy": float(target_K[1, 2]),
        "baseline": baseline,
        "P1": projection_left.tolist(),
        "P2": projection_right.tolist(),
        "Q": q_matrix.tolist(),
        "disparity_convention": "x_left_minus_x_right",
        "rectification_report": str(calibration_path),
        "rectification_report_sha256": sha256_file(calibration_path),
    }
    (output / "camera_info.json").write_text(
        json.dumps(camera_info, indent=2, allow_nan=False) + "\n"
    )

    output_frames = []
    for frame in frames:
        output_index = int(frame["idx"])
        updated = dict(frame)
        updated.pop("_reference_pose_row")
        updated["cam0"] = str(
            (output / "rgb" / f"{output_index:08d}.png").resolve()
        )
        updated["cam1"] = str(
            (output / "stereo_right" / f"{output_index:08d}.png").resolve()
        )
        updated["pose_row"] = output_index
        updated["rectification_integrity"] = integrity_by_index[output_index]
        output_frames.append(updated)

    output_index = {
        **{key: value for key, value in reference_index.items() if key != "frames"},
        "source": str(source),
        "projection_model": "pinhole",
        "stereo_rectified": True,
        "pose_composition": (
            "map_T_source_left_camera @ source_left_camera_T_rectified_left_camera"
        ),
        "fx": float(target_K[0, 0]),
        "fy": float(target_K[1, 1]),
        "cx": float(target_K[0, 2]),
        "cy": float(target_K[1, 2]),
        "baseline": baseline,
        "recommended_max_depth_m": 65.535,
        "time_origin_ns": time_origin_ns,
        "rectification_report": str(calibration_path),
        "rectification_report_sha256": sha256_file(calibration_path),
        "frames": output_frames,
    }
    (output / "tick_index.json").write_text(
        json.dumps(output_index, indent=2, allow_nan=False) + "\n"
    )

    source_manifest_sha256 = sha256_file(source / "manifest.json")
    source_manifest_jsonl_sha256 = sha256_file(source / "manifest.jsonl")
    report = {
        "contract_version": 1,
        "status": "complete",
        "source_dataset": str(source),
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_jsonl_sha256": source_manifest_jsonl_sha256,
        "pose_reference": str(pose_reference),
        "pose_reference_tick_index_sha256": sha256_file(
            pose_reference / "tick_index.json"
        ),
        "calibration_report": str(calibration_path),
        "calibration_report_sha256": sha256_file(calibration_path),
        "input_projection_model": "already_monocular_undistorted_pinhole",
        "fisheye_undistortion_applied": False,
        "stereo_rectification_applied": True,
        "selected_candidate": selected["name"],
        "source_to_rectified_left_H": left_homography.tolist(),
        "source_to_rectified_right_base_H": right_homography.tolist(),
        "right_vertical_model": vertical_model,
        "rectification_left_R": rectification_left.tolist(),
        "output_K": target_K.tolist(),
        "P1": projection_left.tolist(),
        "P2": projection_right.tolist(),
        "Q": q_matrix.tolist(),
        "baseline_m": baseline,
        "fb_px_m": float(target_K[0, 0] * baseline),
        "frames_written": len(output_frames),
        "first_source_index": int(output_frames[0]["source_idx"]),
        "last_source_index": int(output_frames[-1]["source_idx"]),
        "pose_frame": "map",
        "camera_quaternion_order": "xyzw",
        "rectified_image_valid_area": calibration[
            "rectified_image_valid_area"
        ],
        "calibration_holdout_metrics": calibration["holdout_metrics"],
        "recommended_max_depth_m": 65.535,
    }
    (output / "combination_preparation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    # Some existing pipeline diagnostics look for this conventional filename.
    shutil.copyfile(
        output / "combination_preparation_report.json",
        output / "pinhole_preparation_report.json",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "frames_written": len(output_frames),
                "pose_frame": "map",
                "fb_px_m": report["fb_px_m"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
