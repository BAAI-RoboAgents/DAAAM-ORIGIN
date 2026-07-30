#!/usr/bin/env python3
"""Build a zero-remap multi-frame stereo validation index.

The output contains metadata only.  Every frame path points directly at the
recorded ``2d_rect`` PNG, so no image pixels are copied or transformed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-indices", required=True, nargs="+", type=int)
    parser.add_argument(
        "--reference-camera",
        choices=("cam0", "cam1"),
        default="cam0",
        help=(
            "Camera supplied as the model's reference/left input. cam1 creates "
            "a camera-order control without modifying either recorded image."
        ),
    )
    parser.add_argument("--maximum-depth-m", type=float, default=30.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    source = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if not args.source_indices or len(set(args.source_indices)) != len(
        args.source_indices
    ):
        raise ValueError("source-indices must be non-empty and unique")
    if min(args.source_indices) < 0 or args.maximum_depth_m <= 0.0:
        raise ValueError("source indices and maximum depth are invalid")

    calibration_dir = source / "calibrations" / "000000"
    left_calibration = yaml.safe_load(
        (calibration_dir / "calib_cam0_intrinsics.yaml").read_text()
    )["intrinsics"]
    right_calibration = yaml.safe_load(
        (calibration_dir / "calib_cam1_intrinsics.yaml").read_text()
    )["intrinsics"]
    stereo = yaml.safe_load(
        (calibration_dir / "calib_cam0_to_cam1.yaml").read_text()
    )
    left_k = np.asarray(left_calibration["K"], dtype=np.float64).reshape(3, 3)
    right_k = np.asarray(right_calibration["K"], dtype=np.float64).reshape(3, 3)
    left_r = np.asarray(left_calibration["R"], dtype=np.float64).reshape(3, 3)
    right_r = np.asarray(right_calibration["R"], dtype=np.float64).reshape(3, 3)
    left_d = np.asarray(left_calibration["D"], dtype=np.float64)
    right_d = np.asarray(right_calibration["D"], dtype=np.float64)
    left_t_right = np.asarray(
        stereo["transform"]["matrix_4x4"], dtype=np.float64
    )
    reference_camera = args.reference_camera
    partner_camera = "cam1" if reference_camera == "cam0" else "cam0"
    calibration_by_camera = {
        "cam0": {
            "K": left_k,
            "R": left_r,
            "D": left_d,
        },
        "cam1": {
            "K": right_k,
            "R": right_r,
            "D": right_d,
        },
    }
    reference_calibration = calibration_by_camera[reference_camera]
    partner_calibration = calibration_by_camera[partner_camera]
    reference_projection_model = str(
        (
            left_calibration
            if reference_camera == "cam0"
            else right_calibration
        )["distortion_model"]
    )
    baseline = float(np.linalg.norm(left_t_right[:3, 3]))
    if not np.allclose(left_k, right_k, atol=1.0e-12):
        raise ValueError("Post-rectification left/right intrinsics differ")
    if not np.allclose(left_r, np.eye(3)) or not np.allclose(right_r, np.eye(3)):
        raise ValueError("Post-rectification rotations are not identity")
    if not np.allclose(left_d, 0.0) or not np.allclose(right_d, 0.0):
        raise ValueError("Stored rectified pixels declare non-zero distortion")
    if baseline <= 0.0:
        raise ValueError("Stereo baseline must be positive")

    records = {int(item["tick"]): item for item in load_jsonl(source / "manifest.jsonl")}
    frames = []
    integrity_frames = []
    expected_shape = None
    for output_index, source_index in enumerate(args.source_indices):
        if source_index not in records:
            raise IndexError(f"Source frame {source_index} is absent from the manifest")
        record = records[source_index]
        descriptor_by_camera = {
            camera: next(
                item for item in record["images"] if item["camera"] == camera
            )
            for camera in ("cam0", "cam1")
        }
        reference_descriptor = descriptor_by_camera[reference_camera]
        partner_descriptor = descriptor_by_camera[partner_camera]
        reference_path = (source / reference_descriptor["path"]).resolve()
        partner_path = (source / partner_descriptor["path"]).resolve()
        reference_image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        partner_image = cv2.imread(str(partner_path), cv2.IMREAD_COLOR)
        if (
            reference_image is None
            or partner_image is None
            or reference_image.shape != partner_image.shape
        ):
            raise RuntimeError(f"Invalid stereo images for source frame {source_index}")
        if expected_shape is None:
            expected_shape = reference_image.shape
        elif reference_image.shape != expected_shape:
            raise ValueError("Selected stereo frames do not share one image shape")
        source_cam0_time = int(descriptor_by_camera["cam0"]["sensor_time_ns"])
        source_cam1_time = int(descriptor_by_camera["cam1"]["sensor_time_ns"])
        reference_time = int(reference_descriptor["sensor_time_ns"])
        partner_time = int(partner_descriptor["sensor_time_ns"])
        frames.append(
            {
                "idx": output_index,
                "source_idx": source_index,
                "cam0_source_idx": source_index,
                "cam1_source_idx": source_index,
                "pose_row": output_index,
                "cam0": str(reference_path),
                "cam1": str(partner_path),
                "model_reference_camera": reference_camera,
                "model_partner_camera": partner_camera,
                "timestamp": output_index / 30.0,
                "cam0_sensor_time_ns": reference_time,
                "cam1_sensor_time_ns": partner_time,
                "source_cam0_sensor_time_ns": source_cam0_time,
                "source_cam1_sensor_time_ns": source_cam1_time,
                "sensor_time_ns": reference_time,
                "pose_sensor_time_ns": reference_time,
                "stereo_delta_ms": abs(reference_time - partner_time) / 1.0e6,
                "image_geometry_operation": "none",
            }
        )
        integrity_frames.append(
            {
                "output_index": output_index,
                "source_index": source_index,
                "reference": {
                    "camera": reference_camera,
                    "sensor": reference_descriptor["sensor"],
                    "path": str(reference_path),
                    "sha256": sha256(reference_path),
                },
                "partner": {
                    "camera": partner_camera,
                    "sensor": partner_descriptor["sensor"],
                    "path": str(partner_path),
                    "sha256": sha256(partner_path),
                },
                "shape_hwc": list(reference_image.shape),
                "stereo_delta_ms": abs(reference_time - partner_time) / 1.0e6,
            }
        )

    assert expected_shape is not None
    height, width = expected_shape[:2]
    reference_k = reference_calibration["K"]
    fx = float(reference_k[0, 0])
    fy = float(reference_k[1, 1])
    cx = float(reference_k[0, 2])
    cy = float(reference_k[1, 2])
    tick_index = {
        "source": str(source),
        "projection_model": reference_projection_model,
        "stereo_inference_projection_assumption": "pinhole_horizontal_disparity",
        "pose_frame": "map",
        "reference_camera": reference_camera,
        "partner_camera": partner_camera,
        "model_input_order": [reference_camera, partner_camera],
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "baseline": baseline,
        "width": width,
        "height": height,
        "recommended_max_depth_m": args.maximum_depth_m,
        "image_policy": (
            "Direct absolute references to recorded 2d_rect PNGs; no copy, "
            "resize, crop, rotation, remap, or re-encoding"
        ),
        "frames": frames,
    }
    camera_info = {
        "width": width,
        "height": height,
        "model": reference_projection_model,
        "stereo_inference_projection_assumption": "pinhole_horizontal_disparity",
        "reference_camera": reference_camera,
        "partner_camera": partner_camera,
        "intrinsics": reference_k.tolist(),
        "distortion": reference_calibration["D"].tolist(),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "baseline": baseline,
    }
    integrity = {
        "contract": "zero-remap direct rectified stereo validation",
        "source_dataset": str(source),
        "reference_camera": reference_camera,
        "partner_camera": partner_camera,
        "model_input_order": [reference_camera, partner_camera],
        "calibration": {
            "reference_projection_model": reference_projection_model,
            "left_K": left_k.tolist(),
            "right_K": right_k.tolist(),
            "left_R": left_r.tolist(),
            "right_R": right_r.tolist(),
            "left_D": left_d.tolist(),
            "right_D": right_d.tolist(),
            "left_T_right": left_t_right.tolist(),
            "baseline_m": baseline,
            "reference_K": reference_calibration["K"].tolist(),
            "partner_K": partner_calibration["K"].tolist(),
        },
        "operations": {
            "copy": False,
            "resize": False,
            "crop": False,
            "rotation": False,
            "remap": False,
            "decode_reencode": False,
        },
        "frames": integrity_frames,
    }
    output.mkdir(parents=True)
    (output / "tick_index.json").write_text(json.dumps(tick_index, indent=2) + "\n")
    (output / "camera_info.json").write_text(
        json.dumps(camera_info, indent=2) + "\n"
    )
    (output / "input_integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n"
    )
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
