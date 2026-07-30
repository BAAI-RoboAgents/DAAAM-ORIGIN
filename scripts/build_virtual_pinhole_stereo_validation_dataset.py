#!/usr/bin/env python3
"""Build a calibration-only virtual-pinhole stereo validation dataset.

Both stored Kannala-Brandt images are remapped with the same identity optical
rotation into a centred pinhole camera.  No LiDAR, image-derived homography, or
additional roll/pitch/yaw correction is used.
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
    parser.add_argument("--horizontal-fov-deg", type=float, default=120.0)
    parser.add_argument("--maximum-depth-m", type=float, default=30.0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def draw_label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    args = parse_args()
    source = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if (
        not args.source_indices
        or len(set(args.source_indices)) != len(args.source_indices)
        or min(args.source_indices) < 0
    ):
        raise ValueError("source-indices must be unique non-negative values")
    if not 60.0 <= args.horizontal_fov_deg <= 140.0:
        raise ValueError("horizontal-fov-deg must be in [60, 140]")
    if args.maximum_depth_m <= 0.0:
        raise ValueError("maximum-depth-m must be positive")

    calibration_dir = source / "calibrations" / "000000"
    calibration_by_camera = {}
    for camera in ("cam0", "cam1"):
        calibration = yaml.safe_load(
            (calibration_dir / f"calib_{camera}_intrinsics.yaml").read_text()
        )["intrinsics"]
        if calibration["distortion_model"] != "kannala_brandt":
            raise ValueError(
                f"{camera} source projection is not Kannala-Brandt"
            )
        calibration_by_camera[camera] = calibration
    source_k_by_camera = {
        camera: np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3)
        for camera, calibration in calibration_by_camera.items()
    }
    source_d_by_camera = {
        camera: np.asarray(calibration["D"], dtype=np.float64).reshape(4, 1)
        for camera, calibration in calibration_by_camera.items()
    }
    widths = {int(calibration["width"]) for calibration in calibration_by_camera.values()}
    heights = {
        int(calibration["height"]) for calibration in calibration_by_camera.values()
    }
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError("Stereo camera image dimensions differ")
    width = widths.pop()
    height = heights.pop()

    stereo = yaml.safe_load(
        (calibration_dir / "calib_cam0_to_cam1.yaml").read_text()
    )
    cam0_t_cam1 = np.asarray(
        stereo["transform"]["matrix_4x4"], dtype=np.float64
    )
    if cam0_t_cam1.shape != (4, 4):
        raise ValueError("T_cam0_cam1 must be 4x4")
    if not np.allclose(cam0_t_cam1[:3, :3], np.eye(3), atol=1.0e-12):
        raise ValueError(
            "Identity optical orientation was requested but stereo rotation is non-identity"
        )
    if not np.allclose(cam0_t_cam1[1:3, 3], 0.0, atol=1.0e-12):
        raise ValueError("Stereo baseline is not horizontal")
    baseline = abs(float(cam0_t_cam1[0, 3]))
    if baseline <= 0.0:
        raise ValueError("Stereo baseline must be positive")

    focal = (0.5 * (width - 1)) / np.tan(
        np.deg2rad(args.horizontal_fov_deg / 2.0)
    )
    virtual_k = np.array(
        [
            [focal, 0.0, 0.5 * (width - 1)],
            [0.0, focal, 0.5 * (height - 1)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    identity_rotation = np.eye(3, dtype=np.float64)
    maps = {
        camera: cv2.fisheye.initUndistortRectifyMap(
            source_k_by_camera[camera],
            source_d_by_camera[camera],
            identity_rotation,
            virtual_k,
            (width, height),
            cv2.CV_32FC1,
        )
        for camera in ("cam0", "cam1")
    }
    map_valid_ratios = {}
    for camera, (map_x, map_y) in maps.items():
        valid = (
            (map_x >= 0.0)
            & (map_x <= width - 1)
            & (map_y >= 0.0)
            & (map_y <= height - 1)
        )
        map_valid_ratios[camera] = float(valid.mean())
    if min(map_valid_ratios.values()) < 0.995:
        raise ValueError(
            "Virtual view extends outside source images: "
            f"{map_valid_ratios}"
        )

    records = {
        int(record["tick"]): record
        for record in load_jsonl(source / "manifest.jsonl")
    }
    for directory in ("rgb", "stereo_right", "preview"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    frames = []
    integrity_frames = []
    write_options = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    for output_index, source_index in enumerate(args.source_indices):
        if source_index not in records:
            raise IndexError(f"Source frame {source_index} is absent")
        record = records[source_index]
        descriptors = {
            image["camera"]: image for image in record["images"]
        }
        paths = {
            camera: (source / descriptors[camera]["path"]).resolve()
            for camera in ("cam0", "cam1")
        }
        images = {
            camera: cv2.imread(str(path), cv2.IMREAD_COLOR)
            for camera, path in paths.items()
        }
        if (
            images["cam0"] is None
            or images["cam1"] is None
            or images["cam0"].shape != images["cam1"].shape
            or images["cam0"].shape[:2] != (height, width)
        ):
            raise RuntimeError(f"Invalid stereo images for source {source_index}")
        virtual = {
            camera: cv2.remap(
                images[camera],
                maps[camera][0],
                maps[camera][1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            for camera in ("cam0", "cam1")
        }
        left_output = output / "rgb" / f"{output_index:08d}.png"
        right_output = output / "stereo_right" / f"{output_index:08d}.png"
        if not cv2.imwrite(str(left_output), virtual["cam0"], write_options):
            raise RuntimeError(f"Failed to save {left_output}")
        if not cv2.imwrite(str(right_output), virtual["cam1"], write_options):
            raise RuntimeError(f"Failed to save {right_output}")

        preview_scale = 0.5
        preview_size = (
            int(round(width * preview_scale)),
            int(round(height * preview_scale)),
        )
        preview_panels = [
            cv2.resize(
                draw_label(images["cam0"], "source cam0 Kannala-Brandt"),
                preview_size,
                interpolation=cv2.INTER_AREA,
            ),
            cv2.resize(
                draw_label(images["cam1"], "source cam1 Kannala-Brandt"),
                preview_size,
                interpolation=cv2.INTER_AREA,
            ),
            cv2.resize(
                draw_label(virtual["cam0"], "virtual pinhole cam0, R=I"),
                preview_size,
                interpolation=cv2.INTER_AREA,
            ),
            cv2.resize(
                draw_label(virtual["cam1"], "virtual pinhole cam1, R=I"),
                preview_size,
                interpolation=cv2.INTER_AREA,
            ),
        ]
        preview = np.vstack(
            (
                np.hstack(preview_panels[:2]),
                np.hstack(preview_panels[2:]),
            )
        )
        preview_path = output / "preview" / f"{source_index:06d}.png"
        if not cv2.imwrite(str(preview_path), preview, write_options):
            raise RuntimeError(f"Failed to save {preview_path}")

        cam0_time = int(descriptors["cam0"]["sensor_time_ns"])
        cam1_time = int(descriptors["cam1"]["sensor_time_ns"])
        frames.append(
            {
                "idx": output_index,
                "source_idx": source_index,
                "cam0_source_idx": source_index,
                "cam1_source_idx": source_index,
                "pose_row": output_index,
                "cam0": str(left_output.resolve()),
                "cam1": str(right_output.resolve()),
                "reference_camera": "cam0",
                "partner_camera": "cam1",
                "timestamp": output_index / 30.0,
                "cam0_sensor_time_ns": cam0_time,
                "cam1_sensor_time_ns": cam1_time,
                "sensor_time_ns": cam0_time,
                "pose_sensor_time_ns": cam0_time,
                "stereo_delta_ms": abs(cam0_time - cam1_time) / 1.0e6,
                "image_geometry_operation": (
                    "calibration-only Kannala-Brandt to pinhole remap"
                ),
            }
        )
        integrity_frames.append(
            {
                "output_index": output_index,
                "source_index": source_index,
                "source_cam0": {
                    "path": str(paths["cam0"]),
                    "sha256": sha256(paths["cam0"]),
                },
                "source_cam1": {
                    "path": str(paths["cam1"]),
                    "sha256": sha256(paths["cam1"]),
                },
                "output_cam0": {
                    "path": str(left_output.resolve()),
                    "sha256": sha256(left_output),
                },
                "output_cam1": {
                    "path": str(right_output.resolve()),
                    "sha256": sha256(right_output),
                },
                "stereo_delta_ms": abs(cam0_time - cam1_time) / 1.0e6,
            }
        )

    camera_info = {
        "width": width,
        "height": height,
        "model": "pinhole",
        "intrinsics": virtual_k.tolist(),
        "distortion": [0.0, 0.0, 0.0, 0.0],
        "fx": float(virtual_k[0, 0]),
        "fy": float(virtual_k[1, 1]),
        "cx": float(virtual_k[0, 2]),
        "cy": float(virtual_k[1, 2]),
        "baseline": baseline,
        "reference_camera": "cam0",
        "partner_camera": "cam1",
    }
    tick_index = {
        "source": str(source),
        "projection_model": "pinhole",
        "source_projection_model": "kannala_brandt",
        "pose_frame": "map",
        "reference_camera": "cam0",
        "partner_camera": "cam1",
        "model_input_order": ["cam0", "cam1"],
        "fx": camera_info["fx"],
        "fy": camera_info["fy"],
        "cx": camera_info["cx"],
        "cy": camera_info["cy"],
        "baseline": baseline,
        "width": width,
        "height": height,
        "recommended_max_depth_m": args.maximum_depth_m,
        "image_policy": (
            "Both source Kannala-Brandt images remapped into one centred "
            "virtual pinhole camera with identity optical rotation"
        ),
        "frames": frames,
    }
    integrity = {
        "contract": "calibration-only virtual pinhole stereo validation",
        "source_dataset": str(source),
        "source_projection_model": "kannala_brandt",
        "output_projection_model": "pinhole",
        "requested_horizontal_fov_deg": args.horizontal_fov_deg,
        "effective_horizontal_fov_deg": float(
            np.degrees(
                np.arctan2(virtual_k[0, 2], virtual_k[0, 0])
                + np.arctan2(
                    (width - 1) - virtual_k[0, 2], virtual_k[0, 0]
                )
            )
        ),
        "source_K": {
            camera: source_k_by_camera[camera].tolist()
            for camera in ("cam0", "cam1")
        },
        "source_D": {
            camera: source_d_by_camera[camera].reshape(-1).tolist()
            for camera in ("cam0", "cam1")
        },
        "virtual_K": virtual_k.tolist(),
        "cam0_T_cam1": cam0_t_cam1.tolist(),
        "baseline_m": baseline,
        "optical_rotation_source_to_virtual": identity_rotation.tolist(),
        "extra_rotation_applied": False,
        "lidar_used_for_remap": False,
        "image_derived_homography_used": False,
        "crop": False,
        "resize": False,
        "output_dimensions_equal_source": True,
        "remap_valid_ratios": map_valid_ratios,
        "frames": integrity_frames,
    }
    (output / "camera_info.json").write_text(
        json.dumps(camera_info, indent=2) + "\n"
    )
    (output / "tick_index.json").write_text(
        json.dumps(tick_index, indent=2) + "\n"
    )
    (output / "input_integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "frame_count": len(frames),
                "virtual_K": virtual_k.tolist(),
                "remap_valid_ratios": map_valid_ratios,
                "extra_rotation_applied": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
