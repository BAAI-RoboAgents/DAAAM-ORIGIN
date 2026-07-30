#!/usr/bin/env python3
"""Apply a fixed right-only homography to selected raw stereo frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", required=True, type=Path)
    parser.add_argument("--camera-info", required=True, type=Path)
    parser.add_argument("--rectification-report", required=True, type=Path)
    parser.add_argument("--source-indices", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {args.output}")
    source_indices = [int(value) for value in args.source_indices.split(",")]
    if not source_indices or len(set(source_indices)) != len(source_indices):
        raise ValueError("source-indices must contain unique comma-separated integers")
    records = load_jsonl(args.raw_dataset / "manifest.jsonl")
    camera_info = json.loads(args.camera_info.read_text())
    rectification = json.loads(args.rectification_report.read_text())
    homography = np.asarray(
        rectification["right_source_to_rectified_homography"],
        dtype=np.float64,
    )
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("Rectification report contains an invalid homography")
    width = int(camera_info["width"])
    height = int(camera_info["height"])

    rgb_directory = args.output / "rgb"
    right_directory = args.output / "stereo_right"
    rgb_directory.mkdir(parents=True)
    right_directory.mkdir()
    frames = []
    integrity = []
    for output_index, source_index in enumerate(source_indices):
        if not 0 <= source_index < len(records):
            raise IndexError(f"Source index {source_index} is outside the manifest")
        record = records[source_index]
        left_descriptor = next(
            image for image in record["images"] if image["camera"] == "cam0"
        )
        right_descriptor = next(
            image for image in record["images"] if image["camera"] == "cam1"
        )
        left_path = args.raw_dataset / left_descriptor["path"]
        right_path = args.raw_dataset / right_descriptor["path"]
        right_bgr = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if right_bgr is None or right_bgr.shape[:2] != (height, width):
            raise FileNotFoundError(f"Invalid right image for source {source_index}")
        output_left = rgb_directory / f"{output_index:08d}.png"
        output_right = right_directory / f"{output_index:08d}.png"
        shutil.copyfile(left_path, output_left)
        corrected_right = cv2.warpPerspective(
            right_bgr,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        if not cv2.imwrite(str(output_right), corrected_right):
            raise RuntimeError(f"Could not write corrected source {source_index}")
        left_timestamp = int(left_descriptor["sensor_time_ns"])
        right_timestamp = int(right_descriptor["sensor_time_ns"])
        frames.append(
            {
                "idx": output_index,
                "source_idx": source_index,
                "cam0_source_idx": source_index,
                "cam1_source_idx": source_index,
                "pose_row": output_index,
                "cam0": str(output_left.resolve()),
                "cam1": str(output_right.resolve()),
                "timestamp": float(output_index),
                "cam0_sensor_time_ns": left_timestamp,
                "cam1_sensor_time_ns": right_timestamp,
                "sensor_time_ns": left_timestamp,
                "pose_sensor_time_ns": left_timestamp,
                "stereo_delta_ms": (right_timestamp - left_timestamp) / 1e6,
            }
        )
        integrity.append(
            {
                "source_index": source_index,
                "raw_left_sha256": sha256(left_path),
                "output_left_sha256": sha256(output_left),
                "left_byte_identical": sha256(left_path) == sha256(output_left),
            }
        )

    (args.output / "camera_info.json").write_text(
        json.dumps(camera_info, ensure_ascii=False, indent=2) + "\n"
    )
    tick_index = {
        "source": str(args.raw_dataset.resolve()),
        "projection_model": "pinhole",
        "pose_frame": "map",
        "fx": float(camera_info["fx"]),
        "fy": float(camera_info["fy"]),
        "cx": float(camera_info["cx"]),
        "cy": float(camera_info["cy"]),
        "baseline": float(camera_info["baseline"]),
        "width": width,
        "height": height,
        "recommended_max_depth_m": 5.0,
        "frames": frames,
    }
    (args.output / "tick_index.json").write_text(
        json.dumps(tick_index, ensure_ascii=False, indent=2) + "\n"
    )
    report = {
        "raw_dataset": str(args.raw_dataset.resolve()),
        "rectification_report": str(args.rectification_report.resolve()),
        "right_source_to_rectified_homography": homography.tolist(),
        "source_indices": source_indices,
        "left_image_policy": "byte-for-byte original PNG",
        "extra_left_rotation_applied": False,
        "integrity": integrity,
    }
    (args.output / "validation_dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
