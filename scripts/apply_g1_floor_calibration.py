#!/usr/bin/env python3
"""Apply a validated fixed G1 floor/image-frame calibration to a dataset.

FoundationStereo depth is linear in the stereo baseline, so a nominal-depth
run can be converted exactly by scaling valid depths and clipping to the same
metric range. The fixed image-frame rotation is applied on the right of every
world_T_camera pose; absolute capture timestamps and selected frame provenance
remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply an existing G1 floor geometry calibration."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--calibration-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-depth-m", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_poses(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64).reshape(-1, 4, 4)
    if not np.isfinite(poses).all() or not np.allclose(
        poses[:, 3, :], [0.0, 0.0, 0.0, 1.0]
    ):
        raise ValueError(f"Invalid homogeneous poses in {path}")
    return poses


def prepare_output(source: Path, output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("rgb", "stereo_right"):
        source_directory = source / directory
        (output / directory).symlink_to(
            source_directory.resolve(), target_is_directory=True
        )
    (output / "depth").mkdir()
    (output / "pose").mkdir()
    shutil.copy2(
        source / "pose" / "pose_timestamps_ns.txt",
        output / "pose" / "pose_timestamps_ns.txt",
    )
    for name in ("source_manifest.json", "keyframe_selection_report.json"):
        source_path = source / name
        if source_path.exists():
            (output / name).symlink_to(source_path.resolve())
    foundation_report = source / "foundation_stereo_run.json"
    if foundation_report.exists():
        shutil.copy2(foundation_report, output / "foundation_stereo_nominal_run.json")


def main() -> None:
    args = parse_args()
    source = args.dataset.resolve()
    output = args.output.resolve()
    calibration_path = args.calibration_report.resolve()
    calibration = json.loads(calibration_path.read_text())
    camera = json.loads((source / "camera_info.json").read_text())
    tick_index = json.loads((source / "tick_index.json").read_text())
    poses = load_poses(source / "pose" / "poses.txt")
    frame_count = len(tick_index["frames"])
    depth_paths = sorted((source / "depth").glob("*.png"))
    if len(poses) != frame_count or len(depth_paths) != frame_count:
        raise ValueError("Pose, tick_index, and depth counts must agree")

    source_baseline = float(calibration["source_baseline_m"])
    effective_baseline = float(calibration["effective_baseline_m"])
    depth_scale = float(calibration["depth_scale"])
    if not math.isclose(float(camera["baseline"]), source_baseline, abs_tol=1.0e-9):
        raise ValueError(
            "Input baseline does not match the calibration source baseline: "
            f"{camera['baseline']} vs {source_baseline}"
        )
    if not math.isclose(
        effective_baseline / source_baseline, depth_scale, rel_tol=1.0e-9
    ):
        raise ValueError("Calibration baseline and depth scale disagree")
    correction = np.asarray(
        calibration["tf_camera_R_image_camera"], dtype=np.float64
    )
    if correction.shape != (3, 3) or not np.allclose(
        correction.T @ correction, np.eye(3), atol=1.0e-6
    ):
        raise ValueError("Calibration image-frame rotation is invalid")
    max_depth_m = args.max_depth_m or float(
        tick_index.get("recommended_max_depth_m", 3.0)
    )
    if max_depth_m <= 0.0:
        raise ValueError("Maximum depth must be positive")

    prepare_output(source, output, args.overwrite)
    corrected_poses = poses.copy()
    corrected_poses[:, :3, :3] = poses[:, :3, :3] @ correction
    (output / "pose" / "poses.txt").write_text(
        "".join(
            " ".join(f"{value:.12g}" for value in pose.reshape(-1)) + "\n"
            for pose in corrected_poses
        )
    )

    before_ratios = []
    after_ratios = []
    clipped_pixels = 0
    for ordinal, depth_path in enumerate(depth_paths, start=1):
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None or depth_mm.dtype != np.uint16:
            raise ValueError(f"Invalid uint16 depth image: {depth_path}")
        valid_before = depth_mm > 0
        scaled = np.rint(depth_mm.astype(np.float64) * depth_scale)
        clipped = valid_before & (scaled > max_depth_m * 1000.0)
        clipped_pixels += int(clipped.sum())
        scaled[~valid_before | clipped] = 0.0
        corrected_depth = scaled.astype(np.uint16)
        output_path = output / "depth" / depth_path.name
        if not cv2.imwrite(str(output_path), corrected_depth):
            raise RuntimeError(f"Failed to write {output_path}")
        before_ratios.append(float(valid_before.mean()))
        after_ratios.append(float((corrected_depth > 0).mean()))
        if ordinal % 100 == 0 or ordinal == frame_count:
            print(f"Calibrated {ordinal}/{frame_count} depth images", flush=True)

    camera["source_baseline"] = source_baseline
    camera["baseline"] = effective_baseline
    (output / "camera_info.json").write_text(json.dumps(camera, indent=2) + "\n")
    output_tick = copy.deepcopy(tick_index)
    output_tick["source_baseline"] = source_baseline
    output_tick["baseline"] = effective_baseline
    composition_suffix = " @ tf_camera_T_image_camera"
    if not output_tick.get("pose_composition", "").endswith(composition_suffix):
        output_tick["pose_composition"] = (
            output_tick.get("pose_composition", "") + composition_suffix
        )
    output_tick["floor_geometry_calibration"] = {
        "method": "apply_validated_fixed_calibration",
        "source_report": str(calibration_path),
        "application_report": "floor_calibration_application.json",
    }
    for frame in output_tick["frames"]:
        index = int(frame["idx"])
        frame["cam0"] = str(output / "rgb" / f"{index:08d}.png")
        frame["cam1"] = str(output / "stereo_right" / f"{index:08d}.png")
    (output / "tick_index.json").write_text(
        json.dumps(output_tick, indent=2) + "\n"
    )

    report = {
        "source_dataset": str(source),
        "output_dataset": str(output),
        "source_calibration_report": str(calibration_path),
        "frame_count": frame_count,
        "absolute_timestamps_preserved": True,
        "source_baseline_m": source_baseline,
        "effective_baseline_m": effective_baseline,
        "depth_scale": depth_scale,
        "max_depth_m": max_depth_m,
        "clipped_pixels": clipped_pixels,
        "valid_ratio_before_percentiles": np.percentile(
            before_ratios, [0, 5, 50, 95, 100]
        ).tolist(),
        "valid_ratio_after_percentiles": np.percentile(
            after_ratios, [0, 5, 50, 95, 100]
        ).tolist(),
        "tf_camera_R_image_camera": correction.tolist(),
    }
    (output / "floor_calibration_application.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
