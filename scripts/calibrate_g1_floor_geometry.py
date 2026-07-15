#!/usr/bin/env python3
"""Calibrate G1 image-frame pitch and stereo scale from a floor sample."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit the floor in a small nominal-depth batch, align the image "
            "frame with gravity, and correct the effective stereo baseline."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--frame-count", type=int, default=5)
    parser.add_argument("--floor-world-z-m", type=float, default=0.0)
    parser.add_argument("--roi-x-min", type=float, default=0.0)
    parser.add_argument("--roi-x-max", type=float, default=0.52)
    parser.add_argument("--roi-y-min", type=float, default=0.68)
    parser.add_argument("--roi-y-max", type=float, default=1.0)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--residual-threshold", type=float, default=0.02)
    parser.add_argument("--max-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_fraction(name: str, value: float):
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], found {value}")


def ransac_plane(design, inverse_depth, threshold, max_trials, seed):
    rng = np.random.default_rng(seed)
    best_mask = None
    best_error = np.inf
    for _ in range(max_trials):
        sample = rng.choice(len(design), 3, replace=False)
        try:
            coefficients = np.linalg.solve(design[sample], inverse_depth[sample])
        except np.linalg.LinAlgError:
            continue
        residuals = np.abs(design @ coefficients - inverse_depth)
        mask = residuals < threshold
        inliers = int(mask.sum())
        if inliers < 3:
            continue
        error = float(np.median(residuals[mask]))
        if best_mask is None or inliers > int(best_mask.sum()) or (
            inliers == int(best_mask.sum()) and error < best_error
        ):
            best_mask = mask
            best_error = error
    if best_mask is None:
        raise RuntimeError("Floor-plane RANSAC failed")

    for _ in range(3):
        coefficients = np.linalg.lstsq(
            design[best_mask], inverse_depth[best_mask], rcond=None
        )[0]
        residuals = np.abs(design @ coefficients - inverse_depth)
        best_mask = residuals < threshold
    return coefficients, best_mask, residuals


def main():
    args = parse_args()
    dataset = args.dataset.resolve()
    report_path = dataset / "floor_geometry_calibration.json"
    backup_pose_path = dataset / "pose" / "poses_before_floor_calibration.txt"
    if report_path.exists() or backup_pose_path.exists():
        raise RuntimeError(
            "Dataset is already floor-calibrated; rerun the pinhole preparation "
            "step before calibrating again."
        )
    if args.frame_count < 3:
        raise ValueError("--frame-count must be at least 3")
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be positive")
    for name in ("roi_x_min", "roi_x_max", "roi_y_min", "roi_y_max"):
        validate_fraction(f"--{name.replace('_', '-')}", getattr(args, name))
    if args.roi_x_min >= args.roi_x_max or args.roi_y_min >= args.roi_y_max:
        raise ValueError("Floor ROI bounds are empty")

    camera_path = dataset / "camera_info.json"
    tick_path = dataset / "tick_index.json"
    pose_path = dataset / "pose" / "poses.txt"
    camera = json.loads(camera_path.read_text())
    tick_index = json.loads(tick_path.read_text())
    if camera.get("model") != "pinhole" or tick_index.get("projection_model") != "pinhole":
        raise ValueError("Floor calibration requires a pinhole stereo dataset")

    depth_paths = sorted((dataset / "depth").glob("*.png"))[: args.frame_count]
    if len(depth_paths) < args.frame_count:
        raise FileNotFoundError(
            f"Need {args.frame_count} nominal depth frames, found {len(depth_paths)}"
        )
    poses = np.loadtxt(pose_path, dtype=np.float64).reshape(-1, 4, 4)
    frame_indices = [int(path.stem) for path in depth_paths]
    if max(frame_indices) >= len(poses):
        raise ValueError("Depth frame index exceeds pose count")

    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    width = int(camera["width"])
    height = int(camera["height"])
    x0 = int(round(args.roi_x_min * width))
    x1 = int(round(args.roi_x_max * width))
    y0 = int(round(args.roi_y_min * height))
    y1 = int(round(args.roi_y_max * height))
    max_depth = float(tick_index.get("recommended_max_depth_m", 20.0))

    design_parts = []
    inverse_depth_parts = []
    for path in depth_paths:
        depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None or depth_mm.dtype != np.uint16:
            raise ValueError(f"Expected uint16 depth image: {path}")
        depth = depth_mm.astype(np.float64) / 1000.0
        v, u = np.mgrid[
            y0:y1:args.sample_stride,
            x0:x1:args.sample_stride,
        ]
        z = depth[y0:y1:args.sample_stride, x0:x1:args.sample_stride]
        valid = (z >= 0.25) & (z < max_depth)
        design_parts.append(
            np.column_stack(
                (
                    (u[valid] - cx) / fx,
                    (v[valid] - cy) / fy,
                    np.ones(int(valid.sum())),
                )
            )
        )
        inverse_depth_parts.append(1.0 / z[valid])

    design = np.concatenate(design_parts)
    inverse_depth = np.concatenate(inverse_depth_parts)
    if len(design) < 10000:
        raise RuntimeError(f"Too few valid floor samples: {len(design)}")
    coefficients, inlier_mask, residuals = ransac_plane(
        design,
        inverse_depth,
        args.residual_threshold,
        args.max_trials,
        args.seed,
    )
    inlier_ratio = float(inlier_mask.mean())
    if inlier_ratio < 0.5:
        raise RuntimeError(
            f"Floor fit is unreliable: inlier ratio={inlier_ratio:.3f}"
        )

    # inv(z) = a*x/z + b*y/z + c implies a*x+b*y+c*z-1=0.
    normal_scale = float(np.linalg.norm(coefficients))
    floor_normal = coefficients / normal_scale
    plane_offset = -1.0 / normal_scale
    selected_poses = poses[frame_indices]
    current_up = np.mean(
        [pose[:3, :3].T @ np.array([0.0, 0.0, 1.0]) for pose in selected_poses],
        axis=0,
    )
    current_up /= np.linalg.norm(current_up)
    if np.dot(floor_normal, current_up) < 0.0:
        floor_normal *= -1.0
        plane_offset *= -1.0
    if plane_offset <= 0.0:
        raise RuntimeError(
            f"Fitted floor is not below the camera: offset={plane_offset:.3f}m"
        )

    camera_height = float(
        np.median(selected_poses[:, 2, 3]) - args.floor_world_z_m
    )
    depth_scale = camera_height / plane_offset
    if not 0.75 <= depth_scale <= 1.5:
        raise RuntimeError(
            "Stereo scale correction is implausible: "
            f"height={camera_height:.3f}m floor_distance={plane_offset:.3f}m "
            f"scale={depth_scale:.3f}"
        )
    source_baseline = float(camera["baseline"])
    effective_baseline = source_baseline * depth_scale

    correction, _ = Rotation.align_vectors(
        current_up.reshape(1, 3), floor_normal.reshape(1, 3)
    )
    correction_matrix = correction.as_matrix()
    correction_angle = float(np.rad2deg(correction.magnitude()))
    if correction_angle > 45.0:
        raise RuntimeError(
            f"Image-frame correction is too large: {correction_angle:.3f} deg"
        )
    corrected_poses = poses.copy()
    corrected_poses[:, :3, :3] = poses[:, :3, :3] @ correction_matrix

    backup_pose_path.write_text(pose_path.read_text())
    pose_path.write_text(
        "".join(
            " ".join(f"{value:.12g}" for value in pose.reshape(-1)) + "\n"
            for pose in corrected_poses
        )
    )
    camera["source_baseline"] = source_baseline
    camera["baseline"] = effective_baseline
    camera_path.write_text(json.dumps(camera, indent=2) + "\n")
    tick_index["source_baseline"] = source_baseline
    tick_index["baseline"] = effective_baseline
    tick_index["pose_composition"] += " @ tf_camera_T_image_camera"
    tick_path.write_text(json.dumps(tick_index, indent=2) + "\n")

    report = {
        "dataset": str(dataset),
        "frame_indices": frame_indices,
        "floor_roi_pixels": [x0, y0, x1, y1],
        "sample_count": len(design),
        "ransac_inlier_ratio": inlier_ratio,
        "ransac_median_inverse_depth_residual": float(
            np.median(residuals[inlier_mask])
        ),
        "floor_normal_image_frame": floor_normal.tolist(),
        "nominal_floor_distance_m": plane_offset,
        "camera_height_m": camera_height,
        "depth_scale": depth_scale,
        "source_baseline_m": source_baseline,
        "effective_baseline_m": effective_baseline,
        "tf_camera_R_image_camera": correction_matrix.tolist(),
        "tf_camera_R_image_camera_euler_xyz_deg": correction.as_euler(
            "xyz", degrees=True
        ).tolist(),
        "correction_angle_deg": correction_angle,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    for path in (dataset / "depth").glob("*.png"):
        path.unlink()
    (dataset / "foundation_stereo_run.json").unlink(missing_ok=True)
    print(
        f"Calibrated floor geometry from {len(frame_indices)} frames\n"
        f"  inliers={inlier_ratio:.3f}, floor distance={plane_offset:.3f}m, "
        f"camera height={camera_height:.3f}m\n"
        f"  image-frame correction={correction_angle:.3f} deg, "
        f"baseline={source_baseline:.6f}m -> {effective_baseline:.6f}m\n"
        "  nominal depth images removed; rerun FoundationStereo"
    )


if __name__ == "__main__":
    main()
