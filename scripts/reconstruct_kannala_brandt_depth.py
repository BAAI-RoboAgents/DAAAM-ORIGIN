#!/usr/bin/env python3
"""Re-triangulate saved horizontal disparity as Kannala-Brandt camera rays.

This is a diagnostic control for disparity inferred directly on fisheye images.
It does not alter either model input image and does not use LiDAR.  It cannot
repair correspondence errors caused by curved epipolar lines; it only tests
whether the pinhole ``fx * baseline / disparity`` conversion was the dominant
metric-depth error.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", required=True, type=Path)
    parser.add_argument("--depth-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-depth-m", type=float, default=0.25)
    parser.add_argument("--maximum-depth-m", type=float, default=30.0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def invert_kannala_brandt_radius(
    distorted_theta: np.ndarray, distortion: np.ndarray
) -> np.ndarray:
    coefficients = np.zeros(4, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if distortion.size > 4:
        raise ValueError("Kannala-Brandt supports at most four coefficients")
    coefficients[: distortion.size] = distortion
    theta = distorted_theta.copy()
    for _ in range(10):
        theta2 = theta * theta
        polynomial = (
            1.0
            + coefficients[0] * theta2
            + coefficients[1] * theta2**2
            + coefficients[2] * theta2**3
            + coefficients[3] * theta2**4
        )
        value = theta * polynomial - distorted_theta
        derivative = (
            1.0
            + 3.0 * coefficients[0] * theta2
            + 5.0 * coefficients[1] * theta2**2
            + 7.0 * coefficients[2] * theta2**3
            + 9.0 * coefficients[3] * theta2**4
        )
        theta -= np.divide(
            value,
            derivative,
            out=np.zeros_like(value),
            where=np.abs(derivative) > 1.0e-12,
        )
    return theta


def kannala_brandt_rays(
    u: np.ndarray,
    v: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: np.ndarray,
) -> np.ndarray:
    mx = (u - cx) / fx
    my = (v - cy) / fy
    distorted_theta = np.hypot(mx, my)
    theta = invert_kannala_brandt_radius(distorted_theta, distortion)
    radial_scale = np.divide(
        np.sin(theta),
        distorted_theta,
        out=np.ones_like(theta),
        where=distorted_theta > 1.0e-12,
    )
    rays = np.stack(
        (
            mx * radial_scale,
            my * radial_scale,
            np.cos(theta),
        ),
        axis=-1,
    )
    norm = np.linalg.norm(rays, axis=-1, keepdims=True)
    return np.divide(rays, norm, out=np.zeros_like(rays), where=norm > 0.0)


def triangulate_depth(
    disparity: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: np.ndarray,
    reference_t_partner: np.ndarray,
) -> np.ndarray:
    height, width = disparity.shape
    v, u = np.indices((height, width), dtype=np.float64)
    partner_u = u - disparity.astype(np.float64)
    valid = (
        np.isfinite(disparity)
        & (disparity > 0.0)
        & (partner_u >= 0.0)
        & (partner_u <= width - 1)
    )
    reference_rays = kannala_brandt_rays(
        u, v, fx, fy, cx, cy, distortion
    )
    partner_rays = kannala_brandt_rays(
        partner_u, v, fx, fy, cx, cy, distortion
    )
    rotation = reference_t_partner[:3, :3]
    translation = reference_t_partner[:3, 3]
    partner_rays_reference = partner_rays @ rotation.T
    ray_dot = np.sum(reference_rays * partner_rays_reference, axis=-1)
    denominator = 1.0 - ray_dot * ray_dot
    reference_dot_translation = reference_rays @ translation
    partner_dot_translation = partner_rays_reference @ translation
    reference_range = np.divide(
        reference_dot_translation
        - ray_dot * partner_dot_translation,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=np.abs(denominator) > 1.0e-12,
    )
    partner_range = np.divide(
        ray_dot * reference_dot_translation
        - partner_dot_translation,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=np.abs(denominator) > 1.0e-12,
    )
    depth = reference_range * reference_rays[:, :, 2]
    ray_gap = np.linalg.norm(
        reference_range[:, :, None] * reference_rays
        - (
            translation
            + partner_range[:, :, None] * partner_rays_reference
        ),
        axis=-1,
    )
    valid &= (
        np.isfinite(depth)
        & np.isfinite(ray_gap)
        & (reference_range > 0.0)
        & (partner_range > 0.0)
        & (ray_gap <= 0.02)
    )
    output = np.full(disparity.shape, np.nan, dtype=np.float32)
    output[valid] = depth[valid].astype(np.float32)
    return output


def colorize_depth(
    depth: np.ndarray, minimum_depth_m: float, maximum_depth_m: float
) -> np.ndarray:
    valid = (
        np.isfinite(depth)
        & (depth >= minimum_depth_m)
        & (depth <= maximum_depth_m)
    )
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = (
        depth[valid] - minimum_depth_m
    ) / (maximum_depth_m - minimum_depth_m)
    encoded = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(encoded, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def main() -> None:
    args = parse_args()
    raw_dataset = args.raw_dataset.resolve()
    depth_dataset = args.depth_dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if not 0.0 < args.minimum_depth_m < args.maximum_depth_m:
        raise ValueError("Depth range is invalid")

    tick_index = load_json(depth_dataset / "tick_index.json")
    camera = load_json(depth_dataset / "camera_info.json")
    reference_camera = str(
        tick_index.get(
            "reference_camera", camera.get("reference_camera", "cam0")
        )
    )
    if reference_camera not in {"cam0", "cam1"}:
        raise ValueError(f"Unsupported reference camera: {reference_camera}")
    calibration_dir = raw_dataset / "calibrations" / "000000"
    reference_calibration = yaml.safe_load(
        (
            calibration_dir
            / f"calib_{reference_camera}_intrinsics.yaml"
        ).read_text()
    )["intrinsics"]
    if reference_calibration["distortion_model"] != "kannala_brandt":
        raise ValueError("Source reference camera is not Kannala-Brandt")
    distortion = np.asarray(reference_calibration["D"], dtype=np.float64)
    stereo = yaml.safe_load(
        (calibration_dir / "calib_cam0_to_cam1.yaml").read_text()
    )
    cam0_t_cam1 = np.asarray(
        stereo["transform"]["matrix_4x4"], dtype=np.float64
    )
    reference_t_partner = (
        cam0_t_cam1
        if reference_camera == "cam0"
        else np.linalg.inv(cam0_t_cam1)
    )

    product_directories = (
        "raw_disparity",
        "raw_depth_meter",
        "lr_consistency_error",
        "right_disparity",
        "depth",
        "raw_depth_visualization_5m",
        "raw_depth_visualization_30m",
        "raw_depth_overlay_5m",
    )
    output.mkdir(parents=True)
    for name in product_directories:
        (output / name).mkdir()
    frame_reports = []
    for frame in tick_index["frames"]:
        output_index = int(frame["idx"])
        name = f"{output_index:08d}"
        disparity = np.load(
            depth_dataset / "raw_disparity" / f"{name}.npy",
            allow_pickle=False,
        )
        source_filtered_mm = cv2.imread(
            str(depth_dataset / "depth" / f"{name}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        rgb = cv2.imread(str(frame["cam0"]), cv2.IMREAD_COLOR)
        if (
            source_filtered_mm is None
            or rgb is None
            or source_filtered_mm.shape != disparity.shape
            or rgb.shape[:2] != disparity.shape
        ):
            raise ValueError(f"Input shape mismatch for output frame {output_index}")
        depth = triangulate_depth(
            disparity,
            float(camera["fx"]),
            float(camera["fy"]),
            float(camera["cx"]),
            float(camera["cy"]),
            distortion,
            reference_t_partner,
        )
        valid_5m = (
            np.isfinite(depth)
            & (depth >= args.minimum_depth_m)
            & (depth <= 5.0)
        )
        valid_30m = (
            np.isfinite(depth)
            & (depth >= args.minimum_depth_m)
            & (depth <= args.maximum_depth_m)
        )
        filtered_valid = valid_30m & (source_filtered_mm > 0)
        filtered_mm = np.zeros(depth.shape, dtype=np.uint16)
        filtered_mm[filtered_valid] = np.rint(
            np.minimum(depth[filtered_valid], 65.535) * 1000.0
        ).astype(np.uint16)
        color_5m = colorize_depth(depth, args.minimum_depth_m, 5.0)
        color_30m = colorize_depth(
            depth, args.minimum_depth_m, args.maximum_depth_m
        )
        overlay = rgb.copy()
        overlay[valid_5m] = np.rint(
            0.45 * rgb[valid_5m] + 0.55 * color_5m[valid_5m]
        ).astype(np.uint8)

        np.save(output / "raw_disparity" / f"{name}.npy", disparity)
        np.save(output / "raw_depth_meter" / f"{name}.npy", depth)
        for product in ("lr_consistency_error", "right_disparity"):
            shutil.copy2(
                depth_dataset / product / f"{name}.npy",
                output / product / f"{name}.npy",
            )
        if not cv2.imwrite(str(output / "depth" / f"{name}.png"), filtered_mm):
            raise RuntimeError("Failed to save filtered KB depth")
        if not cv2.imwrite(
            str(output / "raw_depth_visualization_5m" / f"{name}.png"),
            color_5m,
        ):
            raise RuntimeError("Failed to save 5 m KB visualization")
        if not cv2.imwrite(
            str(output / "raw_depth_visualization_30m" / f"{name}.png"),
            color_30m,
        ):
            raise RuntimeError("Failed to save 30 m KB visualization")
        if not cv2.imwrite(
            str(output / "raw_depth_overlay_5m" / f"{name}.png"), overlay
        ):
            raise RuntimeError("Failed to save KB overlay")
        frame_reports.append(
            {
                "output_index": output_index,
                "source_index": int(frame["source_idx"]),
                "valid_depth_within_5m_ratio": float(valid_5m.mean()),
                "valid_depth_within_30m_ratio": float(valid_30m.mean()),
                "filtered_valid_ratio": float(filtered_valid.mean()),
            }
        )

    output_tick = tick_index.copy()
    output_tick["depth_reconstruction_model"] = (
        "Kannala-Brandt ray triangulation from horizontal model disparity"
    )
    output_camera = camera.copy()
    output_camera["model"] = "kannala_brandt"
    output_camera["distortion"] = distortion.tolist()
    output_camera["source_disparity_dataset"] = str(depth_dataset)
    (output / "tick_index.json").write_text(
        json.dumps(output_tick, indent=2) + "\n"
    )
    (output / "camera_info.json").write_text(
        json.dumps(output_camera, indent=2) + "\n"
    )
    source_integrity = depth_dataset / "input_integrity.json"
    if source_integrity.is_file():
        shutil.copy2(source_integrity, output / "input_integrity.json")
    report = {
        "contract": (
            "Camera-only diagnostic; original images are unchanged and LiDAR "
            "is not used. Horizontal disparity is re-triangulated as "
            "Kannala-Brandt rays."
        ),
        "source_depth_dataset": str(depth_dataset),
        "output": str(output),
        "reference_camera": reference_camera,
        "distortion_model": reference_calibration["distortion_model"],
        "distortion": distortion.tolist(),
        "reference_T_partner": reference_t_partner.tolist(),
        "frames": frame_reports,
        "aggregate": {
            "mean_valid_depth_within_5m_ratio": float(
                np.mean(
                    [
                        frame["valid_depth_within_5m_ratio"]
                        for frame in frame_reports
                    ]
                )
            ),
            "mean_valid_depth_within_30m_ratio": float(
                np.mean(
                    [
                        frame["valid_depth_within_30m_ratio"]
                        for frame in frame_reports
                    ]
                )
            ),
        },
        "interpretation_guard": (
            "This conversion only corrects ray geometry. It cannot make a "
            "horizontal-disparity network follow curved fisheye epipolar lines."
        ),
    }
    (output / "kannala_brandt_reconstruction.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
