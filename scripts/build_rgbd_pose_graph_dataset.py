#!/usr/bin/env python3
"""Shared dense RGB-D geometry helpers for pose-graph construction.

The loop-closure pipeline works in the OpenCV/Open3D optical frame (X right,
Y down, Z forward). Depth PNGs are uint16 millimetres and RGB images are
stored in OpenCV BGR order.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def create_intrinsic(
    camera: dict, image_scale: float
) -> tuple[o3d.camera.PinholeCameraIntrinsic, int, int]:
    """Scale a pinhole camera model to the dense-registration image size."""
    if not 0.0 < image_scale <= 1.0:
        raise ValueError("image_scale must be in (0, 1]")
    try:
        source_width = int(camera["width"])
        source_height = int(camera["height"])
        fx, fy, cx, cy = (
            float(camera[key]) for key in ("fx", "fy", "cx", "cy")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Camera metadata must contain pinhole width/height/fx/fy/cx/cy"
        ) from error
    if source_width <= 0 or source_height <= 0 or fx <= 0.0 or fy <= 0.0:
        raise ValueError("Camera dimensions and focal lengths must be positive")

    width = max(1, int(round(source_width * image_scale)))
    height = max(1, int(round(source_height * image_scale)))
    scale_x = width / source_width
    scale_y = height / source_height
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        fx * scale_x,
        fy * scale_y,
        cx * scale_x,
        cy * scale_y,
    )
    return intrinsic, width, height


class DenseCloudCache:
    """Load, downscale, and cache metric colored point clouds by frame index."""

    def __init__(
        self,
        dataset: Path,
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
        width: int,
        height: int,
        max_depth_m: float,
    ):
        if width <= 0 or height <= 0 or max_depth_m <= 0.0:
            raise ValueError("Dense cloud dimensions and maximum depth must be positive")
        self.dataset = dataset.resolve()
        self.intrinsic = intrinsic
        self.width = width
        self.height = height
        self.max_depth_m = max_depth_m
        self.clouds: dict[int, o3d.geometry.PointCloud] = {}

    def cloud(self, frame: int) -> o3d.geometry.PointCloud:
        if frame not in self.clouds:
            rgb_path = self.dataset / "rgb" / f"{frame:08d}.png"
            depth_path = self.dataset / "depth" / f"{frame:08d}.png"
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise FileNotFoundError(f"Missing RGB frame {frame}: {rgb_path}")
            if depth is None or depth.dtype != np.uint16:
                raise ValueError(f"Expected uint16 depth frame {frame}: {depth_path}")
            if bgr.shape[:2] != depth.shape:
                raise ValueError(f"RGB/depth dimensions differ for frame {frame}")
            if (bgr.shape[1], bgr.shape[0]) != (self.width, self.height):
                bgr = cv2.resize(
                    bgr, (self.width, self.height), interpolation=cv2.INTER_AREA
                )
                depth = cv2.resize(
                    depth,
                    (self.width, self.height),
                    interpolation=cv2.INTER_NEAREST,
                )

            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            depth = np.ascontiguousarray(depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb),
                o3d.geometry.Image(depth),
                depth_scale=1000.0,
                depth_trunc=self.max_depth_m,
                convert_rgb_to_intensity=False,
            )
            cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, self.intrinsic
            )
            if len(cloud.points) == 0:
                raise RuntimeError(f"Dense RGB-D cloud is empty for frame {frame}")
            self.clouds[frame] = prepare_dense_cloud(cloud)
        return self.clouds[frame]


def prepare_dense_cloud(
    cloud: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    """Match the fixed 2 cm cloud contract used by dense loop thresholds."""
    cloud = cloud.voxel_down_sample(0.02)
    if len(cloud.points) == 0:
        raise RuntimeError("Dense RGB-D cloud is empty after 2 cm voxel sampling")
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30)
    )
    return cloud


def multiscale_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Refine source-to-target registration from coarse to fine."""
    if len(source.points) == 0 or len(target.points) == 0:
        raise ValueError("ICP requires non-empty source and target point clouds")
    transform = np.asarray(initial_transform, dtype=np.float64).copy()
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Initial ICP transform must be a finite 4x4 matrix")

    levels = (
        (0.08, 0.30),
        (0.04, 0.16),
        (0.02, 0.08),
    )
    metrics: list[tuple[float, float]] = []
    for voxel_size, correspondence_distance in levels:
        source_level = source.voxel_down_sample(voxel_size)
        target_level = target.voxel_down_sample(voxel_size)
        if len(source_level.points) < 3 or len(target_level.points) < 3:
            raise RuntimeError(
                f"Too few points for {voxel_size:.3f}m ICP level"
            )
        normal_search = o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 3.0, max_nn=30
        )
        source_level.estimate_normals(normal_search)
        target_level.estimate_normals(normal_search)
        result = o3d.pipelines.registration.registration_icp(
            source_level,
            target_level,
            correspondence_distance,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=50
            ),
        )
        transform = result.transformation
        metrics.append((float(result.fitness), float(result.inlier_rmse)))
    return transform, metrics
