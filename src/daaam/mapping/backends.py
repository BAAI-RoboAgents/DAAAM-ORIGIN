"""Static map backend adapters for the realtime fusion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


def matrix_to_xyzw(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera transform must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("camera transform rotation must be orthonormal")

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


class HydraStaticMapBackend:
    """Feed static-only RGB-D frames into Hydra without importing ROS at startup."""

    def __init__(
        self,
        hydra_config_path: Path | str,
        output_dir: Path | str,
        *,
        labelspace_path: Optional[Path | str] = None,
        labelspace_colors: Optional[Path | str] = None,
        maximum_depth_m: float = 10.0,
        integration_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.hydra_config_path = Path(hydra_config_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.labelspace_path = (
            None if labelspace_path is None else Path(labelspace_path).resolve()
        )
        self.labelspace_colors = (
            None if labelspace_colors is None else Path(labelspace_colors).resolve()
        )
        if not self.hydra_config_path.is_file():
            raise FileNotFoundError(self.hydra_config_path)
        for optional_path in (self.labelspace_path, self.labelspace_colors):
            if optional_path is not None and not optional_path.is_file():
                raise FileNotFoundError(optional_path)
        if maximum_depth_m <= 0.0:
            raise ValueError("maximum depth must be positive")
        self.maximum_depth_m = maximum_depth_m
        self._factory = integration_factory
        self._integration = None
        self._origin_time_ns: Optional[int] = None
        self._finalized = False
        self._frames_rejected = 0

    def initialize(self, width: int, height: int, intrinsics: np.ndarray) -> None:
        camera = np.asarray(intrinsics, dtype=np.float64)
        if width <= 0 or height <= 0 or camera.shape != (3, 3):
            raise ValueError("Hydra camera geometry is invalid")
        if self._factory is None:
            from daaam.hydra.integration import HydraIntegration

            factory = HydraIntegration
        else:
            factory = self._factory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._integration = factory(
            hydra_config_path=str(self.hydra_config_path),
            labelspace_path=(
                None if self.labelspace_path is None else str(self.labelspace_path)
            ),
            labelspace_colors=(
                None
                if self.labelspace_colors is None
                else str(self.labelspace_colors)
            ),
            output_dir=self.output_dir,
            zmq_url=None,
        )
        self._integration.initialize_camera(
            width=width,
            height=height,
            fx=float(camera[0, 0]),
            fy=float(camera[1, 1]),
            cx=float(camera[0, 2]),
            cy=float(camera[1, 2]),
            min_range=0.1,
            max_range=self.maximum_depth_m,
        )
        if not self._integration.initialize_pipeline():
            raise RuntimeError("Hydra pipeline initialization failed")

    def integrate(
        self,
        *,
        sensor_time_ns: int,
        rgb_image: np.ndarray,
        static_depth_m: np.ndarray,
        world_T_camera: np.ndarray,
        semantic_labels: Optional[np.ndarray] = None,
    ) -> None:
        if self._integration is None:
            raise RuntimeError("Hydra map backend is not initialized")
        if sensor_time_ns <= 0:
            raise ValueError("Hydra frame time must be absolute nanoseconds")
        rgb = np.asarray(rgb_image)
        depth = np.asarray(static_depth_m, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError("Hydra RGB-D dimensions do not match")
        if semantic_labels is None:
            labels = np.zeros(depth.shape, dtype=np.int32)
        else:
            labels = np.asarray(semantic_labels, dtype=np.int32)
            if labels.shape != depth.shape:
                raise ValueError("Hydra semantic labels do not match depth")
        if self._origin_time_ns is None:
            self._origin_time_ns = sensor_time_ns
        world_transform = np.asarray(world_T_camera, dtype=np.float64)
        quaternion = matrix_to_xyzw(world_transform)
        pose = np.r_[world_transform[:3, 3], quaternion]
        success = self._integration.process_frame(
            timestamp=(sensor_time_ns - self._origin_time_ns) / 1.0e9,
            timestamp_ns=sensor_time_ns,
            rgb_image=rgb,
            depth_image=depth,
            semantic_labels=labels,
            transform=pose,
        )
        if not success:
            self._frames_rejected += 1
            raise RuntimeError("Hydra rejected the static RGB-D frame")

    def finalize(self) -> None:
        if self._integration is None or self._finalized:
            return
        if not self._integration.save_results(self.output_dir):
            raise RuntimeError("Hydra failed to save the realtime map")
        self._finalized = True

    def stats(self) -> dict:
        values = (
            {} if self._integration is None else dict(self._integration.get_stats())
        )
        processing_times = values.pop("processing_times", [])
        if processing_times:
            milliseconds = np.asarray(processing_times, dtype=np.float64) * 1000.0
            values["processing_time_ms"] = {
                "samples": int(milliseconds.size),
                "p50": float(np.percentile(milliseconds, 50)),
                "p95": float(np.percentile(milliseconds, 95)),
                "p99": float(np.percentile(milliseconds, 99)),
                "maximum": float(np.max(milliseconds)),
            }
        for key, value in list(values.items()):
            if isinstance(value, np.generic):
                values[key] = value.item()
            elif isinstance(value, np.ndarray):
                values[key] = value.tolist()
        values.update(
            {
                "backend": "hydra",
                "frames_rejected": self._frames_rejected,
                "finalized": self._finalized,
                "output_dir": str(self.output_dir),
            }
        )
        return values

    def map_metrics(self) -> dict:
        if not self._finalized:
            raise RuntimeError("Hydra map must be finalized before mesh analysis")
        from daaam.quality.mesh import analyze_ascii_ply_mesh

        return analyze_ascii_ply_mesh(self.output_dir / "backend" / "mesh.ply")

    def close(self, *, finalize: bool = True) -> None:
        if self._integration is None:
            return
        integration = self._integration
        try:
            if finalize and not self._finalized:
                self.finalize()
        finally:
            try:
                integration.shutdown()
            finally:
                self._integration = None
