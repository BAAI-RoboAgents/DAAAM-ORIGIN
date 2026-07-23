"""Safety regressions for the G1 floor calibration report workflow."""

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


multiframe = load_script_module(
    "calibrate_g1_multiframe_floor_geometry_test",
    REPOSITORY_ROOT / "scripts" / "calibrate_g1_multiframe_floor_geometry.py",
)


class FloorGeometryCalibrationTests(unittest.TestCase):
    def test_multiframe_rotation_rejects_inconsistent_plane_normals(self):
        correction = Rotation.from_euler("xz", [7.0, 5.0], degrees=True)
        current_up = np.array([0.0, -1.0, 0.0])
        floor_normal = correction.inv().apply(current_up)
        records = [
            {
                "frame_index": index,
                "floor_normal_image_frame": floor_normal.tolist(),
                "current_up_image_frame": current_up.tolist(),
                "inlier_ratio": 0.5,
            }
            for index in range(30)
        ]
        records.extend(
            {
                "frame_index": 30 + index,
                "floor_normal_image_frame": [1.0, 0.0, 0.0],
                "current_up_image_frame": current_up.tolist(),
                "inlier_ratio": 0.5,
            }
            for index in range(5)
        )

        fitted, _, retained, residuals, *_ = multiframe.fit_shared_rotation(
            records,
            yaw_policy="minimal",
            minimum_yaw_observability_ratio=0.005,
            maximum_frame_angular_residual_deg=10.0,
        )

        self.assertEqual(int(retained.sum()), 30)
        self.assertTrue(np.all(retained[:30]))
        self.assertTrue(np.all(~retained[30:]))
        self.assertLess(float(np.max(residuals[:30])), 0.01)
        np.testing.assert_allclose(
            fitted.apply(floor_normal), current_up, atol=1.0e-6
        )

    def test_report_only_preserves_source_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory) / "dataset"
            (dataset / "depth").mkdir(parents=True)
            (dataset / "pose").mkdir()

            width, height = 640, 480
            fx = fy = 300.0
            cx, cy = 320.0, 240.0
            rows = np.arange(height, dtype=np.float64)[:, None]
            normalized_y = (rows - cy) / fy
            depth_m = np.zeros((height, width), dtype=np.float64)
            valid = normalized_y[:, 0] > 0.0
            depth_m[valid] = 1.5 / normalized_y[valid]
            depth_mm = np.rint(depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
            for index in range(5):
                self.assertTrue(
                    cv2.imwrite(
                        str(dataset / "depth" / f"{index:08d}.png"), depth_mm
                    )
                )

            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
            )
            pose[2, 3] = 1.5
            pose_text = "".join(
                " ".join(str(value) for value in pose.reshape(-1)) + "\n"
                for _ in range(5)
            )
            (dataset / "pose" / "poses.txt").write_text(pose_text)
            camera = {
                "model": "pinhole",
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "baseline": 0.06,
            }
            tick_index = {
                "projection_model": "pinhole",
                "recommended_max_depth_m": 5.0,
                "pose_composition": "map_T_camera",
            }
            camera_path = dataset / "camera_info.json"
            tick_path = dataset / "tick_index.json"
            camera_path.write_text(json.dumps(camera))
            tick_path.write_text(json.dumps(tick_index))
            before = {
                "pose": (dataset / "pose" / "poses.txt").read_bytes(),
                "camera": camera_path.read_bytes(),
                "tick": tick_path.read_bytes(),
                "depth": [path.read_bytes() for path in sorted((dataset / "depth").glob("*.png"))],
            }

            subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts/calibrate_g1_floor_geometry.py"),
                    "--dataset",
                    str(dataset),
                    "--report-only",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            report = json.loads(
                (dataset / "floor_geometry_calibration.json").read_text()
            )
            self.assertTrue(report["report_only"])
            self.assertAlmostEqual(report["depth_scale"], 1.0, delta=0.01)
            self.assertFalse(
                (dataset / "pose" / "poses_before_floor_calibration.txt").exists()
            )
            self.assertEqual(
                before["pose"], (dataset / "pose" / "poses.txt").read_bytes()
            )
            self.assertEqual(before["camera"], camera_path.read_bytes())
            self.assertEqual(before["tick"], tick_path.read_bytes())
            self.assertEqual(
                before["depth"],
                [path.read_bytes() for path in sorted((dataset / "depth").glob("*.png"))],
            )


if __name__ == "__main__":
    unittest.main()
