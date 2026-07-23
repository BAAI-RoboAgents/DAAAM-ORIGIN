"""Tests for reconstructing prepared RGB-D poses in a LiDAR map frame."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


alignment = load_script_module(
    "build_lidar_map_aligned_dataset_test",
    REPOSITORY_ROOT / "scripts/build_lidar_map_aligned_dataset.py",
)


def pose(x: float, yaw_deg: float = 0.0) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    matrix[0, 3] = x
    return matrix


class LidarMapAlignedDatasetTests(unittest.TestCase):
    def test_se3_interpolation_clamps_only_outside_samples(self):
        base_ns = 1_700_000_000_000_000_000
        timestamps = np.asarray([base_ns, base_ns + 10_000_000_000], dtype=np.int64)
        matrices = np.asarray([pose(0.0, 0.0), pose(10.0, 90.0)])
        targets = np.asarray(
            [base_ns - 1, base_ns + 5_000_000_000, base_ns + 10_000_000_001],
            dtype=np.int64,
        )
        interpolated, clamped = alignment.interpolate_transforms(
            timestamps, matrices, targets
        )
        self.assertEqual(clamped, 2)
        np.testing.assert_allclose(interpolated[:, 0, 3], [0.0, 5.0, 10.0])
        midpoint_yaw = Rotation.from_matrix(interpolated[1, :3, :3]).as_euler(
            "zyx", degrees=True
        )[0]
        self.assertAlmostEqual(midpoint_yaw, 45.0, places=8)

    def test_builds_zero_copy_dataset_with_lidar_map_camera_poses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            raw = root / "raw"
            lidar_map = root / "lidar_map"
            output = root / "aligned"
            (dataset / "pose").mkdir(parents=True)
            for name in ("rgb", "stereo_right", "depth", "depth_confidence"):
                (dataset / name).mkdir()
            (raw / "state/000000").mkdir(parents=True)
            (raw / "poses/dense_global/000000").mkdir(parents=True)
            lidar_map.mkdir()

            base_ns = 1_700_000_000_000_000_000
            target_offsets_s = [2, 10, 18]
            target_timestamps = [base_ns + value * 1_000_000_000 for value in target_offsets_s]
            frames = []
            for index, timestamp_ns in enumerate(target_timestamps):
                (dataset / "rgb" / f"{index:08d}.png").touch()
                (dataset / "stereo_right" / f"{index:08d}.png").touch()
                frames.append(
                    {
                        "idx": index,
                        "pose_row": index,
                        "cam0": str(dataset / "rgb" / f"{index:08d}.png"),
                        "cam1": str(dataset / "stereo_right" / f"{index:08d}.png"),
                        "sensor_time_ns": timestamp_ns,
                        "pose_sensor_time_ns": timestamp_ns,
                    }
                )
            (dataset / "tick_index.json").write_text(
                json.dumps(
                    {
                        "source": str(raw),
                        "pose_frame": "odom",
                        "frames": frames,
                    }
                )
            )
            (dataset / "pose/pose_timestamps_ns.txt").write_text(
                "".join(f"{value}\n" for value in target_timestamps)
            )
            alignment.write_pose_file(
                dataset / "pose/poses.txt", np.asarray([pose(99.0)] * len(frames))
            )

            odom_records = []
            aux_records = []
            for tick, offset_s in enumerate((0, 10, 20)):
                timestamp_ns = base_ns + offset_s * 1_000_000_000
                odom_records.append(
                    {
                        "tick": tick,
                        "sensor_time_ns": timestamp_ns,
                        "odom": {
                            "header": {"timestamp_ns": timestamp_ns},
                            "pose": {
                                "pose": {
                                    "position": {"x": offset_s / 10.0, "y": 0.0, "z": 0.0},
                                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                                }
                            },
                        },
                    }
                )
                aux_records.append(
                    {
                        "tick": tick,
                        "poses": {
                            "head_camera": {
                                "target_frame": "base_link",
                                "timestamp_ns": timestamp_ns,
                                "position": [0.25, 0.0, 1.0],
                                # The capture's mislabeled storage is wxyz.
                                "orientation_xyzw": [1.0, 0.0, 0.0, 0.0],
                            }
                        },
                    }
                )
            (raw / "state/000000/odom.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in odom_records)
            )
            (raw / "poses/dense_global/000000/aux_poses.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in aux_records)
            )

            map_T_odom = pose(10.0, 90.0)
            anchor_offsets_s = [5, 15]
            anchor_poses = [
                map_T_odom @ pose(offset_s / 10.0)
                for offset_s in anchor_offsets_s
            ]
            (lidar_map / "poses.txt").write_text(
                "".join(
                    " ".join(f"{value:.12g}" for value in matrix[:3, :].reshape(-1))
                    + "\n"
                    for matrix in anchor_poses
                )
            )
            (lidar_map / "times.txt").write_text(
                "".join(
                    f"{(base_ns + offset_s * 1_000_000_000) / 1e9:.6f}\n"
                    for offset_s in anchor_offsets_s
                )
            )
            pinhole_report = root / "pinhole.json"
            pinhole_report.write_text(
                json.dumps(
                    {
                        "camera_quaternion_order": "wxyz",
                        "original_camera_R_virtual_camera": np.eye(3).tolist(),
                    }
                )
            )
            floor_report = root / "floor.json"
            floor_report.write_text(
                json.dumps({"tf_camera_R_image_camera": np.eye(3).tolist()})
            )

            report = alignment.build_dataset(
                argparse.Namespace(
                    dataset=dataset,
                    raw_dataset=raw,
                    lidar_map=lidar_map,
                    pinhole_report=pinhole_report,
                    floor_calibration_report=floor_report,
                    output=output,
                    validate_lidar_cloud=False,
                    validation_frames=2,
                    validation_points_per_frame=10,
                    validation_threshold_m=0.15,
                    overwrite=False,
                )
            )

            actual = np.loadtxt(output / "pose/poses.txt").reshape(-1, 4, 4)
            expected = np.asarray(
                [
                    map_T_odom @ pose(offset_s / 10.0) @ alignment.transform(
                        [0.25, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]
                    )
                    for offset_s in target_offsets_s
                ]
            )
            np.testing.assert_allclose(actual, expected, atol=1.0e-10)
            self.assertTrue((output / "rgb").is_symlink())
            output_tick = json.loads((output / "tick_index.json").read_text())
            self.assertEqual(output_tick["pose_frame"], "lidar_map")
            self.assertTrue(output_tick["frames"][0]["cam0"].startswith(str(output)))
            self.assertEqual(report["propagation"]["targets_before_first_anchor"], 1)
            self.assertEqual(report["propagation"]["targets_after_last_anchor"], 1)
            self.assertLess(
                report["propagation"]["maximum_anchor_translation_error_m"], 1.0e-10
            )


if __name__ == "__main__":
    unittest.main()
