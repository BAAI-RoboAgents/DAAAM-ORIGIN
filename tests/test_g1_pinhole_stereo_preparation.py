"""Regression tests for fixed G1 stereo calibration preparation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preparation = load_script_module(
    "prepare_g1_pinhole_stereo_dataset_test",
    REPOSITORY_ROOT / "scripts" / "prepare_g1_pinhole_stereo_dataset.py",
)
synchronization = sys.modules["build_synchronized_stereo_dataset"]


class G1PinholeStereoPreparationTests(unittest.TestCase):
    def test_auto_projection_model_avoids_double_warping_rectified_pixels(self):
        records = [
            {
                "images": [
                    {"camera": "cam0", "path": "2d_rect/cam0/000000/000000.png"},
                    {"camera": "cam1", "path": "2d_rect/cam1/000000/000000.png"},
                ]
            }
        ]
        calibration = {
            "distortion_model": "kannala_brandt",
            "roi": {"do_rectify": True},
        }
        model, evidence = preparation.resolve_input_projection_model(
            "auto",
            records,
            calibration,
            np.zeros((4, 1)),
        )
        self.assertEqual(model, "pinhole_rectified")
        self.assertEqual(
            evidence["selection_source"],
            "rectified_path_zero_distortion_and_roi",
        )

        raw_records = [
            {
                "images": [
                    {"camera": "cam0", "path": "2d_raw/cam0/000000.png"},
                    {"camera": "cam1", "path": "2d_raw/cam1/000000.png"},
                ]
            }
        ]
        model, _ = preparation.resolve_input_projection_model(
            "auto",
            raw_records,
            calibration,
            np.zeros((4, 1)),
        )
        self.assertEqual(model, "kannala_brandt")

    def test_pose_coverage_filter_prevents_endpoint_clamping(self):
        matches = [(0, 0, 1), (1, 1, 2), (2, 2, 3)]
        retained, dropped, lower, upper = preparation.filter_matches_to_pose_coverage(
            matches,
            np.asarray([90, 150, 210], dtype=np.int64),
            [100, 200],
            [110, 220],
        )
        self.assertEqual(retained, [(1, 1, 2)])
        self.assertEqual(dropped, [(0, 0, 1), (2, 2, 3)])
        self.assertEqual((lower, upper), (110, 200))

    def test_odom_pose_sample_accepts_legacy_and_ros2_records(self):
        legacy = {
            "tick": 1,
            "odom": {
                "timestamp_ns": 123,
                "position": [1.0, 2.0, 3.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
        }
        ros2 = {
            "tick": 2,
            "sensor_time_ns": 456,
            "odom": {
                "header": {"timestamp_ns": 455},
                "pose": {
                    "pose": {
                        "position": {"x": 4.0, "y": 5.0, "z": 6.0},
                        "orientation": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                            "w": 1.0,
                        },
                    }
                },
            },
        }

        legacy_timestamp, legacy_position, legacy_orientation = (
            synchronization.odom_pose_sample(legacy)
        )
        ros2_timestamp, ros2_position, ros2_orientation = (
            synchronization.odom_pose_sample(ros2)
        )

        self.assertEqual(legacy_timestamp, 123)
        np.testing.assert_allclose(legacy_position, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(legacy_orientation, [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(ros2_timestamp, 455)
        np.testing.assert_allclose(ros2_position, [4.0, 5.0, 6.0])
        np.testing.assert_allclose(ros2_orientation, [0.0, 0.0, 0.0, 1.0])

    def test_map_pose_sample_requires_and_reads_map_to_base_contract(self):
        record = {
            "tick": 7,
            "sensor_time_ns": 999,
            "target_frame": "map",
            "source_frame": "base_link",
            "pose": {
                "target_frame": "map",
                "source_frame": "base_link",
                "timestamp_ns": 998,
                "position": [1.0, 2.0, 0.1],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
        timestamp, position, orientation = synchronization.map_pose_sample(record)
        self.assertEqual(timestamp, 998)
        np.testing.assert_allclose(position, [1.0, 2.0, 0.1])
        np.testing.assert_allclose(orientation, [0.0, 0.0, 0.0, 1.0])

        malformed = dict(record, target_frame="odom")
        with self.assertRaisesRegex(ValueError, "Malformed map pose"):
            synchronization.map_pose_sample(malformed)

    def test_map_base_pose_is_composed_with_time_varying_camera_extrinsic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "state/000000").mkdir(parents=True)
            (root / "poses/dense_global/000000").mkdir(parents=True)
            map_records = []
            aux_records = []
            for tick, timestamp_ns in enumerate((100, 200)):
                map_records.append(
                    {
                        "tick": tick,
                        "sensor_time_ns": timestamp_ns,
                        "target_frame": "map",
                        "source_frame": "base_link",
                        "pose": {
                            "target_frame": "map",
                            "source_frame": "base_link",
                            "timestamp_ns": timestamp_ns,
                            "position": [float(tick), 0.0, 0.0],
                            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                )
                aux_records.append(
                    {
                        "tick": tick,
                        "poses": {
                            "head_camera": {
                                "timestamp_ns": timestamp_ns,
                                "position": [0.25, 0.0, 1.5],
                                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                            }
                        },
                    }
                )
            (root / "state/000000/map_pose.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in map_records)
            )
            (root / "poses/dense_global/000000/aux_poses.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in aux_records)
            )

            poses, base_clamped, camera_clamped = (
                synchronization.compose_global_camera_poses(
                    root,
                    [150],
                    camera_quaternion_order="xyzw",
                    base_pose_source="map",
                )
            )
            self.assertEqual(base_clamped, 0)
            self.assertEqual(camera_clamped, 0)
            np.testing.assert_allclose(poses[0][:3, 3], [0.75, 0.0, 1.5])

    def test_recover_pose_cannot_rewrite_the_recorded_ransac_count(self):
        points = np.zeros((1200, 2), dtype=np.float64)
        ransac_mask = np.ones((1200, 1), dtype=np.uint8)

        def mutate_pose_mask(*_args, **kwargs):
            kwargs["mask"][:600] = 0
            return 1100, np.eye(3), np.array([[-1.0], [0.0], [0.0]]), kwargs["mask"]

        with mock.patch.object(
            preparation.cv2,
            "findEssentialMat",
            return_value=(np.eye(3), ransac_mask),
        ), mock.patch.object(
            preparation.cv2, "recoverPose", side_effect=mutate_pose_mask
        ):
            _, _, evidence = preparation.recover_stereo_pose(
                points, points, baseline=0.06
            )

        self.assertEqual(evidence["ransac_inliers"], 1200)
        self.assertEqual(evidence["ransac_inlier_ratio"], 1.0)
        self.assertEqual(int(np.count_nonzero(ransac_mask)), 1200)

    def test_right_only_rectification_fits_and_inverts_vertical_warp(self):
        rng = np.random.default_rng(7)
        right = np.column_stack(
            (rng.uniform(20.0, 1260.0, 5000), rng.uniform(20.0, 940.0, 5000))
        )
        coefficients = np.array(
            [0.006, 1.01, -4.0, 1.0e-5, -2.0e-6], dtype=np.float64
        )
        a, b, c, g, h = coefficients
        left_y = (
            a * right[:, 0] + b * right[:, 1] + c
        ) / (g * right[:, 0] + h * right[:, 1] + 1.0)
        left_y += rng.normal(0.0, 0.03, len(left_y))
        left = np.column_stack((right[:, 0] - 30.0, left_y))

        rectification = preparation.estimate_rectified_right_vertical_warp(
            left, right
        )
        recovered_source_y = preparation.invert_rectified_right_vertical_warp(
            right[:, 0], left_y, rectification
        )
        recovered_source_x, recovered_source_y_full = (
            preparation.invert_rectified_right_remap(
                right[:, 0], left_y, rectification
            )
        )

        self.assertEqual(
            rectification["model"], "right_y_projective_x_preserving"
        )
        self.assertTrue(rectification["left_image_preserved"])
        self.assertLess(np.median(np.abs(recovered_source_y - right[:, 1])), 0.05)
        np.testing.assert_allclose(recovered_source_x, right[:, 0], atol=1.0e-4)
        np.testing.assert_allclose(
            recovered_source_y_full, recovered_source_y, atol=1.0e-4
        )

    def test_pinhole_virtual_camera_preserves_left_image_orientation(self):
        width, height = 640, 480
        focal = 320.0
        source_cy = 240.0
        K = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, source_cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        down_fov_deg = np.degrees(np.arctan(((height - 1) - source_cy) / focal))
        right_identity = {
            "model": "right_y_projective_x_preserving",
            "coefficients_a_b_c_g_h": [0.0, 1.0, 0.0, 0.0, 0.0],
        }

        (
            virtual_K,
            left_rotation,
            right_rotation,
            left_maps,
            right_maps,
            _,
        ) = preparation.build_virtual_camera(
            K,
            np.zeros((4, 1), dtype=np.float64),
            width,
            height,
            horizontal_fov_deg=90.0,
            down_fov_deg=down_fov_deg,
            optical_x_rotation_deg=0.0,
            camera0_R_camera1=preparation.Rotation.from_euler(
                "z", 12.0, degrees=True
            ).as_matrix(),
            camera0_t_camera1=np.array([-0.06, 0.0, 0.0]),
            input_projection_model="pinhole_rectified",
            right_vertical_rectification=right_identity,
        )

        np.testing.assert_allclose(virtual_K, K, atol=1.0e-12)
        np.testing.assert_allclose(left_rotation, np.eye(3), atol=1.0e-12)
        np.testing.assert_allclose(right_rotation, np.eye(3), atol=1.0e-12)
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        np.testing.assert_allclose(left_maps[0], grid_x, atol=1.0e-4)
        np.testing.assert_allclose(left_maps[1], grid_y, atol=1.0e-4)
        np.testing.assert_allclose(right_maps[0], grid_x, atol=1.0e-4)
        np.testing.assert_allclose(right_maps[1], grid_y, atol=1.0e-4)

    def test_fixed_report_requires_matching_geometry_and_records_provenance(self):
        K = np.array(
            [[417.0, 0.0, 691.0], [0.0, 417.0, 438.0], [0.0, 0.0, 1.0]]
        )
        distortion = np.zeros((4, 1), dtype=np.float64)
        translation = [-0.06, 0.0, 0.0]
        report = {
            "source_intrinsics": K.tolist(),
            "source_distortion": distortion.reshape(-1).tolist(),
            "estimated_stereo_calibration": {
                "camera0_R_camera1": np.eye(3).tolist(),
                "camera0_t_camera1_m": translation,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "pinhole_preparation_report.json"
            report_path.write_text(json.dumps(report))
            _, _, evidence = preparation.load_fixed_stereo_calibration(
                report_path, K, distortion, baseline=0.06
            )
            self.assertEqual(evidence["mode"], "fixed_report")
            self.assertEqual(len(evidence["report_sha256"]), 64)
            self.assertTrue(
                evidence["source_geometry_validation"]["intrinsics_match"]
            )

            mismatched_K = K.copy()
            mismatched_K[0, 0] += 1.0
            with self.assertRaisesRegex(ValueError, "intrinsics do not match"):
                preparation.load_fixed_stereo_calibration(
                    report_path, mismatched_K, distortion, baseline=0.06
                )


if __name__ == "__main__":
    unittest.main()
