"""Regression tests for the RGB-D validation and pose-graph safety gates."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
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


def create_rgbd_dataset(root: Path) -> tuple[Path, list[int]]:
    dataset = root / "source"
    for directory in ("rgb", "stereo_right", "depth", "pose"):
        (dataset / directory).mkdir(parents=True, exist_ok=True)

    timestamps = [
        1_783_933_507_759_540_877,
        1_783_933_507_839_540_877,
        1_783_933_508_069_540_877,
        1_783_933_508_509_540_877,
        1_783_933_508_609_540_877,
    ]
    frames = []
    poses = []
    for index, timestamp_ns in enumerate(timestamps):
        rgb = np.full((24, 32, 3), 40 + index, dtype=np.uint8)
        depth = np.full((24, 32), 1500, dtype=np.uint16)
        cv2.imwrite(str(dataset / "rgb" / f"{index:08d}.png"), rgb)
        cv2.imwrite(str(dataset / "stereo_right" / f"{index:08d}.png"), rgb)
        cv2.imwrite(str(dataset / "depth" / f"{index:08d}.png"), depth)
        pose = np.eye(4, dtype=np.float64)
        poses.append(pose)
        frames.append(
            {
                "idx": index,
                "source_idx": 100 + index,
                "source_frame_idx": 200 + index,
                "pose_row": index,
                "timestamp": (timestamp_ns - timestamps[0]) / 1.0e9,
                "sensor_time_ns": timestamp_ns,
                "cam0_sensor_time_ns": timestamp_ns,
                "cam1_sensor_time_ns": timestamp_ns,
                "pose_sensor_time_ns": timestamp_ns,
                "stereo_delta_ms": 0.0,
                "cam0": str(dataset / "rgb" / f"{index:08d}.png"),
                "cam1": str(dataset / "stereo_right" / f"{index:08d}.png"),
            }
        )

    (dataset / "pose" / "poses.txt").write_text(
        "".join(
            " ".join(f"{value:.12g}" for value in pose.reshape(-1)) + "\n"
            for pose in poses
        )
    )
    (dataset / "pose" / "pose_timestamps_ns.txt").write_text(
        "".join(f"{timestamp_ns}\n" for timestamp_ns in timestamps)
    )
    (dataset / "camera_info.json").write_text(
        json.dumps(
            {
                "model": "pinhole",
                "width": 32,
                "height": 24,
                "fx": 30.0,
                "fy": 30.0,
                "cx": 15.5,
                "cy": 11.5,
                "baseline": 0.07,
            }
        )
    )
    (dataset / "tick_index.json").write_text(
        json.dumps(
            {
                "time_origin_ns": timestamps[0],
                "projection_model": "pinhole",
                "recommended_max_depth_m": 3.0,
                "frames": frames,
            }
        )
    )
    return dataset, timestamps


class RgbdMappingGuardTests(unittest.TestCase):
    def test_gravity_pose_graph_uses_accurate_scaled_sparse_solver(self):
        optimizer = load_script_module(
            "optimize_rgbd_pose_graph_solver_test",
            REPOSITORY_ROOT / "scripts/optimize_rgbd_pose_graph.py",
        )
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        poses[1, 0, 3] = 0.1
        edge = optimizer.GraphEdge(
            0,
            1,
            optimizer.relative_pose(poses, 0, 1),
            0.04,
            1.5,
            "rgbd_odometry",
        )
        captured = {}

        def fake_least_squares(residual, initial, **kwargs):
            captured.update(kwargs)
            values = residual(initial)
            return SimpleNamespace(
                success=True,
                status=2,
                message="ftol satisfied",
                cost=float(np.dot(values, values) / 2.0),
                nfev=1,
                njev=1,
                optimality=0.0,
                x=initial,
                fun=values,
            )

        with mock.patch.object(
            optimizer, "least_squares", side_effect=fake_least_squares
        ):
            _, report = optimizer.optimize_gravity_se3(
                poses,
                [edge],
                max_nfev=10,
                z_sigma_m=0.04,
                roll_pitch_sigma_deg=2.0,
                initial_poses=poses,
            )

        self.assertEqual(captured["method"], "trf")
        self.assertEqual(captured["tr_solver"], "lsmr")
        self.assertEqual(captured["x_scale"], "jac")
        self.assertEqual(
            captured["tr_options"], {"atol": 1.0e-10, "btol": 1.0e-10}
        )
        self.assertEqual(report["status"], 2)
        self.assertEqual(report["optimality"], 0.0)
        self.assertEqual(report["solver"]["lsmr_atol"], 1.0e-10)

    def test_one_click_dry_run_plans_full_chain_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, _ = create_rgbd_dataset(root)
            run_dir = root / "planned-run"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts/run_stereo_mapping.py"),
                    "--adapter",
                    "prepared-stereo",
                    "--src",
                    str(dataset),
                    "--run-dir",
                    str(run_dir),
                    "--hydra-config-path",
                    str(root / "hydra.yaml"),
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            planned_commands = [
                line for line in result.stdout.splitlines() if line.startswith("+ ")
            ]
            self.assertEqual(len(planned_commands), 10)
            self.assertIn("select_mapping_keyframes.py", result.stdout)
            self.assertIn("filter_temporal_depth_consistency.py", result.stdout)
            self.assertIn("run_pipeline.py", result.stdout)
            self.assertIn('"status": "planned"', result.stdout)
            self.assertFalse(run_dir.exists())

    def test_temporal_filter_preserves_frames_poses_and_absolute_times(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, timestamps = create_rgbd_dataset(root)
            output = root / "filtered"
            subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts/filter_temporal_depth_consistency.py"),
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--neighbor-offsets",
                    "1,2",
                    "--filter-scale",
                    "1",
                    "--min-judged-neighbors",
                    "1",
                    "--min-support-ratio",
                    "0.5",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                (output / "pose/poses.txt").read_bytes(),
                (dataset / "pose/poses.txt").read_bytes(),
            )
            self.assertEqual(
                (output / "pose/pose_timestamps_ns.txt").read_bytes(),
                (dataset / "pose/pose_timestamps_ns.txt").read_bytes(),
            )
            metadata = json.loads((output / "tick_index.json").read_text())
            self.assertEqual(
                [int(frame["sensor_time_ns"]) for frame in metadata["frames"]],
                timestamps,
            )
            self.assertEqual(
                [int(frame["source_frame_idx"]) for frame in metadata["frames"]],
                [200, 201, 202, 203, 204],
            )
            self.assertEqual(len(list((output / "depth").glob("*.png"))), 5)
            report = json.loads(
                (output / "temporal_depth_filter_report.json").read_text()
            )
            self.assertTrue(report["absolute_time_contract_validated"])
            self.assertTrue(report["rgb_frames_preserved"])
            self.assertTrue(report["poses_preserved"])

    def test_temporal_filter_preserves_verifiable_left_right_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, timestamps = create_rgbd_dataset(root)
            for directory in (
                "depth_confidence",
                "depth_consistency",
                "depth_occlusion",
                "depth_metadata",
            ):
                (dataset / directory).mkdir()
            for index, timestamp in enumerate(timestamps):
                cv2.imwrite(
                    str(dataset / "depth_confidence" / f"{index:08d}.png"),
                    np.full((24, 32), 220, dtype=np.uint8),
                )
                cv2.imwrite(
                    str(dataset / "depth_consistency" / f"{index:08d}.png"),
                    np.full((24, 32), 255, dtype=np.uint8),
                )
                cv2.imwrite(
                    str(dataset / "depth_occlusion" / f"{index:08d}.png"),
                    np.zeros((24, 32), dtype=np.uint8),
                )
                (dataset / "depth_metadata" / f"{index:08d}.json").write_text(
                    json.dumps(
                        {
                            "frame_idx": index,
                            "sensor_time_ns": timestamp,
                            "confidence_mode": "left-right",
                            "left_right_verified": True,
                            "left_right_consistency": 1.0,
                        }
                    )
                )

            output = root / "filtered-with-evidence"
            subprocess.run(
                [
                    sys.executable,
                    str(
                        REPOSITORY_ROOT
                        / "scripts/filter_temporal_depth_consistency.py"
                    ),
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--neighbor-offsets",
                    "1,2",
                    "--filter-scale",
                    "1",
                    "--min-judged-neighbors",
                    "1",
                    "--min-support-ratio",
                    "0.5",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            for directory in (
                "depth_confidence",
                "depth_consistency",
                "depth_occlusion",
                "depth_metadata",
            ):
                self.assertEqual(len(list((output / directory).iterdir())), 5)
            metadata = json.loads(
                (output / "depth_metadata" / "00000000.json").read_text()
            )
            self.assertTrue(metadata["left_right_verified"])
            self.assertEqual(
                metadata["temporal_filter"]["method"],
                "multi_neighbor_reprojection_consistency",
            )
            self.assertEqual(len(metadata["temporal_filter"]["output_depth_sha256"]), 64)
            report = json.loads(
                (output / "temporal_depth_filter_report.json").read_text()
            )
            self.assertEqual(report["depth_evidence"]["coverage"], 1.0)

    def test_nonuniform_time_temporal_gate_uses_every_adjacent_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, _ = create_rgbd_dataset(root)
            output = root / "temporal"
            subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts/diagnose_temporal_depth_consistency.py"),
                    "--dataset",
                    str(dataset),
                    "--output-dir",
                    str(output),
                    "--frame-step",
                    "1",
                    "--neighbor-offsets",
                    "1",
                    "--pixel-step",
                    "2",
                    "--forward-only",
                    "--require-time-contract",
                    "--window-size-frames",
                    "2",
                    "--max-panels",
                    "0",
                    "--fail-below-adjacent-agreement-rate",
                    "0.99",
                    "--fail-above-adjacent-median-error-m",
                    "0.001",
                    "--fail-below-window-adjacent-agreement-rate",
                    "0.99",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(
                (output / "temporal_depth_consistency_report.json").read_text()
            )
            adjacent = report["summary_by_absolute_offset"]["1"]
            self.assertEqual(adjacent["pairs"], 4)
            self.assertEqual(adjacent["agreement_rate_weighted"], 1.0)
            self.assertEqual(adjacent["absolute_time_delta_s_max"], 0.44)
            self.assertTrue(report["pre_hydra_gate"]["passed"])
            self.assertEqual(len(report["adjacent_window_summary"]), 2)

    def test_pose_graph_rejects_gravity_incompatible_loop(self):
        try:
            optimizer = load_script_module(
                "optimize_rgbd_pose_graph_test",
                REPOSITORY_ROOT / "scripts/optimize_rgbd_pose_graph.py",
            )
        except ModuleNotFoundError as error:
            if error.name == "open3d":
                self.skipTest("open3d is not installed in this test environment")
            raise

        poses = np.repeat(np.eye(4, dtype=np.float64)[None], 50, axis=0)

        def candidate(rotation: np.ndarray, similarity: float) -> dict:
            return {
                "similarity": similarity,
                "quality_ok": True,
                "dense_verification": {
                    "selected_hypothesis": {
                        "forward_fitness_5cm": 0.7,
                        "reverse_fitness_5cm": 0.6,
                    },
                    "pixel_verification": {
                        "depth_agreement_rate": 0.8,
                        "color_agreement_rate_on_depth_agreement": 0.9,
                    },
                },
                "constraint": {
                    "first": 2,
                    "second": 40,
                    "relative_camera_rotation": rotation.tolist(),
                    "relative_camera_translation_m": [0.0, 0.0, 0.0],
                },
            }

        yaw_loop = candidate(Rotation.from_euler("z", 30, degrees=True).as_matrix(), 0.8)
        roll_loop = candidate(Rotation.from_euler("x", 20, degrees=True).as_matrix(), 0.9)
        selected = optimizer.select_loop_candidates(
            {"verified_links": [roll_loop, yaw_loop]},
            poses,
            frame_count=50,
            cluster_radius=5,
            max_edges=4,
            max_gravity_residual_deg=8.0,
        )
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected[0]["similarity"], 0.8)
        self.assertLess(
            selected[0]["gravity_compatibility"][
                "minimum_full_rotation_residual_deg"
            ],
            0.01,
        )
        with self.assertRaisesRegex(RuntimeError, "gravity-compatible loop"):
            optimizer.select_loop_candidates(
                {"verified_links": [roll_loop]},
                poses,
                frame_count=50,
                cluster_radius=5,
                max_edges=4,
                max_gravity_residual_deg=8.0,
            )


if __name__ == "__main__":
    unittest.main()
