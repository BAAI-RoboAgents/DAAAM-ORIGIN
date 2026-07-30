"""Tests for the strict Unitree head-D455 RGB-D adapter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


unitree = load_script_module(
    "prepare_unitree_head_rgbd_dataset_test",
    REPOSITORY_ROOT / "scripts/prepare_unitree_head_rgbd_dataset.py",
)
stereo_runner = load_script_module(
    "run_stereo_mapping_unitree_test",
    REPOSITORY_ROOT / "scripts/run_stereo_mapping.py",
)
realtime_runner = load_script_module(
    "run_realtime_mapping_unitree_test",
    REPOSITORY_ROOT / "scripts/run_realtime_mapping.py",
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def create_capture(root: Path, frame_count: int = 3) -> Path:
    source = root / "capture"
    sequence = "000000"
    for relative in (
        f"2d_rect/cam0/{sequence}",
        f"2d_rect/cam1/{sequence}",
        f"3d_raw/cam0/{sequence}",
        f"3d_raw/cam1/{sequence}",
        f"lidar/{sequence}/lidar0",
        f"calibrations/{sequence}",
        f"state/{sequence}",
        f"poses/dense_global/{sequence}",
    ):
        (source / relative).mkdir(parents=True)

    origin_ns = 1_800_000_000_000_000_000
    records = []
    odometry = []
    auxiliary = []
    for index in range(frame_count):
        timestamp_ns = origin_ns + index * 100_000_000
        rgb_relative = f"2d_rect/cam0/{sequence}/{index:06d}.png"
        depth_relative = f"3d_raw/cam0/{sequence}/{index:06d}.npy"
        secondary_rgb_relative = f"2d_rect/cam1/{sequence}/{index:06d}.png"
        secondary_depth_relative = f"3d_raw/cam1/{sequence}/{index:06d}.npy"
        lidar_relative = f"lidar/{sequence}/lidar0/{index:06d}.npy"
        rgb = np.full((12, 16, 3), 20 + index, dtype=np.uint8)
        depth = np.full((12, 16), 1.0 + index * 0.1, dtype=np.float32)
        assert cv2.imwrite(str(source / rgb_relative), rgb)
        assert cv2.imwrite(str(source / secondary_rgb_relative), rgb)
        np.save(source / depth_relative, depth)
        np.save(source / secondary_depth_relative, depth)
        np.save(source / lidar_relative, np.ones((4, 3), dtype=np.float32))
        records.append(
            {
                "tick": index,
                "anchor_timestamp_ns": timestamp_ns,
                "images": {
                    "cam0": {
                        "path": rgb_relative,
                        "timestamp_ns": timestamp_ns,
                        "host_ns": timestamp_ns,
                    },
                    "cam1": {
                        "path": secondary_rgb_relative,
                        "timestamp_ns": timestamp_ns,
                        "host_ns": timestamp_ns,
                    },
                },
                "depth": {
                    "cam0": {
                        "path": depth_relative,
                        "valid_ratio": 1.0,
                    },
                    "cam1": {
                        "path": secondary_depth_relative,
                        "valid_ratio": 1.0,
                    },
                },
                "lidar_path": lidar_relative,
                "lidar_points": 4,
                "odom": {"timestamp_ns": timestamp_ns},
            }
        )
        pose = [
            index * 0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        odometry.append(
            {
                "tick": index,
                "header": {"timestamp_ns": timestamp_ns, "frame_id": "local"},
                "child_frame_id": "body",
                "pose_xyz_quat_xyzw": pose,
            }
        )
        auxiliary.append(
            {
                "tick": index,
                "poses": {
                    "map": {
                        "target_frame": "local",
                        "source_frame": "body",
                    },
                    "cam0": {"pose_xyz_quat_xyzw": None},
                },
            }
        )

    (source / "manifest.json").write_text(
        json.dumps(
            {
                "layout": "daaam_g1_hardware_v1",
                "cameras": {
                    "cam0": "HEAD_D455",
                    "cam1": "CHEST_D435I",
                },
            }
        )
    )
    write_jsonl(source / "manifest.jsonl", records)
    write_jsonl(source / f"state/{sequence}/odom.jsonl", odometry)
    write_jsonl(
        source / f"poses/dense_global/{sequence}/aux_poses.jsonl",
        auxiliary,
    )
    (source / f"timestamps/{sequence}.txt").parent.mkdir()
    (source / f"timestamps/{sequence}.txt").write_text(
        "".join(
            f"{origin_ns + index * 100_000_000}\n"
            for index in range(frame_count)
        )
    )
    identity = " ".join(str(value) for value in np.eye(4).reshape(-1))
    (source / f"poses/dense_global/{sequence}/poses.txt").write_text(
        "".join(identity + "\n" for _ in range(frame_count))
    )
    (source / f"poses/dense_global/{sequence}/poses_7d.txt").write_text(
        "".join("0 0 0 0 0 0 1\n" for _ in range(frame_count))
    )
    (source / f"calibrations/{sequence}/camera_info.json").write_text(
        json.dumps(
            {
                "cam0": {
                    "sensor": "HEAD_D455",
                    "intrinsics": {
                        "fx": 14.0,
                        "fy": 14.0,
                        "cx": 7.5,
                        "cy": 5.5,
                        "width": 16,
                        "height": 12,
                    },
                },
                "notes": {"cam0_depth_aligned_to_rgb": True},
            }
        )
    )
    counts = {
        name: frame_count
        for name in (
            "frames",
            "timestamps",
            "poses",
            "aux_poses",
            "manifest_records",
            "cam0_images",
            "cam0_depth",
            "lidar",
            "odom",
        )
    }
    counts["ok"] = True
    (source / "quality_report.json").write_text(
        json.dumps(
            {
                "alignment_ok": True,
                "target_hz": 10.0,
                "actual_hz_estimate": 10.0,
                "counts": counts,
                "extra": {"attempts": frame_count, "skipped": 0},
            }
        )
    )
    return source


def create_contracts(root: Path) -> tuple[Path, Path]:
    camera = root / "head_camera_calibration.json"
    camera.write_text(
        json.dumps(
            {
                "schema": "unitree_head_rgbd_calibration_v1",
                "sensor": "HEAD_D455",
                "target_frame": "body",
                "source_frame": "head_d455_color_optical_frame",
                "target_T_camera": [
                    [1.0, 0.0, 0.0, 0.2],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "intrinsics": {
                    "model": "pinhole",
                    "fx": 14.0,
                    "fy": 14.0,
                    "cx": 7.5,
                    "cy": 5.5,
                    "width": 16,
                    "height": 12,
                    "distortion": {
                        "model": "none",
                        "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                    },
                },
                "depth": {
                    "aligned_to_color": True,
                    "unit": "meter",
                    "minimum_valid_depth_m": 0.1,
                    "maximum_valid_depth_m": 8.0,
                    "invalid_values_m": [0.0, 65.535],
                },
                "provenance": {
                    "validated": True,
                    "method": "fixture",
                    "source": "test",
                    "timestamp": "2026-07-28T00:00:00+08:00",
                },
            }
        )
    )
    time_contract = root / "time_contract.json"
    time_contract.write_text(
        json.dumps(
            {
                "schema": "unitree_rgbd_time_contract_v1",
                "camera": "cam0",
                "camera_time_source": "sensor",
                "shared_timebase_verified": True,
                "rgb_depth_same_capture_verified": True,
                "verification_method": "fixture",
                "maximum_pose_interpolation_gap_ms": 150.0,
                "allow_drop_unbracketed_frames": False,
                "maximum_dropped_frames": 0,
            }
        )
    )
    return camera, time_contract


def test_audit_reports_explicit_missing_contracts(tmp_path):
    source = create_capture(tmp_path)
    report = unitree.audit_dataset(source, "000000")
    assert not report["mapping_ready"]
    blockers = {
        item["code"] for item in report["summary"]["hard_blockers"]
    }
    assert blockers == {
        "contract.head_camera_calibration",
        "contract.rgbd_time",
    }


def test_prepare_builds_native_aligned_rgbd_contract(tmp_path):
    source = create_capture(tmp_path)
    camera, time_contract = create_contracts(tmp_path)
    output = tmp_path / "prepared"
    report = unitree.prepare_dataset(
        source,
        output,
        "000000",
        camera_calibration_path=camera,
        time_contract_path=time_contract,
    )
    assert report["status"] == "complete"
    assert report["counts"]["prepared_frames"] == 3
    metadata = json.loads((output / "tick_index.json").read_text())
    assert metadata["input_modality"] == "aligned_rgbd"
    assert metadata["depth_evidence_type"] == "aligned_rgbd_sensor"
    assert "cam1" not in metadata["frames"][0]
    assert stereo_runner.validate_time_contract(output)["valid"]
    provenance = realtime_runner.load_precomputed_depth_provenance(output)
    assert provenance["backend"] == "sensor-aligned-rgbd"
    frames = realtime_runner.build_frames(
        output,
        metadata,
        realtime_runner.read_poses(output),
    )
    assert len(frames) == 3
    assert all(frame.right_path is None for frame in frames)

    poses = np.loadtxt(output / "pose/poses.txt").reshape(-1, 4, 4)
    np.testing.assert_allclose(poses[:, 0, 3], [0.2, 0.3, 0.4])
    np.testing.assert_allclose(poses[:, 2, 3], [0.5, 0.5, 0.5])
    depth = cv2.imread(str(output / "depth/00000000.png"), cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16
    assert np.all(depth == 1000)
