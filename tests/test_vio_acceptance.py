"""Focused tests for exact-stamp OpenVINS and no-GT acceptance gates."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from daaam.slam.openvins import (  # noqa: E402
    OPENVINS_PINNED_COMMIT,
    OpenVinsOdometryStream,
    StampedOdometry,
    load_odometry_jsonl,
)
from daaam.slam.vio_acceptance import (  # noqa: E402
    SensorEvent,
    VioAcceptanceConfig,
    evaluate_vio_acceptance,
    write_vio_acceptance_report,
)
from record_openvins_acceptance import stamp_to_ns  # noqa: E402


ORIGIN_NS = 1_783_933_507_759_540_877
COVARIANCE = np.eye(6, dtype=np.float64) * 0.001


def quaternion_z(degrees: float) -> np.ndarray:
    half = np.deg2rad(degrees) * 0.5
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)])


def sample(
    index: int,
    position: np.ndarray,
    *,
    traversal_id: str = "outbound",
    yaw_deg: float = 0.0,
    sensor_time_ns: int | None = None,
) -> StampedOdometry:
    timestamp = sensor_time_ns or ORIGIN_NS + index * 50_000_000
    return StampedOdometry(
        sensor_time_ns=timestamp,
        pose_time_ns=timestamp,
        frame_id="odom",
        child_frame_id="imu",
        position_m=np.asarray(position, dtype=np.float64),
        orientation_xyzw=quaternion_z(yaw_deg),
        pose_covariance=COVARIANCE,
        lookup_semantics="message_stamp",
        traversal_id=traversal_id,
    )


def repeated_loop_stream(*, endpoint_yaw_deg: float = 0.0) -> OpenVinsOdometryStream:
    x = np.linspace(0.0, 5.0, 80)
    outbound = np.c_[x, 0.17 * np.sin(1.7 * x) + 0.02 * x, 0.03 * np.cos(x)]
    inbound = outbound[::-1] + np.array([0.08, -0.03, 0.01])
    values = []
    for index, position in enumerate(outbound):
        values.append(sample(index, position, traversal_id="outbound"))
    offset = len(values)
    for index, position in enumerate(inbound):
        yaw = endpoint_yaw_deg if index == len(inbound) - 1 else 0.0
        values.append(
            sample(offset + index, position, traversal_id="return", yaw_deg=yaw)
        )
    return OpenVinsOdometryStream(values)


def short_acceptance_config(**overrides) -> VioAcceptanceConfig:
    values = {
        "minimum_duration_s": 0.0,
        "minimum_odometry_rate_hz": 0.0,
        "require_sensor_evidence": False,
        "alignment_sample_count": 64,
    }
    values.update(overrides)
    return VioAcceptanceConfig(**values)


def test_stamped_openvins_contract_rejects_latest_tf_and_mismatched_stamp():
    valid = sample(0, np.zeros(3))
    assert np.allclose(valid.parent_T_child, np.eye(4))
    estimate = valid.to_pose_estimate()
    assert estimate.sensor_time_ns == valid.sensor_time_ns
    assert estimate.source == "openvins:/ov_msckf/odomimu"

    with pytest.raises(ValueError, match="latest-TF semantics are forbidden"):
        StampedOdometry(
            sensor_time_ns=ORIGIN_NS,
            pose_time_ns=ORIGIN_NS,
            frame_id="odom",
            child_frame_id="imu",
            position_m=np.zeros(3),
            orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            pose_covariance=COVARIANCE,
            lookup_semantics="latest",
        )
    with pytest.raises(ValueError, match="pose_time_ns must equal"):
        StampedOdometry(
            sensor_time_ns=ORIGIN_NS,
            pose_time_ns=ORIGIN_NS + 1,
            frame_id="odom",
            child_frame_id="imu",
            position_m=np.zeros(3),
            orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            pose_covariance=COVARIANCE,
        )


def test_ros_recorder_uses_message_header_as_absolute_nanoseconds():
    assert stamp_to_ns(SimpleNamespace(sec=1_783_933_507, nanosec=759_540_877)) == (
        ORIGIN_NS
    )
    with pytest.raises(ValueError, match="timestamp fields"):
        stamp_to_ns(SimpleNamespace(sec=1, nanosec=1_000_000_000))
    with pytest.raises(ValueError, match="absolute nanoseconds"):
        StampedOdometry(
            sensor_time_ns=float(ORIGIN_NS),
            pose_time_ns=float(ORIGIN_NS),
            frame_id="odom",
            child_frame_id="imu",
            position_m=np.zeros(3),
            orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            pose_covariance=COVARIANCE,
        )


def test_openvins_stream_rejects_non_monotonic_time_and_frame_changes():
    stream = OpenVinsOdometryStream([sample(1, np.zeros(3))])
    with pytest.raises(ValueError, match="increase strictly"):
        stream.append(
            sample(
                0,
                np.ones(3),
                sensor_time_ns=ORIGIN_NS + 50_000_000,
            )
        )
    changed_frame = StampedOdometry(
        sensor_time_ns=ORIGIN_NS + 200_000_000,
        pose_time_ns=ORIGIN_NS + 200_000_000,
        frame_id="map",
        child_frame_id="imu",
        position_m=np.zeros(3),
        orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        pose_covariance=COVARIANCE,
    )
    with pytest.raises(ValueError, match="frames changed"):
        stream.append(changed_frame)


def test_jsonl_loader_requires_auditable_timestamp_semantics(tmp_path: Path):
    path = tmp_path / "odometry.jsonl"
    record = {
        "sensor_time_ns": ORIGIN_NS,
        "pose_time_ns": ORIGIN_NS,
        "frame_id": "odom",
        "child_frame_id": "imu",
        "position_m": [0.0, 0.0, 0.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "pose_covariance": COVARIANCE.reshape(-1).tolist(),
        "lookup_semantics": "latest",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1.*latest-TF"):
        load_odometry_jsonl(path)
    del record["lookup_semantics"]
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_odometry_jsonl(path)


def test_no_gt_loop_and_reverse_path_acceptance_passes(tmp_path: Path):
    stream = repeated_loop_stream()
    report = evaluate_vio_acceptance(stream, config=short_acceptance_config())
    assert report["passed"]
    assert report["metrics"]["loop_translation_m"] < 0.50
    assert report["metrics"]["loop_rotation_deg"] == pytest.approx(0.0)
    assert report["metrics"]["reverse_repeated_aligned_p95_m"] < 1.0e-8
    assert report["traversal_comparisons"][0]["reversed"] is True
    assert report["estimator"]["pinned_commit"] == OPENVINS_PINNED_COMMIT

    output = tmp_path / "nested" / "acceptance.json"
    write_vio_acceptance_report(report, output)
    restored = json.loads(output.read_text(encoding="utf-8"))
    assert restored["passed"] is True


def test_no_gt_gate_fails_loop_orientation_and_repeated_path_distortion():
    base = repeated_loop_stream(endpoint_yaw_deg=15.0)
    values = list(base.samples)
    return_indices = [
        index for index, value in enumerate(values) if value.traversal_id == "return"
    ]
    for ordinal, index in enumerate(return_indices):
        original = values[index]
        distortion = np.array([0.0, 0.9 * np.sin(ordinal / 7.0), 0.0])
        values[index] = sample(
            index,
            original.position_m + distortion,
            traversal_id="return",
            yaw_deg=15.0 if ordinal == len(return_indices) - 1 else 0.0,
        )
    report = evaluate_vio_acceptance(
        OpenVinsOdometryStream(values), config=short_acceptance_config()
    )
    assert not report["passed"]
    assert "vio.loop_rotation" in report["failures"]
    assert "vio.repeated_path" in report["failures"]
    assert report["metrics"]["reverse_repeated_aligned_p95_m"] > 0.30


def test_sensor_evidence_requires_imu_stereo_duration_and_monotonicity():
    stream = repeated_loop_stream()
    config = short_acceptance_config(
        minimum_duration_s=1.0,
        require_sensor_evidence=True,
    )
    start = ORIGIN_NS
    valid = [
        SensorEvent(start, name, f"{name}_frame")
        for name in ("imu", "stereo_left", "stereo_right")
    ] + [
        SensorEvent(start + 1_100_000_000, name, f"{name}_frame")
        for name in ("imu", "stereo_left", "stereo_right")
    ]
    report = evaluate_vio_acceptance(stream, sensor_events=valid, config=config)
    assert report["passed"]

    invalid = [
        SensorEvent(start + 1_100_000_000, "imu"),
        SensorEvent(start, "imu"),
        SensorEvent(start, "stereo_left"),
        SensorEvent(start + 1_100_000_000, "stereo_left"),
    ]
    report = evaluate_vio_acceptance(stream, sensor_events=invalid, config=config)
    assert not report["passed"]
    assert "vio.sensor_evidence" in report["failures"]


def test_install_manifest_pins_openvins_to_immutable_commit():
    manifest = yaml.safe_load(
        (REPOSITORY_ROOT / "install" / "packages.yaml").read_text()
    )
    package = manifest["repositories"]["open_vins"]
    assert package["url"] == "https://github.com/rpng/open_vins.git"
    assert package["version"] == OPENVINS_PINNED_COMMIT


def test_cli_writes_auditable_report(tmp_path: Path):
    trajectory_path = tmp_path / "trajectory.jsonl"
    sensor_path = tmp_path / "sensors.jsonl"
    output_path = tmp_path / "report.json"
    trajectory_records = []
    for value in repeated_loop_stream().samples:
        trajectory_records.append(
            {
                "sensor_time_ns": value.sensor_time_ns,
                "pose_time_ns": value.pose_time_ns,
                "frame_id": value.frame_id,
                "child_frame_id": value.child_frame_id,
                "position_m": value.position_m.tolist(),
                "orientation_xyzw": value.orientation_xyzw.tolist(),
                "pose_covariance": value.pose_covariance.reshape(-1).tolist(),
                "lookup_semantics": value.lookup_semantics,
                "source_topic": value.source_topic,
                "traversal_id": value.traversal_id,
            }
        )
    trajectory_path.write_text(
        "".join(json.dumps(record) + "\n" for record in trajectory_records),
        encoding="utf-8",
    )
    sensor_path.write_text(
        "".join(
            json.dumps({"sensor_time_ns": ORIGIN_NS, "stream": name}) + "\n"
            for name in ("imu", "stereo_left", "stereo_right")
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "evaluate_vio_acceptance.py"),
            "--trajectory-jsonl",
            str(trajectory_path),
            "--sensor-jsonl",
            str(sensor_path),
            "--output",
            str(output_path),
            "--minimum-duration-s",
            "0",
            "--minimum-odometry-rate-hz",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"]
    assert report["provenance"]["trajectory_sha256"]
    assert report["provenance"]["sensor_sha256"]
