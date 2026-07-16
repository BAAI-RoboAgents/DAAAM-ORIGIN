#!/usr/bin/env python3
"""Record exact-stamp OpenVINS and stereo/IMU evidence on a ROS 2 robot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--odom-topic", default="/ov_msckf/odomimu")
    parser.add_argument("--imu-topic", required=True)
    parser.add_argument("--left-topic", required=True)
    parser.add_argument("--right-topic", required=True)
    parser.add_argument(
        "--image-message-type",
        choices=("image", "compressed"),
        default="image",
    )
    parser.add_argument(
        "--traversal-topic",
        default="/daaam/vio_traversal",
        help=(
            "std_msgs/String marker. Publish 'outbound', then 'return' or another "
            "stable traversal ID before repeating the route."
        ),
    )
    parser.add_argument("--initial-traversal", default="outbound")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stamp_to_ns(stamp: Any) -> int:
    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError("ROS timestamp fields are invalid")
    value = seconds * 1_000_000_000 + nanoseconds
    if value <= 0:
        raise ValueError("ROS timestamp must be positive absolute nanoseconds")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.duration_s <= 0.0 or not args.initial_traversal.strip():
        raise ValueError("Duration and initial traversal must be positive/non-empty")
    output_dir = args.output_dir.resolve()
    prepare_output(output_dir, args.overwrite)

    try:
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, Image, Imu
        from std_msgs.msg import String
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 Python packages are required on the robot to record VIO evidence"
        ) from error

    trajectory_path = output_dir / "openvins_odometry.jsonl"
    sensors_path = output_dir / "sensor_events.jsonl"

    class AcceptanceRecorder(Node):
        def __init__(self) -> None:
            super().__init__("daaam_openvins_acceptance_recorder")
            self.lock = threading.Lock()
            self.trajectory_file = trajectory_path.open("x", encoding="utf-8")
            self.sensor_file = sensors_path.open("x", encoding="utf-8")
            self.traversal_id = args.initial_traversal.strip()
            self.started_monotonic = time.monotonic()
            self.counts = {
                "odometry": 0,
                "imu": 0,
                "stereo_left": 0,
                "stereo_right": 0,
            }
            self.first_sensor_time_ns: int | None = None
            self.last_sensor_time_ns: int | None = None
            self.failure: str | None = None
            self.traversal_markers = [
                {
                    "traversal_id": self.traversal_id,
                    "host_monotonic_s": self.started_monotonic,
                }
            ]
            image_type = Image if args.image_message_type == "image" else CompressedImage
            self.create_subscription(
                Odometry,
                args.odom_topic,
                self.on_odometry,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Imu,
                args.imu_topic,
                lambda message: self.on_sensor(message, "imu", args.imu_topic),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                image_type,
                args.left_topic,
                lambda message: self.on_sensor(
                    message, "stereo_left", args.left_topic
                ),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                image_type,
                args.right_topic,
                lambda message: self.on_sensor(
                    message, "stereo_right", args.right_topic
                ),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                String,
                args.traversal_topic,
                self.on_traversal,
                10,
            )
            self.create_timer(0.1, self.check_deadline)

        def _observe_time(self, sensor_time_ns: int) -> None:
            self.first_sensor_time_ns = (
                sensor_time_ns
                if self.first_sensor_time_ns is None
                else min(self.first_sensor_time_ns, sensor_time_ns)
            )
            self.last_sensor_time_ns = (
                sensor_time_ns
                if self.last_sensor_time_ns is None
                else max(self.last_sensor_time_ns, sensor_time_ns)
            )

        @staticmethod
        def _write(stream, record: dict[str, Any]) -> None:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()

        def on_odometry(self, message) -> None:
            try:
                sensor_time_ns = stamp_to_ns(message.header.stamp)
                covariance = [float(value) for value in message.pose.covariance]
                if len(covariance) != 36:
                    raise ValueError("OpenVINS pose covariance must contain 36 values")
                record = {
                    "sensor_time_ns": sensor_time_ns,
                    "pose_time_ns": sensor_time_ns,
                    "frame_id": str(message.header.frame_id),
                    "child_frame_id": str(message.child_frame_id),
                    "position_m": [
                        float(message.pose.pose.position.x),
                        float(message.pose.pose.position.y),
                        float(message.pose.pose.position.z),
                    ],
                    "orientation_xyzw": [
                        float(message.pose.pose.orientation.x),
                        float(message.pose.pose.orientation.y),
                        float(message.pose.pose.orientation.z),
                        float(message.pose.pose.orientation.w),
                    ],
                    "pose_covariance": covariance,
                    "lookup_semantics": "message_stamp",
                    "source_topic": args.odom_topic,
                    "traversal_id": self.traversal_id,
                }
                with self.lock:
                    self._write(self.trajectory_file, record)
                    self.counts["odometry"] += 1
                    self._observe_time(sensor_time_ns)
            except Exception as error:
                self.fail(error)

        def on_sensor(self, message, stream_name: str, topic: str) -> None:
            try:
                sensor_time_ns = stamp_to_ns(message.header.stamp)
                record = {
                    "sensor_time_ns": sensor_time_ns,
                    "stream": stream_name,
                    "frame_id": str(message.header.frame_id),
                    "topic": topic,
                }
                with self.lock:
                    self._write(self.sensor_file, record)
                    self.counts[stream_name] += 1
                    self._observe_time(sensor_time_ns)
            except Exception as error:
                self.fail(error)

        def on_traversal(self, message) -> None:
            value = " ".join(str(message.data).split()).strip()
            if not value:
                self.fail(ValueError("Traversal marker cannot be empty"))
                return
            with self.lock:
                self.traversal_id = value
                self.traversal_markers.append(
                    {
                        "traversal_id": value,
                        "host_monotonic_s": time.monotonic(),
                    }
                )

        def fail(self, error: Exception) -> None:
            self.failure = repr(error)
            self.get_logger().error(self.failure)
            rclpy.shutdown()

        def check_deadline(self) -> None:
            if time.monotonic() - self.started_monotonic >= args.duration_s:
                rclpy.shutdown()

        def close_files(self) -> None:
            self.trajectory_file.close()
            self.sensor_file.close()

    rclpy.init()
    node = AcceptanceRecorder()
    interrupted = False
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        elapsed_s = time.monotonic() - node.started_monotonic
        node.close_files()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    complete_streams = all(node.counts[name] > 0 for name in node.counts)
    manifest = {
        "schema_version": 1,
        "status": (
            "failed"
            if node.failure
            else "interrupted" if interrupted else "complete"
        ),
        "failure": node.failure,
        "requested_duration_s": args.duration_s,
        "elapsed_host_seconds": elapsed_s,
        "topics": {
            "odometry": args.odom_topic,
            "imu": args.imu_topic,
            "stereo_left": args.left_topic,
            "stereo_right": args.right_topic,
            "traversal_marker": args.traversal_topic,
        },
        "counts": node.counts,
        "all_required_streams_observed": complete_streams,
        "first_sensor_time_ns": node.first_sensor_time_ns,
        "last_sensor_time_ns": node.last_sensor_time_ns,
        "traversal_markers": node.traversal_markers,
        "timestamp_semantics": "message_header_stamp_only",
        "trajectory_jsonl": str(trajectory_path),
        "trajectory_sha256": _sha256(trajectory_path),
        "sensor_jsonl": str(sensors_path),
        "sensor_sha256": _sha256(sensors_path),
    }
    (output_dir / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))
    if node.failure or not complete_streams:
        raise SystemExit(2)
    if interrupted or elapsed_s < args.duration_s * 0.99:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
