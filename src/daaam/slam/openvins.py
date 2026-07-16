"""Time-safe adapter contract for OpenVINS stamped odometry.

The ROS adapter is deliberately dependency-free: a ROS process can translate
``nav_msgs/msg/Odometry`` into :class:`StampedOdometry` while this module owns
the timestamp, frame, pose, and covariance validation.  In particular, no
``tf2`` lookup at time zero ("latest") is accepted as odometry evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from daaam.realtime.contracts import PoseEstimate


OPENVINS_ODOMETRY_TOPIC = "/ov_msckf/odomimu"
OPENVINS_PINNED_COMMIT = "69488123ed9362dd44b6f28e7f4680abbff1442b"
STAMPED_LOOKUP_SEMANTICS = "message_stamp"


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return result.copy()


def quaternion_xyzw_to_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    """Return a rotation matrix for a finite, non-zero XYZW quaternion."""

    quaternion = _array(quaternion_xyzw, (4,), "orientation_xyzw")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("orientation_xyzw must be non-zero")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class StampedOdometry:
    """Validated OpenVINS pose at its original sensor timestamp.

    ``pose_time_ns`` is separate from ``sensor_time_ns`` on purpose.  A bridge
    must prove that any transform used to form the pose was queried at the
    message stamp; substituting a host-clock timestamp or the latest TF is a
    hard contract violation.
    """

    sensor_time_ns: int
    pose_time_ns: int
    frame_id: str
    child_frame_id: str
    position_m: np.ndarray
    orientation_xyzw: np.ndarray
    pose_covariance: np.ndarray
    lookup_semantics: str = STAMPED_LOOKUP_SEMANTICS
    source_topic: str = OPENVINS_ODOMETRY_TOPIC
    traversal_id: str = "trajectory"

    def __post_init__(self) -> None:
        if (
            isinstance(self.sensor_time_ns, bool)
            or isinstance(self.pose_time_ns, bool)
            or not isinstance(self.sensor_time_ns, Integral)
            or not isinstance(self.pose_time_ns, Integral)
            or self.sensor_time_ns <= 0
            or self.pose_time_ns <= 0
        ):
            raise ValueError(
                "odometry timestamps must be positive absolute nanoseconds"
            )
        if self.lookup_semantics != STAMPED_LOOKUP_SEMANTICS:
            raise ValueError(
                "latest-TF semantics are forbidden; lookup_semantics must be "
                f"{STAMPED_LOOKUP_SEMANTICS!r}"
            )
        if self.pose_time_ns != self.sensor_time_ns:
            raise ValueError("pose_time_ns must equal the odometry sensor_time_ns")
        if not self.frame_id.strip() or not self.child_frame_id.strip():
            raise ValueError("odometry frame_id and child_frame_id are required")
        if self.frame_id == self.child_frame_id:
            raise ValueError("odometry parent and child frames must differ")
        if not self.source_topic.strip() or not self.traversal_id.strip():
            raise ValueError("odometry source_topic and traversal_id are required")

        position = _array(self.position_m, (3,), "position_m")
        orientation = _array(self.orientation_xyzw, (4,), "orientation_xyzw")
        norm = float(np.linalg.norm(orientation))
        if norm <= 1.0e-12:
            raise ValueError("orientation_xyzw must be non-zero")
        orientation /= norm

        covariance = np.asarray(self.pose_covariance, dtype=np.float64)
        if covariance.shape == (36,):
            covariance = covariance.reshape(6, 6)
        if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
            raise ValueError("pose_covariance must be a finite 6x6 or 36-value array")
        if not np.allclose(covariance, covariance.T, atol=1.0e-10):
            raise ValueError("pose_covariance must be symmetric")
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1.0e-10:
            raise ValueError("pose_covariance must be positive semidefinite")

        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "orientation_xyzw", orientation)
        object.__setattr__(self, "pose_covariance", covariance.copy())
        object.__setattr__(self, "sensor_time_ns", int(self.sensor_time_ns))
        object.__setattr__(self, "pose_time_ns", int(self.pose_time_ns))

    @property
    def parent_T_child(self) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = quaternion_xyzw_to_matrix(self.orientation_xyzw)
        transform[:3, 3] = self.position_m
        return transform

    def to_pose_estimate(self) -> PoseEstimate:
        """Convert to the pipeline's generic pose contract after validation."""

        return PoseEstimate(
            sensor_time_ns=self.sensor_time_ns,
            world_T_camera=self.parent_T_child,
            covariance=self.pose_covariance,
            source=f"openvins:{self.source_topic}",
        )

    @classmethod
    def from_json_record(cls, record: Mapping[str, Any]) -> "StampedOdometry":
        """Parse the stable JSONL interchange format used by the acceptance CLI."""

        required = (
            "sensor_time_ns",
            "pose_time_ns",
            "frame_id",
            "child_frame_id",
            "position_m",
            "orientation_xyzw",
            "pose_covariance",
            "lookup_semantics",
        )
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"odometry record is missing required fields: {missing}")
        return cls(
            sensor_time_ns=record["sensor_time_ns"],
            pose_time_ns=record["pose_time_ns"],
            frame_id=str(record["frame_id"]),
            child_frame_id=str(record["child_frame_id"]),
            position_m=np.asarray(record["position_m"], dtype=np.float64),
            orientation_xyzw=np.asarray(record["orientation_xyzw"], dtype=np.float64),
            pose_covariance=np.asarray(record["pose_covariance"], dtype=np.float64),
            lookup_semantics=str(record["lookup_semantics"]),
            source_topic=str(record.get("source_topic", OPENVINS_ODOMETRY_TOPIC)),
            traversal_id=str(record.get("traversal_id", "trajectory")),
        )


class OpenVinsOdometryStream:
    """Strictly ordered, fixed-frame OpenVINS odometry sequence."""

    def __init__(self, samples: Iterable[StampedOdometry] = ()) -> None:
        self._samples: list[StampedOdometry] = []
        self._frames: tuple[str, str] | None = None
        self._source_topic: str | None = None
        for sample in samples:
            self.append(sample)

    @property
    def samples(self) -> tuple[StampedOdometry, ...]:
        return tuple(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def append(self, sample: StampedOdometry) -> None:
        if self._samples and sample.sensor_time_ns <= self._samples[-1].sensor_time_ns:
            raise ValueError("OpenVINS odometry timestamps must increase strictly")
        frames = (sample.frame_id, sample.child_frame_id)
        if self._frames is None:
            self._frames = frames
            self._source_topic = sample.source_topic
        elif frames != self._frames:
            raise ValueError(
                f"OpenVINS odometry frames changed from {self._frames} to {frames}"
            )
        elif sample.source_topic != self._source_topic:
            raise ValueError("OpenVINS odometry source topic changed within one stream")
        self._samples.append(sample)

    def traversals(self) -> dict[str, tuple[StampedOdometry, ...]]:
        result: dict[str, list[StampedOdometry]] = {}
        for sample in self._samples:
            result.setdefault(sample.traversal_id, []).append(sample)
        return {name: tuple(values) for name, values in result.items()}


def load_odometry_jsonl(path: Path | str) -> OpenVinsOdometryStream:
    """Load and validate OpenVINS odometry records from JSON Lines."""

    source = Path(path)
    stream = OpenVinsOdometryStream()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                stream.append(StampedOdometry.from_json_record(record))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid odometry JSONL line {line_number}: {error}"
                ) from error
    if not stream.samples:
        raise ValueError(f"odometry JSONL contains no samples: {source}")
    return stream
