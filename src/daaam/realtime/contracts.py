"""Versioned absolute-time contracts shared by realtime pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import time
from typing import Any, Generic, Mapping, Optional, TypeVar

import numpy as np


PayloadT = TypeVar("PayloadT")


class FrameValue(IntEnum):
    """Content value used by bounded queues; larger values are retained first."""

    STRICT_DUPLICATE = 0
    ROUTINE = 10
    WATCHDOG = 20
    POSE_MOTION = 30
    IMAGE_EVENT_AT_STATIC_POSE = 40
    LOOP_CANDIDATE = 50

    @classmethod
    def parse(cls, value: "FrameValue | str | int") -> "FrameValue":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().upper()
            try:
                return cls[normalized]
            except KeyError as error:
                raise ValueError(f"Unknown frame value: {value}") from error
        return cls(int(value))


@dataclass(frozen=True)
class MessageKey:
    """Identity for a result that may cross asynchronous stage boundaries."""

    sensor_time_ns: int
    map_revision: int = 0
    calibration_revision: int = 0
    entity_id: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.sensor_time_ns, bool) or self.sensor_time_ns <= 0:
            raise ValueError("sensor_time_ns must be a positive absolute nanosecond value")
        if self.map_revision < 0:
            raise ValueError("map_revision must be non-negative")
        if self.calibration_revision < 0:
            raise ValueError("calibration_revision must be non-negative")
        if self.entity_id is not None and not str(self.entity_id).strip():
            raise ValueError("entity_id cannot be empty")


@dataclass(frozen=True)
class PoseEstimate:
    """SE(3) camera pose and uncertainty at an absolute capture time."""

    sensor_time_ns: int
    world_T_camera: np.ndarray
    covariance: np.ndarray
    source: str
    status: str = "tracking"

    def __post_init__(self) -> None:
        transform = np.asarray(self.world_T_camera, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if self.sensor_time_ns <= 0:
            raise ValueError("pose sensor_time_ns must be positive")
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("world_T_camera must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("world_T_camera must be a homogeneous transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("world_T_camera rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.999 or np.linalg.det(rotation) > 1.001:
            raise ValueError("world_T_camera rotation determinant must be one")
        if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
            raise ValueError("pose covariance must be a finite 6x6 matrix")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("pose covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-10:
            raise ValueError("pose covariance must be positive semidefinite")
        if not self.source.strip():
            raise ValueError("pose source is required")
        object.__setattr__(self, "world_T_camera", transform)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class SemanticCorrection:
    """Idempotent semantic update addressed to a stable map entity."""

    operation_id: str
    entity_id: str
    sensor_time_ns: int
    map_revision: int
    label: str
    confidence: float
    source: str = "automatic"
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.operation_id.strip():
            raise ValueError("operation_id is required")
        if not self.entity_id.strip():
            raise ValueError("entity_id is required")
        if self.sensor_time_ns <= 0 or self.map_revision < 0:
            raise ValueError("semantic correction time/revision is invalid")
        if not self.label.strip():
            raise ValueError("semantic label is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("semantic confidence must be in [0, 1]")
        object.__setattr__(self, "aliases", tuple(dict.fromkeys(self.aliases)))


@dataclass(frozen=True)
class MapUpdate:
    """Map revision transition emitted after a verified global correction."""

    previous_revision: int
    map_revision: int
    sensor_time_ns: int
    reason: str
    transforms: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.previous_revision < 0 or self.map_revision <= self.previous_revision:
            raise ValueError("map revision must increase monotonically")
        if self.sensor_time_ns <= 0 or not self.reason.strip():
            raise ValueError("map update requires time and reason")
        checked = {}
        for identifier, transform in self.transforms.items():
            matrix = np.asarray(transform, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"invalid transform for {identifier}")
            checked[str(identifier)] = matrix
        object.__setattr__(self, "transforms", checked)


@dataclass(frozen=True)
class RealtimeEnvelope(Generic[PayloadT]):
    """Transport envelope with capture identity and host-clock scheduling fields."""

    key: MessageKey
    payload: PayloadT
    value: FrameValue = FrameValue.ROUTINE
    source: str = "unknown"
    created_monotonic_ns: int = field(default_factory=time.monotonic_ns)
    deadline_monotonic_ns: Optional[int] = None
    trace_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", FrameValue.parse(self.value))
        if self.created_monotonic_ns <= 0:
            raise ValueError("created_monotonic_ns must be positive")
        if (
            self.deadline_monotonic_ns is not None
            and self.deadline_monotonic_ns < self.created_monotonic_ns
        ):
            raise ValueError("deadline cannot precede envelope creation")
        if not self.source.strip():
            raise ValueError("envelope source is required")
        payload_time = getattr(self.payload, "sensor_time_ns", None)
        if payload_time is not None and int(payload_time) != self.key.sensor_time_ns:
            raise ValueError("payload and envelope sensor_time_ns disagree")
        payload_revision = getattr(self.payload, "map_revision", None)
        if payload_revision is not None and int(payload_revision) != self.key.map_revision:
            raise ValueError("payload and envelope map_revision disagree")

    @property
    def identity(self) -> tuple[int, Optional[str], int, int]:
        return (
            self.key.sensor_time_ns,
            self.key.entity_id,
            self.key.map_revision,
            self.key.calibration_revision,
        )

    def is_expired(self, now_monotonic_ns: Optional[int] = None) -> bool:
        if self.deadline_monotonic_ns is None:
            return False
        now = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        return now > self.deadline_monotonic_ns
