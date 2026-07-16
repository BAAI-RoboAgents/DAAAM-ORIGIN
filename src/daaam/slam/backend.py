"""Validation contract for external VIO, LIO, and odometry pose streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from daaam.realtime.contracts import PoseEstimate


class PoseBackend(Protocol):
    def poll(self, timeout_s: float) -> Optional[PoseEstimate]: ...


@dataclass(frozen=True)
class PoseBackendConfig:
    maximum_gap_s: float = 0.5
    maximum_clock_jump_s: float = 2.0
    maximum_position_std_m: float = 1.0
    maximum_rotation_std_deg: float = 30.0

    def __post_init__(self) -> None:
        if min(
            self.maximum_gap_s,
            self.maximum_clock_jump_s,
            self.maximum_position_std_m,
            self.maximum_rotation_std_deg,
        ) <= 0:
            raise ValueError("pose backend thresholds must be positive")


@dataclass(frozen=True)
class PoseValidation:
    accepted: bool
    status: str
    reason: str
    gap_s: Optional[float]


class PoseInputValidator:
    """Stateful time/covariance gate in front of any external pose backend."""

    def __init__(self, config: PoseBackendConfig = PoseBackendConfig()) -> None:
        self.config = config
        self._last_time_ns: Optional[int] = None
        self._calibration_revision: Optional[int] = None

    def validate(
        self, estimate: PoseEstimate, *, calibration_revision: int
    ) -> PoseValidation:
        if calibration_revision < 0:
            return PoseValidation(False, "rejected", "invalid_calibration_revision", None)
        if self._calibration_revision is None:
            self._calibration_revision = calibration_revision
        elif calibration_revision != self._calibration_revision:
            return PoseValidation(False, "rejected", "calibration_revision_changed", None)

        gap_s = None
        if self._last_time_ns is not None:
            delta_ns = estimate.sensor_time_ns - self._last_time_ns
            if delta_ns <= 0:
                return PoseValidation(False, "rejected", "non_monotonic_time", delta_ns / 1e9)
            gap_s = delta_ns / 1e9
            if gap_s > self.config.maximum_clock_jump_s:
                return PoseValidation(False, "rejected", "clock_jump", gap_s)

        position_std = float(np.sqrt(np.max(np.diag(estimate.covariance)[:3])))
        rotation_std_deg = float(
            np.rad2deg(np.sqrt(np.max(np.diag(estimate.covariance)[3:])))
        )
        if position_std > self.config.maximum_position_std_m:
            return PoseValidation(False, "rejected", "position_covariance", gap_s)
        if rotation_std_deg > self.config.maximum_rotation_std_deg:
            return PoseValidation(False, "rejected", "rotation_covariance", gap_s)

        self._last_time_ns = estimate.sensor_time_ns
        if gap_s is not None and gap_s > self.config.maximum_gap_s:
            return PoseValidation(True, "degraded", "pose_gap", gap_s)
        return PoseValidation(True, "tracking", "accepted", gap_s)
