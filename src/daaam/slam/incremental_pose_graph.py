"""Fixed-window SE(3) pose graph with verified-loop revision updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from daaam.realtime.contracts import PoseEstimate


def _check_transform(transform: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must be homogeneous")
    return matrix


def _se3_increment(vector: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(vector[3:]).as_matrix()
    transform[:3, 3] = vector[:3]
    return transform


def _se3_residual(transform: np.ndarray) -> np.ndarray:
    return np.r_[
        transform[:3, 3],
        Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
    ]


@dataclass(frozen=True)
class PoseGraphConfig:
    window_size: int = 20
    maximum_gravity_residual_deg: float = 8.0
    minimum_loop_verification_score: float = 0.5
    robust_loss_scale: float = 1.0
    maximum_iterations: int = 40

    def __post_init__(self) -> None:
        if self.window_size < 3 or self.maximum_iterations <= 0:
            raise ValueError("pose graph window/iteration settings are invalid")
        if self.maximum_gravity_residual_deg <= 0 or self.robust_loss_scale <= 0:
            raise ValueError("pose graph thresholds must be positive")
        if not 0.0 <= self.minimum_loop_verification_score <= 1.0:
            raise ValueError("loop verification score must be in [0, 1]")


@dataclass(frozen=True)
class PoseConstraint:
    source_time_ns: int
    target_time_ns: int
    source_T_target: np.ndarray
    covariance: np.ndarray
    kind: str = "odometry"
    geometrically_verified: bool = True
    verification_score: float = 1.0
    gravity_residual_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.source_time_ns <= 0 or self.target_time_ns <= 0:
            raise ValueError("constraint timestamps must be positive")
        if self.source_time_ns == self.target_time_ns:
            raise ValueError("constraint endpoints must differ")
        transform = _check_transform(self.source_T_target, "source_T_target")
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
            raise ValueError("constraint covariance must be finite 6x6")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("constraint covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) <= 0:
            raise ValueError("constraint covariance must be positive definite")
        if self.kind not in {"odometry", "rgbd", "loop"}:
            raise ValueError(f"unsupported constraint kind: {self.kind}")
        if not 0.0 <= self.verification_score <= 1.0:
            raise ValueError("verification score must be in [0, 1]")
        object.__setattr__(self, "source_T_target", transform)
        object.__setattr__(self, "covariance", covariance)


@dataclass
class _PoseNode:
    estimate: PoseEstimate
    transform: np.ndarray


@dataclass(frozen=True)
class PoseGraphReport:
    success: bool
    optimized_nodes: int
    constraints_used: int
    cost_before: float
    cost_after: float
    map_revision: int
    loop_applied: bool
    rejected_loops: int
    message: str


class IncrementalPoseGraph:
    """Optimize only a fixed recent window while older poses remain anchors."""

    def __init__(self, config: PoseGraphConfig = PoseGraphConfig()) -> None:
        self.config = config
        self._nodes: dict[int, _PoseNode] = {}
        self._ordered_times: list[int] = []
        self._constraints: list[PoseConstraint] = []
        self._accepted_loop_count = 0
        self._rejected_loop_count = 0
        self._last_optimized_loop_count = 0
        self.map_revision = 0

    @property
    def timestamps(self) -> tuple[int, ...]:
        return tuple(self._ordered_times)

    def add_pose(self, estimate: PoseEstimate) -> None:
        if estimate.sensor_time_ns in self._nodes:
            raise ValueError("duplicate pose timestamp")
        if self._ordered_times and estimate.sensor_time_ns <= self._ordered_times[-1]:
            raise ValueError("poses must be added in strictly increasing time order")
        self._nodes[estimate.sensor_time_ns] = _PoseNode(
            estimate, estimate.world_T_camera.copy()
        )
        self._ordered_times.append(estimate.sensor_time_ns)

    def add_constraint(self, constraint: PoseConstraint) -> bool:
        if (
            constraint.source_time_ns not in self._nodes
            or constraint.target_time_ns not in self._nodes
        ):
            raise KeyError("constraint references an unknown pose")
        if constraint.kind == "loop":
            accepted = (
                constraint.geometrically_verified
                and constraint.verification_score
                >= self.config.minimum_loop_verification_score
                and constraint.gravity_residual_deg
                <= self.config.maximum_gravity_residual_deg
            )
            if not accepted:
                self._rejected_loop_count += 1
                return False
            self._accepted_loop_count += 1
        self._constraints.append(constraint)
        return True

    def pose(self, sensor_time_ns: int) -> np.ndarray:
        return self._nodes[sensor_time_ns].transform.copy()

    def relative_pose(self, source_time_ns: int, target_time_ns: int) -> np.ndarray:
        return np.linalg.inv(self.pose(source_time_ns)) @ self.pose(target_time_ns)

    def optimize(self) -> PoseGraphReport:
        if len(self._ordered_times) < 2:
            return PoseGraphReport(
                False,
                0,
                0,
                0.0,
                0.0,
                self.map_revision,
                False,
                self._rejected_loop_count,
                "not enough poses",
            )
        from scipy.optimize import least_squares

        active_times = self._ordered_times[-self.config.window_size :]
        variable_times = active_times[1:]
        variable_index = {timestamp: index for index, timestamp in enumerate(variable_times)}
        base = {timestamp: self._nodes[timestamp].transform.copy() for timestamp in active_times}
        constraints = [
            constraint
            for constraint in self._constraints
            if (
                constraint.source_time_ns in self._nodes
                and constraint.target_time_ns in self._nodes
                and (
                    constraint.source_time_ns in active_times
                    or constraint.target_time_ns in active_times
                )
            )
        ]
        if not constraints or not variable_times:
            return PoseGraphReport(
                False,
                0,
                len(constraints),
                0.0,
                0.0,
                self.map_revision,
                False,
                self._rejected_loop_count,
                "no optimizable constraints",
            )

        square_root_information = {
            id(constraint): np.linalg.cholesky(np.linalg.inv(constraint.covariance))
            for constraint in constraints
        }

        def transform_for(timestamp: int, parameters: np.ndarray) -> np.ndarray:
            if timestamp not in variable_index:
                return self._nodes[timestamp].transform
            offset = variable_index[timestamp] * 6
            return _se3_increment(parameters[offset : offset + 6]) @ base[timestamp]

        def residuals(parameters: np.ndarray) -> np.ndarray:
            values = []
            for constraint in constraints:
                source = transform_for(constraint.source_time_ns, parameters)
                target = transform_for(constraint.target_time_ns, parameters)
                predicted = np.linalg.inv(source) @ target
                error = np.linalg.inv(constraint.source_T_target) @ predicted
                values.append(
                    square_root_information[id(constraint)] @ _se3_residual(error)
                )
            return np.concatenate(values)

        initial = np.zeros(len(variable_times) * 6, dtype=np.float64)
        before_vector = residuals(initial)
        cost_before = float(0.5 * before_vector @ before_vector)
        result = least_squares(
            residuals,
            initial,
            loss="huber",
            f_scale=self.config.robust_loss_scale,
            max_nfev=self.config.maximum_iterations,
        )
        after_vector = residuals(result.x)
        cost_after = float(0.5 * after_vector @ after_vector)
        if result.success:
            for timestamp in variable_times:
                self._nodes[timestamp].transform = transform_for(timestamp, result.x)

        loop_applied = (
            result.success
            and self._accepted_loop_count > self._last_optimized_loop_count
        )
        if loop_applied:
            self.map_revision += 1
            self._last_optimized_loop_count = self._accepted_loop_count
        return PoseGraphReport(
            bool(result.success),
            len(variable_times),
            len(constraints),
            cost_before,
            cost_after,
            self.map_revision,
            loop_applied,
            self._rejected_loop_count,
            str(result.message),
        )
