"""3D dynamic-object state, association, prediction, and static promotion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import uuid
from typing import Mapping, Optional

import numpy as np


class ObjectState(str, Enum):
    TENTATIVE = "tentative"
    DYNAMIC = "dynamic"
    STATIONARY_CANDIDATE = "stationary_candidate"
    PROMOTED_STATIC = "promoted_static"
    OCCLUDED = "occluded"
    EXPIRED = "expired"


@dataclass(frozen=True)
class DynamicLayerConfig:
    process_acceleration_std_mps2: float = 1.0
    default_position_std_m: float = 0.15
    default_velocity_std_mps: float = 2.0
    association_distance_m: float = 1.0
    moving_score_threshold: float = 0.55
    moving_speed_threshold_mps: float = 0.20
    stable_speed_threshold_mps: float = 0.08
    stable_duration_s: float = 2.0
    stable_observations: int = 5
    occluded_after_s: float = 0.5
    remove_after_s: float = 5.0
    trajectory_limit: int = 512

    def __post_init__(self) -> None:
        positive = (
            self.process_acceleration_std_mps2,
            self.default_position_std_m,
            self.default_velocity_std_mps,
            self.association_distance_m,
            self.stable_duration_s,
            self.occluded_after_s,
            self.remove_after_s,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("dynamic layer distances and durations must be positive")
        if self.stable_observations < 2 or self.trajectory_limit < 2:
            raise ValueError("dynamic layer observation limits are too small")
        if self.remove_after_s <= self.occluded_after_s:
            raise ValueError("remove_after_s must exceed occluded_after_s")


@dataclass(frozen=True)
class ObjectObservation:
    track_id: int
    sensor_time_ns: int
    position_m: np.ndarray
    dimensions_m: np.ndarray
    position_covariance: np.ndarray
    semantic_probabilities: Mapping[str, float] = field(default_factory=dict)
    motion_score: float = 0.0
    entity_id: Optional[str] = None

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64)
        dimensions = np.asarray(self.dimensions_m, dtype=np.float64)
        covariance = np.asarray(self.position_covariance, dtype=np.float64)
        if self.track_id < 0 or self.sensor_time_ns <= 0:
            raise ValueError("observation track/time is invalid")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("observation position must be finite xyz")
        if dimensions.shape != (3,) or np.any(dimensions <= 0):
            raise ValueError("object dimensions must be positive xyz")
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            raise ValueError("position covariance must be finite 3x3")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("position covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-10:
            raise ValueError("position covariance must be positive semidefinite")
        if not 0.0 <= self.motion_score <= 1.0:
            raise ValueError("motion_score must be in [0, 1]")
        probabilities = {str(key): float(value) for key, value in self.semantic_probabilities.items()}
        if any(value < 0.0 for value in probabilities.values()):
            raise ValueError("semantic probabilities cannot be negative")
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {key: value / total for key, value in probabilities.items()}
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "dimensions_m", dimensions)
        object.__setattr__(self, "position_covariance", covariance)
        object.__setattr__(self, "semantic_probabilities", probabilities)


@dataclass
class DynamicObject:
    entity_id: str
    track_ids: set[int]
    state_vector: np.ndarray
    covariance: np.ndarray
    dimensions_m: np.ndarray
    semantic_probabilities: dict[str, float]
    state: ObjectState
    first_seen_ns: int
    last_seen_ns: int
    last_update_ns: int
    stable_since_ns: Optional[int] = None
    stable_observations: int = 0
    observation_count: int = 1
    trajectory: list[dict] = field(default_factory=list)

    @property
    def position_m(self) -> np.ndarray:
        return self.state_vector[:3].copy()

    @property
    def velocity_mps(self) -> np.ndarray:
        return self.state_vector[3:].copy()


class DynamicLayer:
    """Constant-velocity object layer with conservative lifecycle transitions."""

    def __init__(self, config: DynamicLayerConfig = DynamicLayerConfig()) -> None:
        self.config = config
        self._active: dict[str, DynamicObject] = {}
        self._history: dict[str, DynamicObject] = {}
        self._track_to_entity: dict[int, str] = {}

    @property
    def active_objects(self) -> Mapping[str, DynamicObject]:
        return self._active

    @property
    def history(self) -> Mapping[str, DynamicObject]:
        return self._history

    def _predict(self, obj: DynamicObject, timestamp_ns: int) -> None:
        if timestamp_ns < obj.last_update_ns:
            raise ValueError("object updates must be chronological")
        dt = (timestamp_ns - obj.last_update_ns) / 1e9
        if dt <= 0:
            return
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        acceleration_variance = self.config.process_acceleration_std_mps2**2
        process = np.block(
            [
                [np.eye(3) * dt**4 / 4.0, np.eye(3) * dt**3 / 2.0],
                [np.eye(3) * dt**3 / 2.0, np.eye(3) * dt**2],
            ]
        ) * acceleration_variance
        obj.state_vector = transition @ obj.state_vector
        obj.covariance = transition @ obj.covariance @ transition.T + process
        obj.last_update_ns = timestamp_ns

    def _associate(self, observation: ObjectObservation) -> Optional[str]:
        if observation.entity_id in self._active:
            return observation.entity_id
        mapped = self._track_to_entity.get(observation.track_id)
        if mapped in self._active:
            return mapped
        best_entity = None
        best_distance = float("inf")
        for entity_id, obj in self._active.items():
            if observation.sensor_time_ns < obj.last_seen_ns:
                continue
            dt = (observation.sensor_time_ns - obj.last_update_ns) / 1e9
            predicted = obj.position_m + obj.velocity_mps * max(0.0, dt)
            distance = float(np.linalg.norm(observation.position_m - predicted))
            if distance > self.config.association_distance_m:
                continue
            observed_labels = set(observation.semantic_probabilities)
            known_labels = set(obj.semantic_probabilities)
            if observed_labels and known_labels and not (observed_labels & known_labels):
                continue
            if distance < best_distance:
                best_entity = entity_id
                best_distance = distance
        return best_entity

    def _new_object(self, observation: ObjectObservation) -> DynamicObject:
        entity_id = observation.entity_id or f"object-{uuid.uuid4().hex}"
        if entity_id in self._history:
            raise ValueError(f"entity_id has already expired: {entity_id}")
        covariance = np.eye(6, dtype=np.float64)
        covariance[:3, :3] = observation.position_covariance + np.eye(3) * 1e-9
        covariance[3:, 3:] *= self.config.default_velocity_std_mps**2
        state = (
            ObjectState.DYNAMIC
            if observation.motion_score >= self.config.moving_score_threshold
            else ObjectState.TENTATIVE
        )
        obj = DynamicObject(
            entity_id=entity_id,
            track_ids={observation.track_id},
            state_vector=np.r_[observation.position_m, np.zeros(3)],
            covariance=covariance,
            dimensions_m=observation.dimensions_m.copy(),
            semantic_probabilities=dict(observation.semantic_probabilities),
            state=state,
            first_seen_ns=observation.sensor_time_ns,
            last_seen_ns=observation.sensor_time_ns,
            last_update_ns=observation.sensor_time_ns,
            trajectory=[],
        )
        self._record_trajectory(obj, observation.sensor_time_ns, observed=True)
        self._active[entity_id] = obj
        self._track_to_entity[observation.track_id] = entity_id
        return obj

    def _record_trajectory(
        self, obj: DynamicObject, timestamp_ns: int, *, observed: bool
    ) -> None:
        obj.trajectory.append(
            {
                "sensor_time_ns": int(timestamp_ns),
                "position_m": obj.position_m.tolist(),
                "velocity_mps": obj.velocity_mps.tolist(),
                "observed": observed,
                "state": obj.state.value,
            }
        )
        if len(obj.trajectory) > self.config.trajectory_limit:
            del obj.trajectory[: len(obj.trajectory) - self.config.trajectory_limit]

    def update(self, observation: ObjectObservation) -> DynamicObject:
        entity_id = self._associate(observation)
        if entity_id is None:
            return self._new_object(observation)
        obj = self._active[entity_id]
        if observation.sensor_time_ns <= obj.last_seen_ns:
            raise ValueError("object observations must be strictly chronological")
        first_velocity_initialization = obj.observation_count == 1
        previous_position = obj.position_m
        observation_dt = (observation.sensor_time_ns - obj.last_seen_ns) / 1e9
        self._predict(obj, observation.sensor_time_ns)

        measurement = observation.position_m
        measurement_matrix = np.zeros((3, 6), dtype=np.float64)
        measurement_matrix[:, :3] = np.eye(3)
        residual = measurement - measurement_matrix @ obj.state_vector
        innovation = (
            measurement_matrix @ obj.covariance @ measurement_matrix.T
            + observation.position_covariance
            + np.eye(3) * 1e-9
        )
        gain = obj.covariance @ measurement_matrix.T @ np.linalg.inv(innovation)
        obj.state_vector = obj.state_vector + gain @ residual
        identity = np.eye(6)
        # Joseph form keeps covariance symmetric and positive semidefinite.
        correction = identity - gain @ measurement_matrix
        obj.covariance = (
            correction @ obj.covariance @ correction.T
            + gain @ observation.position_covariance @ gain.T
        )
        obj.covariance = (obj.covariance + obj.covariance.T) / 2.0
        if first_velocity_initialization and observation_dt > 0.0:
            obj.state_vector[3:] = (
                observation.position_m - previous_position
            ) / observation_dt
            obj.covariance[3:, 3:] = (
                observation.position_covariance
                + np.eye(3) * self.config.default_position_std_m**2
            ) / observation_dt**2
        obj.dimensions_m = 0.8 * obj.dimensions_m + 0.2 * observation.dimensions_m
        obj.track_ids.add(observation.track_id)
        self._track_to_entity[observation.track_id] = entity_id
        obj.last_seen_ns = observation.sensor_time_ns
        obj.observation_count += 1

        for label in set(obj.semantic_probabilities) | set(observation.semantic_probabilities):
            previous = obj.semantic_probabilities.get(label, 0.0)
            measured = observation.semantic_probabilities.get(label, 0.0)
            obj.semantic_probabilities[label] = 0.8 * previous + 0.2 * measured
        semantic_total = sum(obj.semantic_probabilities.values())
        if semantic_total > 0:
            obj.semantic_probabilities = {
                label: probability / semantic_total
                for label, probability in obj.semantic_probabilities.items()
            }

        speed = float(np.linalg.norm(obj.velocity_mps))
        moving = (
            observation.motion_score >= self.config.moving_score_threshold
            or speed >= self.config.moving_speed_threshold_mps
        )
        stable = (
            observation.motion_score < self.config.moving_score_threshold
            and speed <= self.config.stable_speed_threshold_mps
        )
        if moving:
            obj.state = ObjectState.DYNAMIC
            obj.stable_since_ns = None
            obj.stable_observations = 0
        elif stable:
            if obj.stable_since_ns is None:
                obj.stable_since_ns = observation.sensor_time_ns
                obj.stable_observations = 1
            else:
                obj.stable_observations += 1
            stable_duration = (
                observation.sensor_time_ns - obj.stable_since_ns
            ) / 1e9
            if (
                stable_duration >= self.config.stable_duration_s
                and obj.stable_observations >= self.config.stable_observations
            ):
                obj.state = ObjectState.PROMOTED_STATIC
            else:
                obj.state = ObjectState.STATIONARY_CANDIDATE
        else:
            obj.state = ObjectState.TENTATIVE
        self._record_trajectory(obj, observation.sensor_time_ns, observed=True)
        return obj

    def advance_time(self, sensor_time_ns: int) -> list[str]:
        """Predict missing objects, expire stale ones, and retain trajectory history."""
        expired = []
        for entity_id, obj in list(self._active.items()):
            if sensor_time_ns < obj.last_seen_ns:
                raise ValueError("layer time cannot move backwards")
            gap_s = (sensor_time_ns - obj.last_seen_ns) / 1e9
            self._predict(obj, sensor_time_ns)
            if gap_s >= self.config.remove_after_s:
                obj.state = ObjectState.EXPIRED
                self._record_trajectory(obj, sensor_time_ns, observed=False)
                self._history[entity_id] = obj
                del self._active[entity_id]
                expired.append(entity_id)
                continue
            if gap_s >= self.config.occluded_after_s:
                obj.state = ObjectState.OCCLUDED
                self._record_trajectory(obj, sensor_time_ns, observed=False)
        return expired

    def snapshot(self) -> dict:
        def serialize(obj: DynamicObject) -> dict:
            return {
                "entity_id": obj.entity_id,
                "track_ids": sorted(obj.track_ids),
                "state_vector": obj.state_vector.tolist(),
                "covariance": obj.covariance.tolist(),
                "dimensions_m": obj.dimensions_m.tolist(),
                "semantic_probabilities": obj.semantic_probabilities,
                "state": obj.state.value,
                "first_seen_ns": obj.first_seen_ns,
                "last_seen_ns": obj.last_seen_ns,
                "last_update_ns": obj.last_update_ns,
                "stable_since_ns": obj.stable_since_ns,
                "stable_observations": obj.stable_observations,
                "observation_count": obj.observation_count,
                "trajectory": obj.trajectory,
            }

        return {
            "active": [serialize(obj) for obj in self._active.values()],
            "history": [serialize(obj) for obj in self._history.values()],
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict,
        config: DynamicLayerConfig = DynamicLayerConfig(),
    ) -> "DynamicLayer":
        layer = cls(config)

        def deserialize(data: dict) -> DynamicObject:
            return DynamicObject(
                entity_id=str(data["entity_id"]),
                track_ids={int(value) for value in data["track_ids"]},
                state_vector=np.asarray(data["state_vector"], dtype=np.float64),
                covariance=np.asarray(data["covariance"], dtype=np.float64),
                dimensions_m=np.asarray(data["dimensions_m"], dtype=np.float64),
                semantic_probabilities={
                    str(key): float(value)
                    for key, value in data["semantic_probabilities"].items()
                },
                state=ObjectState(data["state"]),
                first_seen_ns=int(data["first_seen_ns"]),
                last_seen_ns=int(data["last_seen_ns"]),
                last_update_ns=int(data["last_update_ns"]),
                stable_since_ns=(
                    int(data["stable_since_ns"])
                    if data.get("stable_since_ns") is not None
                    else None
                ),
                stable_observations=int(data["stable_observations"]),
                observation_count=int(data["observation_count"]),
                trajectory=list(data["trajectory"]),
            )

        for data in snapshot.get("active", []):
            obj = deserialize(data)
            layer._active[obj.entity_id] = obj
            for track_id in obj.track_ids:
                layer._track_to_entity[track_id] = obj.entity_id
        for data in snapshot.get("history", []):
            obj = deserialize(data)
            layer._history[obj.entity_id] = obj
        return layer
