"""Direction-invariant merging of repeated robot traversals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
import uuid

import numpy as np


@dataclass(frozen=True)
class PathMergeConfig:
    sample_count: int = 64
    maximum_distance_m: float = 0.25
    minimum_overlap_ratio: float = 0.75
    minimum_length_ratio: float = 0.65

    def __post_init__(self) -> None:
        if self.sample_count < 4 or self.maximum_distance_m <= 0:
            raise ValueError("path sampling settings are invalid")
        if not 0.0 < self.minimum_overlap_ratio <= 1.0:
            raise ValueError("minimum_overlap_ratio must be in (0, 1]")
        if not 0.0 < self.minimum_length_ratio <= 1.0:
            raise ValueError("minimum_length_ratio must be in (0, 1]")


@dataclass(frozen=True)
class PathObservation:
    session_id: str
    sensor_times_ns: np.ndarray
    points_m: np.ndarray
    map_revision: int

    def __post_init__(self) -> None:
        times = np.asarray(self.sensor_times_ns, dtype=np.int64)
        points = np.asarray(self.points_m, dtype=np.float64)
        if not self.session_id.strip() or self.map_revision < 0:
            raise ValueError("path session/revision is invalid")
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("path points must be Nx3 with at least two points")
        if times.shape != (len(points),) or np.any(np.diff(times) <= 0):
            raise ValueError("path times must match points and increase strictly")
        if np.any(times <= 0) or not np.all(np.isfinite(points)):
            raise ValueError("path contains invalid time or position")
        object.__setattr__(self, "sensor_times_ns", times)
        object.__setattr__(self, "points_m", points)


@dataclass
class CanonicalPath:
    path_id: str
    points_m: np.ndarray
    observations: list[PathObservation] = field(default_factory=list)
    start_time_ns: int = 0
    end_time_ns: int = 0


def _length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    if cumulative[-1] <= 1e-9:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.stack(
        [np.interp(targets, cumulative, points[:, axis]) for axis in range(3)],
        axis=1,
    )


class PathRepository:
    def __init__(self, config: PathMergeConfig = PathMergeConfig()) -> None:
        self.config = config
        self._paths: dict[str, CanonicalPath] = {}

    @property
    def paths(self) -> Mapping[str, CanonicalPath]:
        return self._paths

    def _similarity(
        self, candidate: np.ndarray, reference: np.ndarray
    ) -> tuple[bool, bool, float, float]:
        first = _resample(candidate, self.config.sample_count)
        second = _resample(reference, self.config.sample_count)
        forward_error = float(np.mean(np.linalg.norm(first - second, axis=1)))
        reverse_error = float(np.mean(np.linalg.norm(first - second[::-1], axis=1)))
        reversed_direction = reverse_error < forward_error
        aligned_second = second[::-1] if reversed_direction else second
        pair_distances = np.linalg.norm(first - aligned_second, axis=1)
        overlap = float(np.mean(pair_distances <= self.config.maximum_distance_m))
        length_ratio = min(_length(candidate), _length(reference)) / max(
            _length(candidate), _length(reference), 1e-9
        )
        matched = (
            overlap >= self.config.minimum_overlap_ratio
            and length_ratio >= self.config.minimum_length_ratio
            and float(np.median(pair_distances)) <= self.config.maximum_distance_m
        )
        return matched, reversed_direction, overlap, float(np.median(pair_distances))

    def add(self, observation: PathObservation) -> tuple[CanonicalPath, dict]:
        best = None
        for path in self._paths.values():
            matched, reversed_direction, overlap, distance = self._similarity(
                observation.points_m, path.points_m
            )
            if matched and (best is None or distance < best[0]):
                best = (distance, path, reversed_direction, overlap)
        if best is None:
            path = CanonicalPath(
                path_id=f"path-{uuid.uuid4().hex}",
                points_m=observation.points_m.copy(),
                observations=[observation],
                start_time_ns=int(observation.sensor_times_ns[0]),
                end_time_ns=int(observation.sensor_times_ns[-1]),
            )
            self._paths[path.path_id] = path
            return path, {
                "merged": False,
                "reversed": False,
                "overlap_ratio": 0.0,
                "median_distance_m": None,
            }

        distance, path, reversed_direction, overlap = best
        incoming = _resample(observation.points_m, self.config.sample_count)
        if reversed_direction:
            incoming = incoming[::-1]
        current = _resample(path.points_m, self.config.sample_count)
        count = len(path.observations)
        path.points_m = (current * count + incoming) / (count + 1)
        path.observations.append(observation)
        path.start_time_ns = min(path.start_time_ns, int(observation.sensor_times_ns[0]))
        path.end_time_ns = max(path.end_time_ns, int(observation.sensor_times_ns[-1]))
        return path, {
            "merged": True,
            "reversed": reversed_direction,
            "overlap_ratio": overlap,
            "median_distance_m": distance,
        }

    def snapshot(self) -> dict:
        return {
            "config": {
                "sample_count": self.config.sample_count,
                "maximum_distance_m": self.config.maximum_distance_m,
                "minimum_overlap_ratio": self.config.minimum_overlap_ratio,
                "minimum_length_ratio": self.config.minimum_length_ratio,
            },
            "paths": [
                {
                    "path_id": path.path_id,
                    "points_m": path.points_m.tolist(),
                    "start_time_ns": path.start_time_ns,
                    "end_time_ns": path.end_time_ns,
                    "observations": [
                        {
                            "session_id": observation.session_id,
                            "sensor_times_ns": observation.sensor_times_ns.tolist(),
                            "points_m": observation.points_m.tolist(),
                            "map_revision": observation.map_revision,
                        }
                        for observation in path.observations
                    ],
                }
                for path in self._paths.values()
            ],
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict,
        config: PathMergeConfig | None = None,
    ) -> "PathRepository":
        repository = cls(config or PathMergeConfig(**snapshot.get("config", {})))
        for data in snapshot.get("paths", []):
            observations = [
                PathObservation(
                    session_id=str(observation["session_id"]),
                    sensor_times_ns=np.asarray(
                        observation["sensor_times_ns"], dtype=np.int64
                    ),
                    points_m=np.asarray(observation["points_m"], dtype=np.float64),
                    map_revision=int(observation["map_revision"]),
                )
                for observation in data["observations"]
            ]
            path = CanonicalPath(
                path_id=str(data["path_id"]),
                points_m=np.asarray(data["points_m"], dtype=np.float64),
                observations=observations,
                start_time_ns=int(data["start_time_ns"]),
                end_time_ns=int(data["end_time_ns"]),
            )
            if path.points_m.ndim != 2 or path.points_m.shape[1] != 3:
                raise ValueError("canonical path points must be Nx3")
            repository._paths[path.path_id] = path
        return repository
