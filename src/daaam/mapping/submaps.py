"""Revisioned submap transforms for non-blocking global corrections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from daaam.realtime.contracts import MapUpdate


def _transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("submap transform must be finite 4x4")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("submap transform must be homogeneous")
    return matrix


@dataclass
class Submap:
    submap_id: str
    start_time_ns: int
    end_time_ns: int
    frame_times_ns: list[int]
    transforms_by_revision: dict[int, np.ndarray]
    local_origin_time_ns: int

    @property
    def latest_revision(self) -> int:
        return max(self.transforms_by_revision)


class SubmapManager:
    """Keep local geometry immutable while global submap transforms change."""

    def __init__(self, *, maximum_frames: int = 30) -> None:
        if maximum_frames <= 0:
            raise ValueError("maximum_frames must be positive")
        self.maximum_frames = maximum_frames
        self.map_revision = 0
        self._submaps: dict[str, Submap] = {}
        self._current_id: Optional[str] = None
        self._next_id = 0

    @property
    def submaps(self) -> Mapping[str, Submap]:
        return self._submaps

    def add_frame(self, sensor_time_ns: int, world_T_camera: np.ndarray) -> str:
        if sensor_time_ns <= 0:
            raise ValueError("frame time must be positive")
        pose = _transform(world_T_camera)
        current = self._submaps.get(self._current_id or "")
        if current is None or len(current.frame_times_ns) >= self.maximum_frames:
            submap_id = f"submap-{self._next_id:06d}"
            self._next_id += 1
            current = Submap(
                submap_id=submap_id,
                start_time_ns=sensor_time_ns,
                end_time_ns=sensor_time_ns,
                frame_times_ns=[],
                transforms_by_revision={self.map_revision: pose.copy()},
                local_origin_time_ns=sensor_time_ns,
            )
            self._submaps[submap_id] = current
            self._current_id = submap_id
        if current.frame_times_ns and sensor_time_ns <= current.frame_times_ns[-1]:
            raise ValueError("submap frame times must be strictly increasing")
        current.frame_times_ns.append(sensor_time_ns)
        current.end_time_ns = sensor_time_ns
        return current.submap_id

    def apply_global_correction(
        self,
        transforms: Mapping[str, np.ndarray],
        *,
        sensor_time_ns: int,
        expected_revision: int,
        reason: str,
    ) -> MapUpdate:
        if expected_revision != self.map_revision:
            raise ValueError("stale map revision for global correction")
        unknown = set(transforms) - set(self._submaps)
        if unknown:
            raise KeyError(f"unknown submaps: {sorted(unknown)}")
        next_revision = self.map_revision + 1
        checked = {identifier: _transform(value) for identifier, value in transforms.items()}
        for submap_id, submap in self._submaps.items():
            previous = submap.transforms_by_revision[self.map_revision]
            submap.transforms_by_revision[next_revision] = checked.get(
                submap_id, previous.copy()
            )
        update = MapUpdate(
            previous_revision=self.map_revision,
            map_revision=next_revision,
            sensor_time_ns=sensor_time_ns,
            reason=reason,
            transforms=checked,
        )
        self.map_revision = next_revision
        return update

    def world_point(
        self, submap_id: str, local_point_m: np.ndarray, *, revision: Optional[int] = None
    ) -> np.ndarray:
        submap = self._submaps[submap_id]
        selected_revision = self.map_revision if revision is None else revision
        if selected_revision not in submap.transforms_by_revision:
            raise KeyError(f"submap has no revision {selected_revision}")
        point = np.asarray(local_point_m, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError("local point must be xyz")
        return (
            submap.transforms_by_revision[selected_revision] @ np.r_[point, 1.0]
        )[:3]

    def reproject_world_point(
        self,
        submap_id: str,
        world_point_m: np.ndarray,
        *,
        from_revision: int,
        to_revision: int,
    ) -> np.ndarray:
        submap = self._submaps[submap_id]
        source = submap.transforms_by_revision[from_revision]
        target = submap.transforms_by_revision[to_revision]
        point = np.asarray(world_point_m, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError("world point must be xyz")
        return (target @ np.linalg.inv(source) @ np.r_[point, 1.0])[:3]

    def snapshot(self) -> dict:
        return {
            "map_revision": self.map_revision,
            "submaps": [
                {
                    "submap_id": submap.submap_id,
                    "start_time_ns": submap.start_time_ns,
                    "end_time_ns": submap.end_time_ns,
                    "frame_times_ns": submap.frame_times_ns,
                    "local_origin_time_ns": submap.local_origin_time_ns,
                    "transforms_by_revision": {
                        str(revision): transform.tolist()
                        for revision, transform in submap.transforms_by_revision.items()
                    },
                }
                for submap in self._submaps.values()
            ],
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict, *, maximum_frames: int = 30) -> "SubmapManager":
        manager = cls(maximum_frames=maximum_frames)
        manager.map_revision = int(snapshot.get("map_revision", 0))
        for data in snapshot.get("submaps", []):
            submap = Submap(
                submap_id=str(data["submap_id"]),
                start_time_ns=int(data["start_time_ns"]),
                end_time_ns=int(data["end_time_ns"]),
                frame_times_ns=[int(value) for value in data["frame_times_ns"]],
                transforms_by_revision={
                    int(revision): np.asarray(transform, dtype=np.float64)
                    for revision, transform in data["transforms_by_revision"].items()
                },
                local_origin_time_ns=int(data["local_origin_time_ns"]),
            )
            manager._submaps[submap.submap_id] = submap
            manager._current_id = submap.submap_id
            try:
                manager._next_id = max(
                    manager._next_id,
                    int(submap.submap_id.rsplit("-", 1)[-1]) + 1,
                )
            except ValueError:
                manager._next_id += 1
        return manager
