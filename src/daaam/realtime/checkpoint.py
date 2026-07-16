"""Atomic replay checkpoint with idempotent per-frame completion tracking."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Mapping, Optional


class RealtimeCheckpoint:
    def __init__(self, path: Path, *, dataset_fingerprint: str) -> None:
        self.path = path
        self.dataset_fingerprint = dataset_fingerprint
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            "checkpoint_version": 1,
            "dataset_fingerprint": dataset_fingerprint,
            "completed_frame_indices": [],
            "dropped_frames": {},
            "last_sensor_time_ns": None,
            "map_revision": 0,
            "dynamic_layer": {"active": [], "history": []},
            "submaps": {"map_revision": 0, "submaps": []},
            "paths": {"paths": []},
            "path_buffer": {"sensor_times_ns": [], "points_m": []},
        }

    def load(self) -> bool:
        with self._lock:
            if not self.path.is_file():
                return False
            state = json.loads(self.path.read_text())
            if state.get("checkpoint_version") != 1:
                raise ValueError("unsupported realtime checkpoint version")
            if state.get("dataset_fingerprint") != self.dataset_fingerprint:
                raise ValueError("checkpoint dataset fingerprint does not match")
            self._state = state
            return True

    @property
    def completed_indices(self) -> set[int]:
        with self._lock:
            return {int(value) for value in self._state["completed_frame_indices"]}

    @property
    def state(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def mark_completed(
        self,
        frame_index: int,
        sensor_time_ns: int,
        *,
        map_revision: int,
        dynamic_layer: Mapping[str, Any],
        submaps: Mapping[str, Any],
        paths: Optional[Mapping[str, Any]] = None,
        path_buffer: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if frame_index < 0 or sensor_time_ns <= 0 or map_revision < 0:
            raise ValueError("checkpoint completion values are invalid")
        with self._lock:
            completed = self.completed_indices
            completed.add(frame_index)
            self._state.update(
                {
                    "completed_frame_indices": sorted(completed),
                    "last_sensor_time_ns": max(
                        sensor_time_ns, self._state.get("last_sensor_time_ns") or 0
                    ),
                    "map_revision": map_revision,
                    "dynamic_layer": dict(dynamic_layer),
                    "submaps": dict(submaps),
                }
            )
            if paths is not None:
                self._state["paths"] = dict(paths)
            if path_buffer is not None:
                self._state["path_buffer"] = dict(path_buffer)
            self._write_atomic()

    def update_mapping_state(
        self,
        *,
        map_revision: int,
        dynamic_layer: Mapping[str, Any],
        submaps: Mapping[str, Any],
        paths: Mapping[str, Any],
        path_buffer: Mapping[str, Any],
    ) -> None:
        if map_revision < 0:
            raise ValueError("checkpoint map revision is invalid")
        with self._lock:
            self._state.update(
                {
                    "map_revision": map_revision,
                    "dynamic_layer": dict(dynamic_layer),
                    "submaps": dict(submaps),
                    "paths": dict(paths),
                    "path_buffer": dict(path_buffer),
                }
            )
            self._write_atomic()

    def mark_dropped(self, frame_index: int, reason: str) -> None:
        if frame_index < 0 or not reason.strip():
            raise ValueError("checkpoint drop values are invalid")
        with self._lock:
            self._state["dropped_frames"][str(frame_index)] = reason
            self._write_atomic()

    def _write_atomic(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._state, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path)
