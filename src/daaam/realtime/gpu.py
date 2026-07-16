"""Cross-process single-GPU staggering for realtime model workers."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
import threading
import time
from typing import Iterator, Optional, Protocol


class StopEvent(Protocol):
    def is_set(self) -> bool: ...


class GpuLeaseCancelled(RuntimeError):
    """Raised when shutdown is requested while waiting for the shared GPU."""


class GpuActivityHeartbeat:
    """Refresh a coordinator activity marker until explicitly stopped."""

    def __init__(
        self,
        coordinator: SharedGpuCoordinator,
        *,
        interval_s: float,
    ) -> None:
        if interval_s <= 0.0:
            raise ValueError("GPU activity heartbeat interval must be positive")
        self.coordinator = coordinator
        self.interval_s = float(interval_s)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._state_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            while not self._stop_event.wait(self.interval_s):
                self.coordinator.touch_activity()
        except BaseException as error:
            with self._state_lock:
                self._error = error
            self._stop_event.set()

    def start(self) -> GpuActivityHeartbeat:
        with self._state_lock:
            if self._thread is not None:
                return self
            self._stop_event.clear()
            self._error = None
            self.coordinator.touch_activity()
            thread = threading.Thread(
                target=self._run,
                name="daaam-gpu-activity-heartbeat",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        return self

    def stop(self, timeout_s: float = 1.0) -> None:
        if timeout_s <= 0.0:
            raise ValueError("GPU activity heartbeat timeout must be positive")
        with self._state_lock:
            thread = self._thread
            self._stop_event.set()
        if thread is None:
            return
        thread.join(timeout_s)
        if thread.is_alive():
            raise RuntimeError("GPU activity heartbeat did not stop")
        with self._state_lock:
            error = self._error
            self._thread = None
            self._error = None
        if error is not None:
            raise RuntimeError("GPU activity heartbeat failed") from error


class SharedGpuCoordinator:
    """Serialize CUDA work and optionally defer background work until idle.

    The lock is advisory and process-wide. Every cooperating CUDA process must use
    the same ``lock_path``. Geometry/frontend activity is represented by a separate
    timestamp file so long-running background DAM work can wait until the realtime
    path has gone idle before acquiring the GPU.
    """

    def __init__(
        self,
        *,
        lock_path: Path | str | None = None,
        activity_path: Path | str | None = None,
        minimum_idle_s: float = 0.0,
        poll_interval_s: float = 0.05,
    ) -> None:
        if minimum_idle_s < 0.0 or poll_interval_s <= 0.0:
            raise ValueError("GPU coordination timing must be non-negative")
        self.lock_path = None if lock_path is None else Path(lock_path).resolve()
        self.activity_path = (
            None if activity_path is None else Path(activity_path).resolve()
        )
        self.minimum_idle_s = float(minimum_idle_s)
        self.poll_interval_s = float(poll_interval_s)

    def touch_activity(self) -> None:
        if self.activity_path is None:
            return
        self.activity_path.parent.mkdir(parents=True, exist_ok=True)
        self.activity_path.touch(exist_ok=True)

    def activity_age_s(self) -> Optional[float]:
        if self.activity_path is None or not self.activity_path.exists():
            return None
        return max(0.0, time.time() - self.activity_path.stat().st_mtime)

    def start_activity_heartbeat(
        self,
        *,
        interval_s: float,
    ) -> GpuActivityHeartbeat:
        return GpuActivityHeartbeat(self, interval_s=interval_s).start()

    @staticmethod
    def _cancelled(stop_event: Optional[StopEvent]) -> bool:
        return bool(stop_event is not None and stop_event.is_set())

    def wait_until_idle(self, stop_event: Optional[StopEvent] = None) -> None:
        if self.minimum_idle_s <= 0.0 or self.activity_path is None:
            return
        while True:
            if self._cancelled(stop_event):
                raise GpuLeaseCancelled("GPU lease wait cancelled")
            age_s = self.activity_age_s()
            if age_s is None or age_s >= self.minimum_idle_s:
                return
            time.sleep(min(self.poll_interval_s, self.minimum_idle_s - age_s))

    @contextmanager
    def lease(self, stop_event: Optional[StopEvent] = None) -> Iterator[None]:
        self.wait_until_idle(stop_event)
        if self.lock_path is None:
            yield
            return
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_file:
            while True:
                if self._cancelled(stop_event):
                    raise GpuLeaseCancelled("GPU lease wait cancelled")
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(self.poll_interval_s)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
