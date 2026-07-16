"""Tests for single-GPU realtime/background staggering."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.realtime.gpu import (  # noqa: E402
    GpuActivityHeartbeat,
    SharedGpuCoordinator,
)


def test_optional_gpu_coordinator_is_a_noop():
    coordinator = SharedGpuCoordinator()
    coordinator.touch_activity()
    with coordinator.lease():
        assert coordinator.activity_age_s() is None


def test_background_gpu_lease_waits_for_frontend_idle(tmp_path):
    activity = tmp_path / "frontend.activity"
    coordinator = SharedGpuCoordinator(
        lock_path=tmp_path / "gpu.lock",
        activity_path=activity,
        minimum_idle_s=0.08,
        poll_interval_s=0.005,
    )
    coordinator.touch_activity()
    started = time.monotonic()
    with coordinator.lease():
        waited = time.monotonic() - started
    assert waited >= 0.06


def test_shared_gpu_lease_serializes_independent_clients(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    first = SharedGpuCoordinator(lock_path=lock_path, poll_interval_s=0.005)
    second = SharedGpuCoordinator(lock_path=lock_path, poll_interval_s=0.005)
    first_acquired = threading.Event()
    release_first = threading.Event()
    second_acquired = threading.Event()

    def hold_first():
        with first.lease():
            first_acquired.set()
            assert release_first.wait(1.0)

    def enter_second():
        assert first_acquired.wait(1.0)
        with second.lease():
            second_acquired.set()

    first_thread = threading.Thread(target=hold_first)
    second_thread = threading.Thread(target=enter_second)
    first_thread.start()
    second_thread.start()
    assert first_acquired.wait(1.0)
    time.sleep(0.03)
    assert not second_acquired.is_set()
    release_first.set()
    first_thread.join(1.0)
    second_thread.join(1.0)
    assert second_acquired.is_set()


def test_activity_heartbeat_defers_background_lease_until_cancelled(tmp_path):
    activity_path = tmp_path / "frontend.activity"
    foreground = SharedGpuCoordinator(activity_path=activity_path)
    background = SharedGpuCoordinator(
        lock_path=tmp_path / "gpu.lock",
        activity_path=activity_path,
        minimum_idle_s=0.06,
        poll_interval_s=0.005,
    )
    heartbeat = foreground.start_activity_heartbeat(interval_s=0.015)
    acquired = threading.Event()

    def acquire_in_background():
        with background.lease():
            acquired.set()

    background_thread = threading.Thread(target=acquire_in_background)
    background_thread.start()
    time.sleep(0.12)
    assert not acquired.is_set()

    heartbeat.stop()
    assert acquired.wait(0.25)
    background_thread.join(1.0)
    assert not heartbeat.is_running


def test_activity_heartbeat_stop_interrupts_long_wait_promptly(tmp_path):
    coordinator = SharedGpuCoordinator(activity_path=tmp_path / "frontend.activity")
    heartbeat = coordinator.start_activity_heartbeat(interval_s=10.0)

    started = time.monotonic()
    heartbeat.stop()

    assert time.monotonic() - started < 0.2
    assert not heartbeat.is_running


def test_virtual_replay_gaps_are_covered_without_busy_polling():
    class VirtualStopEvent:
        def __init__(self, duration_s):
            self.duration_s = duration_s
            self.now_s = 0.0
            self.waits = 0
            self.cancelled = False

        def wait(self, timeout_s):
            self.waits += 1
            if self.cancelled:
                return True
            self.now_s = min(self.duration_s, self.now_s + timeout_s)
            return self.now_s >= self.duration_s

        def set(self):
            self.cancelled = True

        def clear(self):
            self.cancelled = False

    coordinator = SharedGpuCoordinator()
    for source_gap_s in (1.0, 12.6):
        stop_event = VirtualStopEvent(source_gap_s)
        touches = []
        coordinator.touch_activity = lambda: touches.append(stop_event.now_s)
        heartbeat = GpuActivityHeartbeat(coordinator, interval_s=0.25)
        heartbeat._stop_event = stop_event

        coordinator.touch_activity()
        heartbeat._run()

        observed = touches + [source_gap_s]
        assert max(b - a for a, b in zip(observed, observed[1:])) <= 0.25
        assert stop_event.waits <= int(source_gap_s / 0.25) + 1
