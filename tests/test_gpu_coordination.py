"""Tests for single-GPU realtime/background staggering."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.realtime.gpu import SharedGpuCoordinator  # noqa: E402


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
