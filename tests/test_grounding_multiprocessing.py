"""Regression tests for DAM worker multiprocessing context isolation."""

from __future__ import annotations

import ast
import multiprocessing as mp
from pathlib import Path
import queue
from types import SimpleNamespace
import sys
import threading
import time

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.config import WorkerConfig  # noqa: E402
from daaam.grounding.control import (  # noqa: E402
    GroundingCorrectionsDrained,
    GroundingDrainRequest,
)
from daaam.grounding.services import GroundingService  # noqa: E402


def _echo_grounding_worker(
    incoming,
    outgoing,
    stop_event,
    _config,
    ready_queue,
) -> None:
    """Small spawn-safe stand-in for the CUDA DAM worker."""

    ready_queue.put(
        {
            "worker": mp.current_process().name,
            "ready": True,
            "error": None,
        }
    )
    outgoing.put(incoming.get(timeout=5.0))
    stop_event.wait(5.0)


def _inflight_grounding_worker(
    incoming,
    outgoing,
    _stop_event,
    config,
    ready_queue,
) -> None:
    """Hold one dequeued item in-flight until the test releases it."""

    ready_queue.put(
        {
            "worker": mp.current_process().name,
            "ready": True,
            "error": None,
        }
    )
    item = incoming.get(timeout=5.0)
    config["in_flight"].set()
    if not config["release"].wait(5.0):
        raise RuntimeError("test did not release in-flight grounding item")
    outgoing.put(("processed", item))
    request = incoming.get(timeout=5.0)
    if not isinstance(request, GroundingDrainRequest):
        raise TypeError(f"unexpected grounding control message: {type(request)!r}")
    ready_queue.put(
        {
            "worker": mp.current_process().name,
            "drain_token": request.token,
            "drained": True,
            "error": None,
        }
    )


def _fifo_drain_worker(
    incoming,
    outgoing,
    stop_event,
    _config,
    ready_queue,
) -> None:
    ready_queue.put(
        {
            "worker": mp.current_process().name,
            "ready": True,
            "error": None,
        }
    )
    while not stop_event.is_set():
        item = incoming.get(timeout=5.0)
        if isinstance(item, GroundingDrainRequest):
            ready_queue.put(
                {
                    "worker": mp.current_process().name,
                    "drain_token": item.token,
                    "drained": True,
                    "error": None,
                }
            )
            return
        outgoing.put(item)


def _drain_crashing_worker(
    incoming,
    _outgoing,
    _stop_event,
    _config,
    ready_queue,
) -> None:
    ready_queue.put(
        {
            "worker": mp.current_process().name,
            "ready": True,
            "error": None,
        }
    )
    request = incoming.get(timeout=5.0)
    if isinstance(request, GroundingDrainRequest):
        raise RuntimeError("intentional drain crash")


def _drain_without_ack_worker(
    incoming,
    _outgoing,
    _stop_event,
    config,
    ready_queue,
) -> None:
    ready_queue.put(
        {
            "worker": mp.current_process().name,
            "ready": True,
            "error": None,
        }
    )
    request = incoming.get(timeout=5.0)
    if isinstance(request, GroundingDrainRequest):
        config["sentinel_consumed"].set()
    while True:
        time.sleep(0.05)


def _queue_blocking_worker(
    _incoming,
    _outgoing,
    _stop_event,
    _config,
    ready_queue,
) -> None:
    ready_queue.put(
        {
            "worker": mp.current_process().name,
            "ready": True,
            "error": None,
        }
    )
    while True:
        time.sleep(0.05)


def _close_queues(*queues) -> None:
    for queued in queues:
        queued.close()
    for queued in queues:
        queued.join_thread()


def _wait_until_ready(service: GroundingService) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if service.get_worker_health()["all_ready"]:
            return
        time.sleep(0.01)
    raise AssertionError("grounding worker did not report ready")


class _FakeEvent:
    def __init__(self):
        self.was_set = False

    def set(self):
        self.was_set = True


class _FakeProcess:
    def __init__(self, name, *, start_error=None, join_error=None):
        self.name = name
        self.pid = 12345
        self.exitcode = None
        self.start_error = start_error
        self.join_error = join_error
        self.started = False
        self.joined = False

    def start(self):
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    def join(self, timeout=None):
        self.joined = True
        if self.join_error is not None:
            raise self.join_error
        self.exitcode = 0

    def is_alive(self):
        return False

    def terminate(self):
        return None


class _PartialFailureContext:
    def __init__(self, start_error, cleanup_error=None):
        self.start_error = start_error
        self.cleanup_error = cleanup_error
        self.event = _FakeEvent()
        self.processes = []

    def Event(self):
        return self.event

    def Queue(self):
        return queue.Queue()

    def Process(self, *, target, args, name):
        del target, args
        ordinal = len(self.processes)
        process = _FakeProcess(
            name,
            start_error=self.start_error if ordinal == 1 else None,
            join_error=self.cleanup_error if ordinal == 0 else None,
        )
        self.processes.append(process)
        return process


def test_grounding_service_uses_spawn_context_without_global_constructors(
    monkeypatch,
):
    context = mp.get_context("spawn")
    incoming = context.Queue(maxsize=1)
    outgoing = context.Queue(maxsize=1)
    service = GroundingService(WorkerConfig(num_grounding_workers=1))
    monkeypatch.setattr(
        service,
        "_get_grounding_worker_process",
        lambda: _echo_grounding_worker,
    )

    def forbidden_global_constructor(*_args, **_kwargs):
        raise AssertionError(
            "grounding service used the global multiprocessing context"
        )

    monkeypatch.setattr(mp, "Event", forbidden_global_constructor)
    monkeypatch.setattr(mp, "Queue", forbidden_global_constructor)
    monkeypatch.setattr(mp, "Process", forbidden_global_constructor)
    monkeypatch.setattr(mp, "set_start_method", forbidden_global_constructor)
    pipeline_config = SimpleNamespace(
        get_worker_config=lambda _worker: {"grounding_worker": "dam_multi_image"}
    )

    try:
        service.start(incoming, outgoing, pipeline_config)
        incoming.put("spawn-compatible")
        assert outgoing.get(timeout=10.0) == "spawn-compatible"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if service.get_worker_health()["all_ready"]:
                break
            time.sleep(0.01)
        assert service.get_worker_health()["all_ready"]
    finally:
        service.stop()
        incoming.close()
        outgoing.close()
        incoming.join_thread()
        outgoing.join_thread()


def test_dam_module_has_no_import_time_start_method_mutation():
    source_path = (
        REPOSITORY_ROOT / "src" / "daaam" / "query_manager" / "dam" / "services.py"
    )
    syntax = ast.parse(source_path.read_text())
    forbidden_calls = [
        node
        for node in ast.walk(syntax)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_start_method"
    ]
    assert forbidden_calls == []


def test_partial_spawn_failure_rolls_back_started_workers_and_preserves_error(
    monkeypatch,
):
    start_error = RuntimeError("second worker spawn failed")
    cleanup_error = RuntimeError("first worker join failed")
    context = _PartialFailureContext(start_error, cleanup_error)
    service = GroundingService(WorkerConfig(num_grounding_workers=2))
    service._mp_context = context
    monkeypatch.setattr(
        service,
        "_get_grounding_worker_process",
        lambda: _echo_grounding_worker,
    )
    pipeline_config = SimpleNamespace(
        get_worker_config=lambda _worker: {"grounding_worker": "dam_multi_image"}
    )

    try:
        service.start(queue.Queue(), queue.Queue(), pipeline_config)
    except RuntimeError as captured:
        assert captured is start_error
    else:  # pragma: no cover - regression assertion
        raise AssertionError("partial worker spawn unexpectedly succeeded")

    assert context.event.was_set
    assert context.processes[0].started
    assert context.processes[0].joined
    assert not service.workers
    assert not service.is_running()


def test_stop_cleans_workers_even_if_running_flag_was_never_set():
    service = GroundingService(WorkerConfig(num_grounding_workers=1))
    worker = _FakeProcess("partially-started")
    worker.started = True
    event = _FakeEvent()
    service.workers = [worker]
    service.stop_event = event
    service.worker_ready_queue = queue.Queue()
    service._running = False

    service.stop()

    assert event.was_set
    assert worker.joined
    assert not service.workers


def test_drain_waits_for_dequeued_inflight_work_and_publishes_fifo_tail(
    monkeypatch,
):
    context = mp.get_context("spawn")
    incoming = context.Queue(maxsize=1)
    outgoing = context.Queue(maxsize=4)
    in_flight = context.Event()
    release = context.Event()
    service = GroundingService(WorkerConfig(num_grounding_workers=1))
    monkeypatch.setattr(
        service,
        "_get_grounding_worker_process",
        lambda: _inflight_grounding_worker,
    )
    pipeline_config = SimpleNamespace(
        get_worker_config=lambda _worker: {
            "in_flight": in_flight,
            "release": release,
        }
    )

    try:
        service.start(incoming, outgoing, pipeline_config)
        incoming.put("dequeued-but-not-finished")
        assert in_flight.wait(5.0)
        timer = threading.Timer(0.2, release.set)
        timer.start()
        started = time.monotonic()
        shutdown = service.stop(timeout_s=3.0, drain=True)
        elapsed = time.monotonic() - started
        timer.join()

        assert elapsed >= 0.15
        assert outgoing.get(timeout=1.0) == (
            "processed",
            "dequeued-but-not-finished",
        )
        tail = outgoing.get(timeout=1.0)
        assert isinstance(tail, GroundingCorrectionsDrained)
        assert tail.token == shutdown["drain_token"]
        assert shutdown["drain_complete"] is True
        assert shutdown["correction_tail_enqueued"] is True
        health = service.get_worker_health()
        assert health["shutdown"] == shutdown
        assert health["workers"][0]["drained"] is True
        assert health["workers"][0]["forced_termination"] is False
    finally:
        release.set()
        if service.workers:
            service.stop(timeout_s=1.0, drain=False)
        _close_queues(incoming, outgoing)


def test_drain_sends_one_fifo_request_to_each_worker(monkeypatch):
    context = mp.get_context("spawn")
    incoming = context.Queue(maxsize=8)
    outgoing = context.Queue(maxsize=16)
    service = GroundingService(WorkerConfig(num_grounding_workers=2))
    monkeypatch.setattr(
        service,
        "_get_grounding_worker_process",
        lambda: _fifo_drain_worker,
    )
    pipeline_config = SimpleNamespace(get_worker_config=lambda _worker: {})

    try:
        service.start(incoming, outgoing, pipeline_config)
        _wait_until_ready(service)
        for item in range(6):
            incoming.put(item)
        shutdown = service.stop(timeout_s=3.0, drain=True)
        outputs = [outgoing.get(timeout=1.0) for _ in range(7)]

        assert set(outputs[:-1]) == set(range(6))
        assert isinstance(outputs[-1], GroundingCorrectionsDrained)
        assert outputs[-1].token == shutdown["drain_token"]
        health = service.get_worker_health()
        assert len(health["workers"]) == 2
        assert all(worker["drained"] for worker in health["workers"])
        assert all(
            worker["drain_request_enqueued"] for worker in health["workers"]
        )
    finally:
        if service.workers:
            service.stop(timeout_s=1.0, drain=False)
        _close_queues(incoming, outgoing)


def test_drain_reports_worker_crash_before_acknowledgement(monkeypatch):
    context = mp.get_context("spawn")
    incoming = context.Queue(maxsize=1)
    outgoing = context.Queue(maxsize=2)
    service = GroundingService(WorkerConfig(num_grounding_workers=1))
    monkeypatch.setattr(
        service,
        "_get_grounding_worker_process",
        lambda: _drain_crashing_worker,
    )
    pipeline_config = SimpleNamespace(get_worker_config=lambda _worker: {})

    try:
        service.start(incoming, outgoing, pipeline_config)
        _wait_until_ready(service)
        with pytest.raises(RuntimeError, match="exited with code"):
            service.stop(timeout_s=2.0, drain=True)
        health = service.get_worker_health()
        assert health["shutdown"]["drain_complete"] is False
        assert health["shutdown"]["timed_out"] is False
        assert health["workers"][0]["exitcode"] not in (None, 0)
    finally:
        if service.workers:
            service.stop(timeout_s=1.0, drain=False)
        _close_queues(incoming, outgoing)


def test_drain_timeout_forces_worker_that_consumes_sentinel_without_ack(
    monkeypatch,
):
    context = mp.get_context("spawn")
    incoming = context.Queue(maxsize=1)
    outgoing = context.Queue(maxsize=2)
    sentinel_consumed = context.Event()
    service = GroundingService(WorkerConfig(num_grounding_workers=1))
    monkeypatch.setattr(
        service,
        "_get_grounding_worker_process",
        lambda: _drain_without_ack_worker,
    )
    pipeline_config = SimpleNamespace(
        get_worker_config=lambda _worker: {
            "sentinel_consumed": sentinel_consumed,
        }
    )

    try:
        service.start(incoming, outgoing, pipeline_config)
        _wait_until_ready(service)
        with pytest.raises(TimeoutError, match="drain acknowledgements"):
            service.stop(timeout_s=0.5, drain=True)
        assert sentinel_consumed.is_set()
        health = service.get_worker_health()
        assert health["shutdown"]["timed_out"] is True
        assert health["shutdown"]["forced_termination"] is True
        assert health["workers"][0]["timed_out"] is True
        assert health["workers"][0]["forced_termination"] is True
        assert health["workers"][0]["is_alive"] is False
    finally:
        if service.workers:
            service.stop(timeout_s=1.0, drain=False)
        _close_queues(incoming, outgoing)


def test_full_prompt_queue_fails_drain_within_deadline_and_forces_worker(
    monkeypatch,
):
    context = mp.get_context("spawn")
    incoming = context.Queue(maxsize=1)
    outgoing = context.Queue(maxsize=2)
    service = GroundingService(WorkerConfig(num_grounding_workers=1))
    monkeypatch.setattr(
        service,
        "_get_grounding_worker_process",
        lambda: _queue_blocking_worker,
    )
    pipeline_config = SimpleNamespace(get_worker_config=lambda _worker: {})

    try:
        service.start(incoming, outgoing, pipeline_config)
        _wait_until_ready(service)
        incoming.put("occupies-the-only-slot")
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="prompt queue stayed full"):
            service.stop(timeout_s=0.5, drain=True)
        assert time.monotonic() - started < 1.0
        health = service.get_worker_health()
        assert health["shutdown"]["timed_out"] is True
        assert health["shutdown"]["forced_termination"] is True
        assert health["workers"][0]["drain_request_enqueued"] is False
    finally:
        if service.workers:
            service.stop(timeout_s=1.0, drain=False)
        _close_queues(incoming, outgoing)
