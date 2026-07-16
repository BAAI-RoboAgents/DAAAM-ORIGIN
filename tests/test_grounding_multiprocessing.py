"""Regression tests for DAM worker multiprocessing context isolation."""

from __future__ import annotations

import ast
import multiprocessing as mp
from pathlib import Path
import queue
from types import SimpleNamespace
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.config import WorkerConfig  # noqa: E402
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
