"""Independent-rate worker stages connected by bounded value-aware queues."""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
import time
from typing import Callable, Iterable, Optional

from .contracts import RealtimeEnvelope
from .metrics import MetricsCollector
from .queueing import QueueClosed, ValueAwareQueue


Handler = Callable[[RealtimeEnvelope], Optional[RealtimeEnvelope | Iterable[RealtimeEnvelope]]]


@dataclass(frozen=True)
class StageSpec:
    name: str
    handler: Handler
    rate_hz: Optional[float]
    queue_capacity: int
    deadline_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("stage name is required")
        if self.rate_hz is not None and self.rate_hz <= 0:
            raise ValueError("stage rate_hz must be positive")
        if self.queue_capacity <= 0:
            raise ValueError("stage queue_capacity must be positive")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("stage deadline_ms must be positive")


class MultiRateScheduler:
    """Small realtime scheduler whose stages cannot block one another."""

    def __init__(self, *, active_revision: int = 0) -> None:
        self.metrics = MetricsCollector()
        self._active_revision = active_revision
        self._specs: dict[str, StageSpec] = {}
        self._queues: dict[str, ValueAwareQueue] = {}
        self._routes: dict[str, list[str]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._started = False
        self._errors: list[tuple[str, str]] = []
        self._inflight: dict[str, int] = {}
        self._state_lock = threading.Lock()

    @property
    def active_revision(self) -> int:
        with self._state_lock:
            return self._active_revision

    def add_stage(self, spec: StageSpec) -> None:
        if self._started:
            raise RuntimeError("cannot add stages after scheduler start")
        if spec.name in self._specs:
            raise ValueError(f"duplicate stage: {spec.name}")
        self._specs[spec.name] = spec
        self._queues[spec.name] = ValueAwareQueue(
            spec.queue_capacity, active_revision=self._active_revision
        )
        self._routes[spec.name] = []
        self._inflight[spec.name] = 0

    def connect(self, source: str, destination: str) -> None:
        if source not in self._specs or destination not in self._specs:
            raise KeyError("both source and destination stages must exist")
        if destination not in self._routes[source]:
            self._routes[source].append(destination)

    def start(self) -> None:
        if self._started:
            return
        if not self._specs:
            raise RuntimeError("scheduler has no stages")
        self._started = True
        self._stop_event.clear()
        for name in self._specs:
            thread = threading.Thread(
                target=self._run_stage,
                args=(name,),
                name=f"daaam-{name}",
                daemon=True,
            )
            self._threads[name] = thread
            thread.start()

    def submit(self, stage: str, envelope: RealtimeEnvelope) -> bool:
        if stage not in self._specs:
            raise KeyError(f"unknown stage: {stage}")
        spec = self._specs[stage]
        if envelope.deadline_monotonic_ns is None and spec.deadline_ms is not None:
            envelope = replace(
                envelope,
                deadline_monotonic_ns=(
                    time.monotonic_ns() + int(spec.deadline_ms * 1e6)
                ),
            )
        decision = self._queues[stage].put(envelope)
        self.metrics.observe_queue(stage, self._queues[stage].qsize())
        if not decision.accepted:
            self.metrics.record_drop(stage, decision.reason)
        elif decision.reason.startswith("evicted_"):
            self.metrics.record_drop(stage, decision.reason)
        return decision.accepted

    def advance_revision(self, revision: int) -> dict[str, int]:
        with self._state_lock:
            if revision <= self._active_revision:
                raise ValueError("new map revision must be greater than active revision")
            self._active_revision = revision
        removed = {}
        for name, stage_queue in self._queues.items():
            count = stage_queue.set_active_revision(revision)
            removed[name] = count
            for _ in range(count):
                self.metrics.record_drop(name, "stale_revision")
        return removed

    def _run_stage(self, name: str) -> None:
        spec = self._specs[name]
        period_s = None if spec.rate_hz is None else 1.0 / spec.rate_hz
        last_started = 0.0
        while not self._stop_event.is_set():
            try:
                envelope, queue_wait_ms = self._queues[name].get(timeout=0.1)
            except (TimeoutError, QueueClosed):
                continue

            with self._state_lock:
                self._inflight[name] += 1
            if envelope.key.map_revision < self.active_revision:
                self.metrics.record_drop(name, "stale_revision")
                with self._state_lock:
                    self._inflight[name] -= 1
                continue
            if period_s is not None:
                delay = period_s - (time.monotonic() - last_started)
                if delay > 0 and self._stop_event.wait(delay):
                    self.metrics.record_drop(name, "shutdown_before_service")
                    with self._state_lock:
                        self._inflight[name] -= 1
                    break
            service_started_ns = time.monotonic_ns()
            last_started = time.monotonic()
            try:
                result = spec.handler(envelope)
                outputs: list[RealtimeEnvelope]
                if result is None:
                    outputs = []
                elif isinstance(result, RealtimeEnvelope):
                    outputs = [result]
                else:
                    outputs = list(result)
                for output in outputs:
                    if output.key.map_revision < self.active_revision:
                        self.metrics.record_drop(name, "stale_handler_result")
                        continue
                    for destination in self._routes[name]:
                        self.submit(destination, output)
            except Exception as error:  # A stage failure is observable, not process-fatal.
                with self._state_lock:
                    self._errors.append((name, repr(error)))
                self.metrics.record_error(name)
            finally:
                completed_ns = time.monotonic_ns()
                self.metrics.record_processed(
                    name,
                    queue_wait_ms=queue_wait_ms,
                    service_ms=(completed_ns - service_started_ns) / 1e6,
                    end_to_end_ms=(completed_ns - envelope.created_monotonic_ns) / 1e6,
                    queue_size=self._queues[name].qsize(),
                )
                with self._state_lock:
                    self._inflight[name] -= 1

    def wait_until_idle(self, timeout: float = 5.0, stable_for: float = 0.05) -> bool:
        deadline = time.monotonic() + timeout
        idle_since: float | None = None
        while time.monotonic() < deadline:
            with self._state_lock:
                no_inflight = not any(self._inflight.values())
            if no_inflight and all(
                stage_queue.empty() for stage_queue in self._queues.values()
            ):
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - idle_since >= stable_for:
                    return True
            else:
                idle_since = None
            time.sleep(0.005)
        return False

    def stop(self, timeout: float = 5.0, *, drain: bool = True) -> dict:
        if not self._started:
            self.metrics.stop()
            return self.report()
        if drain:
            self.wait_until_idle(timeout=max(0.0, timeout * 0.7))
        self._stop_event.set()
        for stage_queue in self._queues.values():
            stage_queue.close(discard=True)
        deadline = time.monotonic() + timeout
        for thread in self._threads.values():
            thread.join(max(0.0, deadline - time.monotonic()))
        alive = [name for name, thread in self._threads.items() if thread.is_alive()]
        self.metrics.stop()
        self._started = False
        if alive:
            raise RuntimeError(f"scheduler stages did not stop: {alive}")
        return self.report()

    def report(self) -> dict:
        report = self.metrics.report()
        with self._state_lock:
            report["active_revision"] = self._active_revision
            report["handler_errors"] = [
                {"stage": stage, "error": error} for stage, error in self._errors
            ]
        report["queues"] = {
            name: {
                "capacity": stage_queue.capacity,
                "size": stage_queue.qsize(),
                "high_water": stage_queue.high_water,
                "drops": stage_queue.drop_counts(),
            }
            for name, stage_queue in self._queues.items()
        }
        return report
