"""Bounded content-value queue with explicit backpressure decisions."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Generic, Optional, TypeVar

from .contracts import RealtimeEnvelope


PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class QueueDecision:
    accepted: bool
    reason: str
    dropped_identity: Optional[tuple] = None


@dataclass
class _QueuedItem(Generic[PayloadT]):
    envelope: RealtimeEnvelope[PayloadT]
    admitted_monotonic_ns: int
    sequence: int


class QueueClosed(RuntimeError):
    pass


class ValueAwareQueue(Generic[PayloadT]):
    """A small bounded queue optimized for mapping observations, not FIFO purity."""

    def __init__(self, capacity: int, *, active_revision: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("queue capacity must be positive")
        if active_revision < 0:
            raise ValueError("active_revision must be non-negative")
        self.capacity = int(capacity)
        self._active_revision = int(active_revision)
        self._items: list[_QueuedItem[PayloadT]] = []
        self._condition = threading.Condition()
        self._sequence = 0
        self._closed = False
        self._drop_counts: dict[str, int] = {}
        self._high_water = 0

    @property
    def active_revision(self) -> int:
        with self._condition:
            return self._active_revision

    @property
    def high_water(self) -> int:
        with self._condition:
            return self._high_water

    def qsize(self) -> int:
        with self._condition:
            return len(self._items)

    def empty(self) -> bool:
        return self.qsize() == 0

    def _count_drop(self, reason: str) -> None:
        self._drop_counts[reason] = self._drop_counts.get(reason, 0) + 1

    def drop_counts(self) -> dict[str, int]:
        with self._condition:
            return dict(self._drop_counts)

    def set_active_revision(self, revision: int) -> int:
        """Advance map revision and remove queued results tied to older maps."""
        if revision < 0:
            raise ValueError("revision must be non-negative")
        with self._condition:
            if revision < self._active_revision:
                raise ValueError("active revision cannot move backwards")
            self._active_revision = revision
            kept = []
            removed = 0
            for item in self._items:
                if item.envelope.key.map_revision < revision:
                    removed += 1
                    self._count_drop("stale_revision")
                else:
                    kept.append(item)
            self._items = kept
            return removed

    def put(self, envelope: RealtimeEnvelope[PayloadT]) -> QueueDecision:
        now_ns = time.monotonic_ns()
        with self._condition:
            if self._closed:
                self._count_drop("queue_closed")
                return QueueDecision(False, "queue_closed")
            if envelope.key.map_revision < self._active_revision:
                self._count_drop("stale_revision")
                return QueueDecision(False, "stale_revision")
            if envelope.is_expired(now_ns):
                self._count_drop("deadline_expired")
                return QueueDecision(False, "deadline_expired")

            self._sequence += 1
            incoming = _QueuedItem(envelope, now_ns, self._sequence)
            if len(self._items) < self.capacity:
                self._items.append(incoming)
                self._high_water = max(self._high_water, len(self._items))
                self._condition.notify()
                return QueueDecision(True, "accepted")

            # Lowest-value and oldest capture is the first eviction candidate.
            victim_index = min(
                range(len(self._items)),
                key=lambda index: (
                    int(self._items[index].envelope.value),
                    self._items[index].envelope.key.sensor_time_ns,
                    self._items[index].sequence,
                ),
            )
            victim = self._items[victim_index]
            incoming_rank = (
                int(envelope.value),
                envelope.key.sensor_time_ns,
                incoming.sequence,
            )
            victim_rank = (
                int(victim.envelope.value),
                victim.envelope.key.sensor_time_ns,
                victim.sequence,
            )
            if incoming_rank <= victim_rank:
                reason = (
                    "strict_duplicate"
                    if int(envelope.value) == 0
                    else "lower_content_value"
                )
                self._count_drop(reason)
                return QueueDecision(False, reason)

            self._items[victim_index] = incoming
            reason = (
                "evicted_strict_duplicate"
                if int(victim.envelope.value) == 0
                else "evicted_lower_content_value"
            )
            self._count_drop(reason)
            self._condition.notify()
            return QueueDecision(True, reason, victim.envelope.identity)

    def get(self, timeout: Optional[float] = None) -> tuple[RealtimeEnvelope[PayloadT], float]:
        """Return retained items in capture-time order; value only controls eviction."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._items:
                if self._closed:
                    raise QueueClosed("queue is closed")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("queue get timed out")
                self._condition.wait(remaining)

            while self._items:
                selected_index = min(
                    range(len(self._items)),
                    key=lambda index: (
                        self._items[index].envelope.key.sensor_time_ns,
                        self._items[index].sequence,
                    ),
                )
                item = self._items.pop(selected_index)
                now_ns = time.monotonic_ns()
                if item.envelope.key.map_revision < self._active_revision:
                    self._count_drop("stale_revision")
                    continue
                if item.envelope.is_expired(now_ns):
                    self._count_drop("deadline_expired")
                    continue
                wait_ms = (now_ns - item.admitted_monotonic_ns) / 1e6
                return item.envelope, wait_ms
            raise TimeoutError("all queued items expired")

    def close(self, *, discard: bool = False) -> int:
        with self._condition:
            self._closed = True
            discarded = len(self._items) if discard else 0
            if discard:
                self._items.clear()
                for _ in range(discarded):
                    self._count_drop("shutdown_discard")
            self._condition.notify_all()
            return discarded
