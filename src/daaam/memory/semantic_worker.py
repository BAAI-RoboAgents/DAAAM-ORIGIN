"""Asynchronous, retryable delivery of versioned semantic corrections."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

from daaam.realtime.contracts import SemanticCorrection

from .store import CorrectionReceipt, MapMemory


@dataclass(frozen=True)
class DeliveredSemanticCorrection:
    correction: SemanticCorrection
    effective_label: str


class VersionedCorrectionProcessor:
    """Persist first, then deliver to DSG without blocking the geometry frontend."""

    def __init__(
        self,
        memory: MapMemory,
        consumer: Callable[[DeliveredSemanticCorrection], bool],
        *,
        poll_interval_s: float = 0.05,
        maximum_delivery_attempts: int = 3,
    ) -> None:
        if poll_interval_s <= 0 or maximum_delivery_attempts <= 0:
            raise ValueError("semantic processor retry settings are invalid")
        self.memory = memory
        self.consumer = consumer
        self.poll_interval_s = poll_interval_s
        self.maximum_delivery_attempts = maximum_delivery_attempts
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._errors: list[str] = []

    def submit(self, correction: SemanticCorrection) -> CorrectionReceipt:
        receipt = self.memory.enqueue_correction(correction)
        self._wake.set()
        return receipt

    def process_once(self, *, limit: int = 20) -> dict[str, int]:
        applied = self.memory.apply_pending_corrections(limit=limit)
        delivered = retried = failed = 0
        for correction, effective_label in self.memory.claim_semantic_deliveries(
            limit=limit
        ):
            try:
                success = bool(
                    self.consumer(
                        DeliveredSemanticCorrection(correction, effective_label)
                    )
                )
                error = None if success else "consumer_rejected"
            except Exception as exception:
                success = False
                error = repr(exception)
                with self._lock:
                    self._errors.append(error)
            status = self.memory.complete_semantic_delivery(
                correction.operation_id,
                success=success,
                sensor_time_ns=max(correction.sensor_time_ns, time.time_ns()),
                error=error,
                maximum_attempts=self.maximum_delivery_attempts,
            )
            if status == "delivered":
                delivered += 1
            elif status == "retry":
                retried += 1
            elif status == "failed":
                failed += 1
        return {
            "memory_updates": len(applied),
            "delivered": delivered,
            "retried": retried,
            "failed": failed,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            work = self.process_once()
            if not any(work.values()):
                self._wake.wait(self.poll_interval_s)
                self._wake.clear()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="versioned-semantic-corrections",
        )
        self._thread.start()

    def stop(self, timeout_s: float = 5.0, *, drain: bool = True) -> dict:
        if drain:
            deadline = time.monotonic() + timeout_s * 0.7
            while time.monotonic() < deadline:
                work = self.process_once()
                stats = self.memory.delivery_stats()
                if not any(
                    stats[status]
                    for status in ("waiting", "ready", "retry", "delivering")
                ):
                    break
                if not any(work.values()):
                    time.sleep(self.poll_interval_s)
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise RuntimeError("semantic correction processor did not stop")
        return self.stats()

    def stats(self) -> dict:
        with self._lock:
            errors = list(self._errors)
        return {
            "corrections": self.memory.correction_stats(),
            "deliveries": self.memory.delivery_stats(),
            "consumer_errors": errors,
        }
