"""Thread-safe latency and drop metrics for realtime pipeline stages."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import threading
import time
from typing import Any, Iterable


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class MetricsCollector:
    """Collect bounded-run stage metrics without emitting non-JSON values."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_ns = time.monotonic_ns()
        self._ended_ns: int | None = None
        self._latencies: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._processed: Counter[str] = Counter()
        self._errors: Counter[str] = Counter()
        self._drops: dict[str, Counter[str]] = defaultdict(Counter)
        self._queue_high_water: Counter[str] = Counter()

    def record_processed(
        self,
        stage: str,
        *,
        queue_wait_ms: float,
        service_ms: float,
        end_to_end_ms: float,
        queue_size: int,
    ) -> None:
        with self._lock:
            self._processed[stage] += 1
            self._latencies[stage]["queue_wait_ms"].append(max(0.0, queue_wait_ms))
            self._latencies[stage]["service_ms"].append(max(0.0, service_ms))
            self._latencies[stage]["end_to_end_ms"].append(max(0.0, end_to_end_ms))
            self._queue_high_water[stage] = max(
                self._queue_high_water[stage], int(queue_size)
            )

    def record_drop(self, stage: str, reason: str) -> None:
        with self._lock:
            self._drops[stage][reason] += 1

    def record_error(self, stage: str, reason: str = "handler_error") -> None:
        with self._lock:
            self._errors[stage] += 1
            self._drops[stage][reason] += 1

    def observe_queue(self, stage: str, size: int) -> None:
        with self._lock:
            self._queue_high_water[stage] = max(
                self._queue_high_water[stage], int(size)
            )

    def stop(self) -> None:
        with self._lock:
            if self._ended_ns is None:
                self._ended_ns = time.monotonic_ns()

    def report(self) -> dict[str, Any]:
        with self._lock:
            end_ns = self._ended_ns or time.monotonic_ns()
            elapsed_s = max((end_ns - self._started_ns) / 1e9, 1e-9)
            stages = sorted(
                set(self._processed)
                | set(self._errors)
                | set(self._drops)
                | set(self._latencies)
            )
            output: dict[str, Any] = {}
            for stage in stages:
                latency_report = {}
                for name in ("queue_wait_ms", "service_ms", "end_to_end_ms"):
                    values = self._latencies[stage].get(name, [])
                    latency_report[name] = {
                        "samples": len(values),
                        "p50": _percentile(values, 50),
                        "p95": _percentile(values, 95),
                        "p99": _percentile(values, 99),
                        "max": max(values) if values else None,
                    }
                output[stage] = {
                    "processed": self._processed[stage],
                    "errors": self._errors[stage],
                    "throughput_hz": self._processed[stage] / elapsed_s,
                    "queue_high_water": self._queue_high_water[stage],
                    "drops": dict(sorted(self._drops[stage].items())),
                    "latency": latency_report,
                }
            return {
                "elapsed_seconds": elapsed_s,
                "stages": output,
                "totals": {
                    "processed": sum(self._processed.values()),
                    "errors": sum(self._errors.values()),
                    "dropped": sum(sum(counter.values()) for counter in self._drops.values()),
                },
            }
