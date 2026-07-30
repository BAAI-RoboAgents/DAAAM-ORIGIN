"""Append-only experiment telemetry for realtime mapping runs."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time
from typing import Any


def _json_value(value: Any) -> Any:
    """Convert common scalar containers without accepting non-finite numbers."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


class JsonlAuditWriter:
    """Thread-safe JSONL writer with a versioned envelope."""

    def __init__(self, path: Path | str, *, schema: str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.schema = str(schema)
        if not self.schema.strip():
            raise ValueError("audit schema is required")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False

    def write(self, event: str, record: Mapping[str, Any]) -> None:
        if not str(event).strip():
            raise ValueError("audit event is required")
        payload = {
            "schema": self.schema,
            "event": str(event),
            "recorded_time_ns": time.time_ns(),
            "monotonic_time_ns": time.monotonic_ns(),
            **{str(key): _json_value(value) for key, value in record.items()},
        }
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError(f"audit writer is closed: {self.path}")
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
                stream.flush()

    def close(self) -> None:
        with self._lock:
            self._closed = True


class ResourceSampler:
    """Periodically persist host, process, and optional NVIDIA GPU telemetry."""

    def __init__(
        self,
        writer: JsonlAuditWriter,
        *,
        interval_s: float = 0.5,
        repository_root: Path | str | None = None,
    ) -> None:
        if interval_s <= 0.0:
            raise ValueError("resource sampling interval must be positive")
        self.writer = writer
        self.interval_s = float(interval_s)
        self.repository_root = (
            Path(repository_root).resolve()
            if repository_root is not None
            else Path.cwd()
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _process_status() -> dict[str, Any]:
        values: dict[str, Any] = {
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
        }
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith(("VmRSS:", "VmHWM:", "Threads:")):
                    key, raw = line.split(":", 1)
                    fields = raw.split()
                    values[key] = int(fields[0]) * 1024 if fields else None
        except (OSError, ValueError):
            pass
        try:
            stat_fields = Path("/proc/self/stat").read_text().split()
            clock_ticks = os.sysconf("SC_CLK_TCK")
            values["cpu_user_seconds"] = int(stat_fields[13]) / clock_ticks
            values["cpu_system_seconds"] = int(stat_fields[14]) / clock_ticks
            values["minor_page_faults"] = int(stat_fields[9])
            values["major_page_faults"] = int(stat_fields[11])
        except (OSError, ValueError, IndexError):
            pass
        try:
            values["process_io"] = {
                key: int(raw.strip())
                for key, raw in (
                    line.split(":", 1)
                    for line in Path("/proc/self/io").read_text().splitlines()
                    if ":" in line
                )
            }
        except (OSError, ValueError):
            pass
        try:
            memory = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, raw = line.split(":", 1)
                fields = raw.split()
                if fields:
                    memory[key] = int(fields[0]) * 1024
            values["host_memory"] = {
                key: memory.get(key)
                for key in (
                    "MemTotal",
                    "MemAvailable",
                    "SwapTotal",
                    "SwapFree",
                )
            }
        except (OSError, ValueError):
            pass
        try:
            load_1m, load_5m, load_15m = os.getloadavg()
            values["load_average"] = [load_1m, load_5m, load_15m]
        except OSError:
            pass
        return values

    def _gpu_status(self) -> list[dict[str, Any]]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,clocks.sm",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                cwd=self.repository_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=min(2.0, max(0.2, self.interval_s)),
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        if result.returncode != 0:
            return []
        fields = (
            "index",
            "name",
            "utilization_gpu_percent",
            "memory_used_mib",
            "memory_total_mib",
            "temperature_c",
            "power_w",
            "sm_clock_mhz",
        )
        return [
            dict(zip(fields, (part.strip() for part in line.split(","))))
            for line in result.stdout.splitlines()
            if line.strip()
        ]

    def sample(self) -> None:
        disk = shutil.disk_usage(self.repository_root)
        self.writer.write(
            "resource_sample",
            {
                "process": self._process_status(),
                "gpus": self._gpu_status(),
                "repository_disk": {
                    "total_bytes": disk.total,
                    "used_bytes": disk.used,
                    "free_bytes": disk.free,
                },
            },
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.sample()
            except (OSError, RuntimeError, ValueError):
                pass
            remaining = max(0.0, self.interval_s - (time.monotonic() - started))
            self._stop.wait(remaining)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="daaam-resource-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise RuntimeError("resource sampler did not stop")
        self._thread = None


class RealtimeAuditBundle:
    """Own the standard append-only files for one realtime experiment run."""

    SCHEMAS = {
        "frame_metrics": "daaam.experiment.frame_metrics.v1",
        "queue_events": "daaam.experiment.queue_events.v1",
        "resource_samples": "daaam.experiment.resource_samples.v1",
        "semantic_decisions": "daaam.experiment.semantic_decisions.v1",
        "track_events": "daaam.experiment.track_events.v1",
        "binding_candidates": "daaam.experiment.binding_candidates.v1",
        "dam_events": "daaam.experiment.dam_events.v1",
    }

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._writers = {
            name: JsonlAuditWriter(
                self.root / f"{name}.jsonl",
                schema=schema,
            )
            for name, schema in self.SCHEMAS.items()
        }

    def writer(self, name: str) -> JsonlAuditWriter:
        try:
            return self._writers[name]
        except KeyError as error:
            raise KeyError(f"unknown realtime audit stream: {name}") from error

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
