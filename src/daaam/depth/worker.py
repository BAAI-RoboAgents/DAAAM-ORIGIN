"""Restartable JSON-lines client for an isolated stereo inference process."""

from __future__ import annotations

from collections import deque
import json
import queue
import subprocess
import threading
import time
from typing import Any, Mapping, Optional, Sequence
import uuid


class DepthBackendError(RuntimeError):
    pass


class SubprocessDepthBackend:
    """Keep CUDA model failures outside the realtime frontend process."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        startup_timeout_s: float = 60.0,
        request_timeout_s: float = 1.0,
        maximum_retries: int = 1,
        environment: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not command:
            raise ValueError("depth worker command is required")
        if startup_timeout_s <= 0 or request_timeout_s <= 0 or maximum_retries < 0:
            raise ValueError("depth worker timeout/retry settings are invalid")
        self.command = [str(value) for value in command]
        self.startup_timeout_s = startup_timeout_s
        self.request_timeout_s = request_timeout_s
        self.maximum_retries = maximum_retries
        self.environment = dict(environment) if environment is not None else None
        self._process: Optional[subprocess.Popen] = None
        self._stdout_queue: queue.Queue[dict | str] = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._closed = False
        self._recent_worker_logs: deque[str] = deque(maxlen=50)
        self._stats = {
            "requests": 0,
            "completed": 0,
            "failed": 0,
            "timeouts": 0,
            "restarts": 0,
            "worker_logs": 0,
        }

    def _read_stream(self, stream, *, parse_json: bool) -> None:
        try:
            for line in iter(stream.readline, ""):
                cleaned = line.strip()
                if not cleaned:
                    continue
                if parse_json:
                    try:
                        self._stdout_queue.put(json.loads(cleaned))
                    except json.JSONDecodeError:
                        self._stats["worker_logs"] += 1
                        self._recent_worker_logs.append(cleaned)
                        self._stdout_queue.put(cleaned)
                else:
                    self._stats["worker_logs"] += 1
                    self._recent_worker_logs.append(cleaned)
        finally:
            stream.close()

    def _next_message(self, timeout_s: float) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            process = self._process
            if process is not None and process.poll() is not None and self._stdout_queue.empty():
                recent_logs = " | ".join(list(self._recent_worker_logs)[-5:])
                raise DepthBackendError(
                    f"depth worker exited with code {process.returncode}"
                    + (f": {recent_logs}" if recent_logs else "")
                )
            try:
                message = self._stdout_queue.get(
                    timeout=min(0.05, max(0.0, deadline - time.monotonic()))
                )
            except queue.Empty:
                continue
            if isinstance(message, dict):
                return message
        raise TimeoutError("depth worker response timed out")

    def start(self) -> dict:
        with self._lock:
            if self._closed:
                raise DepthBackendError("depth backend is closed")
            if self._process is not None and self._process.poll() is None:
                return {"status": "already_running"}
            self._stdout_queue = queue.Queue()
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=self.environment,
            )
            assert self._process.stdout is not None
            assert self._process.stderr is not None
            self._reader_thread = threading.Thread(
                target=self._read_stream,
                args=(self._process.stdout,),
                kwargs={"parse_json": True},
                daemon=True,
                name="depth-worker-stdout",
            )
            self._stderr_thread = threading.Thread(
                target=self._read_stream,
                args=(self._process.stderr,),
                kwargs={"parse_json": False},
                daemon=True,
                name="depth-worker-stderr",
            )
            self._reader_thread.start()
            self._stderr_thread.start()
            try:
                ready = self._next_message(self.startup_timeout_s)
            except Exception:
                self._terminate_process()
                raise
            if ready.get("status") != "ready":
                self._terminate_process()
                raise DepthBackendError(f"invalid depth worker handshake: {ready}")
            return ready

    def _terminate_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        for thread in (self._reader_thread, self._stderr_thread):
            if thread is not None:
                thread.join(timeout=1.0)

    def restart(self) -> dict:
        with self._lock:
            self._terminate_process()
            self._stats["restarts"] += 1
            return self.start()

    def infer(self, request: Mapping[str, Any]) -> dict:
        with self._lock:
            if self._closed:
                raise DepthBackendError("depth backend is closed")
            self._stats["requests"] += 1
            payload = dict(request)
            payload.setdefault("request_id", uuid.uuid4().hex)
            payload["command"] = "infer"
            last_error: Exception | None = None
            for attempt in range(self.maximum_retries + 1):
                try:
                    self.start()
                    assert self._process is not None and self._process.stdin is not None
                    self._process.stdin.write(json.dumps(payload) + "\n")
                    self._process.stdin.flush()
                    deadline = time.monotonic() + self.request_timeout_s
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("depth request timed out")
                        response = self._next_message(remaining)
                        if response.get("request_id") != payload["request_id"]:
                            continue
                        if response.get("status") == "ok":
                            self._stats["completed"] += 1
                            return response
                        raise DepthBackendError(
                            str(response.get("error", "depth worker inference failed"))
                        )
                except TimeoutError as error:
                    self._stats["timeouts"] += 1
                    last_error = error
                except (BrokenPipeError, OSError, DepthBackendError) as error:
                    last_error = error
                if attempt < self.maximum_retries:
                    self.restart()
            self._stats["failed"] += 1
            self._terminate_process()
            raise DepthBackendError(
                f"depth inference failed after {self.maximum_retries + 1} attempts: {last_error}"
            )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "recent_worker_logs": list(self._recent_worker_logs),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            process = self._process
            if process is not None and process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                    process.stdin.flush()
                    process.wait(timeout=2.0)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    pass
            self._terminate_process()
            self._closed = True

    def __enter__(self) -> "SubprocessDepthBackend":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.close()
