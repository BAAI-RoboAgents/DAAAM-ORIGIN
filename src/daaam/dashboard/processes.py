"""Small, argv-only subprocess supervisor used by the local dashboard."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import threading
from typing import Any
import uuid

from .commands import CommandPreview


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ProcessEvent:
    sequence: int
    timestamp: str
    stream: str
    message: str
    kind: str = "log"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "stream": self.stream,
            "message": self.message,
            "kind": self.kind,
        }


class ManagedProcess:
    def __init__(
        self,
        process_id: str,
        preview: CommandPreview,
        *,
        repository_root: Path,
        maximum_events: int,
    ) -> None:
        self.process_id = process_id
        self.preview = preview
        self.repository_root = repository_root
        self.run_dir = Path(str(preview.parameters["run_dir"])).resolve()
        self.run_id = self.run_dir.name
        self.status = "starting"
        self.started_at = _utc_now()
        self.ended_at: str | None = None
        self.exit_code: int | None = None
        self.message = "正在启动子进程"
        self._events: deque[ProcessEvent] = deque(maxlen=maximum_events)
        self._sequence = 0
        self._lock = threading.RLock()
        self._log_lock = threading.Lock()
        self._cancel_requested = False
        self._popen: subprocess.Popen[str] | None = None

    def _event(self, stream: str, message: str, kind: str = "log") -> None:
        clean = message.rstrip("\r\n")
        if not clean:
            return
        with self._lock:
            self._sequence += 1
            self._events.append(
                ProcessEvent(self._sequence, _utc_now(), stream, clean, kind)
            )
        # The runner normally creates run_dir very early.  Do not create it in
        # advance: both existing runners use directory existence for safety.
        if self.run_dir.is_dir():
            try:
                with self._log_lock:
                    with (self.run_dir / "dashboard_process.log").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(f"[{_utc_now()}] [{stream}] {clean}\n")
            except OSError:
                pass

    def start(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        self._event("dashboard", self.preview.command, "command")
        try:
            popen = subprocess.Popen(
                list(self.preview.argv),
                cwd=self.repository_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=(os.name == "posix"),
            )
        except OSError as error:
            with self._lock:
                self.status = "failed"
                self.ended_at = _utc_now()
                self.message = f"启动失败：{error}"
            self._event("dashboard", self.message, "status")
            return
        with self._lock:
            self._popen = popen
            self.status = "running"
            self.message = "子进程正在运行"
        self._event("dashboard", f"PID {popen.pid} 已启动", "status")
        assert popen.stdout is not None
        assert popen.stderr is not None
        threading.Thread(
            target=self._read_stream,
            args=(popen.stdout, "stdout"),
            daemon=True,
            name=f"daaam-dashboard-stdout-{self.process_id}",
        ).start()
        threading.Thread(
            target=self._read_stream,
            args=(popen.stderr, "stderr"),
            daemon=True,
            name=f"daaam-dashboard-stderr-{self.process_id}",
        ).start()
        threading.Thread(
            target=self._wait,
            daemon=True,
            name=f"daaam-dashboard-wait-{self.process_id}",
        ).start()

    def _read_stream(self, stream, stream_name: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                self._event(stream_name, line)
        finally:
            stream.close()

    def _wait(self) -> None:
        with self._lock:
            popen = self._popen
        if popen is None:
            return
        code = popen.wait()
        with self._lock:
            self.exit_code = code
            self.ended_at = _utc_now()
            if self._cancel_requested:
                self.status = "cancelled"
                self.message = f"运行已停止（退出码 {code}）"
            elif code == 0:
                self.status = "succeeded"
                self.message = "子进程已成功结束"
            else:
                self.status = "failed"
                self.message = f"子进程失败（退出码 {code}）"
        self._event("dashboard", self.message, "status")

    def stop(self, *, grace_seconds: float = 15.0) -> bool:
        with self._lock:
            popen = self._popen
            if popen is None or popen.poll() is not None:
                return False
            self._cancel_requested = True
            self.status = "stopping"
            self.message = "已发送 SIGINT，等待 pipeline drain/cleanup"
        self._event("dashboard", self.message, "status")
        try:
            if os.name == "posix":
                os.killpg(popen.pid, signal.SIGINT)
            else:
                popen.send_signal(signal.SIGINT)
        except ProcessLookupError:
            return False
        threading.Thread(
            target=self._escalate_stop,
            args=(grace_seconds,),
            daemon=True,
            name=f"daaam-dashboard-stop-{self.process_id}",
        ).start()
        return True

    def _escalate_stop(self, grace_seconds: float) -> None:
        with self._lock:
            popen = self._popen
        if popen is None:
            return
        try:
            popen.wait(timeout=max(0.1, grace_seconds))
            return
        except subprocess.TimeoutExpired:
            self._event("dashboard", "SIGINT 超时，升级为 SIGTERM", "status")
        try:
            if os.name == "posix":
                os.killpg(popen.pid, signal.SIGTERM)
            else:
                popen.terminate()
            popen.wait(timeout=5.0)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        self._event("dashboard", "SIGTERM 超时，强制结束进程组", "status")
        try:
            if os.name == "posix":
                os.killpg(popen.pid, signal.SIGKILL)
            else:
                popen.kill()
        except ProcessLookupError:
            pass

    def events_after(self, cursor: int) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            events = [event.to_dict() for event in self._events if event.sequence > cursor]
            next_cursor = self._sequence
        return events, next_cursor

    def info(self) -> dict[str, Any]:
        with self._lock:
            popen = self._popen
            return {
                "process_id": self.process_id,
                "pid": popen.pid if popen is not None else None,
                "workflow_id": self.preview.workflow_id,
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "status": self.status,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "exit_code": self.exit_code,
                "message": self.message,
                "command": self.preview.command,
                "parameters": dict(self.preview.parameters),
                "warnings": list(self.preview.warnings),
            }


class ProcessManager:
    def __init__(
        self,
        repository_root: Path,
        *,
        maximum_events: int = 5000,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.maximum_events = maximum_events
        self._lock = threading.RLock()
        self._processes: dict[str, ManagedProcess] = {}

    def start(self, preview: CommandPreview) -> ManagedProcess:
        run_dir = Path(str(preview.parameters["run_dir"])).resolve()
        with self._lock:
            for managed in self._processes.values():
                info = managed.info()
                if Path(info["run_dir"]) == run_dir and info["status"] in {
                    "starting", "running", "stopping"
                }:
                    raise RuntimeError("该运行目录已有活动进程")
            process_id = uuid.uuid4().hex
            managed = ManagedProcess(
                process_id,
                preview,
                repository_root=self.repository_root,
                maximum_events=self.maximum_events,
            )
            self._processes[process_id] = managed
        managed.start()
        return managed

    def get(self, process_id: str) -> ManagedProcess:
        with self._lock:
            try:
                return self._processes[process_id]
            except KeyError as error:
                raise KeyError(f"Unknown process: {process_id}") from error

    def for_run(self, run_id: str) -> ManagedProcess | None:
        with self._lock:
            matches = [
                managed
                for managed in self._processes.values()
                if managed.run_id == run_id
            ]
        if not matches:
            return None
        return max(matches, key=lambda item: item.started_at)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [managed.info() for managed in self._processes.values()]
