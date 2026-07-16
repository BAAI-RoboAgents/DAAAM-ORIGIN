"""Fault-injection tests for the isolated depth worker client."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.depth.worker import DepthBackendError, SubprocessDepthBackend  # noqa: E402


ECHO_WORKER = r"""
import json, sys
print(json.dumps({'status': 'ready'}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get('command') == 'shutdown':
        break
    print(json.dumps({
        'status': 'ok',
        'request_id': request['request_id'],
        'sensor_time_ns': request['sensor_time_ns'],
    }), flush=True)
"""


def test_subprocess_depth_backend_round_trip_and_clean_shutdown():
    backend = SubprocessDepthBackend(
        [sys.executable, "-u", "-c", ECHO_WORKER],
        startup_timeout_s=1.0,
        request_timeout_s=1.0,
    )
    response = backend.infer({"sensor_time_ns": 123})
    assert response["sensor_time_ns"] == 123
    assert backend.stats()["completed"] == 1
    backend.close()


def test_worker_crash_is_restarted_without_killing_frontend(tmp_path):
    marker = tmp_path / "crashed_once"
    worker = r"""
import json, pathlib, sys
marker = pathlib.Path(sys.argv[1])
print(json.dumps({'status': 'ready'}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if not marker.exists():
        marker.write_text('yes')
        sys.exit(7)
    print(json.dumps({'status': 'ok', 'request_id': request['request_id']}), flush=True)
"""
    backend = SubprocessDepthBackend(
        [sys.executable, "-u", "-c", worker, str(marker)],
        startup_timeout_s=1.0,
        request_timeout_s=0.5,
        maximum_retries=1,
    )
    assert backend.infer({})["status"] == "ok"
    assert backend.stats()["restarts"] == 1
    backend.close()


def test_timeout_is_bounded_and_reported():
    worker = r"""
import json, sys, time
print(json.dumps({'status': 'ready'}), flush=True)
for line in sys.stdin:
    time.sleep(2)
"""
    backend = SubprocessDepthBackend(
        [sys.executable, "-u", "-c", worker],
        startup_timeout_s=1.0,
        request_timeout_s=0.05,
        maximum_retries=0,
    )
    with pytest.raises(DepthBackendError, match="failed after"):
        backend.infer({})
    assert backend.stats()["timeouts"] == 1
    assert backend.stats()["failed"] == 1
    backend.close()
