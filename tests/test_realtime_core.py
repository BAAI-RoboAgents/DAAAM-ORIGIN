"""Acceptance tests for versioned contracts and bounded realtime scheduling."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.realtime.contracts import (  # noqa: E402
    FrameValue,
    MessageKey,
    PoseEstimate,
    RealtimeEnvelope,
)
from daaam.realtime.metrics import MetricsCollector  # noqa: E402
from daaam.realtime.queueing import ValueAwareQueue  # noqa: E402
from daaam.realtime.scheduler import MultiRateScheduler, StageSpec  # noqa: E402


ORIGIN_NS = 1_783_933_507_759_540_877


def envelope(
    offset: int,
    value: FrameValue = FrameValue.ROUTINE,
    revision: int = 0,
) -> RealtimeEnvelope:
    return RealtimeEnvelope(
        key=MessageKey(ORIGIN_NS + offset, map_revision=revision),
        payload={"offset": offset},
        value=value,
        source="test",
    )


def test_contract_rejects_bad_time_revision_and_covariance():
    with pytest.raises(ValueError, match="positive absolute"):
        MessageKey(0)
    with pytest.raises(ValueError, match="non-negative"):
        MessageKey(ORIGIN_NS, map_revision=-1)
    with pytest.raises(ValueError, match="positive semidefinite"):
        PoseEstimate(
            ORIGIN_NS,
            np.eye(4),
            np.diag([1.0, 1.0, 1.0, 1.0, 1.0, -1.0]),
            "test",
        )


def test_envelope_rejects_payload_time_mismatch():
    pose = PoseEstimate(ORIGIN_NS, np.eye(4), np.eye(6), "test")
    with pytest.raises(ValueError, match="sensor_time_ns disagree"):
        RealtimeEnvelope(
            MessageKey(ORIGIN_NS + 1),
            pose,
            source="test",
        )


def test_queue_preserves_visual_event_and_evicts_duplicate():
    queue = ValueAwareQueue(capacity=2)
    assert queue.put(envelope(0, FrameValue.STRICT_DUPLICATE)).accepted
    assert queue.put(envelope(1, FrameValue.ROUTINE)).accepted
    decision = queue.put(envelope(2, FrameValue.IMAGE_EVENT_AT_STATIC_POSE))
    assert decision.accepted
    assert decision.reason == "evicted_strict_duplicate"

    first, _ = queue.get()
    second, _ = queue.get()
    assert first.value is FrameValue.ROUTINE
    assert second.value is FrameValue.IMAGE_EVENT_AT_STATIC_POSE
    assert queue.qsize() <= queue.capacity


def test_queue_value_never_reorders_retained_absolute_times():
    queue = ValueAwareQueue(capacity=3)
    queue.put(envelope(30, FrameValue.LOOP_CANDIDATE))
    queue.put(envelope(10, FrameValue.ROUTINE))
    queue.put(envelope(20, FrameValue.IMAGE_EVENT_AT_STATIC_POSE))
    observed = [queue.get()[0].key.sensor_time_ns for _ in range(3)]
    assert observed == sorted(observed)


def test_queue_rejects_stale_revision_and_flushes_old_work():
    queue = ValueAwareQueue(capacity=3)
    queue.put(envelope(0, revision=0))
    queue.put(envelope(1, revision=1))
    assert queue.set_active_revision(1) == 1
    assert not queue.put(envelope(2, revision=0)).accepted
    current, _ = queue.get()
    assert current.key.map_revision == 1


def test_scheduler_slow_semantic_stage_does_not_block_pose_stage():
    pose_processed: list[int] = []
    semantic_started = threading.Event()
    semantic_release = threading.Event()

    def pose_handler(item):
        pose_processed.append(item.key.sensor_time_ns)
        return None

    def semantic_handler(_item):
        semantic_started.set()
        assert semantic_release.wait(2.0)
        return None

    scheduler = MultiRateScheduler()
    scheduler.add_stage(StageSpec("pose", pose_handler, None, 16, 500.0))
    scheduler.add_stage(StageSpec("semantic", semantic_handler, None, 2, 2000.0))
    scheduler.start()
    assert scheduler.submit("semantic", envelope(0))
    assert semantic_started.wait(1.0)
    for index in range(8):
        assert scheduler.submit("pose", envelope(index + 1))

    deadline = time.monotonic() + 1.0
    while len(pose_processed) < 8 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(pose_processed) == 8
    semantic_release.set()
    report = scheduler.stop()
    assert report["stages"]["pose"]["processed"] == 8
    assert report["stages"]["semantic"]["processed"] == 1


def test_scheduler_rejects_handler_result_after_revision_change():
    handler_started = threading.Event()
    release_handler = threading.Event()
    sink_items = []

    def delayed(item):
        handler_started.set()
        assert release_handler.wait(2.0)
        return item

    scheduler = MultiRateScheduler()
    scheduler.add_stage(StageSpec("delayed", delayed, None, 2))
    scheduler.add_stage(StageSpec("sink", lambda item: sink_items.append(item), None, 2))
    scheduler.connect("delayed", "sink")
    scheduler.start()
    scheduler.submit("delayed", envelope(0, revision=0))
    assert handler_started.wait(1.0)
    scheduler.advance_revision(1)
    release_handler.set()
    assert scheduler.wait_until_idle(2.0)
    report = scheduler.stop()
    assert not sink_items
    assert report["stages"]["delayed"]["drops"]["stale_handler_result"] == 1


def test_wait_until_idle_counts_rate_limited_prefetched_message_as_inflight():
    processed = []
    scheduler = MultiRateScheduler()
    scheduler.add_stage(
        StageSpec("limited", lambda item: processed.append(item), 5.0, 4)
    )
    scheduler.start()
    scheduler.submit("limited", envelope(0))
    scheduler.submit("limited", envelope(1))
    started = time.monotonic()
    assert scheduler.wait_until_idle(1.0, stable_for=0.02)
    elapsed = time.monotonic() - started
    report = scheduler.stop()
    assert len(processed) == 2
    assert elapsed >= 0.15
    assert report["stages"]["limited"]["processed"] == 2


def test_empty_metrics_report_is_strict_json():
    collector = MetricsCollector()
    collector.stop()
    report = collector.report()
    assert report["totals"] == {"processed": 0, "errors": 0, "dropped": 0}
    assert "NaN" not in json.dumps(report, allow_nan=False)
