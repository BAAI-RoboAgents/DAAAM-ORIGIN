"""Acceptance tests for incremental semantic acknowledgements and retries."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.memory import (  # noqa: E402
    MapMemory,
    VersionedCorrectionProcessor,
)
from daaam.realtime.contracts import SemanticCorrection  # noqa: E402


ORIGIN_NS = 1_783_933_507_759_540_877


def setup_memory(path: Path):
    memory = MapMemory(path)
    memory.create_session("session", ORIGIN_NS, canonical=True)
    entity_id, _ = memory.observe_entity(
        "session",
        "local",
        np.zeros(3),
        sensor_time_ns=ORIGIN_NS + 1,
        semantic_label="unknown",
    )
    return memory, entity_id


def test_correction_is_delivered_during_runtime_with_ack(tmp_path):
    memory, entity_id = setup_memory(tmp_path / "memory.sqlite3")
    deliveries = []
    processor = VersionedCorrectionProcessor(
        memory, lambda update: deliveries.append(update) or True
    )
    processor.start()
    receipt = processor.submit(
        SemanticCorrection(
            "operation", entity_id, ORIGIN_NS + 2, 0, "chair", 0.9
        )
    )
    assert receipt.status == "pending"
    deadline = time.monotonic() + 1.0
    while not deliveries and time.monotonic() < deadline:
        time.sleep(0.01)
    stats = processor.stop()
    assert len(deliveries) == 1
    assert deliveries[0].effective_label == "chair"
    assert stats["deliveries"]["delivered"] == 1
    assert stats["deliveries"]["waiting"] == 0
    memory.close()


def test_slow_semantic_consumer_does_not_block_submitter(tmp_path):
    memory, entity_id = setup_memory(tmp_path / "memory.sqlite3")
    entered = threading.Event()
    release = threading.Event()

    def slow(_update):
        entered.set()
        release.wait(2.0)
        return True

    processor = VersionedCorrectionProcessor(memory, slow)
    processor.start()
    started = time.monotonic()
    processor.submit(
        SemanticCorrection("slow", entity_id, ORIGIN_NS + 2, 0, "desk", 0.8)
    )
    assert time.monotonic() - started < 0.1
    assert entered.wait(1.0)
    release.set()
    stats = processor.stop()
    assert stats["deliveries"]["delivered"] == 1
    memory.close()


def test_failed_delivery_retries_idempotently_then_succeeds(tmp_path):
    memory, entity_id = setup_memory(tmp_path / "memory.sqlite3")
    attempts = []

    def flaky(update):
        attempts.append(update.correction.operation_id)
        return len(attempts) >= 2

    processor = VersionedCorrectionProcessor(
        memory, flaky, maximum_delivery_attempts=3
    )
    processor.submit(
        SemanticCorrection("retry", entity_id, ORIGIN_NS + 2, 0, "table", 0.8)
    )
    first = processor.process_once()
    assert first["retried"] == 1
    second = processor.process_once()
    assert second["delivered"] == 1
    assert attempts == ["retry", "retry"]
    assert memory.correction_stats()["applied"] == 1
    assert memory.delivery_stats()["delivered"] == 1
    memory.close()
