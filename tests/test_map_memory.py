"""Acceptance tests for editable, idempotent, cross-session map memory."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.memory import MapMemory, MapMemoryConfig  # noqa: E402
from daaam.realtime.contracts import SemanticCorrection  # noqa: E402


ORIGIN_NS = 1_783_933_507_759_540_877
MEMORY_CLI = REPOSITORY_ROOT / "scripts" / "manage_map_memory.py"


def translation(x: float, y: float = 0.0) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = [x, y, 0.0]
    return transform


def make_memory(path: Path) -> MapMemory:
    return MapMemory(
        path,
        MapMemoryConfig(
            minimum_registration_inliers=5,
            maximum_registration_rms_m=0.2,
            maximum_registration_std=0.3,
            entity_merge_distance_m=0.4,
        ),
    )


def create_table(memory: MapMemory, name="table-1") -> str:
    memory.create_session("session-a", ORIGIN_NS, canonical=True)
    entity_id, created = memory.observe_entity(
        "session-a",
        name,
        np.array([1.0, 0.0, 0.0]),
        sensor_time_ns=ORIGIN_NS + 1,
        semantic_label="table",
        dimensions_m=np.array([1.2, 0.8, 0.7]),
        confidence=0.9,
    )
    assert created
    return entity_id


def test_user_name_and_alias_survive_reopen_and_automatic_correction(tmp_path):
    database = tmp_path / "map_memory.sqlite3"
    memory = make_memory(database)
    entity_id = create_table(memory)
    memory.set_user_name(
        entity_id,
        "dining table",
        sensor_time_ns=ORIGIN_NS + 2,
        aliases=("family table",),
        lock=True,
    )
    correction = SemanticCorrection(
        "operation-1",
        entity_id,
        ORIGIN_NS + 3,
        0,
        "wood furniture",
        0.91,
        aliases=("large table",),
    )
    assert memory.enqueue_correction(correction).status == "pending"
    receipts = memory.apply_pending_corrections()
    assert receipts[0].status == "applied_alias"
    assert memory.get_entity(entity_id)["canonical_name"] == "dining table"
    assert memory.find_by_name("wood furniture")[0]["entity_id"] == entity_id
    assert memory.find_by_name("family table")[0]["entity_id"] == entity_id
    memory.close()

    reopened = make_memory(database)
    entity = reopened.get_entity(entity_id)
    assert entity["canonical_name"] == "dining table"
    assert entity["name_locked"]
    assert "large table" in entity["aliases"]
    reopened.close()


def test_correction_operation_is_idempotent_and_old_revision_is_rejected(tmp_path):
    memory = make_memory(tmp_path / "memory.sqlite3")
    entity_id = create_table(memory)
    correction = SemanticCorrection(
        "same-operation", entity_id, ORIGIN_NS + 2, 0, "desk", 0.8
    )
    first = memory.enqueue_correction(correction)
    duplicate = memory.enqueue_correction(correction)
    assert first.status == "pending"
    assert duplicate.duplicate
    assert memory.apply_pending_corrections()[0].status == "applied"
    assert memory.enqueue_correction(correction).duplicate

    assert memory.advance_revision("verified_loop", ORIGIN_NS + 3) == 1
    stale = SemanticCorrection(
        "stale-operation", entity_id, ORIGIN_NS + 4, 0, "old label", 1.0
    )
    memory.enqueue_correction(stale)
    receipt = memory.apply_pending_corrections()[0]
    assert receipt.status == "rejected"
    assert receipt.reason == "stale_revision"
    assert memory.get_entity(entity_id)["canonical_name"] == "desk"
    memory.close()


def test_pending_updates_for_same_entity_are_coalesced(tmp_path):
    memory = make_memory(tmp_path / "memory.sqlite3")
    entity_id = create_table(memory)
    older = SemanticCorrection(
        "older", entity_id, ORIGIN_NS + 2, 0, "old guess", 0.6
    )
    newer = SemanticCorrection(
        "newer", entity_id, ORIGIN_NS + 3, 0, "new guess", 0.9
    )
    memory.enqueue_correction(older)
    memory.enqueue_correction(newer)
    stats = memory.correction_stats()
    assert stats["pending"] == 1
    assert stats["superseded"] == 1
    receipts = memory.apply_pending_corrections()
    assert [receipt.operation_id for receipt in receipts] == ["newer"]
    assert memory.get_entity(entity_id)["canonical_name"] == "new guess"
    memory.close()


def test_different_start_session_merges_only_after_verified_registration(tmp_path):
    memory = make_memory(tmp_path / "memory.sqlite3")
    entity_id = create_table(memory)
    memory.create_session("session-b", ORIGIN_NS + 1_000_000_000)
    rejected = memory.register_session(
        "session-b",
        translation(-10.0),
        np.eye(6) * 0.01,
        inlier_count=2,
        rms_error_m=0.05,
    )
    assert not rejected.accepted
    with pytest.raises(RuntimeError, match="not registered"):
        memory.observe_entity(
            "session-b",
            "local-table",
            np.array([11.0, 0.0, 0.0]),
            sensor_time_ns=ORIGIN_NS + 1_100_000_000,
            semantic_label="table",
        )

    accepted = memory.register_session(
        "session-b",
        translation(-10.0),
        np.eye(6) * 0.01,
        inlier_count=30,
        rms_error_m=0.04,
    )
    assert accepted.accepted
    merged_id, created = memory.observe_entity(
        "session-b",
        "local-table",
        np.array([11.02, 0.0, 0.0]),
        sensor_time_ns=ORIGIN_NS + 1_100_000_000,
        semantic_label="table",
        confidence=0.8,
    )
    assert not created
    assert merged_id == entity_id
    memory.close()


def test_future_revision_correction_waits_and_tombstone_remains_auditable(tmp_path):
    memory = make_memory(tmp_path / "memory.sqlite3")
    entity_id = create_table(memory)
    future = SemanticCorrection(
        "future", entity_id, ORIGIN_NS + 2, 1, "future label", 0.9
    )
    memory.enqueue_correction(future)
    assert memory.apply_pending_corrections() == []
    assert memory.correction_stats()["pending"] == 1
    memory.advance_revision("loop", ORIGIN_NS + 3)
    assert memory.apply_pending_corrections()[0].status == "applied"
    memory.delete_entity(entity_id, sensor_time_ns=ORIGIN_NS + 4)
    assert memory.get_entity(entity_id)["deleted_ns"] == ORIGIN_NS + 4
    assert any(row["action"] == "entity_deleted" for row in memory.audit_log())
    memory.close()


def test_rollback_creates_new_revision_and_restores_prior_name(tmp_path):
    memory = make_memory(tmp_path / "memory.sqlite3")
    entity_id = create_table(memory)
    memory.advance_revision("before_edit", ORIGIN_NS + 2)
    memory.set_user_name(
        entity_id, "edited name", sensor_time_ns=ORIGIN_NS + 3, lock=True
    )
    memory.advance_revision("after_edit", ORIGIN_NS + 4)
    revision = memory.rollback_to_revision(0, sensor_time_ns=ORIGIN_NS + 5)
    assert revision == 3
    assert memory.get_entity(entity_id)["canonical_name"] == "table"
    assert memory.current_revision == 3
    memory.close()


def test_map_memory_cli_edits_and_queries_locked_user_name(tmp_path):
    database = tmp_path / "memory.sqlite3"
    memory = make_memory(database)
    entity_id = create_table(memory)
    memory.close()

    edit = subprocess.run(
        [
            sys.executable,
            str(MEMORY_CLI),
            "--database",
            str(database),
            "name",
            entity_id,
            "inspection desk",
            "--alias",
            "workbench",
            "--sensor-time-ns",
            str(ORIGIN_NS + 20),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(edit.stdout)["result"]["name_locked"]

    query = subprocess.run(
        [
            sys.executable,
            str(MEMORY_CLI),
            "--database",
            str(database),
            "find",
            "workbench",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(query.stdout)["result"]
    assert result[0]["entity_id"] == entity_id

    reopened = make_memory(database)
    assert reopened.get_entity(entity_id)["canonical_name"] == "inspection desk"
    reopened.close()
