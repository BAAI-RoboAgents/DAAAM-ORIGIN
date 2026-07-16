"""Acceptance tests for reproducible manifests and interruption recovery."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.realtime.checkpoint import RealtimeCheckpoint  # noqa: E402
from daaam.realtime.manifest import (  # noqa: E402
    build_run_manifest,
    validate_run_manifest,
    write_run_manifest,
)


def test_manifest_records_code_dataset_configuration_and_time_contract(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "tick_index.json").write_text('{"frames": []}')
    manifest = build_run_manifest(
        REPOSITORY_ROOT,
        dataset,
        configuration={"queue_capacity": 8},
        model_configuration={"depth": {"profile": "online"}},
        time_contract={"valid": True, "frame_count": 10},
    )
    output = tmp_path / "run_manifest.json"
    write_run_manifest(output, manifest)
    loaded = json.loads(output.read_text())
    assert loaded["repository"]["git_sha"]
    assert loaded["repository"]["foundation_stereo_sha"]
    assert loaded["dataset"]["tick_index_sha256"]
    assert loaded["configuration"]["queue_capacity"] == 8
    assert loaded["time_contract"]["valid"]


def test_invalid_manifest_is_rejected():
    with pytest.raises(ValueError, match="missing fields"):
        validate_run_manifest({"manifest_version": 1})


def test_checkpoint_tracks_sparse_completion_and_restores_state(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = RealtimeCheckpoint(path, dataset_fingerprint="abc")
    checkpoint.mark_completed(
        3,
        100,
        map_revision=2,
        dynamic_layer={"active": [{"entity_id": "one"}], "history": []},
        submaps={"map_revision": 2, "submaps": []},
        paths={"paths": [{"path_id": "path-one"}]},
        path_buffer={"sensor_times_ns": [100], "points_m": [[1.0, 0.0, 0.0]]},
    )
    checkpoint.mark_completed(
        1,
        90,
        map_revision=2,
        dynamic_layer={"active": [], "history": []},
        submaps={"map_revision": 2, "submaps": []},
    )
    checkpoint.mark_dropped(2, "deadline_expired")
    restored = RealtimeCheckpoint(path, dataset_fingerprint="abc")
    assert restored.load()
    assert restored.completed_indices == {1, 3}
    assert restored.state["last_sensor_time_ns"] == 100
    assert restored.state["dropped_frames"] == {"2": "deadline_expired"}
    assert restored.state["paths"]["paths"][0]["path_id"] == "path-one"


def test_checkpoint_refuses_different_dataset(tmp_path):
    path = tmp_path / "checkpoint.json"
    first = RealtimeCheckpoint(path, dataset_fingerprint="first")
    first.mark_dropped(0, "test")
    second = RealtimeCheckpoint(path, dataset_fingerprint="second")
    with pytest.raises(ValueError, match="fingerprint"):
        second.load()
