"""Acceptance tests for clean-HEAD realtime benchmark authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.quality.benchmark import (  # noqa: E402
    validate_benchmark_pair,
    validate_realtime_run,
)


def write_run(root: Path, rate_hz: float, *, dirty: bool = False) -> Path:
    root.mkdir()
    backend = root / "hydra_realtime" / "backend"
    backend.mkdir(parents=True)
    graph = {
        "nodes": [
            {
                "id": 10,
                "layer": 3,
                "partition": 0,
                "attributes": {"type": "PlaceNodeAttributes"},
            },
            {
                "id": 20,
                "layer": 2,
                "partition": 0,
                "attributes": {
                    "type": "KhronosObjectAttributes",
                    "semantic_label": 7,
                    "is_active": True,
                    "metadata": {
                        "entity_id": "entity-chair",
                        "description": "wooden chair",
                    },
                },
            },
        ],
        "edges": [{"source": 10, "target": 20, "info": {}}],
    }
    dsg_path = backend / "dsg.json"
    dsg_with_mesh_path = backend / "dsg_with_mesh.json"
    dsg_path.write_text(json.dumps(graph))
    mesh_graph = {
        **graph,
        "mesh": {
            "points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "faces": [[0, 1, 2]],
        },
    }
    dsg_with_mesh_path.write_text(json.dumps(mesh_graph))
    semantic_commit = {
        "schema": "daaam.semantic_dsg_commit.v1",
        "artifacts": {
            "dsg.json": {
                "sha256": hashlib.sha256(dsg_path.read_bytes()).hexdigest(),
                "requested_include_mesh": False,
                "has_mesh": False,
                "mesh_vertices": 0,
                "mesh_faces": 0,
                "object_count": 1,
            },
            "dsg_with_mesh.json": {
                "sha256": hashlib.sha256(
                    dsg_with_mesh_path.read_bytes()
                ).hexdigest(),
                "requested_include_mesh": True,
                "has_mesh": True,
                "mesh_vertices": 3,
                "mesh_faces": 1,
                "object_count": 1,
            },
        },
        "object_count": 1,
        "verified_entity_count": 1,
        "verified_operation_count": 1,
    }
    semantic_commit_path = backend / "semantic_dsg_commit.json"
    semantic_commit_path.write_text(json.dumps(semantic_commit))
    semantic_commit_sha256 = hashlib.sha256(
        semantic_commit_path.read_bytes()
    ).hexdigest()
    quality_config = REPOSITORY_ROOT / "config" / "realtime_quality_gates.yaml"
    quality_sha256 = hashlib.sha256(quality_config.read_bytes()).hexdigest()
    documents = {
        "run_manifest.json": {
            "repository": {"git_sha": "abc123", "git_dirty": dirty},
            "dataset": {"tick_index_sha256": "dataset-sha"},
            "time_contract": {"frame_count": 20},
            "configuration": {
                "rate_hz": rate_hz,
                "max_frames": None,
                "allow_source_bursts": False,
                "stage_rate_multiplier": 1.0,
                "no_throttle": False,
                "quality_config_sha256": quality_sha256,
                "fault": {"stage": None, "delay_ms": 0.0, "error_every": 0},
            },
            "models": {
                "foundation_stereo": {
                    "valid_iters": 8,
                    "scale": 0.15,
                    "precision": "fp16",
                    "checkpoint_sha256": "model-sha",
                },
                "semantic_frontend": {
                    "fastsam": {"sha256": "fastsam-sha"},
                    "botsort_reid": {"sha256": "reid-sha"},
                    "dam": {"cached_revision": "dam-revision"},
                    "semantic_labelspace": {"sha256": "labels-sha"},
                    "labelspace_colors": {"sha256": "colors-sha"},
                },
            },
        },
        "realtime_metrics.json": {
            "elapsed_seconds": 20.0,
            "totals": {"processed": 100, "dropped": 0, "errors": 0},
            "stages": {
                "segmentation": {
                    "processed": 10,
                    "latency": {"service_ms": {"p95": 100.0}},
                },
                "tracking": {
                    "processed": 20,
                    "latency": {"service_ms": {"p95": 20.0}},
                },
                "global": {
                    "processed": 20,
                    "latency": {"end_to_end_ms": {"p95": 50.0}},
                },
            },
        },
        "quality_report.json": {"passed": True, "hard_failures": 0},
        "realtime_run_report.json": {
            "status": "complete",
            "frames_requested": 20,
            "frames_dispatched": 20,
            "frames_completed": 20,
            "frames_resumed_from": 0,
            "dropped_frames": {},
            "quality_passed": True,
            "semantic_mode": "dam",
            "semantic_stats": {
                "segmentation_calls": 10,
                "tracking_calls": 20,
                "prompts_submitted": 1,
                "corrections_submitted": 1,
                "dsg": {
                    "graph_attached": True,
                    "commit_valid": True,
                    "applied": 1,
                    "pending": 0,
                    "unmapped": 0,
                    "commit_manifest_path": str(semantic_commit_path.resolve()),
                    "commit_manifest_sha256": semantic_commit_sha256,
                    "verified_artifacts": [
                        str(dsg_path.resolve()),
                        str(dsg_with_mesh_path.resolve()),
                    ],
                    "verified_entities": 1,
                    "verified_operations": 1,
                    "errors": [],
                },
            },
            "replay_pacing": {
                "configured_max_rate_hz": rate_hz,
                "source_bursts_allowed": False,
                "absolute_timestamps_preserved": True,
                "sleep_seconds": 19.0 / rate_hz,
            },
        },
    }
    for name, document in documents.items():
        (root / name).write_text(json.dumps(document))
    return root


def test_clean_1_hz_run_is_authoritative(tmp_path):
    run = write_run(tmp_path / "one", 1.0)
    verdict = validate_realtime_run(run, expected_rate_hz=1.0)
    assert verdict["passed"]
    assert verdict["authoritative"]
    assert verdict["expected_rate_hz"] == 1.0


def test_1_hz_authority_rejects_wrong_rate_or_slow_map_cycle(tmp_path):
    wrong_rate = write_run(tmp_path / "wrong-rate", 2.0)
    rate_verdict = validate_realtime_run(wrong_rate, expected_rate_hz=1.0)
    assert "configuration.rate" in rate_verdict["blocking_failures"]

    slow = write_run(tmp_path / "slow", 1.0)
    metrics_path = slow / "realtime_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["stages"]["global"]["latency"]["end_to_end_ms"]["p95"] = 1000.1
    metrics_path.write_text(json.dumps(metrics))
    slow_verdict = validate_realtime_run(slow, expected_rate_hz=1.0)
    assert "runtime.mapping_cycle_p95" in slow_verdict["blocking_failures"]


def test_1_hz_authority_reloads_and_hashes_final_semantic_dsg(tmp_path):
    run = write_run(tmp_path / "tampered-dsg", 1.0)
    dsg_with_mesh = run / "hydra_realtime" / "backend" / "dsg_with_mesh.json"
    document = json.loads(dsg_with_mesh.read_text())
    document["nodes"][1]["attributes"]["metadata"]["description"] = "tampered"
    dsg_with_mesh.write_text(json.dumps(document))

    verdict = validate_realtime_run(run, expected_rate_hz=1.0)

    assert "semantic.final_dsg_artifacts" in verdict["blocking_failures"]
    check = next(
        item
        for item in verdict["checks"]
        if item["code"] == "semantic.final_dsg_artifacts"
    )
    assert "hash changed" in check["detail"]["error"]


def test_development_overrides_never_create_single_run_authority(tmp_path):
    run = write_run(tmp_path / "no-dam", 1.0)
    verdict = validate_realtime_run(run, expected_rate_hz=1.0, require_dam=False)
    assert verdict["passed"]
    assert not verdict["authoritative"]


def test_1_hz_authority_rejects_partial_resume_faults_and_scheduler_drops(tmp_path):
    partial = write_run(tmp_path / "partial", 1.0)
    partial_manifest = json.loads((partial / "run_manifest.json").read_text())
    partial_manifest["time_contract"]["frame_count"] = 21
    (partial / "run_manifest.json").write_text(json.dumps(partial_manifest))
    assert (
        "run.full_dataset"
        in validate_realtime_run(partial, expected_rate_hz=1.0)["blocking_failures"]
    )

    resumed = write_run(tmp_path / "resumed", 1.0)
    resumed_report = json.loads((resumed / "realtime_run_report.json").read_text())
    resumed_report["frames_resumed_from"] = 5
    (resumed / "realtime_run_report.json").write_text(json.dumps(resumed_report))
    assert (
        "run.no_resume"
        in validate_realtime_run(resumed, expected_rate_hz=1.0)["blocking_failures"]
    )

    faulted = write_run(tmp_path / "faulted", 1.0)
    faulted_manifest = json.loads((faulted / "run_manifest.json").read_text())
    faulted_manifest["configuration"]["fault"]["delay_ms"] = 1.0
    (faulted / "run_manifest.json").write_text(json.dumps(faulted_manifest))
    assert (
        "configuration.no_fault_injection"
        in validate_realtime_run(faulted, expected_rate_hz=1.0)["blocking_failures"]
    )

    dropped = write_run(tmp_path / "dropped", 1.0)
    dropped_metrics = json.loads((dropped / "realtime_metrics.json").read_text())
    dropped_metrics["totals"]["dropped"] = 1
    (dropped / "realtime_metrics.json").write_text(json.dumps(dropped_metrics))
    assert (
        "runtime.zero_scheduler_drops"
        in validate_realtime_run(dropped, expected_rate_hz=1.0)["blocking_failures"]
    )


def test_clean_matching_5_and_10_hz_runs_pass_optional_stress_checks(tmp_path):
    run_5hz = write_run(tmp_path / "five", 5.0)
    run_10hz = write_run(tmp_path / "ten", 10.0)
    verdict = validate_benchmark_pair(run_5hz, run_10hz)
    assert verdict["passed"]
    assert not verdict["authoritative"]
    assert verdict["pair_checks"] == {
        "same_clean_commit": True,
        "same_dataset": True,
    }


def test_dirty_development_override_never_becomes_authoritative(tmp_path):
    run_5hz = write_run(tmp_path / "five", 5.0, dirty=True)
    run_10hz = write_run(tmp_path / "ten", 10.0, dirty=True)
    blocked = validate_benchmark_pair(run_5hz, run_10hz)
    assert not blocked["passed"]
    assert "provenance.clean_head" in blocked["runs"][0]["blocking_failures"]

    allowed = validate_benchmark_pair(
        run_5hz,
        run_10hz,
        allow_dirty=True,
    )
    assert allowed["passed"]
    assert not allowed["authoritative"]


def test_pair_rejects_wrong_rate_even_if_individual_reports_claim_success(tmp_path):
    run_5hz = write_run(tmp_path / "five", 4.0)
    run_10hz = write_run(tmp_path / "ten", 10.0)
    verdict = validate_benchmark_pair(run_5hz, run_10hz)
    assert not verdict["passed"]
    assert "configuration.rate" in verdict["runs"][0]["blocking_failures"]
