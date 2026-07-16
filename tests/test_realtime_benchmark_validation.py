"""Acceptance tests for clean-HEAD paired realtime benchmark authority."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.quality.benchmark import validate_benchmark_pair  # noqa: E402


def write_run(root: Path, rate_hz: float, *, dirty: bool = False) -> Path:
    root.mkdir()
    documents = {
        "run_manifest.json": {
            "repository": {"git_sha": "abc123", "git_dirty": dirty},
            "dataset": {"tick_index_sha256": "dataset-sha"},
            "configuration": {
                "rate_hz": rate_hz,
                "stage_rate_multiplier": 1.0,
                "no_throttle": False,
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
            },
        },
        "quality_report.json": {"passed": True, "hard_failures": 0},
        "realtime_run_report.json": {
            "status": "complete",
            "frames_requested": 20,
            "frames_completed": 20,
            "dropped_frames": {},
            "quality_passed": True,
            "semantic_mode": "dam",
            "semantic_stats": {
                "prompts_submitted": 1,
                "corrections_submitted": 1,
                "dsg": {
                    "graph_attached": True,
                    "applied": 1,
                    "pending": 0,
                    "unmapped": 0,
                    "errors": [],
                },
            },
        },
    }
    for name, document in documents.items():
        (root / name).write_text(json.dumps(document))
    return root


def test_clean_matching_5_and_10_hz_runs_are_authoritative(tmp_path):
    run_5hz = write_run(tmp_path / "five", 5.0)
    run_10hz = write_run(tmp_path / "ten", 10.0)
    verdict = validate_benchmark_pair(run_5hz, run_10hz)
    assert verdict["passed"]
    assert verdict["authoritative"]
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
