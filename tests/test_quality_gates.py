"""Acceptance tests for hard stage gates and stable failure codes."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.quality import QualityGateRunner  # noqa: E402


def valid_context():
    return {
        "time": {
            "valid": True,
            "monotonic": True,
            "pose_exact_match": True,
            "relative_time_consistent": True,
            "maximum_stereo_delta_ms": 2.0,
            "projection_model": "pinhole",
        },
        "depth": {
            "valid_ratio": 0.8,
            "temporal_agreement": 0.9,
            "left_right_consistency": 0.85,
            "left_right_coverage": 1.0,
        },
        "pose": {
            "maximum_translation_step_m": 0.1,
            "maximum_rotation_step_deg": 3.0,
            "maximum_position_std_m": 0.1,
            "timestamps_monotonic": True,
        },
        "dynamic": {
            "dynamic_contamination_rate": 0.0,
            "unknown_ratio": 0.1,
        },
        "runtime": {
            "stages": {
                "pose": {"latency": {"service_ms": {"p95": 10.0}}},
                "depth": {"latency": {"service_ms": {"p95": 200.0}}},
            }
        },
        "map": {"largest_component_ratio": 0.8, "connected_components": 20},
        "semantic": {"pending": 0, "applied": 20, "rejected": 0},
    }


def result_codes(report):
    return {result["stage"]: result["code"] for result in report["results"]}


def test_all_valid_stage_evidence_passes():
    report = QualityGateRunner().evaluate(valid_context())
    assert report["passed"]
    assert report["hard_failures"] == 0


def test_each_anomaly_has_an_explainable_failure_code():
    mutations = {
        "time": ("pose_exact_match", False, "time.contract_violation"),
        "depth": ("temporal_agreement", 0.1, "depth.inconsistent"),
        "pose": ("maximum_translation_step_m", 2.0, "pose.jump_or_uncertainty"),
        "dynamic": (
            "dynamic_contamination_rate",
            0.4,
            "dynamic.static_contamination",
        ),
        "map": ("connected_components", 5000, "map.fragmented_mesh"),
        "semantic": ("pending", 50, "semantic.pending_backlog"),
    }
    for stage, (field, value, code) in mutations.items():
        context = valid_context()
        context[stage][field] = value
        report = QualityGateRunner().evaluate(context)
        assert not report["passed"]
        assert result_codes(report)[stage] == code


def test_runtime_latency_failure_and_missing_evidence_block_pipeline():
    context = valid_context()
    context["runtime"]["stages"]["depth"]["latency"]["service_ms"]["p95"] = 500.0
    report = QualityGateRunner().evaluate(context)
    assert result_codes(report)["runtime"] == "runtime.p95_exceeded"
    missing = QualityGateRunner().evaluate({}, required_stages=["time"])
    assert not missing["passed"]
    assert result_codes(missing)["time"] == "time.missing_evidence"


def test_depth_gate_rejects_insufficient_left_right_validation_coverage():
    context = valid_context()
    context["depth"]["left_right_coverage"] = 0.2
    report = QualityGateRunner().evaluate(context)
    depth = next(result for result in report["results"] if result["stage"] == "depth")
    assert depth["code"] == "depth.inconsistent"
    assert depth["metrics"]["left_right_coverage"] == 0.2


def test_depth_gate_distinguishes_missing_lr_provenance_from_bad_consistency():
    context = valid_context()
    context["depth"].update(
        {
            "left_right_consistency": 0.0,
            "left_right_coverage": 0.0,
            "left_right_evidence_available": False,
        }
    )
    report = QualityGateRunner().evaluate(context)
    assert result_codes(report)["depth"] == "depth.missing_lr_evidence"


def test_required_dam_gate_requires_real_worker_and_hydra_ack():
    context = valid_context()
    context["semantic"] = {
        "required": True,
        "submitted": 1,
        "pending": 0,
        "applied": 1,
        "rejected": 0,
        "grounding_workers": {"all_ready": True},
        "dsg": {
            "graph_attached": True,
            "applied": 1,
            "pending": 0,
            "unmapped": 0,
            "errors": [],
        },
    }
    assert QualityGateRunner().evaluate(context)["passed"]

    context["semantic"]["dsg"]["pending"] = 1
    report = QualityGateRunner().evaluate(context)
    assert result_codes(report)["semantic"] == "semantic.dsg_pending"


def test_required_dam_gate_rejects_missing_work_and_worker_readiness():
    context = valid_context()
    context["semantic"] = {
        "required": True,
        "submitted": 0,
        "pending": 0,
        "applied": 0,
        "rejected": 0,
        "grounding_workers": {"all_ready": True},
        "dsg": {"graph_attached": True, "errors": []},
    }
    report = QualityGateRunner().evaluate(context)
    assert result_codes(report)["semantic"] == "semantic.no_requests"

    context["semantic"]["grounding_workers"]["all_ready"] = False
    report = QualityGateRunner().evaluate(context)
    assert result_codes(report)["semantic"] == "semantic.worker_unavailable"


def test_mesh_gate_uses_significant_area_without_hiding_raw_fragmentation():
    context = valid_context()
    context["map"] = {
        "connected_components": 2459,
        "largest_component_area_ratio": 0.71,
        "minimum_significant_component_area_m2": 0.005,
        "significant_connected_components": 466,
        "tiny_component_area_ratio": 0.049,
    }
    report = QualityGateRunner().evaluate(context)
    result = next(value for value in report["results"] if value["stage"] == "map")
    assert result["code"] == "map.connectivity"
    assert result["metrics"]["connected_components"] == 2459
    assert result["metrics"]["significant_connected_components"] == 466


def test_mesh_gate_rejects_excess_tiny_surface_area():
    context = valid_context()
    context["map"] = {
        "connected_components": 900,
        "largest_component_area_ratio": 0.8,
        "minimum_significant_component_area_m2": 0.005,
        "significant_connected_components": 100,
        "tiny_component_area_ratio": 0.051,
    }
    report = QualityGateRunner().evaluate(context)
    assert result_codes(report)["map"] == "map.fragmented_mesh"


def test_runtime_queue_backlog_and_drop_ratio_are_hard_failures():
    context = valid_context()
    context["runtime"]["stages"]["depth"]["latency"]["queue_wait_ms"] = {
        "p95": 900.0
    }
    context["runtime"]["totals"] = {"processed": 80, "dropped": 20}
    report = QualityGateRunner().evaluate(context)
    runtime = next(result for result in report["results"] if result["stage"] == "runtime")
    assert runtime["code"] == "runtime.p95_exceeded"
    assert "depth" in runtime["metrics"]["queue_exceeded"]
    assert runtime["metrics"]["drop_ratio"] == 0.2


def test_runtime_resource_limit_has_distinct_failure_code():
    context = valid_context()
    context["runtime"]["resources"] = {
        "depth_peak_cuda_memory_bytes": 21_000_000_000,
        "depth_peak_worker_rss_bytes": 1_000_000_000,
        "depth_worker_restarts": 0,
    }
    report = QualityGateRunner().evaluate(context)
    assert result_codes(report)["runtime"] == "runtime.resource_exceeded"


def test_runtime_handler_error_is_always_a_hard_failure():
    context = valid_context()
    context["runtime"]["totals"] = {
        "processed": 10,
        "dropped": 0,
        "errors": 1,
    }
    report = QualityGateRunner().evaluate(context)
    assert result_codes(report)["runtime"] == "runtime.stage_error"


def test_quality_cli_returns_nonzero_for_hard_failure(tmp_path):
    context = valid_context()
    context["map"] = {
        "largest_component_ratio": 0.0184,
        "connected_components": 4013,
    }
    (tmp_path / "quality_context.json").write_text(json.dumps(context))
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "evaluate_mapping_quality.py"),
            "--run-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    report = json.loads((tmp_path / "quality_report.json").read_text())
    assert result_codes(report)["map"] == "map.fragmented_mesh"
