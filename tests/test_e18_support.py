from __future__ import annotations

from daaam.experiments.e18_support import (
    compare_exact_repetitions,
    evaluate_durable_commit,
    stable_records_sha256,
    validate_postpass_report,
)


def _report() -> dict:
    return {
        "schema": "daaam.hydra_semantic_postpass.v1",
        "status": "complete",
        "frames_expected": 2,
        "frames_replayed": 2,
        "frames_with_labels": 2,
        "label_coverage": 1.0,
        "missing_frame_indices": [],
        "nonzero_label_frames": 2,
        "nonzero_label_pixels": 10,
        "unique_semantic_labels": [0, 1],
        "label_manifest_sha256": "a" * 64,
        "label_run_configuration_sha256": "b" * 64,
    }


def test_validate_postpass_report_collects_all_contract_violations():
    report = _report()
    assert (
        validate_postpass_report(
            report,
            expected_frames=2,
            expected_label_manifest_sha256="a" * 64,
            expected_run_configuration_sha256="b" * 64,
        )
        == []
    )
    report["frames_replayed"] = 1
    report["label_coverage"] = 0.5
    assert validate_postpass_report(
        report,
        expected_frames=2,
        expected_label_manifest_sha256="a" * 64,
        expected_run_configuration_sha256="b" * 64,
    ) == ["frames_replayed_mismatch", "label_coverage_not_one"]


def test_repetition_comparison_separates_contract_and_byte_stability():
    rows = [
        {
            "report": _report(),
            "artifact_hashes": {
                "backend/mesh.ply": "1",
                "backend/dsg.json": "2",
                "backend/dsg_with_mesh.json": "3",
                "backend/deformation_graph.dgrf": "4",
            },
            "graph_summary": {"objects": 4},
        },
        {
            "report": _report(),
            "artifact_hashes": {
                "backend/mesh.ply": "1",
                "backend/dsg.json": "changed",
                "backend/dsg_with_mesh.json": "3",
                "backend/deformation_graph.dgrf": "4",
            },
            "graph_summary": {"objects": 4},
        },
    ]
    result = compare_exact_repetitions(rows)
    assert result["semantic_contract_stable"] is True
    assert result["formal_product_hash_stable"] is False
    assert result["product_stability"]["backend/dsg.json"] is False
    assert result["graph_summary_stable"] is True


def test_durable_commit_keeps_candidate_review_out_of_delivery_pending():
    result = evaluate_durable_commit(
        applied=54,
        rejected_no_mesh=33,
        delivery_pending=0,
        unmapped=0,
        errors=[],
        graph_reloaded=True,
        output_hash_verified=True,
        candidate_review_pending=11,
    )
    assert result["status"] == "passed"
    assert result["candidate_review_pending"] == 11
    assert result["candidate_review_pending_is_nonblocking"] is True


def test_stable_records_hash_is_order_sensitive_and_key_order_invariant():
    first = stable_records_sha256([{"a": 1, "b": 2}, {"a": 3}])
    same = stable_records_sha256([{"b": 2, "a": 1}, {"a": 3}])
    reversed_rows = stable_records_sha256([{"a": 3}, {"a": 1, "b": 2}])
    assert first == same
    assert first != reversed_rows
