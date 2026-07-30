from __future__ import annotations

import pytest

from daaam.experiments.e17_support import (
    BindingThreshold,
    aabb_evidence,
    build_candidate_rows,
    perturb_matched_candidate,
    summarize_decisions,
    terminal_decisions,
)


def test_aabb_overlap_accepts_strict_gate_even_when_centers_are_far():
    threshold = BindingThreshold("strict", 0.10, 0.025)
    evidence = aabb_evidence(
        [0.0, 0.0, 0.0],
        [4.0, 1.0, 1.0],
        [1.5, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        threshold,
    )
    assert evidence["center_distance_m"] == pytest.approx(1.5)
    assert evidence["aabb_gap_m"] == pytest.approx(0.0)
    assert evidence["accepted_by_center_distance"] is False
    assert evidence["accepted_by_aabb_gap"] is True
    assert evidence["accepted"] is True


def test_candidate_matrix_marks_other_named_semantic_ids_as_reserved():
    threshold = BindingThreshold("wide", 0.75, 0.15)
    entities = [
        {
            "entity_id": "entity-one",
            "semantic_id": 1,
            "description": "one",
            "position_m": [0.0, 0.0, 0.0],
            "dimensions_m": [1.0, 1.0, 1.0],
        }
    ]
    nodes = [
        {
            "node_id": "O(2)",
            "semantic_id": 2,
            "position_m": [0.0, 0.0, 0.0],
            "dimensions_m": [1.0, 1.0, 1.0],
            "mesh_vertices": 12,
        }
    ]
    rows = build_candidate_rows(
        entities,
        nodes,
        threshold,
        {1: "entity-one", 2: "entity-two"},
        geometry_source="test",
    )
    assert len(rows) == 1
    assert rows[0]["accepted"] is True
    assert rows[0]["rejected_reserved_owner"] is True
    assert rows[0]["eligible_before_assignment"] is False


def test_terminal_reduction_and_summary_keep_proxy_names_explicit():
    entities = [
        {"entity_id": "one", "semantic_id": 1, "description": "one"},
        {"entity_id": "two", "semantic_id": 2, "description": "two"},
    ]
    events = [
        {
            "entity_id": "one",
            "semantic_id": 1,
            "status": "matched_real_mesh",
            "node_id": "O(4)",
            "candidate_semantic_id": 1,
            "center_distance_m": 0.2,
            "aabb_gap_m": 0.0,
        },
        {
            "entity_id": "two",
            "semantic_id": 2,
            "status": "rejected_entity_conflict",
        },
        {
            "entity_id": "two",
            "semantic_id": 2,
            "status": "rejected_no_mesh",
        },
    ]
    decisions = terminal_decisions(events, entities)
    summary = summarize_decisions(
        decisions,
        named_entity_count=2,
        real_mesh_count=3,
    )
    assert summary["matched_real_mesh"] == 1
    assert summary["rejected_no_mesh"] == 1
    assert summary["provenance_consistent_semantic_id"] == 1
    assert summary["formal_binding_precision"] is None
    assert decisions[1]["conflict_event_count"] == 1


def test_terminal_reduction_rejects_missing_entity_decision():
    with pytest.raises(ValueError, match="coverage mismatch"):
        terminal_decisions(
            [],
            [{"entity_id": "one", "semantic_id": 1, "description": "one"}],
        )


def test_radial_perturbation_recomputes_exact_gap():
    threshold = BindingThreshold("strict", 0.10, 0.025)
    candidate = {
        "entity_position_m": [0.0, 0.0, 0.0],
        "entity_dimensions_m": [0.2, 0.2, 0.2],
        "node_position_m": [0.0, 0.0, 0.0],
        "node_dimensions_m": [0.2, 0.2, 0.2],
    }
    result = perturb_matched_candidate(
        candidate,
        dose_m=0.3,
        threshold=threshold,
    )
    assert result["aabb_gap_before_m"] == pytest.approx(0.0)
    assert result["aabb_gap_after_m"] == pytest.approx(0.1)
    assert result["chosen_pair_rejected_due_to_injection"] is True
