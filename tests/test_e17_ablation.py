from __future__ import annotations

from daaam.experiments.e17_ablation import (
    GlobalCrossIdGate,
    eligible_global_cross_id_candidates,
    global_one_to_one_assignment,
    symmetric_volume_ratio,
)


def _row(entity, ordinal, node, label, center, gap, iou, first, second):
    return {
        "entity_id": entity,
        "entity_ordinal": ordinal,
        "node_id": node,
        "candidate_semantic_id": label,
        "semantic_id_match": ordinal == label,
        "rejected_reserved_owner": False,
        "center_distance_m": center,
        "aabb_gap_m": gap,
        "aabb_iou": iou,
        "entity_dimensions_m": first,
        "node_dimensions_m": second,
    }


def test_symmetric_volume_ratio_is_order_invariant():
    assert symmetric_volume_ratio([1, 2, 3], [1, 1, 1]) == 6.0
    assert symmetric_volume_ratio([1, 1, 1], [1, 2, 3]) == 6.0


def test_joint_gate_rejects_adjacent_different_track_surface():
    rows = [
        _row(
            "ceiling",
            8,
            "O(77)",
            77,
            0.8,
            0.0,
            0.3,
            [1, 1, 1],
            [1, 1, 1],
        )
    ]
    assert not eligible_global_cross_id_candidates(
        rows,
        {8: {9}, 77: {34}},
        exact_entity_ids=set(),
        exact_node_ids=set(),
    )


def test_global_assignment_selects_lowest_cost_same_track_fragment():
    rows = [
        _row(
            "wall-a",
            33,
            "O(77)",
            77,
            0.86,
            0.0,
            0.25,
            [1.0, 2.0, 1.0],
            [1.0, 2.0, 1.0],
        ),
        _row(
            "wall-b",
            51,
            "O(77)",
            77,
            0.81,
            0.0,
            0.35,
            [1.0, 2.0, 1.0],
            [1.0, 2.0, 1.0],
        ),
    ]
    eligible = eligible_global_cross_id_candidates(
        rows,
        {33: {34}, 51: {34}, 77: {34}},
        exact_entity_ids=set(),
        exact_node_ids=set(),
        gate=GlobalCrossIdGate(),
    )
    selected = global_one_to_one_assignment(eligible)
    assert len(eligible) == 2
    assert len(selected) == 1
    assert selected[0]["entity_id"] == "wall-b"
