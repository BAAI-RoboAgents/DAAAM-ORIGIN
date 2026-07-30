"""Pure helpers for the G1 E17 entity-to-mesh binding experiment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BindingThreshold:
    """One preregistered E17 spatial gate."""

    name: str
    maximum_center_distance_m: float
    maximum_aabb_gap_m: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("binding threshold name cannot be empty")
        if self.maximum_center_distance_m <= 0.0:
            raise ValueError("center-distance threshold must be positive")
        if self.maximum_aabb_gap_m < 0.0:
            raise ValueError("AABB-gap threshold cannot be negative")


E17_THRESHOLDS = (
    BindingThreshold("strict", 0.10, 0.025),
    BindingThreshold("medium", 0.35, 0.075),
    BindingThreshold("wide", 0.75, 0.15),
)


def _vector(value: Any, *, positive: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("geometry vector must contain three finite values")
    if positive and np.any(result <= 0.0):
        raise ValueError("dimensions must be positive")
    return result


def aabb_evidence(
    entity_position_m: Any,
    entity_dimensions_m: Any,
    node_position_m: Any,
    node_dimensions_m: Any,
    threshold: BindingThreshold,
) -> dict[str, Any]:
    """Reproduce the production center/AABB OR gate without DSG dependencies."""

    entity_position = _vector(entity_position_m)
    entity_dimensions = _vector(entity_dimensions_m, positive=True)
    node_position = _vector(node_position_m)
    node_dimensions = _vector(node_dimensions_m, positive=True)
    delta = np.abs(node_position - entity_position)
    separation = np.maximum(
        delta - 0.5 * (node_dimensions + entity_dimensions),
        0.0,
    )
    center_distance = float(np.linalg.norm(node_position - entity_position))
    aabb_gap = float(np.linalg.norm(separation))
    intersection_dimensions = np.maximum(
        np.minimum(
            node_position + 0.5 * node_dimensions,
            entity_position + 0.5 * entity_dimensions,
        )
        - np.maximum(
            node_position - 0.5 * node_dimensions,
            entity_position - 0.5 * entity_dimensions,
        ),
        0.0,
    )
    intersection = float(np.prod(intersection_dimensions))
    union = float(
        np.prod(node_dimensions) + np.prod(entity_dimensions) - intersection
    )
    aabb_iou = intersection / union if union > 0.0 else 0.0
    accepted_by_center = center_distance <= threshold.maximum_center_distance_m
    accepted_by_gap = aabb_gap <= threshold.maximum_aabb_gap_m
    return {
        "center_distance_m": center_distance,
        "aabb_gap_m": aabb_gap,
        "aabb_iou": aabb_iou,
        "accepted_by_center_distance": accepted_by_center,
        "accepted_by_aabb_gap": accepted_by_gap,
        "accepted": accepted_by_center or accepted_by_gap,
    }


def build_candidate_rows(
    entities: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    threshold: BindingThreshold,
    semantic_id_owners: Mapping[int, str],
    *,
    geometry_source: str,
) -> list[dict[str, Any]]:
    """Build the complete order-independent entity × real-mesh evidence matrix."""

    rows: list[dict[str, Any]] = []
    for entity in entities:
        entity_id = str(entity["entity_id"])
        semantic_id = int(entity["semantic_id"])
        for node in nodes:
            candidate_semantic_id = int(node["semantic_id"])
            evidence = aabb_evidence(
                entity["position_m"],
                entity["dimensions_m"],
                node["position_m"],
                node["dimensions_m"],
                threshold,
            )
            reserved_owner = semantic_id_owners.get(candidate_semantic_id)
            semantic_id_match = candidate_semantic_id == semantic_id
            rejected_reserved_owner = bool(
                evidence["accepted"]
                and reserved_owner is not None
                and (
                    reserved_owner != entity_id
                    or candidate_semantic_id != semantic_id
                )
            )
            rows.append(
                {
                    "schema": "daaam.g1_e17_binding_candidate.v1",
                    "geometry_source": geometry_source,
                    "threshold": threshold.name,
                    "maximum_center_distance_m": (
                        threshold.maximum_center_distance_m
                    ),
                    "maximum_aabb_gap_m": threshold.maximum_aabb_gap_m,
                    "entity_id": entity_id,
                    "entity_ordinal": semantic_id,
                    "entity_label": str(entity.get("description") or ""),
                    "entity_position_m": list(map(float, entity["position_m"])),
                    "entity_dimensions_m": list(map(float, entity["dimensions_m"])),
                    "node_id": str(node["node_id"]),
                    "candidate_semantic_id": candidate_semantic_id,
                    "node_position_m": list(map(float, node["position_m"])),
                    "node_dimensions_m": list(map(float, node["dimensions_m"])),
                    "mesh_vertices": int(node["mesh_vertices"]),
                    "semantic_id_match": semantic_id_match,
                    "reserved_owner": reserved_owner,
                    "rejected_reserved_owner": rejected_reserved_owner,
                    "eligible_before_assignment": bool(
                        evidence["accepted"] and not rejected_reserved_owner
                    ),
                    **evidence,
                }
            )
    return rows


def terminal_decisions(
    events: Iterable[Mapping[str, Any]],
    entities: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reduce the append-only production audit to one terminal row per entity."""

    terminal_statuses = {"matched_real_mesh", "rejected_no_mesh"}
    by_entity: dict[str, list[Mapping[str, Any]]] = {}
    all_events = list(events)
    for event in all_events:
        if event.get("status") in terminal_statuses:
            by_entity.setdefault(str(event.get("entity_id")), []).append(event)
    entity_lookup = {str(entity["entity_id"]): entity for entity in entities}
    if set(by_entity) != set(entity_lookup):
        missing = sorted(set(entity_lookup) - set(by_entity))
        unexpected = sorted(set(by_entity) - set(entity_lookup))
        raise ValueError(
            f"terminal binding coverage mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    duplicate = sorted(
        entity_id for entity_id, rows in by_entity.items() if len(rows) != 1
    )
    if duplicate:
        raise ValueError(f"entities have multiple terminal decisions: {duplicate}")

    event_counts: dict[str, Counter[str]] = {}
    for event in all_events:
        entity_id = str(event.get("entity_id") or "")
        if entity_id:
            event_counts.setdefault(entity_id, Counter())[str(event.get("status"))] += 1

    decisions = []
    for entity in sorted(entities, key=lambda item: int(item["semantic_id"])):
        entity_id = str(entity["entity_id"])
        event = dict(by_entity[entity_id][0])
        status = str(event["status"])
        candidate_semantic_id = event.get("candidate_semantic_id")
        semantic_id_match = bool(
            status == "matched_real_mesh"
            and candidate_semantic_id is not None
            and int(candidate_semantic_id) == int(entity["semantic_id"])
        )
        risk_class = (
            "provenance_consistent_semantic_id"
            if semantic_id_match
            else (
                "spatial_fallback_requires_review"
                if status == "matched_real_mesh"
                else "rejected_no_mesh"
            )
        )
        nearest = event.get("nearest_rejected_candidate") or {}
        counts = event_counts.get(entity_id, Counter())
        decisions.append(
            {
                "schema": "daaam.g1_e17_terminal_decision.v1",
                "entity_id": entity_id,
                "entity_ordinal": int(entity["semantic_id"]),
                "entity_label": str(entity.get("description") or ""),
                "status": status,
                "node_id": event.get("node_id"),
                "candidate_semantic_id": candidate_semantic_id,
                "semantic_id_match": semantic_id_match,
                "risk_class": risk_class,
                "center_distance_m": event.get("center_distance_m"),
                "aabb_gap_m": event.get("aabb_gap_m"),
                "aabb_iou": event.get("aabb_iou"),
                "accepted_by": event.get("accepted_by") or [],
                "nearest_rejected_node_id": nearest.get("node_id"),
                "nearest_rejected_semantic_id": nearest.get(
                    "candidate_semantic_id"
                ),
                "nearest_rejected_center_distance_m": nearest.get(
                    "center_distance_m"
                ),
                "nearest_rejected_aabb_gap_m": nearest.get("aabb_gap_m"),
                "conflict_event_count": int(
                    counts.get("rejected_entity_conflict", 0)
                ),
                "reserved_owner_rejection_count": int(
                    counts.get("rejected_reserved_semantic_owner", 0)
                ),
            }
        )
    return decisions


def summarize_decisions(
    decisions: Sequence[Mapping[str, Any]],
    *,
    named_entity_count: int,
    real_mesh_count: int,
) -> dict[str, Any]:
    matched = [row for row in decisions if row["status"] == "matched_real_mesh"]
    exact = [row for row in matched if row["semantic_id_match"]]
    spatial = [row for row in matched if not row["semantic_id_match"]]
    rejected = [row for row in decisions if row["status"] == "rejected_no_mesh"]
    assigned_nodes = [str(row["node_id"]) for row in matched]
    duplicate_nodes = sorted(
        node_id
        for node_id, count in Counter(assigned_nodes).items()
        if count > 1
    )
    if duplicate_nodes:
        raise ValueError(f"mesh nodes assigned more than once: {duplicate_nodes}")
    return {
        "named_entity_count": int(named_entity_count),
        "real_mesh_count": int(real_mesh_count),
        "matched_real_mesh": len(matched),
        "rejected_no_mesh": len(rejected),
        "provenance_consistent_semantic_id": len(exact),
        "spatial_fallback_requires_review": len(spatial),
        "named_entity_binding_coverage_proxy": (
            len(matched) / named_entity_count if named_entity_count else 0.0
        ),
        "semantic_id_consistency_among_matches_proxy": (
            len(exact) / len(matched) if matched else 0.0
        ),
        "mesh_utilization_by_named_entities": (
            len(matched) / real_mesh_count if real_mesh_count else 0.0
        ),
        "entity_conflict_events": int(
            sum(int(row["conflict_event_count"]) for row in decisions)
        ),
        "reserved_owner_rejections": int(
            sum(int(row["reserved_owner_rejection_count"]) for row in decisions)
        ),
        "formal_binding_precision": None,
        "formal_binding_recall": None,
        "formal_binding_f1": None,
    }


def perturb_matched_candidate(
    candidate: Mapping[str, Any],
    *,
    dose_m: float,
    threshold: BindingThreshold,
) -> dict[str, Any]:
    """Move one entity radially away and exactly recompute its chosen-pair gate."""

    if dose_m <= 0.0:
        raise ValueError("perturbation dose must be positive")
    entity_position = _vector(candidate["entity_position_m"])
    node_position = _vector(candidate["node_position_m"])
    direction = entity_position - node_position
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    perturbed = entity_position + float(dose_m) * direction
    before = aabb_evidence(
        entity_position,
        candidate["entity_dimensions_m"],
        node_position,
        candidate["node_dimensions_m"],
        threshold,
    )
    after = aabb_evidence(
        perturbed,
        candidate["entity_dimensions_m"],
        node_position,
        candidate["node_dimensions_m"],
        threshold,
    )
    return {
        "dose_m": float(dose_m),
        "entity_position_before_m": entity_position.tolist(),
        "entity_position_after_m": perturbed.tolist(),
        "center_distance_before_m": before["center_distance_m"],
        "center_distance_after_m": after["center_distance_m"],
        "aabb_gap_before_m": before["aabb_gap_m"],
        "aabb_gap_after_m": after["aabb_gap_m"],
        "accepted_before": before["accepted"],
        "accepted_after": after["accepted"],
        "chosen_pair_rejected_due_to_injection": bool(
            before["accepted"] and not after["accepted"]
        ),
    }
