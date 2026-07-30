"""Pure assignment helpers for the E17-v2 binding policy ablation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class GlobalCrossIdGate:
    """Preregistered D-arm gate for cross-semantic-ID fragment recovery."""

    maximum_center_distance_m: float = 1.0
    maximum_aabb_gap_m: float = 0.075
    minimum_aabb_iou: float = 0.05
    maximum_symmetric_volume_ratio: float = 4.0

    def __post_init__(self) -> None:
        if self.maximum_center_distance_m <= 0.0:
            raise ValueError("maximum center distance must be positive")
        if self.maximum_aabb_gap_m < 0.0:
            raise ValueError("maximum AABB gap cannot be negative")
        if not 0.0 <= self.minimum_aabb_iou <= 1.0:
            raise ValueError("minimum AABB IoU must be in [0, 1]")
        if self.maximum_symmetric_volume_ratio < 1.0:
            raise ValueError("maximum volume ratio must be at least one")


D_GLOBAL_GATE = GlobalCrossIdGate()


def symmetric_volume_ratio(
    first_dimensions_m: Sequence[float],
    second_dimensions_m: Sequence[float],
) -> float:
    first = np.asarray(first_dimensions_m, dtype=np.float64)
    second = np.asarray(second_dimensions_m, dtype=np.float64)
    if (
        first.shape != (3,)
        or second.shape != (3,)
        or np.any(first <= 0.0)
        or np.any(second <= 0.0)
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError("volume ratio requires finite positive 3D dimensions")
    first_volume = float(np.prod(first))
    second_volume = float(np.prod(second))
    return max(first_volume / second_volume, second_volume / first_volume)


def shared_track_ids(
    entity_ordinal: int,
    candidate_semantic_id: int,
    tracks_by_ordinal: Mapping[int, set[int]],
) -> list[int]:
    return sorted(
        tracks_by_ordinal.get(int(entity_ordinal), set())
        & tracks_by_ordinal.get(int(candidate_semantic_id), set())
    )


def eligible_global_cross_id_candidates(
    candidate_rows: Sequence[Mapping[str, Any]],
    tracks_by_ordinal: Mapping[int, set[int]],
    *,
    exact_entity_ids: set[str],
    exact_node_ids: set[str],
    gate: GlobalCrossIdGate = D_GLOBAL_GATE,
) -> list[dict[str, Any]]:
    """Filter cross-ID pairs using same-track provenance and joint geometry."""

    result = []
    for source in candidate_rows:
        entity_id = str(source["entity_id"])
        node_id = str(source["node_id"])
        if entity_id in exact_entity_ids or node_id in exact_node_ids:
            continue
        if bool(source["semantic_id_match"]) or bool(
            source.get("rejected_reserved_owner")
        ):
            continue
        tracks = shared_track_ids(
            int(source["entity_ordinal"]),
            int(source["candidate_semantic_id"]),
            tracks_by_ordinal,
        )
        volume_ratio = symmetric_volume_ratio(
            source["entity_dimensions_m"],
            source["node_dimensions_m"],
        )
        gate_checks = {
            "shared_upstream_track": bool(tracks),
            "center_distance": (
                float(source["center_distance_m"])
                <= gate.maximum_center_distance_m
            ),
            "aabb_gap": (
                float(source["aabb_gap_m"]) <= gate.maximum_aabb_gap_m
            ),
            "aabb_iou": float(source["aabb_iou"]) >= gate.minimum_aabb_iou,
            "volume_ratio": volume_ratio <= gate.maximum_symmetric_volume_ratio,
        }
        if not all(gate_checks.values()):
            continue
        normalized_cost = (
            float(source["center_distance_m"]) / gate.maximum_center_distance_m
            + float(source["aabb_gap_m"])
            / max(gate.maximum_aabb_gap_m, 1.0e-12)
            + (1.0 - float(source["aabb_iou"]))
            + math.log(volume_ratio)
            / math.log(gate.maximum_symmetric_volume_ratio)
        )
        result.append(
            {
                **dict(source),
                "schema": "daaam.g1_e17_v2_global_cross_id_candidate.v1",
                "shared_track_ids": tracks,
                "symmetric_volume_ratio": volume_ratio,
                "joint_gate_checks": gate_checks,
                "global_assignment_cost": normalized_cost,
            }
        )
    return result


def global_one_to_one_assignment(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Minimize total valid cost while maximizing one-to-one assignments."""

    if not candidates:
        return []
    entity_ids = sorted({str(row["entity_id"]) for row in candidates})
    node_ids = sorted({str(row["node_id"]) for row in candidates})
    entity_index = {value: index for index, value in enumerate(entity_ids)}
    node_index = {value: index for index, value in enumerate(node_ids)}
    # One private dummy column per entity guarantees a feasible unmatched option.
    invalid_cost = 1.0e6
    unmatched_cost = 10.0
    matrix = np.full(
        (len(entity_ids), len(node_ids) + len(entity_ids)),
        invalid_cost,
        dtype=np.float64,
    )
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in candidates:
        entity_id = str(row["entity_id"])
        node_id = str(row["node_id"])
        key = (entity_id, node_id)
        existing = lookup.get(key)
        if existing is None or float(row["global_assignment_cost"]) < float(
            existing["global_assignment_cost"]
        ):
            lookup[key] = row
            matrix[entity_index[entity_id], node_index[node_id]] = float(
                row["global_assignment_cost"]
            )
    for index in range(len(entity_ids)):
        matrix[index, len(node_ids) + index] = unmatched_cost
    rows, columns = linear_sum_assignment(matrix)
    assignments = []
    for row_index, column_index in zip(rows, columns):
        if column_index >= len(node_ids):
            continue
        entity_id = entity_ids[row_index]
        node_id = node_ids[column_index]
        source = lookup.get((entity_id, node_id))
        if source is None or matrix[row_index, column_index] >= unmatched_cost:
            continue
        assignments.append(
            {
                **dict(source),
                "schema": "daaam.g1_e17_v2_global_assignment.v1",
                "assignment_method": "scipy_linear_sum_assignment",
            }
        )
    return sorted(
        assignments,
        key=lambda row: (
            int(row["entity_ordinal"]),
            int(row["candidate_semantic_id"]),
            str(row["node_id"]),
        ),
    )


def current_spatial_pending_candidates(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    exact_entity_ids: set[str],
    exact_node_ids: set[str],
) -> list[dict[str, Any]]:
    """Return cross-ID candidates the current center-OR-gap policy would apply."""

    return [
        {
            **dict(row),
            "schema": "daaam.g1_e17_v2_pending_spatial_candidate.v1",
            "pending_reason": "cross_semantic_id_requires_review",
        }
        for row in candidate_rows
        if str(row["entity_id"]) not in exact_entity_ids
        and str(row["node_id"]) not in exact_node_ids
        and not bool(row["semantic_id_match"])
        and bool(row["accepted"])
        and not bool(row.get("rejected_reserved_owner"))
    ]
