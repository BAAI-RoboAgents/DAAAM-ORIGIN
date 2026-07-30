"""Auditable input-support and extraction-decision diagnostics for E16."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


_EXTRACTION_PATTERN = re.compile(
    r"\[MeshObjectExtractor\] "
    r"(?P<action>Dropping|Extracted) track (?P<label>\d+) "
    r"\([^)]*\)(?:: )?(?P<detail>.*)"
)


def track_confidence(observations: int, configured_minimum: int) -> float:
    """Reproduce Khronos ExternalTracker's observation confidence."""

    if configured_minimum <= 0:
        raise ValueError("configured_minimum must be positive")
    return min(float(observations) / float(configured_minimum * 2), 1.0)


def required_observations_for_allocation(
    configured_minimum: int,
    allocation_confidence: float = 0.5,
) -> int:
    """Return the first observation count accepted by MeshObjectExtractor.

    Khronos drops tracks when ``confidence <= allocation_confidence``. With the
    default confidence formula and threshold 0.5, ``min_num_observations=8``
    therefore requires 9 observations, not 8.
    """

    if configured_minimum <= 0:
        raise ValueError("configured_minimum must be positive")
    if not 0.0 <= allocation_confidence < 1.0:
        raise ValueError("allocation_confidence must be in [0, 1)")
    count = 0
    while track_confidence(count, configured_minimum) <= allocation_confidence:
        count += 1
    return count


def _maximum_consecutive(indices: Sequence[int]) -> int:
    if not indices:
        return 0
    longest = current = 1
    for left, right in zip(indices, indices[1:]):
        if right == left + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def _distribution(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"minimum": None, "p50": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": int(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "maximum": int(np.max(array)),
    }


def _update_bounds(
    accumulators: dict[int, dict[str, Any]],
    labels: np.ndarray,
    points: np.ndarray,
) -> None:
    if not len(labels):
        return
    maximum_label = int(np.max(labels))
    minimum = np.full((maximum_label + 1, 3), np.inf, dtype=np.float64)
    maximum = np.full((maximum_label + 1, 3), -np.inf, dtype=np.float64)
    np.minimum.at(minimum, labels, points)
    np.maximum.at(maximum, labels, points)
    for label in np.unique(labels):
        value = accumulators[int(label)]
        value["map_min"] = np.minimum(value["map_min"], minimum[int(label)])
        value["map_max"] = np.maximum(value["map_max"], maximum[int(label)])


def summarize_semantic_support(
    *,
    label_paths: Sequence[Path],
    depth_paths: Sequence[Path],
    frames: Sequence[Mapping[str, Any]],
    maximum_range_m: float,
    minimum_cluster_pixels: int = 20,
    observation_thresholds: Iterable[int] = (4, 6, 8),
) -> list[dict[str, Any]]:
    """Summarize the exact label/depth support presented to InstanceForwarding."""

    if len(label_paths) != len(depth_paths) or len(label_paths) != len(frames):
        raise ValueError("label/depth/frame lengths differ")
    if maximum_range_m <= 0.0:
        raise ValueError("maximum_range_m must be positive")
    if minimum_cluster_pixels <= 0:
        raise ValueError("minimum_cluster_pixels must be positive")

    accumulators: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "visible_frames": [],
            "cluster_frames": [],
            "pixels_by_frame": [],
            "in_range_pixels_by_frame": [],
            "total_pixels": 0,
            "total_in_range_pixels": 0,
            "map_min": np.full(3, np.inf, dtype=np.float64),
            "map_max": np.full(3, -np.inf, dtype=np.float64),
        }
    )

    for frame_position, (label_path, depth_path, frame) in enumerate(
        zip(label_paths, depth_paths, frames)
    ):
        labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if labels is None:
            raise FileNotFoundError(label_path)
        if depth_raw is None:
            raise FileNotFoundError(depth_path)
        if labels.shape != depth_raw.shape:
            raise ValueError(f"label/depth shape mismatch: {frame_position}")

        positive_labels = labels[labels > 0].astype(np.int64, copy=False)
        if len(positive_labels):
            unique, counts = np.unique(positive_labels, return_counts=True)
            for label, count in zip(unique, counts):
                value = accumulators[int(label)]
                value["visible_frames"].append(frame_position)
                value["pixels_by_frame"].append(int(count))
                value["total_pixels"] += int(count)

        depth_m = depth_raw.astype(np.float32) / 1000.0
        intrinsics = np.asarray(frame["intrinsics"], dtype=np.float64)
        world_T_camera = np.asarray(
            frame["world_T_camera"], dtype=np.float64
        )
        if intrinsics.shape != (3, 3) or world_T_camera.shape != (4, 4):
            raise ValueError(f"invalid camera geometry: {frame_position}")

        valid = (labels > 0) & np.isfinite(depth_m) & (depth_m > 0.0)
        rows, columns = np.nonzero(valid)
        if not len(rows):
            continue
        z = depth_m[rows, columns].astype(np.float64, copy=False)
        x = (columns - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (rows - intrinsics[1, 2]) * z / intrinsics[1, 1]
        camera_points = np.column_stack((x, y, z))
        ranges = np.linalg.norm(camera_points, axis=1)
        in_range = ranges <= maximum_range_m
        if not np.any(in_range):
            continue
        selected_rows = rows[in_range]
        selected_columns = columns[in_range]
        selected_labels = labels[
            selected_rows, selected_columns
        ].astype(np.int64, copy=False)
        selected_camera = camera_points[in_range]
        world_points = (
            selected_camera @ world_T_camera[:3, :3].T
            + world_T_camera[:3, 3]
        )

        unique, counts = np.unique(selected_labels, return_counts=True)
        for label, count in zip(unique, counts):
            value = accumulators[int(label)]
            value["in_range_pixels_by_frame"].append(int(count))
            value["total_in_range_pixels"] += int(count)
            if int(count) >= minimum_cluster_pixels:
                value["cluster_frames"].append(frame_position)
        _update_bounds(accumulators, selected_labels, world_points)

    thresholds = tuple(int(value) for value in observation_thresholds)
    rows = []
    for label, value in sorted(accumulators.items()):
        map_min = value["map_min"]
        map_max = value["map_max"]
        has_geometry = bool(np.all(np.isfinite(map_min)))
        dimensions = map_max - map_min if has_geometry else None
        cluster_observations = len(value["cluster_frames"])
        allocation = {}
        for threshold in thresholds:
            confidence = track_confidence(cluster_observations, threshold)
            required = required_observations_for_allocation(threshold)
            allocation[str(threshold)] = {
                "configured_minimum_observations": threshold,
                "required_observations_due_to_strict_confidence_gate": required,
                "predicted_track_confidence": confidence,
                "passes_allocation_confidence_strictly_above_0p5": (
                    confidence > 0.5
                ),
            }
        rows.append(
            {
                "schema": "daaam.g1_e16_semantic_support.v1",
                "semantic_label": label,
                "maximum_object_range_m": maximum_range_m,
                "minimum_cluster_pixels": minimum_cluster_pixels,
                "visible_frame_count": len(value["visible_frames"]),
                "visible_frame_indices": value["visible_frames"],
                "cluster_observation_count": cluster_observations,
                "cluster_frame_indices": value["cluster_frames"],
                "maximum_consecutive_cluster_observations": _maximum_consecutive(
                    value["cluster_frames"]
                ),
                "total_label_pixels": value["total_pixels"],
                "total_in_range_depth_pixels": value["total_in_range_pixels"],
                "in_range_depth_fraction": (
                    value["total_in_range_pixels"] / value["total_pixels"]
                    if value["total_pixels"]
                    else 0.0
                ),
                "label_pixels_per_visible_frame": _distribution(
                    value["pixels_by_frame"]
                ),
                "in_range_pixels_per_supported_frame": _distribution(
                    value["in_range_pixels_by_frame"]
                ),
                "map_aabb_min_m": map_min.tolist() if has_geometry else None,
                "map_aabb_max_m": map_max.tolist() if has_geometry else None,
                "map_aabb_dimensions_m": (
                    dimensions.tolist() if dimensions is not None else None
                ),
                "map_aabb_volume_m3": (
                    float(np.prod(dimensions))
                    if dimensions is not None
                    else None
                ),
                "allocation_gate_by_minimum_observations": allocation,
                "diagnostic_boundary": (
                    "pre-extraction support reconstructed from frozen label/depth/"
                    "pose inputs; final mesh confidence and connectivity require "
                    "the extractor decision log"
                ),
            }
        )
    return rows


def parse_mesh_extraction_decisions(
    log_text: str,
) -> list[dict[str, Any]]:
    """Parse MeshObjectExtractor's label-level terminal decisions."""

    rows = []
    for line_number, line in enumerate(log_text.splitlines(), start=1):
        match = _EXTRACTION_PATTERN.search(line)
        if not match:
            continue
        action = match.group("action")
        detail = match.group("detail").strip()
        rows.append(
            {
                "schema": "daaam.g1_e16_mesh_extraction_decision.v1",
                "semantic_label": int(match.group("label")),
                "decision": "extracted" if action == "Extracted" else "dropped",
                "detail": detail,
                "line_number": line_number,
                "raw_line": line.strip(),
            }
        )
    return rows


def read_mesh_extraction_decisions(
    paths: Sequence[Path],
) -> list[dict[str, Any]]:
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in paths
        if path.is_file()
    )
    return parse_mesh_extraction_decisions(text)


def object_aabb_volume(node: Mapping[str, Any]) -> float:
    """Return an exported object-node AABB volume."""

    dimensions = json.loads(str(node["dimensions_json"]))
    if len(dimensions) != 3:
        raise ValueError("object dimensions must contain three values")
    return float(np.prod(np.asarray(dimensions, dtype=np.float64)))


def choose_adaptive_object_source(
    near_node: Mapping[str, Any] | None,
    far_node: Mapping[str, Any] | None,
    *,
    compact_volume_ratio: float = 0.25,
    minimum_mesh_point_ratio: float = 0.5,
) -> tuple[str, str]:
    """Choose a near/far reconstruction without rewarding volume inflation."""

    if near_node is None and far_node is None:
        raise ValueError("at least one object node is required")
    if near_node is None:
        return "far", "far_only_recovery"
    if far_node is None:
        return "near", "near_only_preservation"
    near_volume = object_aabb_volume(near_node)
    far_volume = object_aabb_volume(far_node)
    near_points = int(near_node["mesh_points"])
    far_points = int(far_node["mesh_points"])
    if (
        near_volume > 0.0
        and far_volume / near_volume <= compact_volume_ratio
        and far_points >= near_points * minimum_mesh_point_ratio
    ):
        return "far", "far_materially_more_compact_with_mesh_support"
    return "near", "near_preferred_to_limit_far_range_contamination"


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
