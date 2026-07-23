#!/usr/bin/env python3
"""Validate query-map geometry against an authoritative lidar point cloud."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


def normalized_description(value: Any) -> str:
    return " ".join(str(value or "").split())


def is_object_node(node: dict[str, Any]) -> bool:
    identifier = int(node.get("id", -1))
    return int(node.get("layer", -1)) == 2 and identifier >> 56 == ord("O")


def validate_mesh_records(
    graph: dict[str, Any],
    tree: cKDTree,
    *,
    maximum_median_distance_m: float,
    maximum_p90_distance_m: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    records: list[dict[str, Any]] = []
    all_distances: list[np.ndarray] = []
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict) or not is_object_node(node):
            continue
        attributes = node.get("attributes") or {}
        metadata = attributes.get("metadata") or {}
        description = normalized_description(metadata.get("description"))
        points = np.asarray((attributes.get("mesh") or {}).get("points") or [], dtype=float)
        position = np.asarray(attributes.get("position") or [], dtype=float)
        if not description or points.ndim != 2 or points.shape[1:] != (3,):
            continue
        if position.shape != (3,):
            continue
        distances = np.asarray(tree.query(points + position, workers=-1)[0], dtype=float)
        median = quantile(distances, 0.5)
        p90 = quantile(distances, 0.9)
        passed = median <= maximum_median_distance_m and p90 <= maximum_p90_distance_m
        records.append(
            {
                "record_id": f"O({int(node['id']) & ((1 << 56) - 1)})",
                "entity_id": str(metadata.get("entity_id") or ""),
                "description": description,
                "position_m": position.tolist(),
                "mesh_vertices": int(len(distances)),
                "nearest_lidar_distance_m": {
                    "minimum": float(np.min(distances)),
                    "median": median,
                    "p90": p90,
                    "maximum": float(np.max(distances)),
                },
                "mesh_vertex_fraction_within": {
                    "0.10_m": float(np.mean(distances <= 0.10)),
                    "0.20_m": float(np.mean(distances <= 0.20)),
                    "0.30_m": float(np.mean(distances <= 0.30)),
                    "0.50_m": float(np.mean(distances <= 0.50)),
                },
                "passed": bool(passed),
            }
        )
        all_distances.append(distances)
    combined = np.concatenate(all_distances) if all_distances else np.empty(0, dtype=float)
    return records, combined


def aabb_gaps(points: np.ndarray, center: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    outside = np.maximum(np.abs(points - center) - dimensions / 2.0, 0.0)
    return np.linalg.norm(outside, axis=1)


def validate_spatial_records(
    semantic_index: dict[str, Any],
    lidar_points: np.ndarray,
    tree: cKDTree,
    *,
    maximum_aabb_gap_m: float,
    evidence_by_node: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in semantic_index.get("records") or []:
        if not isinstance(item, dict) or item.get("geometry_status") != "spatial_only":
            continue
        record_identifier = str(item.get("record_id") or "")
        evidence = (evidence_by_node or {}).get(record_identifier) or {}
        evidence_center = evidence.get("geometry_position_m")
        evidence_dimensions = evidence.get("geometry_dimensions_m")
        use_evidence = evidence_center is not None and evidence_dimensions is not None
        center = np.asarray(
            evidence_center if use_evidence else item.get("position_m") or [], dtype=float
        )
        dimensions = np.asarray(
            evidence_dimensions if use_evidence else item.get("dimensions_m") or [],
            dtype=float,
        )
        if center.shape != (3,) or dimensions.shape != (3,) or np.any(dimensions < 0.0):
            continue
        radius = float(np.linalg.norm(dimensions / 2.0) + max(1.0, maximum_aabb_gap_m))
        indices = tree.query_ball_point(center, radius)
        if indices:
            candidates = lidar_points[np.asarray(indices, dtype=int)]
        else:
            _, nearest = tree.query(center, k=min(256, len(lidar_points)))
            candidates = lidar_points[np.atleast_1d(nearest).astype(int)]
        gaps = aabb_gaps(candidates, center, dimensions)
        minimum_gap = float(np.min(gaps))
        records.append(
            {
                "record_id": record_identifier,
                "entity_id": str(item.get("entity_id") or ""),
                "description": normalized_description(item.get("description")),
                "position_m": center.tolist(),
                "dimensions_m": dimensions.tolist(),
                "geometry_source": (
                    "query_evidence_rgbd" if use_evidence else "semantic_index"
                ),
                "minimum_lidar_to_aabb_gap_m": minimum_gap,
                "lidar_points_inside_or_within": {
                    "0.10_m": int(np.count_nonzero(gaps <= 0.10)),
                    "0.20_m": int(np.count_nonzero(gaps <= 0.20)),
                },
                "passed": bool(minimum_gap <= maximum_aabb_gap_m),
            }
        )
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(bool(item.get("passed")) for item in records)
    return {
        "records": len(records),
        "passed": passed,
        "failed": len(records) - passed,
        "pass_ratio": None if not records else passed / len(records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsg", type=Path, required=True)
    parser.add_argument("--lidar-map", type=Path, required=True)
    parser.add_argument("--semantic-index", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-median-distance-m", type=float, default=0.30)
    parser.add_argument("--maximum-p90-distance-m", type=float, default=0.50)
    parser.add_argument("--maximum-aabb-gap-m", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dsg_path = args.dsg.expanduser().resolve()
    lidar_path = args.lidar_map.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    semantic_path = (
        None if args.semantic_index is None else args.semantic_index.expanduser().resolve()
    )
    evidence_path = None if args.evidence is None else args.evidence.expanduser().resolve()

    graph = json.loads(dsg_path.read_text(encoding="utf-8"))
    cloud = o3d.io.read_point_cloud(str(lidar_path))
    lidar_points = np.asarray(cloud.points, dtype=float)
    if lidar_points.ndim != 2 or lidar_points.shape[1:] != (3,) or not len(lidar_points):
        raise ValueError(f"Lidar map has no XYZ points: {lidar_path}")
    tree = cKDTree(lidar_points)
    mesh_records, mesh_distances = validate_mesh_records(
        graph,
        tree,
        maximum_median_distance_m=args.maximum_median_distance_m,
        maximum_p90_distance_m=args.maximum_p90_distance_m,
    )
    semantic_index = None
    evidence_manifest = None
    evidence_by_node: dict[str, dict[str, Any]] = {}
    if evidence_path is not None:
        evidence_manifest = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence_by_node = {
            str(item.get("node_id") or ""): item
            for item in evidence_manifest.get("objects") or []
            if isinstance(item, dict) and item.get("node_id")
        }
    spatial_records: list[dict[str, Any]] = []
    if semantic_path is not None:
        semantic_index = json.loads(semantic_path.read_text(encoding="utf-8"))
        spatial_records = validate_spatial_records(
            semantic_index,
            lidar_points,
            tree,
            maximum_aabb_gap_m=args.maximum_aabb_gap_m,
            evidence_by_node=evidence_by_node,
        )

    report = {
        "schema": "daaam.dsg_lidar_coordinate_validation.v1",
        "coordinate_frame": "lidar_map",
        "sources": {
            "dsg": str(dsg_path),
            "dsg_sha256": sha256(dsg_path),
            "lidar_map": str(lidar_path),
            "lidar_map_sha256": sha256(lidar_path),
            "semantic_index": None if semantic_path is None else str(semantic_path),
            "semantic_index_sha256": (
                None if semantic_path is None else sha256(semantic_path)
            ),
            "evidence": None if evidence_path is None else str(evidence_path),
            "evidence_sha256": (
                None if evidence_path is None else sha256(evidence_path)
            ),
        },
        "lidar_points": int(len(lidar_points)),
        "thresholds": {
            "maximum_mesh_median_distance_m": args.maximum_median_distance_m,
            "maximum_mesh_p90_distance_m": args.maximum_p90_distance_m,
            "maximum_spatial_aabb_gap_m": args.maximum_aabb_gap_m,
        },
        "summary": {
            "mesh_bound": summarize(mesh_records),
            "spatial_only": summarize(spatial_records),
            "mesh_vertex_weighted": (
                None
                if not len(mesh_distances)
                else {
                    "vertices": int(len(mesh_distances)),
                    "median_distance_m": quantile(mesh_distances, 0.5),
                    "p90_distance_m": quantile(mesh_distances, 0.9),
                    "fraction_within_0.20_m": float(np.mean(mesh_distances <= 0.20)),
                    "fraction_within_0.30_m": float(np.mean(mesh_distances <= 0.30)),
                }
            ),
        },
        "mesh_bound_records": mesh_records,
        "spatial_only_records": spatial_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
