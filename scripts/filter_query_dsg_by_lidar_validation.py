#!/usr/bin/env python3
"""Remove lidar-coordinate outliers from a checksum-bound DSG query index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DESCRIPTION_FIELD = "description"
EMBEDDING_FIELD = "sentence_embedding_feature"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any], *, compact: bool = False) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if compact:
        payload = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    else:
        payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def record_id(node: dict[str, Any]) -> str:
    return f"O({int(node['id']) & ((1 << 56) - 1)})"


def is_object_node(node: dict[str, Any]) -> bool:
    identifier = int(node.get("id", -1))
    return int(node.get("layer", -1)) == 2 and identifier >> 56 == ord("O")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsg", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--semantic-index", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--validation", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dsg_path = args.dsg.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    semantic_path = args.semantic_index.expanduser().resolve()
    validation_path = args.validation.expanduser().resolve()
    evidence_path = None if args.evidence is None else args.evidence.expanduser().resolve()

    graph = read_json(dsg_path)
    manifest = read_json(manifest_path)
    semantic_index = read_json(semantic_path)
    validation = read_json(validation_path)
    evidence_manifest = None if evidence_path is None else read_json(evidence_path)
    if validation.get("schema") != "daaam.dsg_lidar_coordinate_validation.v1":
        raise ValueError("Unsupported lidar coordinate validation report")
    sources = validation.get("sources") or {}
    if str(sources.get("dsg_sha256") or "") != sha256(dsg_path):
        raise ValueError("Validation report does not describe --dsg")
    if str(sources.get("semantic_index_sha256") or "") != sha256(semantic_path):
        raise ValueError("Validation report does not describe --semantic-index")
    if evidence_path is not None and str(sources.get("evidence_sha256") or "") != sha256(
        evidence_path
    ):
        raise ValueError("Validation report does not describe --evidence")

    mesh_results = {
        str(item["record_id"]): item
        for item in validation.get("mesh_bound_records") or []
        if isinstance(item, dict) and item.get("record_id")
    }
    mesh_passed = 0
    mesh_failed = 0
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict) or not is_object_node(node):
            continue
        result = mesh_results.get(record_id(node))
        if result is None:
            continue
        attributes = node.get("attributes") or {}
        metadata = attributes.get("metadata") or {}
        attributes["metadata"] = metadata
        metadata["lidar_coordinate_validation"] = {
            "passed": bool(result.get("passed")),
            "median_distance_m": result["nearest_lidar_distance_m"]["median"],
            "p90_distance_m": result["nearest_lidar_distance_m"]["p90"],
        }
        if result.get("passed"):
            mesh_passed += 1
            p90 = float(result["nearest_lidar_distance_m"]["p90"])
            threshold = float(
                validation["thresholds"]["maximum_mesh_p90_distance_m"]
            )
            metadata["geometry_confidence"] = max(0.0, 1.0 - p90 / threshold)
        else:
            mesh_failed += 1
            metadata.pop(DESCRIPTION_FIELD, None)
            metadata.pop(EMBEDDING_FIELD, None)
            metadata["geometry_confidence"] = 0.0
            metadata["query_exclusion_reason"] = "lidar_coordinate_validation_failed"
    expected_mesh = validation.get("summary", {}).get("mesh_bound", {})
    if mesh_passed != int(expected_mesh.get("passed", -1)) or mesh_failed != int(
        expected_mesh.get("failed", -1)
    ):
        raise ValueError("DSG object nodes do not match the lidar validation report")

    spatial_results = {
        str(item["record_id"]): item
        for item in validation.get("spatial_only_records") or []
        if isinstance(item, dict) and item.get("record_id")
    }
    kept_records: list[dict[str, Any]] = []
    threshold = float(validation["thresholds"]["maximum_spatial_aabb_gap_m"])
    for item in semantic_index.get("records") or []:
        result = spatial_results.get(str(item.get("record_id") or ""))
        if result is None or not result.get("passed"):
            continue
        updated = dict(item)
        gap = float(result["minimum_lidar_to_aabb_gap_m"])
        updated["geometry_confidence"] = max(0.0, 1.0 - gap / threshold)
        updated["lidar_coordinate_validation"] = {
            "passed": True,
            "minimum_aabb_gap_m": gap,
        }
        kept_records.append(updated)

    write_json(dsg_path, graph, compact=True)
    dsg_digest = sha256(dsg_path)
    semantic_index["dsg_file"] = dsg_path.name
    semantic_index["dsg_sha256"] = dsg_digest
    semantic_index["records"] = kept_records
    semantic_index["record_count"] = len(kept_records)
    semantic_index["geometry_counts"] = {
        "image_only": sum(item.get("geometry_status") == "image_only" for item in kept_records),
        "spatial_only": sum(
            item.get("geometry_status") == "spatial_only" for item in kept_records
        ),
    }
    write_json(semantic_path, semantic_index)
    semantic_digest = sha256(semantic_path)

    if evidence_path is not None and evidence_manifest is not None:
        queryable_ids = {
            identifier
            for identifier, result in mesh_results.items()
            if result.get("passed")
        }
        queryable_ids.update(str(item.get("record_id") or "") for item in kept_records)
        evidence_objects = [
            item
            for item in evidence_manifest.get("objects") or []
            if isinstance(item, dict) and str(item.get("node_id") or "") in queryable_ids
        ]
        evidence_manifest["dsg_file"] = dsg_path.name
        evidence_manifest["dsg_sha256"] = dsg_digest
        evidence_manifest["objects"] = evidence_objects
        evidence_manifest["object_count"] = len(evidence_objects)
        evidence_manifest["queryable_object_count"] = len(queryable_ids)
        evidence_manifest["missing_node_ids"] = sorted(
            queryable_ids
            - {str(item.get("node_id") or "") for item in evidence_objects}
        )
        write_json(evidence_path, evidence_manifest)

    spatial_count = semantic_index["geometry_counts"]["spatial_only"]
    image_count = semantic_index["geometry_counts"]["image_only"]
    manifest["dsg_file"] = dsg_path.name
    manifest["dsg_sha256"] = dsg_digest
    manifest["dsg_queryable_objects"] = mesh_passed
    manifest["queryable_objects"] = mesh_passed + len(kept_records)
    manifest["geometry_counts"] = {
        "mesh_bound": mesh_passed,
        "spatial_only": spatial_count,
        "image_only": image_count,
    }
    manifest["semantic_index"] = {
        "schema": semantic_index["schema"],
        "file": semantic_path.name,
        "sha256": semantic_digest,
        "records": len(kept_records),
    }
    manifest["lidar_coordinate_validation"] = {
        "file": validation_path.name,
        "sha256": sha256(validation_path),
        "mesh_excluded": mesh_failed,
        "semantic_excluded": len(spatial_results) - len(kept_records),
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "queryable_objects": manifest["queryable_objects"],
                "mesh_bound": mesh_passed,
                "spatial_only": spatial_count,
                "mesh_excluded": mesh_failed,
                "spatial_excluded": len(spatial_results) - len(kept_records),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
