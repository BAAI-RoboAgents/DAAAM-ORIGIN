#!/usr/bin/env python3
"""Build a deterministic near-preferred E16 object graph with far recovery."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.experiments.e16_support import (  # noqa: E402
    choose_adaptive_object_source,
    object_aabb_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--near-variant", type=Path, required=True)
    parser.add_argument("--far-variant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-volume-ratio", type=float, default=0.25)
    parser.add_argument("--minimum-mesh-point-ratio", type=float, default=0.5)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def object_graph_nodes(graph: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {}
    for node in graph.get("nodes", []):
        if node.get("layer") != 2 or node.get("partition", 0) != 0:
            continue
        label = int((node.get("attributes") or {}).get("semantic_label", -1))
        if label > 0:
            if label in rows:
                raise ValueError(f"duplicate object semantic label: {label}")
            rows[label] = node
    return rows


def exported_rows(path: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["semantic_label"]): row
        for row in read_jsonl(path / "metrics/object_nodes.jsonl")
    }


def main() -> int:
    args = parse_args()
    near = args.near_variant.resolve()
    far = args.far_variant.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    near_graph_path = near / "hydra_realtime/backend/dsg.json"
    far_graph_path = far / "hydra_realtime/backend/dsg.json"
    near_graph = read_json(near_graph_path)
    far_graph = read_json(far_graph_path)
    near_graph_nodes = object_graph_nodes(near_graph)
    far_graph_nodes = object_graph_nodes(far_graph)
    near_exports = exported_rows(near)
    far_exports = exported_rows(far)
    if set(near_graph_nodes) != set(near_exports):
        raise ValueError("near DSG/export semantic labels differ")
    if set(far_graph_nodes) != set(far_exports):
        raise ValueError("far DSG/export semantic labels differ")

    selected_nodes = {}
    selection_rows = []
    for label in sorted(set(near_exports) | set(far_exports)):
        near_export = near_exports.get(label)
        far_export = far_exports.get(label)
        source, reason = choose_adaptive_object_source(
            near_export,
            far_export,
            compact_volume_ratio=args.compact_volume_ratio,
            minimum_mesh_point_ratio=args.minimum_mesh_point_ratio,
        )
        selected_graph_node = deepcopy(
            near_graph_nodes[label]
            if source == "near"
            else far_graph_nodes[label]
        )
        selected_export = (
            near_export if source == "near" else far_export
        )
        selected_nodes[label] = selected_graph_node
        selection_rows.append(
            {
                "schema": "daaam.g1_e16_adaptive_object_selection.v1",
                "semantic_label": label,
                "selected_source": source,
                "selection_reason": reason,
                "near_present": near_export is not None,
                "far_present": far_export is not None,
                "near_aabb_volume_m3": (
                    object_aabb_volume(near_export)
                    if near_export is not None
                    else None
                ),
                "far_aabb_volume_m3": (
                    object_aabb_volume(far_export)
                    if far_export is not None
                    else None
                ),
                "near_mesh_points": (
                    int(near_export["mesh_points"])
                    if near_export is not None
                    else None
                ),
                "far_mesh_points": (
                    int(far_export["mesh_points"])
                    if far_export is not None
                    else None
                ),
                "selected_mesh_points": int(selected_export["mesh_points"]),
                "selected_dimensions_json": selected_export["dimensions_json"],
            }
        )

    graph = deepcopy(near_graph)
    non_object_nodes = [
        node
        for node in graph.get("nodes", [])
        if not (node.get("layer") == 2 and node.get("partition", 0) == 0)
    ]
    used_ids = {int(node["id"]) for node in non_object_nodes}
    next_id = max(
        int(node["id"])
        for node in near_graph.get("nodes", [])
        if node.get("layer") == 2 and node.get("partition", 0) == 0
    ) + 1
    normalized_objects = []
    for label, node in sorted(selected_nodes.items()):
        if label in near_graph_nodes:
            node["id"] = int(near_graph_nodes[label]["id"])
        else:
            while next_id in used_ids:
                next_id += 1
            node["id"] = next_id
            next_id += 1
        used_ids.add(int(node["id"]))
        normalized_objects.append(node)
    graph["nodes"] = non_object_nodes + normalized_objects

    graph_edges = []
    valid_ids = {int(node["id"]) for node in graph["nodes"]}
    for edge in graph.get("edges", []):
        source = int(edge.get("source", -1))
        target = int(edge.get("target", -1))
        if source in valid_ids and target in valid_ids:
            graph_edges.append(edge)
    graph["edges"] = graph_edges

    backend = output / "hydra_realtime/backend"
    shutil.copytree(near / "hydra_realtime/backend", backend)
    write_json(backend / "dsg.json", graph)
    write_jsonl(output / "selection_manifest.jsonl", selection_rows)
    write_csv(output / "selection_manifest.csv", selection_rows)
    write_json(
        output / "SUMMARY.json",
        {
            "schema": "daaam.g1_e16_adaptive_object_graph_summary.v1",
            "near_variant": str(near),
            "far_variant": str(far),
            "near_dsg_sha256": sha256_file(near_graph_path),
            "far_dsg_sha256": sha256_file(far_graph_path),
            "compact_volume_ratio": args.compact_volume_ratio,
            "minimum_mesh_point_ratio": args.minimum_mesh_point_ratio,
            "near_object_count": len(near_exports),
            "far_object_count": len(far_exports),
            "adaptive_object_count": len(selected_nodes),
            "near_selected": sum(
                row["selected_source"] == "near" for row in selection_rows
            ),
            "far_selected": sum(
                row["selected_source"] == "far" for row in selection_rows
            ),
            "far_only_recoveries": sum(
                row["selection_reason"] == "far_only_recovery"
                for row in selection_rows
            ),
            "formal_object_recall": None,
            "status": "derived_candidate_pending_target_review",
        },
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
