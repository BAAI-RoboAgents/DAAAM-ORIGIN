#!/usr/bin/env python3
"""Bind DAM semantics from one DSG onto real object meshes in another DSG."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move MapMemory semantics onto credible real Hydra object meshes and "
            "restore standard relative/absolute observation intervals."
        )
    )
    parser.add_argument(
        "--dsg",
        required=True,
        type=Path,
        help="Target candidate geometry DSG whose real object meshes receive semantics.",
    )
    parser.add_argument(
        "--semantic-source-dsg",
        type=Path,
        help=(
            "DSG that owns entity IDs, DAM descriptions, and embeddings. "
            "Defaults to --dsg for backward-compatible in-place-source migration."
        ),
    )
    parser.add_argument(
        "--semantic-source-report",
        type=Path,
        help=(
            "Baseline realtime_run_report.json. rejected_no_mesh entities are "
            "recovered from semantic_stats.dsg.rejection_audit and named from "
            "the same MapMemory database."
        ),
    )
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument(
        "--semantic-config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "labels_pseudo.yaml",
    )
    parser.add_argument(
        "--labelspace-colors",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "labels_pseudo.csv",
    )
    parser.add_argument("--maximum-center-distance-m", type=float, default=0.75)
    parser.add_argument("--maximum-aabb-gap-m", type=float, default=0.15)
    parser.add_argument(
        "--time-origin-ns",
        type=int,
        help="Dataset time-contract origin; otherwise inferred from run_manifest.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a previous migration output; the source DSG is never overwritten.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh_counts(graph: DynamicSceneGraph) -> dict[str, int]:
    from spark_dsg import DsgLayers

    main_vertices = main_faces = 0
    if graph.has_mesh():
        main_vertices = int(graph.mesh.num_vertices())
        main_faces = int(graph.mesh.num_faces())
    object_vertices = 0
    object_meshes = 0
    for node in graph.get_layer(DsgLayers.OBJECTS).nodes:
        mesh = node.attributes.mesh()
        vertices = 0 if mesh is None else int(mesh.num_vertices())
        object_vertices += vertices
        object_meshes += int(vertices > 0)
    return {
        "main_mesh_vertices": main_vertices,
        "main_mesh_faces": main_faces,
        "object_meshes": object_meshes,
        "object_mesh_vertices": object_vertices,
    }


def _semantic_records(graph: DynamicSceneGraph) -> list[dict]:
    from spark_dsg import DsgLayers

    records: dict[str, dict] = {}
    for node in graph.get_layer(DsgLayers.OBJECTS).nodes:
        metadata = dict(node.attributes.metadata.get() or {})
        entity_id = str(metadata.get("entity_id") or "").strip()
        if not entity_id:
            continue
        semantic_id = int(node.attributes.semantic_label)
        existing = records.get(entity_id)
        if existing is not None and existing["semantic_id"] != semantic_id:
            raise ValueError(f"entity {entity_id} has multiple semantic IDs")
        records[entity_id] = {
            "entity_id": entity_id,
            "semantic_id": semantic_id,
            "description": str(metadata.get("description") or "").strip(),
            "embedding": metadata.get("sentence_embedding_feature"),
            "selectframe_clip_feature": metadata.get("selectframe_clip_feature"),
            "record_sources": ["semantic_source_dsg"],
            "description_source": "semantic_source_dsg",
        }
    return sorted(records.values(), key=lambda record: record["semantic_id"])


def _report_rejection_records(report: dict) -> list[dict]:
    """Recover strict entity/semantic identities from a baseline rejection audit."""

    if not isinstance(report, dict):
        raise ValueError("semantic source report must be a JSON object")
    semantic_stats = report.get("semantic_stats")
    dsg = semantic_stats.get("dsg") if isinstance(semantic_stats, dict) else None
    if not isinstance(dsg, dict) or "rejection_audit" not in dsg:
        raise ValueError(
            "semantic source report is missing semantic_stats.dsg.rejection_audit"
        )
    audit = dsg["rejection_audit"]
    if not isinstance(audit, list):
        raise ValueError("semantic rejection audit must be a list")
    records: dict[str, dict] = {}
    for index, event in enumerate(audit):
        if not isinstance(event, dict):
            raise ValueError(f"semantic rejection audit entry {index} is invalid")
        if event.get("status") != "rejected_no_mesh":
            continue
        entity_id = str(event.get("entity_id") or "").strip()
        try:
            semantic_id = int(event.get("semantic_id"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"semantic rejection audit entry {index} has no semantic ID"
            ) from error
        if not entity_id or semantic_id <= 0:
            raise ValueError(
                f"semantic rejection audit entry {index} has invalid identity"
            )
        existing = records.get(entity_id)
        if existing is not None and existing["semantic_id"] != semantic_id:
            raise ValueError(
                f"rejection audit entity {entity_id} has multiple semantic IDs"
            )
        records[entity_id] = {
            "entity_id": entity_id,
            "semantic_id": semantic_id,
            "description": "",
            "embedding": None,
            "selectframe_clip_feature": None,
            "record_sources": ["semantic_source_report"],
            "description_source": "map_memory_canonical_name",
        }
    return sorted(
        records.values(), key=lambda record: (record["semantic_id"], record["entity_id"])
    )


def _same_optional_value(first: object, second: object) -> bool:
    return json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def _merge_semantic_records(*record_groups: list[dict]) -> list[dict]:
    """Merge semantic identities without silently resolving any conflict."""

    merged: dict[str, dict] = {}
    semantic_owners: dict[int, str] = {}
    for record in (record for group in record_groups for record in group):
        entity_id = str(record.get("entity_id") or "").strip()
        semantic_id = int(record.get("semantic_id"))
        if not entity_id or semantic_id <= 0:
            raise ValueError("semantic record identity is invalid")
        owner = semantic_owners.get(semantic_id)
        if owner is not None and owner != entity_id:
            raise ValueError(
                f"semantic ID {semantic_id} belongs to both {owner} and {entity_id}"
            )
        semantic_owners[semantic_id] = entity_id
        incoming = dict(record)
        incoming["entity_id"] = entity_id
        incoming["semantic_id"] = semantic_id
        incoming["record_sources"] = list(record.get("record_sources") or [])
        existing = merged.get(entity_id)
        if existing is None:
            merged[entity_id] = incoming
            continue
        if existing["semantic_id"] != semantic_id:
            raise ValueError(f"entity {entity_id} has conflicting semantic IDs")
        for key in ("description", "embedding", "selectframe_clip_feature"):
            old_value = existing.get(key)
            new_value = incoming.get(key)
            if old_value in (None, "", []):
                existing[key] = new_value
            elif new_value not in (None, "", []) and not _same_optional_value(
                old_value, new_value
            ):
                raise ValueError(f"entity {entity_id} has conflicting {key}")
        existing["record_sources"] = sorted(
            set(existing["record_sources"]) | set(incoming["record_sources"])
        )
        if "semantic_source_dsg" in existing["record_sources"]:
            existing["description_source"] = "semantic_source_dsg"
        else:
            existing["description_source"] = incoming.get("description_source")
    return sorted(
        merged.values(), key=lambda record: (record["semantic_id"], record["entity_id"])
    )


def _semantic_id_owners(records: list[dict]) -> dict[int, str]:
    """Build the complete immutable identity view used by every binding attempt."""

    owners: dict[int, str] = {}
    for record in records:
        semantic_id = int(record["semantic_id"])
        entity_id = str(record["entity_id"] or "").strip()
        if semantic_id <= 0 or not entity_id:
            raise ValueError("semantic record identity is invalid")
        owner = owners.get(semantic_id)
        if owner is not None and owner != entity_id:
            raise ValueError(
                f"semantic ID {semantic_id} belongs to both {owner} and {entity_id}"
            )
        owners[semantic_id] = entity_id
    return owners


def _copy_database(source: Path, target: Path) -> None:
    source_uri = f"file:{source.resolve()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source_connection:
        with sqlite3.connect(target) as target_connection:
            source_connection.backup(target_connection)


def _resolve_time_origin_ns(
    explicit: int | None,
    *source_paths: Path,
) -> tuple[int | None, str]:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("--time-origin-ns must be positive")
        return int(explicit), "explicit_cli"
    checked = set()
    for source_path in source_paths:
        for directory in (source_path.parent, *source_path.parents):
            manifest_path = directory / "run_manifest.json"
            if manifest_path in checked or not manifest_path.is_file():
                continue
            checked.add(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            origin = (manifest.get("time_contract") or {}).get("time_origin_ns")
            if origin is not None:
                return int(origin), f"run_manifest:{manifest_path}"
    return None, "map_memory_derived_first_entity_observation"


def _verify_entities(
    graph: DynamicSceneGraph,
    matched_entity_ids: set[str],
    rejected_entity_ids: set[str],
) -> dict:
    from spark_dsg import DsgLayers

    nodes_by_entity: dict[str, list] = {}
    described_unmeshed_nodes = []
    for node in graph.get_layer(DsgLayers.OBJECTS).nodes:
        metadata = dict(node.attributes.metadata.get() or {})
        mesh = node.attributes.mesh()
        has_mesh = bool(mesh is not None and mesh.num_vertices() > 0)
        if str(metadata.get("description") or "").strip() and not has_mesh:
            described_unmeshed_nodes.append(str(node.id))
        entity_id = str(metadata.get("entity_id") or "").strip()
        if entity_id:
            nodes_by_entity.setdefault(entity_id, []).append(node)

    missing = sorted(matched_entity_ids - set(nodes_by_entity))
    duplicates = sorted(
        entity_id for entity_id, nodes in nodes_by_entity.items() if len(nodes) != 1
    )
    rejected_still_materialized = sorted(rejected_entity_ids & set(nodes_by_entity))
    invalid = []
    described_mesh_entities = 0
    for entity_id, nodes in sorted(nodes_by_entity.items()):
        node = nodes[0]
        metadata = dict(node.attributes.metadata.get() or {})
        history = dict(metadata.get("temporal_history") or {})
        mesh = node.attributes.mesh()
        has_mesh = bool(mesh is not None and mesh.num_vertices() > 0)
        description = str(metadata.get("description") or "").strip()
        if description:
            described_mesh_entities += 1
            if not has_mesh:
                invalid.append(f"{entity_id}: described node has no real object mesh")
        if entity_id in matched_entity_ids:
            if metadata.get("mesh_binding_status") != "matched_real_mesh":
                invalid.append(f"{entity_id}: missing matched_real_mesh status")
            if not description:
                invalid.append(f"{entity_id}: matched node is missing description")
            if not has_mesh:
                invalid.append(f"{entity_id}: matched status without object mesh")
        if history.get("first_observed") is None or history.get("last_observed") is None:
            invalid.append(f"{entity_id}: missing relative observation interval")
        if (
            history.get("first_observed_ns") is None
            or history.get("last_observed_ns") is None
        ):
            invalid.append(f"{entity_id}: missing absolute observation interval")
    if (
        missing
        or duplicates
        or rejected_still_materialized
        or described_unmeshed_nodes
        or invalid
    ):
        raise RuntimeError(
            json.dumps(
                {
                    "missing": missing,
                    "duplicates": duplicates,
                    "rejected_still_materialized": rejected_still_materialized,
                    "described_unmeshed_nodes": described_unmeshed_nodes,
                    "invalid": invalid,
                },
                sort_keys=True,
            )
        )
    return {
        "matched_real_mesh": len(matched_entity_ids),
        "rejected_no_mesh": len(rejected_entity_ids),
        "described_mesh_entities": described_mesh_entities,
        "authoritative_ready": bool(matched_entity_ids),
    }


def migrate(args: argparse.Namespace) -> dict:
    from daaam.grounding.models import ObjectAnnotation
    from daaam.memory import MapMemory
    from daaam.scene_graph.services import ObjectBindingPolicy, SceneGraphService
    from spark_dsg import DynamicSceneGraph

    source = args.dsg.expanduser().resolve()
    semantic_source_arg = getattr(args, "semantic_source_dsg", None)
    semantic_source = (
        source
        if semantic_source_arg is None
        else semantic_source_arg.expanduser().resolve()
    )
    semantic_report_arg = getattr(args, "semantic_source_report", None)
    semantic_report = (
        None
        if semantic_report_arg is None
        else semantic_report_arg.expanduser().resolve()
    )
    memory_path = args.memory.expanduser().resolve()
    output = args.output.expanduser().resolve()
    audit_output = (
        args.audit_output.expanduser().resolve()
        if args.audit_output is not None
        else output.with_suffix(output.suffix + ".binding.json")
    )
    input_paths = [source, semantic_source, memory_path]
    if semantic_report is not None:
        input_paths.append(semantic_report)
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output in set(input_paths):
        raise ValueError(
            "migration output must differ from every geometry, semantic, and memory input"
        )
    if output.exists() and not args.force:
        raise FileExistsError(output)
    if audit_output.exists() and not args.force:
        raise FileExistsError(audit_output)
    reserved_outputs = {source, semantic_source, memory_path, output}
    if semantic_report is not None:
        reserved_outputs.add(semantic_report)
    if audit_output in reserved_outputs:
        raise ValueError("audit output must be a separate file")

    graph = DynamicSceneGraph.load(str(source))
    source_mesh_counts = _mesh_counts(graph)
    semantic_graph = (
        graph
        if semantic_source == source
        else DynamicSceneGraph.load(str(semantic_source))
    )
    dsg_records = _semantic_records(semantic_graph)
    report_records: list[dict] = []
    if semantic_report is not None:
        with semantic_report.open() as stream:
            report_records = _report_rejection_records(json.load(stream))
    records = _merge_semantic_records(dsg_records, report_records)
    semantic_id_owners = _semantic_id_owners(records)
    report_only_records = sum(
        record["record_sources"] == ["semantic_source_report"]
        for record in records
    )
    overlapping_records = sum(len(record["record_sources"]) > 1 for record in records)
    policy = ObjectBindingPolicy(
        maximum_center_distance_m=args.maximum_center_distance_m,
        maximum_aabb_gap_m=args.maximum_aabb_gap_m,
        # A rejected entity can emit one conflict for every existing real
        # object mesh before its terminal rejected_no_mesh event.  Size the
        # journal from the actual candidate count so those terminal events are
        # never evicted before the checksum-bound query index is generated.
        audit_capacity=max(
            1000,
            len(records) * (int(source_mesh_counts["object_meshes"]) + 4),
        ),
    )
    service = SceneGraphService(
        args.semantic_config.expanduser().resolve(),
        args.labelspace_colors.expanduser().resolve(),
        defer_dsg_processing=False,
        enable_background_objects=False,
        object_binding_policy=policy,
    )
    service.set_scene_graph(graph)
    time_origin_sources = [source, semantic_source, memory_path]
    if semantic_report is not None:
        time_origin_sources.append(semantic_report)
    contract_time_origin_ns, contract_time_origin_source = _resolve_time_origin_ns(
        args.time_origin_ns, *time_origin_sources
    )

    with tempfile.TemporaryDirectory(prefix="daaam-map-memory-") as temporary_dir:
        memory_copy = Path(temporary_dir) / "map_memory.sqlite3"
        _copy_database(memory_path, memory_copy)
        with MapMemory(memory_copy) as memory:
            if contract_time_origin_ns is not None:
                memory.set_time_origin_ns(contract_time_origin_ns)
            matched_entity_ids = set()
            rejected_entity_ids = set()
            for record in records:
                snapshot = memory.get_entity(record["entity_id"])
                canonical_name = " ".join(
                    str(snapshot.get("canonical_name") or "").split()
                ).strip()
                if record.get("description_source") == "map_memory_canonical_name":
                    if not canonical_name or canonical_name.casefold() == "unknown":
                        raise ValueError(
                            "rejection-audit entity has no DAM canonical name in "
                            f"MapMemory: {record['entity_id']}"
                        )
                    description = canonical_name
                else:
                    description = record["description"] or canonical_name
                history = snapshot["temporal_history"]
                sensor_time_ns = int(
                    history.get("last_observed_ns") or snapshot["updated_ns"]
                )
                ensured = service.ensure_object_node(
                    semantic_id=record["semantic_id"],
                    entity_id=record["entity_id"],
                    position_m=snapshot.get("position_m"),
                    dimensions_m=snapshot.get("dimensions_m"),
                    sensor_time_ns=sensor_time_ns,
                    temporal_history=history,
                    time_origin_ns=memory.time_origin_ns,
                    allow_unmeshed_fallback=False,
                    semantic_id_owners=semantic_id_owners,
                )
                if not ensured:
                    rejected_entity_ids.add(record["entity_id"])
                    continue
                matched_entity_ids.add(record["entity_id"])
                service.add_correction(
                    ObjectAnnotation(
                        semantic_id=record["semantic_id"],
                        entity_id=record["entity_id"],
                        semantic_label=description,
                        confidence=1.0,
                        embedding=record["embedding"],
                        selectframe_clip_feature=record["selectframe_clip_feature"],
                        timestamp=sensor_time_ns / 1.0e9,
                        sensor_time_ns=sensor_time_ns,
                    )
                )
            service.apply_corrections()
            time_origin_ns = memory.time_origin_ns

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(
        f".{output.stem}.{uuid.uuid4().hex}.tmp{output.suffix}"
    )
    try:
        graph.save(str(temporary_output), include_mesh=True)
        reloaded = DynamicSceneGraph.load(str(temporary_output))
        output_mesh_counts = _mesh_counts(reloaded)
        if source_mesh_counts != output_mesh_counts:
            raise RuntimeError("DSG migration changed real mesh vertex/face counts")
        verification = _verify_entities(
            reloaded,
            matched_entity_ids,
            rejected_entity_ids,
        )
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)

    audit = {
        "schema": "daaam.dsg_semantic_binding.v1",
        # Backward-compatible aliases: source_dsg has always meant the target
        # geometry whose mesh counts must remain invariant.
        "source_dsg": str(source),
        "source_dsg_sha256": _sha256(source),
        "target_dsg": str(source),
        "target_dsg_sha256": _sha256(source),
        "semantic_source_dsg": str(semantic_source),
        "semantic_source_dsg_sha256": _sha256(semantic_source),
        "semantic_source_is_target": semantic_source == source,
        "semantic_source_report": (
            None if semantic_report is None else str(semantic_report)
        ),
        "semantic_source_report_sha256": (
            None if semantic_report is None else _sha256(semantic_report)
        ),
        "source_memory": str(memory_path),
        "source_memory_sha256": _sha256(memory_path),
        "output_dsg": str(output),
        "output_dsg_sha256": _sha256(output),
        "time_origin_ns": time_origin_ns,
        "time_source": contract_time_origin_source,
        "observation_time_source": (
            "map_memory_entity_observations_with_legacy_versions"
        ),
        "entity_count": len(records),
        "record_sources": {
            "semantic_source_dsg_records": len(dsg_records),
            "semantic_source_report_rejected_no_mesh_records": len(report_records),
            "report_only_records": report_only_records,
            "overlapping_records": overlapping_records,
            "merged_records": len(records),
            "report_only_description_source": "map_memory_canonical_name",
        },
        "policy": {
            "maximum_center_distance_m": policy.maximum_center_distance_m,
            "maximum_aabb_gap_m": policy.maximum_aabb_gap_m,
            "acceptance": "center_distance OR AABB_gap",
            "conflict_policy": "never_rebind_a_node_owned_by_another_entity",
        },
        "mesh_counts": output_mesh_counts,
        "verification": verification,
        "events": service.object_binding_audit,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_audit = audit_output.with_name(
        f".{audit_output.stem}.{uuid.uuid4().hex}.tmp{audit_output.suffix}"
    )
    try:
        temporary_audit.write_text(
            json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        json.loads(temporary_audit.read_text())
        temporary_audit.replace(audit_output)
    finally:
        temporary_audit.unlink(missing_ok=True)
    audit["audit_output"] = str(audit_output)
    audit["audit_output_sha256"] = _sha256(audit_output)
    return audit


def main() -> int:
    args = parse_args()
    try:
        result = migrate(args)
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "error": str(error)},
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
