#!/usr/bin/env python3
"""Re-embed existing DSG object descriptions for multilingual local queries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import click
import numpy as np
import spark_dsg as sdsg
import torch

from daaam.semantic_query import DEFAULT_SENTENCE_MODEL
from daaam.query_index import QueryIndexError, write_query_index
from daaam.utils.embedding import SentenceEmbeddingHandler


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_list(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def attach_description_embeddings(
    graph: sdsg.DynamicSceneGraph,
    *,
    model_name: str,
    device: str,
    handler: Any = None,
) -> tuple[int, int]:
    """Encode every described object and return ``(updated, dimension)``."""
    described_nodes: list[tuple[Any, str]] = []
    for node in graph.get_layer(sdsg.DsgLayers.OBJECTS).nodes:
        metadata = dict(node.attributes.metadata.get() or {})
        description = str(metadata.get("description", "")).strip()
        if description:
            described_nodes.append((node, description))
    if not described_nodes:
        raise ValueError("The DSG contains no object metadata.description values")

    # DAM often produces repeated descriptions. Encode each unique string once,
    # then attach the corresponding normalized vector to every concrete object.
    unique_descriptions = list(dict.fromkeys(text for _, text in described_nodes))
    if handler is None:
        handler = SentenceEmbeddingHandler(model_name=model_name, device=device)
    encoded = handler.extract_text_embeddings(
        unique_descriptions, show_progress=True
    )
    vectors = {
        description: _as_list(vector)
        for description, vector in zip(unique_descriptions, encoded)
    }
    dimension = len(next(iter(vectors.values())))

    for node, description in described_nodes:
        metadata = dict(node.attributes.metadata.get() or {})
        metadata["sentence_embedding_feature"] = vectors[description]
        node.attributes.metadata.set(metadata)

    scene_metadata = dict(graph.metadata.get() or {})
    scene_metadata["query_embedding"] = {
        "schema_version": 1,
        "model": model_name,
        "dimension": dimension,
        "normalized": True,
    }
    graph.metadata.set(scene_metadata)
    return len(described_nodes), dimension


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_spatial_query_records(
    binding_report: Path,
    *,
    source_dsg: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover DAM-described entities rejected only for lacking an object mesh."""

    report = _read_json(binding_report)
    if report.get("schema") != "daaam.dsg_semantic_binding.v1":
        raise ValueError(f"Unsupported binding report: {binding_report}")
    expected_dsg_digest = str(report.get("output_dsg_sha256") or "").lower()
    if expected_dsg_digest != _sha256(source_dsg):
        raise ValueError("Binding report does not describe --dsg-file")

    terminal: dict[str, tuple[int, str]] = {}
    for event in report.get("events") or []:
        if not isinstance(event, dict):
            raise ValueError("Binding report events must be objects")
        status = str(event.get("status") or "")
        if status not in {"matched_real_mesh", "rejected_no_mesh"}:
            continue
        entity_id = str(event.get("entity_id") or "").strip()
        semantic_id = int(event.get("semantic_id") or 0)
        if not entity_id or semantic_id <= 0:
            raise ValueError("Binding report has an invalid terminal entity event")
        previous = terminal.get(entity_id)
        current = (semantic_id, status)
        if previous is not None and previous != current:
            raise ValueError(f"Binding report conflicts for entity {entity_id}")
        terminal[entity_id] = current

    rejected = {
        entity_id: semantic_id
        for entity_id, (semantic_id, status) in terminal.items()
        if status == "rejected_no_mesh"
    }
    expected_rejected = int(
        dict(report.get("verification") or {}).get("rejected_no_mesh", -1)
    )
    if expected_rejected != len(rejected):
        raise ValueError(
            "Binding report rejected entity count is inconsistent: "
            f"{len(rejected)} != {expected_rejected}"
        )

    memory_path = Path(str(report.get("source_memory") or "")).expanduser().resolve()
    if not memory_path.is_file():
        raise ValueError(f"Binding report MapMemory is missing: {memory_path}")
    expected_memory_digest = str(report.get("source_memory_sha256") or "").lower()
    if expected_memory_digest and expected_memory_digest != _sha256(memory_path):
        raise ValueError("Binding report MapMemory checksum changed")

    connection = sqlite3.connect(
        f"file:{memory_path}?mode=ro", uri=True, timeout=30.0
    )
    connection.row_factory = sqlite3.Row
    try:
        origin_row = connection.execute(
            "SELECT value FROM metadata WHERE key='time_origin_ns'"
        ).fetchone()
        time_origin_ns = None if origin_row is None else int(origin_row[0])
        records: list[dict[str, Any]] = []
        for entity_id, semantic_id in sorted(
            rejected.items(), key=lambda item: (item[1], item[0])
        ):
            row = connection.execute(
                """SELECT * FROM entities
                    WHERE entity_id=? AND deleted_ns IS NULL""",
                (entity_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Binding entity is absent from MapMemory: {entity_id}")
            description = " ".join(str(row["canonical_name"] or "").split()).strip()
            if not description or description.casefold() == "unknown":
                raise ValueError(f"Binding entity has no DAM description: {entity_id}")
            timestamps = [
                int(value[0])
                for value in connection.execute(
                    """SELECT sensor_time_ns FROM entity_observations
                        WHERE entity_id=?
                        UNION
                        SELECT sensor_time_ns FROM entity_versions
                        WHERE entity_id=?
                          AND action IN ('entity_created', 'entity_observed')
                        ORDER BY sensor_time_ns""",
                    (entity_id, entity_id),
                ).fetchall()
            ]
            first_ns = timestamps[0] if timestamps else int(row["created_ns"])
            last_ns = timestamps[-1] if timestamps else int(row["updated_ns"])
            position = None if row["position_m"] is None else json.loads(row["position_m"])
            dimensions = (
                None if row["dimensions_m"] is None else json.loads(row["dimensions_m"])
            )
            geometry_status = "spatial_only" if position is not None else "image_only"
            records.append(
                {
                    "record_id": f"M({semantic_id})",
                    "entity_id": entity_id,
                    "semantic_label": semantic_id,
                    "description": description,
                    "position_m": position,
                    "dimensions_m": dimensions,
                    "geometry_status": geometry_status,
                    "geometry_confidence": float(row["geometry_confidence"]),
                    "first_observed_ns": first_ns,
                    "last_observed_ns": last_ns,
                    "first_observed_s": (
                        None
                        if time_origin_ns is None
                        else (first_ns - time_origin_ns) / 1.0e9
                    ),
                    "last_observed_s": (
                        None
                        if time_origin_ns is None
                        else (last_ns - time_origin_ns) / 1.0e9
                    ),
                    "observation_count": len(timestamps),
                    "binding_status": "rejected_no_mesh",
                    "source": "map_memory",
                }
            )
    finally:
        connection.close()
    source = {
        "binding_report": str(binding_report),
        "binding_report_sha256": _sha256(binding_report),
        "map_memory": str(memory_path),
        "map_memory_sha256": _sha256(memory_path),
        "time_origin_ns": time_origin_ns,
    }
    return records, source


def attach_record_embeddings(
    records: list[dict[str, Any]],
    *,
    handler: Any,
    expected_dimension: int,
) -> None:
    """Attach normalized sentence vectors to lower-confidence query records."""

    if not records:
        return
    descriptions = list(dict.fromkeys(record["description"] for record in records))
    encoded = handler.extract_text_embeddings(descriptions, show_progress=True)
    vectors = {
        description: _as_list(vector)
        for description, vector in zip(descriptions, encoded)
    }
    dimensions = {len(vector) for vector in vectors.values()}
    if dimensions != {expected_dimension}:
        raise ValueError(
            f"Semantic sidecar embedding dimensions {sorted(dimensions)} do not match "
            f"DSG dimension {expected_dimension}"
        )
    for record in records:
        record["embedding"] = vectors[record["description"]]


def write_manifest(
    output_path: Path,
    source_path: Path,
    *,
    source_sha256: str,
    model_name: str,
    dimension: int,
    queryable_objects: int,
    dsg_queryable_objects: int | None = None,
    geometry_counts: dict[str, int] | None = None,
    semantic_index: dict[str, Any] | None = None,
) -> Path:
    """Atomically write the v1 sidecar understood by old and new services."""
    manifest = {
        "schema_version": 1,
        "dsg_file": output_path.name,
        "dsg_sha256": _sha256(output_path),
        "queryable_objects": queryable_objects,
        "embedding": {
            "field": "attributes.metadata.sentence_embedding_feature",
            "model": model_name,
            "dimension": dimension,
            "normalized": True,
        },
        "source": {
            "dsg_file": str(source_path),
            "dsg_sha256": source_sha256,
        },
    }
    if dsg_queryable_objects is not None:
        manifest["dsg_queryable_objects"] = int(dsg_queryable_objects)
    if geometry_counts is not None:
        manifest["geometry_counts"] = {
            str(key): int(value) for key, value in geometry_counts.items()
        }
    if semantic_index is not None:
        manifest["semantic_index"] = dict(semantic_index)
    manifest_path = output_path.with_suffix(".manifest.json")
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest_path


@click.command()
@click.option(
    "--dsg-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="DSG whose object nodes already contain metadata.description.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="New query-ready DSG. The authoritative source is never overwritten.",
)
@click.option(
    "--sentence-model-name",
    default=DEFAULT_SENTENCE_MODEL,
    show_default=True,
    envvar="DAAAM_QUERY_SENTENCE_EMBEDDING_MODEL_NAME",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    show_default=True,
)
@click.option(
    "--require-all/--allow-undescribed",
    default=False,
    show_default=True,
    help="Fail if any object lacks a description and therefore an embedding.",
)
@click.option(
    "--binding-report",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Strict mesh-binding report. Rejected DAM entities are kept in a "
        "checksum-bound semantic sidecar instead of being dropped from queries."
    ),
)
def main(
    dsg_file: Path,
    output_file: Path,
    sentence_model_name: str,
    device: str,
    require_all: bool,
    binding_report: Path | None,
) -> None:
    """Create multilingual embeddings and a checksum-bound model manifest."""
    source_path = dsg_file.expanduser().resolve()
    output_path = output_file.expanduser().resolve()
    if output_path == source_path:
        raise click.UsageError("--output-file must not overwrite --dsg-file")
    selected_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    if selected_device == "auto":
        selected_device = "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise click.ClickException("CUDA was requested but torch.cuda.is_available() is false")

    source_digest = _sha256(source_path)
    graph = sdsg.DynamicSceneGraph.load(str(source_path))
    object_count = graph.get_layer(sdsg.DsgLayers.OBJECTS).num_nodes()
    if binding_report is None:
        discovered_report = source_path.with_name(f"{source_path.stem}.binding.json")
        if discovered_report.is_file():
            binding_report = discovered_report
    handler = SentenceEmbeddingHandler(
        model_name=sentence_model_name, device=selected_device
    )
    try:
        updated, dimension = attach_description_embeddings(
            graph,
            model_name=sentence_model_name,
            device=selected_device,
            handler=handler,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if require_all and updated != object_count:
        raise click.ClickException(
            f"Only {updated}/{object_count} objects have descriptions"
        )

    spatial_records: list[dict[str, Any]] = []
    spatial_source: dict[str, Any] = {}
    if binding_report is not None:
        try:
            spatial_records, spatial_source = load_spatial_query_records(
                binding_report.expanduser().resolve(), source_dsg=source_path
            )
            attach_record_embeddings(
                spatial_records,
                handler=handler,
                expected_dimension=dimension,
            )
        except (OSError, ValueError, sqlite3.Error) as exc:
            raise click.ClickException(str(exc)) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.tmp{output_path.suffix}"
    )
    graph.save(str(temporary))
    temporary.replace(output_path)
    semantic_index_manifest = None
    if spatial_records:
        try:
            index_path, index_digest = write_query_index(
                output_path,
                spatial_records,
                source=spatial_source,
            )
        except (OSError, QueryIndexError) as exc:
            raise click.ClickException(str(exc)) from exc
        semantic_index_manifest = {
            "schema": "daaam.semantic_query_index.v1",
            "file": index_path.name,
            "sha256": index_digest,
            "records": len(spatial_records),
        }
    geometry_counts = {
        "mesh_bound": updated,
        "spatial_only": sum(
            record["geometry_status"] == "spatial_only" for record in spatial_records
        ),
        "image_only": sum(
            record["geometry_status"] == "image_only" for record in spatial_records
        ),
    }
    total_queryable = updated + len(spatial_records)
    manifest_path = write_manifest(
        output_path,
        source_path,
        source_sha256=source_digest,
        model_name=sentence_model_name,
        dimension=dimension,
        queryable_objects=total_queryable,
        dsg_queryable_objects=updated,
        geometry_counts=geometry_counts,
        semantic_index=semantic_index_manifest,
    )
    click.echo(
        f"Saved {output_path}: {total_queryable} queryable "
        f"({updated} mesh-bound, {len(spatial_records)} semantic sidecar), {dimension}D, "
        f"model={sentence_model_name}, device={selected_device}"
    )
    click.echo(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
