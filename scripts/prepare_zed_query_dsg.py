#!/usr/bin/env python3
"""Attach DAAAM descriptions and sentence embeddings to a Hydra ZED DSG."""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import click
import numpy as np
import spark_dsg as sdsg
import torch
import yaml

from daaam.utils.embedding import SentenceEmbeddingHandler


UNKNOWN_LABELS = {"", "unknown", "none", "null"}


def _as_list(value: Any) -> Any:
    return value.tolist() if isinstance(value, np.ndarray) else value


def choose_latest_corrections_file(run_dir: Path) -> Path:
    """Return the newest corrections.yaml below a standalone pipeline run."""
    candidates = list(run_dir.glob("out_*/corrections.yaml"))
    if (run_dir / "corrections.yaml").exists():
        candidates.append(run_dir / "corrections.yaml")
    if not candidates:
        raise FileNotFoundError(f"No corrections.yaml found under {run_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def build_description_map(corrections: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Index all usable DAAAM labels by their Hydra semantic ID."""
    descriptions: Dict[int, Dict[str, Any]] = {}
    for entry in corrections.get("label_names", []):
        name = str(entry.get("name", "")).strip()
        if entry.get("label") is None or name.lower() in UNKNOWN_LABELS:
            continue
        descriptions[int(entry["label"])] = {
            "description": name,
            "temporal_history": dict(entry.get("temporal_history") or {}),
        }
    if not descriptions:
        raise ValueError("corrections.yaml contains no usable descriptions")
    return descriptions


def compute_sentence_embeddings(
    descriptions: Mapping[int, Mapping[str, Any]], model_name: str
) -> Dict[int, Any]:
    """Encode each description using the same query embedding family as DAAAM."""
    handler = SentenceEmbeddingHandler(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    semantic_ids = list(descriptions)
    texts = [str(descriptions[semantic_id]["description"]) for semantic_id in semantic_ids]
    embeddings = handler.extract_text_embeddings(texts, show_progress=True)
    return {
        semantic_id: _as_list(embedding)
        for semantic_id, embedding in zip(semantic_ids, embeddings)
    }


def attach_query_metadata(
    scene_graph: sdsg.DynamicSceneGraph,
    descriptions: Mapping[int, Mapping[str, Any]],
    embeddings: Mapping[int, Any],
) -> int:
    """Attach text metadata to object nodes and the graph-level feature table."""
    features: Dict[str, Dict[str, Any]] = {
        str(semantic_id): {"sentence_embedding_feature": _as_list(embedding)}
        for semantic_id, embedding in embeddings.items()
    }
    updated = 0
    for node in scene_graph.get_layer(sdsg.DsgLayers.OBJECTS).nodes:
        semantic_id = int(node.attributes.semantic_label)
        if semantic_id not in descriptions:
            continue
        metadata = dict(node.attributes.metadata.get() or {})
        metadata["description"] = descriptions[semantic_id]["description"]
        metadata["temporal_history"] = descriptions[semantic_id]["temporal_history"]
        metadata["sentence_embedding_feature"] = embeddings[semantic_id]
        node.attributes.metadata.set(metadata)
        updated += 1

    scene_metadata = dict(scene_graph.metadata.get() or {})
    scene_features = dict(scene_metadata.get("features", {}))
    scene_features.update(features)
    scene_metadata["features"] = scene_features
    scene_graph.metadata.set(scene_metadata)
    return updated


@click.command()
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option(
    "--dsg-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Input DSG. Defaults to hydra_output/backend/dsg_with_mesh.json.",
)
@click.option(
    "--corrections-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Defaults to the newest out_*/corrections.yaml below --run-dir.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False),
    default=None,
    help="Defaults to <run-dir>/dsg_updated.json.",
)
@click.option(
    "--sentence-model-name",
    default="sentence-transformers/sentence-t5-large",
    show_default=True,
    envvar="DAAAM_QUERY_SENTENCE_EMBEDDING_MODEL_NAME",
)
@click.option(
    "--require-all/--allow-unmatched",
    default=False,
    show_default=True,
    help="Fail when any Hydra object lacks a usable DAAAM description.",
)
def main(
    run_dir: str,
    dsg_file: Optional[str],
    corrections_file: Optional[str],
    output_file: Optional[str],
    sentence_model_name: str,
    require_all: bool,
) -> None:
    """Build a query-ready scene graph from one standalone pipeline run."""
    run_path = Path(run_dir).resolve()
    dsg_path = (
        Path(dsg_file).resolve()
        if dsg_file
        else run_path / "hydra_output/backend/dsg_with_mesh.json"
    )
    corrections_path = (
        Path(corrections_file).resolve()
        if corrections_file
        else choose_latest_corrections_file(run_path)
    )
    output_path = Path(output_file).resolve() if output_file else run_path / "dsg_updated.json"

    with corrections_path.open("r", encoding="utf-8") as stream:
        descriptions = build_description_map(yaml.safe_load(stream) or {})
    embeddings = compute_sentence_embeddings(descriptions, sentence_model_name)
    scene_graph = sdsg.DynamicSceneGraph.load(str(dsg_path))
    updated = attach_query_metadata(scene_graph, descriptions, embeddings)
    object_count = scene_graph.get_layer(sdsg.DsgLayers.OBJECTS).num_nodes()
    if updated == 0:
        raise ValueError("No Hydra object nodes matched DAAAM descriptions")
    if require_all and updated != object_count:
        raise ValueError(
            f"Only {updated}/{object_count} object nodes matched DAAAM descriptions"
        )
    scene_graph.save(str(output_path))
    click.echo(
        f"Saved {output_path}: {updated}/{object_count} object nodes, "
        f"{len(descriptions)} descriptions, {len(next(iter(embeddings.values())))}D embeddings"
    )


if __name__ == "__main__":
    main()
