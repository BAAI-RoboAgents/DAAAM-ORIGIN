"""Optional integration test for live semantic corrections on a real Spark DSG."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


spark_dsg = pytest.importorskip("spark_dsg")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.grounding.models import ObjectAnnotation  # noqa: E402
from daaam.memory import DeliveredSemanticCorrection  # noqa: E402
from daaam.realtime.contracts import SemanticCorrection  # noqa: E402
from daaam.realtime.semantic import HydraDsgSemanticSink  # noqa: E402
from daaam.scene_graph.services import SceneGraphService  # noqa: E402
from spark_dsg import (  # noqa: E402
    DsgLayers,
    DynamicSceneGraph,
    KhronosObjectAttributes,
    Labelspace,
    NodeSymbol,
    PlaceNodeAttributes,
)


def test_live_correction_updates_real_dsg_and_acknowledges_delivery():
    service = SceneGraphService(
        REPOSITORY_ROOT / "config" / "labels_pseudo.yaml",
        REPOSITORY_ROOT / "config" / "labels_pseudo.csv",
        defer_dsg_processing=False,
        enable_background_objects=False,
    )
    graph = DynamicSceneGraph()
    attributes = KhronosObjectAttributes()
    attributes.semantic_label = 42
    attributes.position = [1.0, 2.0, 3.0]
    assert graph.add_node(
        DsgLayers.OBJECTS,
        NodeSymbol("O", 1),
        attributes,
    )
    service.set_scene_graph(graph)

    service.store_correction(
        ObjectAnnotation(
            semantic_id=42,
            semantic_label="chair",
            confidence=0.9,
        )
    )

    node = next(iter(graph.get_layer(DsgLayers.OBJECTS).nodes))
    stats = service.get_correction_stats()
    assert node.attributes.metadata.get()["description"] == "chair"
    assert stats["applied_corrections"] == 1
    assert stats["pending_corrections"] == 0
    assert stats["application_events"] == 1


def _delivered(operation_id: str, label: str) -> DeliveredSemanticCorrection:
    return DeliveredSemanticCorrection(
        SemanticCorrection(
            operation_id=operation_id,
            entity_id="entity-chair",
            sensor_time_ns=1_700_000_000_000_000_000,
            map_revision=0,
            label=label,
            confidence=0.9,
            source="dam:test",
        ),
        label,
    )


def _service() -> SceneGraphService:
    return SceneGraphService(
        REPOSITORY_ROOT / "config" / "labels_pseudo.yaml",
        REPOSITORY_ROOT / "config" / "labels_pseudo.csv",
        defer_dsg_processing=False,
        enable_background_objects=False,
    )


def test_saved_dsg_flush_materializes_and_reload_verifies_object(tmp_path):
    graph_path = tmp_path / "dsg.json"
    graph_with_mesh_path = tmp_path / "dsg_with_mesh.json"
    graph = DynamicSceneGraph()
    place_attributes = PlaceNodeAttributes()
    place_attributes.position = [0.9, 2.1, 3.0]
    assert graph.add_node(
        DsgLayers.PLACES,
        NodeSymbol("p", 1),
        place_attributes,
    )
    graph.save(str(graph_path), include_mesh=False)
    graph.save(str(graph_with_mesh_path), include_mesh=True)
    entity = {
        "canonical_name": "Final Wooden Chair",
        "position_m": [1.0, 2.0, 3.0],
        "dimensions_m": [0.6, 0.7, 1.1],
    }
    sink = HydraDsgSemanticSink(
        _service(),
        entity_lookup=lambda _entity_id: entity,
    )
    sink.register_entity("entity-chair", 42)
    assert sink(_delivered("old-operation", "stale label"))
    assert sink(_delivered("final-operation", "Final Wooden Chair"))
    assert sink.stats()["pending"] == 2

    stats = sink.attach_saved_graph(graph_path)

    assert stats["applied"] == 2
    assert stats["pending"] == 0
    assert stats["verified_operations"] == 2
    assert stats["verified_entities"] == 1
    assert stats["verified_artifacts"] == [
        str(graph_path),
        str(graph_with_mesh_path),
    ]
    for artifact_path in (graph_path, graph_with_mesh_path):
        reloaded = DynamicSceneGraph.load(str(artifact_path))
        nodes = list(reloaded.get_layer(DsgLayers.OBJECTS).nodes)
        assert len(nodes) == 1
        node = nodes[0]
        metadata = node.attributes.metadata.get()
        assert node.attributes.semantic_label == 42
        assert list(node.attributes.position) == pytest.approx([1.0, 2.0, 3.0])
        assert list(node.attributes.bounding_box.dimensions) == pytest.approx(
            [0.6, 0.7, 1.1]
        )
        assert metadata["entity_id"] == "entity-chair"
        assert metadata["geometry_source"] == "map_memory"
        assert metadata["description"] == "final wooden chair"
        assert metadata["first_observed_ns"] == 1_700_000_000_000_000_000
        assert metadata["last_observed_ns"] == 1_700_000_000_000_000_000
        assert metadata["parent_binding"] == "nearest_place"
        assert node.attributes.is_active is True
        assert node.has_parent()
        assert int(node.get_parent()) == NodeSymbol("p", 1).value
        assert (
            reloaded.get_labelspace(2, 0).labels_to_names[42]
            == "final wooden chair"
        )
        if artifact_path == graph_with_mesh_path:
            assert (
                reloaded.get_labelspace("mesh").labels_to_names[42]
                == "final wooden chair"
            )
    manifest_path = tmp_path / "semantic_dsg_commit.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "daaam.semantic_dsg_commit.v1"
    assert manifest["verified_entity_count"] == 1
    assert manifest["verified_operation_count"] == 2
    assert set(manifest["artifacts"]) == {"dsg.json", "dsg_with_mesh.json"}
    assert manifest["artifacts"]["dsg.json"]["has_mesh"] is False
    assert manifest["artifacts"]["dsg_with_mesh.json"]["has_mesh"] is False
    assert (
        manifest["artifacts"]["dsg_with_mesh.json"]["requested_include_mesh"]
        is True
    )
    assert stats["commit_manifest_path"] == str(manifest_path)
    assert len(stats["commit_manifest_sha256"]) == 64
    graph_path.write_text(graph_path.read_text() + "\n")
    invalidated = sink.stats()
    assert invalidated["commit_valid"] is False
    assert invalidated["graph_attached"] is False
    assert invalidated["applied"] == 0
    assert invalidated["pending"] == 2


def test_saved_dsg_flush_keeps_ack_pending_without_real_geometry(tmp_path):
    graph_path = tmp_path / "dsg.json"
    DynamicSceneGraph().save(str(graph_path))
    sink = HydraDsgSemanticSink(
        _service(),
        entity_lookup=lambda _entity_id: {
            "canonical_name": "chair",
            "position_m": None,
            "dimensions_m": None,
        },
    )
    sink.register_entity("entity-chair", 42)
    assert sink(_delivered("operation", "chair"))

    stats = sink.attach_saved_graph(graph_path)

    assert stats["applied"] == 0
    assert stats["pending"] == 1
    reloaded = DynamicSceneGraph.load(str(graph_path))
    assert not list(reloaded.get_layer(DsgLayers.OBJECTS).nodes)


def test_existing_unbound_object_without_geometry_does_not_ack(tmp_path):
    graph_path = tmp_path / "dsg.json"
    graph = DynamicSceneGraph()
    attributes = KhronosObjectAttributes()
    attributes.semantic_label = 42
    attributes.position = [99.0, 99.0, 99.0]
    assert graph.add_node(DsgLayers.OBJECTS, NodeSymbol("O", 42), attributes)
    graph.save(str(graph_path), include_mesh=False)
    sink = HydraDsgSemanticSink(
        _service(),
        entity_lookup=lambda _entity_id: {
            "canonical_name": "chair",
            "position_m": None,
            "dimensions_m": None,
        },
    )
    sink.register_entity("entity-chair", 42)
    assert sink(_delivered("operation", "chair"))

    stats = sink.attach_saved_graph(graph_path)

    assert stats["applied"] == 0
    assert stats["pending"] == 1
    reloaded = DynamicSceneGraph.load(str(graph_path))
    node = next(iter(reloaded.get_layer(DsgLayers.OBJECTS).nodes))
    assert not node.attributes.metadata.get().get("entity_id")


def test_conflicting_entity_binding_does_not_ack(tmp_path):
    graph_path = tmp_path / "dsg.json"
    graph = DynamicSceneGraph()
    attributes = KhronosObjectAttributes()
    attributes.semantic_label = 42
    attributes.position = [1.0, 2.0, 3.0]
    attributes.metadata.set(
        {
            "entity_id": "different-entity",
            "description": "original label",
        }
    )
    assert graph.add_node(DsgLayers.OBJECTS, NodeSymbol("O", 42), attributes)
    graph.set_labelspace(Labelspace({42: "original label"}), 2, 0)
    graph.save(str(graph_path), include_mesh=False)
    sink = HydraDsgSemanticSink(
        _service(),
        entity_lookup=lambda _entity_id: {
            "canonical_name": "chair",
            "position_m": [1.0, 2.0, 3.0],
            "dimensions_m": [0.6, 0.7, 1.1],
        },
    )
    sink.register_entity("entity-chair", 42)
    assert sink(_delivered("operation", "chair"))

    stats = sink.attach_saved_graph(graph_path)

    assert stats["applied"] == 0
    assert stats["pending"] == 1
    assert "already bound" in " ".join(stats["errors"])
    reloaded = DynamicSceneGraph.load(str(graph_path))
    node = next(iter(reloaded.get_layer(DsgLayers.OBJECTS).nodes))
    assert node.attributes.metadata.get()["entity_id"] == "different-entity"
    assert node.attributes.metadata.get()["description"] == "original label"
    assert reloaded.get_labelspace(2, 0).labels_to_names[42] == "original label"


def test_correction_updates_only_canonical_fragment_for_semantic_id(tmp_path):
    graph_path = tmp_path / "dsg.json"
    graph = DynamicSceneGraph()
    place_attributes = PlaceNodeAttributes()
    place_attributes.position = [0.0, 0.0, 0.0]
    assert graph.add_node(
        DsgLayers.PLACES,
        NodeSymbol("p", 1),
        place_attributes,
    )
    for index, position in ((1, [1.0, 0.0, 0.0]), (2, [9.0, 0.0, 0.0])):
        attributes = KhronosObjectAttributes()
        attributes.semantic_label = 42
        attributes.position = position
        assert graph.add_node(
            DsgLayers.OBJECTS,
            NodeSymbol("O", index),
            attributes,
        )
    graph.save(str(graph_path), include_mesh=False)
    sink = HydraDsgSemanticSink(
        _service(),
        entity_lookup=lambda _entity_id: {
            "canonical_name": "chair",
            "position_m": [1.1, 0.0, 0.0],
            "dimensions_m": [0.6, 0.7, 1.1],
        },
    )
    sink.register_entity("entity-chair", 42)
    assert sink(_delivered("operation", "chair"))

    stats = sink.attach_saved_graph(graph_path)

    assert stats["applied"] == 1
    reloaded = DynamicSceneGraph.load(str(graph_path))
    nodes = sorted(
        reloaded.get_layer(DsgLayers.OBJECTS).nodes,
        key=lambda node: node.id.category_id,
    )
    canonical_metadata = nodes[0].attributes.metadata.get()
    fragment_metadata = nodes[1].attributes.metadata.get()
    assert canonical_metadata["entity_id"] == "entity-chair"
    assert canonical_metadata["description"] == "chair"
    assert nodes[0].attributes.is_active
    assert nodes[0].has_parent()
    assert not fragment_metadata.get("entity_id")
    assert not fragment_metadata.get("description")
