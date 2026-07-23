"""Optional integration test for live semantic corrections on a real Spark DSG."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
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
    BoundingBoxType,
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


def _delivered_for_entity(
    operation_id: str,
    label: str,
    entity_id: str,
) -> DeliveredSemanticCorrection:
    return DeliveredSemanticCorrection(
        SemanticCorrection(
            operation_id=operation_id,
            entity_id=entity_id,
            sensor_time_ns=1_700_000_000_000_000_000,
            map_revision=0,
            label=label,
            confidence=0.9,
            source="dam:test",
        ),
        label,
    )


def _delivered(operation_id: str, label: str) -> DeliveredSemanticCorrection:
    return _delivered_for_entity(operation_id, label, "entity-chair")


def _service() -> SceneGraphService:
    return SceneGraphService(
        REPOSITORY_ROOT / "config" / "labels_pseudo.yaml",
        REPOSITORY_ROOT / "config" / "labels_pseudo.csv",
        defer_dsg_processing=False,
        enable_background_objects=False,
    )


def test_saved_dsg_flush_binds_real_mesh_and_reload_verifies_object(tmp_path):
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
    assert graph.add_node(
        DsgLayers.OBJECTS,
        NodeSymbol("O", 1),
        _mesh_object(900, [1.0, 2.0, 3.0], [0.6, 0.7, 1.1]),
    )
    graph.save(str(graph_path), include_mesh=False)
    graph.save(str(graph_with_mesh_path), include_mesh=True)
    entity = {
        "canonical_name": "Final Wooden Chair",
        "position_m": [1.0, 2.0, 3.0],
        "dimensions_m": [0.6, 0.7, 1.1],
        "time_origin_ns": 1_699_999_990_000_000_000,
        "temporal_history": {
            "time_origin_ns": 1_699_999_990_000_000_000,
            "first_observed_ns": 1_699_999_995_000_000_000,
            "last_observed_ns": 1_700_000_000_000_000_000,
            "observation_count": 3,
        },
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
        assert metadata["geometry_source"] == "hydra_object_mesh"
        assert metadata["mesh_binding_status"] == "matched_real_mesh"
        assert node.attributes.mesh().num_vertices() == 3
        assert metadata["description"] == "final wooden chair"
        assert metadata["first_observed_ns"] == 1_699_999_995_000_000_000
        assert metadata["last_observed_ns"] == 1_700_000_000_000_000_000
        assert metadata["temporal_history"]["first_observed"] == 5.0
        assert metadata["temporal_history"]["last_observed"] == 10.0
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


def test_sink_observation_priority_cannot_steal_registered_semantic_mesh(tmp_path):
    graph_path = tmp_path / "dsg.json"
    graph_with_mesh_path = tmp_path / "dsg_with_mesh.json"
    graph = DynamicSceneGraph()
    assert graph.add_node(
        DsgLayers.OBJECTS,
        NodeSymbol("O", 1),
        _mesh_object(12, [0.0, 0.0, 0.0], [0.8, 0.8, 0.8]),
    )
    graph.save(str(graph_path), include_mesh=False)
    graph.save(str(graph_with_mesh_path), include_mesh=True)
    origin_ns = 1_700_000_000_000_000_000
    snapshots = {
        "entity-twelve": {
            "canonical_name": "semantic twelve object",
            "position_m": [0.0, 0.0, 0.0],
            "dimensions_m": [0.8, 0.8, 0.8],
            "time_origin_ns": origin_ns,
            "temporal_history": {
                "first_observed_ns": origin_ns,
                "last_observed_ns": origin_ns,
                "observation_count": 1,
            },
        },
        "entity-fifteen": {
            "canonical_name": "cardboard box",
            "position_m": [0.0, 0.0, 0.0],
            "dimensions_m": [0.8, 0.8, 0.8],
            "time_origin_ns": origin_ns,
            "temporal_history": {
                "first_observed_ns": origin_ns,
                "last_observed_ns": origin_ns,
                "observation_count": 40,
            },
        },
    }
    service = _service()
    sink = HydraDsgSemanticSink(
        service,
        entity_lookup=lambda entity_id: snapshots[entity_id],
    )
    sink.register_entity("entity-twelve", 12)
    sink.register_entity("entity-fifteen", 15)
    assert sink(
        _delivered_for_entity("operation-fifteen", "box", "entity-fifteen")
    )

    stats = sink.attach_saved_graph(graph_path)

    assert stats["applied"] == 0
    assert stats["rejected"] == 1
    assert stats["rejected_no_mesh"] == 1
    assert not stats["errors"]
    initially_reloaded = DynamicSceneGraph.load(str(graph_with_mesh_path))
    initially_reserved_node = next(
        iter(initially_reloaded.get_layer(DsgLayers.OBJECTS).nodes)
    )
    assert initially_reserved_node.attributes.semantic_label == 12
    assert not (
        initially_reserved_node.attributes.metadata.get() or {}
    ).get("entity_id")

    assert sink(
        _delivered_for_entity("operation-twelve", "object", "entity-twelve")
    )
    sink.persist()
    stats = sink.stats()

    assert stats["applied"] == 1
    assert stats["rejected"] == 1
    reloaded = DynamicSceneGraph.load(str(graph_with_mesh_path))
    node = next(iter(reloaded.get_layer(DsgLayers.OBJECTS).nodes))
    assert node.attributes.semantic_label == 12
    assert node.attributes.metadata.get()["entity_id"] == "entity-twelve"
    assert any(
        event["status"] == "rejected_reserved_semantic_owner"
        and event["entity_id"] == "entity-fifteen"
        and event["candidate_semantic_id"] == 12
        and event["reserved_entity_id"] == "entity-twelve"
        for event in service.object_binding_audit
    )


def test_saved_dsg_flush_rejects_without_real_geometry(tmp_path):
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
    assert stats["pending"] == 0
    assert stats["rejected"] == 1
    assert stats["rejected_no_mesh"] == 1
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
    assert stats["pending"] == 0
    assert stats["rejected"] == 1
    assert stats["rejected_no_mesh"] == 1
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
    assert stats["pending"] == 0
    assert stats["rejected"] == 1
    assert stats["rejected_no_mesh"] == 0
    assert "already bound" in " ".join(stats["errors"])
    reloaded = DynamicSceneGraph.load(str(graph_path))
    node = next(iter(reloaded.get_layer(DsgLayers.OBJECTS).nodes))
    assert node.attributes.metadata.get()["entity_id"] == "different-entity"
    assert node.attributes.metadata.get()["description"] == "original label"
    assert reloaded.get_labelspace(2, 0).labels_to_names[42] == "original label"


def test_correction_rejects_unmeshed_semantic_fragments(tmp_path):
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

    assert stats["applied"] == 0
    assert stats["pending"] == 0
    assert stats["rejected_no_mesh"] == 1
    reloaded = DynamicSceneGraph.load(str(graph_path))
    nodes = sorted(
        reloaded.get_layer(DsgLayers.OBJECTS).nodes,
        key=lambda node: node.id.category_id,
    )
    assert all(not (node.attributes.metadata.get() or {}).get("entity_id") for node in nodes)
    assert all(not (node.attributes.metadata.get() or {}).get("description") for node in nodes)


def _mesh_object(
    semantic_id: int,
    position: list[float],
    dimensions: list[float],
) -> KhronosObjectAttributes:
    attributes = KhronosObjectAttributes()
    attributes.semantic_label = semantic_id
    attributes.position = position
    attributes.bounding_box.type = BoundingBoxType.AABB
    attributes.bounding_box.world_P_center = position
    attributes.bounding_box.dimensions = dimensions
    attributes.mesh().set_vertices(np.zeros((6, 3), dtype=np.float64))
    return attributes


def test_existing_fallback_is_replaced_by_spatially_verified_real_mesh():
    graph = DynamicSceneGraph()
    mesh_attributes = _mesh_object(900, [1.0, 2.0, 3.0], [0.8, 0.8, 1.2])
    assert graph.add_node(
        DsgLayers.OBJECTS,
        NodeSymbol("O", 1),
        mesh_attributes,
    )
    fallback_attributes = KhronosObjectAttributes()
    fallback_attributes.semantic_label = 42
    fallback_attributes.position = [1.1, 2.0, 3.0]
    fallback_attributes.bounding_box.type = BoundingBoxType.AABB
    fallback_attributes.bounding_box.world_P_center = [1.1, 2.0, 3.0]
    fallback_attributes.bounding_box.dimensions = [0.6, 0.7, 1.1]
    fallback_attributes.metadata.set(
        {
            "entity_id": "entity-chair",
            "description": "old chair",
            "geometry_source": "map_memory",
        }
    )
    assert graph.add_node(
        DsgLayers.OBJECTS,
        NodeSymbol("O", 2),
        fallback_attributes,
    )
    service = _service()
    service.set_scene_graph(graph)
    origin_ns = 1_700_000_000_000_000_000

    assert service.ensure_object_node(
        semantic_id=42,
        entity_id="entity-chair",
        position_m=[1.1, 2.0, 3.0],
        dimensions_m=[0.6, 0.7, 1.1],
        sensor_time_ns=origin_ns + 9_000_000_000,
        time_origin_ns=origin_ns,
        temporal_history={
            "first_observed_ns": origin_ns + 1_000_000_000,
            "last_observed_ns": origin_ns + 9_000_000_000,
            "observation_count": 5,
        },
        semantic_id_owners={42: "entity-chair"},
    )
    service.add_correction(
        ObjectAnnotation(
            semantic_id=42,
            entity_id="entity-chair",
            semantic_label="Office Chair",
            embedding=[0.1, 0.2],
        )
    )
    service.apply_corrections()

    nodes = list(graph.get_layer(DsgLayers.OBJECTS).nodes)
    assert len(nodes) == 1
    node = nodes[0]
    metadata = node.attributes.metadata.get()
    assert node.attributes.mesh().num_vertices() == 3
    assert node.attributes.semantic_label == 42
    assert metadata["entity_id"] == "entity-chair"
    assert metadata["description"] == "office chair"
    assert metadata["geometry_source"] == "hydra_object_mesh"
    assert metadata["mesh_binding_status"] == "matched_real_mesh"
    assert metadata["has_object_mesh"] is True
    assert metadata["sentence_embedding_feature"] == pytest.approx([0.1, 0.2])
    history = metadata["temporal_history"]
    assert history["time_origin_ns"] == origin_ns
    assert history["first_observed"] == pytest.approx(1.0)
    assert history["last_observed"] == pytest.approx(9.0)
    assert history["first_observed_ns"] == origin_ns + 1_000_000_000
    assert history["last_observed_ns"] == origin_ns + 9_000_000_000
    audit = service.get_correction_stats()["object_binding_recent"][-1]
    assert audit["status"] == "matched_real_mesh"
    assert audit["replaced_unmeshed_nodes"] == ["O(2)"]
    assert audit["thresholds"] == {
        "maximum_center_distance_m": 0.75,
        "maximum_aabb_gap_m": 0.15,
    }
    assert audit["candidate_semantic_id"] == 900


def test_reserved_semantic_mesh_survives_earlier_spatial_fallback_claim():
    graph = DynamicSceneGraph()
    assert graph.add_node(
        DsgLayers.OBJECTS,
        NodeSymbol("O", 1),
        _mesh_object(12, [0.0, 0.0, 0.0], [0.8, 0.8, 0.8]),
    )
    service = _service()
    service.set_scene_graph(graph)
    owners = {12: "entity-twelve", 15: "entity-fifteen"}

    assert not service.ensure_object_node(
        semantic_id=15,
        entity_id="entity-fifteen",
        position_m=[0.0, 0.0, 0.0],
        dimensions_m=[0.8, 0.8, 0.8],
        sensor_time_ns=1_700_000_000_000_000_015,
        allow_unmeshed_fallback=False,
        semantic_id_owners=owners,
    )

    node = next(iter(graph.get_layer(DsgLayers.OBJECTS).nodes))
    assert node.attributes.semantic_label == 12
    assert not (node.attributes.metadata.get() or {}).get("entity_id")
    reservation = next(
        event
        for event in service.object_binding_audit
        if event["status"] == "rejected_reserved_semantic_owner"
    )
    assert reservation["entity_id"] == "entity-fifteen"
    assert reservation["semantic_id"] == 15
    assert reservation["candidate_semantic_id"] == 12
    assert reservation["reserved_entity_id"] == "entity-twelve"

    assert service.ensure_object_node(
        semantic_id=12,
        entity_id="entity-twelve",
        position_m=[0.0, 0.0, 0.0],
        dimensions_m=[0.8, 0.8, 0.8],
        sensor_time_ns=1_700_000_000_000_000_012,
        allow_unmeshed_fallback=False,
        semantic_id_owners=owners,
    )
    node = next(iter(graph.get_layer(DsgLayers.OBJECTS).nodes))
    assert node.attributes.semantic_label == 12
    assert node.attributes.metadata.get()["entity_id"] == "entity-twelve"


def test_owner_mapping_validates_current_binding_request():
    service = _service()
    service.set_scene_graph(DynamicSceneGraph())

    with pytest.raises(ValueError, match="owner mapping"):
        service.ensure_object_node(
            semantic_id=12,
            entity_id="wrong-entity",
            position_m=[0.0, 0.0, 0.0],
            dimensions_m=[0.8, 0.8, 0.8],
            sensor_time_ns=1_700_000_000_000_000_012,
            semantic_id_owners={12: "entity-twelve"},
        )


def test_distant_mesh_is_not_claimed_and_fallback_is_removed():
    graph = DynamicSceneGraph()
    mesh_attributes = _mesh_object(900, [10.0, 0.0, 0.0], [0.2, 0.2, 0.2])
    assert graph.add_node(
        DsgLayers.OBJECTS,
        NodeSymbol("O", 1),
        mesh_attributes,
    )
    fallback_attributes = KhronosObjectAttributes()
    fallback_attributes.semantic_label = 42
    fallback_attributes.position = [0.0, 0.0, 0.0]
    fallback_attributes.bounding_box.type = BoundingBoxType.AABB
    fallback_attributes.bounding_box.world_P_center = [0.0, 0.0, 0.0]
    fallback_attributes.bounding_box.dimensions = [0.5, 0.5, 0.5]
    fallback_attributes.metadata.set({"entity_id": "entity-chair"})
    assert graph.add_node(
        DsgLayers.OBJECTS,
        NodeSymbol("O", 2),
        fallback_attributes,
    )
    service = _service()
    service.set_scene_graph(graph)

    assert not service.ensure_object_node(
        semantic_id=42,
        entity_id="entity-chair",
        position_m=[0.0, 0.0, 0.0],
        dimensions_m=[0.5, 0.5, 0.5],
        sensor_time_ns=1_700_000_000_000_000_000,
        allow_unmeshed_fallback=False,
    )

    nodes = list(graph.get_layer(DsgLayers.OBJECTS).nodes)
    assert len(nodes) == 1
    assert nodes[0].attributes.mesh().num_vertices() == 3
    assert not (nodes[0].attributes.metadata.get() or {}).get("entity_id")
    assert service.object_binding_audit[-1]["status"] == "rejected_no_mesh"
    assert service.object_binding_audit[-1]["removed_unmeshed_nodes"] == ["O(2)"]
