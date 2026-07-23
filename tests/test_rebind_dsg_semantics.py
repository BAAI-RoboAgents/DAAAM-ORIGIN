"""Pure record recovery tests for cross-run semantic rebinding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from rebind_dsg_semantics import (  # noqa: E402
    _merge_semantic_records,
    _report_rejection_records,
    _semantic_id_owners,
    migrate,
)


def test_rejection_records_merge_report_only_entities_without_losing_dsg_data():
    report = {
        "semantic_stats": {
            "dsg": {
                "rejection_audit": [
                    {
                        "status": "rejected_no_mesh",
                        "entity_id": "entity-chair",
                        "semantic_id": 7,
                    },
                    {
                        "status": "rejected_no_mesh",
                        "entity_id": "entity-lamp",
                        "semantic_id": 9,
                    },
                    {
                        "status": "rejected_binding_error",
                        "entity_id": "ignored-error",
                        "semantic_id": 11,
                    },
                ]
            }
        }
    }
    dsg_records = [
        {
            "entity_id": "entity-chair",
            "semantic_id": 7,
            "description": "black office chair",
            "embedding": [0.1, 0.2],
            "selectframe_clip_feature": [0.3],
            "record_sources": ["semantic_source_dsg"],
            "description_source": "semantic_source_dsg",
        }
    ]

    recovered = _report_rejection_records(report)
    merged = _merge_semantic_records(dsg_records, recovered)

    assert [record["entity_id"] for record in recovered] == [
        "entity-chair",
        "entity-lamp",
    ]
    assert merged[0]["description"] == "black office chair"
    assert merged[0]["record_sources"] == [
        "semantic_source_dsg",
        "semantic_source_report",
    ]
    assert merged[1]["entity_id"] == "entity-lamp"
    assert merged[1]["description_source"] == "map_memory_canonical_name"


@pytest.mark.parametrize(
    "dsg_records, report_records, message",
    [
        (
            [{"entity_id": "one", "semantic_id": 1}],
            [{"entity_id": "one", "semantic_id": 2}],
            "conflicting semantic IDs",
        ),
        (
            [{"entity_id": "one", "semantic_id": 1}],
            [{"entity_id": "two", "semantic_id": 1}],
            "belongs to both",
        ),
    ],
)
def test_semantic_record_merge_rejects_identity_conflicts(
    dsg_records,
    report_records,
    message,
):
    with pytest.raises(ValueError, match=message):
        _merge_semantic_records(dsg_records, report_records)


def test_explicit_report_requires_rejection_audit_contract():
    with pytest.raises(ValueError, match="rejection_audit"):
        _report_rejection_records({"semantic_stats": {"dsg": {}}})


def test_owner_map_uses_every_merged_semantic_record():
    records = _merge_semantic_records(
        [
            {
                "entity_id": "entity-twelve",
                "semantic_id": 12,
                "record_sources": ["semantic_source_dsg"],
            }
        ],
        [
            {
                "entity_id": "entity-fifteen",
                "semantic_id": 15,
                "record_sources": ["semantic_source_report"],
            }
        ],
    )

    assert _semantic_id_owners(records) == {
        12: "entity-twelve",
        15: "entity-fifteen",
    }


def test_rebind_reserves_mesh_for_later_matching_semantic_record(tmp_path):
    pytest.importorskip("spark_dsg")
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from daaam.memory import MapMemory
    from spark_dsg import (
        BoundingBoxType,
        DsgLayers,
        DynamicSceneGraph,
        KhronosObjectAttributes,
        NodeSymbol,
    )

    origin_ns = 1_700_000_000_000_000_000
    memory_path = tmp_path / "map_memory.sqlite3"
    with MapMemory(memory_path) as memory:
        memory.set_time_origin_ns(origin_ns)
        memory.create_session("session", origin_ns, canonical=True)
        wrong_entity_id, _ = memory.observe_entity(
            "session",
            "wrong-entity",
            np.asarray([0.0, 0.0, 0.0]),
            sensor_time_ns=origin_ns + 1,
            semantic_label="wrong object",
            dimensions_m=np.asarray([0.8, 0.8, 0.8]),
        )
        matching_entity_id, _ = memory.observe_entity(
            "session",
            "matching-entity",
            np.asarray([0.0, 0.0, 0.0]),
            sensor_time_ns=origin_ns + 2,
            semantic_label="matching object",
            dimensions_m=np.asarray([0.8, 0.8, 0.8]),
        )

    target_graph = DynamicSceneGraph()
    target_attributes = KhronosObjectAttributes()
    target_attributes.semantic_label = 12
    target_attributes.position = [0.0, 0.0, 0.0]
    target_attributes.bounding_box.type = BoundingBoxType.AABB
    target_attributes.bounding_box.world_P_center = [0.0, 0.0, 0.0]
    target_attributes.bounding_box.dimensions = [0.8, 0.8, 0.8]
    target_attributes.mesh().set_vertices(np.zeros((6, 3), dtype=np.float64))
    assert target_graph.add_node(
        DsgLayers.OBJECTS,
        NodeSymbol("O", 1),
        target_attributes,
    )
    target_path = tmp_path / "target.json"
    target_graph.save(str(target_path), include_mesh=True)

    semantic_graph = DynamicSceneGraph()
    for node_index, semantic_id, entity_id, description in (
        (1, 11, wrong_entity_id, "wrong object"),
        (2, 12, matching_entity_id, "matching object"),
    ):
        attributes = KhronosObjectAttributes()
        attributes.semantic_label = semantic_id
        attributes.position = [0.0, 0.0, 0.0]
        attributes.metadata.set(
            {"entity_id": entity_id, "description": description}
        )
        assert semantic_graph.add_node(
            DsgLayers.OBJECTS,
            NodeSymbol("O", node_index),
            attributes,
        )
    semantic_path = tmp_path / "semantic-source.json"
    semantic_graph.save(str(semantic_path), include_mesh=False)

    output_path = tmp_path / "rebound.json"
    audit_path = tmp_path / "rebound.binding.json"
    audit = migrate(
        SimpleNamespace(
            dsg=target_path,
            semantic_source_dsg=semantic_path,
            semantic_source_report=None,
            memory=memory_path,
            output=output_path,
            audit_output=audit_path,
            semantic_config=REPOSITORY_ROOT / "config" / "labels_pseudo.yaml",
            labelspace_colors=REPOSITORY_ROOT / "config" / "labels_pseudo.csv",
            maximum_center_distance_m=0.75,
            maximum_aabb_gap_m=0.15,
            time_origin_ns=origin_ns,
            force=False,
        )
    )

    rebound = DynamicSceneGraph.load(str(output_path))
    node = next(iter(rebound.get_layer(DsgLayers.OBJECTS).nodes))
    metadata = dict(node.attributes.metadata.get() or {})
    assert node.attributes.semantic_label == 12
    assert metadata["entity_id"] == matching_entity_id
    assert metadata["description"] == "matching object"
    assert audit["verification"]["matched_real_mesh"] == 1
    assert audit["verification"]["rejected_no_mesh"] == 1
    assert any(
        event["status"] == "rejected_reserved_semantic_owner"
        and event["semantic_id"] == 11
        and event["candidate_semantic_id"] == 12
        and event["reserved_entity_id"] == matching_entity_id
        for event in audit["events"]
    )
