"""Tests for high-recall semantic queries beside a strict mesh DSG."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

import daaam.semantic_query as semantic_query
from daaam.query_api import create_app
from daaam.query_evidence import QueryEvidence
from daaam.query_index import write_query_index
from daaam.semantic_query import (
    DEFAULT_SENTENCE_MODEL,
    ObjectRecord,
    SemanticQueryEngine,
    SemanticQueryError,
)


class FakeEncoder:
    def extract_text_embeddings(self, texts, *, show_progress=False):
        del show_progress
        vectors = {
            "纸箱": [0.0, 1.0],
            "显示器": [1.0, 0.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def _mesh_record() -> ObjectRecord:
    return ObjectRecord(
        node_id="O(7)",
        semantic_label=7,
        description="a computer monitor",
        position=np.asarray([1.0, 2.0, 0.8]),
        first_observed=1.0,
        last_observed=4.0,
        embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        entity_id="entity-mesh",
        dimensions=np.asarray([0.7, 0.2, 0.5]),
        geometry_status="mesh_bound",
    )


def _write_high_recall_contract(tmp_path):
    dsg_path = tmp_path / "dsg_updated.json"
    dsg_path.write_text("{}\n", encoding="utf-8")
    index_path, index_digest = write_query_index(
        dsg_path,
        [
            {
                "record_id": "M(11)",
                "entity_id": "entity-memory",
                "semantic_label": 11,
                "description": "a cardboard box",
                "position_m": [4.0, 5.0, 0.5],
                "dimensions_m": [0.6, 0.4, 0.5],
                "first_observed_s": 2.0,
                "last_observed_s": 8.0,
                "first_observed_ns": 102,
                "last_observed_ns": 108,
                "geometry_status": "spatial_only",
                "geometry_confidence": 0.8,
                "source": "map_memory",
                "embedding": [0.0, 1.0],
            }
        ],
    )
    manifest = {
        "schema_version": 1,
        "dsg_file": dsg_path.name,
        "dsg_sha256": hashlib.sha256(dsg_path.read_bytes()).hexdigest(),
        "queryable_objects": 2,
        "dsg_queryable_objects": 1,
        "geometry_counts": {
            "mesh_bound": 1,
            "spatial_only": 1,
            "image_only": 0,
        },
        "embedding": {
            "model": DEFAULT_SENTENCE_MODEL,
            "dimension": 2,
            "normalized": True,
        },
        "semantic_index": {
            "schema": "daaam.semantic_query_index.v1",
            "file": index_path.name,
            "sha256": index_digest,
            "records": 1,
        },
    }
    dsg_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return dsg_path, index_path


def test_default_query_includes_spatial_only_and_mesh_filter_is_optional(
    tmp_path, monkeypatch
) -> None:
    dsg_path, _ = _write_high_recall_contract(tmp_path)
    monkeypatch.setattr(
        semantic_query, "load_object_records", lambda _path: [_mesh_record()]
    )
    engine = SemanticQueryEngine(dsg_path, encoder=FakeEncoder())

    decision = engine.retrieve_with_decision("纸箱", top_k=1)
    assert decision.found is True
    assert decision.matches[0][1].node_id == "M(11)"
    assert decision.matches[0][1].geometry_status == "spatial_only"

    mesh_only = engine.retrieve_with_decision(
        "纸箱", top_k=1, require_mesh=True, min_similarity=-1.0
    )
    assert mesh_only.matches[0][1].node_id == "O(7)"
    assert mesh_only.matches[0][1].geometry_status == "mesh_bound"

    client = TestClient(create_app(dsg_path, engine=engine))
    payload = client.post(
        "/v1/query/retrieve", json={"query": "纸箱", "top_k": 1}
    ).json()
    assert payload["matches"][0]["geometry_status"] == "spatial_only"
    assert payload["matches"][0]["position_m"] == [4.0, 5.0, 0.5]
    health = client.get("/health").json()
    assert health["queryable_objects"] == 2
    assert health["mesh_bound_objects"] == 1
    assert health["spatial_only_objects"] == 1


def test_tampered_semantic_sidecar_blocks_query_service(tmp_path, monkeypatch) -> None:
    dsg_path, index_path = _write_high_recall_contract(tmp_path)
    index_path.write_text('{"tampered": true}\n', encoding="utf-8")
    monkeypatch.setattr(
        semantic_query, "load_object_records", lambda _path: [_mesh_record()]
    )
    with pytest.raises(SemanticQueryError, match="checksum mismatch"):
        SemanticQueryEngine(dsg_path, encoder=FakeEncoder())


def test_rgbd_evidence_geometry_replaces_legacy_spatial_position(
    tmp_path, monkeypatch
) -> None:
    dsg_path, _ = _write_high_recall_contract(tmp_path)
    monkeypatch.setattr(
        semantic_query, "load_object_records", lambda _path: [_mesh_record()]
    )
    evidence = QueryEvidence(
        evidence_id="M_11",
        node_id="M(11)",
        semantic_label=11,
        frame_index=1,
        sensor_time_ns=1_700_000_000_000_000_001,
        observed_s=1.0,
        bbox_xyxy=(0, 0, 2, 2),
        mask_pixels=4,
        mask_source="fastsam_segmentation",
        image_path=Path("dummy.png"),
        image_sha256="a" * 64,
        source_image_sha256="b" * 64,
        mask_sha256="c" * 64,
        geometry_position_m=(1.5, 2.5, 0.05),
        geometry_dimensions_m=(0.7, 0.4, 0.1),
        geometry_source="fastsam_masked_rgbd_joint_backprojection",
    )
    monkeypatch.setattr(
        semantic_query,
        "load_query_evidence",
        lambda _path: ({"M(11)": evidence}, {"M_11": evidence}),
    )

    engine = SemanticQueryEngine(dsg_path, encoder=FakeEncoder())
    record = next(record for record in engine.records if record.node_id == "M(11)")
    assert np.allclose(record.position, [1.5, 2.5, 0.05])
    assert np.allclose(record.dimensions, [0.7, 0.4, 0.1])
    assert record.source == "fastsam_masked_rgbd"
