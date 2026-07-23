"""Tests for multilingual map contracts and open-set semantic retrieval."""

from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest

import daaam.semantic_query as semantic_query
from daaam.query_api import create_app
import daaam.utils.embedding as embedding_utils
from daaam.semantic_query import (
    DEFAULT_SENTENCE_MODEL,
    ObjectRecord,
    SemanticQueryEngine,
    SemanticQueryError,
)
import scripts.prepare_query_dsg_embeddings as embedding_prep
from spark_dsg import DsgLayers, DynamicSceneGraph, KhronosObjectAttributes, NodeSymbol


class FakeEncoder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def extract_text_embeddings(
        self, texts: list[str], *, show_progress: bool
    ) -> np.ndarray:
        del show_progress
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def _record(node_id: str, description: str, embedding: list[float]) -> ObjectRecord:
    vector = np.asarray(embedding, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return ObjectRecord(
        node_id=node_id,
        semantic_label=int(node_id.removeprefix("O(").removesuffix(")")),
        description=description,
        position=np.asarray([1.0, 2.0, 3.0]),
        first_observed=1.0,
        last_observed=2.0,
        embedding=vector,
    )


def _write_map_contract(tmp_path, *, model: str = DEFAULT_SENTENCE_MODEL):
    dsg_path = tmp_path / "dsg_updated.json"
    dsg_path.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(dsg_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "dsg_file": dsg_path.name,
        "dsg_sha256": digest,
        "queryable_objects": 2,
        "embedding": {
            "model": model,
            "dimension": 2,
            "normalized": True,
        },
    }
    dsg_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return dsg_path


@pytest.fixture
def query_engine(tmp_path, monkeypatch) -> SemanticQueryEngine:
    records = [
        _record("O(1)", "a white ceiling light", [1.0, 0.0]),
        _record("O(2)", "a white wall light", [0.98, 0.199]),
    ]
    monkeypatch.setattr(semantic_query, "load_object_records", lambda _: records)
    dsg_path = _write_map_contract(tmp_path)
    return SemanticQueryEngine(
        dsg_path,
        encoder=FakeEncoder(
            {
                "白色天花板灯": [1.0, 0.0],
                "不存在的消防栓": [0.0, 1.0],
            }
        ),
    )


def test_default_model_is_multilingual_mpnet() -> None:
    assert DEFAULT_SENTENCE_MODEL == (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )


def test_sentence_embedding_handler_honors_explicit_cpu(monkeypatch) -> None:
    captured = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs) -> None:
            captured["model_name"] = model_name
            captured.update(kwargs)

    monkeypatch.setattr(embedding_utils, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(embedding_utils.torch.cuda, "is_available", lambda: True)
    handler = embedding_utils.SentenceEmbeddingHandler(
        model_name=DEFAULT_SENTENCE_MODEL,
        device="cpu",
    )
    assert handler.device == "cpu"
    assert captured["device"] == "cpu"


def test_retrieval_accepts_chinese_query_and_rejects_low_similarity(
    query_engine: SemanticQueryEngine,
) -> None:
    accepted = query_engine.retrieve_with_decision("白色天花板灯", top_k=2)
    assert accepted.found is True
    assert accepted.rejection_reason is None
    assert [record.node_id for _, record in accepted.matches] == ["O(1)", "O(2)"]

    rejected = query_engine.retrieve_with_decision("不存在的消防栓", top_k=2)
    assert rejected.found is False
    assert rejected.rejection_reason == "below_min_similarity"
    assert rejected.matches == []
    assert rejected.top_score < rejected.min_similarity


def test_optional_top1_margin_rejects_ambiguous_result(
    query_engine: SemanticQueryEngine,
) -> None:
    decision = query_engine.retrieve_with_decision(
        "白色天花板灯", top_k=2, min_margin=0.05
    )
    assert decision.found is False
    assert decision.rejection_reason == "below_min_margin"
    assert decision.matches == []
    assert decision.top1_margin is not None
    assert decision.top1_margin < 0.05


def test_api_returns_found_false_and_no_forced_matches(
    query_engine: SemanticQueryEngine,
) -> None:
    client = TestClient(create_app(query_engine.dsg_path, engine=query_engine))
    response = client.post(
        "/v1/query/retrieve",
        json={"query": "不存在的消防栓", "top_k": 2},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["found"] is False
    assert payload["rejection_reason"] == "below_min_similarity"
    assert payload["matches"] == []
    assert payload["top_score"] < payload["min_similarity"]


def test_api_supports_per_query_margin_and_reports_health_defaults(
    query_engine: SemanticQueryEngine,
) -> None:
    client = TestClient(create_app(query_engine.dsg_path, engine=query_engine))
    response = client.post(
        "/v1/query/retrieve",
        json={"query": "白色天花板灯", "top_k": 2, "min_margin": 0.05},
    )
    assert response.status_code == 200
    assert response.json()["rejection_reason"] == "below_min_margin"

    health = client.get("/health").json()
    assert health["sentence_model"] == DEFAULT_SENTENCE_MODEL
    assert health["embedding_model_verified"] is True
    assert health["default_min_similarity"] == pytest.approx(0.55)
    assert health["default_min_margin"] == pytest.approx(0.0)


def test_model_identity_mismatch_fails_before_loading_encoder(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        semantic_query,
        "load_object_records",
        lambda _: pytest.fail("records must not load after a model mismatch"),
    )
    dsg_path = _write_map_contract(
        tmp_path, model="sentence-transformers/sentence-t5-large"
    )
    with pytest.raises(SemanticQueryError, match="Re-embed the map"):
        SemanticQueryEngine(dsg_path, encoder=FakeEncoder({}))


def test_checksum_bound_manifest_is_required_by_default(tmp_path, monkeypatch) -> None:
    dsg_path = tmp_path / "legacy.json"
    dsg_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        semantic_query,
        "load_object_records",
        lambda _: [_record("O(1)", "light", [1.0, 0.0])],
    )
    with pytest.raises(SemanticQueryError, match="checksum-bound"):
        SemanticQueryEngine(dsg_path, encoder=FakeEncoder({}))


def test_manifest_checksum_mismatch_is_rejected(tmp_path) -> None:
    dsg_path = _write_map_contract(tmp_path)
    dsg_path.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(SemanticQueryError, match="checksum does not match"):
        SemanticQueryEngine(dsg_path, encoder=FakeEncoder({}))


def test_embedding_preparation_attaches_contract_and_writes_compatible_manifest(
    tmp_path, monkeypatch
) -> None:
    graph = DynamicSceneGraph()
    attributes = KhronosObjectAttributes()
    attributes.semantic_label = 7
    attributes.position = [1.0, 2.0, 3.0]
    attributes.metadata.set({"description": "白色天花板灯"})
    assert graph.add_node(DsgLayers.OBJECTS, NodeSymbol("O", 7), attributes)

    class FakeHandler:
        def __init__(self, **kwargs) -> None:
            assert kwargs["model_name"] == DEFAULT_SENTENCE_MODEL

        def extract_text_embeddings(self, texts, show_progress=True):
            assert texts == ["白色天花板灯"]
            assert show_progress is True
            return np.asarray([[0.6, 0.8]], dtype=np.float32)

    monkeypatch.setattr(embedding_prep, "SentenceEmbeddingHandler", FakeHandler)
    updated, dimension = embedding_prep.attach_description_embeddings(
        graph,
        model_name=DEFAULT_SENTENCE_MODEL,
        device="cpu",
    )
    assert (updated, dimension) == (1, 2)
    node = next(iter(graph.get_layer(DsgLayers.OBJECTS).nodes))
    assert node.attributes.metadata.get()["sentence_embedding_feature"] == pytest.approx(
        [0.6, 0.8]
    )
    assert graph.metadata.get()["query_embedding"]["model"] == DEFAULT_SENTENCE_MODEL

    source = tmp_path / "source.json"
    output = tmp_path / "dsg_updated.json"
    DynamicSceneGraph().save(str(source))
    graph.save(str(output))
    manifest_path = embedding_prep.write_manifest(
        output,
        source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        model_name=DEFAULT_SENTENCE_MODEL,
        dimension=dimension,
        queryable_objects=updated,
    )
    manifest = semantic_query.load_query_manifest(output)
    assert manifest_path == output.with_suffix(".manifest.json")
    assert manifest["embedding"]["model"] == DEFAULT_SENTENCE_MODEL
    assert manifest["embedding"]["dimension"] == 2
