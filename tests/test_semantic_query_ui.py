from __future__ import annotations

import json
from pathlib import Path

import cv2
from fastapi.testclient import TestClient
import numpy as np

from daaam.semantic_query_ui import SemanticMapService, create_semantic_query_app
from daaam.query_evidence import QueryEvidence, sha256_file
from daaam.semantic_query import ObjectRecord, RetrievalDecision


class _FakeQueryEngine:
    def __init__(
        self, dsg_path: Path, record: ObjectRecord, evidence: QueryEvidence
    ) -> None:
        self.dsg_path = dsg_path
        self.records = [record]
        self.evidence_by_node = {record.node_id: evidence}
        self.geometry_counts = {
            "mesh_bound": 1,
            "spatial_only": 0,
            "image_only": 0,
        }
        self.embedding_dim = 2
        self.sentence_model_name = "fake-multilingual-model"
        self.encoder_device = "cpu"
        self.min_similarity = 0.55
        self.min_margin = 0.0
        self._record = record
        self._evidence = evidence

    def evidence_for_node(self, node_id: str) -> QueryEvidence | None:
        return self._evidence if node_id == self._record.node_id else None

    def retrieve_with_decision(
        self,
        query: str,
        top_k: int,
        *,
        min_similarity: float | None,
        min_margin: float | None,
        require_mesh: bool,
    ) -> RetrievalDecision:
        assert query == "黑色显示器"
        assert top_k == 3
        assert require_mesh is True
        return RetrievalDecision(
            found=True,
            matches=[(0.81, self._record)],
            rejection_reason=None,
            top_score=0.81,
            top1_margin=None,
            min_similarity=self.min_similarity if min_similarity is None else min_similarity,
            min_margin=self.min_margin if min_margin is None else min_margin,
        )


def _make_query_service(tmp_path: Path) -> tuple[TestClient, Path]:
    output = tmp_path / "output"
    run = output / "semantic-run"
    run.mkdir(parents=True)
    dsg_path = run / "dsg_updated.json"
    dsg_path.write_text(
        json.dumps(
            {
                "mesh": {
                    "points": [
                        [0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [2.0, 2.0, 0.0],
                        [0.0, 2.0, 0.0],
                    ],
                    "colors": [
                        {"r": 200, "g": 210, "b": 220, "a": 255},
                        {"r": 160, "g": 180, "b": 200, "a": 255},
                        {"r": 120, "g": 150, "b": 180, "a": 255},
                        {"r": 220, "g": 225, "b": 230, "a": 255},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    image_path = run / "O_40.png"
    image = np.full((80, 120, 3), 230, dtype=np.uint8)
    cv2.rectangle(image, (50, 18), (100, 70), (0, 110, 255), 3)
    assert cv2.imwrite(str(image_path), image)
    evidence = QueryEvidence(
        evidence_id="O_40",
        node_id="O(40)",
        semantic_label=180,
        frame_index=542,
        sensor_time_ns=123,
        observed_s=71.313,
        bbox_xyxy=(50, 18, 100, 70),
        mask_pixels=1800,
        mask_source="fastsam_segmentation",
        image_path=image_path,
        image_sha256=sha256_file(image_path),
        source_image_sha256="a" * 64,
        mask_sha256="b" * 64,
        camera_position_m=(0.3, 0.4, 1.4),
    )
    record = ObjectRecord(
        node_id="O(40)",
        semantic_label=180,
        description="a slim black computer monitor",
        position=np.asarray([1.2, 1.4, 0.9]),
        dimensions=np.asarray([0.5, 0.1, 0.7]),
        first_observed=70.0,
        last_observed=72.0,
        embedding=np.asarray([1.0, 0.0]),
        entity_id="entity-monitor",
    )
    engine = _FakeQueryEngine(dsg_path, record, evidence)
    service = SemanticMapService(output, engine_factory=lambda _path: engine)
    app = create_semantic_query_app(output, semantic_map_service=service)
    return TestClient(app), run


def test_independent_query_ui_loads_map_queries_and_serves_evidence(
    tmp_path: Path,
) -> None:
    client, run = _make_query_service(tmp_path)

    page = client.get("/")
    assert page.status_code == 200
    assert "语义地图查询" in page.text
    assert client.get("/api/health").json()["service"] == "semantic-query-ui"

    discovered = client.get("/api/maps").json()["maps"]
    assert [item["run_name"] for item in discovered] == ["semantic-run"]

    opened = client.post("/api/map/open", json={"run_path": str(run)})
    assert opened.status_code == 200
    assert opened.json()["queryable_objects"] == 1
    assert opened.json()["evidence_coverage"] == 1.0

    preview = client.get(opened.json()["mesh_preview_url"])
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
    assert cv2.imdecode(
        np.frombuffer(preview.content, dtype=np.uint8), cv2.IMREAD_COLOR
    ).size > 0

    queried = client.post(
        "/api/query",
        json={
            "run_path": str(run),
            "query": "黑色显示器",
            "top_k": 3,
            "require_mesh": True,
        },
    )
    assert queried.status_code == 200
    result = queried.json()
    assert result["found"] is True
    assert result["matches"][0]["node_id"] == "O(40)"
    assert result["matches"][0]["evidence"]["frame_index"] == 542

    assert client.get(result["topdown_image_url"]).status_code == 200
    evidence_response = client.get(result["matches"][0]["evidence"]["image_url"])
    assert evidence_response.status_code == 200
    assert evidence_response.content == (run / "O_40.png").read_bytes()
    report = client.get(result["report_url"])
    assert report.status_code == 200
    assert report.json()["query"] == "黑色显示器"


def test_independent_query_ui_rejects_maps_and_files_outside_output_root(
    tmp_path: Path,
) -> None:
    client, run = _make_query_service(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "dsg_updated.json").write_text("{}", encoding="utf-8")

    rejected_map = client.post("/api/map/open", json={"run_path": str(outside)})
    assert rejected_map.status_code == 400
    assert "输出根目录" in rejected_map.json()["detail"]

    rejected_file = client.get(
        "/api/file",
        params={"run_path": str(run), "path": "../secret.json"},
    )
    assert rejected_file.status_code == 403
