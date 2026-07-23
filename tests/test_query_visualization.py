from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from daaam.query_evidence import QueryEvidence, sha256_file
from daaam.query_visualization import (
    QUERY_VISUAL_REPORT_SCHEMA,
    write_query_visuals,
)
from daaam.semantic_query import ObjectRecord


class _FakeEngine:
    def __init__(self, dsg_path: Path, evidence: QueryEvidence) -> None:
        self.dsg_path = dsg_path
        self._evidence = evidence

    def evidence_for_node(self, node_id: str) -> QueryEvidence | None:
        return self._evidence if node_id == self._evidence.node_id else None


def test_write_query_visuals_copies_evidence_and_marks_mesh(tmp_path: Path) -> None:
    dsg_path = tmp_path / "dsg_updated.json"
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
                        {"r": 255, "g": 0, "b": 0, "a": 255},
                        {"r": 0, "g": 255, "b": 0, "a": 255},
                        {"r": 0, "g": 0, "b": 255, "a": 255},
                        {"r": 255, "g": 255, "b": 255, "a": 255},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    source_evidence = tmp_path / "O_40.png"
    source_evidence.write_bytes(b"checksum-bound image evidence")
    evidence = QueryEvidence(
        evidence_id="O_40",
        node_id="O(40)",
        semantic_label=40,
        frame_index=542,
        sensor_time_ns=123,
        observed_s=10.5,
        bbox_xyxy=(10, 20, 30, 40),
        mask_pixels=400,
        mask_source="fastsam_segmentation",
        image_path=source_evidence,
        image_sha256=sha256_file(source_evidence),
        source_image_sha256="a" * 64,
        mask_sha256="b" * 64,
        camera_position_m=(0.25, 0.5, 1.0),
    )
    record = ObjectRecord(
        node_id="O(40)",
        semantic_label=40,
        description="black computer monitor",
        position=np.asarray([1.25, 1.5, 0.8]),
        first_observed=10.0,
        last_observed=11.0,
        embedding=np.asarray([1.0, 0.0]),
    )

    artifacts = write_query_visuals(
        engine=_FakeEngine(dsg_path, evidence),  # type: ignore[arg-type]
        query="黑色电脑显示器",
        matches=[(0.78, record)],
        output_root=tmp_path / "query_results",
        found=True,
        rejection_reason=None,
        top_score=0.78,
        top1_margin=0.12,
    )

    assert artifacts.topdown_image.is_file()
    assert artifacts.topdown_image.stat().st_size > 1_000
    assert len(artifacts.evidence_images) == 1
    assert artifacts.evidence_images[0].read_bytes() == source_evidence.read_bytes()

    report = json.loads(artifacts.report.read_text(encoding="utf-8"))
    assert report["schema"] == QUERY_VISUAL_REPORT_SCHEMA
    assert report["query"] == "黑色电脑显示器"
    assert report["dsg_sha256"] == sha256_file(dsg_path)
    assert report["topdown_image"] == artifacts.topdown_image.name
    assert report["matches"][0]["node_id"] == "O(40)"
    assert report["matches"][0]["position_m"] == [1.25, 1.5, 0.8]
    assert report["matches"][0]["evidence_image"] == artifacts.evidence_images[0].name
