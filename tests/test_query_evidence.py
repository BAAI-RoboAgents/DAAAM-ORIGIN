"""Tests for checksum-bound top-1 FastSAM query evidence."""

from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest

import daaam.semantic_query as semantic_query
from daaam.query_api import create_app
from daaam.query_evidence import EVIDENCE_SCHEMA, infer_segmentation_frame_indices
from daaam.semantic_query import (
    DEFAULT_SENTENCE_MODEL,
    ObjectRecord,
    SemanticQueryEngine,
    SemanticQueryError,
)
from scripts.prepare_query_evidence import (
    accumulated_segmentation_frame_indices,
    largest_component,
    render_evidence_image,
    render_masked_cutout,
)


class FakeEncoder:
    def extract_text_embeddings(self, texts, *, show_progress=False):
        del show_progress
        vectors = {"纸箱": [1.0, 0.0], "消防栓": [0.0, 1.0]}
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def _record() -> ObjectRecord:
    return ObjectRecord(
        node_id="O(6)",
        semantic_label=15,
        description="a cardboard box",
        position=np.asarray([1.0, 2.0, 3.0]),
        first_observed=0.0,
        last_observed=4.0,
        embedding=np.asarray([1.0, 0.0], dtype=np.float32),
    )


def _write_contract(tmp_path):
    dsg_path = tmp_path / "dsg_updated.json"
    dsg_path.write_text("{}\n", encoding="utf-8")
    dsg_digest = hashlib.sha256(dsg_path.read_bytes()).hexdigest()
    dsg_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dsg_sha256": dsg_digest,
                "queryable_objects": 1,
                "embedding": {
                    "model": DEFAULT_SENTENCE_MODEL,
                    "dimension": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    image_directory = tmp_path / "query_evidence"
    image_directory.mkdir()
    image_path = image_directory / "O_6.png"
    image_path.write_bytes(b"valid-enough-for-file-response")
    image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    cutout_directory = image_directory / "cutouts"
    cutout_directory.mkdir()
    cutout_path = cutout_directory / "O_6.png"
    cutout_path.write_bytes(b"checksum-bound-masked-cutout")
    cutout_digest = hashlib.sha256(cutout_path.read_bytes()).hexdigest()
    dsg_path.with_suffix(".evidence.json").write_text(
        json.dumps(
            {
                "schema": EVIDENCE_SCHEMA,
                "dsg_sha256": dsg_digest,
                "objects": [
                    {
                        "evidence_id": "O_6",
                        "node_id": "O(6)",
                        "semantic_label": 15,
                        "frame_index": 7,
                        "sensor_time_ns": 1_700_000_000_000_000_007,
                        "observed_s": 2.5,
                        "bbox_xyxy": [2, 3, 8, 9],
                        "mask_pixels": 24,
                        "mask_source": "fastsam_segmentation",
                        "image": "query_evidence/O_6.png",
                        "image_sha256": image_digest,
                        "cutout": "query_evidence/cutouts/O_6.png",
                        "cutout_sha256": cutout_digest,
                        "camera_position_m": [4.0, 5.0, 1.2],
                        "source_image_sha256": "a" * 64,
                        "mask_sha256": "b" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return dsg_path, image_path


def test_segmentation_schedule_matches_timestamp_rate_gate() -> None:
    origin = 1_700_000_000_000_000_000
    times = [
        origin,
        origin + 100_000_000,
        origin + 200_000_000,
        origin + 399_999_999,
        origin + 400_000_000,
    ]
    assert infer_segmentation_frame_indices(times, 5.0) == [0, 2, 4]


def test_segmentation_schedule_restarts_at_resume_boundary() -> None:
    origin = 1_700_000_000_000_000_000
    times = [origin + index * 100_000_000 for index in range(8)]
    accumulated, reported = accumulated_segmentation_frame_indices(
        times, 5.0, frames_resumed_from=5
    )
    assert accumulated == [0, 2, 4, 5, 7]
    assert reported == [5, 7]


def test_largest_component_and_render_preserve_exact_mask() -> None:
    labels = np.zeros((20, 30), dtype=np.uint16)
    labels[1:3, 1:3] = 15
    labels[5:15, 8:25] = 15
    mask, bbox, area, border_touch = largest_component(labels, 15)
    assert bbox == (8, 5, 25, 15)
    assert area == 170
    assert border_touch is False
    assert int(mask.sum()) == area
    image = np.full((20, 30, 3), 80, dtype=np.uint8)
    annotated = render_evidence_image(
        image,
        mask,
        bbox_xyxy=bbox,
        node_id="O(6)",
        semantic_label=15,
        observed_s=2.5,
    )
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)
    cutout = render_masked_cutout(
        image, mask, bbox_xyxy=bbox, max_size_px=64
    )
    assert cutout.dtype == np.uint8
    assert cutout.shape[2] == 4
    assert max(cutout.shape[:2]) <= 64
    assert np.any(cutout[:, :, 3] == 0)
    assert np.any(cutout[:, :, 3] > 0)
    assert np.all(cutout[cutout[:, :, 3] == 0, :3] == 0)


def test_api_returns_and_serves_top1_evidence(tmp_path, monkeypatch) -> None:
    dsg_path, image_path = _write_contract(tmp_path)
    monkeypatch.setattr(semantic_query, "load_object_records", lambda _: [_record()])
    engine = SemanticQueryEngine(dsg_path, encoder=FakeEncoder())
    loaded_evidence = engine.evidence_by_node["O(6)"]
    assert loaded_evidence.cutout_path is not None
    assert loaded_evidence.camera_position_m == (4.0, 5.0, 1.2)
    client = TestClient(create_app(dsg_path, engine=engine))

    response = client.post(
        "/v1/query/retrieve", json={"query": "纸箱", "top_k": 1}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["found"] is True
    assert payload["top1_evidence"] == {
        "evidence_id": "O_6",
        "image_url": "/v1/evidence/O_6.png",
        "frame_index": 7,
        "sensor_time_ns": 1_700_000_000_000_000_007,
        "observed_s": 2.5,
        "bbox_xyxy": [2, 3, 8, 9],
        "mask_pixels": 24,
        "mask_source": "fastsam_segmentation",
        "source_image_sha256": "a" * 64,
        "annotated_image_sha256": hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest(),
    }
    evidence = client.get(payload["top1_evidence"]["image_url"])
    assert evidence.status_code == 200
    assert evidence.content == image_path.read_bytes()
    assert evidence.headers["x-daaam-mask-source"] == "fastsam_segmentation"
    assert client.get("/health").json()["evidence_available_objects"] == 1

    rejected = client.post(
        "/v1/query/retrieve", json={"query": "消防栓", "top_k": 1}
    ).json()
    assert rejected["found"] is False
    assert rejected["top1_evidence"] is None


def test_tampered_evidence_image_blocks_service_start(tmp_path, monkeypatch) -> None:
    dsg_path, image_path = _write_contract(tmp_path)
    image_path.write_bytes(b"tampered")
    monkeypatch.setattr(semantic_query, "load_object_records", lambda _: [_record()])
    with pytest.raises(SemanticQueryError, match="checksum mismatch"):
        SemanticQueryEngine(dsg_path, encoder=FakeEncoder())
