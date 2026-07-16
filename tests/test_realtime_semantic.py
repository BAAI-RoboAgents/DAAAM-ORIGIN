"""Acceptance tests for the non-blocking real semantic sidecar contracts."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import queue
import sys
import threading

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.grounding.models import ObjectAnnotation  # noqa: E402
from daaam.grounding.workers.dam_grounding import (  # noqa: E402
    DAMGroundingWorkerMultiImage,
)
from daaam.memory import MapMemory  # noqa: E402
from daaam.pipeline.models import PromptRecord  # noqa: E402
from daaam.realtime.contracts import (  # noqa: E402
    FrameValue,
    MessageKey,
    RealtimeEnvelope,
)
from daaam.realtime.semantic import (  # noqa: E402
    HydraDsgSemanticSink,
    RealtimeSemanticAdapter,
    RealtimeSemanticConfig,
)
from daaam.tracking.models import Track  # noqa: E402


ORIGIN_NS = 1_783_933_507_759_540_877


class FakeSegmenter:
    def __init__(self):
        self.calls = 0

    def warmup(self):
        return None

    def segment_checked(self, image):
        self.calls += 1
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[8:24, 12:28] = True
        return np.asarray([[12, 8, 28, 24, 0.9, 0]], dtype=np.float32), [mask]


class FakeTracker:
    def __init__(self):
        self.calls = []

    def warmup(self):
        return None

    def update(self, detections, _image):
        self.calls.append(len(detections))
        if not len(detections):
            return np.empty((0, 8), dtype=np.float32)
        return np.asarray([[12, 8, 28, 24, 1, 0.9, 0, 0]], dtype=np.float32)


class FakePredictingTracker(FakeTracker):
    def update(self, detections, _image):
        self.calls.append(len(detections))
        if len(detections):
            return np.asarray([[12, 8, 28, 24, 1, 0.9, 0, 0]], dtype=np.float32)
        return np.asarray([[16, 8, 32, 24, 1, 0.8, 0, -1]], dtype=np.float32)


class FakeDuplicateSegmenter(FakeSegmenter):
    def segment_checked(self, image):
        self.calls += 1
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[8:24, 12:28] = True
        detections = np.asarray(
            [
                [12, 8, 28, 24, 0.9, 0],
                [12, 8, 28, 24, 0.8, 0],
            ],
            dtype=np.float32,
        )
        return detections, [mask, mask.copy()]


class FakeDuplicateTracker(FakeTracker):
    def update(self, detections, _image):
        self.calls.append(len(detections))
        return np.asarray(
            [
                [12, 8, 28, 24, 1, 0.9, 0, 0],
                [12, 8, 28, 24, 2, 0.8, 0, 1],
            ],
            dtype=np.float32,
        )


class FakeDsgSink:
    def __init__(self):
        self.service = SimpleNamespace(color_map=None)
        self.mappings = {}
        self.updates = []

    def register_entity(self, entity_id, semantic_id):
        self.mappings[entity_id] = semantic_id

    def __call__(self, update):
        self.updates.append(update)
        return True

    def stats(self):
        return {
            "mapped_entities": len(self.mappings),
            "applied": len(self.updates),
            "pending": 0,
            "unmapped": 0,
            "graph_attached": True,
            "errors": [],
        }

    def persist(self):
        return None


class StrictFakeDsgSink(FakeDsgSink):
    def register_entity(self, entity_id, semantic_id):
        existing = self.mappings.get(entity_id)
        if existing is not None and existing != semantic_id:
            raise ValueError("stable entity was assigned multiple semantic IDs")
        super().register_entity(entity_id, semantic_id)


def pipeline_config():
    return SimpleNamespace(
        segmentation=SimpleNamespace(polygon_epsilon_factor=0.001),
        tracking=SimpleNamespace(),
        workers=SimpleNamespace(),
        semantic_config_path=str(REPOSITORY_ROOT / "config/labels_pseudo.yaml"),
        labelspace_colors_path=str(REPOSITORY_ROOT / "config/labels_pseudo.csv"),
    )


def envelope(sensor_time_ns):
    source = SimpleNamespace(
        frame_index=0,
        intrinsics=np.asarray([[60.0, 0.0, 31.5], [0.0, 60.0, 23.5], [0.0, 0.0, 1.0]]),
        world_T_camera=np.eye(4),
    )
    payload = SimpleNamespace(
        source=source,
        rgb_image=np.full((48, 64, 3), 80, dtype=np.uint8),
        depth_m=np.full((48, 64), 1.5, dtype=np.float32),
    )
    return RealtimeEnvelope(
        MessageKey(sensor_time_ns), payload, FrameValue.ROUTINE, source="test"
    )


def test_semantic_frontend_segments_at_5hz_but_ticks_tracker_every_frame(tmp_path):
    memory = MapMemory(tmp_path / "memory.sqlite3")
    memory.create_session("replay", ORIGIN_NS, canonical=True)
    segmenter = FakeSegmenter()
    tracker = FakeTracker()
    sink = FakeDsgSink()
    adapter = RealtimeSemanticAdapter(
        pipeline_config(),
        memory,
        session_id="replay",
        output_dir=tmp_path / "semantic",
        config=RealtimeSemanticConfig(
            segmentation_rate_hz=5.0,
            minimum_observations=1,
            grounding_enabled=False,
            gpu_activity_path=tmp_path / "gpu.activity",
        ),
        segmentation_service=segmenter,
        tracking_service=tracker,
        dsg_sink=sink,
    )
    adapter.start()
    segmented_time_ns = ORIGIN_NS + 1
    propagated_time_ns = ORIGIN_NS + 100_000_001
    adapter.handle(envelope(segmented_time_ns))
    adapter.handle(envelope(propagated_time_ns))
    labels = adapter.label_image_for(segmented_time_ns)
    propagated = adapter.label_image_for(propagated_time_ns)
    audit = adapter.label_audit_for(propagated_time_ns)
    stats = adapter.stop()
    assert segmenter.calls == 1
    assert tracker.calls == [1, 0]
    assert labels is not None and int(labels.max()) == 1
    assert propagated is not None and np.array_equal(propagated, labels)
    assert audit is not None
    assert audit["sensor_time_ns"] == propagated_time_ns
    assert audit["mode"] == "propagation"
    assert audit["tracks"][0]["source_sensor_time_ns"] == segmented_time_ns
    assert audit["tracks"][0]["method"] == "carry_forward"
    assert stats["tracking_calls"] == 2
    assert stats["propagation_frames"] == 1
    assert stats["propagation_instances"] == 1
    assert stats["propagation_carry_forwards"] == 1
    assert stats["propagation"]["active_track_ids"] == [1]
    assert stats["dsg"]["mapped_entities"] == 1
    assert memory.stats()["entities"]["active"] == 1
    assert (tmp_path / "gpu.activity").is_file()
    memory.close()


def test_non_segmentation_mask_follows_botsort_bbox_with_exact_provenance(tmp_path):
    memory = MapMemory(tmp_path / "memory.sqlite3")
    memory.create_session("replay", ORIGIN_NS, canonical=True)
    adapter = RealtimeSemanticAdapter(
        pipeline_config(),
        memory,
        session_id="replay",
        output_dir=tmp_path / "semantic",
        config=RealtimeSemanticConfig(
            segmentation_rate_hz=5.0,
            minimum_observations=1,
            grounding_enabled=False,
        ),
        segmentation_service=FakeSegmenter(),
        tracking_service=FakePredictingTracker(),
        dsg_sink=FakeDsgSink(),
    )
    first_time_ns = ORIGIN_NS + 1
    second_time_ns = ORIGIN_NS + 100_000_001
    adapter.handle(envelope(first_time_ns))
    adapter.handle(envelope(second_time_ns))
    labels = adapter.label_image_for(second_time_ns)
    audit = adapter.label_audit_for(second_time_ns)
    stats = adapter.stats()
    assert labels is not None
    assert not np.any(labels[8:24, 12:16])
    assert np.all(labels[8:24, 16:32] == 1)
    assert adapter.label_image_for(second_time_ns + 1) is None
    assert audit is not None
    assert audit["tracks"][0]["method"] == "botsort_bbox_warp"
    assert audit["tracks"][0]["age_ns"] == second_time_ns - first_time_ns
    assert audit["tracks"][0]["propagation_steps"] == 1
    assert len(audit["tracks"][0]["mask_sha256"]) == 64
    assert stats["propagation_bbox_warps"] == 1
    assert stats["propagation_carry_forwards"] == 0
    assert stats["dsg"]["mapped_entities"] == 1
    memory.close()


def test_mapmemory_entity_owns_canonical_semantic_id_across_duplicate_tracks(
    tmp_path,
):
    memory = MapMemory(tmp_path / "memory.sqlite3")
    memory.create_session("replay", ORIGIN_NS, canonical=True)
    sink = StrictFakeDsgSink()
    adapter = RealtimeSemanticAdapter(
        pipeline_config(),
        memory,
        session_id="replay",
        output_dir=tmp_path / "semantic",
        config=RealtimeSemanticConfig(grounding_enabled=False),
        segmentation_service=FakeDuplicateSegmenter(),
        tracking_service=FakeDuplicateTracker(),
        dsg_sink=sink,
    )
    sensor_time_ns = ORIGIN_NS + 1
    adapter.handle(envelope(sensor_time_ns))
    labels = adapter.label_image_for(sensor_time_ns)
    stats = adapter.stats()
    assert labels is not None and set(np.unique(labels)) == {0, 1}
    assert len(sink.mappings) == 1
    assert stats["entity_semantic_merges"] == 1
    assert memory.stats()["entities"]["active"] == 1
    memory.close()


def test_propagation_state_and_audit_history_are_bounded(tmp_path):
    memory = MapMemory(tmp_path / "memory.sqlite3")
    memory.create_session("replay", ORIGIN_NS, canonical=True)
    adapter = RealtimeSemanticAdapter(
        pipeline_config(),
        memory,
        session_id="replay",
        output_dir=tmp_path / "semantic",
        config=RealtimeSemanticConfig(
            segmentation_rate_hz=1.0,
            propagation_max_frames=1,
            label_cache_frames=2,
            grounding_enabled=False,
        ),
        segmentation_service=FakeSegmenter(),
        tracking_service=FakeTracker(),
        dsg_sink=FakeDsgSink(),
    )
    times_ns = [ORIGIN_NS + 1, ORIGIN_NS + 100_000_001, ORIGIN_NS + 200_000_001]
    for sensor_time_ns in times_ns:
        adapter.handle(envelope(sensor_time_ns))
    expired_labels = adapter.label_image_for(times_ns[-1])
    stats = adapter.stats()
    assert expired_labels is not None and not np.any(expired_labels)
    assert adapter.label_image_for(times_ns[0]) is None
    assert stats["propagation_expired"] == 1
    assert stats["propagation_audit_evictions"] == 1
    assert stats["propagation"]["active_tracks"] == 0
    assert stats["propagation"]["cached_sensor_times_ns"] == times_ns[1:]
    assert len(stats["propagation"]["recent_audit"]) == 2
    memory.close()


def test_dam_annotation_becomes_idempotent_versioned_memory_update(tmp_path):
    memory = MapMemory(tmp_path / "memory.sqlite3")
    memory.create_session("replay", ORIGIN_NS, canonical=True)
    sink = FakeDsgSink()
    adapter = RealtimeSemanticAdapter(
        pipeline_config(),
        memory,
        session_id="replay",
        output_dir=tmp_path / "semantic",
        config=RealtimeSemanticConfig(minimum_observations=1, grounding_enabled=False),
        segmentation_service=FakeSegmenter(),
        tracking_service=FakeTracker(),
        dsg_sink=sink,
    )
    adapter.handle(envelope(ORIGIN_NS + 1))
    entity_id = next(iter(sink.mappings))
    adapter.process_annotation(
        ObjectAnnotation(
            semantic_id=1,
            semantic_label="wooden chair",
            confidence=0.0,
            entity_id=entity_id,
            request_id="request",
            sensor_time_ns=ORIGIN_NS + 1,
            map_revision=0,
            source_model="nvidia/DAM-3B",
        )
    )
    adapter.processor.process_once()
    adapter.processor.process_once()
    assert memory.get_entity(entity_id)["canonical_name"] == "wooden chair"
    assert memory.correction_stats()["applied"] == 1
    assert len(sink.updates) == 1
    memory.close()


class FakeSceneGraphService:
    def __init__(self):
        self.scene_graph_is_set = False
        self.correction_lock = threading.RLock()
        self.scene_graph_lock = threading.RLock()
        self.applied_correction_ids = set()

    def store_correction(self, correction):
        if self.scene_graph_is_set:
            self.applied_correction_ids.add(correction.semantic_id)


def test_dsg_sink_has_separate_pending_and_applied_ack():
    service = FakeSceneGraphService()
    sink = HydraDsgSemanticSink(service)
    sink.register_entity("entity", 7)
    correction = SimpleNamespace(
        operation_id="operation",
        entity_id="entity",
        sensor_time_ns=ORIGIN_NS,
        map_revision=0,
        confidence=0.8,
        source="dam:test",
    )
    update = SimpleNamespace(correction=correction, effective_label="chair")
    assert sink(update)
    assert sink.stats()["pending"] == 1
    service.scene_graph_is_set = True
    sink.register_entity("entity", 7)
    assert sink.stats()["pending"] == 0
    assert sink.stats()["applied"] == 1


class FakeAgedDamWorker(DAMGroundingWorkerMultiImage):
    def _initialize_models(self):
        self.dam_agent = object()
        self.compute_full_image_description = False
        self.sentence_embedding_handler = None
        self.clip_handler = None
        self.save_grounding_images = False
        self.output_dir = None
        self.color_map = None

    def _process_aggregated_batch(self):
        if not self.aggregated_records:
            return None
        record = self.aggregated_records[0]
        self.aggregated_records = []
        self.total_masks = 0
        self._batch_started_monotonic = None
        return [
            ObjectAnnotation(
                semantic_id=record.object_labels[record.tracks[0].id],
                semantic_label="chair",
            )
        ]


def test_underfilled_dam_batch_flushes_by_age():
    incoming = queue.Queue()
    outgoing = queue.Queue()
    stop = threading.Event()
    worker = FakeAgedDamWorker(
        incoming,
        outgoing,
        stop,
        {
            "multi_image_min_n_masks": 16,
            "max_batch_age_s": 0.05,
            "enable_selectframe_clip_features": False,
        },
    )
    mask = np.ones((8, 8), dtype=bool)
    track = Track.from_mask(1, mask, np.asarray([0, 0, 8, 8]))
    incoming.put(
        PromptRecord(
            frame=np.zeros((8, 8, 3), dtype=np.uint8),
            tracks=[track],
            object_labels={1: 1},
        )
    )
    thread = threading.Thread(target=worker.run)
    thread.start()
    correction = outgoing.get(timeout=1.0)
    stop.set()
    thread.join(1.0)
    assert correction.semantic_label == "chair"
    assert not thread.is_alive()
