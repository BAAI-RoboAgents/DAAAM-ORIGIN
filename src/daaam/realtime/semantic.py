"""Non-blocking real segmentation, tracking, and DAM semantic sidecar."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Mapping, Optional
import uuid

import cv2
import numpy as np

from daaam.grounding.control import GroundingCorrectionsDrained
from daaam.grounding.models import ObjectAnnotation
from daaam.memory import (
    DeliveredSemanticCorrection,
    MapMemory,
    VersionedCorrectionProcessor,
)
from daaam.pipeline.models import PromptRecord
from daaam.realtime.contracts import RealtimeEnvelope, SemanticCorrection
from daaam.realtime.gpu import SharedGpuCoordinator
from daaam.tracking.models import Track


@dataclass(frozen=True)
class RealtimeSemanticConfig:
    segmentation_rate_hz: float = 5.0
    minimum_observations: int = 5
    prompt_queue_capacity: int = 20
    correction_queue_capacity: int = 50
    label_cache_frames: int = 32
    propagation_max_frames: int = 2
    propagation_track_capacity: int = 256
    grounding_enabled: bool = True
    automatic_confidence: float = 0.5
    gpu_lock_path: Path | str | None = None
    gpu_activity_path: Path | str | None = None

    def __post_init__(self) -> None:
        if self.segmentation_rate_hz <= 0.0:
            raise ValueError("segmentation rate must be positive")
        if self.minimum_observations <= 0:
            raise ValueError("minimum semantic observations must be positive")
        if min(self.prompt_queue_capacity, self.correction_queue_capacity) <= 0:
            raise ValueError("semantic queue capacities must be positive")
        if self.label_cache_frames <= 0:
            raise ValueError("label cache size must be positive")
        if self.propagation_max_frames < 0:
            raise ValueError("maximum propagation frames must be non-negative")
        if self.propagation_track_capacity <= 0:
            raise ValueError("propagation track capacity must be positive")
        if not 0.0 <= self.automatic_confidence <= 1.0:
            raise ValueError("automatic semantic confidence must be in [0, 1]")


@dataclass
class _TrackMaskState:
    """Bounded mask state used only between authoritative segmentations."""

    track_id: int
    semantic_id: int
    mask: np.ndarray
    bbox: np.ndarray
    source_sensor_time_ns: int
    last_sensor_time_ns: int
    propagation_steps: int
    confidence: float
    entity_id: Optional[str] = None


class HydraDsgSemanticSink:
    """Apply versioned memory updates to a saved/live Hydra DSG with a second ACK."""

    def __init__(
        self,
        scene_graph_service: Any,
        *,
        entity_lookup: Optional[Callable[[str], Mapping[str, Any]]] = None,
    ) -> None:
        self.service = scene_graph_service
        self._entity_lookup = entity_lookup
        self._entity_to_semantic_id: dict[str, int] = {}
        self._semantic_id_to_entity: dict[int, str] = {}
        self._effective_label_by_entity: dict[str, str] = {}
        self._pending: dict[str, DeliveredSemanticCorrection] = {}
        self._applied: set[str] = set()
        self._applied_entities: set[str] = set()
        self._errors: list[str] = []
        self._attached_path: Optional[Path] = None
        self._attached_with_mesh_path: Optional[Path] = None
        self._commit_manifest_path: Optional[Path] = None
        self._commit_manifest_sha256: Optional[str] = None
        self._verified_artifacts: list[str] = []
        self._verified_entities = 0
        self._verified_operations = 0
        self._source_mesh_counts: Optional[tuple[int, int]] = None
        self._lock = threading.RLock()

    def register_entity(self, entity_id: str, semantic_id: int) -> None:
        if not entity_id.strip() or semantic_id <= 0:
            raise ValueError("DSG entity mapping is invalid")
        with self._lock:
            existing = self._entity_to_semantic_id.get(entity_id)
            if existing is not None and existing != semantic_id:
                raise ValueError("stable entity was assigned multiple semantic IDs")
            semantic_owner = self._semantic_id_to_entity.get(semantic_id)
            if semantic_owner is not None and semantic_owner != entity_id:
                raise ValueError("semantic ID was assigned to multiple stable entities")
            self._entity_to_semantic_id[entity_id] = semantic_id
            self._semantic_id_to_entity[semantic_id] = entity_id

    def __call__(self, update: DeliveredSemanticCorrection) -> bool:
        with self._lock:
            self._pending[update.correction.operation_id] = update
            self._effective_label_by_entity[update.correction.entity_id] = (
                update.effective_label
            )
        # MapMemory delivery acknowledges durable handoff to this independently
        # audited sink. The DSG ACK remains pending until a mapped node is updated.
        return True

    def _entity_snapshot_locked(self, entity_id: str) -> Optional[Mapping[str, Any]]:
        if self._entity_lookup is None:
            return None
        try:
            return self._entity_lookup(entity_id)
        except KeyError:
            return None

    def _effective_label_locked(self, entity_id: str) -> str:
        snapshot = self._entity_snapshot_locked(entity_id)
        if snapshot is not None:
            canonical_name = str(snapshot.get("canonical_name") or "").strip()
            if canonical_name:
                return canonical_name
        return self._effective_label_by_entity[entity_id]

    @staticmethod
    def _labelspace_has_update(
        graph: Any,
        semantic_id: int,
        effective_label: str,
        *labelspace_key: Any,
    ) -> bool:
        expected = " ".join(effective_label.split()).strip().casefold()
        try:
            labelspace = graph.get_labelspace(*labelspace_key)
            stored = labelspace.labels_to_names[int(semantic_id)]
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError):
            return False
        return " ".join(str(stored).split()).strip().casefold() == expected

    @staticmethod
    def _graph_has_update(
        graph: Any,
        semantic_id: int,
        effective_label: str,
        entity_id: str,
    ) -> bool:
        from spark_dsg import DsgLayers

        expected = " ".join(effective_label.split()).strip().casefold()
        try:
            nodes = graph.get_layer(DsgLayers.OBJECTS).nodes
            requires_parent = bool(
                list(graph.get_layer(DsgLayers.PLACES).nodes)
            )
        except Exception:
            return False
        if not HydraDsgSemanticSink._labelspace_has_update(
            graph,
            semantic_id,
            effective_label,
            2,
            0,
        ):
            return False
        for node in nodes:
            if int(node.attributes.semantic_label) != semantic_id:
                continue
            metadata = node.attributes.metadata.get() or {}
            description = " ".join(str(metadata.get("description", "")).split())
            stored_entity = str(metadata.get("entity_id", "")).strip()
            if description.casefold() != expected:
                continue
            if stored_entity != entity_id:
                continue
            if requires_parent and not node.has_parent():
                continue
            return True
        return False

    def _artifact_targets_locked(self) -> list[tuple[Path, bool]]:
        if self._attached_path is None:
            return []
        targets = [(self._attached_path, False)]
        if self._attached_with_mesh_path is not None:
            targets.append((self._attached_with_mesh_path, True))
        return targets

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _object_signature(graph: Any) -> tuple[tuple[Any, ...], ...]:
        from spark_dsg import DsgLayers

        signature = []
        for node in graph.get_layer(DsgLayers.OBJECTS).nodes:
            metadata = node.attributes.metadata.get() or {}
            signature.append(
                (
                    int(node.id.value),
                    int(node.attributes.semantic_label),
                    str(metadata.get("entity_id", "")),
                    str(metadata.get("description", "")),
                    bool(node.attributes.is_active),
                    int(node.get_parent()) if node.has_parent() else None,
                )
            )
        return tuple(sorted(signature))

    @staticmethod
    def _mesh_counts(graph: Any) -> Optional[tuple[int, int]]:
        if not graph.has_mesh():
            return None
        return int(graph.mesh.num_vertices()), int(graph.mesh.num_faces())

    def _commit_is_valid_locked(self) -> bool:
        manifest_path = self._commit_manifest_path
        if (
            manifest_path is None
            or self._commit_manifest_sha256 is None
            or not manifest_path.is_file()
        ):
            return False
        try:
            if self._file_sha256(manifest_path) != self._commit_manifest_sha256:
                return False
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("schema") != "daaam.semantic_dsg_commit.v1":
                return False
            records = manifest["artifacts"]
            expected_names = {
                target.name for target, _ in self._artifact_targets_locked()
            }
            if set(records) != expected_names:
                return False
            for name, record in records.items():
                artifact = manifest_path.with_name(name)
                if (
                    not artifact.is_file()
                    or self._file_sha256(artifact) != record["sha256"]
                ):
                    return False
            return True
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def _persist_and_reload_locked(
        self,
        expected_updates: tuple[tuple[str, int, str], ...] = (),
        *,
        verified_entity_ids: Optional[set[str]] = None,
        verified_operation_count: Optional[int] = None,
    ) -> tuple[Any, ...]:
        if self._attached_path is None:
            graph = getattr(self.service, "scene_graph", None)
            return () if graph is None else (graph,)
        from spark_dsg import DynamicSceneGraph

        token = uuid.uuid4().hex
        temporary_paths: list[Path] = []
        candidates: list[tuple[Path, Path, bool, Any, str]] = []
        manifest_path = self._attached_path.with_name("semantic_dsg_commit.json")
        self._commit_manifest_path = None
        self._commit_manifest_sha256 = None
        self._verified_artifacts = []
        self._verified_entities = 0
        self._verified_operations = 0
        try:
            if manifest_path.is_file():
                stale_manifest = manifest_path.with_name(
                    f".{manifest_path.stem}.stale.{token}{manifest_path.suffix}"
                )
                manifest_path.replace(stale_manifest)
                temporary_paths.append(stale_manifest)

            with self.service.scene_graph_lock:
                for target, include_mesh in self._artifact_targets_locked():
                    temporary = target.with_name(
                        f".{target.stem}.semantic.{token}.tmp{target.suffix}"
                    )
                    temporary_paths.append(temporary)
                    self.service.scene_graph.save(
                        str(temporary),
                        include_mesh=include_mesh,
                    )
                    candidate_graph = DynamicSceneGraph.load(str(temporary))
                    candidates.append(
                        (
                            target,
                            temporary,
                            include_mesh,
                            candidate_graph,
                            self._file_sha256(temporary),
                        )
                    )

            for target, _, include_mesh, graph, _ in candidates:
                mesh_counts = self._mesh_counts(graph)
                if not include_mesh and mesh_counts is not None:
                    raise RuntimeError("mesh leaked into plain DSG artifact")
                if (
                    include_mesh
                    and self._source_mesh_counts is not None
                    and mesh_counts != self._source_mesh_counts
                ):
                    raise RuntimeError("semantic DSG candidate mesh counts changed")
                for entity_id, semantic_id, effective_label in expected_updates:
                    if not self._graph_has_update(
                        graph,
                        semantic_id,
                        effective_label,
                        entity_id,
                    ):
                        raise RuntimeError(
                            "semantic DSG candidate failed node verification: "
                            f"{entity_id}"
                        )
                    if (
                        target == self._attached_with_mesh_path
                        and not self._labelspace_has_update(
                            graph,
                            semantic_id,
                            effective_label,
                            "mesh",
                        )
                    ):
                        raise RuntimeError(
                            "semantic DSG mesh labelspace failed verification: "
                            f"{entity_id}"
                        )

            signatures = {
                self._object_signature(graph) for _, _, _, graph, _ in candidates
            }
            if len(signatures) != 1:
                raise RuntimeError("semantic DSG candidate artifacts disagree")

            for target, temporary, _, _, _ in candidates:
                temporary.replace(target)

            reloaded = tuple(
                DynamicSceneGraph.load(str(target))
                for target, _, _, _, _ in candidates
            )
            for (target, _, include_mesh, _, candidate_sha256), graph in zip(
                candidates,
                reloaded,
                strict=True,
            ):
                if self._file_sha256(target) != candidate_sha256:
                    raise RuntimeError("semantic DSG final artifact hash changed")
                mesh_counts = self._mesh_counts(graph)
                if not include_mesh and mesh_counts is not None:
                    raise RuntimeError("mesh leaked into final plain DSG artifact")
                if (
                    include_mesh
                    and self._source_mesh_counts is not None
                    and mesh_counts != self._source_mesh_counts
                ):
                    raise RuntimeError("semantic DSG final mesh counts changed")
                for entity_id, semantic_id, effective_label in expected_updates:
                    if not self._graph_has_update(
                        graph,
                        semantic_id,
                        effective_label,
                        entity_id,
                    ):
                        raise RuntimeError(
                            "semantic DSG final artifact failed node verification: "
                            f"{entity_id}"
                        )
                    if (
                        target == self._attached_with_mesh_path
                        and not self._labelspace_has_update(
                            graph,
                            semantic_id,
                            effective_label,
                            "mesh",
                        )
                    ):
                        raise RuntimeError(
                            "semantic DSG final mesh labelspace failed verification: "
                            f"{entity_id}"
                        )

            final_signatures = {self._object_signature(graph) for graph in reloaded}
            if len(final_signatures) != 1:
                raise RuntimeError("semantic DSG final artifacts disagree")

            artifact_records = {}
            for (target, include_mesh), graph in zip(
                self._artifact_targets_locked(),
                reloaded,
                strict=True,
            ):
                mesh_counts = self._mesh_counts(graph)
                artifact_records[target.name] = {
                    "sha256": self._file_sha256(target),
                    "requested_include_mesh": include_mesh,
                    "has_mesh": mesh_counts is not None,
                    "mesh_vertices": 0 if mesh_counts is None else mesh_counts[0],
                    "mesh_faces": 0 if mesh_counts is None else mesh_counts[1],
                    "object_count": len(self._object_signature(graph)),
                }
            bound_entities = {
                record[2]
                for record in next(iter(final_signatures))
                if record[2] and record[3]
            }
            verified_entities = (
                set(self._applied_entities)
                if verified_entity_ids is None
                else set(verified_entity_ids)
            )
            if not verified_entities.issubset(bound_entities):
                raise RuntimeError("verified DSG entities are missing from artifacts")
            operation_count = (
                len(self._applied)
                if verified_operation_count is None
                else int(verified_operation_count)
            )
            manifest = {
                "schema": "daaam.semantic_dsg_commit.v1",
                "committed_ns": time.time_ns(),
                "artifacts": artifact_records,
                "object_count": len(next(iter(final_signatures))),
                "verified_entity_count": len(verified_entities),
                "verified_operation_count": operation_count,
            }
            manifest_temporary = manifest_path.with_name(
                f".{manifest_path.stem}.{token}.tmp{manifest_path.suffix}"
            )
            temporary_paths.append(manifest_temporary)
            manifest_temporary.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            json.loads(manifest_temporary.read_text())
            manifest_temporary.replace(manifest_path)
            manifest_sha256 = self._file_sha256(manifest_path)

            self._commit_manifest_path = manifest_path
            self._commit_manifest_sha256 = manifest_sha256
            self._verified_artifacts = [
                str(target) for target, _ in self._artifact_targets_locked()
            ]
            self._verified_entities = len(verified_entities)
            self._verified_operations = operation_count
            return reloaded
        finally:
            for temporary in temporary_paths:
                temporary.unlink(missing_ok=True)

    def _drain_pending_locked(self) -> None:
        if (
            not self._pending
            or self._attached_path is None
            or not bool(
                getattr(self.service, "scene_graph_is_set", False)
            )
        ):
            return

        # Deliver state, not stale intermediate events: every operation for one
        # entity converges to MapMemory's current effective label.
        updates_by_entity: dict[str, list[tuple[str, DeliveredSemanticCorrection]]] = {}
        for operation_id, update in self._pending.items():
            updates_by_entity.setdefault(update.correction.entity_id, []).append(
                (operation_id, update)
            )

        staged: list[tuple[str, int, str, list[str]]] = []
        try:
            for entity_id, entity_updates in updates_by_entity.items():
                semantic_id = self._entity_to_semantic_id.get(entity_id)
                if semantic_id is None:
                    continue
                operation_ids = [operation_id for operation_id, _ in entity_updates]
                representative = entity_updates[-1][1]
                effective_label = self._effective_label_locked(entity_id)
                correction = ObjectAnnotation(
                    semantic_id=semantic_id,
                    semantic_label=effective_label,
                    confidence=min(1.0, representative.correction.confidence),
                    timestamp=representative.correction.sensor_time_ns / 1.0e9,
                    entity_id=entity_id,
                    request_id=representative.correction.operation_id,
                    sensor_time_ns=representative.correction.sensor_time_ns,
                    map_revision=representative.correction.map_revision,
                    source_model=representative.correction.source,
                )
                snapshot = self._entity_snapshot_locked(entity_id)
                ensure_node = getattr(self.service, "ensure_object_node", None)
                if snapshot is None or not callable(ensure_node):
                    continue
                try:
                    ensured = bool(
                        ensure_node(
                            semantic_id=semantic_id,
                            entity_id=entity_id,
                            position_m=snapshot.get("position_m"),
                            dimensions_m=snapshot.get("dimensions_m"),
                            sensor_time_ns=representative.correction.sensor_time_ns,
                        )
                    )
                except (RuntimeError, ValueError) as error:
                    rendered = repr(error)
                    if not self._errors or self._errors[-1] != rendered:
                        self._errors.append(rendered)
                    continue
                if not ensured:
                    continue

                add_correction = getattr(self.service, "add_correction", None)
                if callable(add_correction):
                    add_correction(correction)
                else:
                    self.service.store_correction(correction)
                staged.append(
                    (entity_id, semantic_id, effective_label, operation_ids)
                )

            apply_corrections = getattr(self.service, "apply_corrections", None)
            if callable(apply_corrections):
                apply_corrections()

            live_graph = self.service.scene_graph
            verified_entries = tuple(
                (entity_id, semantic_id, effective_label, operation_ids)
                for entity_id, semantic_id, effective_label, operation_ids in staged
                if live_graph is not None
                and self._graph_has_update(
                    live_graph,
                    semantic_id,
                    effective_label,
                    entity_id,
                )
            )
            expected_updates = tuple(entry[:3] for entry in verified_entries)
            verified_operation_ids = {
                operation_id
                for _, _, _, operation_ids in verified_entries
                for operation_id in operation_ids
            }
            verified_entity_ids = {
                entity_id for entity_id, _, _, _ in verified_entries
            }
            verification_graphs = self._persist_and_reload_locked(
                expected_updates,
                verified_entity_ids=(
                    self._applied_entities | verified_entity_ids
                ),
                verified_operation_count=len(
                    self._applied | verified_operation_ids
                ),
            )
            for entity_id, semantic_id, effective_label, operation_ids in staged:
                if not verification_graphs or not all(
                    self._graph_has_update(
                        graph,
                        semantic_id,
                        effective_label,
                        entity_id,
                    )
                    for graph in verification_graphs
                ):
                    continue
                self._applied.update(operation_ids)
                self._applied_entities.add(entity_id)
                for operation_id in operation_ids:
                    self._pending.pop(operation_id, None)
        except Exception as error:  # pragma: no cover - depends on Spark I/O
            rendered = repr(error)
            if not self._errors or self._errors[-1] != rendered:
                self._errors.append(rendered)

    def attach_saved_graph(self, path: Path | str) -> dict[str, Any]:
        requested_path = Path(path).resolve()
        if not requested_path.is_file():
            raise FileNotFoundError(requested_path)
        from spark_dsg import DynamicSceneGraph

        if requested_path.name == "dsg_with_mesh.json":
            graph_path = requested_path.with_name("dsg.json")
            with_mesh_path = requested_path
        else:
            graph_path = requested_path
            with_mesh_path = requested_path.with_name("dsg_with_mesh.json")
        source_path = with_mesh_path if with_mesh_path.is_file() else requested_path
        graph = DynamicSceneGraph.load(str(source_path))
        with self._lock:
            self._attached_path = graph_path
            self._attached_with_mesh_path = (
                with_mesh_path if with_mesh_path.is_file() else None
            )
            self._source_mesh_counts = self._mesh_counts(graph)
            self.service.set_scene_graph(graph)
            had_pending = bool(self._pending)
            self._drain_pending_locked()
            if not had_pending:
                self._persist_and_reload_locked(
                    verified_entity_ids=set(self._applied_entities),
                    verified_operation_count=len(self._applied),
                )
        return self.stats()

    def persist(self) -> None:
        with self._lock:
            if self._attached_path is None or not self.service.scene_graph_is_set:
                return
            had_pending = bool(self._pending)
            self._drain_pending_locked()
            if not had_pending:
                self._persist_and_reload_locked(
                    verified_entity_ids=set(self._applied_entities),
                    verified_operation_count=len(self._applied),
                )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            commit_valid = self._commit_is_valid_locked()
            unmapped = sum(
                update.correction.entity_id not in self._entity_to_semantic_id
                for update in self._pending.values()
            )
            durable_applied = len(self._applied) if commit_valid else 0
            durable_pending = len(self._pending) + (
                0 if commit_valid else len(self._applied)
            )
            return {
                "mapped_entities": len(self._entity_to_semantic_id),
                "applied": durable_applied,
                "pending": durable_pending,
                "unmapped": int(unmapped),
                "graph_attached": commit_valid,
                "commit_valid": commit_valid,
                "commit_manifest_path": (
                    None
                    if self._commit_manifest_path is None
                    else str(self._commit_manifest_path)
                ),
                "commit_manifest_sha256": self._commit_manifest_sha256,
                "verified_artifacts": (
                    list(self._verified_artifacts) if commit_valid else []
                ),
                "verified_entities": (
                    self._verified_entities if commit_valid else 0
                ),
                "verified_operations": (
                    self._verified_operations if commit_valid else 0
                ),
                "errors": list(self._errors),
            }


class RealtimeSemanticAdapter:
    """Run FastSAM/BotSort on a bounded side branch and DAM in a subprocess."""

    def __init__(
        self,
        pipeline_config: Any,
        memory: MapMemory,
        *,
        session_id: str,
        output_dir: Path | str,
        config: RealtimeSemanticConfig = RealtimeSemanticConfig(),
        segmentation_service: Any = None,
        tracking_service: Any = None,
        grounding_service: Any = None,
        dsg_sink: Optional[HydraDsgSemanticSink] = None,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.memory = memory
        self.session_id = session_id
        self.output_dir = Path(output_dir).resolve()
        self.config = config
        if segmentation_service is None:
            from daaam.segmentation import SegmentationService

            segmentation_service = SegmentationService(pipeline_config.segmentation)
        if tracking_service is None:
            from daaam.tracking import TrackingService

            tracking_service = TrackingService(pipeline_config.tracking)
        if grounding_service is None and config.grounding_enabled:
            from daaam.grounding import GroundingService

            grounding_service = GroundingService(pipeline_config.workers)
        if dsg_sink is None:
            from daaam.scene_graph.services import SceneGraphService

            scene_service = SceneGraphService(
                Path(pipeline_config.semantic_config_path),
                Path(pipeline_config.labelspace_colors_path),
                defer_dsg_processing=False,
                enable_background_objects=False,
            )
            dsg_sink = HydraDsgSemanticSink(
                scene_service,
                entity_lookup=memory.get_entity,
            )
        self.segmentation_service = segmentation_service
        self.tracking_service = tracking_service
        self.grounding_service = grounding_service
        self.dsg_sink = dsg_sink
        self.gpu_coordinator = SharedGpuCoordinator(
            lock_path=config.gpu_lock_path,
            activity_path=config.gpu_activity_path,
        )
        self._mp_context = mp.get_context("spawn")
        self.query_queue: mp.Queue = self._mp_context.Queue(
            maxsize=config.prompt_queue_capacity
        )
        self.correction_queue: mp.Queue = self._mp_context.Queue(
            maxsize=config.correction_queue_capacity
        )
        self.processor = VersionedCorrectionProcessor(memory, dsg_sink)
        self._correction_thread: Optional[threading.Thread] = None
        self._correction_stop = threading.Event()
        self._correction_drained = threading.Event()
        self._correction_drain_token: Optional[str] = None
        self._lock = threading.RLock()
        self._last_segmentation_time_ns: Optional[int] = None
        self._semantic_ids_by_track: dict[int, int] = {}
        self._semantic_id_by_entity: dict[str, int] = {}
        self._entity_by_semantic_id: dict[int, str] = {}
        self._context_by_semantic_id: dict[int, tuple[int, int]] = {}
        self._observations_by_entity: dict[str, int] = {}
        self._prompted_entity_revisions: set[tuple[str, int]] = set()
        self._next_semantic_id = 1
        self._label_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._propagation_audit: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._track_mask_states: OrderedDict[int, _TrackMaskState] = OrderedDict()
        self._started = False
        self._processor_start_attempted = False
        self._grounding_start_attempted = False
        self._correction_thread_start_attempted = False
        self._stats: dict[str, Any] = {
            "frames": 0,
            "segmentation_calls": 0,
            "segmentation_empty": 0,
            "segmentation_failures": 0,
            "detections": 0,
            "tracking_calls": 0,
            "tracking_failures": 0,
            "tracked_instances": 0,
            "entity_semantic_merges": 0,
            "entity_semantic_reassignments": 0,
            "label_frames_cached": 0,
            "propagation_frames": 0,
            "propagation_frames_with_labels": 0,
            "propagation_instances": 0,
            "propagation_bbox_warps": 0,
            "propagation_carry_forwards": 0,
            "propagation_expired": 0,
            "propagation_state_evictions": 0,
            "propagation_audit_evictions": 0,
            "propagation_overlap_pixels": 0,
            "prompts_submitted": 0,
            "prompt_queue_full": 0,
            "corrections_received": 0,
            "corrections_submitted": 0,
            "corrections_skipped": 0,
            "label_cache_hits": 0,
            "label_cache_misses": 0,
            "cleanup_errors": [],
            "segmentation_service_ms": [],
            "tracking_service_ms": [],
        }

    def start(self) -> None:
        if self._started:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.gpu_coordinator.touch_activity()
            with self.gpu_coordinator.lease():
                self.segmentation_service.warmup()
                self.tracking_service.warmup()
            self._processor_start_attempted = True
            self.processor.start()
            if self.config.grounding_enabled:
                if self.grounding_service is None:
                    raise RuntimeError("grounding service is required in DAM mode")
                self._grounding_start_attempted = True
                self.grounding_service.start(
                    self.query_queue,
                    self.correction_queue,
                    self.pipeline_config,
                    output_dir=str(self.output_dir),
                    color_map=getattr(self.dsg_sink.service, "color_map", None),
                    log_dir=str(self.output_dir / "logs"),
                )
                self._correction_stop.clear()
                self._correction_drained.clear()
                self._correction_drain_token = None
                self._correction_thread = threading.Thread(
                    target=self._correction_loop,
                    daemon=True,
                    name="realtime-dam-corrections",
                )
                self._correction_thread_start_attempted = True
                self._correction_thread.start()
            self._started = True
        except BaseException:
            # Startup is transactional. Cleanup failures remain auditable but
            # must never replace the model/service exception that triggered it.
            try:
                self.stop(timeout_s=5.0, drain=False)
            except BaseException:
                pass
            raise

    @staticmethod
    def _source_payload(
        envelope: RealtimeEnvelope,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        payload = envelope.payload
        source = getattr(payload, "source", None)
        if source is None:
            raise ValueError("semantic sidecar requires a depth-frame source")
        rgb = np.asarray(payload.rgb_image)
        depth = np.asarray(payload.depth_m, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[:2] != depth.shape:
            raise ValueError("semantic RGB/depth dimensions do not match")
        return source, rgb, depth

    def _segmentation_due(self, sensor_time_ns: int) -> bool:
        period_ns = int(round(1.0e9 / self.config.segmentation_rate_hz))
        return (
            self._last_segmentation_time_ns is None
            or sensor_time_ns - self._last_segmentation_time_ns >= period_ns
        )

    def _semantic_id(self, track_id: int) -> int:
        semantic_id = self._semantic_ids_by_track.get(track_id)
        if semantic_id is None:
            semantic_id = self._next_semantic_id
            self._next_semantic_id += 1
            self._semantic_ids_by_track[track_id] = semantic_id
        return semantic_id

    def _canonical_entity_semantic_id(
        self,
        state: _TrackMaskState,
        entity_id: str,
    ) -> int:
        """Make the stable MapMemory entity, not a transient track, authoritative."""

        existing = self._semantic_id_by_entity.get(entity_id)
        if existing is not None:
            if existing != state.semantic_id:
                with self._lock:
                    self._stats["entity_semantic_merges"] += 1
            semantic_id = existing
        else:
            semantic_id = state.semantic_id
            owner = self._entity_by_semantic_id.get(semantic_id)
            if owner is not None and owner != entity_id:
                semantic_id = self._next_semantic_id
                self._next_semantic_id += 1
                with self._lock:
                    self._stats["entity_semantic_reassignments"] += 1
            self._semantic_id_by_entity[entity_id] = semantic_id
        state.semantic_id = semantic_id
        self._semantic_ids_by_track[state.track_id] = semantic_id
        self._entity_by_semantic_id[semantic_id] = entity_id
        return semantic_id

    @staticmethod
    def _track_rows(tracks: Any) -> list[np.ndarray]:
        array = np.asarray(tracks)
        if array.size == 0:
            return []
        if array.ndim != 2 or array.shape[1] < 8:
            raise ValueError("semantic tracker output must be an Mx8 array")
        rows = [
            row for row in array if np.all(np.isfinite(row[:8])) and int(row[4]) > 0
        ]
        rows.sort(
            key=lambda row: (
                int(row[4]),
                -float(row[5]),
                *(float(value) for value in row[:4]),
            )
        )
        # A tracker should emit one row per ID. Deterministically retain the
        # highest-confidence row if an adapter violates that contract.
        unique: dict[int, np.ndarray] = {}
        for row in rows:
            unique.setdefault(int(row[4]), row)
        return list(unique.values())

    @staticmethod
    def _bbox_bounds(
        bbox: np.ndarray, shape: tuple[int, int]
    ) -> Optional[tuple[int, int, int, int]]:
        values = np.asarray(bbox, dtype=np.float64).reshape(-1)
        if values.size < 4 or not np.all(np.isfinite(values[:4])):
            return None
        height, width = shape
        x1 = int(np.clip(np.floor(min(values[0], values[2])), 0, width))
        y1 = int(np.clip(np.floor(min(values[1], values[3])), 0, height))
        x2 = int(np.clip(np.ceil(max(values[0], values[2])), 0, width))
        y2 = int(np.clip(np.ceil(max(values[1], values[3])), 0, height))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @classmethod
    def _warp_mask_to_bbox(
        cls,
        mask: np.ndarray,
        source_bbox: np.ndarray,
        target_bbox: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Deterministically translate/scale a mask with a BotSort bbox."""

        source = cls._bbox_bounds(source_bbox, mask.shape)
        target = cls._bbox_bounds(target_bbox, mask.shape)
        if source is None or target is None:
            return None
        sx1, sy1, sx2, sy2 = source
        tx1, ty1, tx2, ty2 = target
        crop = np.asarray(mask[sy1:sy2, sx1:sx2], dtype=np.uint8)
        if crop.size == 0 or not np.any(crop):
            return None
        resized = cv2.resize(
            crop,
            (tx2 - tx1, ty2 - ty1),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        propagated = np.zeros(mask.shape, dtype=bool)
        propagated[ty1:ty2, tx1:tx2] = resized
        return propagated if np.any(propagated) else None

    def _segmented_track_states(
        self,
        rows: list[np.ndarray],
        masks: list[np.ndarray],
        image_shape: tuple[int, int],
        sensor_time_ns: int,
    ) -> list[tuple[_TrackMaskState, str]]:
        previous = self._track_mask_states
        candidates: list[_TrackMaskState] = []
        for row in rows:
            track_id = int(row[4])
            mask_index = int(row[7])
            if not 0 <= mask_index < len(masks):
                continue
            mask = np.asarray(masks[mask_index], dtype=bool)
            if mask.shape != image_shape:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image_shape[1], image_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if not np.any(mask):
                continue
            old_state = previous.get(track_id)
            candidates.append(
                _TrackMaskState(
                    track_id=track_id,
                    semantic_id=self._semantic_id(track_id),
                    mask=mask.copy(),
                    bbox=np.asarray(row[:4], dtype=np.float64).copy(),
                    source_sensor_time_ns=sensor_time_ns,
                    last_sensor_time_ns=sensor_time_ns,
                    propagation_steps=0,
                    confidence=float(np.clip(row[5], 0.0, 1.0)),
                    entity_id=None if old_state is None else old_state.entity_id,
                )
            )

        capacity = self.config.propagation_track_capacity
        retained = sorted(
            candidates,
            key=lambda state: (-state.confidence, state.track_id),
        )[:capacity]
        evicted = len(candidates) - len(retained)
        retained.sort(key=lambda state: state.track_id)
        self._track_mask_states = OrderedDict(
            (state.track_id, state) for state in retained
        )
        if evicted:
            with self._lock:
                self._stats["propagation_state_evictions"] += evicted
        return [(state, "segmentation") for state in retained]

    def _propagated_track_states(
        self,
        rows: list[np.ndarray],
        sensor_time_ns: int,
    ) -> list[tuple[_TrackMaskState, str]]:
        rows_by_track = {int(row[4]): row for row in rows}
        propagated: list[tuple[_TrackMaskState, str]] = []
        expired = 0
        for track_id, old_state in list(self._track_mask_states.items()):
            if old_state.propagation_steps >= self.config.propagation_max_frames:
                self._track_mask_states.pop(track_id, None)
                expired += 1
                continue
            row = rows_by_track.get(track_id)
            mask = None
            method = "carry_forward"
            bbox = old_state.bbox
            confidence = old_state.confidence
            if row is not None:
                mask = self._warp_mask_to_bbox(
                    old_state.mask,
                    old_state.bbox,
                    np.asarray(row[:4], dtype=np.float64),
                )
                if mask is not None:
                    method = "botsort_bbox_warp"
                    bbox = np.asarray(row[:4], dtype=np.float64).copy()
                confidence = float(np.clip(row[5], 0.0, 1.0))
            if mask is None:
                mask = old_state.mask.copy()
            state = _TrackMaskState(
                track_id=track_id,
                semantic_id=old_state.semantic_id,
                mask=mask,
                bbox=np.asarray(bbox, dtype=np.float64).copy(),
                source_sensor_time_ns=old_state.source_sensor_time_ns,
                last_sensor_time_ns=sensor_time_ns,
                propagation_steps=old_state.propagation_steps + 1,
                confidence=confidence,
                entity_id=old_state.entity_id,
            )
            self._track_mask_states[track_id] = state
            propagated.append((state, method))
        with self._lock:
            self._stats["propagation_frames"] += 1
            self._stats["propagation_frames_with_labels"] += int(bool(propagated))
            self._stats["propagation_instances"] += len(propagated)
            self._stats["propagation_bbox_warps"] += sum(
                method == "botsort_bbox_warp" for _, method in propagated
            )
            self._stats["propagation_carry_forwards"] += sum(
                method == "carry_forward" for _, method in propagated
            )
            self._stats["propagation_expired"] += expired
        return propagated

    @staticmethod
    def _mask_digest(mask: np.ndarray) -> str:
        shape = np.asarray(mask.shape, dtype="<u4").tobytes()
        packed = np.packbits(np.asarray(mask, dtype=np.uint8), axis=None).tobytes()
        return hashlib.sha256(shape + packed).hexdigest()

    def _cache_label_frame(
        self,
        sensor_time_ns: int,
        map_revision: int,
        label_image: np.ndarray,
        *,
        segmentation_due: bool,
        overlap_pixels: int,
        tracks: list[dict[str, Any]],
    ) -> None:
        audit = {
            "sensor_time_ns": sensor_time_ns,
            "map_revision": map_revision,
            "mode": "segmentation" if segmentation_due else "propagation",
            "overlap_pixels": int(overlap_pixels),
            "tracks": tracks,
        }
        with self._lock:
            self._label_cache[sensor_time_ns] = label_image.copy()
            self._label_cache.move_to_end(sensor_time_ns)
            self._propagation_audit[sensor_time_ns] = audit
            self._propagation_audit.move_to_end(sensor_time_ns)
            while len(self._label_cache) > self.config.label_cache_frames:
                evicted_time_ns, _ = self._label_cache.popitem(last=False)
                if self._propagation_audit.pop(evicted_time_ns, None) is not None:
                    self._stats["propagation_audit_evictions"] += 1
            while len(self._propagation_audit) > self.config.label_cache_frames:
                self._propagation_audit.popitem(last=False)
                self._stats["propagation_audit_evictions"] += 1
            self._stats["label_frames_cached"] += 1
            self._stats["propagation_overlap_pixels"] += overlap_pixels

    def handle(self, envelope: RealtimeEnvelope) -> None:
        source, rgb, depth = self._source_payload(envelope)
        sensor_time_ns = envelope.key.sensor_time_ns
        self.gpu_coordinator.touch_activity()
        segmentation_due = self._segmentation_due(sensor_time_ns)
        detections = np.empty((0, 6), dtype=np.float32)
        masks: list[np.ndarray] = []
        with self.gpu_coordinator.lease():
            if segmentation_due:
                started = time.perf_counter()
                try:
                    detections, masks = self.segmentation_service.segment_checked(rgb)
                except Exception:
                    with self._lock:
                        self._stats["segmentation_failures"] += 1
                    raise
                finally:
                    with self._lock:
                        self._stats["segmentation_service_ms"].append(
                            (time.perf_counter() - started) * 1000.0
                        )
                self._last_segmentation_time_ns = sensor_time_ns
                with self._lock:
                    self._stats["segmentation_calls"] += 1
                    self._stats["detections"] += len(detections)
                    self._stats["segmentation_empty"] += int(len(detections) == 0)

            tracking_started = time.perf_counter()
            try:
                tracks = self.tracking_service.update(detections, rgb)
            except Exception:
                with self._lock:
                    self._stats["tracking_failures"] += 1
                raise
            finally:
                with self._lock:
                    self._stats["tracking_service_ms"].append(
                        (time.perf_counter() - tracking_started) * 1000.0
                    )
        with self._lock:
            self._stats["tracking_calls"] += 1
            self._stats["frames"] += 1

        rows = self._track_rows(tracks)
        if segmentation_due:
            candidates = self._segmented_track_states(
                rows,
                masks,
                depth.shape,
                sensor_time_ns,
            )
        else:
            candidates = self._propagated_track_states(rows, sensor_time_ns)

        label_image = np.zeros(depth.shape, dtype=np.int32)
        prompt_tracks: list[Track] = []
        object_labels: dict[int, int] = {}
        entity_ids: dict[int, str] = {}
        audit_tracks: list[dict[str, Any]] = []
        overlap_pixels = 0
        for state, propagation_method in candidates:
            track_id = state.track_id
            semantic_id = state.semantic_id
            mask = state.mask
            overlap_pixels += int(np.count_nonzero(mask & (label_image != 0)))
            label_image[mask & (label_image == 0)] = semantic_id
            valid_depth = depth[mask & np.isfinite(depth) & (depth > 0.0)]
            depth_valid = valid_depth.size >= max(1, int(mask.sum() * 0.25))
            audit_track = {
                "track_id": track_id,
                "semantic_id": semantic_id,
                "source_sensor_time_ns": state.source_sensor_time_ns,
                "age_ns": sensor_time_ns - state.source_sensor_time_ns,
                "propagation_steps": state.propagation_steps,
                "method": propagation_method,
                "mask_pixels": int(np.count_nonzero(mask)),
                "mask_sha256": self._mask_digest(mask),
                "depth_valid": bool(depth_valid),
                "entity_id": state.entity_id,
            }
            audit_tracks.append(audit_track)
            if not depth_valid:
                continue
            median_depth = float(np.median(valid_depth))
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            u = float(np.median(xs))
            v = float(np.median(ys))
            intrinsics = np.asarray(source.intrinsics, dtype=np.float64)
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            camera_point = np.asarray(
                [
                    (u - cx) * median_depth / fx,
                    (v - cy) * median_depth / fy,
                    median_depth,
                ]
            )
            world_point = (
                np.asarray(source.world_T_camera, dtype=np.float64)
                @ np.r_[camera_point, 1.0]
            )[:3]
            width_m = max(0.05, (xs.max() - xs.min() + 1) * median_depth / fx)
            height_m = max(0.05, (ys.max() - ys.min() + 1) * median_depth / fy)
            entity_id, _ = self.memory.observe_entity(
                self.session_id,
                f"botsort:{track_id}",
                world_point,
                sensor_time_ns=sensor_time_ns,
                semantic_label="unknown",
                dimensions_m=np.asarray([width_m, 0.2, height_m]),
                confidence=state.confidence,
            )
            provisional_semantic_id = semantic_id
            semantic_id = self._canonical_entity_semantic_id(state, entity_id)
            if semantic_id != provisional_semantic_id:
                label_image[
                    mask & (label_image == provisional_semantic_id)
                ] = semantic_id
                audit_track["semantic_id"] = semantic_id
            state.entity_id = entity_id
            audit_track["entity_id"] = entity_id
            self.dsg_sink.register_entity(entity_id, semantic_id)
            self._context_by_semantic_id[semantic_id] = (
                sensor_time_ns,
                envelope.key.map_revision,
            )
            if segmentation_due:
                observations = self._observations_by_entity.get(entity_id, 0) + 1
                self._observations_by_entity[entity_id] = observations
                track = Track.from_mask(
                    id=track_id,
                    mask=mask,
                    bbox=np.asarray(state.bbox, dtype=np.int32),
                    epsilon_factor=(
                        self.pipeline_config.segmentation.polygon_epsilon_factor
                    ),
                    depth_valid=True,
                    median_depth=median_depth,
                )
                prompt_tracks.append(track)
                object_labels[track_id] = semantic_id
                entity_ids[track_id] = entity_id
            with self._lock:
                self._stats["tracked_instances"] += 1

        self._cache_label_frame(
            sensor_time_ns,
            envelope.key.map_revision,
            label_image,
            segmentation_due=segmentation_due,
            overlap_pixels=overlap_pixels,
            tracks=audit_tracks,
        )
        if segmentation_due:
            self._enqueue_prompt(
                sensor_time_ns,
                envelope.key.map_revision,
                int(getattr(source, "frame_index", -1)),
                rgb,
                prompt_tracks,
                object_labels,
                entity_ids,
            )

    def _enqueue_prompt(
        self,
        sensor_time_ns: int,
        map_revision: int,
        frame_id: int,
        rgb: np.ndarray,
        tracks: list[Track],
        object_labels: dict[int, int],
        entity_ids: dict[int, str],
    ) -> None:
        if not self.config.grounding_enabled:
            return
        selected = [
            track
            for track in tracks
            if self._observations_by_entity.get(entity_ids[track.id], 0)
            >= self.config.minimum_observations
            and (entity_ids[track.id], map_revision)
            not in self._prompted_entity_revisions
        ]
        if not selected:
            return
        selected_track_ids = {track.id for track in selected}
        selected_labels = {
            track_id: semantic_id
            for track_id, semantic_id in object_labels.items()
            if track_id in selected_track_ids
        }
        selected_entities = {
            track_id: entity_id
            for track_id, entity_id in entity_ids.items()
            if track_id in selected_track_ids
        }
        request_material = "|".join(
            [
                self.session_id,
                str(sensor_time_ns),
                str(map_revision),
                *(selected_entities[key] for key in sorted(selected_entities)),
            ]
        )
        request_id = hashlib.sha256(request_material.encode()).hexdigest()
        record = PromptRecord(
            frame=rgb.copy(),
            tracks=selected,
            object_labels=selected_labels,
            frame_id=frame_id,
            timestamp=sensor_time_ns / 1.0e9,
            sensor_time_ns=sensor_time_ns,
            map_revision=map_revision,
            request_id=request_id,
            entity_ids=selected_entities,
        )
        try:
            self.query_queue.put_nowait(record)
        except queue.Full:
            with self._lock:
                self._stats["prompt_queue_full"] += 1
            return
        self._prompted_entity_revisions.update(
            (entity_id, map_revision) for entity_id in selected_entities.values()
        )
        with self._lock:
            self._stats["prompts_submitted"] += 1

    def label_image_for(self, sensor_time_ns: int) -> Optional[np.ndarray]:
        with self._lock:
            labels = self._label_cache.get(sensor_time_ns)
            if labels is None:
                self._stats["label_cache_misses"] += 1
                return None
            self._stats["label_cache_hits"] += 1
            return labels.copy()

    def label_audit_for(self, sensor_time_ns: int) -> Optional[dict[str, Any]]:
        """Return exact-time provenance for one provisional Hydra label image."""

        with self._lock:
            audit = self._propagation_audit.get(sensor_time_ns)
            if audit is None:
                return None
            return {
                **audit,
                "tracks": [dict(track) for track in audit["tracks"]],
            }

    def _correction_loop(self) -> None:
        while not self._correction_stop.is_set():
            try:
                annotation = self.correction_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(annotation, GroundingCorrectionsDrained):
                self._correction_drain_token = annotation.token
                self._correction_drained.set()
                return
            self.process_annotation(annotation)

    def process_annotation(self, annotation: Any) -> None:
        """Convert one DAM result into a stable, idempotent map correction."""
        with self._lock:
            self._stats["corrections_received"] += 1
        if not isinstance(annotation, ObjectAnnotation):
            with self._lock:
                self._stats["corrections_skipped"] += 1
            return
        label = " ".join(annotation.semantic_label.split()).strip()
        entity_id = annotation.entity_id or self._entity_by_semantic_id.get(
            annotation.semantic_id
        )
        context = self._context_by_semantic_id.get(annotation.semantic_id)
        sensor_time_ns = annotation.sensor_time_ns or (
            context[0] if context is not None else 0
        )
        map_revision = (
            annotation.map_revision
            if annotation.map_revision is not None
            else (context[1] if context is not None else self.memory.current_revision)
        )
        if (
            not entity_id
            or sensor_time_ns <= 0
            or not label
            or label.casefold() == "unknown"
        ):
            with self._lock:
                self._stats["corrections_skipped"] += 1
            return
        source = f"dam:{annotation.source_model or 'unknown'}"
        operation_material = "|".join(
            [
                source,
                annotation.request_id or "",
                entity_id,
                str(map_revision),
                label.casefold(),
            ]
        )
        operation_id = hashlib.sha256(operation_material.encode()).hexdigest()
        confidence = float(annotation.confidence or self.config.automatic_confidence)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        self.processor.submit(
            SemanticCorrection(
                operation_id=operation_id,
                entity_id=entity_id,
                sensor_time_ns=sensor_time_ns,
                map_revision=map_revision,
                label=label,
                confidence=confidence,
                source=source,
            )
        )
        with self._lock:
            self._stats["corrections_submitted"] += 1

    def attach_hydra_dsg(self, path: Path | str) -> dict[str, int]:
        return self.dsg_sink.attach_saved_graph(path)

    @staticmethod
    def _latency(values: list[float]) -> dict[str, Optional[float]]:
        if not values:
            return {"samples": 0, "p50": None, "p95": None, "p99": None, "max": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "samples": int(array.size),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "max": float(np.max(array)),
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            values = dict(self._stats)
            values["cleanup_errors"] = list(values["cleanup_errors"])
            segmentation_ms = list(values.pop("segmentation_service_ms"))
            tracking_ms = list(values.pop("tracking_service_ms"))
            recent_audit = [
                {
                    **audit,
                    "tracks": [dict(track) for track in audit["tracks"]],
                }
                for audit in self._propagation_audit.values()
            ]
            active_track_ids = list(self._track_mask_states)
            cached_sensor_times_ns = list(self._label_cache)
        values["latency"] = {
            "segmentation_ms": self._latency(segmentation_ms),
            "tracking_ms": self._latency(tracking_ms),
        }
        values["propagation"] = {
            "maximum_frames": self.config.propagation_max_frames,
            "track_capacity": self.config.propagation_track_capacity,
            "active_track_ids": active_track_ids,
            "active_tracks": len(active_track_ids),
            "history_capacity": self.config.label_cache_frames,
            "cached_sensor_times_ns": cached_sensor_times_ns,
            "recent_audit": recent_audit,
        }
        values["memory"] = self.processor.stats()
        values["dsg"] = self.dsg_sink.stats()
        if self.grounding_service is not None:
            values["grounding_workers"] = self.grounding_service.get_worker_health()
        return values

    def stop(self, timeout_s: float = 30.0, *, drain: bool = True) -> dict[str, Any]:
        if timeout_s <= 0.0:
            raise ValueError("semantic stop timeout must be positive")
        deadline = time.monotonic() + timeout_s
        cleanup_errors: list[BaseException] = []
        grounding_shutdown: Optional[dict[str, Any]] = None
        grounding_stop_succeeded = False
        if self._grounding_start_attempted and self.grounding_service is not None:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("semantic drain budget expired before DAM stop")
                grounding_shutdown = self.grounding_service.stop(
                    timeout_s=remaining,
                    drain=drain,
                )
                grounding_stop_succeeded = True
            except BaseException as error:
                cleanup_errors.append(error)
            finally:
                self._grounding_start_attempted = False

        thread = self._correction_thread
        if self._correction_thread_start_attempted or thread is not None:
            try:
                if drain and grounding_stop_succeeded:
                    drain_token = (
                        grounding_shutdown.get("drain_token")
                        if isinstance(grounding_shutdown, dict)
                        else None
                    )
                    if not drain_token:
                        raise RuntimeError(
                            "grounding service returned no correction drain token"
                        )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0 or not self._correction_drained.wait(remaining):
                        raise TimeoutError(
                            "DAM correction consumer did not observe drain completion"
                        )
                    if self._correction_drain_token != drain_token:
                        raise RuntimeError(
                            "DAM correction drain token did not match grounding workers"
                        )
            except BaseException as error:
                cleanup_errors.append(error)
            finally:
                self._correction_stop.set()
            try:
                if thread is not None and thread.is_alive():
                    thread.join(max(0.0, deadline - time.monotonic()))
                    if thread.is_alive():
                        raise RuntimeError("DAM correction consumer did not stop")
            except BaseException as error:
                cleanup_errors.append(error)
            finally:
                self._correction_thread_start_attempted = False
                self._correction_thread = None

        if self._processor_start_attempted:
            try:
                self.processor.stop(
                    timeout_s=max(0.1, deadline - time.monotonic()), drain=drain
                )
            except BaseException as error:
                cleanup_errors.append(error)
            finally:
                self._processor_start_attempted = False
        try:
            self.dsg_sink.persist()
        except BaseException as error:
            cleanup_errors.append(error)
        finally:
            self._started = False

        if cleanup_errors:
            with self._lock:
                self._stats["cleanup_errors"].extend(
                    repr(error) for error in cleanup_errors
                )
            raise cleanup_errors[0]
        return self.stats()
