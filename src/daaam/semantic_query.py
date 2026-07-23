"""Reusable semantic retrieval and grounded-answer primitives for a DSG.

The retrieval path is entirely local: a multilingual SentenceTransformer
encodes a text query and compares it with vectors persisted on object nodes in
``dsg_updated.json``.  A sidecar manifest binds those vectors to the exact model
used to create them; matching vector dimensions alone are not sufficient.
The optional grounded-answer path uses an OpenAI-compatible text endpoint only
after those local candidates have been selected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from hmac import compare_digest
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Optional, Protocol

import numpy as np
import spark_dsg as sdsg
import torch

from daaam.query_evidence import (
    QueryEvidence,
    QueryEvidenceError,
    load_query_evidence,
)
from daaam.query_index import QueryIndexError, load_query_index
from daaam.utils.embedding import SentenceEmbeddingHandler


DEFAULT_LLM_BASE_URL = (
    "https://llm-g3o8d3j71xbf6prc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_LLM_MODEL = "qwen3.7-plus"
DEFAULT_SENTENCE_MODEL = (
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
# A deliberately conservative first deployment value.  It prevents unrelated
# low-cosine queries from always returning an object while leaving class-level
# queries with several valid instances usable.  Deployments should calibrate it
# against labelled positive/negative queries for their own scene vocabulary.
DEFAULT_MIN_SIMILARITY = 0.55
DEFAULT_MIN_MARGIN = 0.0


class SemanticQueryError(RuntimeError):
    """Base exception for a semantic query operation."""


class LLMUnavailableError(SemanticQueryError):
    """Raised when the optional LLM path has no configured credential."""


class LLMRequestError(SemanticQueryError):
    """Raised when an OpenAI-compatible completion request fails."""


class TextEncoder(Protocol):
    """Minimal interface shared by the production and test text encoders."""

    def extract_text_embeddings(
        self, texts: list[str], *, show_progress: bool
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class ObjectRecord:
    """One queryable scene-graph object and its sentence embedding."""

    node_id: str
    semantic_label: int
    description: str
    position: np.ndarray
    first_observed: Optional[float]
    last_observed: Optional[float]
    embedding: np.ndarray
    entity_id: str = ""
    dimensions: Optional[np.ndarray] = None
    first_observed_ns: Optional[int] = None
    last_observed_ns: Optional[int] = None
    geometry_status: str = "mesh_bound"
    geometry_confidence: Optional[float] = None
    source: str = "dsg"


@dataclass(frozen=True)
class GroundedAnswer:
    """A model answer with the exact local retrieval evidence that supported it."""

    retrieval_query: str
    matches: list[tuple[float, ObjectRecord]]
    answer: str
    model: str
    found: bool
    rejection_reason: Optional[str]
    top_score: float
    top1_margin: Optional[float]


@dataclass(frozen=True)
class RetrievalDecision:
    """Ranked matches plus the explicit open-set retrieval decision."""

    found: bool
    matches: list[tuple[float, ObjectRecord]]
    rejection_reason: Optional[str]
    top_score: float
    top1_margin: Optional[float]
    min_similarity: float
    min_margin: float


def _normalized(values: Any) -> Optional[np.ndarray]:
    """Convert a persisted embedding to a finite, unit-length vector."""
    embedding = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if embedding.size == 0 or not np.isfinite(embedding).all() or norm == 0.0:
        return None
    return embedding / norm


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_query_manifest(dsg_path: Path | str) -> Optional[dict[str, Any]]:
    """Load and checksum-validate ``<dsg stem>.manifest.json`` when present."""
    path = Path(dsg_path).expanduser().resolve()
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticQueryError(
            f"Failed to read query-map manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise SemanticQueryError(
            f"Query-map manifest must be a JSON object: {manifest_path}"
        )
    schema_version = manifest.get("schema_version")
    if schema_version is not None and schema_version != 1:
        raise SemanticQueryError(
            f"Unsupported query-map manifest schema_version={schema_version!r}: "
            f"{manifest_path}"
        )

    expected_digest = manifest.get("dsg_sha256")
    if expected_digest is not None:
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise SemanticQueryError(
                f"Query-map manifest has an invalid dsg_sha256: {manifest_path}"
            )
        try:
            actual_digest = _sha256(path)
        except OSError as exc:
            raise SemanticQueryError(f"Failed to hash query map: {path}") from exc
        if not compare_digest(actual_digest, expected_digest.lower()):
            raise SemanticQueryError(
                f"Query-map manifest checksum does not match DSG JSON: {manifest_path}"
            )
    return manifest


def _validate_thresholds(min_similarity: float, min_margin: float) -> None:
    if not np.isfinite(min_similarity) or not -1.0 <= min_similarity <= 1.0:
        raise SemanticQueryError("min_similarity must be finite and in [-1, 1].")
    if not np.isfinite(min_margin) or not 0.0 <= min_margin <= 2.0:
        raise SemanticQueryError("min_margin must be finite and in [0, 2].")


def load_object_records(dsg_path: Path | str) -> list[ObjectRecord]:
    """Load descriptions and Sentence-T5 vectors attached to DSG object nodes."""
    graph = sdsg.DynamicSceneGraph.load(str(dsg_path))
    records: list[ObjectRecord] = []

    for node in graph.get_layer(sdsg.DsgLayers.OBJECTS).nodes:
        metadata = dict(node.attributes.metadata.get() or {})
        description = str(metadata.get("description", "")).strip()
        embedding = _normalized(metadata.get("sentence_embedding_feature", []))
        if not description or embedding is None:
            continue

        history = dict(metadata.get("temporal_history") or {})
        first_ns = metadata.get("first_observed_ns", history.get("first_observed_ns"))
        last_ns = metadata.get("last_observed_ns", history.get("last_observed_ns"))
        time_origin_ns = metadata.get("time_origin_ns", history.get("time_origin_ns"))
        first_observed = history.get("first_observed")
        last_observed = history.get("last_observed")
        if first_observed is None and first_ns is not None and time_origin_ns is not None:
            first_observed = (int(first_ns) - int(time_origin_ns)) / 1.0e9
        if last_observed is None and last_ns is not None and time_origin_ns is not None:
            last_observed = (int(last_ns) - int(time_origin_ns)) / 1.0e9
        try:
            mesh = node.attributes.mesh()
            has_object_mesh = bool(mesh is not None and mesh.num_vertices() > 0)
        except (AttributeError, RuntimeError, TypeError):
            has_object_mesh = False
        dimensions = np.asarray(
            node.attributes.bounding_box.dimensions, dtype=float
        ).reshape(-1).copy()
        records.append(
            ObjectRecord(
                node_id=str(node.id),
                semantic_label=int(node.attributes.semantic_label),
                description=description,
                # spark_dsg exposes an Eigen-backed view; copy it before the
                # temporary node wrapper is released on the next iteration.
                position=np.asarray(node.attributes.position, dtype=float).reshape(-1).copy(),
                first_observed=first_observed,
                last_observed=last_observed,
                embedding=embedding,
                entity_id=str(metadata.get("entity_id", "")).strip(),
                dimensions=dimensions if dimensions.size == 3 else None,
                first_observed_ns=None if first_ns is None else int(first_ns),
                last_observed_ns=None if last_ns is None else int(last_ns),
                geometry_status="mesh_bound" if has_object_mesh else "spatial_only",
                geometry_confidence=(
                    None
                    if metadata.get("geometry_confidence") is None
                    else float(metadata["geometry_confidence"])
                ),
                source="dsg",
            )
        )

    if not records:
        raise SemanticQueryError(
            "No object node contains both 'description' and "
            "'sentence_embedding_feature'. Use dsg_updated.json produced by "
            "scripts/prepare_zed_query_dsg.py."
        )
    return records


def _optional_vector(values: Any, *, field: str, record_id: str) -> Optional[np.ndarray]:
    if values is None:
        return None
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != 3 or not np.isfinite(vector).all():
        raise SemanticQueryError(f"{field} is invalid for {record_id}")
    return vector.copy()


def load_sidecar_records(
    dsg_path: Path | str, manifest: Optional[dict[str, Any]]
) -> list[ObjectRecord]:
    """Load lower-confidence DAM records excluded from the authoritative DSG."""

    try:
        raw_records = load_query_index(dsg_path, manifest=manifest)
    except QueryIndexError as exc:
        raise SemanticQueryError(str(exc)) from exc
    records: list[ObjectRecord] = []
    for item in raw_records:
        record_id = item["record_id"]
        embedding = _normalized(item.get("embedding", []))
        if embedding is None:
            raise SemanticQueryError(f"semantic query record has no embedding: {record_id}")
        position = _optional_vector(
            item.get("position_m"), field="position_m", record_id=record_id
        )
        dimensions = _optional_vector(
            item.get("dimensions_m"), field="dimensions_m", record_id=record_id
        )
        geometry_status = item["geometry_status"]
        if geometry_status == "spatial_only" and position is None:
            raise SemanticQueryError(
                f"spatial-only query record has no position: {record_id}"
            )
        confidence = item.get("geometry_confidence")
        if confidence is not None:
            confidence = float(confidence)
            if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise SemanticQueryError(
                    f"geometry_confidence is invalid for {record_id}"
                )
        records.append(
            ObjectRecord(
                node_id=record_id,
                semantic_label=int(item["semantic_label"]),
                description=item["description"],
                position=(np.asarray([], dtype=float) if position is None else position),
                first_observed=(
                    None
                    if item.get("first_observed_s") is None
                    else float(item["first_observed_s"])
                ),
                last_observed=(
                    None
                    if item.get("last_observed_s") is None
                    else float(item["last_observed_s"])
                ),
                embedding=embedding,
                entity_id=item["entity_id"],
                dimensions=dimensions,
                first_observed_ns=(
                    None
                    if item.get("first_observed_ns") is None
                    else int(item["first_observed_ns"])
                ),
                last_observed_ns=(
                    None
                    if item.get("last_observed_ns") is None
                    else int(item["last_observed_ns"])
                ),
                geometry_status=geometry_status,
                geometry_confidence=confidence,
                source=str(item.get("source") or "semantic_sidecar"),
            )
        )
    return records


def _timestamp(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.3f}s"


def results_to_dicts(
    results: Iterable[tuple[float, ObjectRecord]],
) -> list[dict[str, Any]]:
    """Make retrieval records JSON serializable for CLI and HTTP callers."""
    return [
        {
            "rank": rank,
            "score": score,
            "node_id": record.node_id,
            "semantic_label": record.semantic_label,
            "description": record.description,
            "entity_id": record.entity_id or None,
            "position_m": (
                record.position[:3].tolist() if record.position.size >= 3 else None
            ),
            "dimensions_m": (
                None if record.dimensions is None else record.dimensions[:3].tolist()
            ),
            "first_observed_s": record.first_observed,
            "last_observed_s": record.last_observed,
            "geometry_status": record.geometry_status,
            "geometry_confidence": record.geometry_confidence,
            "source": record.source,
        }
        for rank, (score, record) in enumerate(results, start=1)
    ]


def build_evidence(results: Iterable[tuple[float, ObjectRecord]]) -> str:
    """Serialize retrieved nodes as a bounded, citation-ready LLM context."""
    evidence: list[str] = []
    for score, record in results:
        position = (
            "unavailable"
            if record.position.size < 3
            else f"({record.position[0]:.3f}, {record.position[1]:.3f}, "
            f"{record.position[2]:.3f})"
        )
        evidence.append(
            "\n".join(
                [
                    f"[{record.node_id}] similarity={score:.4f}",
                    f"description: {record.description}",
                    f"position_m: {position}",
                    f"geometry_status: {record.geometry_status}",
                    f"observed_s: {_timestamp(record.first_observed)} to {_timestamp(record.last_observed)}",
                ]
            )
        )
    return "\n\n".join(evidence)


class SemanticQueryEngine:
    """Preloaded local retriever with an optional grounded-answer capability."""

    def __init__(
        self,
        dsg_path: Path | str,
        *,
        sentence_model_name: str = DEFAULT_SENTENCE_MODEL,
        llm_base_url: str = DEFAULT_LLM_BASE_URL,
        llm_model: str = DEFAULT_LLM_MODEL,
        api_key_env: str = "DAAAM_KEY",
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        min_margin: float = DEFAULT_MIN_MARGIN,
        require_verified_model: bool = True,
        encoder: Optional[TextEncoder] = None,
    ) -> None:
        _validate_thresholds(min_similarity, min_margin)
        self.dsg_path = Path(dsg_path).expanduser().resolve()
        self.map_manifest = load_query_manifest(self.dsg_path)
        embedding_manifest = (
            self.map_manifest.get("embedding", {}) if self.map_manifest else {}
        )
        if not isinstance(embedding_manifest, dict):
            raise SemanticQueryError("Query-map manifest 'embedding' must be an object.")
        declared_model = embedding_manifest.get("model")
        has_checksum = bool(self.map_manifest and self.map_manifest.get("dsg_sha256"))
        if require_verified_model and (not declared_model or not has_checksum):
            raise SemanticQueryError(
                "The query map must have a checksum-bound sidecar manifest declaring "
                "embedding.model. Rebuild it with scripts/prepare_query_dsg_embeddings.py "
                "or explicitly allow unverified embeddings."
            )
        if declared_model and str(declared_model) != sentence_model_name:
            raise SemanticQueryError(
                f"Map embeddings were generated by {declared_model!r}, but the configured "
                f"query model is {sentence_model_name!r}. Re-embed the map; equal vector "
                "dimensions do not make models compatible."
            )
        dsg_records = load_object_records(self.dsg_path)
        sidecar_records = load_sidecar_records(self.dsg_path, self.map_manifest)
        self.records = dsg_records + sidecar_records
        if not self.records:
            raise SemanticQueryError("The query map contains no described objects.")
        for field, values in (
            ("node/record ID", [record.node_id for record in self.records]),
            (
                "semantic label",
                [record.semantic_label for record in self.records],
            ),
        ):
            if len(values) != len(set(values)):
                raise SemanticQueryError(f"Query records contain duplicate {field}s.")
        entity_ids = [record.entity_id for record in self.records if record.entity_id]
        if len(entity_ids) != len(set(entity_ids)):
            raise SemanticQueryError("Query records contain duplicate entity IDs.")
        try:
            self.evidence_by_node, self.evidence_by_id = load_query_evidence(
                self.dsg_path
            )
        except QueryEvidenceError as exc:
            raise SemanticQueryError(str(exc)) from exc
        records_by_node = {record.node_id: record for record in self.records}
        unknown_evidence_nodes = sorted(set(self.evidence_by_node) - set(records_by_node))
        if unknown_evidence_nodes:
            raise SemanticQueryError(
                "Query evidence references non-queryable DSG nodes: "
                + ", ".join(unknown_evidence_nodes)
            )
        for node_id, evidence in self.evidence_by_node.items():
            record = records_by_node[node_id]
            if evidence.semantic_label != record.semantic_label:
                raise SemanticQueryError(
                    f"Query evidence semantic label does not match {node_id}."
                )
        self.records = [
            replace(
                record,
                position=np.asarray(
                    self.evidence_by_node[record.node_id].geometry_position_m,
                    dtype=float,
                ),
                dimensions=np.asarray(
                    self.evidence_by_node[record.node_id].geometry_dimensions_m,
                    dtype=float,
                ),
                source="fastsam_masked_rgbd",
            )
            if record.geometry_status != "mesh_bound"
            and record.node_id in self.evidence_by_node
            and self.evidence_by_node[record.node_id].geometry_position_m is not None
            and self.evidence_by_node[record.node_id].geometry_dimensions_m is not None
            else record
            for record in self.records
        ]
        dimensions = {int(record.embedding.size) for record in self.records}
        if len(dimensions) != 1:
            raise SemanticQueryError(
                f"DSG contains inconsistent embedding dimensions: {sorted(dimensions)}"
            )
        self.embedding_dim = dimensions.pop()
        declared_dimension = embedding_manifest.get("dimension")
        if declared_dimension is not None:
            try:
                compatible_dimension = int(declared_dimension) == self.embedding_dim
            except (TypeError, ValueError) as exc:
                raise SemanticQueryError(
                    f"Map manifest has invalid embedding.dimension={declared_dimension!r}."
                ) from exc
            if not compatible_dimension:
                raise SemanticQueryError(
                    f"Map manifest declares {declared_dimension}-dimensional embeddings, "
                    f"but DSG objects contain {self.embedding_dim}."
                )
        declared_count = (
            self.map_manifest.get("queryable_objects") if self.map_manifest else None
        )
        if declared_count is not None:
            try:
                compatible_count = int(declared_count) == len(self.records)
            except (TypeError, ValueError) as exc:
                raise SemanticQueryError(
                    f"Map manifest has invalid queryable_objects={declared_count!r}."
                ) from exc
            if not compatible_count:
                raise SemanticQueryError(
                    f"Map manifest declares {declared_count} queryable objects, but DSG "
                    f"contains {len(self.records)}."
                )
        declared_dsg_count = (
            self.map_manifest.get("dsg_queryable_objects") if self.map_manifest else None
        )
        if declared_dsg_count is not None and int(declared_dsg_count) != len(dsg_records):
            raise SemanticQueryError(
                "Map manifest dsg_queryable_objects does not match the DSG records."
            )
        self.geometry_counts = {
            status: sum(record.geometry_status == status for record in self.records)
            for status in ("mesh_bound", "spatial_only", "image_only")
        }
        declared_geometry_counts = (
            self.map_manifest.get("geometry_counts") if self.map_manifest else None
        )
        if declared_geometry_counts is not None:
            if not isinstance(declared_geometry_counts, dict) or any(
                int(declared_geometry_counts.get(status, 0)) != count
                for status, count in self.geometry_counts.items()
            ):
                raise SemanticQueryError(
                    "Map manifest geometry_counts do not match the query records."
                )
        self.sentence_model_name = sentence_model_name
        self.embedding_model_verified = bool(declared_model and has_checksum)
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.api_key_env = api_key_env
        self.min_similarity = float(min_similarity)
        self.min_margin = float(min_margin)
        self._embeddings = np.stack([record.embedding for record in self.records])
        if encoder is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            encoder = SentenceEmbeddingHandler(
                model_name=sentence_model_name, device=device
            )
            self.encoder_device = device
        else:
            self.encoder_device = "custom"
        self.encoder = encoder
        # A single encoder instance is shared by REST requests. Serialize its
        # forward passes to avoid concurrent GPU access from FastAPI workers.
        self._encoder_lock = Lock()

    @property
    def llm_available(self) -> bool:
        return bool(os.getenv(self.api_key_env))

    def evidence_for_node(self, node_id: str) -> Optional[QueryEvidence]:
        """Return checksum-verified FastSAM evidence for one queryable node."""

        return self.evidence_by_node.get(node_id)

    def retrieve_with_decision(
        self,
        query: str,
        top_k: int = 5,
        *,
        min_similarity: Optional[float] = None,
        min_margin: Optional[float] = None,
        require_mesh: bool = False,
    ) -> RetrievalDecision:
        """Rank objects and reject low-confidence or ambiguous open-set queries."""
        normalized_query = query.strip()
        if not normalized_query:
            raise SemanticQueryError("Query text must not be empty.")
        if top_k < 1:
            raise SemanticQueryError("top_k must be at least 1.")
        selected_min_similarity = (
            self.min_similarity if min_similarity is None else float(min_similarity)
        )
        selected_min_margin = self.min_margin if min_margin is None else float(min_margin)
        _validate_thresholds(selected_min_similarity, selected_min_margin)

        with self._encoder_lock:
            text_embedding = self.encoder.extract_text_embeddings(
                [normalized_query], show_progress=False
            )[0]
        text_embedding = _normalized(text_embedding)
        if text_embedding is None:
            raise SemanticQueryError("The query embedding is empty or invalid.")

        if text_embedding.size != self.embedding_dim:
            raise SemanticQueryError(
                f"Query embedding dimension {text_embedding.size} does not match DSG "
                f"dimension {self.embedding_dim}; use the model declared by the map."
            )
        scores = self._embeddings @ text_embedding
        eligible = np.asarray(
            [
                not require_mesh or record.geometry_status == "mesh_bound"
                for record in self.records
            ],
            dtype=bool,
        )
        eligible_indices = np.flatnonzero(eligible)
        if eligible_indices.size == 0:
            raise SemanticQueryError("The query map contains no mesh-bound objects.")
        indices = eligible_indices[
            np.argsort(-scores[eligible_indices], kind="stable")
        ]
        top_score = float(scores[indices[0]])
        top1_margin = (
            float(top_score - scores[indices[1]]) if len(indices) > 1 else None
        )

        rejection_reason: Optional[str] = None
        if top_score < selected_min_similarity:
            rejection_reason = "below_min_similarity"
        elif (
            selected_min_margin > 0.0
            and top1_margin is not None
            and top1_margin < selected_min_margin
        ):
            rejection_reason = "below_min_margin"

        found = rejection_reason is None
        count = min(top_k, len(indices))
        matches = (
            [
                (float(scores[index]), self.records[int(index)])
                for index in indices[:count]
            ]
            if found
            else []
        )
        return RetrievalDecision(
            found=found,
            matches=matches,
            rejection_reason=rejection_reason,
            top_score=top_score,
            top1_margin=top1_margin,
            min_similarity=selected_min_similarity,
            min_margin=selected_min_margin,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        min_similarity: Optional[float] = None,
        min_margin: Optional[float] = None,
        require_mesh: bool = False,
    ) -> list[tuple[float, ObjectRecord]]:
        """Return accepted matches, or an empty list when the query is rejected."""
        return self.retrieve_with_decision(
            query,
            top_k,
            min_similarity=min_similarity,
            min_margin=min_margin,
            require_mesh=require_mesh,
        ).matches

    def _api_key(self) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise LLMUnavailableError(
                f"{self.api_key_env} is not set on the query-service process."
            )
        return api_key

    def _chat_completion(self, system_prompt: str, user_prompt: str, model: str) -> str:
        """Make one bounded request to the configured compatible endpoint."""
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self._api_key(),
                base_url=self.llm_base_url,
                timeout=60.0,
                max_retries=1,
            )
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except LLMUnavailableError:
            raise
        except Exception as exc:
            raise LLMRequestError(
                "OpenAI-compatible completion request failed. Check the service, "
                "model name, and server-side credential."
            ) from exc

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise LLMRequestError("OpenAI-compatible endpoint returned an empty response.")
        return content.strip()

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        model: Optional[str] = None,
        *,
        min_similarity: Optional[float] = None,
        min_margin: Optional[float] = None,
        require_mesh: bool = False,
    ) -> GroundedAnswer:
        """Rewrite, retrieve locally, then answer only from retrieved evidence."""
        normalized_question = question.strip()
        if not normalized_question:
            raise SemanticQueryError("Question text must not be empty.")
        selected_model = model or self.llm_model
        retrieval_query = self._chat_completion(
            system_prompt=(
                "Convert the user's request into one short English visual description "
                "for semantic object retrieval in a robot map. Preserve distinctive "
                "attributes such as color, material, shape, and object type. Output only "
                "the retrieval phrase; do not answer the question or add punctuation."
            ),
            user_prompt=normalized_question,
            model=selected_model,
        )
        decision = self.retrieve_with_decision(
            retrieval_query,
            top_k,
            min_similarity=min_similarity,
            min_margin=min_margin,
            require_mesh=require_mesh,
        )
        answer = self._chat_completion(
            system_prompt=(
                "Answer the user's question using only the supplied robot-map evidence. "
                "Respond in the language of the question. Cite every factual answer with "
                "the corresponding node ID in square brackets, such as [O(52)]. If the "
                "evidence is insufficient, say so plainly; do not invent objects, "
                "locations, or observations."
            ),
            user_prompt=(
                f"Question:\n{normalized_question}\n\n"
                f"Retrieval found an acceptable match: {decision.found}\n"
                f"Rejection reason: {decision.rejection_reason or 'none'}\n\n"
                f"Retrieved evidence:\n{build_evidence(decision.matches)}"
            ),
            model=selected_model,
        )
        return GroundedAnswer(
            retrieval_query=retrieval_query,
            matches=decision.matches,
            answer=answer,
            model=selected_model,
            found=decision.found,
            rejection_reason=decision.rejection_reason,
            top_score=decision.top_score,
            top1_margin=decision.top1_margin,
        )
