"""Reusable semantic retrieval and grounded-answer primitives for a DSG.

The retrieval path is entirely local: Sentence-T5 encodes a text query and
compares it with vectors persisted on object nodes in ``dsg_updated.json``.
The optional grounded-answer path uses an OpenAI-compatible text endpoint only
after those local candidates have been selected.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Optional

import numpy as np
import spark_dsg as sdsg
import torch

from daaam.utils.embedding import SentenceEmbeddingHandler


DEFAULT_LLM_BASE_URL = (
    "https://llm-g3o8d3j71xbf6prc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_LLM_MODEL = "qwen3.7-plus"


class SemanticQueryError(RuntimeError):
    """Base exception for a semantic query operation."""


class LLMUnavailableError(SemanticQueryError):
    """Raised when the optional LLM path has no configured credential."""


class LLMRequestError(SemanticQueryError):
    """Raised when an OpenAI-compatible completion request fails."""


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


@dataclass(frozen=True)
class GroundedAnswer:
    """A model answer with the exact local retrieval evidence that supported it."""

    retrieval_query: str
    matches: list[tuple[float, ObjectRecord]]
    answer: str
    model: str


def _normalized(values: Any) -> Optional[np.ndarray]:
    """Convert a persisted embedding to a finite, unit-length vector."""
    embedding = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if embedding.size == 0 or not np.isfinite(embedding).all() or norm == 0.0:
        return None
    return embedding / norm


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
        records.append(
            ObjectRecord(
                node_id=str(node.id),
                semantic_label=int(node.attributes.semantic_label),
                description=description,
                # spark_dsg exposes an Eigen-backed view; copy it before the
                # temporary node wrapper is released on the next iteration.
                position=np.asarray(node.attributes.position, dtype=float).reshape(-1).copy(),
                first_observed=history.get("first_observed"),
                last_observed=history.get("last_observed"),
                embedding=embedding,
            )
        )

    if not records:
        raise SemanticQueryError(
            "No object node contains both 'description' and "
            "'sentence_embedding_feature'. Use dsg_updated.json produced by "
            "scripts/prepare_zed_query_dsg.py."
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
            "position_m": record.position[:3].tolist(),
            "first_observed_s": record.first_observed,
            "last_observed_s": record.last_observed,
        }
        for rank, (score, record) in enumerate(results, start=1)
    ]


def build_evidence(results: Iterable[tuple[float, ObjectRecord]]) -> str:
    """Serialize retrieved nodes as a bounded, citation-ready LLM context."""
    evidence: list[str] = []
    for score, record in results:
        x, y, z = record.position[:3]
        evidence.append(
            "\n".join(
                [
                    f"[{record.node_id}] similarity={score:.4f}",
                    f"description: {record.description}",
                    f"position_m: ({x:.3f}, {y:.3f}, {z:.3f})",
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
        sentence_model_name: str = "sentence-transformers/sentence-t5-large",
        llm_base_url: str = DEFAULT_LLM_BASE_URL,
        llm_model: str = DEFAULT_LLM_MODEL,
        api_key_env: str = "DAAAM_KEY",
    ) -> None:
        self.dsg_path = Path(dsg_path)
        self.records = load_object_records(self.dsg_path)
        self.sentence_model_name = sentence_model_name
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.api_key_env = api_key_env
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceEmbeddingHandler(model_name=sentence_model_name, device=device)
        # A single encoder instance is shared by REST requests. Serialize its
        # forward passes to avoid concurrent GPU access from FastAPI workers.
        self._encoder_lock = Lock()

    @property
    def llm_available(self) -> bool:
        return bool(os.getenv(self.api_key_env))

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[float, ObjectRecord]]:
        """Rank objects by cosine similarity to a text description."""
        normalized_query = query.strip()
        if not normalized_query:
            raise SemanticQueryError("Query text must not be empty.")
        if top_k < 1:
            raise SemanticQueryError("top_k must be at least 1.")

        with self._encoder_lock:
            text_embedding = self.encoder.extract_text_embeddings(
                [normalized_query], show_progress=False
            )[0]
        text_embedding = _normalized(text_embedding)
        if text_embedding is None:
            raise SemanticQueryError("The query embedding is empty or invalid.")

        scored = [
            (float(record.embedding @ text_embedding), record)
            for record in self.records
            if record.embedding.shape == text_embedding.shape
        ]
        if not scored:
            raise SemanticQueryError(
                "No graph embeddings have the same dimension as the query model. "
                "Use the Sentence-T5 model that produced dsg_updated.json."
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

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

    def answer_question(self, question: str, top_k: int = 5, model: Optional[str] = None) -> GroundedAnswer:
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
        matches = self.retrieve(retrieval_query, top_k)
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
                f"Retrieved evidence:\n{build_evidence(matches)}"
            ),
            model=selected_model,
        )
        return GroundedAnswer(
            retrieval_query=retrieval_query,
            matches=matches,
            answer=answer,
            model=selected_model,
        )
