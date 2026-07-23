"""FastAPI application exposing local and API-backed DSG semantic queries."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from daaam.semantic_query import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_SENTENCE_MODEL,
    LLMRequestError,
    LLMUnavailableError,
    RetrievalDecision,
    SemanticQueryEngine,
    SemanticQueryError,
    results_to_dicts,
)


class QueryRequest(BaseModel):
    """Input shared by the local retrieval and grounded-answer endpoints."""

    query: str = Field(..., description="Natural-language object description or question.")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum retrieved objects.")
    require_mesh: bool = Field(
        default=False,
        description=(
            "Restrict retrieval to real Hydra object meshes. By default the API "
            "also returns DAM-described MapMemory entities with spatial evidence."
        ),
    )
    min_similarity: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional per-query cosine floor; otherwise use the server default.",
    )
    min_margin: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description=(
            "Optional minimum top-1 minus top-2 score margin. Zero disables "
            "ambiguity rejection."
        ),
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class AnswerRequest(QueryRequest):
    """Input for a grounded answer; model is optional server-side selection."""

    model: Optional[str] = Field(
        default=None,
        description="Optional OpenAI-compatible model override. Defaults to the server model.",
    )

    @field_validator("model")
    @classmethod
    def model_must_not_be_blank(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("model must not be blank")
        return value


class RetrievalMatch(BaseModel):
    rank: int
    score: float
    node_id: str
    semantic_label: int
    description: str
    entity_id: Optional[str]
    position_m: Optional[list[float]]
    dimensions_m: Optional[list[float]]
    first_observed_s: Optional[float]
    last_observed_s: Optional[float]
    geometry_status: str
    geometry_confidence: Optional[float]
    source: str


class EvidenceResponse(BaseModel):
    evidence_id: str
    image_url: str
    frame_index: int
    sensor_time_ns: int
    observed_s: Optional[float]
    bbox_xyxy: list[int]
    mask_pixels: int
    mask_source: str
    source_image_sha256: str
    annotated_image_sha256: str


class RetrievalResponse(BaseModel):
    query: str
    found: bool
    rejection_reason: Optional[str]
    top_score: float
    top1_margin: Optional[float]
    min_similarity: float
    min_margin: float
    matches: list[RetrievalMatch]
    top1_evidence: Optional[EvidenceResponse]


class AnswerResponse(BaseModel):
    question: str
    retrieval_query: str
    model: str
    found: bool
    rejection_reason: Optional[str]
    top_score: float
    top1_margin: Optional[float]
    matches: list[RetrievalMatch]
    top1_evidence: Optional[EvidenceResponse]
    answer: str


class HealthResponse(BaseModel):
    status: str
    queryable_objects: int
    mesh_bound_objects: int
    spatial_only_objects: int
    image_only_objects: int
    evidence_available_objects: int
    embedding_dimension: int
    sentence_model: str
    embedding_model_verified: bool
    encoder_device: str
    default_min_similarity: float
    default_min_margin: float
    llm_enabled: bool
    default_llm_model: str


def _as_matches(results) -> list[RetrievalMatch]:
    return [RetrievalMatch(**item) for item in results_to_dicts(results)]


def _top1_evidence(
    engine: SemanticQueryEngine, results
) -> Optional[EvidenceResponse]:
    results = list(results)
    if not results:
        return None
    evidence = engine.evidence_for_node(results[0][1].node_id)
    if evidence is None:
        return None
    return EvidenceResponse(
        evidence_id=evidence.evidence_id,
        image_url=f"/v1/evidence/{evidence.evidence_id}.png",
        frame_index=evidence.frame_index,
        sensor_time_ns=evidence.sensor_time_ns,
        observed_s=evidence.observed_s,
        bbox_xyxy=list(evidence.bbox_xyxy),
        mask_pixels=evidence.mask_pixels,
        mask_source=evidence.mask_source,
        source_image_sha256=evidence.source_image_sha256,
        annotated_image_sha256=evidence.image_sha256,
    )


def _retrieval_response(
    query: str, decision: RetrievalDecision, engine: SemanticQueryEngine
) -> RetrievalResponse:
    return RetrievalResponse(
        query=query,
        found=decision.found,
        rejection_reason=decision.rejection_reason,
        top_score=decision.top_score,
        top1_margin=decision.top1_margin,
        min_similarity=decision.min_similarity,
        min_margin=decision.min_margin,
        matches=_as_matches(decision.matches),
        top1_evidence=_top1_evidence(engine, decision.matches),
    )


def create_app(
    dsg_path: Path | str,
    *,
    sentence_model_name: str = DEFAULT_SENTENCE_MODEL,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    api_key_env: str = "DAAAM_KEY",
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    min_margin: float = DEFAULT_MIN_MARGIN,
    require_verified_model: bool = True,
    engine: Optional[SemanticQueryEngine] = None,
) -> FastAPI:
    """Create a preloaded REST service for one query-ready scene graph."""
    if engine is None:
        engine = SemanticQueryEngine(
            dsg_path,
            sentence_model_name=sentence_model_name,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            api_key_env=api_key_env,
            min_similarity=min_similarity,
            min_margin=min_margin,
            require_verified_model=require_verified_model,
        )
    app = FastAPI(
        title="DAAAM Semantic Query API",
        version="1.3.0",
        description=(
            "Local multilingual retrieval with open-set rejection, plus optional "
            "API-backed answers grounded in retrieved DSG nodes. API credentials "
            "stay server-side."
        ),
    )
    app.state.semantic_query_engine = engine

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            queryable_objects=len(engine.records),
            mesh_bound_objects=engine.geometry_counts["mesh_bound"],
            spatial_only_objects=engine.geometry_counts["spatial_only"],
            image_only_objects=engine.geometry_counts["image_only"],
            evidence_available_objects=len(engine.evidence_by_node),
            embedding_dimension=engine.embedding_dim,
            sentence_model=engine.sentence_model_name,
            embedding_model_verified=engine.embedding_model_verified,
            encoder_device=engine.encoder_device,
            default_min_similarity=engine.min_similarity,
            default_min_margin=engine.min_margin,
            llm_enabled=engine.llm_available,
            default_llm_model=engine.llm_model,
        )

    @app.post("/v1/query/retrieve", response_model=RetrievalResponse, tags=["query"])
    def retrieve(request: QueryRequest) -> RetrievalResponse:
        """Run local multilingual retrieval with explicit open-set rejection."""
        try:
            decision = engine.retrieve_with_decision(
                request.query,
                request.top_k,
                min_similarity=request.min_similarity,
                min_margin=request.min_margin,
                require_mesh=request.require_mesh,
            )
        except SemanticQueryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _retrieval_response(request.query, decision, engine)

    @app.get(
        "/v1/evidence/{evidence_id}.png",
        response_class=FileResponse,
        tags=["evidence"],
    )
    def evidence_image(evidence_id: str) -> FileResponse:
        """Serve one checksum-verified original-frame FastSAM overlay."""

        evidence = engine.evidence_by_id.get(evidence_id)
        if evidence is None:
            raise HTTPException(status_code=404, detail="evidence image not found")
        return FileResponse(
            evidence.image_path,
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "ETag": f'"{evidence.image_sha256}"',
                "X-DAAAM-Mask-Source": evidence.mask_source,
            },
        )

    @app.post("/v1/query/ask", response_model=AnswerResponse, tags=["query"])
    def ask(request: AnswerRequest) -> AnswerResponse:
        """Answer from local matches using the server-side compatible-model key."""
        try:
            result = engine.answer_question(
                request.query,
                request.top_k,
                request.model,
                min_similarity=request.min_similarity,
                min_margin=request.min_margin,
                require_mesh=request.require_mesh,
            )
        except LLMUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LLMRequestError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except SemanticQueryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return AnswerResponse(
            question=request.query,
            retrieval_query=result.retrieval_query,
            model=result.model,
            found=result.found,
            rejection_reason=result.rejection_reason,
            top_score=result.top_score,
            top1_margin=result.top1_margin,
            matches=_as_matches(result.matches),
            top1_evidence=_top1_evidence(engine, result.matches),
            answer=result.answer,
        )

    return app
