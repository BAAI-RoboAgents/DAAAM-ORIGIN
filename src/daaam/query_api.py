"""FastAPI application exposing local and API-backed DSG semantic queries."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from daaam.semantic_query import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    LLMRequestError,
    LLMUnavailableError,
    SemanticQueryEngine,
    SemanticQueryError,
    results_to_dicts,
)


class QueryRequest(BaseModel):
    """Input shared by the local retrieval and grounded-answer endpoints."""

    query: str = Field(..., description="Natural-language object description or question.")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum retrieved objects.")

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
    position_m: list[float]
    first_observed_s: Optional[float]
    last_observed_s: Optional[float]


class RetrievalResponse(BaseModel):
    query: str
    matches: list[RetrievalMatch]


class AnswerResponse(BaseModel):
    question: str
    retrieval_query: str
    model: str
    matches: list[RetrievalMatch]
    answer: str


class HealthResponse(BaseModel):
    status: str
    queryable_objects: int
    sentence_model: str
    llm_enabled: bool
    default_llm_model: str


def _as_matches(results) -> list[RetrievalMatch]:
    return [RetrievalMatch(**item) for item in results_to_dicts(results)]


def create_app(
    dsg_path: Path | str,
    *,
    sentence_model_name: str = "sentence-transformers/sentence-t5-large",
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    api_key_env: str = "DAAAM_KEY",
) -> FastAPI:
    """Create a preloaded REST service for one query-ready scene graph."""
    engine = SemanticQueryEngine(
        dsg_path,
        sentence_model_name=sentence_model_name,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        api_key_env=api_key_env,
    )
    app = FastAPI(
        title="DAAAM Semantic Query API",
        version="1.0.0",
        description=(
            "Local Sentence-T5 retrieval plus optional API-backed answers that are "
            "grounded in retrieved DSG nodes. API credentials stay server-side."
        ),
    )
    app.state.semantic_query_engine = engine

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            queryable_objects=len(engine.records),
            sentence_model=engine.sentence_model_name,
            llm_enabled=engine.llm_available,
            default_llm_model=engine.llm_model,
        )

    @app.post("/v1/query/retrieve", response_model=RetrievalResponse, tags=["query"])
    def retrieve(request: QueryRequest) -> RetrievalResponse:
        """Run local-only Sentence-T5 cosine-similarity retrieval."""
        try:
            matches = engine.retrieve(request.query, request.top_k)
        except SemanticQueryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RetrievalResponse(query=request.query, matches=_as_matches(matches))

    @app.post("/v1/query/ask", response_model=AnswerResponse, tags=["query"])
    def ask(request: AnswerRequest) -> AnswerResponse:
        """Answer from local matches using the server-side compatible-model key."""
        try:
            result = engine.answer_question(request.query, request.top_k, request.model)
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
            matches=_as_matches(result.matches),
            answer=result.answer,
        )

    return app
