#!/usr/bin/env python3
"""Serve a query-ready DAAAM DSG through a local HTTP REST API."""

from __future__ import annotations

import os
from pathlib import Path

import click
import uvicorn

from daaam.query_api import create_app
from daaam.semantic_query import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_SENTENCE_MODEL,
    SemanticQueryError,
)


@click.command()
@click.option(
    "--dsg",
    "dsg_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Query-ready dsg_updated.json to preload once at startup.",
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, type=click.IntRange(1, 65535), show_default=True)
@click.option(
    "--model",
    default=lambda: os.getenv("DAAAM_LLM_MODEL", DEFAULT_LLM_MODEL),
    show_default="DAAAM_LLM_MODEL or qwen3.7-plus",
)
@click.option(
    "--base-url",
    default=lambda: os.getenv("DAAAM_LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
    show_default="DAAAM_LLM_BASE_URL or configured endpoint",
)
@click.option("--api-key-env", default="DAAAM_KEY", show_default=True)
@click.option(
    "--sentence-model-name",
    default=DEFAULT_SENTENCE_MODEL,
    show_default=True,
    envvar="DAAAM_QUERY_SENTENCE_EMBEDDING_MODEL_NAME",
)
@click.option(
    "--min-similarity",
    type=click.FloatRange(-1.0, 1.0),
    default=DEFAULT_MIN_SIMILARITY,
    show_default=True,
    envvar="DAAAM_QUERY_MIN_SIMILARITY",
    help="Default cosine floor for found=true.",
)
@click.option(
    "--min-margin",
    type=click.FloatRange(0.0, 2.0),
    default=DEFAULT_MIN_MARGIN,
    show_default=True,
    envvar="DAAAM_QUERY_MIN_MARGIN",
    help="Default top-1/top-2 margin; zero disables ambiguity rejection.",
)
@click.option(
    "--allow-unverified-embeddings",
    is_flag=True,
    help="Allow a legacy map without a checksum-bound embedding model manifest.",
)
def main(
    dsg_path: Path,
    host: str,
    port: int,
    model: str,
    base_url: str,
    api_key_env: str,
    sentence_model_name: str,
    min_similarity: float,
    min_margin: float,
    allow_unverified_embeddings: bool,
) -> None:
    """Start one preloaded query service; do not use multiple GPU workers."""
    try:
        app = create_app(
            dsg_path,
            sentence_model_name=sentence_model_name,
            llm_base_url=base_url,
            llm_model=model,
            api_key_env=api_key_env,
            min_similarity=min_similarity,
            min_margin=min_margin,
            require_verified_model=not allow_unverified_embeddings,
        )
    except SemanticQueryError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Serving {len(app.state.semantic_query_engine.records)} queryable objects at "
        f"http://{host}:{port} (OpenAPI: /docs)"
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
