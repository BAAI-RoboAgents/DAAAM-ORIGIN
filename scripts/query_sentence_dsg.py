#!/usr/bin/env python3
"""CLI for the reusable DAAAM semantic-query engine.

Use this for ad-hoc local retrieval. External modules should use the HTTP API
started by ``scripts/serve_query_api.py`` instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import click

from daaam.semantic_query import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    LLMUnavailableError,
    SemanticQueryEngine,
    SemanticQueryError,
)


def _timestamp(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.3f}s"


def print_results(results) -> None:
    """Print retrieval results in a compact, copyable form."""
    if not results:
        click.echo("No compatible object embeddings found.")
        return

    for rank, (score, record) in enumerate(results, start=1):
        x, y, z = record.position[:3]
        click.echo(
            f"{rank:>2}. score={score:.4f}  node={record.node_id}  "
            f"label={record.semantic_label}  xyz=({x:.3f}, {y:.3f}, {z:.3f})"
        )
        click.echo(f"    {record.description}")
        click.echo(
            f"    observed: {_timestamp(record.first_observed)} "
            f"to {_timestamp(record.last_observed)}"
        )


@click.command()
@click.option(
    "--dsg",
    "dsg_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Query-ready dsg_updated.json.",
)
@click.option("--query", "query_text", type=str, default=None, help="Text to retrieve.")
@click.option("--top-k", type=click.IntRange(min=1), default=5, show_default=True)
@click.option(
    "--answer-with-llm",
    is_flag=True,
    help="Use the compatible API to rewrite the question and answer from retrieved evidence.",
)
@click.option(
    "--model",
    default=lambda: os.getenv("DAAAM_LLM_MODEL", DEFAULT_LLM_MODEL),
    show_default="DAAAM_LLM_MODEL or qwen3.7-plus",
    help="OpenAI-compatible model name.",
)
@click.option(
    "--base-url",
    default=lambda: os.getenv("DAAAM_LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
    show_default="DAAAM_LLM_BASE_URL or configured endpoint",
    help="OpenAI-compatible API base URL.",
)
@click.option(
    "--api-key-env",
    default="DAAAM_KEY",
    show_default=True,
    help="Environment-variable name containing the API key.",
)
@click.option(
    "--sentence-model-name",
    default="sentence-transformers/sentence-t5-large",
    show_default=True,
    envvar="DAAAM_QUERY_SENTENCE_EMBEDDING_MODEL_NAME",
    help="Must match the model used when creating dsg_updated.json.",
)
def main(
    dsg_path: Path,
    query_text: Optional[str],
    top_k: int,
    answer_with_llm: bool,
    model: str,
    base_url: str,
    api_key_env: str,
    sentence_model_name: str,
) -> None:
    """Run local retrieval, with optional API-backed grounded answers."""
    if answer_with_llm and query_text is None:
        raise click.UsageError("--answer-with-llm requires --query.")

    try:
        engine = SemanticQueryEngine(
            dsg_path,
            sentence_model_name=sentence_model_name,
            llm_base_url=base_url,
            llm_model=model,
            api_key_env=api_key_env,
        )
    except SemanticQueryError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Loaded {len(engine.records)} queryable objects from {dsg_path}")
    click.echo("Sentence-T5 is local; no LLM/API calls are made for retrieval-only queries.")

    def run_query(text: str) -> None:
        normalized_text = text.strip()
        if not normalized_text:
            return
        try:
            if not answer_with_llm:
                click.echo(f"\nQuery: {normalized_text}")
                print_results(engine.retrieve(normalized_text, top_k))
                return

            result = engine.answer_question(normalized_text, top_k=top_k)
            click.echo(f"Retrieval phrase: {result.retrieval_query}")
            print_results(result.matches)
            click.echo("\nGrounded answer:")
            click.echo(result.answer)
        except (SemanticQueryError, LLMUnavailableError) as exc:
            raise click.ClickException(str(exc)) from exc

    if query_text is not None:
        run_query(query_text)
        return

    click.echo("Enter an English description (empty line or Ctrl+C exits).")
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            click.echo()
            return
        if not text.strip():
            return
        run_query(text)


if __name__ == "__main__":
    main()
