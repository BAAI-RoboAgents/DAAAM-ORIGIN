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
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_SENTENCE_MODEL,
    LLMUnavailableError,
    RetrievalDecision,
    SemanticQueryEngine,
    SemanticQueryError,
)
from daaam.query_visualization import (
    QueryVisualizationError,
    write_query_visuals,
)


def _timestamp(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.3f}s"


def print_results(results) -> None:
    """Print retrieval results in a compact, copyable form."""
    if not results:
        click.echo("No accepted object match.")
        return

    for rank, (score, record) in enumerate(results, start=1):
        position = (
            "n/a"
            if record.position.size < 3
            else f"({record.position[0]:.3f}, {record.position[1]:.3f}, "
            f"{record.position[2]:.3f})"
        )
        click.echo(
            f"{rank:>2}. score={score:.4f}  node={record.node_id}  "
            f"label={record.semantic_label}  xyz={position}  "
            f"geometry={record.geometry_status}"
        )
        click.echo(f"    {record.description}")
        click.echo(
            f"    observed: {_timestamp(record.first_observed)} "
            f"to {_timestamp(record.last_observed)}"
        )


def print_top1_evidence(engine: SemanticQueryEngine, results) -> None:
    """Print the local annotated FastSAM image for an accepted top-1 result."""

    results = list(results)
    if not results:
        return
    evidence = engine.evidence_for_node(results[0][1].node_id)
    if evidence is None:
        click.echo("top1_evidence: unavailable")
        return
    click.echo(
        f"top1_evidence: {evidence.image_path} "
        f"(frame={evidence.frame_index}, t={_timestamp(evidence.observed_s)}, "
        f"source={evidence.mask_source})"
    )


def print_decision(
    decision: RetrievalDecision, engine: Optional[SemanticQueryEngine] = None
) -> None:
    """Print acceptance diagnostics without leaking rejected candidates."""
    margin = "n/a" if decision.top1_margin is None else f"{decision.top1_margin:.4f}"
    click.echo(
        f"found={str(decision.found).lower()}  top_score={decision.top_score:.4f}  "
        f"top1_margin={margin}  min_similarity={decision.min_similarity:.4f}  "
        f"min_margin={decision.min_margin:.4f}"
    )
    if decision.rejection_reason:
        click.echo(f"rejection_reason={decision.rejection_reason}")
    print_results(decision.matches)
    if engine is not None:
        print_top1_evidence(engine, decision.matches)


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
    default=DEFAULT_SENTENCE_MODEL,
    show_default=True,
    envvar="DAAAM_QUERY_SENTENCE_EMBEDDING_MODEL_NAME",
    help="Must match the model used when creating dsg_updated.json.",
)
@click.option(
    "--min-similarity",
    type=click.FloatRange(-1.0, 1.0),
    default=DEFAULT_MIN_SIMILARITY,
    show_default=True,
    envvar="DAAAM_QUERY_MIN_SIMILARITY",
)
@click.option(
    "--min-margin",
    type=click.FloatRange(0.0, 2.0),
    default=DEFAULT_MIN_MARGIN,
    show_default=True,
    envvar="DAAAM_QUERY_MIN_MARGIN",
    help="Minimum top-1/top-2 score gap; zero disables this rejection rule.",
)
@click.option(
    "--allow-unverified-embeddings",
    is_flag=True,
    help="Allow a legacy map without a checksum-bound embedding model manifest.",
)
@click.option(
    "--require-mesh",
    is_flag=True,
    help="Return only descriptions bound to real Hydra object meshes.",
)
@click.option(
    "--visual-output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "For every query, copy all available result evidence images and write a "
        "marked mesh top view plus query_result.json below this directory."
    ),
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
    min_similarity: float,
    min_margin: float,
    allow_unverified_embeddings: bool,
    require_mesh: bool,
    visual_output_dir: Optional[Path],
) -> None:
    """Run local retrieval, with optional API-backed grounded answers."""
    try:
        engine = SemanticQueryEngine(
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
        f"Loaded {len(engine.records)} queryable objects from {dsg_path}: "
        f"mesh_bound={engine.geometry_counts['mesh_bound']}, "
        f"spatial_only={engine.geometry_counts['spatial_only']}, "
        f"image_only={engine.geometry_counts['image_only']}"
    )
    click.echo(
        "Multilingual embeddings are local; retrieval-only queries make no LLM/API calls."
    )

    def write_visual_result(
        text: str,
        matches,
        *,
        found: bool,
        rejection_reason: Optional[str],
        top_score: float,
        top1_margin: Optional[float],
    ) -> None:
        if visual_output_dir is None:
            return
        try:
            artifacts = write_query_visuals(
                engine=engine,
                query=text,
                matches=matches,
                output_root=visual_output_dir,
                found=found,
                rejection_reason=rejection_reason,
                top_score=top_score,
                top1_margin=top1_margin,
            )
        except QueryVisualizationError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"visual_output: {artifacts.output_directory}")
        click.echo(f"mesh_topdown: {artifacts.topdown_image}")
        if artifacts.evidence_images:
            for evidence_image in artifacts.evidence_images:
                click.echo(f"evidence_image: {evidence_image}")
        else:
            click.echo("evidence_image: unavailable")
        click.echo(f"query_report: {artifacts.report}")

    def run_query(text: str) -> None:
        normalized_text = text.strip()
        if not normalized_text:
            return
        try:
            if not answer_with_llm:
                click.echo(f"\nQuery: {normalized_text}")
                decision = engine.retrieve_with_decision(
                    normalized_text, top_k, require_mesh=require_mesh
                )
                print_decision(decision, engine)
                write_visual_result(
                    normalized_text,
                    decision.matches,
                    found=decision.found,
                    rejection_reason=decision.rejection_reason,
                    top_score=decision.top_score,
                    top1_margin=decision.top1_margin,
                )
                return

            result = engine.answer_question(
                normalized_text, top_k=top_k, require_mesh=require_mesh
            )
            click.echo(f"Retrieval phrase: {result.retrieval_query}")
            margin = "n/a" if result.top1_margin is None else f"{result.top1_margin:.4f}"
            click.echo(
                f"found={str(result.found).lower()}  top_score={result.top_score:.4f}  "
                f"top1_margin={margin}"
            )
            if result.rejection_reason:
                click.echo(f"rejection_reason={result.rejection_reason}")
            print_results(result.matches)
            print_top1_evidence(engine, result.matches)
            click.echo("\nGrounded answer:")
            click.echo(result.answer)
            write_visual_result(
                normalized_text,
                result.matches,
                found=result.found,
                rejection_reason=result.rejection_reason,
                top_score=result.top_score,
                top1_margin=result.top1_margin,
            )
        except (SemanticQueryError, LLMUnavailableError) as exc:
            raise click.ClickException(str(exc)) from exc

    if query_text is not None:
        run_query(query_text)
        return

    click.echo("Enter a Chinese or English description (empty line or Ctrl+C exits).")
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
