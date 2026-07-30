#!/usr/bin/env python3
"""Evaluate a frozen multilingual open-set query set with evidence retention."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.semantic_query import (  # noqa: E402
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_SENTENCE_MODEL,
    SemanticQueryEngine,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsg", required=True, type=Path)
    parser.add_argument("--query-set", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--minimum-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--minimum-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument("--require-mesh", action="store_true")
    return parser.parse_args()


def normalized(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    query_set = yaml.safe_load(args.query_set.read_text()) or {}
    if query_set.get("schema") != "daaam.semantic_query_set.v1":
        raise ValueError("unsupported query-set schema")
    queries = list(query_set.get("queries") or [])
    if not queries or len({query["id"] for query in queries}) != len(queries):
        raise ValueError("query set must contain unique non-empty queries")
    output.mkdir(parents=True)
    evidence_directory = output / "evidence"
    evidence_directory.mkdir()
    raw_directory = output / "per_query"
    raw_directory.mkdir()
    engine = SemanticQueryEngine(
        args.dsg,
        sentence_model_name=args.sentence_model,
        min_similarity=args.minimum_similarity,
        min_margin=args.minimum_margin,
    )
    records = []
    for query in queries:
        started = time.perf_counter()
        decision = engine.retrieve_with_decision(
            str(query["text"]),
            top_k=args.top_k,
            min_similarity=args.minimum_similarity,
            min_margin=args.minimum_margin,
            require_mesh=args.require_mesh,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        matches = []
        copied_evidence = []
        for rank, (score, match) in enumerate(decision.matches, start=1):
            evidence = engine.evidence_for_node(match.node_id)
            evidence_copy = None
            if evidence is not None and evidence.image_path.is_file():
                suffix = evidence.image_path.suffix or ".png"
                target = (
                    evidence_directory
                    / f"{query['id']}_rank{rank}_{match.node_id}{suffix}"
                )
                shutil.copy2(evidence.image_path, target)
                evidence_copy = str(target)
                copied_evidence.append(evidence_copy)
            matches.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "node_id": match.node_id,
                    "semantic_label": match.semantic_label,
                    "description": match.description,
                    "position": match.position.tolist(),
                    "geometry_status": match.geometry_status,
                    "entity_id": match.entity_id,
                    "evidence_copy": evidence_copy,
                }
            )
        expected_found = bool(query["expected_found"])
        top_text = (
            normalized(matches[0]["description"]) if matches else ""
        )
        expected_terms = [
            normalized(value) for value in query.get("expected_terms", [])
        ]
        term_match = (
            any(term in top_text for term in expected_terms)
            if expected_found and expected_terms
            else True
        )
        passed = decision.found == expected_found and (
            not expected_found or term_match
        )
        record = {
            "query_id": str(query["id"]),
            "language": str(query.get("language", "unknown")),
            "query": str(query["text"]),
            "expected_found": expected_found,
            "expected_terms": query.get("expected_terms", []),
            "found": decision.found,
            "rejection_reason": decision.rejection_reason,
            "top_score": decision.top_score,
            "top1_margin": decision.top1_margin,
            "latency_ms": latency_ms,
            "term_match": term_match,
            "passed": passed,
            "matches": matches,
            "evidence_files": copied_evidence,
        }
        (raw_directory / f"{query['id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        records.append(record)
    positive = [record for record in records if record["expected_found"]]
    negative = [record for record in records if not record["expected_found"]]
    summary = {
        "schema": "daaam.semantic_query_evaluation.v1",
        "dsg": str(args.dsg.resolve()),
        "query_set": str(args.query_set.resolve()),
        "sentence_model": args.sentence_model,
        "thresholds": {
            "minimum_similarity": args.minimum_similarity,
            "minimum_margin": args.minimum_margin,
            "require_mesh": args.require_mesh,
            "top_k": args.top_k,
        },
        "queries": len(records),
        "passed": sum(record["passed"] for record in records),
        "pass_rate": sum(record["passed"] for record in records) / len(records),
        "positive_recall": (
            sum(record["passed"] for record in positive) / len(positive)
            if positive
            else None
        ),
        "negative_rejection_rate": (
            sum(record["passed"] for record in negative) / len(negative)
            if negative
            else None
        ),
        "evidence_coverage": (
            sum(bool(record["evidence_files"]) for record in positive) / len(positive)
            if positive
            else None
        ),
        "latency_ms": {
            "mean": sum(record["latency_ms"] for record in records) / len(records),
            "maximum": max(record["latency_ms"] for record in records),
        },
        "records": records,
    }
    (output / "query_evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    with (output / "query_evaluation.csv").open("w", newline="") as stream:
        fields = (
            "query_id",
            "language",
            "query",
            "expected_found",
            "found",
            "top_score",
            "top1_margin",
            "latency_ms",
            "term_match",
            "passed",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: record[field] for field in fields} for record in records
        )
    row_height = 38
    svg_rows = []
    for index, record in enumerate(records):
        y = 70 + index * row_height
        score = float(record["top_score"] or 0.0)
        margin = float(record["top1_margin"] or 0.0)
        color = "#31a354" if record["passed"] else "#de2d26"
        svg_rows.append(
            f'<rect x="15" y="{y - 20}" width="9" height="25" fill="{color}"/>'
            f'<text x="35" y="{y}" font-size="14">'
            f'{escape(record["query_id"])} · {escape(record["query"])}</text>'
            f'<rect x="690" y="{y - 17}" width="{max(0.0, score) * 260:.2f}" '
            f'height="10" fill="#3182bd"/>'
            f'<rect x="690" y="{y - 4}" width="{max(0.0, margin) * 260:.2f}" '
            f'height="8" fill="#756bb1"/>'
            f'<text x="970" y="{y}" font-size="12">'
            f'score={score:.3f} margin={margin:.3f} '
            f'latency={record["latency_ms"]:.1f}ms</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1320" '
        f'height="{100 + row_height * len(records)}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="15" y="30" font-size="22" font-weight="bold">'
        'Frozen multilingual semantic-query evaluation</text>'
        '<text x="690" y="48" font-size="12">'
        'blue=top score, purple=top-1 margin; green=pass, red=fail</text>'
        + "".join(svg_rows)
        + "</svg>"
    )
    (output / "query_evaluation.svg").write_text(svg + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
