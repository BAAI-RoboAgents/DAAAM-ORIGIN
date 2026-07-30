#!/usr/bin/env python3
"""Independently verify E15 accounting, embeddings, rankings, and provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPOSITORY_ROOT
    / "experiments/g1_20260724_473_573_v1_1/runs"
    / "diagnostic_gt_free_e15_safe035_increment_20260730"
)
INVENTORY_EXCLUDES = {
    "artifact_inventory.csv",
    "artifact_inventory.jsonl",
    "inventory_summary.json",
    "COMPLETION.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_root(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                f"{row['relative_path']}\t{row['size_bytes']}\t"
                f"{row['sha256']}\n"
            ).encode()
        )
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_inventory(root: Path) -> dict[str, Any]:
    rows = read_jsonl(root / "artifact_inventory.jsonl")
    by_path = {str(row["relative_path"]): row for row in rows}
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in INVENTORY_EXCLUDES
    )
    check(actual == sorted(by_path), "inventory path set")
    for relative in actual:
        path = root / relative
        row = by_path[relative]
        check(path.stat().st_size == int(row["size_bytes"]), f"size: {relative}")
        check(sha256_file(path) == row["sha256"], f"sha256: {relative}")
    summary = read_json(root / "inventory_summary.json")
    observed_root = inventory_root(rows)
    check(
        observed_root == summary["inventory_root_sha256"],
        "inventory root hash",
    )
    check(len(rows) == int(summary["file_count"]), "inventory file count")
    return {
        "file_count": len(rows),
        "root_sha256": observed_root,
    }


def status_counts(values: Iterable[str]) -> dict[str, Any]:
    sequence = list(values)
    counts = {
        status: sum(value == status for value in sequence)
        for status in ("success", "partial", "failure")
    }
    total = len(sequence)
    return {
        "total": total,
        **counts,
        "strict_success_rate": counts["success"] / total,
        "lenient_success_rate": (
            counts["success"] + counts["partial"]
        )
        / total,
    }


def selected_row(
    rows: Iterable[Mapping[str, Any]],
    key: str,
    value: Any,
) -> dict[str, Any]:
    matches = [dict(row) for row in rows if row.get(key) == value]
    check(len(matches) == 1, f"one source row: {key}={value!r}")
    return matches[0]


def close(left: float, right: float, tolerance: float = 1.0e-6) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> int:
    args = parse_args()
    root = args.run.resolve()
    audit_path = root / "INDEPENDENT_AUDIT.json"
    check(root.is_dir(), "run directory")
    check(not audit_path.exists(), "audit already exists")
    inventory = verify_inventory(root)

    frozen = read_json(root / "FROZEN_INPUTS.json")
    source_paths: dict[str, Path] = {}
    for key, reference in frozen.items():
        if key in {"schema", "captured_at"}:
            continue
        path = Path(str(reference["path"]))
        source_paths[key] = path
        check(path.is_file(), f"{key}: source exists")
        check(path.stat().st_size == int(reference["size_bytes"]), f"{key}: size")
        check(sha256_file(path) == reference["sha256"], f"{key}: sha256")

    completion = read_json(root / "COMPLETION.json")
    check(
        completion["status"] == "complete_pending_independent_audit",
        "pre-audit completion status",
    )
    protocol = read_json(source_paths["protocol_query_set"])
    check(protocol["queries"] == [], "formal query set remains empty")
    check(
        protocol["status"] == "not_generated_pending_reviewed_L5_GT",
        "formal query status",
    )

    old_census = read_jsonl(source_paths["old_census"])
    targets = read_jsonl(source_paths["target_comparison"])
    labels = read_jsonl(source_paths["e14_labels"])
    stage_rows = read_jsonl(root / "tables/stage_status.jsonl")
    check(len(old_census) == len(targets) == len(stage_rows) == 20, "20 targets")
    check(len(labels) == 87, "87 final DAM labels")
    old_by_id = {row["instance_id"]: row for row in old_census}
    target_by_id = {row["instance_id"]: row for row in targets}
    stage_by_id = {row["instance_id"]: row for row in stage_rows}
    check(
        set(old_by_id) == set(target_by_id) == set(stage_by_id),
        "target identity sets",
    )
    for instance_id, row in stage_by_id.items():
        old = old_by_id[instance_id]
        target = target_by_id[instance_id]
        check(row["E11_status"] == old["stages"]["E11"], f"{instance_id}: E11")
        check(row["E12_status"] == old["stages"]["E12"], f"{instance_id}: E12")
        check(
            row["E13_status"] == target["new_e13_status"],
            f"{instance_id}: E13",
        )
        check(
            row["E14_status"] == target["new_e14_status"],
            f"{instance_id}: E14",
        )
        check(
            row["primary_evidence_entities_json"]
            == target["primary_evidence_entities_json"],
            f"{instance_id}: evidence entities",
        )
    stage_summary = read_json(root / "tables/stage_status_summary.json")
    for stage in ("E11", "E12", "E13", "E14"):
        recomputed = status_counts(
            row[f"{stage}_status"] for row in stage_rows
        )
        check(recomputed == stage_summary[stage], f"{stage}: status summary")

    label_embeddings = np.load(
        root / "query_proxy/label_embeddings.npy",
        allow_pickle=False,
    )
    query_embeddings = np.load(
        root / "query_proxy/query_embeddings.npy",
        allow_pickle=False,
    )
    scores = np.load(
        root / "query_proxy/score_matrix.npy",
        allow_pickle=False,
    )
    check(label_embeddings.shape == (87, 768), "label embedding shape")
    check(query_embeddings.shape == (20, 768), "query embedding shape")
    check(scores.shape == (20, 87), "score matrix shape")
    check(np.isfinite(label_embeddings).all(), "finite label embeddings")
    check(np.isfinite(query_embeddings).all(), "finite query embeddings")
    check(np.isfinite(scores).all(), "finite scores")
    check(
        np.allclose(np.linalg.norm(label_embeddings, axis=1), 1.0, atol=1.0e-5),
        "normalized label embeddings",
    )
    check(
        np.allclose(np.linalg.norm(query_embeddings, axis=1), 1.0, atol=1.0e-5),
        "normalized query embeddings",
    )
    recomputed_scores = query_embeddings @ label_embeddings.T
    maximum_score_error = float(np.max(np.abs(scores - recomputed_scores)))
    check(maximum_score_error <= 1.0e-6, "score matrix dot product")

    query_rows = read_jsonl(root / "query_proxy/query_rankings.jsonl")
    query_summary = read_json(root / "query_proxy/query_summary.json")
    check(len(query_rows) == 20, "20 query rows")
    top_k = sorted(
        int(value) for value in query_summary["recall_at_k_proxy"]
    )
    check(top_k == [1, 3, 5, 10], "frozen top-k")
    minimum_similarity = float(query_summary["minimum_similarity"])
    check(minimum_similarity == 0.55, "frozen minimum similarity")
    query_by_id = {row["instance_id"]: row for row in query_rows}
    raw_hits = {value: 0 for value in top_k}
    thresholded_hits = {value: 0 for value in top_k}
    for query_index, target in enumerate(targets):
        record = query_by_id[target["instance_id"]]
        row_scores = scores[query_index]
        order = np.argsort(-row_scores, kind="stable")
        relevant = {
            int(value)
            for value in json.loads(
                target["primary_evidence_entities_json"]
            )
        }
        relevant_ranks = [
            rank
            for rank, label_index in enumerate(order, start=1)
            if int(labels[int(label_index)]["entity_ordinal"]) in relevant
        ]
        best_rank = min(relevant_ranks) if relevant_ranks else None
        top_score = float(row_scores[int(order[0])])
        check(record["query"] == target["instance_name"], "query text order")
        check(record["best_relevant_rank"] == best_rank, "best relevant rank")
        check(close(record["top_score"], top_score), "top score")
        check(
            record["passes_minimum_similarity"]
            == (top_score >= minimum_similarity),
            "threshold decision",
        )
        rankings = json.loads(record["rankings_json"])
        check(len(rankings) == 10, "top-10 ranking length")
        for rank, (ranking, label_index) in enumerate(
            zip(rankings, order[:10]),
            start=1,
        ):
            label = labels[int(label_index)]
            check(int(ranking["rank"]) == rank, "ranking ordinal")
            check(
                int(ranking["entity_ordinal"])
                == int(label["entity_ordinal"]),
                "ranking entity",
            )
            check(
                close(ranking["score"], row_scores[int(label_index)]),
                "ranking score",
            )
        for value in top_k:
            hit = best_rank is not None and best_rank <= value
            raw_hits[value] += int(hit)
            thresholded_hits[value] += int(
                hit and top_score >= minimum_similarity
            )
    for value in top_k:
        check(
            close(
                query_summary["recall_at_k_proxy"][str(value)],
                raw_hits[value] / 20,
            ),
            f"R@{value}",
        )
        check(
            close(
                query_summary["thresholded_recall_at_k_proxy"][str(value)],
                thresholded_hits[value] / 20,
            ),
            f"thresholded R@{value}",
        )

    e11_rows = read_json(source_paths["e11_summary"])
    e12_rows = read_json(source_paths["e12_summary"])
    e13_rows = read_json(source_paths["e13_summary"])
    e14_rows = read_json(source_paths["e14_summary"])
    e11 = selected_row(
        e11_rows, "cell_id", "conf_0p3__area_0300__iou_0p5"
    )
    e12 = selected_row(e12_rows, "variant_id", "buffer_10")
    e13 = selected_row(e13_rows, "variant_id", "safe_merge_0p35m")
    e14 = selected_row(e14_rows, "threshold_observations", 8)
    funnel = read_json(root / "tables/incremental_funnel.json")
    funnel_by_variant = {row["variant"]: row for row in funnel}
    check(set(funnel_by_variant) == {"geometry_only", "frontend", "dam"}, "variants")
    check(
        int(funnel_by_variant["frontend"]["mask_count"])
        == int(e11["kept_instance_count"]),
        "frontend masks",
    )
    check(
        int(funnel_by_variant["frontend"]["tracked_observation_count"])
        == int(e12["tracked_observation_count"]),
        "frontend tracked observations",
    )
    check(
        int(funnel_by_variant["frontend"]["entity_count"])
        == int(e13["entity_count"]),
        "frontend entity count",
    )
    check(
        int(funnel_by_variant["dam"]["named_entity_count"])
        == int(e14["responded_entity_count"]),
        "DAM named count",
    )
    check(
        int(funnel_by_variant["dam"]["query_candidate_count"]) == len(labels),
        "DAM query candidates",
    )

    cost = read_json(root / "tables/cost_summary.json")
    e11_total = float(e11["inference_latency_mean_ms"]) * 101
    e12_total = float(e12["tracking_latency_mean_ms"]) * 101
    e13_total = float(e13["observe_latency_ms_mean"]) * int(
        e13["input_geometry_observations"]
    )
    e14_total = float(e14["batch_latency_s"]["mean"]) * int(
        e14["batch_latency_s"]["count"]
    ) * 1000.0
    for key, expected in (
        ("e11", e11_total),
        ("e12", e12_total),
        ("e13", e13_total),
        ("e14", e14_total),
    ):
        check(close(cost[key]["total_ms"], expected), f"{key}: total cost")
    check(
        close(
            cost["cumulative"]["total_ms"],
            e11_total + e12_total + e13_total + e14_total,
        ),
        "cumulative cost",
    )

    model_manifest = read_json(root / "query_proxy/MODEL_MANIFEST.json")
    for name in (
        "label_embeddings.npy",
        "query_embeddings.npy",
        "score_matrix.npy",
    ):
        key = name.removesuffix(".npy") + "_sha256"
        check(
            sha256_file(root / "query_proxy" / name) == model_manifest[key],
            f"{name}: manifest hash",
        )
    e14_database = (
        source_paths["e14_completion"].parent
        / "cells/obs_08/seed_0/map_memory.sqlite3"
    )
    with sqlite3.connect(e14_database) as connection:
        sqlite_integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
    check(sqlite_integrity == "ok", "source E14 SQLite integrity")

    summary = read_json(root / "SUMMARY.json")
    check(summary["hypothesis_result"] == "partial_pass", "hypothesis result")
    check(summary["winner"] is None, "no E15 winner")
    check(summary["formal_claims_permitted"] is False, "claim boundary")
    verifier_path = Path(__file__).resolve()
    audit = {
        "schema": "daaam.g1_no_gt_e15_independent_audit.v1",
        "audited_at": utc_now(),
        "auditor": "independent deterministic verifier",
        "verifier": {
            "path": str(verifier_path),
            "sha256": sha256_file(verifier_path),
        },
        "passed": True,
        "result": "PASS",
        "checks": {
            "inventory": inventory,
            "frozen_input_reference_count": len(source_paths),
            "target_count": len(targets),
            "label_count": len(labels),
            "stage_accounting_checks": 4 * len(stage_rows),
            "embedding_shapes": {
                "query": list(query_embeddings.shape),
                "label": list(label_embeddings.shape),
                "score": list(scores.shape),
            },
            "maximum_score_recompute_error": maximum_score_error,
            "ranking_check_count": len(query_rows) * 10,
            "query_proxy": {
                "raw_hits": raw_hits,
                "thresholded_hits": thresholded_hits,
            },
            "source_e14_sqlite_integrity_check": sqlite_integrity,
        },
        "correctness_audited": False,
        "correctness_reason": (
            "The verifier proves provenance, accounting, embedding arithmetic, "
            "rankings, costs, and SQLite integrity. Target relevance remains a "
            "post-hoc Codex approximate-GT proxy and formal L5 queries are absent."
        ),
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
