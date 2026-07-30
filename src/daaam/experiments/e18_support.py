"""Pure helpers for the G1 E18 exact semantic-label postpass experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


POSTPASS_CONTRACT_FIELDS = (
    "status",
    "frames_expected",
    "frames_replayed",
    "frames_with_labels",
    "label_coverage",
    "missing_frame_indices",
    "nonzero_label_frames",
    "nonzero_label_pixels",
    "unique_semantic_labels",
    "label_manifest_sha256",
    "label_run_configuration_sha256",
)

DETERMINISTIC_PRODUCT_FILES = (
    "backend/mesh.ply",
    "backend/dsg.json",
    "backend/dsg_with_mesh.json",
    "backend/deformation_graph.dgrf",
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                dict(record),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def validate_postpass_report(
    report: Mapping[str, Any],
    *,
    expected_frames: int,
    expected_label_manifest_sha256: str,
    expected_run_configuration_sha256: str,
) -> list[str]:
    """Return every exact-postpass contract violation without short-circuiting."""

    issues: list[str] = []
    if report.get("schema") != "daaam.hydra_semantic_postpass.v1":
        issues.append("unsupported_schema")
    if report.get("status") != "complete":
        issues.append("status_not_complete")
    for field in ("frames_expected", "frames_replayed", "frames_with_labels"):
        if int(report.get(field, -1)) != int(expected_frames):
            issues.append(f"{field}_mismatch")
    if float(report.get("label_coverage", 0.0)) != 1.0:
        issues.append("label_coverage_not_one")
    if list(report.get("missing_frame_indices") or []):
        issues.append("missing_frame_indices_nonempty")
    if report.get("label_manifest_sha256") != expected_label_manifest_sha256:
        issues.append("label_manifest_mismatch")
    if (
        report.get("label_run_configuration_sha256")
        != expected_run_configuration_sha256
    ):
        issues.append("run_configuration_mismatch")
    return issues


def compare_exact_repetitions(
    repetitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare semantic invariants and byte-level formal products."""

    if len(repetitions) < 2:
        raise ValueError("exact postpass comparison requires at least two repetitions")
    contract_values: dict[str, list[Any]] = {
        field: [row["report"].get(field) for row in repetitions]
        for field in POSTPASS_CONTRACT_FIELDS
    }
    contract_stability = {
        field: all(value == values[0] for value in values[1:])
        for field, values in contract_values.items()
    }
    product_hashes: dict[str, list[str | None]] = {
        relative: [
            dict(row.get("artifact_hashes") or {}).get(relative)
            for row in repetitions
        ]
        for relative in DETERMINISTIC_PRODUCT_FILES
    }
    product_stability = {
        relative: (
            all(value is not None for value in values)
            and all(value == values[0] for value in values[1:])
        )
        for relative, values in product_hashes.items()
    }
    graph_summaries = [dict(row.get("graph_summary") or {}) for row in repetitions]
    graph_summary_stable = all(
        summary == graph_summaries[0] for summary in graph_summaries[1:]
    )
    return {
        "schema": "daaam.g1_e18_repetition_comparison.v1",
        "repetition_count": len(repetitions),
        "contract_stability": contract_stability,
        "semantic_contract_stable": all(contract_stability.values()),
        "product_hashes": product_hashes,
        "product_stability": product_stability,
        "formal_product_hash_stable": all(product_stability.values()),
        "graph_summaries": graph_summaries,
        "graph_summary_stable": graph_summary_stable,
    }


def evaluate_durable_commit(
    *,
    applied: int,
    rejected_no_mesh: int,
    delivery_pending: int,
    unmapped: int,
    errors: Sequence[str],
    graph_reloaded: bool,
    output_hash_verified: bool,
    candidate_review_pending: int,
) -> dict[str, Any]:
    """Evaluate the E18 delivery gate while keeping review candidates separate."""

    reasons: list[str] = []
    if applied <= 0:
        reasons.append("no_applied_bindings")
    if delivery_pending:
        reasons.append("delivery_pending_nonzero")
    if unmapped:
        reasons.append("unmapped_nonzero")
    if errors:
        reasons.append("delivery_errors_nonempty")
    if not graph_reloaded:
        reasons.append("graph_reload_failed")
    if not output_hash_verified:
        reasons.append("output_hash_mismatch")
    return {
        "schema": "daaam.g1_e18_durable_commit_gate.v1",
        "status": "passed" if not reasons else "failed",
        "applied": int(applied),
        "rejected_no_mesh": int(rejected_no_mesh),
        "delivery_pending": int(delivery_pending),
        "unmapped": int(unmapped),
        "errors": list(errors),
        "graph_reloaded": bool(graph_reloaded),
        "output_hash_verified": bool(output_hash_verified),
        "candidate_review_pending": int(candidate_review_pending),
        "candidate_review_pending_is_nonblocking": True,
        "failure_reasons": reasons,
    }
