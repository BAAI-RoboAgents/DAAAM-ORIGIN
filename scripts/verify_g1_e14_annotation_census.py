#!/usr/bin/env python3
"""Independent deterministic verifier for the E14 dual-pass census evidence."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kappa(left: list[Any], right: list[Any]) -> float:
    count = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / count
    expected = sum(
        (left.count(category) / count) * (right.count(category) / count)
        for category in set(left) | set(right)
    )
    return (observed - expected) / (1.0 - expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.census_dir.resolve()
    output = root / "CENSUS_INDEPENDENT_AUDIT.json"

    prereg = read_json(root / "PRE_REGISTRATION.json")
    manifest = read_jsonl(root / "CENSUS_MANIFEST.jsonl")
    reviewer_a = read_jsonl(root / "REVIEW_RESULTS_REVIEWER_A.jsonl")
    reviewer_b = read_jsonl(root / "REVIEW_RESULTS_REVIEWER_B.jsonl")
    consensus = read_jsonl(root / "REVIEW_CONSENSUS.jsonl")
    disagreements = read_jsonl(root / "REVIEW_DISAGREEMENTS.jsonl")
    failures = read_jsonl(root / "FAILURE_CASES.jsonl")
    salient = read_jsonl(root / "SALIENT_OBJECT_COVERAGE.jsonl")
    summary = read_json(root / "AUDIT_SUMMARY.json")
    integrity = read_json(root / "EVIDENCE_INTEGRITY.json")
    adjudicated = read_jsonl(root / "REVIEW_RESULTS_ADJUDICATED.jsonl")
    salient_targets = read_jsonl(root / "SALIENT_TARGET_COVERAGE.jsonl")
    salient_target_summary = read_json(root / "SALIENT_TARGET_SUMMARY.json")
    adjudicated_summary = read_json(root / "ADJUDICATED_SUMMARY.json")
    adjudicated_integrity = read_json(root / "ADJUDICATED_INTEGRITY.json")

    expected = int(prereg["finite_population_count"])
    identity_fields = ("census_index", "entity_id", "entity_ordinal")

    def keys(rows: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
        return {tuple(row[field] for field in identity_fields) for row in rows}

    manifest_keys = keys(manifest)
    check(len(manifest) == expected == len(manifest_keys), "manifest population")
    for name, rows in (
        ("reviewer_a", reviewer_a),
        ("reviewer_b", reviewer_b),
        ("consensus", consensus),
        ("adjudicated", adjudicated),
    ):
        check(len(rows) == expected, f"{name} row count")
        check(keys(rows) == manifest_keys, f"{name} identity match")

    panel_hash_checks = 0
    rgb_hash_checks = 0
    mask_hash_checks = 0
    for row in manifest:
        panel = Path(row["review_panel_path"])
        check(panel.is_file(), f"panel exists: {panel}")
        check(sha256(panel) == row["review_panel_sha256"], f"panel hash: {panel}")
        panel_hash_checks += 1
        rgb = Path(row["rgb_path"])
        check(rgb.is_file(), f"RGB exists: {rgb}")
        check(sha256(rgb) == row["rgb_sha256"], f"RGB hash: {rgb}")
        rgb_hash_checks += 1
        for record in row["mask_records"]:
            mask = Path(record["materialized_mask_path"])
            check(mask.is_file(), f"mask exists: {mask}")
            check(
                sha256(mask) == record["materialized_mask_sha256"],
                f"mask hash: {mask}",
            )
            mask_hash_checks += 1

    a_by_key = {
        tuple(row[field] for field in identity_fields): row for row in reviewer_a
    }
    b_by_key = {
        tuple(row[field] for field in identity_fields): row for row in reviewer_b
    }
    severity = {
        "correct": 0,
        "partially_correct": 1,
        "incorrect": 2,
        "unjudgeable": 3,
    }
    semantic_disagreements = 0
    any_disagreements = 0
    failure_count = 0
    for row in consensus:
        key = tuple(row[field] for field in identity_fields)
        left = a_by_key[key]
        right = b_by_key[key]
        expected_semantic = max(
            (left["semantic_verdict"], right["semantic_verdict"]),
            key=severity.__getitem__,
        )
        check(
            row["semantic_verdict_conservative"] == expected_semantic,
            f"conservative semantic: {key}",
        )
        check(
            row["mask_acceptable_both"]
            == (bool(left["mask_acceptable"]) and bool(right["mask_acceptable"])),
            f"mask consensus: {key}",
        )
        semantic_diff = left["semantic_verdict"] != right["semantic_verdict"]
        any_diff = (
            semantic_diff
            or left["mask_acceptable"] != right["mask_acceptable"]
            or left["main_identity_correct"] != right["main_identity_correct"]
        )
        semantic_disagreements += semantic_diff
        any_disagreements += any_diff
        failure_count += (
            not row["mask_acceptable_both"]
            or expected_semantic != "correct"
            or not row["semantic_complete_both"]
            or not row["main_identity_correct_both"]
        )

    check(len(disagreements) == any_disagreements, "disagreement queue count")
    check(len(failures) == failure_count, "failure queue count")
    semantic_a = [row["semantic_verdict"] for row in reviewer_a]
    semantic_b = [row["semantic_verdict"] for row in reviewer_b]
    mask_a = [bool(row["mask_acceptable"]) for row in reviewer_a]
    mask_b = [bool(row["mask_acceptable"]) for row in reviewer_b]
    strict = sum(
        a == "correct" and b == "correct"
        for a, b in zip(semantic_a, semantic_b, strict=True)
    )
    lenient = sum(
        a != "incorrect" and b != "incorrect"
        for a, b in zip(semantic_a, semantic_b, strict=True)
    )
    mask_both = sum(
        a and b for a, b in zip(mask_a, mask_b, strict=True)
    )
    joint_strict = sum(
        ma
        and mb
        and sa == "correct"
        and sb == "correct"
        for ma, mb, sa, sb in zip(
            mask_a, mask_b, semantic_a, semantic_b, strict=True
        )
    )
    check(strict == 19, "strict conservative count")
    check(lenient == 39, "lenient conservative count")
    check(mask_both == 28, "mask both count")
    check(joint_strict == 13, "joint strict count")
    check(semantic_disagreements == 13, "semantic disagreement count")
    check(
        abs(kappa(semantic_a, semantic_b) - 0.6046783625730994) < 1e-12,
        "semantic kappa",
    )
    check(
        abs(kappa(mask_a, mask_b) - 0.7182662538699691) < 1e-12,
        "mask kappa",
    )

    check(
        summary["conservative_consensus"]["strict_visual_correctness"]["count"]
        == strict,
        "summary strict",
    )
    check(
        summary["conservative_consensus"]["lenient_visual_correctness"]["count"]
        == lenient,
        "summary lenient",
    )
    check(
        summary["conservative_consensus"]["mask_acceptable_both"]["count"]
        == mask_both,
        "summary mask",
    )
    check(
        summary["conservative_consensus"][
            "mask_acceptable_and_strict_correct"
        ]["count"]
        == joint_strict,
        "summary joint strict",
    )
    salient_counts = collections.Counter(row["coverage_status"] for row in salient)
    check(
        salient_counts
        == {
            "clean_full_entity": 2,
            "covered_fragmented_or_conflicted": 10,
            "missed_as_whole": 3,
        },
        "salient status counts",
    )
    salient_evidence_checks = 0
    for row in salient:
        for relative in row["evidence"]:
            check((root / relative).resolve().is_file(), f"salient evidence: {relative}")
            salient_evidence_checks += 1

    for section in ("inputs", "generated"):
        for relative, record in integrity[section].items():
            path = root / relative
            check(path.is_file(), f"integrity path: {relative}")
            check(path.stat().st_size == record["bytes"], f"integrity size: {relative}")
            check(sha256(path) == record["sha256"], f"integrity hash: {relative}")

    adjudicated_semantic = collections.Counter(
        row["semantic_verdict"] for row in adjudicated
    )
    adjudicated_masks = collections.Counter(row["mask_verdict"] for row in adjudicated)
    check(
        adjudicated_semantic
        == {"correct": 20, "partially_correct": 24, "incorrect": 8},
        "adjudicated semantic counts",
    )
    check(
        adjudicated_masks
        == {
            "acceptable": 30,
            "acceptable_boundary_truncated": 2,
            "oversegmented": 12,
            "partial": 6,
            "wrong_region": 2,
        },
        "adjudicated mask counts",
    )
    check(
        sum(bool(row["mask_acceptable"]) for row in adjudicated) == 32,
        "adjudicated mask acceptable",
    )
    check(
        sum(bool(row["semantic_complete"]) for row in adjudicated) == 50,
        "adjudicated semantic complete",
    )
    object_targets = [
        row for row in salient_targets if row["target_type"] != "scene_stuff"
    ]
    scene_stuff_targets = [
        row for row in salient_targets if row["target_type"] == "scene_stuff"
    ]
    check(len(object_targets) == 13, "salient object target count")
    check(len(scene_stuff_targets) == 3, "scene stuff target count")
    check(
        sum(row["region_response"] == "confirmed" for row in object_targets) == 12,
        "confirmed object region responses",
    )
    check(
        sum(row["region_response"] == "probable" for row in object_targets) == 1,
        "probable object region responses",
    )
    check(
        sum(bool(row["usable_independent_entity"]) for row in object_targets) == 7,
        "usable object entities",
    )
    check(
        sum(bool(row["fully_clean_at_target_granularity"]) for row in object_targets)
        == 4,
        "fully clean object targets",
    )
    check(
        sum(bool(row["usable_independent_entity"]) for row in scene_stuff_targets)
        == 3,
        "usable scene-stuff entities",
    )
    check(
        adjudicated_summary["adjudicated"]["strict_visual_correctness"]["count"]
        == 20,
        "adjudicated summary strict",
    )
    check(
        adjudicated_summary["adjudicated"]["lenient_visual_correctness"]["count"]
        == 44,
        "adjudicated summary lenient",
    )
    check(
        salient_target_summary["object_targets_with_usable_independent_entity"]
        == 7,
        "salient summary usable",
    )
    check(
        salient_target_summary["object_targets_fully_clean_at_target_granularity"]
        == 4,
        "salient summary fully clean",
    )
    for relative, digest in adjudicated_integrity["sha256"].items():
        path = (root / relative).resolve()
        check(path.is_file(), f"adjudicated integrity path: {relative}")
        check(sha256(path) == digest, f"adjudicated integrity hash: {relative}")

    audit = {
        "schema": "daaam.g1_no_gt_e14_annotation_census_independent_audit.v1",
        "result": "PASS",
        "passed": True,
        "auditor": "independent deterministic verifier; no visual re-adjudication",
        "verifier": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__)),
        },
        "checks": {
            "finite_population": expected,
            "review_passes": 2,
            "review_rows": len(reviewer_a) + len(reviewer_b),
            "consensus_rows": len(consensus),
            "panel_hash_checks": panel_hash_checks,
            "rgb_hash_checks": rgb_hash_checks,
            "mask_hash_checks": mask_hash_checks,
            "semantic_disagreements": semantic_disagreements,
            "all_field_disagreements": any_disagreements,
            "failure_queue_rows": failure_count,
            "salient_rows": len(salient),
            "salient_evidence_path_checks": salient_evidence_checks,
            "strict_conservative": strict,
            "lenient_conservative": lenient,
            "mask_acceptable_both": mask_both,
            "joint_strict": joint_strict,
            "adjudicated_rows": len(adjudicated),
            "adjudicated_strict": adjudicated_semantic["correct"],
            "adjudicated_lenient": (
                adjudicated_semantic["correct"]
                + adjudicated_semantic["partially_correct"]
            ),
            "adjudicated_mask_acceptable": sum(
                bool(row["mask_acceptable"]) for row in adjudicated
            ),
            "salient_object_targets": len(object_targets),
            "salient_object_targets_with_usable_entity": sum(
                bool(row["usable_independent_entity"]) for row in object_targets
            ),
            "salient_object_targets_fully_clean": sum(
                bool(row["fully_clean_at_target_granularity"])
                for row in object_targets
            ),
            "scene_stuff_targets": len(scene_stuff_targets),
            "adjudicated_integrity_hash_entries": len(
                adjudicated_integrity["sha256"]
            ),
        },
        "semantic_correctness_re_adjudicated": False,
        "permitted_claims": [
            "52/52 population coverage for each of two Codex review records",
            "exact panel/RGB/mask identity and review-to-manifest alignment",
            "recomputed agreement, conservative counts, and queue accounting",
            "recomputed 52-row adjudicated counts and 13-object/3-stuff target accounting",
        ],
        "forbidden_claims": [
            "human ground-truth accuracy",
            "formal object recall",
            "independent human adjudication",
            "E14 winner",
        ],
    }
    output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
