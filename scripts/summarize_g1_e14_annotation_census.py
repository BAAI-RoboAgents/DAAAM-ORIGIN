#!/usr/bin/env python3
"""Validate and summarize the full E14 annotation census.

This script does not make review judgments. It treats REVIEW_RESULTS.jsonl and
SALIENT_OBJECT_COVERAGE.jsonl as frozen reviewer inputs, validates their finite
population coverage, and materializes deterministic summary evidence.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rate(count: int, denominator: int) -> dict[str, Any]:
    return {
        "count": count,
        "denominator": denominator,
        "fraction": count / denominator if denominator else None,
        "percent": round(100.0 * count / denominator, 1) if denominator else None,
    }


def cohen_kappa(left: list[Any], right: list[Any]) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("kappa inputs must have equal non-zero length")
    count = len(left)
    categories = set(left) | set(right)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / count
    expected = sum(
        (left.count(category) / count) * (right.count(category) / count)
        for category in categories
    )
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    return {
        "observed_agreement": observed,
        "expected_agreement": expected,
        "cohen_kappa": kappa,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--census-dir",
        type=Path,
        required=True,
        help="annotation_census_obs_08_seed_0 directory",
    )
    args = parser.parse_args()
    census_dir = args.census_dir.resolve()

    prereg_path = census_dir / "PRE_REGISTRATION.json"
    manifest_path = census_dir / "CENSUS_MANIFEST.jsonl"
    review_a_path = census_dir / "REVIEW_RESULTS_REVIEWER_A.jsonl"
    review_b_path = census_dir / "REVIEW_RESULTS_REVIEWER_B.jsonl"
    salient_path = census_dir / "SALIENT_OBJECT_COVERAGE.jsonl"
    prereg = read_json(prereg_path)
    manifest = read_jsonl(manifest_path)
    review_sets = {
        "reviewer_a": read_jsonl(review_a_path),
        "reviewer_b": read_jsonl(review_b_path),
    }
    salient = read_jsonl(salient_path)
    expected = int(prereg["finite_population_count"])
    identity_fields = ("census_index", "entity_id", "entity_ordinal")
    manifest_key = {
        tuple(row[field] for field in identity_fields) for row in manifest
    }
    if len(manifest) != expected or len(manifest_key) != expected:
        raise ValueError("manifest does not contain the unique finite population")
    panel_count = 0
    mask_record_count = 0
    rgb_hash_checks = 0
    mask_hash_checks = 0
    for row in manifest:
        panel = Path(row["review_panel_path"])
        if not panel.is_file() or sha256(panel) != row["review_panel_sha256"]:
            raise ValueError(f"review panel identity mismatch: {panel}")
        panel_count += 1
        rgb = Path(row["rgb_path"])
        if not rgb.is_file() or sha256(rgb) != row["rgb_sha256"]:
            raise ValueError(f"RGB identity mismatch: {rgb}")
        rgb_hash_checks += 1
        for mask_record in row["mask_records"]:
            mask = Path(mask_record["materialized_mask_path"])
            if (
                not mask.is_file()
                or sha256(mask) != mask_record["materialized_mask_sha256"]
            ):
                raise ValueError(f"mask identity mismatch: {mask}")
            mask_record_count += 1
            mask_hash_checks += 1

    semantic_allowed = set(prereg["semantic_verdicts"])
    mask_allowed = set(prereg["mask_verdicts"])
    for reviewer, rows in review_sets.items():
        review_key = {
            tuple(row[field] for field in identity_fields) for row in rows
        }
        if len(rows) != expected or review_key != manifest_key:
            raise ValueError(f"{reviewer} does not exactly match the manifest")
        for row in rows:
            if row["semantic_verdict"] not in semantic_allowed:
                raise ValueError(f"invalid semantic verdict in {reviewer}: {row}")
            if row["mask_verdict"] not in mask_allowed:
                raise ValueError(f"invalid mask verdict in {reviewer}: {row}")
            if not str(row.get("review_notes", "")).strip():
                raise ValueError(f"missing review note in {reviewer}: {row}")

    by_reviewer = {
        reviewer: {
            tuple(row[field] for field in identity_fields): row for row in rows
        }
        for reviewer, rows in review_sets.items()
    }
    severity = {
        "correct": 0,
        "partially_correct": 1,
        "incorrect": 2,
        "unjudgeable": 3,
    }
    consensus: list[dict[str, Any]] = []
    for item in sorted(manifest, key=lambda row: int(row["census_index"])):
        key = tuple(item[field] for field in identity_fields)
        left = by_reviewer["reviewer_a"][key]
        right = by_reviewer["reviewer_b"][key]
        conservative = max(
            (left["semantic_verdict"], right["semantic_verdict"]),
            key=severity.__getitem__,
        )
        consensus.append(
            {
                **{field: item[field] for field in identity_fields},
                "reviewer_a": {
                    key: left[key]
                    for key in (
                        "artifact_complete",
                        "mask_acceptable",
                        "mask_verdict",
                        "semantic_complete",
                        "semantic_verdict",
                        "main_identity_correct",
                        "review_notes",
                    )
                },
                "reviewer_b": {
                    key: right[key]
                    for key in (
                        "artifact_complete",
                        "mask_acceptable",
                        "mask_verdict",
                        "semantic_complete",
                        "semantic_verdict",
                        "main_identity_correct",
                        "review_notes",
                    )
                },
                "artifact_complete_both": bool(left["artifact_complete"])
                and bool(right["artifact_complete"]),
                "mask_acceptable_both": bool(left["mask_acceptable"])
                and bool(right["mask_acceptable"]),
                "semantic_complete_both": bool(left["semantic_complete"])
                and bool(right["semantic_complete"]),
                "main_identity_correct_both": bool(left["main_identity_correct"])
                and bool(right["main_identity_correct"]),
                "semantic_verdict_agreement": left["semantic_verdict"]
                == right["semantic_verdict"],
                "semantic_verdict_conservative": conservative,
            }
        )
    write_jsonl(census_dir / "REVIEW_CONSENSUS.jsonl", consensus)

    semantic_a = [row["semantic_verdict"] for row in review_sets["reviewer_a"]]
    semantic_b = [row["semantic_verdict"] for row in review_sets["reviewer_b"]]
    mask_a = [bool(row["mask_acceptable"]) for row in review_sets["reviewer_a"]]
    mask_b = [bool(row["mask_acceptable"]) for row in review_sets["reviewer_b"]]
    main_a = [
        bool(row["main_identity_correct"]) for row in review_sets["reviewer_a"]
    ]
    main_b = [
        bool(row["main_identity_correct"]) for row in review_sets["reviewer_b"]
    ]
    semantic_agreement = cohen_kappa(semantic_a, semantic_b)
    mask_agreement = cohen_kappa(mask_a, mask_b)
    main_agreement = cohen_kappa(main_a, main_b)
    conservative_counts = collections.Counter(
        row["semantic_verdict_conservative"] for row in consensus
    )
    strict = conservative_counts["correct"]
    lenient = strict + conservative_counts["partially_correct"]
    both_mask = sum(row["mask_acceptable_both"] for row in consensus)
    joint_strict_mask = sum(
        row["mask_acceptable_both"]
        and row["semantic_verdict_conservative"] == "correct"
        for row in consensus
    )
    joint_lenient_mask = sum(
        row["mask_acceptable_both"]
        and row["semantic_verdict_conservative"]
        in {"correct", "partially_correct"}
        for row in consensus
    )

    reviewer_stats: dict[str, Any] = {}
    for reviewer, rows in review_sets.items():
        semantic_counts = collections.Counter(
            row["semantic_verdict"] for row in rows
        )
        reviewer_stats[reviewer] = {
            "semantic_verdict_counts": dict(sorted(semantic_counts.items())),
            "mask_acceptable": rate(
                sum(bool(row["mask_acceptable"]) for row in rows), expected
            ),
            "strict_visual_correctness": rate(
                semantic_counts["correct"], expected - semantic_counts["unjudgeable"]
            ),
            "lenient_visual_correctness": rate(
                semantic_counts["correct"]
                + semantic_counts["partially_correct"],
                expected - semantic_counts["unjudgeable"],
            ),
            "main_identity_correct": rate(
                sum(bool(row["main_identity_correct"]) for row in rows), expected
            ),
        }

    salient_allowed = {
        "clean_full_entity",
        "covered_fragmented_or_conflicted",
        "missed_as_whole",
    }
    if (
        not salient
        or len({row["salient_id"] for row in salient}) != len(salient)
        or {row["coverage_status"] for row in salient} - salient_allowed
    ):
        raise ValueError("invalid salient-object coverage inventory")
    salient_evidence_checks = 0
    for row in salient:
        for evidence in row["evidence"]:
            evidence_path = (census_dir / evidence).resolve()
            if not evidence_path.is_file():
                raise ValueError(f"missing salient evidence: {evidence_path}")
            salient_evidence_checks += 1
    salient_counts = collections.Counter(row["coverage_status"] for row in salient)

    summary = {
        "schema": "daaam.g1_no_gt_e14_annotation_census_summary.v2",
        "scope": prereg["scope"],
        "population": {
            "eligible_final_entities": expected,
            "reviewed_unique_entities_per_reviewer": expected,
            "review_passes": 2,
            "coverage_per_reviewer": rate(expected, expected),
            "stopping_rule_satisfied": True,
            "sampling_uncertainty_for_this_finite_population": "none; full census",
        },
        "reviewers": reviewer_stats,
        "inter_reviewer_agreement": {
            "semantic_verdict": semantic_agreement,
            "mask_acceptable": mask_agreement,
            "main_identity_correct": main_agreement,
            "semantic_exact_agreement": rate(
                sum(row["semantic_verdict_agreement"] for row in consensus),
                expected,
            ),
            "semantic_disagreement_count": sum(
                not row["semantic_verdict_agreement"] for row in consensus
            ),
        },
        "conservative_consensus": {
            "rule": (
                "strict requires both reviewers=correct; lenient requires neither "
                "reviewer=incorrect; mask acceptable requires both=True"
            ),
            "artifact_complete_both": rate(
                sum(row["artifact_complete_both"] for row in consensus), expected
            ),
            "semantic_complete_both": rate(
                sum(row["semantic_complete_both"] for row in consensus), expected
            ),
            "semantic_verdict_counts": dict(sorted(conservative_counts.items())),
            "strict_visual_correctness": rate(strict, expected),
            "lenient_visual_correctness": rate(lenient, expected),
            "mask_acceptable_both": rate(both_mask, expected),
            "main_identity_correct_both": rate(
                sum(row["main_identity_correct_both"] for row in consensus),
                expected,
            ),
            "mask_acceptable_and_strict_correct": rate(
                joint_strict_mask, expected
            ),
            "mask_acceptable_and_lenient_correct": rate(
                joint_lenient_mask, expected
            ),
        },
        "salient_object_inventory": {
            "scope": (
                "15 diagnostic prominent foreground objects/groups visible "
                "across the audited sequence"
            ),
            "count": len(salient),
            "status_counts": dict(sorted(salient_counts.items())),
            "clean_full_entity": rate(
                salient_counts["clean_full_entity"], len(salient)
            ),
            "at_least_partly_covered": rate(
                len(salient) - salient_counts["missed_as_whole"], len(salient)
            ),
            "missed_as_whole": rate(
                salient_counts["missed_as_whole"], len(salient)
            ),
            "all_prominent_objects_cleanly_recognized": False,
        },
        "conclusion": {
            "engineering": (
                "Both complete review passes independently yield 22/52 strict "
                "correct. Conservative agreement yields 19/52 strict correct, "
                "28/52 masks accepted by both, and 13/52 jointly strict with "
                "a mask accepted by both. Prominent objects are not all cleanly "
                "recognized."
            ),
            "formal_accuracy_claim_permitted": False,
            "limitation": (
                "Two Codex visual review passes are not two human annotators or "
                "independent adjudication; there is no sealed held-out human GT."
            ),
            "salient_inventory_limitation": (
                "The 15-item salient inventory is diagnostic and not formal GT."
            ),
        },
    }
    write_json(census_dir / "AUDIT_SUMMARY.json", summary)

    disagreements = [
        row
        for row in consensus
        if not row["semantic_verdict_agreement"]
        or row["reviewer_a"]["mask_acceptable"]
        != row["reviewer_b"]["mask_acceptable"]
        or row["reviewer_a"]["main_identity_correct"]
        != row["reviewer_b"]["main_identity_correct"]
    ]
    write_jsonl(census_dir / "REVIEW_DISAGREEMENTS.jsonl", disagreements)
    failures = [
        {
            **{field: row[field] for field in identity_fields},
            "mask_acceptable_both": row["mask_acceptable_both"],
            "semantic_verdict_conservative": row[
                "semantic_verdict_conservative"
            ],
            "semantic_complete_both": row["semantic_complete_both"],
            "main_identity_correct_both": row["main_identity_correct_both"],
            "reviewer_a": row["reviewer_a"],
            "reviewer_b": row["reviewer_b"],
        }
        for row in consensus
        if not row["mask_acceptable_both"]
        or row["semantic_verdict_conservative"] != "correct"
        or not row["semantic_complete_both"]
        or not row["main_identity_correct_both"]
    ]
    write_jsonl(census_dir / "FAILURE_CASES.jsonl", failures)

    semantic_percent = 100.0 * semantic_agreement["observed_agreement"]
    mask_percent = 100.0 * mask_agreement["observed_agreement"]
    report = f"""# E14 obs=8 seed=0 全量双遍视觉复核

## 结论

**不是所有非常明显的物体都被干净识别。** 当前有限总体已由两套 Codex 视觉审阅记录
各自完成 52/52 全量复核，因此不再受原 6/52 小样本抽查的抽样波动支配。

两个审阅遍次分别都判定严格正确 22/52（42.3%），但逐实体只有 19/52（36.5%）
得到双方一致的“正确”。采用保守共识口径后：

- 证据产物双方均完整：52/52（100%）
- mask 双方均接受：{both_mask}/52（{100.0 * both_mask / expected:.1f}%）
- 严格正确（双方都判正确）：{strict}/52（{100.0 * strict / expected:.1f}%）
- 宽松正确（双方均未判错误）：{lenient}/52（{100.0 * lenient / expected:.1f}%）
- mask 双方接受且严格正确：{joint_strict_mask}/52（{100.0 * joint_strict_mask / expected:.1f}%）

这足以形成稳健的否定结论：DAM 返回非空文本不等于正确语义实体。主要问题来自
E11/E12/E13 遗留的切碎、粘连、跨实例合并，以及 E14 对局部区域的过度描述和属性臆测。

## 两遍审阅的一致性

| 指标 | 完全一致 | Cohen κ |
| --- | ---: | ---: |
| 语义三分类 | {sum(row['semantic_verdict_agreement'] for row in consensus)}/52（{semantic_percent:.1f}%） | {semantic_agreement['cohen_kappa']:.3f} |
| mask 是否可接受 | {sum(a == b for a, b in zip(mask_a, mask_b, strict=True))}/52（{mask_percent:.1f}%） | {mask_agreement['cohen_kappa']:.3f} |
| 主体身份是否正确 | {sum(a == b for a, b in zip(main_a, main_b, strict=True))}/52（{100.0 * main_agreement['observed_agreement']:.1f}%） | {main_agreement['cohen_kappa']:.3f} |

语义判定有 {sum(not row['semantic_verdict_agreement'] for row in consensus)} 项分歧，已全部保存到
`REVIEW_DISAGREEMENTS.jsonl`，没有用单方判断静默覆盖。

## 保守共识语义分布

| 判定 | 数量 | 比例 |
| --- | ---: | ---: |
| 双方都正确 | {conservative_counts['correct']} | {100.0 * conservative_counts['correct'] / expected:.1f}% |
| 至少一方部分正确、且无人判错 | {conservative_counts['partially_correct']} | {100.0 * conservative_counts['partially_correct'] / expected:.1f}% |
| 至少一方判错 | {conservative_counts['incorrect']} | {100.0 * conservative_counts['incorrect'] / expected:.1f}% |
| 至少一方不可判断 | {conservative_counts['unjudgeable']} | {100.0 * conservative_counts['unjudgeable'] / expected:.1f}% |

## 显著物体场景级覆盖

对跨帧可见、面积较大或语义显著的 15 个前景物体/物体组建立诊断清单：

- 干净完整实体：{salient_counts['clean_full_entity']}/15
- 有覆盖但碎片化、重复或冲突：{salient_counts['covered_fragmented_or_conflicted']}/15
- 作为整体漏识：{salient_counts['missed_as_whole']}/15

明确的整体漏识包括：左侧大型黑色展示/冷藏柜、贯穿场景的长条高桌、桌上浅棕色
托盘/篮筐。长桌只留下 E83 桌腿；出现“藤编篮子”文本的 E18 掩码主要覆盖桌面、
地面和右侧大区域，不能算正确定位。大型植物、白色书架、椅子组和两台售货机虽然
至少有相关描述，但存在切碎、重复、跨实例合并或属性冲突。

## 证据

- `review_panels/`：52 张逐实体 RGB、同帧全部 mask response 与最终描述面板。
- `REVIEW_RESULTS_REVIEWER_A.jsonl`、`REVIEW_RESULTS_REVIEWER_B.jsonl`：两套原始逐项判定。
- `REVIEW_CONSENSUS.jsonl`：逐实体并列保存两遍判断及保守共识。
- `REVIEW_DISAGREEMENTS.jsonl`：所有语义、mask 或主体判定分歧。
- `FAILURE_CASES.jsonl`：未同时满足双方 mask、语义、主体与完整性要求的风险项。
- `SALIENT_OBJECT_COVERAGE.jsonl`：15 个显著物体/物体组覆盖清单。
- `AUDIT_SUMMARY.json`、`EVIDENCE_INTEGRITY.json`：机器可读统计与哈希核验。

## 结论边界

52/52 双遍复核消除了当前 obs=8、seed=0、52 个 eligible final entity 有限总体内的
抽样误差，并量化了 Codex 审阅口径差异；但两遍 Codex 复核不等于两名独立人工标注员，
也没有人工裁决或 held-out GT。因此这些是工程诊断统计，不得写成正式 accuracy、
recall 或生产验收指标。15 项显著物体清单同样是诊断枚举，不是封存 GT。
"""
    (census_dir / "REPORT.md").write_text(report, encoding="utf-8")

    generated = [
        census_dir / "REVIEW_CONSENSUS.jsonl",
        census_dir / "REVIEW_DISAGREEMENTS.jsonl",
        census_dir / "AUDIT_SUMMARY.json",
        census_dir / "FAILURE_CASES.jsonl",
        census_dir / "REPORT.md",
    ]
    integrity = {
        "schema": "daaam.g1_no_gt_e14_annotation_census_integrity.v2",
        "status": "PASS",
        "checks": {
            "finite_population_expected": expected,
            "manifest_rows": len(manifest),
            "reviewer_a_rows": len(review_sets["reviewer_a"]),
            "reviewer_b_rows": len(review_sets["reviewer_b"]),
            "both_reviews_exactly_match_manifest": True,
            "consensus_rows": len(consensus),
            "review_panel_hash_checks": panel_count,
            "rgb_hash_checks": rgb_hash_checks,
            "materialized_mask_hash_checks": mask_hash_checks,
            "materialized_mask_records": mask_record_count,
            "salient_inventory_rows": len(salient),
            "salient_evidence_path_checks": salient_evidence_checks,
            "all_verdicts_in_preregistered_vocabulary": True,
        },
        "inputs": {
            str(path.relative_to(census_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (
                prereg_path,
                manifest_path,
                review_a_path,
                review_b_path,
                salient_path,
            )
        },
        "generated": {
            str(path.relative_to(census_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in generated
        },
    }
    write_json(census_dir / "EVIDENCE_INTEGRITY.json", integrity)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
