#!/usr/bin/env python3
"""Summarize the 20-target retrospective safety-candidate validation."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    REPOSITORY_ROOT / "experiments/g1_20260724_473_573_v1_1"
)
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT
    / "comparisons/e13_e14_safety_validation_20260730"
)
DEFAULT_OLD_AUDIT = (
    EXPERIMENT_ROOT / "comparisons/codex_approx_gt_e11_e14_20260730"
)
DEFAULT_SAFETY_RUN = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e13_e14_safety_ablation_20260730"
)
DEFAULT_E14_RUN = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e14_safe035_unique_obs8_seed0_20260730"
)
STATUSES = ("success", "partial", "failure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--old-audit", type=Path, default=DEFAULT_OLD_AUDIT)
    parser.add_argument("--safety-run", type=Path, default=DEFAULT_SAFETY_RUN)
    parser.add_argument("--e14-run", type=Path, default=DEFAULT_E14_RUN)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def reference(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def status_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    counts = Counter(str(row[key]) for row in rows)
    total = len(rows)
    return {
        "total": total,
        "success": counts["success"],
        "partial": counts["partial"],
        "failure": counts["failure"],
        "strict_success_rate": counts["success"] / total,
        "lenient_success_rate": (
            counts["success"] + counts["partial"]
        ) / total,
    }


def transition_rows(
    rows: Sequence[Mapping[str, Any]], old_key: str, new_key: str, stage: str
) -> list[dict[str, Any]]:
    counts = Counter((str(row[old_key]), str(row[new_key])) for row in rows)
    return [
        {
            "stage": stage,
            "old_status": old,
            "new_status": new,
            "count": counts[(old, new)],
        }
        for old in STATUSES
        for new in STATUSES
        if counts[(old, new)]
    ]


def build_figure(
    output: Path,
    summaries: Mapping[str, Mapping[str, Any]],
) -> None:
    labels = ["old E13", "safe E13", "old E14", "new E14"]
    keys = ["old_e13", "new_e13", "old_e14", "new_e14"]
    strict = [summaries[key]["strict_success_rate"] for key in keys]
    lenient = [summaries[key]["lenient_success_rate"] for key in keys]
    failures = [summaries[key]["failure"] for key in keys]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(x - 0.18, strict, 0.36, label="strict")
    axes[0].bar(x + 0.18, lenient, 0.36, label="lenient")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("20-target diagnostic rate")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, failures, color=["#999999", "#3b82f6", "#999999", "#3b82f6"])
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("hard failures / 20")
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(failures):
        axes[1].text(index, value + 0.15, str(value), ha="center")
    path = output / "visualizations/target_status_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(
    output: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    engineering: Mapping[str, Any],
) -> None:
    old_e13 = summaries["old_e13"]
    new_e13 = summaries["new_e13"]
    old_e14 = summaries["old_e14"]
    new_e14 = summaries["new_e14"]
    report = f"""# safe E13 + unique-safe E14 实验验证

## 结论

在同一组 20 个冻结显著物体上，`safe + 0.35 m` 的 E13 近似真值结果由旧方案的
{old_e13['success']} 成功 / {old_e13['partial']} 部分 / {old_e13['failure']} 失败，
变为 **{new_e13['success']} / {new_e13['partial']} / {new_e13['failure']}**。
E13 硬失败从 {old_e13['failure']}/20 降为
**{new_e13['failure']}/20**；这与全量账本中同帧额外-track冲突由 581 降为 0
相互印证。

经过真实 DAM 后，E14 从 {old_e14['success']} / {old_e14['partial']} /
{old_e14['failure']} 变为 **{new_e14['success']} / {new_e14['partial']} /
{new_e14['failure']}**。严格成功率从 {old_e14['strict_success_rate']:.1%}
升至 **{new_e14['strict_success_rate']:.1%}**；宽松可用率从
{old_e14['lenient_success_rate']:.1%} 升至
**{new_e14['lenient_success_rate']:.1%}**，硬失败从
{old_e14['failure']} 降至 **{new_e14['failure']}**。

这说明修复最显著的价值是避免 E13 把上游可用物体破坏成硬失败；严格语义只增加
{new_e14['success'] - old_e14['success']} 个，原因是 E11 part-only、E12 ID
复用和 DAM 属性幻觉仍然存在。

## 工程链路结果

- 旧基线 obs=8：{engineering['old_eligible_entities']} 个 eligible entity、
  {engineering['old_mask_requests']} 个 DAM mask，其中
  {engineering['old_duplicate_masks']} 个为同实体重复 mask。
- 新候选：{engineering['new_eligible_entities']} 个 eligible entity、
  {engineering['new_mask_requests']} 个 DAM mask，重复 mask 为
  **{engineering['new_duplicate_masks']}**。
- 新候选真实 DAM 返回 {engineering['new_responses']} 条非空响应，
  {engineering['new_corrections_applied']} 条 correction 全部写入，SQLite
  `integrity_check={engineering['sqlite_integrity_check']}`。
- 5 个 batch 的总 DAM 延迟为 {engineering['total_dam_latency_s']:.3f} s。

## 仍未解决的失败

1. 后排两把蓝灰椅仍因 part-only/ID-switch 没有可用完整实体；一个 805 px 桌上小条
   甚至被 DAM 描述成 chair。
2. 白色货架不再与黑柜错并，但仍只有 tag/drawer/mug 等部件实体。
3. 小盆栽的叶片与花盆仍分属两个实体。
4. 四个墙面糖果盒已经逐个分开，但 DAM 只给 generic box，颜色/功能不严格。
5. 桌面篮子触发 mask 和文本正确，但 E30 历史仍含 5 条桌面碎片；几何身份未完全纯化。
6. 整套墙面标语仍依赖整墙 mask 与单词碎片，OCR 结果不能弥补错误边界。

## 证据与限制

- `TARGET_COMPARISON.*`：20 个物体逐项旧/新状态、实体、原因。
- `EVIDENCE_INDEX.jsonl`：每个主证据 panel 的路径与 SHA-256。
- E13 全量证据：9 个 SQLite、2,886×9 条 merge event、909 张 RGB overlay。
- E14 全量证据：87 个原 mask/crop/panel、87 条原始 DAM response、修订后 SQLite。

这是单一 Codex 视觉复核形成的**诊断估计**，且协议为事后审计；没有双人独立标注、
裁决或 held-out GT。因此不得把上述比例写成正式模型准确率，也不得据此宣布生产
winner。`safe+0.35 m` 是下一轮 held-out 的优先候选，不是已冻结最佳参数。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    review_path = args.output / "TARGET_REVIEW.jsonl"
    protocol_path = args.output / "AUDIT_PROTOCOL.json"
    reviews = read_jsonl(review_path)
    old_rows = read_jsonl(args.old_audit / "OBJECT_INSTANCE_CENSUS.jsonl")
    old_by_id = {str(row["instance_id"]): row for row in old_rows}
    if len(reviews) != 20 or len({row["instance_id"] for row in reviews}) != 20:
        raise ValueError("target review must contain exactly 20 unique instances")
    if set(old_by_id) != {str(row["instance_id"]) for row in reviews}:
        raise ValueError("target review population differs from frozen census")

    census_root = args.e14_run / "annotation_census_obs_08_seed_0"
    manifest = read_jsonl(census_root / "CENSUS_MANIFEST.jsonl")
    panel_by_ordinal = {
        int(row["entity_ordinal"]): row for row in manifest
    }
    rows = []
    evidence = []
    for review in reviews:
        old = old_by_id[str(review["instance_id"])]
        primary = [int(value) for value in review["primary_evidence_entities"]]
        for ordinal in primary:
            panel = panel_by_ordinal.get(ordinal)
            if panel is None:
                continue
            evidence.append(
                {
                    "schema": "daaam.g1_e13_e14_safety_evidence.v1",
                    "instance_id": review["instance_id"],
                    "entity_ordinal": ordinal,
                    "review_panel_path": panel["review_panel_path"],
                    "review_panel_sha256": panel["review_panel_sha256"],
                }
            )
        rows.append(
            {
                "schema": "daaam.g1_e13_e14_safety_target_comparison.v1",
                "instance_id": review["instance_id"],
                "target_id": old["target_id"],
                "instance_name": old["instance_name"],
                "old_e13_status": old["stages"]["E13"],
                "new_e13_status": review["new_e13_status"],
                "old_e14_status": old["stages"]["E14"],
                "new_e14_status": review["new_e14_status"],
                "new_entity_ordinals_json": json.dumps(
                    review["entity_ordinals"]
                ),
                "primary_evidence_entities_json": json.dumps(primary),
                "reason_e13": review["reason_e13"],
                "reason_e14": review["reason_e14"],
                "review_basis": "retrospective_codex_approximate_gt",
            }
        )

    summaries = {
        "old_e13": status_summary(rows, "old_e13_status"),
        "new_e13": status_summary(rows, "new_e13_status"),
        "old_e14": status_summary(rows, "old_e14_status"),
        "new_e14": status_summary(rows, "new_e14_status"),
    }
    transitions = [
        *transition_rows(rows, "old_e13_status", "new_e13_status", "E13"),
        *transition_rows(rows, "old_e14_status", "new_e14_status", "E14"),
    ]
    old_e14_summary = json.loads(
        (
            EXPERIMENT_ROOT
            / "runs/diagnostic_gt_free_e14_e13fed_dam_20260729"
            / "tables/threshold_summary.json"
        ).read_text()
    )
    old_obs8 = next(
        row
        for row in old_e14_summary
        if int(row["threshold_observations"]) == 8
    )
    new_cell = json.loads(
        (args.e14_run / "tables/cell_summary.json").read_text()
    )[0]
    engineering = {
        "old_eligible_entities": old_obs8["eligible_entity_count"],
        "old_mask_requests": old_obs8["prompt_mask_request_count"],
        "old_duplicate_masks": old_obs8[
            "same_entity_extra_mask_request_count"
        ],
        "new_eligible_entities": new_cell["eligible_entity_count"],
        "new_mask_requests": new_cell["prompt_mask_request_count"],
        "new_duplicate_masks": new_cell[
            "same_entity_extra_mask_request_count"
        ],
        "new_responses": new_cell["nonempty_response_count"],
        "new_corrections_applied": new_cell["correction_applied_count"],
        "sqlite_integrity_check": new_cell["sqlite_integrity_check"],
        "total_dam_latency_s": (
            new_cell["batch_latency_s"]["mean"]
            * new_cell["batch_latency_s"]["count"]
        ),
    }
    write_json(args.output / "SUMMARY.json", {
        "schema": "daaam.g1_e13_e14_safety_validation_summary.v1",
        "status": "complete_retrospective_codex_approximate_gt",
        "target_count": len(rows),
        "stage_summaries": summaries,
        "engineering": engineering,
        "formal_accuracy_claim_permitted": False,
    })
    write_jsonl(args.output / "TARGET_COMPARISON.jsonl", rows)
    write_csv(args.output / "TARGET_COMPARISON.csv", rows)
    write_json(args.output / "TRANSITIONS.json", transitions)
    write_csv(args.output / "TRANSITIONS.csv", transitions)
    write_jsonl(args.output / "EVIDENCE_INDEX.jsonl", evidence)
    build_figure(args.output, summaries)
    write_report(args.output, summaries, engineering)

    sources = {
        "audit_protocol": reference(protocol_path),
        "target_review": reference(review_path),
        "old_object_census": reference(
            args.old_audit / "OBJECT_INSTANCE_CENSUS.jsonl"
        ),
        "e13_safety_completion": reference(
            args.safety_run / "COMPLETION.json"
        ),
        "e14_candidate_completion": reference(
            args.e14_run / "COMPLETION.json"
        ),
        "e14_census_manifest": reference(
            census_root / "CENSUS_MANIFEST.jsonl"
        ),
    }
    generated = [
        path
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and path.name != "INTEGRITY.json"
    ]
    write_json(
        args.output / "INTEGRITY.json",
        {
            "schema": "daaam.g1_e13_e14_safety_validation_integrity.v1",
            "sources": sources,
            "generated_files": [
                {
                    "relative_path": path.relative_to(args.output).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in generated
            ],
            "evidence_panel_count": len(evidence),
            "all_evidence_panel_hashes_match": all(
                sha256_file(Path(row["review_panel_path"]))
                == row["review_panel_sha256"]
                for row in evidence
            ),
        },
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
