#!/usr/bin/env python3
"""Evaluate the GT-free E15 geometry -> frontend -> DAM increment chain."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    REPOSITORY_ROOT / "experiments/g1_20260724_473_573_v1_1"
)
DEFAULT_E11_RUN = (
    EXPERIMENT_ROOT / "runs/diagnostic_gt_free_e11_fastsam_20260729"
)
DEFAULT_E12_RUN = (
    EXPERIMENT_ROOT / "runs/diagnostic_gt_free_e12_e11fed_botsort_20260729"
)
DEFAULT_E13_RUN = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e13_e14_safety_ablation_20260730"
)
DEFAULT_E14_RUN = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e14_safe035_unique_obs8_seed0_20260730"
)
DEFAULT_OLD_CENSUS = (
    EXPERIMENT_ROOT
    / "comparisons/codex_approx_gt_e11_e14_20260730"
    / "OBJECT_INSTANCE_CENSUS.jsonl"
)
DEFAULT_TARGET_COMPARISON = (
    EXPERIMENT_ROOT
    / "comparisons/e13_e14_safety_validation_20260730"
    / "TARGET_COMPARISON.jsonl"
)
DEFAULT_PROTOCOL_QUERY_SET = EXPERIMENT_ROOT / "protocol/query_set.json"
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e15_safe035_increment_20260730"
)
DEFAULT_SENTENCE_MODEL = (
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
DEFAULT_TOP_K = (1, 3, 5, 10)
INVENTORY_EXCLUDES = {
    "artifact_inventory.csv",
    "artifact_inventory.jsonl",
    "inventory_summary.json",
    "COMPLETION.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run E15 on the frozen 473-573 semantic chain. Formal L5 GT/query "
            "metrics remain unavailable; target and retrieval results are proxies."
        )
    )
    parser.add_argument("--e11-run", type=Path, default=DEFAULT_E11_RUN)
    parser.add_argument("--e12-run", type=Path, default=DEFAULT_E12_RUN)
    parser.add_argument("--e13-run", type=Path, default=DEFAULT_E13_RUN)
    parser.add_argument("--e14-run", type=Path, default=DEFAULT_E14_RUN)
    parser.add_argument("--old-census", type=Path, default=DEFAULT_OLD_CENSUS)
    parser.add_argument(
        "--target-comparison",
        type=Path,
        default=DEFAULT_TARGET_COMPARISON,
    )
    parser.add_argument(
        "--protocol-query-set",
        type=Path,
        default=DEFAULT_PROTOCOL_QUERY_SET,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    parser.add_argument(
        "--minimum-similarity",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=list(DEFAULT_TOP_K),
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Permit a missing sentence-transformer model to be downloaded.",
    )
    parser.add_argument(
        "--reseal-existing",
        action="store_true",
        help="Only rebuild inventory and completion after independent audit.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def frozen_reference(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def status_counts(values: Iterable[str]) -> dict[str, Any]:
    ordered = list(values)
    counts = {
        status: sum(value == status for value in ordered)
        for status in ("success", "partial", "failure")
    }
    total = len(ordered)
    return {
        "total": total,
        **counts,
        "strict_success_rate": counts["success"] / total if total else None,
        "lenient_success_rate": (
            (counts["success"] + counts["partial"]) / total if total else None
        ),
    }


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ("git", *args),
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_cache_revision(model_name: str) -> dict[str, Any]:
    cache_name = "models--" + model_name.replace("/", "--")
    cache_root = Path.home() / ".cache/huggingface/hub" / cache_name
    reference = cache_root / "refs/main"
    result: dict[str, Any] = {
        "cache_root": str(cache_root),
        "cache_exists": cache_root.is_dir(),
    }
    if reference.is_file():
        result["revision"] = reference.read_text(encoding="utf-8").strip()
        result["reference"] = frozen_reference(reference)
    return result


def inventory_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in INVENTORY_EXCLUDES:
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


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


def seal_output(
    root: Path,
    *,
    status: str = "complete_pending_independent_audit",
) -> None:
    rows = inventory_rows(root)
    write_jsonl(root / "artifact_inventory.jsonl", rows)
    write_csv(root / "artifact_inventory.csv", rows)
    root_hash = inventory_root(rows)
    summary = {
        "schema": "daaam.g1_no_gt_e15_inventory.v1",
        "generated_at": utc_now(),
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "inventory_root_sha256": root_hash,
    }
    write_json(root / "inventory_summary.json", summary)
    audited = status == "complete_independently_audited"
    write_json(
        root / "COMPLETION.json",
        {
            "schema": "daaam.g1_no_gt_e15_completion.v1",
            "status": status,
            "generated_at": utc_now(),
            "artifact_inventory_root_sha256": root_hash,
            "artifact_inventory_file_count": len(rows),
            "formal_claims_permitted": False,
            "independent_audit": "passed" if audited else "pending",
        },
    )


def verify_source_status(path: Path, accepted: set[str]) -> None:
    status = str(read_json(path)["status"])
    if status not in accepted:
        raise ValueError(f"source is not complete: {path}: {status}")


def selected_row(
    rows: Iterable[Mapping[str, Any]],
    key: str,
    value: Any,
) -> dict[str, Any]:
    selected = [dict(row) for row in rows if row.get(key) == value]
    if len(selected) != 1:
        raise ValueError(f"expected one row where {key}={value!r}, got {len(selected)}")
    return selected[0]


def retrieval_rows(
    targets: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    top_k: Sequence[int],
    minimum_similarity: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    maximum_k = max(top_k)
    rows = []
    for target_index, (target, row_scores) in enumerate(zip(targets, scores)):
        order = np.argsort(-row_scores, kind="stable")
        relevant = {
            int(value)
            for value in json.loads(
                str(target["primary_evidence_entities_json"])
            )
        }
        relevant_ranks = [
            rank
            for rank, label_index in enumerate(order, start=1)
            if int(labels[int(label_index)]["entity_ordinal"]) in relevant
        ]
        best_relevant_rank = min(relevant_ranks) if relevant_ranks else None
        top_score = float(row_scores[int(order[0])])
        rankings = [
            {
                "rank": rank,
                "entity_id": str(labels[int(label_index)]["entity_id"]),
                "entity_ordinal": int(
                    labels[int(label_index)]["entity_ordinal"]
                ),
                "score": float(row_scores[int(label_index)]),
                "is_relevant_proxy": int(
                    labels[int(label_index)]["entity_ordinal"]
                )
                in relevant,
                "description": str(labels[int(label_index)]["final_label"]),
            }
            for rank, label_index in enumerate(order[:maximum_k], start=1)
        ]
        result = {
            "schema": "daaam.g1_no_gt_e15_query_proxy.v1",
            "target_index": target_index,
            "instance_id": str(target["instance_id"]),
            "query": str(target["instance_name"]),
            "relevant_entity_ordinals_json": json.dumps(sorted(relevant)),
            "top_score": top_score,
            "passes_minimum_similarity": top_score >= minimum_similarity,
            "best_relevant_rank": best_relevant_rank,
            "hit_at_1": best_relevant_rank is not None and best_relevant_rank <= 1,
            "hit_at_3": best_relevant_rank is not None and best_relevant_rank <= 3,
            "hit_at_5": best_relevant_rank is not None and best_relevant_rank <= 5,
            "hit_at_10": (
                best_relevant_rank is not None and best_relevant_rank <= 10
            ),
            "rankings_json": json.dumps(rankings, ensure_ascii=False),
            "evaluation_basis": (
                "posthoc_codex_approximate_gt_positive_query_proxy"
            ),
        }
        rows.append(result)

    metrics: dict[str, Any] = {
        "schema": "daaam.g1_no_gt_e15_query_proxy_summary.v1",
        "query_count": len(rows),
        "candidate_count": len(labels),
        "minimum_similarity": minimum_similarity,
        "positive_queries_only": True,
        "formal_query_accuracy": None,
        "formal_false_accept_rate": None,
        "evaluation_basis": (
            "posthoc Codex approximate target-to-entity relevance; no L5 GT"
        ),
        "recall_at_k_proxy": {},
        "thresholded_recall_at_k_proxy": {},
    }
    for value in top_k:
        raw = sum(
            row["best_relevant_rank"] is not None
            and int(row["best_relevant_rank"]) <= value
            for row in rows
        )
        thresholded = sum(
            row["best_relevant_rank"] is not None
            and int(row["best_relevant_rank"]) <= value
            and bool(row["passes_minimum_similarity"])
            for row in rows
        )
        metrics["recall_at_k_proxy"][str(value)] = raw / len(rows)
        metrics["thresholded_recall_at_k_proxy"][str(value)] = (
            thresholded / len(rows)
        )
    return rows, metrics


def make_visualizations(
    output: Path,
    e11: Mapping[str, Any],
    e12: Mapping[str, Any],
    e13: Mapping[str, Any],
    e14: Mapping[str, Any],
    stage_status: Mapping[str, Mapping[str, Any]],
    query_summary: Mapping[str, Any],
    cost: Mapping[str, Any],
) -> None:
    visualizations = output / "visualizations"
    watermark = "NO HUMAN GT / PROXY"

    labels = ("E11 masks", "E12 tracked", "E13 entities", "E14 named")
    values = (
        int(e11["kept_instance_count"]),
        int(e12["tracked_observation_count"]),
        int(e13["entity_count"]),
        int(e14["responded_entity_count"]),
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=("#3182bd", "#6baed6", "#74c476", "#fd8d3c"))
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Exact persisted artifact count")
    ax.set_title("E15 semantic artifact funnel (units change by layer)")
    ax.text(
        0.99,
        0.96,
        "counts are not conversion accuracy",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(visualizations / "01_artifact_funnel.png", dpi=180)
    plt.close(fig)

    stages = ("E11 mask", "E12 track", "E13 entity", "E14 DAM")
    success = [int(stage_status[key]["success"]) for key in ("E11", "E12", "E13", "E14")]
    partial = [int(stage_status[key]["partial"]) for key in ("E11", "E12", "E13", "E14")]
    failure = [int(stage_status[key]["failure"]) for key in ("E11", "E12", "E13", "E14")]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, success, label="success", color="#31a354")
    ax.bar(x, partial, bottom=success, label="partial", color="#fdae6b")
    ax.bar(
        x,
        failure,
        bottom=np.asarray(success) + np.asarray(partial),
        label="failure",
        color="#de2d26",
    )
    ax.set_xticks(x, stages)
    ax.set_ylim(0, 22)
    ax.set_ylabel("Frozen salient targets (n=20)")
    ax.set_title("Target status through Mask → Track → Entity → DAM")
    ax.legend()
    ax.text(
        0.99,
        0.02,
        watermark,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#b30000",
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(visualizations / "02_target_status_chain.png", dpi=180)
    plt.close(fig)

    query_k = [int(value) for value in query_summary["recall_at_k_proxy"]]
    raw = [
        float(query_summary["recall_at_k_proxy"][str(value)])
        for value in query_k
    ]
    thresholded = [
        float(query_summary["thresholded_recall_at_k_proxy"][str(value)])
        for value in query_k
    ]
    x = np.arange(len(query_k))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.36
    ax.bar(x - width / 2, raw, width, label="rank-only", color="#3182bd")
    ax.bar(
        x + width / 2,
        thresholded,
        width,
        label="top score ≥ 0.55",
        color="#9ecae1",
    )
    ax.set_xticks(x, [f"R@{value}" for value in query_k])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Positive-query entity hit proxy")
    ax.set_title("DAM description retrieval on 20 post-hoc Chinese queries")
    ax.legend()
    ax.text(
        0.99,
        0.02,
        watermark,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#b30000",
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(visualizations / "03_query_proxy_recall.png", dpi=180)
    plt.close(fig)

    operators = ("E11", "E12", "E13", "E14 DAM")
    milliseconds = [
        float(cost["e11"]["total_ms"]),
        float(cost["e12"]["total_ms"]),
        float(cost["e13"]["total_ms"]),
        float(cost["e14"]["total_ms"]),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(operators, milliseconds, color=("#3182bd", "#6baed6", "#74c476", "#fd8d3c"))
    ax.bar_label(bars, fmt="%.0f ms", padding=3)
    ax.set_ylabel("Observed standalone operator time (ms)")
    ax.set_title("E15 measured semantic cost on 101 frames")
    ax.text(
        0.99,
        0.96,
        "isolated timings; not end-to-end wall clock",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(visualizations / "04_incremental_cost.png", dpi=180)
    plt.close(fig)


def build_report(
    stage_status: Mapping[str, Mapping[str, Any]],
    query: Mapping[str, Any],
    cost: Mapping[str, Any],
    e11: Mapping[str, Any],
    e12: Mapping[str, Any],
    e13: Mapping[str, Any],
    e14: Mapping[str, Any],
) -> str:
    r = query["recall_at_k_proxy"]
    tr = query["thresholded_recall_at_k_proxy"]
    return f"""# E15 语义增量链实验（新 E13/E14 输入）

## 结论

本轮固定 473–573 共 {int(e11["frame_count"])} 帧，前端采用
E11 `conf=0.3, area=300, IoU=0.5`、E12 `buffer=10`，实体层采用
`safe_merge_0p35m`，DAM 采用 `unique_safe, observations=8, seed=0`。

E15 假设仅**部分成立**：前端成功把几何观察组织成可寻址实体，DAM 再把其中
{int(e14["responded_entity_count"])}/{int(e13["entity_count"])}
（{float(e14["all_entity_description_coverage"]):.1%}）变为有自然语言描述的实体，
从而首次获得文本检索能力；但严格目标状态从 E13 的
{float(stage_status["E13"]["strict_success_rate"]):.1%} 降为 E14 的
{float(stage_status["E14"]["strict_success_rate"]):.1%}。这说明 DAM 增加了功能，
没有提高严格实体正确性，且新增了描述错误。

## Mask → track → entity → query 漏斗

| 层 | 精确工程产物 | 20 目标：成功/部分/失败 | 能否文本查询 |
| --- | ---: | ---: | --- |
| geometry-only | 101 个冻结 RGB-D 帧 | 不适用 | 否，无语义候选 |
| E11 mask | {int(e11["kept_instance_count"])} 个 mask | {stage_status["E11"]["success"]}/{stage_status["E11"]["partial"]}/{stage_status["E11"]["failure"]} | 否，FastSAM 为类无关 mask |
| E12 track | {int(e12["tracked_observation_count"])} 个 tracked observations，{int(e12["unique_track_count"])} tracks | {stage_status["E12"]["success"]}/{stage_status["E12"]["partial"]}/{stage_status["E12"]["failure"]} | 否，没有自然语言描述 |
| E13 entity | {int(e13["input_geometry_observations"])} 个 3D observations，{int(e13["entity_count"])} entities | {stage_status["E13"]["success"]}/{stage_status["E13"]["partial"]}/{stage_status["E13"]["failure"]} | 否，只有实体几何和 ID |
| E14 DAM | {int(e14["responded_entity_count"])} 个 named entities | {stage_status["E14"]["success"]}/{stage_status["E14"]["partial"]}/{stage_status["E14"]["failure"]} | 是，诊断型正查询 proxy |

## 文本检索 proxy

使用缓存的 `paraphrase-multilingual-mpnet-base-v2`，以 20 个事后 Codex
目标中文名称为 query，以 87 条原始 DAM 最终描述为候选；相关实体集合冻结自此前
`TARGET_COMPARISON`。这不是 L5 查询 GT，也没有负查询，不能计算正式 Recall/MRR/FAR。

| 指标 | 仅排序 | 要求 top score ≥ 0.55 |
| --- | ---: | ---: |
| R@1 proxy | {float(r["1"]):.1%} | {float(tr["1"]):.1%} |
| R@3 proxy | {float(r["3"]):.1%} | {float(tr["3"]):.1%} |
| R@5 proxy | {float(r["5"]):.1%} | {float(tr["5"]):.1%} |
| R@10 proxy | {float(r["10"]):.1%} | {float(tr["10"]):.1%} |

R@1 只有 {int(round(float(r["1"]) * 20))}/20，说明自然语言能力已经出现，
但相似类别和部件描述仍导致明显排序歧义。R@3 升到
{int(round(float(r["3"]) * 20))}/20，表明 top-k 查询比单一 top-1 更适合当前结果。

## 增量成本

以下是隔离实验中可直接归因的算子时间，不是同一端到端进程的 wall-clock：

| 增量 | 总观测时间 | 折算到 101 帧 |
| --- | ---: | ---: |
| E11 FastSAM inference | {float(cost["e11"]["total_ms"]):.1f} ms | {float(cost["e11"]["per_frame_ms"]):.2f} ms/frame |
| E12 BotSort | {float(cost["e12"]["total_ms"]):.1f} ms | {float(cost["e12"]["per_frame_ms"]):.2f} ms/frame |
| E13 MapMemory observe | {float(cost["e13"]["total_ms"]):.1f} ms | {float(cost["e13"]["per_frame_ms"]):.2f} ms/frame |
| E14 DAM | {float(cost["e14"]["total_ms"]):.1f} ms | {float(cost["e14"]["per_frame_ms"]):.2f} ms/frame |
| 语义链算术合计 | {float(cost["cumulative"]["total_ms"]):.1f} ms | {float(cost["cumulative"]["per_frame_ms"]):.2f} ms/frame |

DAM 占已测语义算子时间的 {float(cost["e14"]["fraction_of_measured_semantic_cost"]):.1%}，
是主要成本来源。E11 postprocess、模型启动、并发/队列和 I/O 的计时口径不完全一致，
因此算术合计只用于成本归因，不得当作实时延迟。

## 判定与下一步

- `geometry → frontend`：通过工程能力门。产生稳定、可持久化的实体 ID，但没有文本查询能力。
- `frontend → DAM`：通过“新增自然语言查询能力”门；未通过“严格目标质量提升”门。
- 当前 E15 不能选 winner：正式 query set 在 reviewed L5 GT 前按协议保持空集。
- E16/E17 可以继续做几何 mesh 与实体绑定，但 Q1 的正式查询排名必须等待 L5。
- 查询侧建议默认返回 top-3 和证据图；top-1 不能作为可靠单答案。

## 证据

- `FROZEN_INPUTS.json`：所有 E11–E14、目标审计和协议输入的路径、大小与 SHA-256。
- `tables/stage_status.*`：20 个目标逐层状态。
- `tables/incremental_funnel.*`：各层产物、覆盖和查询能力。
- `query_proxy/query_rankings.*`：20 条 query 的 top-10 完整描述、分数和相关实体命中。
- `query_proxy/*.npy`：query/label embedding 与 20×87 score matrix。
- `tables/cost_summary.json`：各层独立计时和口径。
- `visualizations/`：漏斗、目标状态、检索 proxy 和成本图。
- `artifact_inventory.*`、`COMPLETION.json`：文件级封存。
"""


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if args.reseal_existing:
        if not output.is_dir():
            raise FileNotFoundError(output)
        audit_path = output / "INDEPENDENT_AUDIT.json"
        audit_passed = (
            audit_path.is_file()
            and read_json(audit_path).get("passed") is True
        )
        seal_output(
            output,
            status=(
                "complete_independently_audited"
                if audit_passed
                else "complete_pending_independent_audit"
            ),
        )
        return 0
    if output.exists():
        raise FileExistsError(f"refuse to overwrite existing output: {output}")
    if not args.top_k or any(value <= 0 for value in args.top_k):
        raise ValueError("--top-k values must be positive")
    if sorted(set(args.top_k)) != sorted(args.top_k):
        raise ValueError("--top-k values must be unique and increasing")
    if not 0.0 <= args.minimum_similarity <= 1.0:
        raise ValueError("--minimum-similarity must be in [0, 1]")

    paths = {
        "e11_completion": args.e11_run / "COMPLETION.json",
        "e11_audit": args.e11_run / "INDEPENDENT_AUDIT.json",
        "e11_summary": args.e11_run / "tables/cell_summary.json",
        "e12_completion": args.e12_run / "COMPLETION.json",
        "e12_audit": args.e12_run / "INDEPENDENT_AUDIT.json",
        "e12_summary": args.e12_run / "tables/variant_summary.json",
        "e13_completion": args.e13_run / "COMPLETION.json",
        "e13_summary": args.e13_run / "tables/variant_summary.json",
        "e14_completion": args.e14_run / "COMPLETION.json",
        "e14_summary": args.e14_run / "tables/cell_summary.json",
        "e14_labels": args.e14_run / "tables/final_labels.jsonl",
        "old_census": args.old_census,
        "target_comparison": args.target_comparison,
        "protocol_query_set": args.protocol_query_set,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing E15 inputs: " + ", ".join(missing))

    verify_source_status(paths["e11_completion"], {"complete"})
    verify_source_status(paths["e12_completion"], {"complete"})
    verify_source_status(
        paths["e13_completion"], {"complete_independently_audited"}
    )
    verify_source_status(
        paths["e14_completion"], {"complete_independently_audited"}
    )
    e11_audit = read_json(paths["e11_audit"])
    e12_audit = read_json(paths["e12_audit"])
    if not all(
        check.get("passed") is True
        for check in e11_audit.get("checks", {}).values()
    ):
        raise ValueError("E11 independent deterministic checks did not all pass")
    if not all(
        check.get("passed") is True
        for check in e12_audit.get("checks", {}).values()
    ):
        raise ValueError("E12 independent deterministic checks did not all pass")
    protocol_query_set = read_json(paths["protocol_query_set"])
    if protocol_query_set.get("queries") != []:
        raise ValueError(
            "formal protocol query set unexpectedly populated; use the formal Q1 path"
        )

    output.mkdir(parents=True)
    (output / "tables").mkdir()
    (output / "query_proxy/per_query").mkdir(parents=True)
    (output / "visualizations").mkdir()
    (output / "source_snapshot").mkdir()
    shutil.copy2(
        Path(__file__),
        output / "source_snapshot/run_g1_no_gt_e15_semantic_increment.py",
    )
    preregistration = {
        "schema": "daaam.g1_no_gt_e15_preregistration.v1",
        "created_at": utc_now(),
        "frozen_chain": {
            "E11": "conf_0p3__area_0300__iou_0p5",
            "E12": "buffer_10",
            "E13": "safe_merge_0p35m",
            "E14": "unique_safe_obs8_seed0",
        },
        "variants": ("geometry_only", "frontend", "dam"),
        "top_k": sorted(args.top_k),
        "minimum_similarity": args.minimum_similarity,
        "sentence_model": args.sentence_model,
        "query_proxy": (
            "20 existing retrospective Codex target names; positive-only; "
            "relevant entity ordinals from the frozen target comparison"
        ),
        "formal_query_set_status": protocol_query_set["status"],
        "formal_claims_permitted": False,
        "decision_rule": (
            "A layer adds task capability only if it creates a new persisted "
            "artifact/capability. Proxy target/query metrics cannot select a winner."
        ),
    }
    write_json(output / "PRE_REGISTRATION.json", preregistration)
    frozen_inputs = {
        "schema": "daaam.g1_no_gt_e15_frozen_inputs.v1",
        "captured_at": utc_now(),
        **{key: frozen_reference(path) for key, path in paths.items()},
    }
    write_json(output / "FROZEN_INPUTS.json", frozen_inputs)
    write_json(
        output / "invocation.json",
        {
            "schema": "daaam.g1_no_gt_e15_invocation.v1",
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "started_at": utc_now(),
        },
    )

    e11 = selected_row(
        read_json(paths["e11_summary"]),
        "cell_id",
        "conf_0p3__area_0300__iou_0p5",
    )
    e12 = selected_row(
        read_json(paths["e12_summary"]), "variant_id", "buffer_10"
    )
    e13 = selected_row(
        read_json(paths["e13_summary"]),
        "variant_id",
        "safe_merge_0p35m",
    )
    e14 = selected_row(
        read_json(paths["e14_summary"]), "threshold_observations", 8
    )
    frame_counts = {
        int(e11["frame_count"]),
        int(e12["frame_count"]),
        101,
    }
    if frame_counts != {101}:
        raise ValueError(f"input frame count mismatch: {frame_counts}")
    if int(e13["input_geometry_observations"]) != 2886:
        raise ValueError("unexpected E13 geometry observation count")
    if e13["association_policy"] != "safe" or float(e13["threshold_m"]) != 0.35:
        raise ValueError("E13 is not the frozen safe 0.35 m variant")
    if int(e14["total_e13_entities"]) != int(e13["entity_count"]):
        raise ValueError("E13/E14 entity count mismatch")
    if str(read_json(paths["e14_completion"])["independent_audit"]) != "passed":
        raise ValueError("E14 independent audit is not passed")

    old_census = read_jsonl(paths["old_census"])
    targets = read_jsonl(paths["target_comparison"])
    labels = read_jsonl(paths["e14_labels"])
    old_by_instance = {row["instance_id"]: row for row in old_census}
    target_by_instance = {row["instance_id"]: row for row in targets}
    if (
        len(old_by_instance) != 20
        or len(target_by_instance) != 20
        or set(old_by_instance) != set(target_by_instance)
    ):
        raise ValueError("expected the same 20 frozen salient targets")
    target_rows = []
    for instance_id in sorted(target_by_instance):
        old = old_by_instance[instance_id]
        target = target_by_instance[instance_id]
        target_rows.append(
            {
                "schema": "daaam.g1_no_gt_e15_target_stage.v1",
                "instance_id": instance_id,
                "instance_name": target["instance_name"],
                "E11_status": old["stages"]["E11"],
                "E12_status": old["stages"]["E12"],
                "E13_status": target["new_e13_status"],
                "E14_status": target["new_e14_status"],
                "primary_evidence_entities_json": target[
                    "primary_evidence_entities_json"
                ],
                "evaluation_basis": (
                    "retrospective_codex_approximate_gt_not_human_gt"
                ),
            }
        )
    stage_status = {
        stage: status_counts(row[f"{stage}_status"] for row in target_rows)
        for stage in ("E11", "E12", "E13", "E14")
    }
    write_jsonl(output / "tables/stage_status.jsonl", target_rows)
    write_csv(output / "tables/stage_status.csv", target_rows)
    write_json(output / "tables/stage_status_summary.json", stage_status)

    model_manifest = {
        "schema": "daaam.g1_no_gt_e15_embedding_model.v1",
        "model": args.sentence_model,
        "local_files_only": not args.allow_model_download,
        **model_cache_revision(args.sentence_model),
    }
    model_started = time.perf_counter()
    model = SentenceTransformer(
        args.sentence_model,
        device="cpu",
        local_files_only=not args.allow_model_download,
    )
    model_manifest["load_latency_s"] = time.perf_counter() - model_started
    model_manifest["embedding_dimension"] = (
        model.get_sentence_embedding_dimension()
    )
    queries = [str(row["instance_name"]) for row in targets]
    descriptions = [str(row["final_label"]) for row in labels]
    encode_started = time.perf_counter()
    label_embeddings = np.asarray(
        model.encode(
            descriptions,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    query_embeddings = np.asarray(
        model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    model_manifest["encode_latency_s"] = time.perf_counter() - encode_started
    scores = query_embeddings @ label_embeddings.T
    np.save(output / "query_proxy/label_embeddings.npy", label_embeddings)
    np.save(output / "query_proxy/query_embeddings.npy", query_embeddings)
    np.save(output / "query_proxy/score_matrix.npy", scores)
    query_rows, query_summary = retrieval_rows(
        targets,
        labels,
        scores,
        sorted(args.top_k),
        args.minimum_similarity,
    )
    for row in query_rows:
        write_json(
            output
            / "query_proxy/per_query"
            / f"{row['instance_id']}.json",
            row,
        )
    write_jsonl(output / "query_proxy/query_rankings.jsonl", query_rows)
    write_csv(output / "query_proxy/query_rankings.csv", query_rows)
    write_json(output / "query_proxy/query_summary.json", query_summary)
    model_manifest["label_embeddings_sha256"] = sha256_file(
        output / "query_proxy/label_embeddings.npy"
    )
    model_manifest["query_embeddings_sha256"] = sha256_file(
        output / "query_proxy/query_embeddings.npy"
    )
    model_manifest["score_matrix_sha256"] = sha256_file(
        output / "query_proxy/score_matrix.npy"
    )
    write_json(output / "query_proxy/MODEL_MANIFEST.json", model_manifest)

    e11_total_ms = float(e11["inference_latency_mean_ms"]) * 101
    e12_total_ms = float(e12["tracking_latency_mean_ms"]) * 101
    e13_total_ms = float(e13["observe_latency_ms_mean"]) * int(
        e13["input_geometry_observations"]
    )
    e14_total_ms = float(e14["batch_latency_s"]["mean"]) * int(
        e14["batch_latency_s"]["count"]
    ) * 1000.0
    total_ms = e11_total_ms + e12_total_ms + e13_total_ms + e14_total_ms
    cost = {
        "schema": "daaam.g1_no_gt_e15_cost_summary.v1",
        "scope": (
            "isolated persisted operator timings; arithmetic attribution only, "
            "not end-to-end wall clock"
        ),
        "e11": {
            "total_ms": e11_total_ms,
            "per_frame_ms": e11_total_ms / 101,
            "source_metric": "inference_latency_mean_ms",
        },
        "e12": {
            "total_ms": e12_total_ms,
            "per_frame_ms": e12_total_ms / 101,
            "source_metric": "tracking_latency_mean_ms",
        },
        "e13": {
            "total_ms": e13_total_ms,
            "per_frame_ms": e13_total_ms / 101,
            "source_metric": (
                "observe_latency_ms_mean x input_geometry_observations"
            ),
        },
        "e14": {
            "total_ms": e14_total_ms,
            "per_frame_ms": e14_total_ms / 101,
            "source_metric": "sum of five persisted DAM batch latencies",
            "fraction_of_measured_semantic_cost": e14_total_ms / total_ms,
        },
        "frontend_cumulative": {
            "total_ms": e11_total_ms + e12_total_ms + e13_total_ms,
            "per_frame_ms": (
                e11_total_ms + e12_total_ms + e13_total_ms
            )
            / 101,
        },
        "cumulative": {
            "total_ms": total_ms,
            "per_frame_ms": total_ms / 101,
        },
        "formal_end_to_end_latency": None,
    }
    write_json(output / "tables/cost_summary.json", cost)

    funnel = [
        {
            "variant": "geometry_only",
            "frame_count": 101,
            "mask_count": 0,
            "tracked_observation_count": 0,
            "track_count": 0,
            "geometry_observation_count": 0,
            "entity_count": 0,
            "named_entity_count": 0,
            "query_candidate_count": 0,
            "query_capability": "unavailable_no_semantic_records",
            "incremental_operator_cost_ms": 0.0,
            "formal_task_metric": None,
        },
        {
            "variant": "frontend",
            "frame_count": 101,
            "mask_count": int(e11["kept_instance_count"]),
            "tracked_observation_count": int(e12["tracked_observation_count"]),
            "track_count": int(e12["unique_track_count"]),
            "geometry_observation_count": int(
                e13["input_geometry_observations"]
            ),
            "entity_count": int(e13["entity_count"]),
            "named_entity_count": 0,
            "query_candidate_count": 0,
            "query_capability": "unavailable_no_natural_language_descriptions",
            "incremental_operator_cost_ms": cost["frontend_cumulative"][
                "total_ms"
            ],
            "formal_task_metric": None,
        },
        {
            "variant": "dam",
            "frame_count": 101,
            "mask_count": int(e11["kept_instance_count"]),
            "tracked_observation_count": int(e12["tracked_observation_count"]),
            "track_count": int(e12["unique_track_count"]),
            "geometry_observation_count": int(
                e13["input_geometry_observations"]
            ),
            "entity_count": int(e13["entity_count"]),
            "named_entity_count": int(e14["responded_entity_count"]),
            "query_candidate_count": len(labels),
            "query_capability": "positive_only_posthoc_proxy_available",
            "incremental_operator_cost_ms": e14_total_ms,
            "formal_task_metric": None,
        },
    ]
    write_json(output / "tables/incremental_funnel.json", funnel)
    write_csv(output / "tables/incremental_funnel.csv", funnel)
    summary = {
        "schema": "daaam.g1_no_gt_e15_summary.v1",
        "status": "complete_diagnostic_gt_free_no_winner",
        "frame_count": 101,
        "frozen_chain": preregistration["frozen_chain"],
        "stage_status": stage_status,
        "query_proxy": query_summary,
        "named_entity_coverage": float(
            e14["all_entity_description_coverage"]
        ),
        "hypothesis_result": "partial_pass",
        "hypothesis_reason": (
            "DAM adds the first natural-language retrieval capability, but "
            "strict target success drops from E13 entity quality to E14 "
            "semantic quality."
        ),
        "winner": None,
        "formal_claims_permitted": False,
    }
    write_json(output / "SUMMARY.json", summary)
    make_visualizations(
        output,
        e11,
        e12,
        e13,
        e14,
        stage_status,
        query_summary,
        cost,
    )
    (output / "REPORT.md").write_text(
        build_report(stage_status, query_summary, cost, e11, e12, e13, e14),
        encoding="utf-8",
    )
    seal_output(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "hypothesis_result": summary["hypothesis_result"],
                "named_entity_coverage": summary["named_entity_coverage"],
                "query_proxy": query_summary["recall_at_k_proxy"],
                "completion": "complete_pending_independent_audit",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
