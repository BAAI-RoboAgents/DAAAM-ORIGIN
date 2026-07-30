#!/usr/bin/env python3
"""Replay frozen G1 E13 geometry through safer association/E14 trigger policies."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from run_g1_no_gt_e13_entity_merge import (  # noqa: E402
    read_jsonl,
    run_variant,
    seal_output,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_SOURCE = (
    REPOSITORY_ROOT
    / "experiments/g1_20260724_473_573_v1_1/runs"
    / "diagnostic_gt_free_e13_e12fed_mapmemory_20260729"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "experiments/g1_20260724_473_573_v1_1/runs"
    / "diagnostic_gt_free_e13_e14_safety_ablation_20260730"
)
DEFAULT_POLICIES = ("legacy", "safe", "track_only")
DEFAULT_THRESHOLDS_M = (0.20, 0.35, 0.50)
DEFAULT_E14_MINIMUM_OBSERVATIONS = 8

# These relations are frozen from the Codex approximate-GT audit.  A shared
# MapMemory entity between the left/right track sets means the relation remains
# merged at least once.  They are diagnostic labels, not independent human GT.
KNOWN_RELATIONS = {
    "bad_shelf_vs_cabinet": {
        "expected": "separate",
        "left_tracks": (3, 7, 11, 36, 39, 43),
        "right_tracks": (22,),
    },
    "bad_device_vs_pots": {
        "expected": "separate",
        "left_tracks": (5,),
        "right_tracks": (21, 62, 124),
    },
    "bad_upper_tray_pair": {
        "expected": "separate",
        "left_tracks": (16,),
        "right_tracks": (17,),
    },
    "bad_lower_tray_pair": {
        "expected": "separate",
        "left_tracks": (19,),
        "right_tracks": (20,),
    },
    "bad_table_vs_basket": {
        "expected": "separate",
        "left_tracks": (31, 88),
        "right_tracks": (123,),
    },
    "good_bin_fragments": {
        "expected": "merge",
        "left_tracks": (12,),
        "right_tracks": (113,),
    },
    "good_chair_fragments": {
        "expected": "merge",
        "left_tracks": (45,),
        "right_tracks": (120,),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a frozen E13 association and E14 trigger safety ablation without "
            "recomputing E11/E12/depth geometry."
        )
    )
    parser.add_argument("--source-e13-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--association-policies",
        nargs="+",
        choices=DEFAULT_POLICIES,
        default=list(DEFAULT_POLICIES),
    )
    parser.add_argument(
        "--thresholds-m",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS_M),
    )
    parser.add_argument(
        "--e14-minimum-observations",
        type=int,
        default=DEFAULT_E14_MINIMUM_OBSERVATIONS,
    )
    parser.add_argument(
        "--reseal-existing",
        action="store_true",
        help="Only rebuild the artifact inventory and completion seal.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def frozen_reference(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def entities_by_track(
    events: Iterable[Mapping[str, Any]],
) -> dict[int, set[str]]:
    result: dict[int, set[str]] = defaultdict(set)
    for event in events:
        result[int(event["track_id"])].add(str(event["entity_id"]))
    return result


def relation_audit(
    variant_id: str,
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    mapping = entities_by_track(events)
    rows = []
    for relation_id, relation in KNOWN_RELATIONS.items():
        left_entities = set().union(
            *(mapping.get(track_id, set()) for track_id in relation["left_tracks"])
        )
        right_entities = set().union(
            *(mapping.get(track_id, set()) for track_id in relation["right_tracks"])
        )
        shared = sorted(left_entities & right_entities)
        expected = str(relation["expected"])
        passed = (expected == "separate" and not shared) or (
            expected == "merge" and bool(shared)
        )
        rows.append(
            {
                "schema": "daaam.g1_e13_known_relation_audit.v1",
                "variant_id": variant_id,
                "relation_id": relation_id,
                "expected": expected,
                "left_track_ids_json": json.dumps(relation["left_tracks"]),
                "right_track_ids_json": json.dumps(relation["right_tracks"]),
                "shared_entity_ids_json": json.dumps(shared),
                "relation_is_merged": bool(shared),
                "passed": passed,
                "evidence_basis": "codex_approximate_gt_not_independent_human_gt",
            }
        )
    return rows


def e14_trigger_simulation(
    variant_id: str,
    events: Sequence[Mapping[str, Any]],
    minimum_observations: int,
) -> dict[str, Any]:
    by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        by_frame[int(event["frame_index"])].append(event)

    raw_counts: dict[str, int] = defaultdict(int)
    unique_frame_counts: dict[str, int] = defaultdict(int)
    collision_counts: dict[str, int] = defaultdict(int)
    legacy_prompted: set[str] = set()
    corrected_prompted: set[str] = set()
    collisions_seen: set[str] = set()
    legacy_requests = 0
    corrected_requests = 0

    for frame_index in sorted(by_frame):
        current = sorted(
            by_frame[frame_index],
            key=lambda row: (
                int(row["track_id"]),
                int(row["e11_instance_id"]),
            ),
        )
        current_by_entity: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for event in current:
            entity_id = str(event["entity_id"])
            raw_counts[entity_id] += 1
            current_by_entity[entity_id].append(event)

        for entity_id, entity_events in current_by_entity.items():
            unique_frame_counts[entity_id] += 1
            extra_tracks = len({int(row["track_id"]) for row in entity_events}) - 1
            if extra_tracks > 0:
                collision_counts[entity_id] += extra_tracks
                collisions_seen.add(entity_id)

        legacy_selected = [
            event
            for event in current
            if raw_counts[str(event["entity_id"])] >= minimum_observations
            and str(event["entity_id"]) not in legacy_prompted
        ]
        legacy_requests += len(legacy_selected)
        legacy_prompted.update(
            str(event["entity_id"]) for event in legacy_selected
        )

        corrected_selected = [
            entity_id
            for entity_id in current_by_entity
            if unique_frame_counts[entity_id] >= minimum_observations
            and entity_id not in collisions_seen
            and entity_id not in corrected_prompted
        ]
        corrected_requests += len(corrected_selected)
        corrected_prompted.update(corrected_selected)

    all_entities = set(raw_counts)
    raw_eligible = {
        entity_id
        for entity_id, count in raw_counts.items()
        if count >= minimum_observations
    }
    unique_eligible = {
        entity_id
        for entity_id, count in unique_frame_counts.items()
        if count >= minimum_observations
    }
    sealed_collision_free_eligible = unique_eligible - {
        entity_id for entity_id, count in collision_counts.items() if count > 0
    }
    return {
        "schema": "daaam.g1_e14_trigger_ablation.v1",
        "variant_id": variant_id,
        "minimum_observations": minimum_observations,
        "entity_count": len(all_entities),
        "raw_observation_eligible_entities": len(raw_eligible),
        "unique_frame_eligible_entities": len(unique_eligible),
        "sealed_collision_free_eligible_entities": len(
            sealed_collision_free_eligible
        ),
        "legacy_online_prompted_entities": len(legacy_prompted),
        "legacy_online_mask_requests": legacy_requests,
        "legacy_duplicate_mask_requests": legacy_requests - len(legacy_prompted),
        "corrected_online_prompted_entities": len(corrected_prompted),
        "corrected_online_mask_requests": corrected_requests,
        "corrected_duplicate_mask_requests": (
            corrected_requests - len(corrected_prompted)
        ),
        "entities_with_same_frame_collision": sum(
            count > 0 for count in collision_counts.values()
        ),
        "same_frame_extra_track_observations": sum(collision_counts.values()),
        "counting_contract": {
            "legacy": "one increment per entity-track observation",
            "corrected": "one increment per entity per segmentation frame",
            "corrected_mask_selection": "one deterministic mask per entity",
            "corrected_collision_gate": (
                "reject after the first observed same-frame distinct-track collision"
            ),
            "sealed_collision_free": (
                "offline diagnostic upper bound using full-run collision knowledge"
            ),
        },
    }


def create_summary_figure(
    output: Path,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    labels = [
        f"{row['association_policy']}\n{row['threshold_m']:.2f}m"
        for row in summaries
    ]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(
        2, 1, figsize=(max(12, len(labels) * 1.6), 9), constrained_layout=True
    )
    axes[0].bar(
        x - 0.2,
        [int(row["known_bad_relations_remaining"]) for row in summaries],
        0.4,
        label="known bad relations remaining",
    )
    axes[0].bar(
        x + 0.2,
        [int(row["known_good_relations_preserved"]) for row in summaries],
        0.4,
        label="known good merges preserved",
    )
    axes[0].set_ylabel("relations")
    axes[0].set_xticks(x, labels)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        x - 0.2,
        [
            int(row["same_frame_multi_track_collision_count_proxy"])
            for row in summaries
        ],
        0.4,
        label="same-frame extra tracks",
    )
    axes[1].bar(
        x + 0.2,
        [int(row["track_to_multiple_entity_count_proxy"]) for row in summaries],
        0.4,
        label="tracks split across entities",
    )
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_ylabel("count (symlog)")
    axes[1].set_xticks(x, labels)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    path = output / "visualizations/e13_policy_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(
    output: Path,
    summaries: Sequence[Mapping[str, Any]],
    e14_rows: Sequence[Mapping[str, Any]],
) -> None:
    e14_by_variant = {str(row["variant_id"]): row for row in e14_rows}
    table = "\n".join(
        (
            f"| {row['association_policy']} | {row['threshold_m']:.2f} | "
            f"{row['entity_count']} | {row['known_bad_relations_remaining']}/5 | "
            f"{row['known_good_relations_preserved']}/2 | "
            f"{row['same_frame_multi_track_collision_count_proxy']} | "
            f"{row['track_to_multiple_entity_count_proxy']} | "
            f"{e14_by_variant[str(row['variant_id'])]['legacy_online_mask_requests']} | "
            f"{e14_by_variant[str(row['variant_id'])]['corrected_online_mask_requests']} |"
        )
        for row in summaries
    )
    report = f"""# E13–E14 安全策略反事实实验

## 结论边界

本实验复用冻结 E13 的 2,886 条几何观察，逐条调用当前分支的生产
`MapMemory.observe_entity()`。它比较 legacy、safe 和 track_only 三种关联策略及
0.20/0.35/0.50 m 门限，并在同一事件账本上重放 E14 的 observation≥
{e14_rows[0]['minimum_observations']} 触发逻辑。

五条“应分开”与两条“应合并”关系来自 Codex 近似真值审计，不是双人独立人工 GT；
所以结果可验证故障机制和工程回归，不可宣称正式准确率。

## 结果

| 策略 | 距离 m | entity | 已知错并仍存在 | 已知正确合并保留 | 同帧额外 track | 跨 entity track | 旧 E14 mask 请求 | 新 E14 mask 请求 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

`safe` 对新 track 加入同帧互斥、尺寸比和历史空间扩张门，并在先前合并的 track
后来同帧共现时立即拆开。`track_only` 是 pre-DAM 保守上界：不同 BotSort track
不在未知语义阶段合并。E14 新规则按“entity×segmentation frame”唯一计数，一个
entity 每帧只选一个 mask，并拒绝已经出现同帧多 track 冲突的 entity。

## 如何解释

- “已知错并仍存在”越低越好；“已知正确合并保留”越高越好，两者必须同时看。
- 同帧冲突为强 over-merge 证据；跨 entity track 是 fragmentation/ID reuse
  复核信号，不可直接当作错误率。
- track_only 的零跨-track错并以不自动恢复 track fragment 为代价；它是安全基线，
  不是最终的跨-track实体合并器。
- safe 只能消除有几何冲突证据的错并。桌面–篮子这种不同时间、近中心、近尺寸关系
  仍需要语义/外观证据或 delayed confirmation，单靠本轮几何门无法证明可解。

## 证据

- `FROZEN_INPUTS.json`：源账本路径、字节数和 SHA-256。
- `variants/*/map_memory.sqlite3`：每个反事实的原生数据库。
- `variants/*/merge_events.*`：2,886 条逐观察决策。
- `variants/*/frames/*/entity_overlay.jpg`：101 帧 RGB 实体 overlay。
- `tables/known_relation_audit.*`：七条冻结关系逐项结果。
- `tables/e14_trigger_ablation.*`：原计数与唯一帧/单 mask 计数对照。
- `artifact_inventory.*` 与 `COMPLETION.json`：完整文件哈希封存。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.reseal_existing:
        if not args.output.exists():
            raise FileNotFoundError(args.output)
        audit_path = args.output / "INDEPENDENT_AUDIT.json"
        audited = audit_path.is_file() and json.loads(
            audit_path.read_text(encoding="utf-8")
        ).get("passed")
        seal_output(
            args.output,
            status=(
                "complete_independently_audited"
                if audited
                else "complete_pending_independent_audit"
            ),
        )
        return 0
    if args.output.exists():
        raise FileExistsError(f"refuse to overwrite existing output: {args.output}")
    if args.e14_minimum_observations <= 0:
        raise ValueError("E14 minimum observations must be positive")
    if any(threshold <= 0.0 for threshold in args.thresholds_m):
        raise ValueError("all E13 thresholds must be positive")
    args.output.mkdir(parents=True)

    frame_path = args.source_e13_run / "input_manifests/e12_frames.jsonl"
    geometry_path = args.source_e13_run / "tables/geometry_observations.jsonl"
    completion_path = args.source_e13_run / "COMPLETION.json"
    frames = read_jsonl(frame_path)
    observations = read_jsonl(geometry_path)
    write_jsonl(args.output / "input_manifests/e12_frames.jsonl", frames)
    write_csv(args.output / "input_manifests/e12_frames.csv", frames)
    write_json(
        args.output / "FROZEN_INPUTS.json",
        {
            "schema": "daaam.g1_e13_e14_safety_ablation_inputs.v1",
            "captured_at": utc_now(),
            "source_e13_run": str(args.source_e13_run.resolve()),
            "source_completion": frozen_reference(completion_path),
            "frames": frozen_reference(frame_path),
            "geometry_observations": frozen_reference(geometry_path),
            "frame_count": len(frames),
            "observation_count": len(observations),
            "formal_gt_status": "codex_approximate_gt_not_independent_human_gt",
        },
    )
    write_json(
        args.output / "PRE_REGISTRATION.json",
        {
            "schema": "daaam.g1_e13_e14_safety_ablation_preregistration.v1",
            "registered_at": utc_now(),
            "policies": args.association_policies,
            "thresholds_m": args.thresholds_m,
            "e14_minimum_observations": args.e14_minimum_observations,
            "primary_safety_metric": "known_bad_relations_remaining",
            "counter_metric": "known_good_relations_preserved",
            "secondary_metrics": [
                "same_frame_multi_track_collision_count_proxy",
                "track_to_multiple_entity_count_proxy",
                "corrected_online_mask_requests",
                "corrected_duplicate_mask_requests",
            ],
            "selection_rule": (
                "No production winner without independent held-out GT; identify "
                "Pareto behavior and unresolved failure mechanisms only."
            ),
        },
    )

    summaries = []
    relation_rows = []
    e14_rows = []
    for policy in args.association_policies:
        for threshold_m in args.thresholds_m:
            summary, events = run_variant(
                args.output,
                float(threshold_m),
                frames,
                observations,
                association_policy=policy,
            )
            relations = relation_audit(str(summary["variant_id"]), events)
            relation_rows.extend(relations)
            summary["known_bad_relations_remaining"] = sum(
                row["expected"] == "separate" and not row["passed"]
                for row in relations
            )
            summary["known_good_relations_preserved"] = sum(
                row["expected"] == "merge" and row["passed"]
                for row in relations
            )
            e14 = e14_trigger_simulation(
                str(summary["variant_id"]),
                events,
                args.e14_minimum_observations,
            )
            e14_rows.append(e14)
            summaries.append(summary)

    write_json(args.output / "tables/variant_summary.json", summaries)
    write_csv(args.output / "tables/variant_summary.csv", summaries)
    write_jsonl(args.output / "tables/known_relation_audit.jsonl", relation_rows)
    write_csv(args.output / "tables/known_relation_audit.csv", relation_rows)
    write_json(args.output / "tables/e14_trigger_ablation.json", e14_rows)
    write_csv(args.output / "tables/e14_trigger_ablation.csv", e14_rows)
    create_summary_figure(args.output, summaries)
    write_report(args.output, summaries, e14_rows)
    write_json(
        args.output / "RUN_SUMMARY.json",
        {
            "schema": "daaam.g1_e13_e14_safety_ablation_summary.v1",
            "status": "complete_pending_independent_gt",
            "completed_at": utc_now(),
            "variant_count": len(summaries),
            "frame_count": len(frames),
            "observation_count": len(observations),
            "variant_summaries": summaries,
            "e14_trigger_summaries": e14_rows,
            "formal_accuracy_claim_permitted": False,
        },
    )
    seal_output(args.output)
    print(f"complete: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
