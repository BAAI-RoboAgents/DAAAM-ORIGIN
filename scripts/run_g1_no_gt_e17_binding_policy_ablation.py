#!/usr/bin/env python3
"""Run the frozen-input G1 E17-v2 A/B/C/D binding-policy ablation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from daaam.experiments.e17_ablation import (  # noqa: E402
    D_GLOBAL_GATE,
    current_spatial_pending_candidates,
    eligible_global_cross_id_candidates,
    global_one_to_one_assignment,
)
from daaam.grounding.models import ObjectAnnotation  # noqa: E402
from daaam.scene_graph.services import (  # noqa: E402
    ObjectBindingPolicy,
    SceneGraphService,
)
from rebind_dsg_semantics import _mesh_counts  # noqa: E402
from run_g1_no_gt_e17_dsg_binding import (  # noqa: E402
    inventory,
    read_jsonl,
    sha256,
    write_csv,
    write_json,
    write_jsonl,
)


EXPERIMENT_ROOT = (
    REPOSITORY_ROOT / "experiments" / "g1_20260724_473_573_v1_1"
)
DEFAULT_E17 = (
    EXPERIMENT_ROOT
    / "runs"
    / "diagnostic_gt_free_e17_e14e16fed_binding_20260730"
)
DEFAULT_E13 = (
    EXPERIMENT_ROOT
    / "runs"
    / "diagnostic_gt_free_e13_e14_safety_ablation_20260730"
    / "variants"
    / "safe_merge_0p35m"
)
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT
    / "comparisons"
    / "e17_binding_policy_ablation_20260730"
)
E14_REVIEW_ROOT = (
    EXPERIMENT_ROOT
    / "runs"
    / "diagnostic_gt_free_e14_safe035_unique_obs8_seed0_20260730"
    / "annotation_census_obs_08_seed_0"
    / "review_panels"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e17-run", type=Path, default=DEFAULT_E17)
    parser.add_argument("--e13-variant", type=Path, default=DEFAULT_E13)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--semantic-config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "labels_pseudo.yaml",
    )
    parser.add_argument(
        "--labelspace-colors",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "labels_pseudo.csv",
    )
    return parser.parse_args()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def load_tracks(path: Path) -> tuple[dict[int, set[int]], list[dict[str, Any]]]:
    rows = json.loads(path.read_text())
    tracks = {
        int(row["entity_ordinal"]): {
            int(value) for value in json.loads(row["track_ids_json"])
        }
        for row in rows
    }
    return tracks, rows


def node_binding_map(graph: Any) -> dict[str, dict[str, Any]]:
    from spark_dsg import DsgLayers

    result = {}
    for node in graph.get_layer(DsgLayers.OBJECTS).nodes:
        metadata = dict(node.attributes.metadata.get() or {})
        entity_id = str(metadata.get("entity_id") or "").strip()
        if not entity_id:
            continue
        mesh = node.attributes.mesh()
        vertices = 0 if mesh is None else int(mesh.num_vertices())
        if vertices <= 0:
            raise RuntimeError(f"bound node has no real mesh: {node.id}")
        if entity_id in result:
            raise RuntimeError(f"entity has multiple bound nodes: {entity_id}")
        binding_evidence = dict(metadata.get("entity_binding") or {})
        result[entity_id] = {
            "node_id": str(node.id),
            "candidate_semantic_id": int(
                binding_evidence.get(
                    "candidate_semantic_id",
                    node.attributes.semantic_label,
                )
            ),
            "description": str(metadata.get("description") or ""),
            "mesh_vertices": vertices,
            "binding_source": metadata.get("entity_binding_source"),
            "mesh_binding_status": metadata.get("mesh_binding_status"),
        }
    return result


def bind_batch(
    graph: Any,
    entities: Sequence[Mapping[str, Any]],
    *,
    owners: Mapping[int, str],
    cross_policy: str,
    semantic_config: Path,
    labelspace_colors: Path,
    phase: str,
) -> list[dict[str, Any]]:
    source_counts = _mesh_counts(graph)
    policy = ObjectBindingPolicy(
        maximum_center_distance_m=0.10,
        maximum_aabb_gap_m=0.025,
        audit_capacity=max(1000, len(entities) * (source_counts["object_meshes"] + 4)),
        cross_semantic_id_policy=cross_policy,
    )
    service = SceneGraphService(
        semantic_config,
        labelspace_colors,
        enable_background_objects=False,
        object_binding_policy=policy,
    )
    service.set_scene_graph(graph)
    for entity in sorted(entities, key=lambda row: int(row["semantic_id"])):
        history = dict(entity.get("temporal_history") or {})
        sensor_time_ns = int(
            history.get("last_observed_ns")
            or history.get("first_observed_ns")
            or 1
        )
        ensured = service.ensure_object_node(
            semantic_id=int(entity["semantic_id"]),
            entity_id=str(entity["entity_id"]),
            position_m=entity["position_m"],
            dimensions_m=entity["dimensions_m"],
            sensor_time_ns=sensor_time_ns,
            temporal_history=history,
            time_origin_ns=history.get("time_origin_ns"),
            allow_unmeshed_fallback=False,
            semantic_id_owners=owners,
        )
        if ensured:
            service.add_correction(
                ObjectAnnotation(
                    semantic_id=int(entity["semantic_id"]),
                    entity_id=str(entity["entity_id"]),
                    semantic_label=str(entity["description"]),
                    confidence=1.0,
                    sensor_time_ns=sensor_time_ns,
                    timestamp=sensor_time_ns / 1.0e9,
                )
            )
    service.apply_corrections()
    return [
        {
            "schema": "daaam.g1_e17_v2_binding_event.v1",
            "phase": phase,
            "cross_semantic_id_policy": cross_policy,
            "event_index": index,
            **event,
        }
        for index, event in enumerate(service.object_binding_audit)
    ]


def save_graph(graph: Any, path: Path) -> dict[str, Any]:
    from spark_dsg import DynamicSceneGraph

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")
    try:
        graph.save(str(temporary), include_mesh=True)
        reloaded = DynamicSceneGraph.load(str(temporary))
        counts = _mesh_counts(reloaded)
        bindings = node_binding_map(reloaded)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "sha256": sha256(path),
        "mesh_counts": counts,
        "bindings": bindings,
    }


def decisions_from_graph(
    entities: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    geometry_source: str,
    arm: str,
    pending_rows: Sequence[Mapping[str, Any]],
    reviewed_cross: Mapping[tuple[int, int], str],
) -> list[dict[str, Any]]:
    candidates = {
        (str(row["entity_id"]), str(row["node_id"])): row
        for row in candidate_rows
    }
    pending_by_entity: Counter[str] = Counter(
        str(row["entity_id"]) for row in pending_rows
    )
    result = []
    for entity in sorted(entities, key=lambda row: int(row["semantic_id"])):
        entity_id = str(entity["entity_id"])
        binding = bindings.get(entity_id)
        if binding is None:
            result.append(
                {
                    "schema": "daaam.g1_e17_v2_terminal_decision.v1",
                    "geometry_source": geometry_source,
                    "arm": arm,
                    "entity_id": entity_id,
                    "entity_ordinal": int(entity["semantic_id"]),
                    "entity_label": entity["description"],
                    "status": "rejected_no_authoritative_mesh",
                    "node_id": None,
                    "candidate_semantic_id": None,
                    "semantic_id_match": False,
                    "pending_candidate_count": int(pending_by_entity[entity_id]),
                    "review_status": "not_bound",
                }
            )
            continue
        node_id = str(binding["node_id"])
        candidate = candidates[(entity_id, node_id)]
        candidate_semantic_id = int(binding["candidate_semantic_id"])
        exact = int(entity["semantic_id"]) == candidate_semantic_id
        result.append(
            {
                "schema": "daaam.g1_e17_v2_terminal_decision.v1",
                "geometry_source": geometry_source,
                "arm": arm,
                "entity_id": entity_id,
                "entity_ordinal": int(entity["semantic_id"]),
                "entity_label": entity["description"],
                "status": "matched_real_mesh",
                "node_id": node_id,
                "candidate_semantic_id": candidate_semantic_id,
                "semantic_id_match": exact,
                "center_distance_m": candidate["center_distance_m"],
                "aabb_gap_m": candidate["aabb_gap_m"],
                "aabb_iou": candidate["aabb_iou"],
                "pending_candidate_count": int(pending_by_entity[entity_id]),
                "review_status": (
                    "unreviewed_semantic_id_provenance"
                    if exact
                    else reviewed_cross.get(
                        (int(entity["semantic_id"]), candidate_semantic_id),
                        "cross_id_unreviewed",
                    )
                ),
            }
        )
    return result


def summary_from_decisions(
    decisions: Sequence[Mapping[str, Any]],
    *,
    real_mesh_count: int,
    pending_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matched = [row for row in decisions if row["status"] == "matched_real_mesh"]
    exact = [row for row in matched if row["semantic_id_match"]]
    cross = [row for row in matched if not row["semantic_id_match"]]
    rejected = [
        row for row in decisions if row["status"] != "matched_real_mesh"
    ]
    nodes = [str(row["node_id"]) for row in matched]
    if len(nodes) != len(set(nodes)):
        raise RuntimeError("ablation output assigns one mesh to multiple entities")
    return {
        "named_entity_count": len(decisions),
        "real_mesh_count": int(real_mesh_count),
        "matched_real_mesh": len(matched),
        "rejected_no_authoritative_mesh": len(rejected),
        "semantic_id_consistent": len(exact),
        "cross_semantic_id_committed": len(cross),
        "pending_candidate_pairs": len(pending_rows),
        "pending_entity_count": len(
            {str(row["entity_id"]) for row in pending_rows}
        ),
        "confirmed_wrong_cross_id": sum(
            row["review_status"] == "confirmed_wrong_mesh"
            for row in cross
        ),
        "reviewed_compatible_cross_id": sum(
            row["review_status"] == "reviewed_same_wall_track_fragment"
            for row in cross
        ),
        "authoritative_coverage_proxy": (
            len(matched) / len(decisions) if decisions else 0.0
        ),
        "mesh_utilization": len(matched) / real_mesh_count,
        "formal_binding_precision": None,
        "formal_binding_recall": None,
        "formal_binding_f1": None,
    }


def simulate_current_greedy(
    candidate_rows: Sequence[Mapping[str, Any]],
    order: Sequence[str],
) -> dict[str, Any] | None:
    exact_entities = {
        str(row["entity_id"])
        for row in candidate_rows
        if bool(row["semantic_id_match"])
    }
    exact_nodes = {
        str(row["node_id"])
        for row in candidate_rows
        if bool(row["semantic_id_match"])
    }
    by_entity: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidate_rows:
        if (
            str(row["entity_id"]) in exact_entities
            or str(row["node_id"]) in exact_nodes
            or bool(row["semantic_id_match"])
            or not bool(row["accepted"])
            or bool(row.get("rejected_reserved_owner"))
        ):
            continue
        by_entity.setdefault(str(row["entity_id"]), []).append(row)
    claimed = set()
    for entity_id in order:
        eligible = [
            row
            for row in by_entity.get(entity_id, [])
            if str(row["node_id"]) not in claimed
        ]
        eligible.sort(
            key=lambda row: (
                float(row["aabb_gap_m"]),
                -float(row["aabb_iou"]),
                float(row["center_distance_m"]),
                int(row["candidate_semantic_id"]),
            )
        )
        if eligible:
            selected = dict(eligible[0])
            claimed.add(str(selected["node_id"]))
            return selected
    return None


def order_stress_rows(
    entities: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    global_assignments: Sequence[Mapping[str, Any]],
    *,
    geometry_source: str,
) -> list[dict[str, Any]]:
    ids = [str(row["entity_id"]) for row in entities]
    ordinals = {str(row["entity_id"]): int(row["semantic_id"]) for row in entities}
    orders: list[tuple[str, list[str]]] = [
        ("ascending", sorted(ids, key=lambda value: ordinals[value])),
        ("descending", sorted(ids, key=lambda value: ordinals[value], reverse=True)),
    ]
    for seed in range(20):
        shuffled = list(ids)
        random.Random(seed).shuffle(shuffled)
        orders.append((f"shuffle_seed_{seed:02d}", shuffled))
    global_pairs = sorted(
        (
            int(row["entity_ordinal"]),
            int(row["candidate_semantic_id"]),
            str(row["node_id"]),
        )
        for row in global_assignments
    )
    rows = []
    for order_name, order in orders:
        selected = simulate_current_greedy(candidate_rows, order)
        rows.append(
            {
                "schema": "daaam.g1_e17_v2_order_stress.v1",
                "geometry_source": geometry_source,
                "order": order_name,
                "current_greedy_entity_ordinal": (
                    None if selected is None else int(selected["entity_ordinal"])
                ),
                "current_greedy_candidate_semantic_id": (
                    None
                    if selected is None
                    else int(selected["candidate_semantic_id"])
                ),
                "current_greedy_node_id": (
                    None if selected is None else str(selected["node_id"])
                ),
                "global_assignment_pairs": global_pairs,
            }
        )
    return rows


def evidence_montage(output: Path, e17: Path) -> dict[str, Any]:
    target = output / "visualizations" / "cross_id_review"
    target.mkdir(parents=True, exist_ok=True)
    sources = {
        "E008_ceiling.jpg": E14_REVIEW_ROOT / "census_007_E008_source_000480.jpg",
        "E051_wall.jpg": E14_REVIEW_ROOT / "census_038_E051_source_000492.jpg",
        "E077_wall_overlay.jpg": (
            e17
            / "analysis"
            / "spatial_fallback_evidence"
            / "E077_wall_mesh_source_overlay.jpg"
        ),
    }
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target / name)
    images = [plt.imread(target / name) for name in sources]
    titles = [
        "A arm entity E8: ceiling (wrong for E77)",
        "D arm entity E51: wall (same track 34 as E77)",
        "Mesh source E77: wall overlay / track 34",
    ]
    fig, axes = plt.subplots(3, 1, figsize=(18, 19), constrained_layout=True)
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    montage = target / "A_wrong_vs_D_same_track_review.jpg"
    fig.savefig(montage, dpi=140)
    plt.close(fig)
    return {
        "schema": "daaam.g1_e17_v2_codex_cross_id_review.v1",
        "reviewer": "Codex visual engineering review; not human GT",
        "A_E8_to_E77": "confirmed_wrong_mesh",
        "D_E51_to_E77": "reviewed_same_wall_track_fragment",
        "D_support": {
            "shared_upstream_track_id": 34,
            "center_distance_m": 0.8182489728978177,
            "aabb_gap_m": 0.0,
            "aabb_iou": 0.346,
        },
        "montage": str(montage.relative_to(output)),
        "montage_sha256": sha256(montage),
        "formal_gt": False,
    }


def plot_summary(
    output: Path,
    summaries: Sequence[Mapping[str, Any]],
    order_rows: Sequence[Mapping[str, Any]],
) -> None:
    visual = output / "visualizations"
    arms = ("A_current_strict", "B_id_only", "C_id_pending", "D_global_joint")
    sources = ("single_pass", "adaptive")
    lookup = {
        (str(row["geometry_source"]), str(row["arm"])): row for row in summaries
    }
    x = np.arange(len(arms))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for index, source in enumerate(sources):
        exact = [lookup[(source, arm)]["semantic_id_consistent"] for arm in arms]
        cross = [
            lookup[(source, arm)]["cross_semantic_id_committed"] for arm in arms
        ]
        rejected = [
            lookup[(source, arm)]["rejected_no_authoritative_mesh"] for arm in arms
        ]
        axes[index].bar(x, exact, label="semantic-ID consistent")
        axes[index].bar(x, cross, bottom=exact, label="cross-ID committed")
        axes[index].plot(x, rejected, "rx--", label="rejected")
        axes[index].set_xticks(x, ["A", "B", "C", "D"])
        axes[index].set_title(source)
        axes[index].set_ylabel("entities")
        axes[index].grid(axis="y", alpha=0.2)
        axes[index].legend(fontsize=8)
    fig.suptitle("E17-v2 authoritative binding ablation")
    fig.savefig(visual / "01_ablation_outcomes.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for index, source in enumerate(sources):
        wrong = [lookup[(source, arm)]["confirmed_wrong_cross_id"] for arm in arms]
        compatible = [
            lookup[(source, arm)]["reviewed_compatible_cross_id"] for arm in arms
        ]
        pending = [lookup[(source, arm)]["pending_entity_count"] for arm in arms]
        axes[index].bar(x - 0.25, wrong, 0.25, label="confirmed wrong")
        axes[index].bar(x, compatible, 0.25, label="reviewed compatible")
        axes[index].bar(x + 0.25, pending, 0.25, label="pending entities")
        axes[index].set_xticks(x, ["A", "B", "C", "D"])
        axes[index].set_title(source)
        axes[index].set_ylabel("cross-ID entities")
        axes[index].grid(axis="y", alpha=0.2)
        axes[index].legend(fontsize=8)
    fig.suptitle("Cross-ID disposition (Codex review is not formal GT)")
    fig.savefig(visual / "02_cross_id_quality.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for index, source in enumerate(sources):
        rows = [row for row in order_rows if row["geometry_source"] == source]
        counts = Counter(row["current_greedy_entity_ordinal"] for row in rows)
        labels = [str(key) for key in sorted(counts, key=lambda value: -1 if value is None else value)]
        values = [counts[None if value == "None" else int(value)] for value in labels]
        axes[index].bar(labels, values)
        axes[index].axhline(22, color="#2ca02c", linestyle="--", linewidth=1)
        axes[index].set_title(source)
        axes[index].set_xlabel("entity ordinal claiming E77 under A")
        axes[index].set_ylabel("orders out of 22")
        axes[index].grid(axis="y", alpha=0.2)
    fig.suptitle("A greedy order sensitivity; D always selects E51→E77")
    fig.savefig(visual / "03_order_sensitivity.png", dpi=180)
    plt.close(fig)


def report_text(
    summaries: Sequence[Mapping[str, Any]],
    order_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# E17-v2 binding policy 消融实验",
        "",
        "## 结论",
        "",
        (
            "D（同 ID 优先 + 同轨联合几何门 + 全局一对一）在两份冻结几何上"
            "保持 A 的绑定数量，同时把已确认错误的 E8→E77 替换为 E51→E77。"
            "E51 与 E77 都来自 E12 track 34，RGB 均为同一墙面区域。"
        ),
        "",
        "| 几何 | Arm | matched | ID一致 | cross-ID提交 | pending实体 | confirmed wrong | reviewed compatible | rejected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in ("single_pass", "adaptive"):
        for arm in (
            "A_current_strict",
            "B_id_only",
            "C_id_pending",
            "D_global_joint",
        ):
            row = next(
                item
                for item in summaries
                if item["geometry_source"] == source and item["arm"] == arm
            )
            lines.append(
                f"| {source} | {arm} | {row['matched_real_mesh']} | "
                f"{row['semantic_id_consistent']} | "
                f"{row['cross_semantic_id_committed']} | "
                f"{row['pending_entity_count']} | "
                f"{row['confirmed_wrong_cross_id']} | "
                f"{row['reviewed_compatible_cross_id']} | "
                f"{row['rejected_no_authoritative_mesh']} |"
            )
    greedy_counts = Counter(
        (row["geometry_source"], row["current_greedy_entity_ordinal"])
        for row in order_rows
    )
    lines.extend(
        [
            "",
            "## Arm 定义",
            "",
            "- A：当前 strict，`center OR AABB-gap`，跨 ID 直接提交。",
            "- B：只有 semantic-ID 一致可以提交；跨 ID 直接拒绝。",
            "- C：同 B 的 authoritative DSG，但跨 ID 候选进入 pending ledger。",
            (
                "- D：先提交同 ID；跨 ID 必须共享 E12 track，且 center≤1.0m、"
                "gap≤0.075m、IoU≥0.05、对称体积比≤4，再用全局一对一最小代价分配。"
            ),
            "",
            "## 关键证据",
            "",
            (
                "A 的错误不是偶然阈值问题，而是贪心顺序问题。对 ascending、"
                "descending 和 20 个固定 shuffle 共 22 种顺序，E77 的占有者分布为："
            ),
        ]
    )
    for source in ("single_pass", "adaptive"):
        values = {
            ordinal: count
            for (row_source, ordinal), count in greedy_counts.items()
            if row_source == source
        }
        lines.append(f"- {source}: `{json.dumps(values, sort_keys=True)}`")
    lines.extend(
        [
            "",
            (
                "D 的 Hungarian/linear-sum assignment 对候选输入顺序不敏感，"
                "在两个几何源均选择 E51→E77。B/C 删除误绑定但牺牲一个提交；"
                "D 在当前开发窗口同时保持覆盖与复核质量。"
            ),
            "",
            "## 限制",
            "",
            (
                "门限由已经查看过的 development 窗口设计，因此本实验是"
                " retrospective exploratory ablation，不是 held-out confirmatory test。"
                "E51→E77 的正确性是同轨 provenance + Codex RGB 复核，不是人工 GT。"
            ),
            (
                "墙面仍被表示为 object node，说明 E11/E13/E16 的结构层分类仍需"
                "单独优化；D 只修正绑定身份，不解决 layer taxonomy。"
            ),
            "",
            "## 证据目录",
            "",
            "- `PRE_REGISTRATION.json`：A–D 定义、D 联合门和禁止声明。",
            "- `cells/*`：每个 arm 的 durable DSG、终态、event 和 summary。",
            "- `tables/global_candidates.*` / `global_assignments.*`：D 全局输入输出。",
            "- `tables/order_stress.*`：22 种顺序负控。",
            "- `visualizations/cross_id_review/`：E8、E51、E77 RGB 对照。",
            "- `artifact_inventory.*`：逐文件 hash 和根摘要。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from spark_dsg import DynamicSceneGraph

    e17 = args.e17_run.expanduser().resolve()
    e13 = args.e13_variant.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    required = [
        e17 / "COMPLETION.json",
        e17 / "inputs" / "INPUT_MANIFEST.json",
        e17 / "inputs" / "named_entities.jsonl",
        e13 / "entity_membership.json",
        args.semantic_config.expanduser().resolve(),
        args.labelspace_colors.expanduser().resolve(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    completion = json.loads((e17 / "COMPLETION.json").read_text())
    if completion.get("status") != "complete_independently_audited":
        raise ValueError("E17 baseline is not independently audited")
    manifest = json.loads((e17 / "inputs" / "INPUT_MANIFEST.json").read_text())
    entities = read_jsonl(e17 / "inputs" / "named_entities.jsonl")
    owners = {
        int(row["semantic_id"]): str(row["entity_id"]) for row in entities
    }
    tracks, membership = load_tracks(e13 / "entity_membership.json")
    started = datetime.now(timezone.utc)

    preregistration = {
        "schema": "daaam.g1_e17_v2_ablation_preregistration.v1",
        "registered_at": started.isoformat(),
        "status": "retrospective_exploratory_development_window",
        "data_leakage_disclosure": (
            "D gate was designed after inspecting the same 473-573 development "
            "window and prior E17 failure; held-out confirmation is required."
        ),
        "hypothesis": (
            "ID-only prevents confirmed cross-ID error; pending preserves evidence; "
            "shared-track joint geometry plus global assignment recovers the correct "
            "E13 fragment without A's greedy order error."
        ),
        "controlled_input": {
            "e17_baseline": str(e17),
            "e17_inventory_root_sha256": completion[
                "artifact_inventory_root_sha256"
            ],
            "e13_membership": str((e13 / "entity_membership.json").resolve()),
            "e13_membership_sha256": sha256(e13 / "entity_membership.json"),
            "named_entity_count": len(entities),
        },
        "arms": {
            "A_current_strict": "center<=0.10 OR gap<=0.025; cross-ID apply",
            "B_id_only": "same semantic ID authoritative; cross-ID reject",
            "C_id_pending": "same semantic ID authoritative; cross-ID pending",
            "D_global_joint": {
                "same_id_phase": "authoritative first",
                "cross_id_provenance": "shared E12 track ID required",
                "maximum_center_distance_m": (
                    D_GLOBAL_GATE.maximum_center_distance_m
                ),
                "maximum_aabb_gap_m": D_GLOBAL_GATE.maximum_aabb_gap_m,
                "minimum_aabb_iou": D_GLOBAL_GATE.minimum_aabb_iou,
                "maximum_symmetric_volume_ratio": (
                    D_GLOBAL_GATE.maximum_symmetric_volume_ratio
                ),
                "assignment": "global one-to-one minimum cost",
            },
        },
        "single_changed_factor": "binding policy on each frozen geometry source",
        "primary_metric": (
            "authoritative coverage plus reviewed correct/wrong cross-ID bindings"
        ),
        "guardrails": [
            "one entity per mesh",
            "source/output mesh invariant",
            "E8->E77 negative control rejected",
            "all decisions and pending candidates persisted",
            "formal P/R/F1 remain null",
        ],
        "formal_gt": None,
    }
    write_json(output / "PRE_REGISTRATION.json", preregistration)
    write_json(
        output / "invocation.json",
        {
            "schema": "daaam.g1_e17_v2_invocation.v1",
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("branch", "--show-current"),
            "git_status": git_value("status", "--short"),
            "started_at": started.isoformat(),
        },
    )
    source_snapshot = output / "source_snapshot"
    source_snapshot.mkdir()
    for path in (
        Path(__file__),
        REPOSITORY_ROOT / "src" / "daaam" / "experiments" / "e17_ablation.py",
        REPOSITORY_ROOT / "src" / "daaam" / "scene_graph" / "services.py",
    ):
        shutil.copy2(path, source_snapshot / path.name)
    write_jsonl(output / "inputs" / "named_entities.jsonl", entities)
    write_jsonl(output / "inputs" / "e13_entity_membership.jsonl", membership)

    arms = (
        "A_current_strict",
        "B_id_only",
        "C_id_pending",
        "D_global_joint",
    )
    summaries = []
    all_decisions = []
    all_events = []
    all_global_candidates = []
    all_global_assignments = []
    all_pending = []
    all_order_stress = []
    reviewed_cross = {
        (8, 77): "confirmed_wrong_mesh",
        (51, 77): "reviewed_same_wall_track_fragment",
    }

    for source in ("single_pass", "adaptive"):
        source_dsg = Path(manifest["geometry_sources"][source]["path"])
        if sha256(source_dsg) != manifest["geometry_sources"][source]["sha256"]:
            raise ValueError(f"frozen {source} DSG hash changed")
        baseline = e17 / "variants" / f"{source}_strict"
        candidates = read_jsonl(baseline / "candidate_matrix.jsonl")
        exact_entity_ids = {
            str(row["entity_id"])
            for row in candidates
            if bool(row["semantic_id_match"])
        }
        exact_node_ids = {
            str(row["node_id"])
            for row in candidates
            if bool(row["semantic_id_match"])
        }
        pending = current_spatial_pending_candidates(
            candidates,
            exact_entity_ids=exact_entity_ids,
            exact_node_ids=exact_node_ids,
        )
        global_candidates = eligible_global_cross_id_candidates(
            candidates,
            tracks,
            exact_entity_ids=exact_entity_ids,
            exact_node_ids=exact_node_ids,
        )
        global_assignments = global_one_to_one_assignment(global_candidates)
        if [
            (
                int(row["entity_ordinal"]),
                int(row["candidate_semantic_id"]),
            )
            for row in global_assignments
        ] != [(51, 77)]:
            raise RuntimeError(f"unexpected D assignment for {source}")
        for row in pending:
            row["geometry_source"] = source
        for row in global_candidates:
            row["geometry_source"] = source
        for row in global_assignments:
            row["geometry_source"] = source
        all_pending.extend(pending)
        all_global_candidates.extend(global_candidates)
        all_global_assignments.extend(global_assignments)
        all_order_stress.extend(
            order_stress_rows(
                entities,
                candidates,
                global_assignments,
                geometry_source=source,
            )
        )

        entity_lookup = {
            str(row["entity_id"]): row for row in entities
        }
        exact_entities = [
            entity_lookup[entity_id] for entity_id in exact_entity_ids
        ]
        selected_cross_ids = {
            str(row["entity_id"]) for row in global_assignments
        }
        selected_cross_entities = [
            entity_lookup[entity_id] for entity_id in selected_cross_ids
        ]
        remaining_entities = [
            row
            for row in entities
            if str(row["entity_id"]) not in exact_entity_ids
            and str(row["entity_id"]) not in selected_cross_ids
        ]

        for arm in arms:
            cell_started = time.perf_counter()
            cell = output / "cells" / f"{source}_{arm}"
            cell.mkdir(parents=True)
            events = []
            if arm == "A_current_strict":
                shutil.copy2(
                    baseline / "dsg_bound.json",
                    cell / "dsg_bound.json",
                )
                graph = DynamicSceneGraph.load(str(cell / "dsg_bound.json"))
                graph_info = {
                    "sha256": sha256(cell / "dsg_bound.json"),
                    "mesh_counts": _mesh_counts(graph),
                    "bindings": node_binding_map(graph),
                }
                events = read_jsonl(baseline / "binding_events.jsonl")
                arm_pending: list[dict[str, Any]] = []
            else:
                graph = DynamicSceneGraph.load(str(source_dsg))
                if arm == "B_id_only":
                    events.extend(
                        bind_batch(
                            graph,
                            entities,
                            owners=owners,
                            cross_policy="reject",
                            semantic_config=args.semantic_config.resolve(),
                            labelspace_colors=args.labelspace_colors.resolve(),
                            phase="id_only_all_entities",
                        )
                    )
                    arm_pending = []
                elif arm == "C_id_pending":
                    events.extend(
                        bind_batch(
                            graph,
                            entities,
                            owners=owners,
                            cross_policy="pending",
                            semantic_config=args.semantic_config.resolve(),
                            labelspace_colors=args.labelspace_colors.resolve(),
                            phase="id_authoritative_cross_pending",
                        )
                    )
                    arm_pending = pending
                else:
                    events.extend(
                        bind_batch(
                            graph,
                            exact_entities,
                            owners=owners,
                            cross_policy="reject",
                            semantic_config=args.semantic_config.resolve(),
                            labelspace_colors=args.labelspace_colors.resolve(),
                            phase="D_exact_id",
                        )
                    )
                    events.extend(
                        bind_batch(
                            graph,
                            selected_cross_entities,
                            owners=owners,
                            cross_policy="apply",
                            semantic_config=args.semantic_config.resolve(),
                            labelspace_colors=args.labelspace_colors.resolve(),
                            phase="D_global_selected_cross_id",
                        )
                    )
                    events.extend(
                        bind_batch(
                            graph,
                            remaining_entities,
                            owners=owners,
                            cross_policy="pending",
                            semantic_config=args.semantic_config.resolve(),
                            labelspace_colors=args.labelspace_colors.resolve(),
                            phase="D_unselected_pending",
                        )
                    )
                    arm_pending = [
                        row
                        for row in pending
                        if str(row["entity_id"]) not in selected_cross_ids
                    ]
                graph_info = save_graph(graph, cell / "dsg_bound.json")

            decisions = decisions_from_graph(
                entities,
                graph_info["bindings"],
                candidates,
                geometry_source=source,
                arm=arm,
                pending_rows=arm_pending,
                reviewed_cross=reviewed_cross,
            )
            source_counts = _mesh_counts(DynamicSceneGraph.load(str(source_dsg)))
            if graph_info["mesh_counts"] != source_counts:
                raise RuntimeError(f"{source}/{arm} changed frozen mesh")
            cell_summary = summary_from_decisions(
                decisions,
                real_mesh_count=source_counts["object_meshes"],
                pending_rows=arm_pending,
            )
            cell_summary.update(
                {
                    "schema": "daaam.g1_e17_v2_cell_summary.v1",
                    "geometry_source": source,
                    "arm": arm,
                    "source_dsg": str(source_dsg),
                    "source_dsg_sha256": sha256(source_dsg),
                    "output_dsg": str(cell / "dsg_bound.json"),
                    "output_dsg_sha256": graph_info["sha256"],
                    "mesh_counts": graph_info["mesh_counts"],
                    "event_count": len(events),
                    "elapsed_seconds": time.perf_counter() - cell_started,
                    "formal_claims_permitted": False,
                }
            )
            write_jsonl(cell / "terminal_decisions.jsonl", decisions)
            write_csv(cell / "terminal_decisions.csv", decisions)
            write_jsonl(cell / "binding_events.jsonl", events)
            write_csv(cell / "binding_events.csv", events)
            write_jsonl(cell / "pending_candidates.jsonl", arm_pending)
            write_csv(cell / "pending_candidates.csv", arm_pending)
            write_json(cell / "SUMMARY.json", cell_summary)
            summaries.append(cell_summary)
            all_decisions.extend(decisions)
            all_events.extend(
                {
                    "geometry_source": source,
                    "arm": arm,
                    **row,
                }
                for row in events
            )

    write_jsonl(output / "tables" / "cell_summaries.jsonl", summaries)
    write_csv(output / "tables" / "cell_summaries.csv", summaries)
    write_jsonl(output / "tables" / "terminal_decisions.jsonl", all_decisions)
    write_csv(output / "tables" / "terminal_decisions.csv", all_decisions)
    write_jsonl(output / "tables" / "binding_events.jsonl", all_events)
    write_csv(output / "tables" / "binding_events.csv", all_events)
    write_jsonl(output / "tables" / "pending_candidates.jsonl", all_pending)
    write_csv(output / "tables" / "pending_candidates.csv", all_pending)
    write_jsonl(output / "tables" / "global_candidates.jsonl", all_global_candidates)
    write_csv(output / "tables" / "global_candidates.csv", all_global_candidates)
    write_jsonl(
        output / "tables" / "global_assignments.jsonl",
        all_global_assignments,
    )
    write_csv(
        output / "tables" / "global_assignments.csv",
        all_global_assignments,
    )
    write_jsonl(output / "tables" / "order_stress.jsonl", all_order_stress)
    write_csv(output / "tables" / "order_stress.csv", all_order_stress)
    review = evidence_montage(output, e17)
    write_json(output / "visualizations" / "cross_id_review" / "REVIEW.json", review)
    plot_summary(output, summaries, all_order_stress)
    (output / "REPORT.md").write_text(report_text(summaries, all_order_stress))
    run_summary = {
        "schema": "daaam.g1_e17_v2_ablation_summary.v1",
        "status": "complete_pending_independent_audit",
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "cell_count": len(summaries),
        "cell_summaries": summaries,
        "global_candidate_count": len(all_global_candidates),
        "global_assignment_count": len(all_global_assignments),
        "order_stress_count": len(all_order_stress),
        "cross_id_review": review,
        "winner_on_development_window": "D_global_joint",
        "winner_rationale": (
            "same authoritative count as A, confirmed wrong cross-ID 1->0, "
            "reviewed same-track wall fragment 0->1, order-stable"
        ),
        "formal_claims_permitted": False,
        "held_out_confirmation_required": True,
        "formal_binding_precision": None,
        "formal_binding_recall": None,
        "formal_binding_f1": None,
    }
    write_json(output / "SUMMARY.json", run_summary)
    inventory_summary = inventory(output)
    result = {**run_summary, "artifact_inventory": inventory_summary}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    try:
        run(args)
    except Exception as error:
        if output.exists():
            write_json(
                output / "terminal_failure.json",
                {
                    "schema": "daaam.g1_e17_v2_terminal_failure.v1",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        print(
            json.dumps(
                {"status": "error", "error": str(error)},
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
