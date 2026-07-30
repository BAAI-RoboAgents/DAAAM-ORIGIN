#!/usr/bin/env python3
"""Independently verify the G1 E17-v2 A/B/C/D binding-policy ablation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from daaam.experiments.e17_ablation import (  # noqa: E402
    D_GLOBAL_GATE,
    global_one_to_one_assignment,
    symmetric_volume_ratio,
)
from daaam.experiments.e17_support import (  # noqa: E402
    BindingThreshold,
    aabb_evidence,
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


DEFAULT_RUN = (
    REPOSITORY_ROOT
    / "experiments"
    / "g1_20260724_473_573_v1_1"
    / "comparisons"
    / "e17_binding_policy_ablation_20260730"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def reload_graph(path: Path) -> dict[str, Any]:
    from spark_dsg import DsgLayers, DynamicSceneGraph

    graph = DynamicSceneGraph.load(str(path))
    counts = _mesh_counts(graph)
    bindings = {}
    node_owners = {}
    for node in graph.get_layer(DsgLayers.OBJECTS).nodes:
        metadata = dict(node.attributes.metadata.get() or {})
        entity_id = str(metadata.get("entity_id") or "").strip()
        if not entity_id:
            continue
        if entity_id in bindings:
            raise RuntimeError(f"duplicate durable entity: {entity_id}")
        if str(node.id) in node_owners:
            raise RuntimeError(f"duplicate durable mesh owner: {node.id}")
        mesh = node.attributes.mesh()
        vertices = 0 if mesh is None else int(mesh.num_vertices())
        evidence = dict(metadata.get("entity_binding") or {})
        candidate_semantic_id = int(
            evidence.get("candidate_semantic_id", node.attributes.semantic_label)
        )
        bindings[entity_id] = {
            "node_id": str(node.id),
            "committed_semantic_id": int(node.attributes.semantic_label),
            "candidate_semantic_id": candidate_semantic_id,
            "mesh_vertices": vertices,
            "description": str(metadata.get("description") or ""),
            "mesh_binding_status": metadata.get("mesh_binding_status"),
        }
        node_owners[str(node.id)] = entity_id
    return {"mesh_counts": counts, "bindings": bindings}


def exact_d_offset_rows(assignments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    threshold = BindingThreshold(
        "D_global_joint",
        D_GLOBAL_GATE.maximum_center_distance_m,
        D_GLOBAL_GATE.maximum_aabb_gap_m,
    )
    rows = []
    for assignment in assignments:
        entity_position = np.asarray(
            assignment["entity_position_m"],
            dtype=np.float64,
        )
        node_position = np.asarray(
            assignment["node_position_m"],
            dtype=np.float64,
        )
        direction = entity_position - node_position
        norm = float(np.linalg.norm(direction))
        direction = (
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            if norm <= 1.0e-12
            else direction / norm
        )
        volume_ratio = symmetric_volume_ratio(
            assignment["entity_dimensions_m"],
            assignment["node_dimensions_m"],
        )
        for dose in (0.0, 0.1, 0.3, 0.6):
            perturbed = entity_position + dose * direction
            evidence = aabb_evidence(
                perturbed,
                assignment["entity_dimensions_m"],
                node_position,
                assignment["node_dimensions_m"],
                threshold,
            )
            checks = {
                "shared_upstream_track": bool(assignment["shared_track_ids"]),
                "center_distance": (
                    evidence["center_distance_m"]
                    <= D_GLOBAL_GATE.maximum_center_distance_m
                ),
                "aabb_gap": (
                    evidence["aabb_gap_m"]
                    <= D_GLOBAL_GATE.maximum_aabb_gap_m
                ),
                "aabb_iou": (
                    evidence["aabb_iou"] >= D_GLOBAL_GATE.minimum_aabb_iou
                ),
                "volume_ratio": (
                    volume_ratio
                    <= D_GLOBAL_GATE.maximum_symmetric_volume_ratio
                ),
            }
            rows.append(
                {
                    "schema": "daaam.g1_e17_v2_D_offset_probe.v1",
                    "geometry_source": assignment["geometry_source"],
                    "entity_ordinal": assignment["entity_ordinal"],
                    "candidate_semantic_id": assignment[
                        "candidate_semantic_id"
                    ],
                    "node_id": assignment["node_id"],
                    "dose_m": dose,
                    "injection_direction": (
                        "radially_away_from_selected_candidate_center"
                    ),
                    "center_distance_m": evidence["center_distance_m"],
                    "aabb_gap_m": evidence["aabb_gap_m"],
                    "aabb_iou": evidence["aabb_iou"],
                    "symmetric_volume_ratio": volume_ratio,
                    "joint_gate_checks": checks,
                    "accepted_by_D_joint_gate": all(checks.values()),
                }
            )
    return rows


def plot_offset(run: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for source, color in (("single_pass", "#1f77b4"), ("adaptive", "#ff7f0e")):
        selected = [row for row in rows if row["geometry_source"] == source]
        axis.plot(
            [float(row["dose_m"]) for row in selected],
            [float(row["center_distance_m"]) for row in selected],
            marker="o",
            color=color,
            label=f"{source} center distance",
        )
    axis.axhline(
        D_GLOBAL_GATE.maximum_center_distance_m,
        color="#d62728",
        linestyle="--",
        label="D center gate 1.0m",
    )
    axis.set_xlabel("radial-away entity offset dose (m)")
    axis.set_ylabel("E51–E77 center distance (m)")
    axis.set_title("D selected-pair position-offset response")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(run / "visualizations" / "04_D_offset_response.png", dpi=180)
    plt.close(fig)


def append_audit_report(
    run: Path,
    *,
    order_counts: Mapping[str, Counter[int]],
    offset_rows: Sequence[Mapping[str, Any]],
) -> None:
    medium_heavy = [
        row
        for row in offset_rows
        if float(row["dose_m"]) >= 0.3
    ]
    detected = sum(not row["accepted_by_D_joint_gate"] for row in medium_heavy)
    lines = [
        "",
        "## 独立审计补充",
        "",
        (
            "八份 durable DSG 已全部独立重载；每份 87/87 终态完整、"
            "mesh 所有权唯一、描述和 real mesh 有效，且输入/输出 mesh 计数与"
            "顶点数完全一致。B 与 C 的 DSG hash 在每个几何源上相同，证明 C "
            "只增加 pending evidence，没有偷偷提交跨 ID 语义。"
        ),
        "",
        (
            "A 在 22 种输入顺序下，single-pass 的 E77 占有者出现 "
            f"{len(order_counts['single_pass'])} 种、adaptive 出现 "
            f"{len(order_counts['adaptive'])} 种；D 对 100 次候选乱序始终输出 "
            "E51→E77。"
        ),
        "",
        (
            f"D 的 0.3/0.6m 中重度位置偏移共 {len(medium_heavy)} 个运行，"
            f"联合门拒绝 {detected}/{len(medium_heavy)}；0.1m 轻度偏移仍保留。"
            "这是 chosen-pair gate 灵敏度，不是正式 wrong-mesh 检出率。"
        ),
        "",
        (
            "开发窗口上的推荐为 D：single-pass 保持 54/87、adaptive 保持 "
            "56/87，同时已确认 wrong cross-ID 从 1 降至 0，并得到一个同 track "
            "墙面碎片绑定。由于仅复核了一个跨 ID 正例与一个负例，不能外推为"
            "总体 precision=100%。"
        ),
        "",
    ]
    with (run / "REPORT.md").open("a") as stream:
        stream.write("\n".join(lines))


def verify(run: Path) -> dict[str, Any]:
    if not run.is_dir():
        raise FileNotFoundError(run)
    if (run / "terminal_failure.json").exists():
        raise RuntimeError("ablation contains terminal_failure.json")
    summary = json.loads((run / "SUMMARY.json").read_text())
    preregistration = json.loads((run / "PRE_REGISTRATION.json").read_text())
    if summary.get("status") != "complete_pending_independent_audit":
        raise ValueError("ablation is not ready for audit")
    if int(summary.get("cell_count", 0)) != 8:
        raise ValueError("ablation must contain eight cells")
    if preregistration.get("formal_gt") is not None:
        raise ValueError("no-GT ablation unexpectedly names formal GT")

    checks = []
    cell_lookup = {
        (str(row["geometry_source"]), str(row["arm"])): row
        for row in summary["cell_summaries"]
    }
    decision_lookup = {}
    for source in ("single_pass", "adaptive"):
        source_counts = None
        for arm in (
            "A_current_strict",
            "B_id_only",
            "C_id_pending",
            "D_global_joint",
        ):
            cell = run / "cells" / f"{source}_{arm}"
            output_dsg = cell / "dsg_bound.json"
            decisions = read_jsonl(cell / "terminal_decisions.jsonl")
            pending = read_jsonl(cell / "pending_candidates.jsonl")
            events = read_jsonl(cell / "binding_events.jsonl")
            cell_summary = json.loads((cell / "SUMMARY.json").read_text())
            expected = cell_lookup[(source, arm)]
            if cell_summary != expected:
                raise ValueError(f"{source}/{arm}: summary copies differ")
            if len(decisions) != 87 or len(
                {str(row["entity_id"]) for row in decisions}
            ) != 87:
                raise ValueError(f"{source}/{arm}: terminal coverage failed")
            loaded = reload_graph(output_dsg)
            if sha256(output_dsg) != expected["output_dsg_sha256"]:
                raise ValueError(f"{source}/{arm}: DSG hash mismatch")
            if loaded["mesh_counts"] != expected["mesh_counts"]:
                raise ValueError(f"{source}/{arm}: mesh counts mismatch")
            if source_counts is None:
                source_counts = loaded["mesh_counts"]
            elif loaded["mesh_counts"] != source_counts:
                raise ValueError(f"{source}/{arm}: arm changed frozen geometry")
            matched = [
                row for row in decisions if row["status"] == "matched_real_mesh"
            ]
            rejected = [
                row for row in decisions if row["status"] != "matched_real_mesh"
            ]
            if len(matched) != int(expected["matched_real_mesh"]):
                raise ValueError(f"{source}/{arm}: matched mismatch")
            if len(rejected) != int(
                expected["rejected_no_authoritative_mesh"]
            ):
                raise ValueError(f"{source}/{arm}: rejected mismatch")
            node_ids = [str(row["node_id"]) for row in matched]
            if len(node_ids) != len(set(node_ids)):
                raise ValueError(f"{source}/{arm}: duplicate mesh ownership")
            if set(loaded["bindings"]) != {
                str(row["entity_id"]) for row in matched
            }:
                raise ValueError(f"{source}/{arm}: durable entities differ")
            for row in matched:
                durable = loaded["bindings"][str(row["entity_id"])]
                if (
                    durable["node_id"] != str(row["node_id"])
                    or durable["candidate_semantic_id"]
                    != int(row["candidate_semantic_id"])
                    or durable["mesh_vertices"] <= 0
                    or not durable["description"]
                    or durable["mesh_binding_status"] != "matched_real_mesh"
                ):
                    raise ValueError(
                        f"{source}/{arm}: invalid durable entity "
                        f"{row['entity_ordinal']}"
                    )
            if len(pending) != int(expected["pending_candidate_pairs"]):
                raise ValueError(f"{source}/{arm}: pending count mismatch")
            if len(events) != int(expected["event_count"]):
                raise ValueError(f"{source}/{arm}: event count mismatch")
            if any(
                expected.get(key) is not None
                for key in (
                    "formal_binding_precision",
                    "formal_binding_recall",
                    "formal_binding_f1",
                )
            ):
                raise ValueError(f"{source}/{arm}: formal metric populated")
            decision_lookup[(source, arm)] = decisions
            checks.append(
                {
                    "geometry_source": source,
                    "arm": arm,
                    "terminal_count": len(decisions),
                    "matched": len(matched),
                    "rejected": len(rejected),
                    "pending_pairs": len(pending),
                    "events": len(events),
                    "output_hash_verified": True,
                    "mesh_invariant_verified": True,
                    "durable_bindings_verified": True,
                }
            )

        b = cell_lookup[(source, "B_id_only")]
        c = cell_lookup[(source, "C_id_pending")]
        if b["output_dsg_sha256"] != c["output_dsg_sha256"]:
            raise ValueError(f"{source}: B and C authoritative DSG differ")
        a_cross = [
            row
            for row in decision_lookup[(source, "A_current_strict")]
            if row["status"] == "matched_real_mesh"
            and not row["semantic_id_match"]
        ]
        d_cross = [
            row
            for row in decision_lookup[(source, "D_global_joint")]
            if row["status"] == "matched_real_mesh"
            and not row["semantic_id_match"]
        ]
        if [
            (
                int(row["entity_ordinal"]),
                int(row["candidate_semantic_id"]),
                row["review_status"],
            )
            for row in a_cross
        ] != [(8, 77, "confirmed_wrong_mesh")]:
            raise ValueError(f"{source}: A negative control changed")
        if [
            (
                int(row["entity_ordinal"]),
                int(row["candidate_semantic_id"]),
                row["review_status"],
            )
            for row in d_cross
        ] != [(51, 77, "reviewed_same_wall_track_fragment")]:
            raise ValueError(f"{source}: D assignment changed")

    order_rows = read_jsonl(run / "tables" / "order_stress.jsonl")
    if len(order_rows) != 44:
        raise ValueError("order-stress ledger must contain 44 rows")
    order_counts = {
        source: Counter(
            int(row["current_greedy_entity_ordinal"])
            for row in order_rows
            if row["geometry_source"] == source
        )
        for source in ("single_pass", "adaptive")
    }
    if any(len(counts) < 2 for counts in order_counts.values()):
        raise ValueError("A greedy assignment did not respond to order stress")
    assignments = read_jsonl(run / "tables" / "global_assignments.jsonl")
    if len(assignments) != 2:
        raise ValueError("D must contain one assignment per geometry source")
    for source in ("single_pass", "adaptive"):
        source_candidates = [
            row
            for row in read_jsonl(run / "tables" / "global_candidates.jsonl")
            if row["geometry_source"] == source
        ]
        expected = [
            (
                int(row["entity_ordinal"]),
                int(row["candidate_semantic_id"]),
                str(row["node_id"]),
            )
            for row in assignments
            if row["geometry_source"] == source
        ]
        for seed in range(100):
            shuffled = list(source_candidates)
            random.Random(seed).shuffle(shuffled)
            actual = [
                (
                    int(row["entity_ordinal"]),
                    int(row["candidate_semantic_id"]),
                    str(row["node_id"]),
                )
                for row in global_one_to_one_assignment(shuffled)
            ]
            if actual != expected:
                raise ValueError(f"{source}: D changed under candidate shuffle")

    offset_rows = exact_d_offset_rows(assignments)
    for source in ("single_pass", "adaptive"):
        source_rows = [
            row for row in offset_rows if row["geometry_source"] == source
        ]
        accepted = {
            float(row["dose_m"]): bool(row["accepted_by_D_joint_gate"])
            for row in source_rows
        }
        if accepted != {0.0: True, 0.1: True, 0.3: False, 0.6: False}:
            raise ValueError(f"{source}: D offset response is unexpected")
    write_jsonl(run / "analysis" / "D_offset_probe.jsonl", offset_rows)
    write_csv(run / "analysis" / "D_offset_probe.csv", offset_rows)
    plot_offset(run, offset_rows)
    write_json(
        run / "analysis" / "order_stability_summary.json",
        {
            "schema": "daaam.g1_e17_v2_order_stability_summary.v1",
            "A_distinct_owner_count": {
                source: len(counts) for source, counts in order_counts.items()
            },
            "A_owner_histogram": {
                source: dict(sorted(counts.items()))
                for source, counts in order_counts.items()
            },
            "D_candidate_shuffle_runs_per_source": 100,
            "D_assignment_invariant": True,
            "D_assignment": "E51->E77",
        },
    )
    append_audit_report(
        run,
        order_counts=order_counts,
        offset_rows=offset_rows,
    )
    shutil.copy2(
        Path(__file__),
        run / "source_snapshot" / Path(__file__).name,
    )
    inventory_summary = inventory(run)
    audit = {
        "schema": "daaam.g1_e17_v2_independent_audit.v1",
        "status": "passed_development_window_exploratory",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "cell_checks": checks,
        "cell_count": len(checks),
        "terminal_decision_count": sum(row["terminal_count"] for row in checks),
        "all_dsg_outputs_reload": True,
        "mesh_invariant": True,
        "unique_mesh_ownership": True,
        "B_C_authoritative_outputs_identical": True,
        "A_confirmed_wrong_negative_control_reproduced": True,
        "D_reviewed_same_track_assignment_reproduced": True,
        "A_order_sensitive": True,
        "A_distinct_owner_count": {
            source: len(counts) for source, counts in order_counts.items()
        },
        "D_candidate_shuffle_runs": 200,
        "D_assignment_order_invariant": True,
        "D_medium_heavy_offset_detection": "4/4",
        "formal_metrics_remain_unavailable": True,
        "held_out_confirmation_required": True,
        "artifact_inventory": inventory_summary,
    }
    write_json(run / "INDEPENDENT_AUDIT.json", audit)
    completion = {
        "schema": "daaam.g1_e17_v2_completion.v1",
        "status": "complete_independently_audited",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "winner_on_development_window": "D_global_joint",
        "artifact_inventory_file_count": inventory_summary["file_count"],
        "artifact_inventory_root_sha256": inventory_summary["root_sha256"],
        "terminal_decision_count": audit["terminal_decision_count"],
        "formal_claims_permitted": False,
        "held_out_confirmation_required": True,
    }
    write_json(run / "COMPLETION.json", completion)
    print(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False))
    return audit


def main() -> int:
    args = parse_args()
    try:
        verify(args.run.expanduser().resolve())
    except Exception as error:
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
