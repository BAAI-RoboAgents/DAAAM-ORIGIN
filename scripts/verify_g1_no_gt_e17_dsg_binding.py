#!/usr/bin/env python3
"""Independently verify and enrich the G1 no-GT E17 binding run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    REPOSITORY_ROOT / "experiments" / "g1_20260724_473_573_v1_1"
)
DEFAULT_RUN = (
    EXPERIMENT_ROOT
    / "runs"
    / "diagnostic_gt_free_e17_e14e16fed_binding_20260730"
)
E14_E8_PANEL = (
    EXPERIMENT_ROOT
    / "runs"
    / "diagnostic_gt_free_e14_safe035_unique_obs8_seed0_20260730"
    / "annotation_census_obs_08_seed_0"
    / "review_panels"
    / "census_007_E008_source_000480.jpg"
)
E13_E77_OVERLAY = (
    EXPERIMENT_ROOT
    / "runs"
    / "diagnostic_gt_free_e13_e14_safety_ablation_20260730"
    / "variants"
    / "safe_merge_0p35m"
    / "frames"
    / "00000037"
    / "entity_overlay.jpg"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def inventory(run: Path) -> dict[str, Any]:
    excluded = {
        "artifact_inventory.csv",
        "artifact_inventory.jsonl",
        "inventory_summary.json",
        "COMPLETION.json",
        "INDEPENDENT_AUDIT.json",
        "terminal_failure.json",
    }
    rows = []
    for path in sorted(run.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    root = hashlib.sha256()
    for row in rows:
        root.update(
            (
                f"{row['relative_path']}\0{row['size_bytes']}\0{row['sha256']}\n"
            ).encode()
        )
    write_jsonl(run / "artifact_inventory.jsonl", rows)
    write_csv(run / "artifact_inventory.csv", rows)
    result = {
        "schema": "daaam.g1_e17_artifact_inventory.v1",
        "file_count": len(rows),
        "total_size_bytes": int(sum(row["size_bytes"] for row in rows)),
        "root_sha256": root.hexdigest(),
        "excluded_relative_paths": sorted(excluded),
    }
    write_json(run / "inventory_summary.json", result)
    return result


def independently_load_bound_dsg(path: Path) -> dict[str, Any]:
    from spark_dsg import DsgLayers, DynamicSceneGraph

    graph = DynamicSceneGraph.load(str(path))
    real_mesh_nodes = 0
    real_mesh_vertices = 0
    entities = {}
    invalid = []
    for node in graph.get_layer(DsgLayers.OBJECTS).nodes:
        mesh = node.attributes.mesh()
        vertices = 0 if mesh is None else int(mesh.num_vertices())
        if vertices > 0:
            real_mesh_nodes += 1
            real_mesh_vertices += vertices
        metadata = dict(node.attributes.metadata.get() or {})
        entity_id = str(metadata.get("entity_id") or "").strip()
        if not entity_id:
            continue
        if entity_id in entities:
            invalid.append(f"duplicate entity in DSG: {entity_id}")
        entities[entity_id] = {
            "node_id": str(node.id),
            "semantic_id": int(node.attributes.semantic_label),
            "has_real_mesh": vertices > 0,
            "description": str(metadata.get("description") or "").strip(),
            "mesh_binding_status": metadata.get("mesh_binding_status"),
        }
    if invalid:
        raise RuntimeError("; ".join(invalid))
    return {
        "real_mesh_nodes": real_mesh_nodes,
        "real_mesh_vertices": real_mesh_vertices,
        "entities": entities,
    }


def nearest_rejection_rows(
    *,
    variant: str,
    decisions: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_entity: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        by_entity.setdefault(str(candidate["entity_id"]), []).append(candidate)
    rows = []
    for decision in decisions:
        if decision["status"] != "rejected_no_mesh":
            continue
        entity_id = str(decision["entity_id"])
        options = by_entity[entity_id]
        nearest_any = min(
            options,
            key=lambda row: (
                float(row["aabb_gap_m"]),
                float(row["center_distance_m"]),
                int(row["candidate_semantic_id"]),
            ),
        )
        eligible = [
            row for row in options if bool(row["eligible_before_assignment"])
        ]
        nearest_eligible = (
            min(
                eligible,
                key=lambda row: (
                    not bool(row["semantic_id_match"]),
                    float(row["aabb_gap_m"]),
                    float(row["center_distance_m"]),
                    int(row["candidate_semantic_id"]),
                ),
            )
            if eligible
            else None
        )
        exact = [
            row for row in options if bool(row["semantic_id_match"])
        ]
        nearest_exact = (
            min(
                exact,
                key=lambda row: (
                    float(row["aabb_gap_m"]),
                    float(row["center_distance_m"]),
                ),
            )
            if exact
            else None
        )
        rows.append(
            {
                "schema": "daaam.g1_e17_rejected_nearest_candidate.v1",
                "variant": variant,
                "geometry_source": decision["geometry_source"],
                "threshold": decision["threshold"],
                "entity_id": entity_id,
                "entity_ordinal": decision["entity_ordinal"],
                "entity_label": decision["entity_label"],
                "terminal_status": decision["status"],
                "nearest_any_node_id": nearest_any["node_id"],
                "nearest_any_semantic_id": nearest_any["candidate_semantic_id"],
                "nearest_any_center_distance_m": nearest_any["center_distance_m"],
                "nearest_any_aabb_gap_m": nearest_any["aabb_gap_m"],
                "nearest_any_reserved_owner": nearest_any["reserved_owner"],
                "eligible_before_assignment_count": len(eligible),
                "nearest_eligible_node_id": (
                    None if nearest_eligible is None else nearest_eligible["node_id"]
                ),
                "nearest_eligible_semantic_id": (
                    None
                    if nearest_eligible is None
                    else nearest_eligible["candidate_semantic_id"]
                ),
                "nearest_eligible_center_distance_m": (
                    None
                    if nearest_eligible is None
                    else nearest_eligible["center_distance_m"]
                ),
                "nearest_eligible_aabb_gap_m": (
                    None
                    if nearest_eligible is None
                    else nearest_eligible["aabb_gap_m"]
                ),
                "exact_semantic_candidate_exists": nearest_exact is not None,
                "nearest_exact_node_id": (
                    None if nearest_exact is None else nearest_exact["node_id"]
                ),
                "nearest_exact_center_distance_m": (
                    None
                    if nearest_exact is None
                    else nearest_exact["center_distance_m"]
                ),
                "nearest_exact_aabb_gap_m": (
                    None if nearest_exact is None else nearest_exact["aabb_gap_m"]
                ),
                "conflict_event_count": decision["conflict_event_count"],
                "reserved_owner_rejection_count": (
                    decision["reserved_owner_rejection_count"]
                ),
            }
        )
    return rows


def save_spatial_fallback_evidence(run: Path) -> dict[str, Any]:
    if not E14_E8_PANEL.is_file() or not E13_E77_OVERLAY.is_file():
        raise FileNotFoundError("spatial fallback RGB evidence is incomplete")
    target = run / "analysis" / "spatial_fallback_evidence"
    target.mkdir(parents=True, exist_ok=True)
    e8_copy = target / "E008_ceiling_entity_evidence.jpg"
    e77_copy = target / "E077_wall_mesh_source_overlay.jpg"
    shutil.copy2(E14_E8_PANEL, e8_copy)
    shutil.copy2(E13_E77_OVERLAY, e77_copy)

    first = plt.imread(e8_copy)
    second = plt.imread(e77_copy)
    fig, axes = plt.subplots(2, 1, figsize=(18, 13), constrained_layout=True)
    axes[0].imshow(first)
    axes[0].set_title("Binding entity E8: ceiling / recessed lights")
    axes[1].imshow(second)
    axes[1].set_title(
        "Chosen mesh source E77: large wall mask (label E77 near wall slogan)"
    )
    for axis in axes:
        axis.axis("off")
    montage = target / "E008_to_E077_wrong_mesh_review.jpg"
    fig.savefig(montage, dpi=150)
    plt.close(fig)

    finding = {
        "schema": "daaam.g1_e17_codex_spatial_fallback_review.v1",
        "reviewer": "Codex visual engineering review; not independent human GT",
        "entity_ordinal": 8,
        "entity_identity": "ceiling with recessed lights",
        "candidate_semantic_id": 77,
        "candidate_identity_from_rgb_mask": "large wall surface around wall slogan",
        "verdict": "incorrect_wrong_mesh",
        "applies_to_geometry_sources": ["single_pass", "adaptive"],
        "applies_to_thresholds": ["strict", "medium", "wide"],
        "center_distance_m": 1.7323761131599904,
        "aabb_gap_m": 0.01512843869703806,
        "aabb_iou": 0.0,
        "accepted_by": ["aabb_gap"],
        "why": (
            "E8 crop/mask is the ceiling strip; E77 overlay is the wall. "
            "They are adjacent structural surfaces, not the same entity. "
            "The binding passed solely because AABB gap 1.51cm is below even "
            "the strict 2.5cm gate, despite 1.732m center distance and zero IoU."
        ),
        "evidence": [
            {
                "path": str(e8_copy.relative_to(run)),
                "sha256": sha256(e8_copy),
            },
            {
                "path": str(e77_copy.relative_to(run)),
                "sha256": sha256(e77_copy),
            },
            {
                "path": str(montage.relative_to(run)),
                "sha256": sha256(montage),
            },
        ],
        "formal_gt": False,
    }
    write_json(target / "REVIEW.json", finding)
    return finding


def append_verified_findings(
    run: Path,
    *,
    summary: Mapping[str, Any],
    spatial_finding: Mapping[str, Any],
    nearest_rows: Sequence[Mapping[str, Any]],
) -> None:
    cell_lookup = {
        str(row["variant"]): row for row in summary["cell_summaries"]
    }
    lines = [
        "",
        "## 独立审计后的关键发现",
        "",
        (
            "六份输出 DSG 已独立重载；每份均有 87/87 个终态、无 mesh 多重"
            "所有权、无 rejected 实体残留，输入/输出 real-mesh 数量与顶点数一致。"
        ),
        "",
        (
            "三个门限在同一几何源上的最终 entity→node 映射完全一致："
            "medium/wide 没有增加实体，也没有改绑。single-pass 始终为 "
            f"{cell_lookup['single_pass_strict']['matched_real_mesh']} matched / "
            f"{cell_lookup['single_pass_strict']['rejected_no_mesh']} rejected；"
            "adaptive 始终为 "
            f"{cell_lookup['adaptive_strict']['matched_real_mesh']} / "
            f"{cell_lookup['adaptive_strict']['rejected_no_mesh']}。"
        ),
        "",
        (
            "但 strict/medium/wide 都发生同一个已复核错误：E8 天花板被绑到 "
            "E77 墙面 mesh。中心距离 1.732m、AABB IoU=0，只因 AABB gap=1.51cm "
            "而通过 strict 的 2.5cm 门。这是当前 `center OR AABB-gap` 策略在"
            "相邻结构面上的确定失败样本。"
        ),
        "",
        (
            f"对 {len(nearest_rows)} 个 rejected×variant 终态，"
            "`analysis/rejected_nearest_candidates.*` 已补齐 nearest-any、"
            "nearest-eligible、nearest-exact、冲突与 reserved-owner 字段。"
        ),
        "",
        "### 当前可执行建议",
        "",
        (
            "1. 正式 authoritative binding 默认只提交 semantic-ID 一致的 mesh；"
            "不同 ID 的空间候选保留为 `pending_spatial_review`，不得直接写入 DSG。"
        ),
        (
            "2. 若必须启用跨 ID fallback，至少同时要求类别/描述兼容，且采用 "
            "`center AND gap` 或非零 IoU/形状约束；E8→E77 应被拒绝。"
        ),
        (
            "3. 在当前六个候选中 strict 没有损失任何最终覆盖、候选对最少，"
            "因此只可作为相对较保守的诊断默认值；在修复跨 ID fallback 前，"
            "仍不能称为 precision-safe。"
        ),
        (
            "4. 以 provenance-only 后验门计算，single-pass 可提交 53 个、"
            "adaptive 可提交 55 个；这仍不是 53/55 个真实正确物体，因为 "
            "E11–E16 的 mask、名称、碎片和物理实例正确性没有完整 GT。"
        ),
        "",
        (
            f"Codex 错绑复核记录：`{spatial_finding['verdict']}`；双图证据见 "
            "`analysis/spatial_fallback_evidence/E008_to_E077_wrong_mesh_review.jpg`。"
        ),
        "",
    ]
    with (run / "REPORT.md").open("a") as stream:
        stream.write("\n".join(lines))


def verify(run: Path) -> dict[str, Any]:
    if not run.is_dir():
        raise FileNotFoundError(run)
    if (run / "terminal_failure.json").exists():
        raise RuntimeError("run contains terminal_failure.json")
    summary = json.loads((run / "RUN_SUMMARY.json").read_text())
    manifest = json.loads((run / "inputs" / "INPUT_MANIFEST.json").read_text())
    if summary.get("status") != "complete_pending_independent_audit":
        raise ValueError("run is not ready for independent audit")
    if int(summary.get("named_entity_count", 0)) != 87:
        raise ValueError("unexpected E17 named entity count")
    if int(summary.get("cell_count", 0)) != 6:
        raise ValueError("E17 must contain six cells")
    if any(
        summary.get(key) is not None
        for key in (
            "formal_binding_precision",
            "formal_binding_recall",
            "formal_binding_f1",
        )
    ):
        raise ValueError("formal no-GT binding metrics were populated")

    checks = []
    nearest_rows = []
    source_hashes = {
        name: value["sha256"]
        for name, value in manifest["geometry_sources"].items()
    }
    for cell in summary["cell_summaries"]:
        variant_name = str(cell["variant"])
        variant = run / "variants" / variant_name
        audit_path = variant / "binding_audit.json"
        output_path = variant / "dsg_bound.json"
        candidate_path = variant / "candidate_matrix.jsonl"
        terminal_path = variant / "terminal_decisions.jsonl"
        for path in (
            audit_path,
            output_path,
            candidate_path,
            terminal_path,
            variant / "binding_events.jsonl",
            variant / "offset_probe.jsonl",
            variant / "review_queue.jsonl",
            variant / "SUMMARY.json",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        audit = json.loads(audit_path.read_text())
        candidates = read_jsonl(candidate_path)
        decisions = read_jsonl(terminal_path)
        if len(candidates) != 87 * int(cell["real_mesh_count"]):
            raise ValueError(f"{variant_name}: incomplete candidate matrix")
        if len(decisions) != 87:
            raise ValueError(f"{variant_name}: incomplete terminal decisions")
        if len({row["entity_id"] for row in decisions}) != 87:
            raise ValueError(f"{variant_name}: duplicate terminal entity")
        matched = [row for row in decisions if row["status"] == "matched_real_mesh"]
        rejected = [row for row in decisions if row["status"] == "rejected_no_mesh"]
        if len(matched) != int(cell["matched_real_mesh"]):
            raise ValueError(f"{variant_name}: matched count mismatch")
        if len(rejected) != int(cell["rejected_no_mesh"]):
            raise ValueError(f"{variant_name}: rejected count mismatch")
        node_ids = [str(row["node_id"]) for row in matched]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError(f"{variant_name}: duplicate mesh ownership")
        if sha256(output_path) != cell["output_dsg_sha256"]:
            raise ValueError(f"{variant_name}: output DSG hash mismatch")
        if sha256(audit_path) != cell["binding_audit_sha256"]:
            raise ValueError(f"{variant_name}: audit hash mismatch")
        source = str(cell["geometry_source"])
        source_path = Path(manifest["geometry_sources"][source]["path"])
        if sha256(source_path) != source_hashes[source]:
            raise ValueError(f"{variant_name}: frozen source DSG changed")
        loaded = independently_load_bound_dsg(output_path)
        if loaded["real_mesh_nodes"] != int(cell["real_mesh_count"]):
            raise ValueError(f"{variant_name}: real mesh count changed")
        if (
            loaded["real_mesh_vertices"]
            != int(audit["mesh_counts"]["object_mesh_vertices"])
        ):
            raise ValueError(f"{variant_name}: real mesh vertices changed")
        entity_nodes = loaded["entities"]
        if set(entity_nodes) != {str(row["entity_id"]) for row in matched}:
            raise ValueError(f"{variant_name}: DSG entity set mismatch")
        for decision in matched:
            node = entity_nodes[str(decision["entity_id"])]
            if (
                not node["has_real_mesh"]
                or not node["description"]
                or node["mesh_binding_status"] != "matched_real_mesh"
                or node["node_id"] != str(decision["node_id"])
            ):
                raise ValueError(
                    f"{variant_name}: invalid durable binding "
                    f"{decision['entity_id']}"
                )
        nearest_rows.extend(
            nearest_rejection_rows(
                variant=variant_name,
                decisions=decisions,
                candidates=candidates,
            )
        )
        checks.append(
            {
                "variant": variant_name,
                "terminal_entity_count": len(decisions),
                "matched_real_mesh": len(matched),
                "rejected_no_mesh": len(rejected),
                "unique_assigned_mesh_nodes": len(set(node_ids)),
                "reloaded_real_mesh_nodes": loaded["real_mesh_nodes"],
                "reloaded_real_mesh_vertices": loaded["real_mesh_vertices"],
                "source_hash_verified": True,
                "output_hash_verified": True,
                "audit_hash_verified": True,
                "durable_entity_bindings_verified": True,
            }
        )

    # Verify all thresholds preserve the exact same final assignment per source.
    transitions = read_jsonl(run / "tables" / "threshold_transitions.jsonl")
    if any(
        bool(row["gained_by_medium"])
        or bool(row["gained_by_wide"])
        or bool(row["reassigned_when_widened"])
        for row in transitions
    ):
        raise ValueError("threshold invariance disagrees with transition ledger")

    write_jsonl(
        run / "analysis" / "rejected_nearest_candidates.jsonl",
        nearest_rows,
    )
    write_csv(
        run / "analysis" / "rejected_nearest_candidates.csv",
        nearest_rows,
    )
    spatial_finding = save_spatial_fallback_evidence(run)
    finding_rows = []
    for cell in summary["cell_summaries"]:
        finding_rows.append(
            {
                "schema": "daaam.g1_e17_reviewed_binding_policy_result.v1",
                "variant": cell["variant"],
                "matched_real_mesh": cell["matched_real_mesh"],
                "provenance_consistent_semantic_id": (
                    cell["provenance_consistent_semantic_id"]
                ),
                "reviewed_spatial_fallback": (
                    cell["spatial_fallback_requires_review"]
                ),
                "reviewed_spatial_fallback_confirmed_wrong": 1,
                "confirmed_wrong_mesh_lower_bound_among_matches": (
                    1.0 / float(cell["matched_real_mesh"])
                ),
                "provenance_only_authoritative_candidate_count": (
                    cell["provenance_consistent_semantic_id"]
                ),
                "formal_binding_precision": None,
                "formal_binding_recall": None,
                "formal_binding_f1": None,
            }
        )
    write_jsonl(run / "analysis" / "reviewed_policy_results.jsonl", finding_rows)
    write_csv(run / "analysis" / "reviewed_policy_results.csv", finding_rows)
    append_verified_findings(
        run,
        summary=summary,
        spatial_finding=spatial_finding,
        nearest_rows=nearest_rows,
    )
    shutil.copy2(
        Path(__file__),
        run / "source_snapshot" / Path(__file__).name,
    )
    inventory_summary = inventory(run)
    audit = {
        "schema": "daaam.g1_no_gt_e17_independent_audit.v1",
        "status": "passed_with_confirmed_wrong_spatial_fallback",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "run": str(run),
        "checks": checks,
        "check_count": len(checks),
        "terminal_decision_count": sum(
            int(row["terminal_entity_count"]) for row in checks
        ),
        "all_six_dsg_outputs_reload": True,
        "all_entities_have_one_terminal_decision": True,
        "mesh_ownership_unique": True,
        "source_output_mesh_invariant": True,
        "threshold_final_assignments_invariant_within_source": True,
        "confirmed_wrong_spatial_fallback": spatial_finding,
        "confirmed_wrong_mesh_binding_count_per_variant": 1,
        "nearest_rejected_evidence_row_count": len(nearest_rows),
        "formal_binding_metrics_remain_unavailable": True,
        "artifact_inventory": inventory_summary,
    }
    write_json(run / "INDEPENDENT_AUDIT.json", audit)
    completion = {
        "schema": "daaam.g1_no_gt_e17_completion.v1",
        "status": "complete_independently_audited",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "independent_audit": (
            "passed engineering integrity; confirmed one wrong spatial fallback "
            "in every variant"
        ),
        "artifact_inventory_file_count": inventory_summary["file_count"],
        "artifact_inventory_root_sha256": inventory_summary["root_sha256"],
        "named_entity_terminal_decisions": 87 * 6,
        "formal_claims_permitted": False,
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
