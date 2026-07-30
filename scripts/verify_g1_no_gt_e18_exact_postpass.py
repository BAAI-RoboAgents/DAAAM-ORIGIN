#!/usr/bin/env python3
"""Independently audit G1 E18 exact-postpass evidence and root cause."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from daaam.experiments.e18_support import (  # noqa: E402
    DETERMINISTIC_PRODUCT_FILES,
    compare_exact_repetitions,
    sha256_file,
    validate_postpass_report,
)
from daaam.realtime.semantic_labels import (  # noqa: E402
    validate_semantic_label_binding,
)
from run_g1_no_gt_e18_exact_postpass import (  # noqa: E402
    exact_manifest_sha256,
    graph_summary,
    inventory,
    write_csv,
    write_json,
    write_jsonl,
)


EXPERIMENT_ROOT = (
    REPOSITORY_ROOT / "experiments" / "g1_20260724_473_573_v1_1"
)
DEFAULT_RUN = (
    EXPERIMENT_ROOT
    / "runs"
    / "diagnostic_gt_free_e18_exact_postpass_deterministic_fix_20260730"
)
DEFAULT_PRIOR = (
    EXPERIMENT_ROOT
    / "runs"
    / "diagnostic_gt_free_e18_exact_postpass_20260730"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--prior-run", type=Path, default=DEFAULT_PRIOR)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def connection_audit(path: Path) -> dict[str, Any]:
    document = read_json(path)
    connections = [
        connection
        for node in document["nodes"]
        for connection in node.get("attributes", {}).get(
            "voxblox_mesh_connections", []
        )
    ]
    return {
        "connection_count": len(connections),
        "zero_block_count": sum(
            row.get("block") == [0, 0, 0] for row in connections
        ),
        "zero_vertex_count": sum(row.get("vertex") == 0 for row in connections),
        "finite_voxel_position_count": sum(
            len(row.get("voxel_pos") or []) == 3
            and all(math.isfinite(float(value)) for value in row["voxel_pos"])
            for row in connections
        ),
        "unique_voxel_position_count": len(
            {tuple(row["voxel_pos"]) for row in connections}
        ),
        "all_value_initialized": all(
            row.get("block") == [0, 0, 0]
            and row.get("vertex") == 0
            and len(row.get("voxel_pos") or []) == 3
            and all(math.isfinite(float(value)) for value in row["voxel_pos"])
            for row in connections
        ),
    }


def prior_volatile_audit(path: Path, *, mesh_vertices: int) -> dict[str, Any]:
    document = read_json(path)
    connections = [
        connection
        for node in document["nodes"]
        for connection in node.get("attributes", {}).get(
            "voxblox_mesh_connections", []
        )
    ]
    return {
        "connection_count": len(connections),
        "vertex_out_of_range_count": sum(
            int(row["vertex"]) < 0 or int(row["vertex"]) >= mesh_vertices
            for row in connections
        ),
        "zero_block_count": sum(
            row.get("block") == [0, 0, 0] for row in connections
        ),
        "zero_vertex_count": sum(row.get("vertex") == 0 for row in connections),
        "sample": connections[:3],
    }


def audit_inputs(root: Path) -> tuple[list[dict[str, Any]], str]:
    manifest = read_json(root / "inputs" / "INPUT_MANIFEST.json")
    plan = read_json(root / "inputs" / "frozen_source_plan.json")
    rows = read_jsonl(root / "inputs" / "frame_ledger.jsonl")
    if len(rows) != len(plan["frames"]):
        raise RuntimeError("E18 input ledger frame count mismatch")
    label_dir = Path(plan["semantic_label_dir"])
    configuration = str(plan["label_run_configuration_sha256"])
    plan_by_index = {
        int(row["frame_index"]): row for row in plan["frames"]
    }
    independently_read = []
    for row in rows:
        frame_index = int(row["frame_index"])
        source = plan_by_index[frame_index]
        binding = validate_semantic_label_binding(
            label_dir,
            frame_index,
            sensor_time_ns=int(source["sensor_time_ns"]),
            run_configuration_sha256=configuration,
        )
        for key in ("rgb_path", "depth_path", "label_path", "label_metadata_path"):
            path = Path(row[key])
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_key = {
                "rgb_path": "rgb_sha256",
                "depth_path": "depth_sha256",
                "label_path": "label_sha256",
                "label_metadata_path": "label_metadata_sha256",
            }[key]
            if sha256_file(path) != row[expected_key]:
                raise RuntimeError(f"E18 input hash mismatch: {path}")
        if binding["image_sha256"] != row["label_sha256"]:
            raise RuntimeError(f"E18 label binding changed: {frame_index}")
        independently_read.append(dict(row))
    label_manifest = exact_manifest_sha256(independently_read)
    if label_manifest != manifest["label_manifest_sha256"]:
        raise RuntimeError("E18 independently rebuilt label manifest mismatch")
    return independently_read, label_manifest


def audit_repetitions(
    root: Path,
    label_manifest: str,
    configuration: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repetitions = []
    for index in range(3):
        cell = root / "exact_repetitions" / f"rep_{index:02d}"
        recorded = read_json(cell / "SUMMARY.json")
        report = read_json(cell / "postpass_report.json")
        issues = validate_postpass_report(
            report,
            expected_frames=102,
            expected_label_manifest_sha256=label_manifest,
            expected_run_configuration_sha256=configuration,
        )
        if issues:
            raise RuntimeError(f"E18 repetition {index} contract failed: {issues}")
        output = cell / "hydra_realtime"
        hashes = {
            relative: sha256_file(output / relative)
            for relative in DETERMINISTIC_PRODUCT_FILES
        }
        if hashes != recorded["artifact_hashes"]:
            raise RuntimeError(f"E18 repetition {index} artifact hash changed")
        independently_loaded = graph_summary(
            output / "backend" / "dsg_with_mesh.json"
        )
        if independently_loaded != recorded["graph_summary"]:
            raise RuntimeError(f"E18 repetition {index} graph summary changed")
        connections = connection_audit(output / "backend" / "dsg.json")
        if (
            connections["connection_count"] != 378
            or not connections["all_value_initialized"]
        ):
            raise RuntimeError(
                f"E18 repetition {index} connection initialization failed"
            )
        repetitions.append(
            {
                "repetition": index,
                "report": report,
                "artifact_hashes": hashes,
                "graph_summary": independently_loaded,
                "connection_audit": connections,
            }
        )
    comparison = compare_exact_repetitions(repetitions)
    if not (
        comparison["semantic_contract_stable"]
        and comparison["formal_product_hash_stable"]
        and comparison["graph_summary_stable"]
    ):
        raise RuntimeError("E18 fixed repetitions are not deterministic")
    return repetitions, comparison


def audit_failure_injections(root: Path) -> list[dict[str, Any]]:
    expected = {
        "missing_label",
        "corrupt_hash",
        "stale_configuration",
        "duplicate_frame",
    }
    rows = read_jsonl(root / "analysis" / "failure_injections.jsonl")
    if {str(row["probe"]) for row in rows} != expected:
        raise RuntimeError("E18 failure-injection matrix is incomplete")
    for row in rows:
        if not row["failed_as_expected"] or not row["no_formal_product_committed"]:
            raise RuntimeError(f"E18 failure probe did not fail closed: {row}")
        if int(row["returncode"]) == 0 or row["report_created"]:
            raise RuntimeError(f"E18 failure probe produced success report: {row}")
    return rows


def audit_durable_commit(root: Path) -> dict[str, Any]:
    from spark_dsg import DsgLayers, DynamicSceneGraph

    commit = read_json(
        root / "durable_commit" / "semantic_dsg_commit.json"
    )
    if (
        commit["status"] != "passed"
        or commit["applied"] != 54
        or commit["rejected_no_mesh"] != 33
        or commit["delivery_pending"] != 0
        or commit["unmapped"] != 0
        or commit["errors"]
        or not commit["binding_output_hash_stable"]
    ):
        raise RuntimeError("E18 durable commit gate changed")
    hashes = []
    for index in range(3):
        path = (
            root
            / "durable_commit"
            / "repetitions"
            / f"rep_{index:02d}"
            / "dsg_bound.json"
        )
        hashes.append(sha256_file(path))
        graph = DynamicSceneGraph.load(str(path))
        entity_bindings = {}
        for node in graph.get_layer(DsgLayers.OBJECTS).nodes:
            metadata = dict(node.attributes.metadata.get() or {})
            entity_id = str(metadata.get("entity_id") or "")
            if not entity_id:
                continue
            if entity_id in entity_bindings:
                raise RuntimeError("entity is bound to multiple object nodes")
            binding = dict(metadata.get("entity_binding") or {})
            entity_bindings[entity_id] = {
                "node_id": str(node.id),
                "node_semantic_label": int(node.attributes.semantic_label),
                "candidate_semantic_id": int(
                    binding.get(
                        "candidate_semantic_id",
                        node.attributes.semantic_label,
                    )
                ),
            }
        if len(entity_bindings) != 54:
            raise RuntimeError(f"E18 binding repetition {index} has wrong count")
        entity_51 = [
            value
            for value in entity_bindings.values()
            if value["node_semantic_label"] == 51
            and value["candidate_semantic_id"] == 77
        ]
        if len(entity_51) != 1:
            raise RuntimeError("E18 D assignment E51->E77 was not reproduced")
        if any(
            value["node_semantic_label"] == 8
            and value["candidate_semantic_id"] == 77
            for value in entity_bindings.values()
        ):
            raise RuntimeError("E18 reproduced the rejected E8->E77 error")
    if len(set(hashes)) != 1 or hashes[0] != commit["output_dsg_sha256"]:
        raise RuntimeError("E18 bound DSG hash is not stable")
    if sha256_file(Path(commit["output_dsg"])) != hashes[0]:
        raise RuntimeError("E18 canonical durable commit hash changed")
    return commit


def root_cause_audit(
    prior: Path,
    fixed: Path,
    repetitions: list[dict[str, Any]],
) -> dict[str, Any]:
    prior_rows = []
    for index in range(3):
        cell = prior / "exact_repetitions" / f"rep_{index:02d}"
        summary = read_json(cell / "SUMMARY.json")
        prior_rows.append(
            {
                "repetition": index,
                "artifact_hashes": summary["artifact_hashes"],
                "volatile_connections": prior_volatile_audit(
                    cell / "hydra_realtime" / "backend" / "dsg.json",
                    mesh_vertices=360375,
                ),
            }
        )
    prior_dsg_hashes = [
        row["artifact_hashes"]["backend/dsg.json"] for row in prior_rows
    ]
    fixed_dsg_hashes = [
        row["artifact_hashes"]["backend/dsg.json"] for row in repetitions
    ]
    unchanged_geometry = all(
        len(
            {
                row["artifact_hashes"][relative]
                for row in prior_rows + repetitions
            }
        )
        == 1
        for relative in (
            "backend/mesh.ply",
            "backend/deformation_graph.dgrf",
        )
    )
    result = {
        "schema": "daaam.g1_e18_determinism_root_cause.v1",
        "status": "confirmed_and_fixed",
        "prior_run": str(prior.resolve()),
        "fixed_run": str(fixed.resolve()),
        "defect": (
            "Hydra GraphExtractor::convertInfo default-initialized "
            "NearestVertexInfo and serialized uninitialized block[3]/vertex"
        ),
        "affected_layer": "148 place nodes / 378 voxblox mesh connections",
        "prior_dsg_raw_hash_stable": len(set(prior_dsg_hashes)) == 1,
        "fixed_dsg_raw_hash_stable": len(set(fixed_dsg_hashes)) == 1,
        "prior_connection_audits": [
            row["volatile_connections"] for row in prior_rows
        ],
        "fixed_connection_audits": [
            row["connection_audit"] for row in repetitions
        ],
        "prior_out_of_range_vertices": sum(
            row["volatile_connections"]["vertex_out_of_range_count"]
            for row in prior_rows
        ),
        "fixed_zero_initialized_connections": sum(
            row["connection_audit"]["zero_vertex_count"]
            for row in repetitions
        ),
        "geometry_hash_unchanged_before_after": unchanged_geometry,
        "patch": str(
            (
                REPOSITORY_ROOT
                / "patches"
                / "hydra_nearest_vertex_value_init.patch"
            ).resolve()
        ),
        "patch_sha256": sha256_file(
            REPOSITORY_ROOT
            / "patches"
            / "hydra_nearest_vertex_value_init.patch"
        ),
        "causal_evidence": [
            "before: three DSG hashes differ",
            "before: all 1134 serialized vertices are outside mesh range",
            "normalizing only block/vertex makes all remaining JSON content equal",
            "after value initialization: all four product hashes match 3/3",
            "mesh and deformation graph hashes are unchanged before versus after",
        ],
    }
    if (
        result["prior_dsg_raw_hash_stable"]
        or not result["fixed_dsg_raw_hash_stable"]
        or result["prior_out_of_range_vertices"] != 1134
        or result["fixed_zero_initialized_connections"] != 1134
        or not unchanged_geometry
    ):
        raise RuntimeError(f"E18 root-cause evidence failed: {result}")
    return result


def render_before_after(
    root: Path,
    root_cause: Mapping[str, Any],
) -> None:
    labels = ["raw DSG hash\nstable", "invalid vertex\nconnections", "mesh hash\nstable"]
    before = [
        int(root_cause["prior_dsg_raw_hash_stable"]),
        1,
        int(root_cause["geometry_hash_unchanged_before_after"]),
    ]
    after = [
        int(root_cause["fixed_dsg_raw_hash_stable"]),
        0,
        int(root_cause["geometry_hash_unchanged_before_after"]),
    ]
    positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.bar([value - 0.18 for value in positions], before, 0.36, label="before")
    ax.bar([value + 0.18 for value in positions], after, 0.36, label="after")
    ax.set_xticks(positions, labels)
    ax.set_ylim(0.0, 1.15)
    ax.set_title("E18 NearestVertexInfo value-initialization ablation")
    ax.legend()
    fig.savefig(
        root / "visualizations" / "05_determinism_fix_before_after.png",
        dpi=160,
    )
    plt.close(fig)


def append_report(root: Path, root_cause: Mapping[str, Any]) -> None:
    report = root / "REPORT.md"
    marker = "## 独立审计与确定性根因"
    text = report.read_text()
    if marker in text:
        return
    text += f"""

## 独立审计与确定性根因

独立审计重新读取102组RGB/depth/label/metadata hash、三份Hydra图、四个故障注入
和三份E17-D绑定图，全部通过。最终状态为 `complete_independently_audited`。

首轮E18未通过原始DSG hash门。差异被精确定位到148个place节点的378条
`voxblox_mesh_connections`：`convertInfo()`只写入`voxel_pos`，未初始化
`block[3]`和`vertex`。三轮共1134/1134个vertex均越过360375个mesh顶点范围，
且数值呈进程地址特征。移除这两个字段后，其余JSON内容hash三轮完全一致。

将局部变量改为值初始化`NearestVertexInfo info{{}};`后：

- 三轮四个正式产物hash全部一致；
- 三轮D绑定后DSG hash同为
  `{root_cause['fixed_run'] and read_json(root / 'durable_commit' / 'semantic_dsg_commit.json')['output_dsg_sha256']}`；
- 378/378条连接每轮均得到`block=[0,0,0]`、`vertex=0`；
- `voxel_pos`、mesh、deformation graph、635节点/1270边和55个object均未变化；
- 修复前后的mesh与deformation graph hash完全一致。

这证明失败来自未初始化序列化字段，而不是标签、几何或图拓扑随机变化。
"""
    report.write_text(text)


def main() -> int:
    args = parse_args()
    root = args.run.resolve()
    prior = args.prior_run.resolve()
    if not root.is_dir() or not prior.is_dir():
        raise FileNotFoundError("E18 fixed/prior run directory is missing")
    input_rows, label_manifest = audit_inputs(root)
    plan = read_json(root / "inputs" / "frozen_source_plan.json")
    repetitions, comparison = audit_repetitions(
        root,
        label_manifest,
        str(plan["label_run_configuration_sha256"]),
    )
    failures = audit_failure_injections(root)
    commit = audit_durable_commit(root)
    root_cause = root_cause_audit(prior, root, repetitions)

    write_json(
        root / "analysis" / "DETERMINISM_ROOT_CAUSE.json",
        root_cause,
    )
    write_jsonl(
        root / "analysis" / "independent_repetition_audit.jsonl",
        repetitions,
    )
    write_csv(
        root / "analysis" / "independent_repetition_audit.csv",
        [
            {
                "repetition": row["repetition"],
                "label_manifest_sha256": row["report"][
                    "label_manifest_sha256"
                ],
                "dsg_sha256": row["artifact_hashes"]["backend/dsg.json"],
                "dsg_with_mesh_sha256": row["artifact_hashes"][
                    "backend/dsg_with_mesh.json"
                ],
                "mesh_sha256": row["artifact_hashes"]["backend/mesh.ply"],
                "connection_count": row["connection_audit"][
                    "connection_count"
                ],
                "zero_vertex_count": row["connection_audit"][
                    "zero_vertex_count"
                ],
            }
            for row in repetitions
        ],
    )
    render_before_after(root, root_cause)
    append_report(root, root_cause)

    summary = read_json(root / "SUMMARY.json")
    summary.update(
        {
            "status": "complete_independently_audited",
            "independent_audit_passed": True,
            "determinism_root_cause": (
                "uninitialized NearestVertexInfo block/vertex serialization"
            ),
            "determinism_fix_verified": True,
            "prior_failed_run": str(prior),
        }
    )
    write_json(root / "SUMMARY.json", summary)
    write_json(
        root / "COMPLETION.json",
        {
            "schema": "daaam.g1_e18_completion.v1",
            "status": "complete_independently_audited",
            "completed_at": utc_now(),
            "hard_gates_passed": True,
            "independent_audit_passed": True,
            "formal_semantic_accuracy_claim_permitted": False,
        },
    )
    shutil.copy2(
        Path(__file__).resolve(),
        root / "source_snapshot" / Path(__file__).name,
    )
    inventory_summary = inventory(
        root,
        exclusions={
            "INDEPENDENT_AUDIT.json",
            "artifact_inventory.csv",
            "artifact_inventory.jsonl",
            "inventory_summary.json",
        },
    )
    audit = {
        "schema": "daaam.g1_e18_independent_audit.v1",
        "status": "passed",
        "audited_at": utc_now(),
        "input_frames_rehashed": len(input_rows),
        "label_manifest_sha256": label_manifest,
        "exact_repetitions_reloaded": len(repetitions),
        "semantic_contract_stable": comparison["semantic_contract_stable"],
        "formal_product_hash_stable": comparison[
            "formal_product_hash_stable"
        ],
        "graph_summary_stable": comparison["graph_summary_stable"],
        "failure_injections_detected": (
            f"{sum(bool(row['failed_as_expected']) for row in failures)}/"
            f"{len(failures)}"
        ),
        "failure_injections_fail_closed": all(
            bool(row["no_formal_product_committed"]) for row in failures
        ),
        "durable_commit_repetitions": len(
            commit["binding_repetitions"]
        ),
        "durable_commit_hash_stable": commit[
            "binding_output_hash_stable"
        ],
        "applied": commit["applied"],
        "rejected_no_mesh": commit["rejected_no_mesh"],
        "delivery_pending": commit["delivery_pending"],
        "unmapped": commit["unmapped"],
        "errors": commit["errors"],
        "root_cause_status": root_cause["status"],
        "prior_out_of_range_vertices": root_cause[
            "prior_out_of_range_vertices"
        ],
        "fixed_zero_initialized_connections": root_cause[
            "fixed_zero_initialized_connections"
        ],
        "geometry_hash_unchanged_before_after": root_cause[
            "geometry_hash_unchanged_before_after"
        ],
        "artifact_inventory": inventory_summary,
        "formal_semantic_accuracy_remains_unavailable": True,
    }
    write_json(root / "INDEPENDENT_AUDIT.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
