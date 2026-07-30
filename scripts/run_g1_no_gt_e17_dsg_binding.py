#!/usr/bin/env python3
"""Run the GT-free G1 E17 entity-to-Hydra-mesh binding experiment."""

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
from types import SimpleNamespace
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from daaam.experiments.e17_support import (  # noqa: E402
    E17_THRESHOLDS,
    BindingThreshold,
    build_candidate_rows,
    perturb_matched_candidate,
    summarize_decisions,
    terminal_decisions,
)
from rebind_dsg_semantics import migrate  # noqa: E402


EXPERIMENT_ROOT = (
    REPOSITORY_ROOT
    / "experiments"
    / "g1_20260724_473_573_v1_1"
)
DEFAULT_E14 = (
    EXPERIMENT_ROOT
    / "runs"
    / "diagnostic_gt_free_e14_safe035_unique_obs8_seed0_20260730"
)
DEFAULT_E16_SWEEP = (
    EXPERIMENT_ROOT
    / "runs"
    / "diagnostic_gt_free_e16_12cm_obs_range_sweep_20260730"
)
DEFAULT_E16_COMPARISON = (
    EXPERIMENT_ROOT
    / "comparisons"
    / "e16_obs_range_adaptive_validation_20260730"
)
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT
    / "runs"
    / "diagnostic_gt_free_e17_e14e16fed_binding_20260730"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e14-run", type=Path, default=DEFAULT_E14)
    parser.add_argument("--e16-sweep", type=Path, default=DEFAULT_E16_SWEEP)
    parser.add_argument(
        "--e16-comparison",
        type=Path,
        default=DEFAULT_E16_COMPARISON,
    )
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key))
                seen.add(key)
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


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required E17 inputs: {missing}")


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def load_entities(
    final_labels_path: Path,
    final_entities_path: Path,
) -> list[dict[str, Any]]:
    labels = read_jsonl(final_labels_path)
    snapshots = {
        str(row["entity_id"]): row
        for row in json.loads(final_entities_path.read_text())
    }
    records = []
    ordinals = set()
    for label in labels:
        entity_id = str(label["entity_id"])
        if entity_id not in snapshots:
            raise ValueError(f"E14 label has no entity snapshot: {entity_id}")
        semantic_id = int(label["entity_ordinal"])
        if semantic_id <= 0 or semantic_id in ordinals:
            raise ValueError(f"invalid or duplicate entity ordinal: {semantic_id}")
        ordinals.add(semantic_id)
        snapshot = snapshots[entity_id]
        position = snapshot.get("position_m")
        dimensions = snapshot.get("dimensions_m")
        if position is None or dimensions is None:
            raise ValueError(f"named entity has no valid geometry: {entity_id}")
        records.append(
            {
                "entity_id": entity_id,
                "semantic_id": semantic_id,
                "description": str(label["final_label"]).strip(),
                "position_m": list(map(float, position)),
                "dimensions_m": list(map(float, dimensions)),
                "temporal_history": snapshot.get("temporal_history") or {},
            }
        )
    return sorted(records, key=lambda row: int(row["semantic_id"]))


def build_semantic_source(
    path: Path,
    entities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from spark_dsg import (
        BoundingBoxType,
        DsgLayers,
        DynamicSceneGraph,
        KhronosObjectAttributes,
        NodeSymbol,
    )

    graph = DynamicSceneGraph()
    for entity in entities:
        semantic_id = int(entity["semantic_id"])
        attributes = KhronosObjectAttributes()
        attributes.semantic_label = semantic_id
        attributes.position = entity["position_m"]
        attributes.bounding_box.type = BoundingBoxType.AABB
        attributes.bounding_box.world_P_center = entity["position_m"]
        attributes.bounding_box.dimensions = entity["dimensions_m"]
        attributes.metadata.set(
            {
                "entity_id": entity["entity_id"],
                "description": entity["description"],
                "semantic_source": "E14_obs8_seed0_final_labels",
            }
        )
        if not graph.add_node(
            DsgLayers.OBJECTS,
            NodeSymbol("O", semantic_id),
            attributes,
        ):
            raise RuntimeError(f"failed to add semantic source entity {semantic_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.save(str(path), include_mesh=False)
    reloaded = DynamicSceneGraph.load(str(path))
    count = len(list(reloaded.get_layer(DsgLayers.OBJECTS).nodes))
    if count != len(entities):
        raise RuntimeError("semantic-source DSG entity count changed on reload")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "entity_count": count,
        "contains_real_object_mesh": False,
        "purpose": "frozen E14 identity/description source; not E17 geometry",
    }


def real_mesh_nodes(path: Path) -> list[dict[str, Any]]:
    from spark_dsg import DsgLayers, DynamicSceneGraph

    graph = DynamicSceneGraph.load(str(path))
    result = []
    for node in graph.get_layer(DsgLayers.OBJECTS).nodes:
        mesh = node.attributes.mesh()
        vertices = 0 if mesh is None else int(mesh.num_vertices())
        if vertices <= 0:
            continue
        dimensions = np.asarray(
            node.attributes.bounding_box.dimensions,
            dtype=np.float64,
        )
        position = np.asarray(node.attributes.position, dtype=np.float64)
        if (
            dimensions.shape != (3,)
            or position.shape != (3,)
            or np.any(dimensions <= 0.0)
            or not np.all(np.isfinite(dimensions))
            or not np.all(np.isfinite(position))
        ):
            raise ValueError(f"real mesh node has invalid geometry: {node.id}")
        result.append(
            {
                "node_id": str(node.id),
                "node_id_value": int(node.id.value),
                "semantic_id": int(node.attributes.semantic_label),
                "position_m": position.tolist(),
                "dimensions_m": dimensions.tolist(),
                "mesh_vertices": vertices,
                "mesh_faces": int(mesh.num_faces()),
            }
        )
    if not result:
        raise ValueError(f"geometry DSG contains no real object mesh: {path}")
    return sorted(
        result,
        key=lambda row: (int(row["semantic_id"]), int(row["node_id_value"])),
    )


def make_preregistration(
    *,
    inputs: Mapping[str, Path],
    output: Path,
) -> dict[str, Any]:
    return {
        "schema": "daaam.g1_no_gt_e17_preregistration.v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "stage": "E17",
        "status": "exploratory_diagnostic_no_human_gt",
        "hypothesis": (
            "strict/medium/wide spatial gates expose a precision-coverage trade-off; "
            "widening must not be selected solely from higher match count."
        ),
        "controlled_input": {
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in inputs.items()
        },
        "geometry_arms": {
            "single_pass": (
                "12cm obs6 range8m; production-like one-pass E16 recommendation"
            ),
            "adaptive": (
                "derived obs6 near5/far8 graph; diagnostic upper-bound candidate"
            ),
        },
        "single_changed_factor": (
            "within each frozen geometry arm: object-binding center/AABB thresholds"
        ),
        "variants": [
            {
                "name": threshold.name,
                "maximum_center_distance_m": (
                    threshold.maximum_center_distance_m
                ),
                "maximum_aabb_gap_m": threshold.maximum_aabb_gap_m,
                "acceptance": "center_distance OR AABB_gap",
            }
            for threshold in E17_THRESHOLDS
        ],
        "expected_improvement": (
            "medium may recover credible same-object offsets rejected by strict."
        ),
        "expected_failure_signature": (
            "wide increases spatial fallback, entity conflict, or attachment to "
            "neighboring/fragment meshes."
        ),
        "primary_metric": (
            "matched/rejected counts plus semantic-ID consistency and reviewed "
            "wrong-mesh risk; formal binding P/R/F1 unavailable"
        ),
        "guardrail_metrics": [
            "one terminal decision per named entity",
            "one entity per mesh node",
            "source/output mesh counts invariant",
            "all rejected-no-mesh events auditable",
            "spatial fallback never reported as proven correct",
            "E16 reviewed physical-object quality retained as upstream guardrail",
        ],
        "challenge_tags": [
            "large_structural_AABB",
            "adjacent_chairs",
            "machine_part_oversegmentation",
            "far_range_objects",
            "missing_mesh",
        ],
        "hard_gate": (
            "87/87 named entities have exactly one terminal event; no duplicate "
            "mesh ownership; all six outputs reload; inventory verifies."
        ),
        "upstream_oracle": (
            "frozen E14 obs8 entities and frozen E16 graphs; not GT entity/mesh, "
            "therefore formal binding precision/recall/F1 are prohibited"
        ),
        "visualizations": [
            "entity↔mesh top-down lines",
            "binding outcome bars",
            "semantic-ID consistency",
            "entity threshold transition map",
            "candidate gate distributions",
            "0.1/0.3/0.6m chosen-pair perturbation response",
        ],
        "failure_taxonomy_codes": [
            "F-BIND-rejected-no-mesh",
            "F-BIND-spatial-fallback",
            "F-BIND-entity-conflict",
            "F-BIND-reserved-owner",
            "F-BIND-upstream-wrong-or-fragment-mesh",
        ],
        "existing_d0_qualification": {
            "path": str(
                (
                    EXPERIMENT_ROOT
                    / "runs"
                    / "diagnostic_gt_free_d0_contract_probes_20260728"
                    / "07_entity_position_offset"
                    / "aggregate.json"
                ).resolve()
            ),
            "result": (
                "E17/Q1 collector detected 0.3m and 0.6m injections at 100%; "
                "wrong-mesh correctness remained unqualified"
            ),
        },
        "output": str(output.resolve()),
    }


def copy_source_preview(source_root: Path, target: Path) -> None:
    preview = source_root / "hydra_map_preview.png"
    if preview.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preview, target)


def topdown_plot(
    path: Path,
    *,
    geometry_source: str,
    threshold: BindingThreshold,
    nodes: Sequence[Mapping[str, Any]],
    entities: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> None:
    entity_lookup = {str(row["entity_id"]): row for row in entities}
    node_lookup = {str(row["node_id"]): row for row in nodes}
    fig, axis = plt.subplots(figsize=(12, 9), constrained_layout=True)
    node_xy = np.asarray([row["position_m"][:2] for row in nodes])
    axis.scatter(
        node_xy[:, 0],
        node_xy[:, 1],
        s=24,
        c="#1f77b4",
        alpha=0.55,
        label=f"real mesh nodes ({len(nodes)})",
    )
    for row in nodes:
        axis.text(
            row["position_m"][0],
            row["position_m"][1],
            f"M{row['semantic_id']}",
            fontsize=5,
            color="#174f78",
        )
    for decision in decisions:
        entity = entity_lookup[str(decision["entity_id"])]
        start = np.asarray(entity["position_m"][:2], dtype=np.float64)
        if decision["status"] == "matched_real_mesh":
            node = node_lookup[str(decision["node_id"])]
            end = np.asarray(node["position_m"][:2], dtype=np.float64)
            color = "#2ca02c" if decision["semantic_id_match"] else "#ff7f0e"
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                alpha=0.65,
                linewidth=0.9,
            )
        else:
            axis.scatter(
                [start[0]],
                [start[1]],
                marker="x",
                s=18,
                color="#d62728",
                alpha=0.7,
            )
    axis.scatter([], [], c="#2ca02c", label="semantic-ID consistent binding")
    axis.scatter([], [], c="#ff7f0e", label="spatial fallback (review required)")
    axis.scatter([], [], marker="x", c="#d62728", label="rejected no mesh")
    axis.set_title(
        f"E17 {geometry_source} / {threshold.name}: entity ↔ real-mesh binding"
    )
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def offset_rows(
    *,
    geometry_source: str,
    threshold: BindingThreshold,
    decisions: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidates = {
        (str(row["entity_id"]), str(row["node_id"])): row
        for row in candidate_rows
    }
    rows = []
    for decision in decisions:
        if decision["status"] != "matched_real_mesh":
            continue
        key = (str(decision["entity_id"]), str(decision["node_id"]))
        candidate = candidates[key]
        for dose in (0.1, 0.3, 0.6):
            result = perturb_matched_candidate(
                candidate,
                dose_m=dose,
                threshold=threshold,
            )
            rows.append(
                {
                    "schema": "daaam.g1_e17_chosen_pair_offset_probe.v1",
                    "geometry_source": geometry_source,
                    "threshold": threshold.name,
                    "entity_id": decision["entity_id"],
                    "entity_ordinal": decision["entity_ordinal"],
                    "node_id": decision["node_id"],
                    "candidate_semantic_id": decision["candidate_semantic_id"],
                    "semantic_id_match": decision["semantic_id_match"],
                    "injection_direction": (
                        "radially_away_from_chosen_candidate_center"
                    ),
                    "scope": (
                        "exact chosen-pair gate recomputation; alternative-node "
                        "rematching is not simulated"
                    ),
                    **result,
                }
            )
    return rows


def threshold_transitions(
    source: str,
    decisions_by_threshold: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    by_threshold = {
        name: {str(row["entity_id"]): row for row in rows}
        for name, rows in decisions_by_threshold.items()
    }
    entity_ids = sorted(by_threshold["strict"])
    rows = []
    for entity_id in entity_ids:
        values = {name: by_threshold[name][entity_id] for name in by_threshold}
        strict = values["strict"]
        medium = values["medium"]
        wide = values["wide"]
        rows.append(
            {
                "schema": "daaam.g1_e17_threshold_transition.v1",
                "geometry_source": source,
                "entity_id": entity_id,
                "entity_ordinal": strict["entity_ordinal"],
                "entity_label": strict["entity_label"],
                "strict_status": strict["status"],
                "strict_node_id": strict["node_id"],
                "strict_candidate_semantic_id": strict["candidate_semantic_id"],
                "medium_status": medium["status"],
                "medium_node_id": medium["node_id"],
                "medium_candidate_semantic_id": medium["candidate_semantic_id"],
                "wide_status": wide["status"],
                "wide_node_id": wide["node_id"],
                "wide_candidate_semantic_id": wide["candidate_semantic_id"],
                "gained_by_medium": (
                    strict["status"] == "rejected_no_mesh"
                    and medium["status"] == "matched_real_mesh"
                ),
                "gained_by_wide": (
                    medium["status"] == "rejected_no_mesh"
                    and wide["status"] == "matched_real_mesh"
                ),
                "reassigned_when_widened": (
                    len(
                        {
                            str(value["node_id"])
                            for value in values.values()
                            if value["node_id"] is not None
                        }
                    )
                    > 1
                ),
                "spatial_fallback_in_any_variant": any(
                    value["risk_class"] == "spatial_fallback_requires_review"
                    for value in values.values()
                ),
            }
        )
    return rows


def aggregate_visualizations(
    output: Path,
    cell_summaries: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    all_candidates: Sequence[Mapping[str, Any]],
    all_offsets: Sequence[Mapping[str, Any]],
) -> None:
    visual = output / "visualizations"
    sources = ("single_pass", "adaptive")
    thresholds = ("strict", "medium", "wide")
    lookup = {
        (str(row["geometry_source"]), str(row["threshold"])): row
        for row in cell_summaries
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(thresholds))
    width = 0.36
    for index, source in enumerate(sources):
        matched = [lookup[(source, name)]["matched_real_mesh"] for name in thresholds]
        rejected = [lookup[(source, name)]["rejected_no_mesh"] for name in thresholds]
        axes[index].bar(x - width / 2, matched, width, label="matched real mesh")
        axes[index].bar(x + width / 2, rejected, width, label="rejected no mesh")
        axes[index].set_xticks(x, thresholds)
        axes[index].set_ylim(0, max(max(matched), max(rejected)) * 1.15)
        axes[index].set_title(source)
        axes[index].set_ylabel("named E14 entities")
        axes[index].grid(axis="y", alpha=0.2)
        axes[index].legend(fontsize=8)
    fig.suptitle("E17 binding outcomes (formal P/R/F1 unavailable)")
    fig.savefig(visual / "01_binding_outcomes.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for index, source in enumerate(sources):
        exact = [
            lookup[(source, name)]["provenance_consistent_semantic_id"]
            for name in thresholds
        ]
        fallback = [
            lookup[(source, name)]["spatial_fallback_requires_review"]
            for name in thresholds
        ]
        axes[index].bar(x, exact, label="semantic-ID consistent")
        axes[index].bar(x, fallback, bottom=exact, label="spatial fallback")
        axes[index].set_xticks(x, thresholds)
        axes[index].set_title(source)
        axes[index].set_ylabel("matched entities")
        axes[index].grid(axis="y", alpha=0.2)
        axes[index].legend(fontsize=8)
    fig.suptitle("E17 match provenance—not a correctness precision estimate")
    fig.savefig(visual / "02_binding_provenance.png", dpi=180)
    plt.close(fig)

    transition_lookup = {
        (str(row["geometry_source"]), int(row["entity_ordinal"])): row
        for row in transitions
    }
    ordinals = sorted({key[1] for key in transition_lookup})
    matrix = np.full((len(ordinals), 6), np.nan)
    for row_index, ordinal in enumerate(ordinals):
        for source_index, source in enumerate(sources):
            row = transition_lookup[(source, ordinal)]
            for threshold_index, threshold in enumerate(thresholds):
                status = row[f"{threshold}_status"]
                candidate = row[f"{threshold}_candidate_semantic_id"]
                if status == "rejected_no_mesh":
                    value = 0
                elif candidate == ordinal:
                    value = 2
                else:
                    value = 1
                matrix[row_index, source_index * 3 + threshold_index] = value
    fig, axis = plt.subplots(
        figsize=(12, max(8, len(ordinals) * 0.14)),
        constrained_layout=True,
    )
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=2)
    axis.set_xticks(
        range(6),
        [
            "single\nstrict",
            "single\nmedium",
            "single\nwide",
            "adaptive\nstrict",
            "adaptive\nmedium",
            "adaptive\nwide",
        ],
    )
    axis.set_yticks(range(len(ordinals)), [f"E{x:03d}" for x in ordinals], fontsize=5)
    axis.set_title("Per-entity E17 outcome: 0 reject, 1 spatial fallback, 2 ID-consistent")
    colorbar = fig.colorbar(image, ax=axis, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(["reject", "spatial", "ID-consistent"])
    fig.savefig(visual / "03_entity_threshold_matrix.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for source, color in (("single_pass", "#1f77b4"), ("adaptive", "#ff7f0e")):
        rows = [
            row
            for row in all_candidates
            if row["geometry_source"] == source and row["threshold"] == "wide"
        ]
        exact = [row for row in rows if row["semantic_id_match"]]
        other = [row for row in rows if not row["semantic_id_match"]]
        axes[0].hist(
            [row["center_distance_m"] for row in exact],
            bins=30,
            alpha=0.45,
            color=color,
            label=f"{source} exact-ID",
        )
        axes[1].hist(
            [min(float(row["aabb_gap_m"]), 2.0) for row in other],
            bins=30,
            alpha=0.45,
            color=color,
            label=f"{source} nonmatching-ID",
        )
    axes[0].set_xlabel("center distance (m)")
    axes[0].set_title("Exact semantic-ID candidate center distance")
    axes[1].set_xlabel("AABB gap clipped at 2m")
    axes[1].set_title("Nonmatching semantic-ID candidate AABB gap")
    for axis in axes:
        axis.set_ylabel("candidate pairs")
        axis.set_yscale("symlog")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.savefig(visual / "04_candidate_gate_distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for index, source in enumerate(sources):
        for threshold in thresholds:
            values = []
            for dose in (0.1, 0.3, 0.6):
                rows = [
                    row
                    for row in all_offsets
                    if row["geometry_source"] == source
                    and row["threshold"] == threshold
                    and float(row["dose_m"]) == dose
                ]
                values.append(
                    float(
                        np.mean(
                            [
                                bool(row["chosen_pair_rejected_due_to_injection"])
                                for row in rows
                            ]
                        )
                    )
                    if rows
                    else 0.0
                )
            axes[index].plot((0.1, 0.3, 0.6), values, marker="o", label=threshold)
        axes[index].set_title(source)
        axes[index].set_xlabel("radial-away entity offset dose (m)")
        axes[index].set_ylabel("chosen-pair rejection fraction")
        axes[index].set_ylim(-0.03, 1.03)
        axes[index].grid(alpha=0.2)
        axes[index].legend()
    fig.suptitle(
        "E17 exact chosen-pair perturbation response (no alternative rematching)"
    )
    fig.savefig(visual / "05_entity_offset_response.png", dpi=180)
    plt.close(fig)


def inventory(output: Path) -> dict[str, Any]:
    excluded = {
        "artifact_inventory.csv",
        "artifact_inventory.jsonl",
        "inventory_summary.json",
        "COMPLETION.json",
        "INDEPENDENT_AUDIT.json",
        "terminal_failure.json",
    }
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
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
    write_jsonl(output / "artifact_inventory.jsonl", rows)
    write_csv(output / "artifact_inventory.csv", rows)
    summary = {
        "schema": "daaam.g1_e17_artifact_inventory.v1",
        "file_count": len(rows),
        "total_size_bytes": int(sum(row["size_bytes"] for row in rows)),
        "root_sha256": root.hexdigest(),
        "excluded_relative_paths": sorted(excluded),
    }
    write_json(output / "inventory_summary.json", summary)
    return summary


def build_report(
    *,
    cell_summaries: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    upstream_summary: Mapping[str, Any],
    output: Path,
) -> str:
    lookup = {
        (str(row["geometry_source"]), str(row["threshold"])): row
        for row in cell_summaries
    }
    lines = [
        "# G1 E17：E14 实体 ↔ E16 Hydra real-mesh 绑定",
        "",
        "## 结论先行",
        "",
        (
            "本实验已完成 single-pass 与 adaptive 两个冻结 E16 图上的 "
            "strict / medium / wide 六个绑定单元。所有数字均为无人工 GT 的工程"
            "计数或 provenance proxy；**binding precision / recall / F1 仍不可正式计算**。"
        ),
        "",
        "| E16 几何 | 门限 | matched real mesh | rejected no mesh | ID 一致 | 空间兜底（需复核） | 命名实体覆盖 proxy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in ("single_pass", "adaptive"):
        for threshold in ("strict", "medium", "wide"):
            row = lookup[(source, threshold)]
            lines.append(
                f"| {source} | {threshold} | {row['matched_real_mesh']} | "
                f"{row['rejected_no_mesh']} | "
                f"{row['provenance_consistent_semantic_id']} | "
                f"{row['spatial_fallback_requires_review']} | "
                f"{100.0 * row['named_entity_binding_coverage_proxy']:.1f}% |"
            )
    gained_medium = sum(bool(row["gained_by_medium"]) for row in transitions)
    gained_wide = sum(bool(row["gained_by_wide"]) for row in transitions)
    reassigned = sum(bool(row["reassigned_when_widened"]) for row in transitions)
    spatial_any = sum(bool(row["spatial_fallback_in_any_variant"]) for row in transitions)
    lines.extend(
        [
            "",
            "## 门限含义与本次结果",
            "",
            (
                "生产实现采用 `center_distance <= limit OR AABB_gap <= limit`。"
                "因此，即使实体中心相距超过 1m，只要两个 AABB 重叠（gap=0），"
                "strict 仍会放行。大桌、地面、墙面或被污染的大框会削弱 strict 的"
                "实际约束力；这不是等价于“中心在 10cm 内”。"
            ),
            "",
            f"- strict→medium 新增绑定：{gained_medium}",
            f"- medium→wide 新增绑定：{gained_wide}",
            f"- 放宽后改绑到不同 node：{reassigned}",
            f"- 任一门限出现空间兜底的实体：{spatial_any}",
            "",
            (
                "门限选择不能按 matched 数量单独决定。`semantic-ID consistent` "
                "只证明 E13 entity ordinal 与 E16 label provenance 对齐，不证明 mask、"
                "mesh 或 DAM 名称在现实语义上正确；空间兜底的正确性更是未标定。"
            ),
            "",
            "## 上游 E16 守恒风险",
            "",
        ]
    )
    estimates = {
        str(row["variant"]): row
        for row in upstream_summary.get("variant_codex_estimates", [])
    }
    single = estimates.get("12cm_obs6_range8m", {})
    adaptive = estimates.get("adaptive_obs6_near5_far8", {})
    lines.extend(
        [
            (
                "- single-pass obs6/range8 的 Codex 近似物理目标评估为 "
                f"{single.get('strict')}/{single.get('partial')}/"
                f"{single.get('failure')}（严格/部分/失败，共19）。"
            ),
            (
                "- adaptive 为 "
                f"{adaptive.get('strict')}/{adaptive.get('partial')}/"
                f"{adaptive.get('failure')}；但需双路顺序运行约 274.61s，"
                "不是单路生产配置。"
            ),
            (
                "- E17 只能把已有实体连接到已有 object mesh，无法恢复 E11–E16 "
                "漏掉的两把后排椅，也无法把售货机的多个部件 node 自动合并成两个"
                "物理机器；这些必须在上游分割/实体合并/对象提取或后续关系层处理。"
            ),
            "",
            "## D0 位置偏移检查",
            "",
            (
                "保存了每个实际 matched pair 在 0.1/0.3/0.6m 径向远离后的精确"
                "中心距离与 AABB gap 重算。该表只判断原 chosen pair 是否会被门限"
                "拒绝，不模拟它随后改绑到另一 mesh，因此不能作为 wrong-mesh 检出率。"
            ),
            "",
            "## 证据索引",
            "",
            "- `PRE_REGISTRATION.json`：运行前冻结的假设、单变量和禁止声明。",
            "- `inputs/`：E14 语义源 DSG、输入 hash 与 upstream 快照。",
            "- `variants/*/binding_audit.json`：生产绑定函数的完整 append-only event。",
            "- `variants/*/candidate_matrix.{jsonl,csv}`：每个实体×每个 real mesh 的完整矩阵。",
            "- `variants/*/terminal_decisions.{jsonl,csv}`：87 个实体逐项最终结果。",
            "- `variants/*/offset_probe.{jsonl,csv}`：0.1/0.3/0.6m 扰动证据。",
            "- `variants/*/dsg_bound.json`：含 real object mesh 的最终绑定 DSG。",
            "- `visualizations/`：结果柱图、逐实体矩阵、候选分布、扰动曲线及六张绑定线图。",
            "- `tables/threshold_transitions.{jsonl,csv}`：同实体跨门限迁移。",
            "- `artifact_inventory.*`：逐文件 hash 与根摘要。",
            "",
            "## 结论边界",
            "",
            (
                "正式规范要求 `GT entity + frozen Hydra object/mesh`。本轮按用户当前"
                "阶段要求跳过人工 GT，使用冻结 E14/E16 现有数据，所以只能判断工程"
                "完整性、门限行为、provenance 一致性和风险，不能宣称达到规范建议的"
                " `binding precision >=95%`。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"E17 output already exists; use a new run directory: {output}"
        )
    output.mkdir(parents=True)

    e14 = args.e14_run.expanduser().resolve()
    e16_sweep = args.e16_sweep.expanduser().resolve()
    e16_comparison = args.e16_comparison.expanduser().resolve()
    labels_path = e14 / "tables" / "final_labels.jsonl"
    entities_path = e14 / "cells" / "obs_08" / "seed_0" / "final_entities.json"
    memory_path = e14 / "cells" / "obs_08" / "seed_0" / "map_memory.sqlite3"
    single_root = (
        e16_sweep / "variants" / "voxel_12cm_obs6_range8m_vol0p005"
    )
    single_dsg = single_root / "hydra_realtime" / "backend" / "dsg.json"
    adaptive_root = (
        e16_comparison / "adaptive_obs6_near5_far8_complete"
    )
    adaptive_dsg = adaptive_root / "hydra_realtime" / "backend" / "dsg.json"
    e16_summary_path = e16_comparison / "SUMMARY.json"
    e16_review_path = e16_comparison / "TARGET_REVIEW.jsonl"
    d0_path = (
        EXPERIMENT_ROOT
        / "runs"
        / "diagnostic_gt_free_d0_contract_probes_20260728"
        / "07_entity_position_offset"
        / "aggregate.json"
    )
    required = [
        labels_path,
        entities_path,
        memory_path,
        single_dsg,
        adaptive_dsg,
        e16_summary_path,
        e16_review_path,
        d0_path,
        args.semantic_config.expanduser().resolve(),
        args.labelspace_colors.expanduser().resolve(),
    ]
    require_files(required)

    preregistration = make_preregistration(
        inputs={
            "e14_final_labels": labels_path,
            "e14_final_entities": entities_path,
            "e14_map_memory": memory_path,
            "e16_single_pass_dsg": single_dsg,
            "e16_adaptive_dsg": adaptive_dsg,
            "e16_codex_review": e16_review_path,
        },
        output=output,
    )
    write_json(output / "PRE_REGISTRATION.json", preregistration)

    started = datetime.now(timezone.utc)
    entities = load_entities(labels_path, entities_path)
    owners = {int(row["semantic_id"]): str(row["entity_id"]) for row in entities}
    semantic_source_path = output / "inputs" / "e14_semantic_source_dsg.json"
    semantic_source = build_semantic_source(semantic_source_path, entities)
    write_jsonl(output / "inputs" / "named_entities.jsonl", entities)
    write_csv(output / "inputs" / "named_entities.csv", entities)
    shutil.copy2(e16_summary_path, output / "inputs" / "e16_upstream_summary.json")
    shutil.copy2(e16_review_path, output / "inputs" / "e16_target_review.jsonl")
    shutil.copy2(d0_path, output / "inputs" / "d0_entity_offset_aggregate.json")
    copy_source_preview(
        single_root,
        output / "inputs" / "single_pass_hydra_map_preview.png",
    )
    copy_source_preview(
        adaptive_root,
        output / "inputs" / "adaptive_hydra_map_preview.png",
    )
    input_manifest = {
        "schema": "daaam.g1_no_gt_e17_input_manifest.v1",
        "semantic_source": semantic_source,
        "named_entity_count": len(entities),
        "geometry_sources": {
            "single_pass": {
                "path": str(single_dsg),
                "sha256": sha256(single_dsg),
                "selection": "12cm_obs6_range8m_vol0p005",
                "execution_mode": "one_pass_production_candidate",
            },
            "adaptive": {
                "path": str(adaptive_dsg),
                "sha256": sha256(adaptive_dsg),
                "selection": "obs6_near5_plus_far8_far_only",
                "execution_mode": "two_pass_derived_diagnostic_upper_bound",
            },
        },
        "memory": {"path": str(memory_path), "sha256": sha256(memory_path)},
        "identity_contract": (
            "E13/E14 entity_ordinal equals the semantic label written to E16 label maps"
        ),
        "formal_gt": None,
    }
    write_json(output / "inputs" / "INPUT_MANIFEST.json", input_manifest)
    write_json(
        output / "invocation.json",
        {
            "schema": "daaam.g1_no_gt_e17_invocation.v1",
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("branch", "--show-current"),
            "git_status_porcelain": git_value("status", "--short"),
            "started_at": started.isoformat(),
        },
    )
    source_snapshot = output / "source_snapshot"
    source_snapshot.mkdir()
    shutil.copy2(Path(__file__), source_snapshot / Path(__file__).name)
    shutil.copy2(
        REPOSITORY_ROOT / "src" / "daaam" / "experiments" / "e17_support.py",
        source_snapshot / "e17_support.py",
    )
    shutil.copy2(
        REPOSITORY_ROOT / "scripts" / "rebind_dsg_semantics.py",
        source_snapshot / "rebind_dsg_semantics.py",
    )

    geometry_sources = {
        "single_pass": (single_dsg, single_root),
        "adaptive": (adaptive_dsg, adaptive_root),
    }
    all_candidates: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_offsets: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []
    decisions_by_source: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for source_name, (source_dsg, _) in geometry_sources.items():
        nodes = real_mesh_nodes(source_dsg)
        write_jsonl(output / "inputs" / f"{source_name}_real_mesh_nodes.jsonl", nodes)
        write_csv(output / "inputs" / f"{source_name}_real_mesh_nodes.csv", nodes)
        decisions_by_source[source_name] = {}
        for threshold in E17_THRESHOLDS:
            cell_started = time.perf_counter()
            variant_name = f"{source_name}_{threshold.name}"
            variant = output / "variants" / variant_name
            variant.mkdir(parents=True)
            candidates = build_candidate_rows(
                entities,
                nodes,
                threshold,
                owners,
                geometry_source=source_name,
            )
            write_jsonl(variant / "candidate_matrix.jsonl", candidates)
            write_csv(variant / "candidate_matrix.csv", candidates)

            bound_dsg = variant / "dsg_bound.json"
            binding_audit = variant / "binding_audit.json"
            audit = migrate(
                SimpleNamespace(
                    dsg=source_dsg,
                    semantic_source_dsg=semantic_source_path,
                    semantic_source_report=None,
                    memory=memory_path,
                    output=bound_dsg,
                    audit_output=binding_audit,
                    semantic_config=args.semantic_config.expanduser().resolve(),
                    labelspace_colors=args.labelspace_colors.expanduser().resolve(),
                    maximum_center_distance_m=(
                        threshold.maximum_center_distance_m
                    ),
                    maximum_aabb_gap_m=threshold.maximum_aabb_gap_m,
                    time_origin_ns=None,
                    force=False,
                )
            )
            decisions = terminal_decisions(audit["events"], entities)
            for row in decisions:
                row["geometry_source"] = source_name
                row["threshold"] = threshold.name
                row["maximum_center_distance_m"] = (
                    threshold.maximum_center_distance_m
                )
                row["maximum_aabb_gap_m"] = threshold.maximum_aabb_gap_m
            summary = summarize_decisions(
                decisions,
                named_entity_count=len(entities),
                real_mesh_count=len(nodes),
            )
            summary.update(
                {
                    "schema": "daaam.g1_no_gt_e17_cell_summary.v1",
                    "variant": variant_name,
                    "geometry_source": source_name,
                    "threshold": threshold.name,
                    "maximum_center_distance_m": (
                        threshold.maximum_center_distance_m
                    ),
                    "maximum_aabb_gap_m": threshold.maximum_aabb_gap_m,
                    "candidate_pair_count": len(candidates),
                    "accepted_candidate_pair_count_before_assignment": int(
                        sum(bool(row["eligible_before_assignment"]) for row in candidates)
                    ),
                    "source_dsg_sha256": sha256(source_dsg),
                    "output_dsg_sha256": sha256(bound_dsg),
                    "binding_audit_sha256": sha256(binding_audit),
                    "mesh_counts": audit["mesh_counts"],
                    "verification": audit["verification"],
                    "elapsed_seconds": time.perf_counter() - cell_started,
                    "formal_claims_permitted": False,
                }
            )
            offsets = offset_rows(
                geometry_source=source_name,
                threshold=threshold,
                decisions=decisions,
                candidate_rows=candidates,
            )
            event_rows = []
            for event_index, event in enumerate(audit["events"]):
                event_rows.append(
                    {
                        "schema": "daaam.g1_e17_binding_event.v1",
                        "variant": variant_name,
                        "event_index": event_index,
                        **event,
                    }
                )
            write_jsonl(variant / "binding_events.jsonl", event_rows)
            write_csv(variant / "binding_events.csv", event_rows)
            write_jsonl(variant / "terminal_decisions.jsonl", decisions)
            write_csv(variant / "terminal_decisions.csv", decisions)
            write_jsonl(variant / "offset_probe.jsonl", offsets)
            write_csv(variant / "offset_probe.csv", offsets)
            write_json(
                variant / "SUMMARY.json",
                summary,
            )
            risky = [
                row
                for row in decisions
                if row["risk_class"] != "provenance_consistent_semantic_id"
            ]
            write_jsonl(variant / "review_queue.jsonl", risky)
            write_csv(variant / "review_queue.csv", risky)
            topdown_plot(
                output
                / "visualizations"
                / f"binding_lines_{source_name}_{threshold.name}.png",
                geometry_source=source_name,
                threshold=threshold,
                nodes=nodes,
                entities=entities,
                decisions=decisions,
            )

            decisions_by_source[source_name][threshold.name] = decisions
            all_candidates.extend(candidates)
            all_decisions.extend(decisions)
            all_events.extend(event_rows)
            all_offsets.extend(offsets)
            cell_summaries.append(summary)

    transitions = []
    for source_name, decisions_by_threshold in decisions_by_source.items():
        transitions.extend(
            threshold_transitions(source_name, decisions_by_threshold)
        )
    write_jsonl(output / "tables" / "cell_summary.jsonl", cell_summaries)
    write_csv(output / "tables" / "cell_summary.csv", cell_summaries)
    write_jsonl(output / "tables" / "all_terminal_decisions.jsonl", all_decisions)
    write_csv(output / "tables" / "all_terminal_decisions.csv", all_decisions)
    write_jsonl(output / "tables" / "all_binding_events.jsonl", all_events)
    write_csv(output / "tables" / "all_binding_events.csv", all_events)
    write_jsonl(output / "tables" / "threshold_transitions.jsonl", transitions)
    write_csv(output / "tables" / "threshold_transitions.csv", transitions)
    write_jsonl(output / "tables" / "all_offset_probes.jsonl", all_offsets)
    write_csv(output / "tables" / "all_offset_probes.csv", all_offsets)

    upstream_summary = json.loads(e16_summary_path.read_text())
    aggregate_visualizations(
        output,
        cell_summaries,
        transitions,
        all_candidates,
        all_offsets,
    )
    report = build_report(
        cell_summaries=cell_summaries,
        transitions=transitions,
        upstream_summary=upstream_summary,
        output=output,
    )
    (output / "REPORT.md").write_text(report)
    completed = datetime.now(timezone.utc)
    run_summary = {
        "schema": "daaam.g1_no_gt_e17_run_summary.v1",
        "status": "complete_pending_independent_audit",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "named_entity_count": len(entities),
        "geometry_source_count": len(geometry_sources),
        "threshold_count": len(E17_THRESHOLDS),
        "cell_count": len(cell_summaries),
        "cell_summaries": cell_summaries,
        "threshold_transition_counts": {
            "gained_by_medium": int(
                sum(bool(row["gained_by_medium"]) for row in transitions)
            ),
            "gained_by_wide": int(
                sum(bool(row["gained_by_wide"]) for row in transitions)
            ),
            "reassigned_when_widened": int(
                sum(bool(row["reassigned_when_widened"]) for row in transitions)
            ),
            "spatial_fallback_in_any_variant": int(
                sum(
                    bool(row["spatial_fallback_in_any_variant"])
                    for row in transitions
                )
            ),
        },
        "formal_claims_permitted": False,
        "formal_binding_precision": None,
        "formal_binding_recall": None,
        "formal_binding_f1": None,
        "evaluation_basis": (
            "exact engineering audit + semantic-ID provenance proxy + frozen "
            "single-Codex upstream E16 review; no human binding GT"
        ),
    }
    write_json(output / "RUN_SUMMARY.json", run_summary)
    inventory_summary = inventory(output)
    result = {
        **run_summary,
        "artifact_inventory": inventory_summary,
        "output": str(output),
    }
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
                    "schema": "daaam.g1_no_gt_e17_terminal_failure.v1",
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
