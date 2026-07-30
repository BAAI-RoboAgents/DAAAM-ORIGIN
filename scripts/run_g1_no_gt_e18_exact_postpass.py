#!/usr/bin/env python3
"""Run the frozen-input G1 E18 exact semantic-label postpass experiment."""

from __future__ import annotations

import argparse
from collections import Counter
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

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from daaam.experiments.e18_support import (  # noqa: E402
    DETERMINISTIC_PRODUCT_FILES,
    compare_exact_repetitions,
    evaluate_durable_commit,
    sha256_file,
    stable_records_sha256,
    validate_postpass_report,
)
from daaam.realtime.semantic_labels import (  # noqa: E402
    load_semantic_label,
    semantic_label_metadata_path,
    semantic_label_path,
    validate_semantic_label_binding,
)
from run_hydra_semantic_postpass import (  # noqa: E402
    validate_plan_frame_records,
)


EXPERIMENT_ROOT = (
    REPOSITORY_ROOT / "experiments" / "g1_20260724_473_573_v1_1"
)
SOURCE_VARIANT = (
    EXPERIMENT_ROOT
    / "runs"
    / "diagnostic_gt_free_e16_12cm_obs_range_sweep_20260730"
    / "variants"
    / "voxel_12cm_obs6_range8m_vol0p005"
)
DEFAULT_PLAN = SOURCE_VARIANT / "hydra_postpass_plan.json"
DEFAULT_SOURCE_REPORT = SOURCE_VARIANT / "hydra_postpass_report.json"
DEFAULT_E17_CELL = (
    EXPERIMENT_ROOT
    / "comparisons"
    / "e17_binding_policy_ablation_20260730"
    / "cells"
    / "single_pass_D_global_joint"
)
E17_ABLATION_ROOT = DEFAULT_E17_CELL.parents[1]
E17_BASELINE_ROOT = (
    EXPERIMENT_ROOT
    / "runs"
    / "diagnostic_gt_free_e17_e14e16fed_binding_20260730"
)
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT
    / "runs"
    / "diagnostic_gt_free_e18_exact_postpass_20260730"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--source-report", type=Path, default=DEFAULT_SOURCE_REPORT
    )
    parser.add_argument("--e17-cell", type=Path, default=DEFAULT_E17_CELL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    temporary.replace(path)


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def source_environment() -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "schema": "daaam.g1_e18_environment.v1",
        "captured_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_branch": git_value("branch", "--show-current"),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_short": git_value("status", "--short"),
        "hydra_git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT / ".repro" / "ros2_ws" / "src" / "hydra",
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip(),
        "hydra_git_diff": subprocess.run(
            ["git", "diff", "--", "src/places/graph_extractor.cpp"],
            cwd=REPOSITORY_ROOT / ".repro" / "ros2_ws" / "src" / "hydra",
            text=True,
            capture_output=True,
            check=False,
        ).stdout,
        "hydra_determinism_patch": str(
            (REPOSITORY_ROOT / "patches" / "hydra_nearest_vertex_value_init.patch")
            .resolve()
        ),
        "hydra_determinism_patch_sha256": sha256_file(
            REPOSITORY_ROOT / "patches" / "hydra_nearest_vertex_value_init.patch"
        ),
        "gpu": gpu.stdout.strip(),
    }


def exact_manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                f"{row['frame_index']}:{row['sensor_time_ns']}:"
                f"{row['label_sha256']}:{row['label_metadata_sha256']}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def build_input_ledger(
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    records = validate_plan_frame_records(plan.get("frames"))
    run_dir = Path(plan["run_dir"])
    label_dir = Path(plan["semantic_label_dir"])
    configuration = str(plan["label_run_configuration_sha256"])
    rows: list[dict[str, Any]] = []
    for record in records:
        frame_index = int(record["frame_index"])
        sensor_time_ns = int(record["sensor_time_ns"])
        rgb_path = Path(record["rgb_path"])
        depth_path = run_dir / "static_depth" / f"{frame_index:08d}.png"
        binding = validate_semantic_label_binding(
            label_dir,
            frame_index,
            sensor_time_ns=sensor_time_ns,
            run_configuration_sha256=configuration,
        )
        labels = load_semantic_label(label_dir, frame_index)
        label_path = semantic_label_path(label_dir, frame_index)
        metadata_path = semantic_label_metadata_path(label_dir, frame_index)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None:
            raise FileNotFoundError(rgb_path)
        if depth is None:
            raise FileNotFoundError(depth_path)
        if rgb.shape[:2] != labels.shape or depth.shape != labels.shape:
            raise ValueError(f"input shape mismatch for frame {frame_index}")
        rows.append(
            {
                "schema": "daaam.g1_e18_input_frame.v1",
                "frame_index": frame_index,
                "sensor_time_ns": sensor_time_ns,
                "rgb_path": str(rgb_path.resolve()),
                "rgb_sha256": sha256_file(rgb_path),
                "rgb_size_bytes": rgb_path.stat().st_size,
                "depth_path": str(depth_path.resolve()),
                "depth_sha256": sha256_file(depth_path),
                "depth_size_bytes": depth_path.stat().st_size,
                "label_path": str(label_path.resolve()),
                "label_sha256": str(binding["image_sha256"]),
                "label_size_bytes": label_path.stat().st_size,
                "label_metadata_path": str(metadata_path.resolve()),
                "label_metadata_sha256": str(binding["metadata_sha256"]),
                "label_metadata_size_bytes": metadata_path.stat().st_size,
                "shape": list(labels.shape),
                "dtype": str(labels.dtype),
                "minimum_label": int(labels.min(initial=0)),
                "maximum_label": int(labels.max(initial=0)),
                "nonzero_pixels": int(np.count_nonzero(labels)),
                "unique_label_count": int(np.unique(labels).size),
                "run_configuration_sha256": configuration,
                "binding_valid": True,
            }
        )
    return rows, exact_manifest_sha256(rows)


def graph_summary(path: Path) -> dict[str, Any]:
    from spark_dsg import DsgLayers, DynamicSceneGraph

    graph = DynamicSceneGraph.load(str(path))
    objects = list(graph.get_layer(DsgLayers.OBJECTS).nodes)
    semantic_labels = Counter(
        int(node.attributes.semantic_label) for node in objects
    )
    mesh_vertices = []
    for node in objects:
        mesh = node.attributes.mesh()
        mesh_vertices.append(0 if mesh is None else int(mesh.num_vertices()))
    layers = {}
    for layer in graph.layers:
        layers[str(layer.id)] = {
            "nodes": int(layer.num_nodes()),
            "edges": int(layer.num_edges()),
        }
    return {
        "schema": "daaam.g1_e18_graph_summary.v1",
        "total_nodes": int(graph.num_nodes()),
        "total_edges": int(graph.num_edges()),
        "layers": layers,
        "object_nodes": len(objects),
        "object_nodes_with_mesh": sum(value > 0 for value in mesh_vertices),
        "object_mesh_vertices": int(sum(mesh_vertices)),
        "object_semantic_label_histogram": {
            str(key): int(value) for key, value in sorted(semantic_labels.items())
        },
    }


def artifact_hashes(output_dir: Path) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for relative in DETERMINISTIC_PRODUCT_FILES:
        path = output_dir / relative
        result[relative] = sha256_file(path) if path.is_file() else None
    return result


def read_resource_usage(path: Path) -> dict[str, Any]:
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            if ": " not in line:
                continue
            key, value = line.strip().rsplit(": ", 1)
            values[key] = value.strip()
    return {
        "maximum_resident_set_kbytes": int(
            values.get("Maximum resident set size (kbytes)", "0") or 0
        ),
        "user_time_seconds": float(
            values.get("User time (seconds)", "0") or 0.0
        ),
        "system_time_seconds": float(
            values.get("System time (seconds)", "0") or 0.0
        ),
        "wall_clock": values.get(
            "Elapsed (wall clock) time (h:mm:ss or m:ss)", ""
        ),
        "exit_status": int(values.get("Exit status", "-1") or -1),
    }


def run_exact_repetition(
    *,
    repetition: int,
    source_plan: Mapping[str, Any],
    output_root: Path,
    expected_manifest: str,
) -> dict[str, Any]:
    cell = output_root / "exact_repetitions" / f"rep_{repetition:02d}"
    hydra_output = cell / "hydra_realtime"
    plan = dict(source_plan)
    plan["frames"] = [dict(row) for row in source_plan["frames"]]
    plan["output_dir"] = str(hydra_output.resolve())
    plan_path = cell / "postpass_plan.json"
    report_path = cell / "postpass_report.json"
    stdout_path = cell / "stdout.log"
    stderr_path = cell / "stderr.log"
    resource_path = cell / "resource_usage.txt"
    write_json(plan_path, plan)
    command = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(resource_path),
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_hydra_semantic_postpass.py"),
        "--plan",
        str(plan_path),
        "--report",
        str(report_path),
    ]
    (cell / "COMMAND.txt").write_text(" ".join(command) + "\n")
    started = time.monotonic()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=600.0,
        )
    wall_seconds = time.monotonic() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"E18 repetition {repetition} failed: "
            + stderr_path.read_text(errors="replace")[-4000:]
        )
    report = json.loads(report_path.read_text())
    issues = validate_postpass_report(
        report,
        expected_frames=len(source_plan["frames"]),
        expected_label_manifest_sha256=expected_manifest,
        expected_run_configuration_sha256=str(
            source_plan["label_run_configuration_sha256"]
        ),
    )
    if issues:
        raise RuntimeError(f"E18 repetition {repetition} contract: {issues}")
    required = [hydra_output / relative for relative in DETERMINISTIC_PRODUCT_FILES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"E18 repetition {repetition} missing: {missing}")
    summary = {
        "schema": "daaam.g1_e18_exact_repetition.v1",
        "repetition": repetition,
        "status": "complete",
        "wall_seconds": wall_seconds,
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "output_dir": str(hydra_output.resolve()),
        "report": report,
        "contract_issues": issues,
        "artifact_hashes": artifact_hashes(hydra_output),
        "graph_summary": graph_summary(
            hydra_output / "backend" / "dsg_with_mesh.json"
        ),
        "resource_usage": read_resource_usage(resource_path),
    }
    write_json(cell / "SUMMARY.json", summary)
    return summary


def make_label_overlay(
    rgb: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    maximum = int(labels.max(initial=0))
    palette = np.zeros((maximum + 1, 3), dtype=np.uint8)
    for semantic_id in range(1, maximum + 1):
        palette[semantic_id] = [
            (semantic_id * 37 + 29) % 255,
            (semantic_id * 67 + 71) % 255,
            (semantic_id * 97 + 113) % 255,
        ]
    color = palette[labels]
    overlay = rgb.copy()
    mask = labels > 0
    overlay[mask] = cv2.addWeighted(
        rgb[mask],
        0.45,
        color[mask],
        0.55,
        0.0,
    )
    boundary = mask & (
        (labels != np.roll(labels, 1, axis=0))
        | (labels != np.roll(labels, -1, axis=0))
        | (labels != np.roll(labels, 1, axis=1))
        | (labels != np.roll(labels, -1, axis=1))
    )
    overlay[boundary] = [0, 255, 255]
    return color, overlay


def render_label_evidence(
    plan: Mapping[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    records = plan["frames"]
    offsets = sorted({0, len(records) // 4, len(records) // 2, 3 * len(records) // 4, len(records) - 2})
    label_dir = Path(plan["semantic_label_dir"])
    rows: list[dict[str, Any]] = []
    panels = []
    for offset in offsets:
        record = records[offset]
        frame_index = int(record["frame_index"])
        rgb = cv2.imread(str(record["rgb_path"]), cv2.IMREAD_COLOR)
        labels = load_semantic_label(label_dir, frame_index)
        color, overlay = make_label_overlay(rgb, labels)
        triptych = np.concatenate([rgb, color, overlay], axis=1)
        cv2.putText(
            triptych,
            f"frame {frame_index}: RGB | uint16 labels | exact overlay",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        path = (
            output_root
            / "visualizations"
            / "label_overlays"
            / f"frame_{frame_index:08d}.jpg"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), triptych, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(path)
        panels.append(cv2.resize(triptych, (960, 240)))
        rows.append(
            {
                "frame_index": frame_index,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "nonzero_pixels": int(np.count_nonzero(labels)),
                "unique_label_count": int(np.unique(labels).size),
            }
        )
    montage = np.concatenate(panels, axis=0)
    montage_path = output_root / "visualizations" / "01_label_overlay_montage.jpg"
    if not cv2.imwrite(
        str(montage_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 95]
    ):
        raise OSError(montage_path)
    write_json(output_root / "visualizations" / "label_overlays.json", rows)
    return rows


def link_label_overlay(
    source: Path,
    destination: Path,
    *,
    omitted_png_index: int | None = None,
    corrupt_metadata_index: int | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for source_path in sorted(source.iterdir()):
        if not source_path.is_file():
            continue
        frame_index = int(source_path.stem)
        if (
            omitted_png_index is not None
            and frame_index == omitted_png_index
            and source_path.suffix == ".png"
        ):
            continue
        target = destination / source_path.name
        if (
            corrupt_metadata_index is not None
            and frame_index == corrupt_metadata_index
            and source_path.suffix == ".json"
        ):
            record = json.loads(source_path.read_text())
            record["image_sha256"] = "0" * 64
            write_json(target, record)
        else:
            os.symlink(source_path.resolve(), target)


def run_failure_probe(
    *,
    name: str,
    plan: Mapping[str, Any],
    output_root: Path,
    expected_error: str,
) -> dict[str, Any]:
    cell = output_root / "failure_injections" / name
    probe_plan = dict(plan)
    probe_plan["frames"] = [dict(row) for row in plan["frames"]]
    probe_plan["output_dir"] = str((cell / "hydra_realtime").resolve())
    plan_path = cell / "postpass_plan.json"
    report_path = cell / "postpass_report.json"
    stdout_path = cell / "stdout.log"
    stderr_path = cell / "stderr.log"
    write_json(plan_path, probe_plan)
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_hydra_semantic_postpass.py"),
        "--plan",
        str(plan_path),
        "--report",
        str(report_path),
    ]
    (cell / "COMMAND.txt").write_text(" ".join(command) + "\n")
    started = time.monotonic()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=90.0,
        )
    stderr_text = stderr_path.read_text(errors="replace")
    required_outputs = {
        relative: (cell / "hydra_realtime" / relative).is_file()
        for relative in DETERMINISTIC_PRODUCT_FILES
    }
    row = {
        "schema": "daaam.g1_e18_failure_injection.v1",
        "probe": name,
        "expected_error": expected_error,
        "returncode": result.returncode,
        "failed_as_expected": result.returncode != 0
        and expected_error in stderr_text,
        "elapsed_seconds": time.monotonic() - started,
        "report_created": report_path.is_file(),
        "required_outputs_created": required_outputs,
        "no_formal_product_committed": not any(required_outputs.values()),
        "stderr_tail": stderr_text[-4000:],
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
    }
    write_json(cell / "SUMMARY.json", row)
    return row


def failure_injections(
    source_plan: Mapping[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    label_source = Path(source_plan["semantic_label_dir"])
    target_index = int(source_plan["frames"][len(source_plan["frames"]) // 2]["frame_index"])
    rows: list[dict[str, Any]] = []

    missing_plan = dict(source_plan)
    missing_dir = output_root / "failure_injections" / "missing_label" / "labels"
    link_label_overlay(
        label_source,
        missing_dir,
        omitted_png_index=target_index,
    )
    missing_plan["semantic_label_dir"] = str(missing_dir.resolve())
    rows.append(
        run_failure_probe(
            name="missing_label",
            plan=missing_plan,
            output_root=output_root,
            expected_error="100% exact-frame label coverage",
        )
    )

    corrupt_plan = dict(source_plan)
    corrupt_dir = output_root / "failure_injections" / "corrupt_hash" / "labels"
    link_label_overlay(
        label_source,
        corrupt_dir,
        corrupt_metadata_index=target_index,
    )
    corrupt_plan["semantic_label_dir"] = str(corrupt_dir.resolve())
    rows.append(
        run_failure_probe(
            name="corrupt_hash",
            plan=corrupt_plan,
            output_root=output_root,
            expected_error="image hash mismatch",
        )
    )

    stale_plan = dict(source_plan)
    stale_plan["label_run_configuration_sha256"] = "f" * 64
    rows.append(
        run_failure_probe(
            name="stale_configuration",
            plan=stale_plan,
            output_root=output_root,
            expected_error="run-configuration binding mismatch",
        )
    )

    duplicate_plan = dict(source_plan)
    duplicate_records = [dict(row) for row in source_plan["frames"]]
    duplicate_records.insert(len(duplicate_records) // 2, dict(duplicate_records[len(duplicate_records) // 2]))
    duplicate_plan["frames"] = duplicate_records
    rows.append(
        run_failure_probe(
            name="duplicate_frame",
            plan=duplicate_plan,
            output_root=output_root,
            expected_error="frame indices must be unique",
        )
    )
    return rows


def durable_commit(
    e17_cell: Path,
    repetitions: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    from spark_dsg import DsgLayers, DynamicSceneGraph

    from run_g1_no_gt_e17_binding_policy_ablation import (
        bind_batch,
        node_binding_map,
        save_graph,
    )

    decisions = [
        json.loads(line)
        for line in (e17_cell / "terminal_decisions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    pending = [
        json.loads(line)
        for line in (e17_cell / "pending_candidates.jsonl").read_text().splitlines()
        if line.strip()
    ]
    matched = [row for row in decisions if row["status"] == "matched_real_mesh"]
    rejected = [
        row
        for row in decisions
        if row["status"]
        in {"rejected_no_mesh", "rejected_no_authoritative_mesh"}
    ]
    entities = [
        json.loads(line)
        for line in (
            E17_ABLATION_ROOT / "inputs" / "named_entities.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]
    candidates = [
        json.loads(line)
        for line in (
            E17_BASELINE_ROOT
            / "variants"
            / "single_pass_strict"
            / "candidate_matrix.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]
    exact_entity_ids = {
        str(row["entity_id"]) for row in candidates if row["semantic_id_match"]
    }
    selected_cross_ids = {
        str(row["entity_id"])
        for row in (
            json.loads(line)
            for line in (
                E17_ABLATION_ROOT / "tables" / "global_assignments.jsonl"
            ).read_text().splitlines()
            if line.strip()
        )
        if row["geometry_source"] == "single_pass"
    }
    entity_lookup = {str(row["entity_id"]): row for row in entities}
    exact_entities = [entity_lookup[value] for value in exact_entity_ids]
    selected_cross_entities = [
        entity_lookup[value] for value in selected_cross_ids
    ]
    remaining_entities = [
        row
        for row in entities
        if str(row["entity_id"]) not in exact_entity_ids
        and str(row["entity_id"]) not in selected_cross_ids
    ]
    owners = {
        int(row["semantic_id"]): str(row["entity_id"]) for row in entities
    }
    binding_runs = []
    for repetition in repetitions:
        index = int(repetition["repetition"])
        graph = DynamicSceneGraph.load(
            str(Path(repetition["output_dir"]) / "backend" / "dsg.json")
        )
        events = []
        events.extend(
            bind_batch(
                graph,
                exact_entities,
                owners=owners,
                cross_policy="reject",
                semantic_config=REPOSITORY_ROOT / "config" / "labels_pseudo.yaml",
                labelspace_colors=REPOSITORY_ROOT / "config" / "labels_pseudo.csv",
                phase="E18_D_exact_id",
            )
        )
        events.extend(
            bind_batch(
                graph,
                selected_cross_entities,
                owners=owners,
                cross_policy="apply",
                semantic_config=REPOSITORY_ROOT / "config" / "labels_pseudo.yaml",
                labelspace_colors=REPOSITORY_ROOT / "config" / "labels_pseudo.csv",
                phase="E18_D_global_selected_cross_id",
            )
        )
        events.extend(
            bind_batch(
                graph,
                remaining_entities,
                owners=owners,
                cross_policy="pending",
                semantic_config=REPOSITORY_ROOT / "config" / "labels_pseudo.yaml",
                labelspace_colors=REPOSITORY_ROOT / "config" / "labels_pseudo.csv",
                phase="E18_D_unselected_pending",
            )
        )
        cell = output_root / "durable_commit" / "repetitions" / f"rep_{index:02d}"
        saved = save_graph(graph, cell / "dsg_bound.json")
        binding_map = node_binding_map(graph)
        if len(binding_map) != len(matched):
            raise RuntimeError(
                f"E18 durable commit repetition {index} bound "
                f"{len(binding_map)} entities, expected {len(matched)}"
            )
        write_jsonl(cell / "binding_events.jsonl", events)
        binding_run = {
            "schema": "daaam.g1_e18_binding_repetition.v1",
            "repetition": index,
            "source_exact_dsg": str(
                (Path(repetition["output_dir"]) / "backend" / "dsg.json").resolve()
            ),
            "source_exact_dsg_sha256": repetition["artifact_hashes"][
                "backend/dsg.json"
            ],
            "output_dsg": str((cell / "dsg_bound.json").resolve()),
            "output_dsg_sha256": saved["sha256"],
            "bound_entity_count": len(binding_map),
            "event_count": len(events),
        }
        write_json(cell / "SUMMARY.json", binding_run)
        binding_runs.append(binding_run)
    binding_hashes = [row["output_dsg_sha256"] for row in binding_runs]
    binding_output_hash_stable = len(set(binding_hashes)) == 1
    source = Path(binding_runs[0]["output_dsg"])
    destination = output_root / "durable_commit" / "dsg_bound.json"
    shutil.copy2(source, destination)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    graph = DynamicSceneGraph.load(str(destination))
    objects = list(graph.get_layer(DsgLayers.OBJECTS).nodes)
    entity_nodes = [
        node
        for node in objects
        if str((node.attributes.metadata.get() or {}).get("entity_id") or "")
    ]
    gate = evaluate_durable_commit(
        applied=len(matched),
        rejected_no_mesh=len(rejected),
        delivery_pending=0,
        unmapped=0,
        errors=[],
        graph_reloaded=True,
        output_hash_verified=(
            source_hash == destination_hash and binding_output_hash_stable
        ),
        candidate_review_pending=len({row["entity_id"] for row in pending}),
    )
    result = {
        **gate,
        "committed_at": utc_now(),
        "source_e17_cell": str(e17_cell.resolve()),
        "source_dsg": str(source.resolve()),
        "source_dsg_sha256": source_hash,
        "output_dsg": str(destination.resolve()),
        "output_dsg_sha256": destination_hash,
        "terminal_decision_count": len(decisions),
        "bound_entity_nodes_after_reload": len(entity_nodes),
        "binding_repetitions": binding_runs,
        "binding_output_hash_stable": binding_output_hash_stable,
        "graph_summary": graph_summary(destination),
        "review_queue_semantics": (
            "candidate_review_pending is optional cross-ID evidence, not an "
            "undelivered MapMemory correction"
        ),
    }
    write_json(output_root / "durable_commit" / "semantic_dsg_commit.json", result)
    return result


def plot_results(
    input_rows: Sequence[Mapping[str, Any]],
    repetitions: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> None:
    visualization = output_root / "visualizations"
    visualization.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    indices = [int(row["frame_index"]) for row in input_rows]
    nonzero = [int(row["nonzero_pixels"]) for row in input_rows]
    unique = [int(row["unique_label_count"]) for row in input_rows]
    axes[0].plot(indices, nonzero, linewidth=1.5)
    axes[0].set_title("E18 exact label nonzero-pixel coverage")
    axes[0].set_ylabel("nonzero pixels")
    axes[0].grid(alpha=0.25)
    axes[1].plot(indices, unique, color="tab:orange", linewidth=1.5)
    axes[1].set_title("Per-frame unique semantic IDs (including zero)")
    axes[1].set_xlabel("postpass frame index")
    axes[1].set_ylabel("unique labels")
    axes[1].grid(alpha=0.25)
    fig.savefig(visualization / "02_label_coverage_timeline.png", dpi=160)
    plt.close(fig)

    names = [f"rep {row['repetition']}" for row in repetitions]
    wall = [float(row["wall_seconds"]) for row in repetitions]
    rss = [
        float(row["resource_usage"]["maximum_resident_set_kbytes"]) / 1024.0
        for row in repetitions
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(names, wall, color="tab:blue")
    axes[0].set_title("Exact postpass wall time")
    axes[0].set_ylabel("seconds")
    axes[1].bar(names, rss, color="tab:purple")
    axes[1].set_title("Exact postpass peak RSS")
    axes[1].set_ylabel("MiB")
    fig.savefig(visualization / "03_repetition_resources.png", dpi=160)
    plt.close(fig)

    probes = [str(row["probe"]) for row in failure_rows]
    detected = [int(bool(row["failed_as_expected"])) for row in failure_rows]
    clean = [int(bool(row["no_formal_product_committed"])) for row in failure_rows]
    positions = np.arange(len(probes))
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.bar(positions - 0.18, detected, 0.36, label="rejected as expected")
    ax.bar(positions + 0.18, clean, 0.36, label="no formal product committed")
    ax.set_xticks(positions, probes, rotation=15)
    ax.set_ylim(0.0, 1.15)
    ax.set_title("E18 fail-closed contract probes")
    ax.legend()
    fig.savefig(visualization / "04_failure_injection_matrix.png", dpi=160)
    plt.close(fig)


def inventory(
    root: Path,
    *,
    exclusions: set[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in exclusions:
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    root_hash = stable_records_sha256(rows)
    write_jsonl(root / "artifact_inventory.jsonl", rows)
    write_csv(root / "artifact_inventory.csv", rows)
    summary = {
        "schema": "daaam.g1_e18_artifact_inventory.v1",
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "root_sha256": root_hash,
        "excluded_relative_paths": sorted(exclusions),
    }
    write_json(root / "inventory_summary.json", summary)
    return summary


def report_text(
    *,
    input_rows: Sequence[Mapping[str, Any]],
    repetitions: Sequence[Mapping[str, Any]],
    comparison: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    commit: Mapping[str, Any],
    source_report: Mapping[str, Any],
    source_hash_reproduced: bool,
) -> str:
    elapsed = [float(row["wall_seconds"]) for row in repetitions]
    rss = [
        float(row["resource_usage"]["maximum_resident_set_kbytes"]) / 1024.0
        for row in repetitions
    ]
    product_lines = "\n".join(
        f"| `{name}` | {'是' if stable else '否'} |"
        for name, stable in comparison["product_stability"].items()
    )
    probe_lines = "\n".join(
        f"| {row['probe']} | {row['returncode']} | "
        f"{'通过' if row['failed_as_expected'] else '失败'} | "
        f"{'是' if row['no_formal_product_committed'] else '否'} |"
        for row in failures
    )
    return f"""# G1 E18：Exact semantic-label postpass

## 结论

E18 exact formal 在同一冻结输入上独立运行 {len(repetitions)} 次；每次均重放
{len(input_rows)}/{len(input_rows)} 帧，标签覆盖率 100%，缺帧为 0。
输入标签 manifest hash 为
`{repetitions[0]['report']['label_manifest_sha256']}`，三次一致。

语义合同稳定：**{'通过' if comparison['semantic_contract_stable'] else '失败'}**；
正式产物逐字节稳定：**{'通过' if comparison['formal_product_hash_stable'] else '失败'}**；
DSG 图统计稳定：**{'通过' if comparison['graph_summary_stable'] else '失败'}**。
新结果与 E16 冻结源 DSG hash
**{'一致' if source_hash_reproduced else '不一致'}**。

E17-D durable commit 重载后应用 {commit['applied']} 个实体，delivery pending=0、
unmapped=0、error=0；另有 {commit['candidate_review_pending']} 个非阻断跨 ID
候选等待人工复核，不能混称为未提交 correction。最终提交门：
**{commit['status']}**。

## 正式输入

- 候选：E16 `12cm / obs6 / max_range=8m / min_volume=0.005m³`；
- 帧数：{len(input_rows)}，其中真实帧 101 + flush 帧 1；
- 非零标签帧：{source_report['nonzero_label_frames']}；
- 非零标签像素：{source_report['nonzero_label_pixels']}；
- unique semantic ID 数：{len(source_report['unique_semantic_labels'])}；
- live provisional：冻结 E16 运行没有保留替换前图，因此记为
  `not_retained_unscorable`，不伪造对照数字。

## 三重复

| repetition | wall (s) | peak RSS (MiB) | frames | coverage |
| ---: | ---: | ---: | ---: | ---: |
{chr(10).join(f"| {row['repetition']} | {row['wall_seconds']:.3f} | {row['resource_usage']['maximum_resident_set_kbytes']/1024.0:.1f} | {row['report']['frames_replayed']} | {row['report']['label_coverage']:.3f} |" for row in repetitions)}

Wall 范围 {min(elapsed):.3f}–{max(elapsed):.3f}s；峰值 RSS 范围
{min(rss):.1f}–{max(rss):.1f} MiB。Hydra finalize 的长尾仍属于离线
postpass 成本，不能宣称实时。

## 字节确定性

| 产物 | 三次 hash 相同 |
| --- | --- |
{product_lines}

## 故障注入

| probe | return code | 检出预期错误 | 未提交正式产物 |
| --- | ---: | --- | --- |
{probe_lines}

缺标签、损坏 hash、过期 run configuration 和重复帧均要求 fail closed。
覆盖率检查发生在首帧融合前；重复/乱序身份检查发生在 Hydra 分配状态前。

## 证据

- `inputs/frame_ledger.*`：逐帧 RGB/depth/label/metadata hash；
- `exact_repetitions/rep_*/`：完整 plan、stdout/stderr、资源记录与 Hydra 输出；
- `failure_injections/`：四类反事实输入、stderr 与未提交检查；
- `durable_commit/`：E17-D 最终 DSG、commit manifest 与重载结果；
- `visualizations/`：标签 overlay、覆盖时序、资源与故障矩阵；
- `artifact_inventory.*`：逐文件 hash 与根摘要。

## 结论边界

E18证明的是标签持久化、精确帧绑定、重放确定性和提交完整性，不证明标签语义本身
正确。E11–E17 已存在的漏检、错分、过合并和无 mesh 拒绝不会被 postpass 修复。
"""


def main() -> int:
    args = parse_args()
    if args.repetitions < 3:
        raise ValueError("E18 formal validation requires at least three repetitions")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable E18 run directory: {output}"
        )
    output.mkdir(parents=True)
    (output / "RUN_COMMAND.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
        + "\n"
    )
    write_json(output / "ENVIRONMENT.json", source_environment())

    source_plan_path = args.source_plan.resolve()
    source_report_path = args.source_report.resolve()
    e17_cell = args.e17_cell.resolve()
    source_plan = json.loads(source_plan_path.read_text())
    source_report = json.loads(source_report_path.read_text())
    records = validate_plan_frame_records(source_plan.get("frames"))
    source_plan["frames"] = records
    input_rows, input_manifest = build_input_ledger(source_plan)
    if input_manifest != source_report.get("label_manifest_sha256"):
        raise RuntimeError(
            "frozen label manifest does not match the E16 source report"
        )
    write_jsonl(output / "inputs" / "frame_ledger.jsonl", input_rows)
    write_csv(output / "inputs" / "frame_ledger.csv", input_rows)
    write_json(
        output / "inputs" / "INPUT_MANIFEST.json",
        {
            "schema": "daaam.g1_e18_input_manifest.v1",
            "source_plan": str(source_plan_path),
            "source_plan_sha256": sha256_file(source_plan_path),
            "source_report": str(source_report_path),
            "source_report_sha256": sha256_file(source_report_path),
            "source_e17_cell": str(e17_cell),
            "frame_count": len(input_rows),
            "frame_ledger_sha256": stable_records_sha256(input_rows),
            "label_manifest_sha256": input_manifest,
            "label_run_configuration_sha256": source_plan[
                "label_run_configuration_sha256"
            ],
            "rgb_depth_label_bindings_valid": True,
        },
    )
    write_json(output / "inputs" / "frozen_source_plan.json", source_plan)
    write_json(output / "inputs" / "frozen_source_report.json", source_report)
    write_json(
        output / "PRE_REGISTRATION.json",
        {
            "schema": "daaam.g1_e18_preregistration.v1",
            "registered_at": utc_now(),
            "stage": "E18",
            "hypothesis": (
                "exact postpass produces 100% frame-bound labels, stable hashes, "
                "fail-closed invalid input behavior, and a reloadable durable DSG"
            ),
            "formal_candidate": "exact_postpass_commit",
            "diagnostic_candidate": "live_provisional_not_retained_unscorable",
            "repetitions": args.repetitions,
            "required_product_hashes": list(DETERMINISTIC_PRODUCT_FILES),
            "hard_gates": {
                "label_coverage": 1.0,
                "missing_frames": 0,
                "semantic_contract_stable": True,
                "formal_product_hash_stable": True,
                "delivery_pending": 0,
                "unmapped": 0,
                "errors": 0,
                "failure_injections_detected": "4/4",
            },
            "failure_injections": [
                "missing_label",
                "corrupt_hash",
                "stale_configuration",
                "duplicate_frame",
            ],
            "retain_all_intermediates": True,
            "formal_semantic_accuracy_claim_permitted": False,
        },
    )

    repetitions = [
        run_exact_repetition(
            repetition=index,
            source_plan=source_plan,
            output_root=output,
            expected_manifest=input_manifest,
        )
        for index in range(args.repetitions)
    ]
    comparison = compare_exact_repetitions(repetitions)
    write_json(output / "analysis" / "repetition_comparison.json", comparison)

    failures = failure_injections(source_plan, output)
    write_jsonl(output / "analysis" / "failure_injections.jsonl", failures)
    write_csv(output / "analysis" / "failure_injections.csv", failures)
    commit = durable_commit(e17_cell, repetitions, output)
    overlay_rows = render_label_evidence(source_plan, output)
    plot_results(input_rows, repetitions, failures, output)

    source_dsg_hash = sha256_file(
        SOURCE_VARIANT / "hydra_realtime" / "backend" / "dsg.json"
    )
    repetition_dsg_hashes = [
        row["artifact_hashes"]["backend/dsg.json"] for row in repetitions
    ]
    source_hash_reproduced = all(
        value == source_dsg_hash for value in repetition_dsg_hashes
    )
    summary = {
        "schema": "daaam.g1_e18_summary.v1",
        "status": "complete_pending_independent_audit",
        "formal_candidate": "exact_postpass_commit",
        "live_provisional_status": "not_retained_unscorable",
        "frame_count": len(input_rows),
        "label_manifest_sha256": input_manifest,
        "label_coverage": repetitions[0]["report"]["label_coverage"],
        "missing_frame_indices": repetitions[0]["report"][
            "missing_frame_indices"
        ],
        "repetition_count": len(repetitions),
        "semantic_contract_stable": comparison["semantic_contract_stable"],
        "formal_product_hash_stable": comparison[
            "formal_product_hash_stable"
        ],
        "graph_summary_stable": comparison["graph_summary_stable"],
        "source_dsg_hash_reproduced": source_hash_reproduced,
        "failure_injections_detected": (
            f"{sum(bool(row['failed_as_expected']) for row in failures)}/"
            f"{len(failures)}"
        ),
        "failure_injections_fail_closed": all(
            bool(row["no_formal_product_committed"]) for row in failures
        ),
        "durable_commit_status": commit["status"],
        "durable_commit": commit,
        "overlay_count": len(overlay_rows),
        "formal_semantic_accuracy": None,
        "formal_semantic_accuracy_claim_permitted": False,
    }
    write_json(output / "SUMMARY.json", summary)
    (output / "REPORT.md").write_text(
        report_text(
            input_rows=input_rows,
            repetitions=repetitions,
            comparison=comparison,
            failures=failures,
            commit=commit,
            source_report=source_report,
            source_hash_reproduced=source_hash_reproduced,
        )
    )

    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in (
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "scripts" / "run_hydra_semantic_postpass.py",
        REPOSITORY_ROOT / "src" / "daaam" / "experiments" / "e18_support.py",
        REPOSITORY_ROOT / "tests" / "test_e18_support.py",
        REPOSITORY_ROOT / "tests" / "test_semantic_label_postpass.py",
    ):
        shutil.copy2(source, snapshot / source.name)

    hard_pass = (
        comparison["semantic_contract_stable"]
        and comparison["formal_product_hash_stable"]
        and comparison["graph_summary_stable"]
        and all(bool(row["failed_as_expected"]) for row in failures)
        and all(bool(row["no_formal_product_committed"]) for row in failures)
        and commit["status"] == "passed"
    )
    write_json(
        output / "COMPLETION.json",
        {
            "schema": "daaam.g1_e18_completion.v1",
            "status": (
                "complete_pending_independent_audit"
                if hard_pass
                else "complete_with_failed_hard_gate"
            ),
            "completed_at": utc_now(),
            "hard_gates_passed": hard_pass,
            "formal_semantic_accuracy_claim_permitted": False,
        },
    )
    inventory_summary = inventory(
        output,
        exclusions={
            "INDEPENDENT_AUDIT.json",
            "artifact_inventory.csv",
            "artifact_inventory.jsonl",
            "inventory_summary.json",
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "summary": summary,
                "hard_gates_passed": hard_pass,
                "inventory": inventory_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if hard_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
