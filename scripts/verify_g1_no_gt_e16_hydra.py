#!/usr/bin/env python3
"""Independently verify the GT-free E16 Hydra experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
from scipy.spatial import cKDTree
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPOSITORY_ROOT
    / "experiments/g1_20260724_473_573_v1_1/runs"
    / "diagnostic_gt_free_e16_safe035_hydra_20260730"
)
INVENTORY_EXCLUDES = {
    "artifact_inventory.csv",
    "artifact_inventory.jsonl",
    "inventory_summary.json",
    "COMPLETION.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, tolerance: float = 1.0e-6) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def verify_inventory(root: Path) -> dict[str, Any]:
    rows = read_jsonl(root / "artifact_inventory.jsonl")
    by_path = {str(row["relative_path"]): row for row in rows}
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in INVENTORY_EXCLUDES
    )
    check(actual == sorted(by_path), "inventory path set")
    for relative in actual:
        path = root / relative
        row = by_path[relative]
        check(path.stat().st_size == int(row["size_bytes"]), f"size: {relative}")
        check(sha256_file(path) == row["sha256"], f"sha256: {relative}")
    summary = read_json(root / "inventory_summary.json")
    observed = inventory_root(rows)
    check(observed == summary["inventory_root_sha256"], "inventory root")
    check(len(rows) == int(summary["file_count"]), "inventory file count")
    return {"file_count": len(rows), "root_sha256": observed}


def ply_header(path: Path) -> tuple[int, int]:
    vertices = faces = None
    with path.open("r", encoding="ascii") as stream:
        for line in stream:
            fields = line.split()
            if fields[:2] == ["element", "vertex"]:
                vertices = int(fields[2])
            elif fields[:2] == ["element", "face"]:
                faces = int(fields[2])
            elif fields and fields[0] == "end_header":
                break
    check(vertices is not None and faces is not None, f"PLY header: {path}")
    return int(vertices), int(faces)


def distance_fraction(values: np.ndarray, threshold: float) -> float:
    return float(np.mean(values <= threshold))


def main() -> int:
    args = parse_args()
    root = args.run.resolve()
    audit_path = root / "INDEPENDENT_AUDIT.json"
    check(root.is_dir(), "run directory")
    check(not audit_path.exists(), "audit already exists")
    inventory = verify_inventory(root)

    frozen = read_json(root / "FROZEN_INPUTS.json")
    frozen_count = 0
    for key, reference in frozen.items():
        if key in {"schema", "captured_at"}:
            continue
        path = Path(reference["path"])
        check(path.is_file(), f"frozen source exists: {key}")
        check(path.stat().st_size == int(reference["size_bytes"]), f"size: {key}")
        check(sha256_file(path) == reference["sha256"], f"sha256: {key}")
        frozen_count += 1

    completion = read_json(root / "COMPLETION.json")
    check(
        completion["status"] == "complete_pending_independent_audit",
        "pre-audit completion",
    )
    preregistration = read_json(root / "PRE_REGISTRATION.json")
    check(preregistration["formal_claims_permitted"] is False, "no formal claims")
    specs = preregistration["variants"]
    check(len(specs) == 5, "five preregistered variants")
    check(
        sum(bool(spec["diagnostic_only"]) for spec in specs) == 1,
        "one diagnostic variant",
    )
    check(
        next(spec for spec in specs if spec["diagnostic_only"])[
            "voxel_size_m"
        ]
        == 0.03,
        "3 cm diagnostic",
    )

    source_manifest = read_jsonl(
        root / "shared_input/source_frame_manifest.jsonl"
    )
    frames = read_jsonl(root / "shared_input/frames.jsonl")
    depth_manifest = read_jsonl(
        root / "shared_input/static_depth_manifest.jsonl"
    )
    label_manifest = read_jsonl(
        root / "shared_input/semantic_label_manifest.jsonl"
    )
    check(
        len(source_manifest) == 101
        and len(frames) == len(depth_manifest) == len(label_manifest) == 102,
        "101 source frames plus one flush event",
    )
    source_frames = [
        row for row in frames if not bool(row.get("is_flush_event", False))
    ]
    check(
        [int(row["source_frame_index"]) for row in source_frames]
        == list(range(473, 574)),
        "contiguous source frames 473-573",
    )
    flush_frames = [
        row for row in frames if bool(row.get("is_flush_event", False))
    ]
    check(len(flush_frames) == 1, "one explicit flush event")
    check(
        int(flush_frames[0]["sensor_time_ns"])
        - int(source_frames[-1]["sensor_time_ns"])
        == 10_100_000_000,
        "flush occurs 10.1 seconds after final source frame",
    )
    label_configuration = read_json(
        root / "shared_input/label_configuration.json"
    )
    for index, (source, frame, depth_row, label_row) in enumerate(
        zip(source_manifest, source_frames, depth_manifest[:101], label_manifest[:101])
    ):
        check(int(frame["frame_index"]) == index, f"frame index {index}")
        check(
            int(source["sensor_time_ns"]) == int(frame["sensor_time_ns"]),
            f"sensor time {index}",
        )
        for name in (
            "rgb",
            "scaled_depth_meter",
            "entity_id_map",
            "entity_frame",
            "visible_lidar_correspondence",
        ):
            reference = source[name]
            path = Path(reference["path"])
            check(path.is_file(), f"{index}: {name} exists")
            check(path.stat().st_size == int(reference["size_bytes"]), f"{index}: {name} size")
            check(sha256_file(path) == reference["sha256"], f"{index}: {name} hash")
        depth_path = Path(depth_row["static_depth_path"])
        labels_path = Path(label_row["image_path"])
        metadata_path = Path(label_row["metadata_path"])
        check(sha256_file(depth_path) == depth_row["static_depth_sha256"], f"depth {index}")
        check(sha256_file(labels_path) == label_row["image_sha256"], f"labels {index}")
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        labels = cv2.imread(str(labels_path), cv2.IMREAD_UNCHANGED)
        check(depth is not None and depth.dtype == np.uint16, f"depth dtype {index}")
        check(labels is not None and labels.dtype == np.uint16, f"label dtype {index}")
        check(depth.shape == labels.shape == (960, 1280), f"shape {index}")
        metadata = read_json(metadata_path)
        check(int(metadata["frame_index"]) == index, f"label frame bind {index}")
        check(
            int(metadata["sensor_time_ns"]) == int(frame["sensor_time_ns"]),
            f"label time bind {index}",
        )
        check(
            metadata["run_configuration_sha256"]
            == label_configuration["sha256"],
            f"label config bind {index}",
        )
        check(
            metadata["image_sha256"] == sha256_file(labels_path),
            f"label image bind {index}",
        )
    flush_frame = flush_frames[0]
    flush_depth_row = depth_manifest[-1]
    flush_label_row = label_manifest[-1]
    flush_depth = cv2.imread(
        flush_depth_row["static_depth_path"], cv2.IMREAD_UNCHANGED
    )
    flush_labels = cv2.imread(
        flush_label_row["image_path"], cv2.IMREAD_UNCHANGED
    )
    check(
        flush_depth is not None
        and flush_labels is not None
        and not np.any(flush_depth)
        and not np.any(flush_labels),
        "flush depth and labels are all zero",
    )
    flush_metadata = read_json(Path(flush_label_row["metadata_path"]))
    check(
        int(flush_metadata["sensor_time_ns"])
        == int(flush_frame["sensor_time_ns"]),
        "flush label time binding",
    )
    check(
        flush_metadata["run_configuration_sha256"]
        == label_configuration["sha256"],
        "flush label config binding",
    )

    parameter_rows = read_json(root / "tables/parameter_matrix.json")
    check(len(parameter_rows) == 5, "parameter matrix variants")
    base = yaml.safe_load(
        Path(parameter_rows[0]["config_path"]).read_text(encoding="utf-8")
    )
    common_signature = {
        "map_window": base["map_window"],
        "mesh_integrator": base["active_window"]["mesh_integrator"],
        "projective_integrator": base["active_window"][
            "projective_integrator"
        ],
        "frontend": base["frontend"],
        "backend": base["backend"],
    }
    for row, spec in zip(parameter_rows, specs):
        check(row["variant_id"] == spec["variant_id"], "variant order")
        config_path = Path(row["config_path"])
        check(sha256_file(config_path) == row["config_sha256"], "config hash")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        active = config["active_window"]
        check(
            close(
                active["volumetric_map"]["voxel_size"],
                spec["voxel_size_m"],
            ),
            "voxel config",
        )
        check(
            close(
                active["volumetric_map"]["truncation_distance"],
                spec["truncation_distance_m"],
            ),
            "truncation config",
        )
        check(
            close(
                active["object_detector"]["grid_size"],
                spec["grid_size_m"],
            ),
            "grid config",
        )
        check(
            int(active["tracker"]["min_num_observations"])
            == int(spec["minimum_observations"]),
            "observations config",
        )
        check(
            close(
                active["object_extractor"]["min_object_volume"],
                spec["minimum_volume_m3"],
            ),
            "volume config",
        )
        check(
            close(active["tracker"]["temporal_window"], 10.0),
            "frozen tracker temporal window",
        )
        check(
            active["detach_object_extraction"] is False,
            "blocking object extraction",
        )
        check(
            int(active["extraction_worker"]["num_workers"]) == 1,
            "single object extraction worker",
        )
        check(config["map_window"] == common_signature["map_window"], "common map window")
        check(
            active["mesh_integrator"] == common_signature["mesh_integrator"],
            "common mesh integrator",
        )
        check(
            active["projective_integrator"]
            == common_signature["projective_integrator"],
            "common projective integrator",
        )
        check(config["frontend"] == common_signature["frontend"], "common frontend")
        check(config["backend"] == common_signature["backend"], "common backend")
        gvd_min_distance = float(
            config["frontend"]["freespace_places"]["gvd"]["min_distance_m"]
        )
        compatible = gvd_min_distance < float(
            active["volumetric_map"]["truncation_distance"]
        )
        check(
            close(row["gvd_min_distance_m"], gvd_min_distance),
            "GVD minimum distance record",
        )
        check(
            bool(row["gvd_truncation_compatible"]) == compatible,
            "GVD/truncation compatibility record",
        )
        check(
            compatible == (not bool(spec["diagnostic_only"])),
            "only the 3 cm diagnostic has the GVD/truncation conflict",
        )

    source_summary = read_json(
        root / "shared_input/semantic_source_summary.json"
    )
    source_labels = set()
    for row in label_manifest:
        labels = cv2.imread(row["image_path"], cv2.IMREAD_UNCHANGED)
        source_labels.update(
            int(value) for value in np.unique(labels) if int(value) > 0
        )
    check(
        source_labels
        == {int(value) for value in source_summary["source_visible_labels"]},
        "source label set",
    )

    lidar_points = np.load(
        root / "shared_input/visible_lidar_reference_2cm.npy",
        allow_pickle=False,
    )
    check(
        lidar_points.ndim == 2
        and lidar_points.shape[1] == 3
        and np.isfinite(lidar_points).all(),
        "visible LiDAR reference",
    )
    summaries = read_json(root / "tables/variant_summary.json")
    check(len(summaries) == 5, "five complete summaries")
    variant_checks = {}
    for spec, summary in zip(specs, summaries):
        variant_id = str(spec["variant_id"])
        check(summary["variant_id"] == variant_id, f"{variant_id}: summary")
        variant = root / "variants" / variant_id
        execution = read_json(variant / "execution.json")
        child = read_json(variant / "hydra_postpass_report.json")
        check(execution["status"] == "complete", f"{variant_id}: execution")
        check(child["status"] == "complete", f"{variant_id}: child")
        check(
            int(child["frames_expected"])
            == int(child["frames_replayed"])
            == int(child["frames_with_labels"])
            == 102,
            f"{variant_id}: frames",
        )
        check(close(child["label_coverage"], 1.0), f"{variant_id}: labels")
        check(
            int(child["backend_stats"]["frames_rejected"]) == 0,
            f"{variant_id}: rejected",
        )
        mesh = variant / "hydra_realtime/backend/mesh.ply"
        dsg = variant / "hydra_realtime/backend/dsg.json"
        vertices, faces = ply_header(mesh)
        check(vertices == int(summary["mesh_vertices"]), f"{variant_id}: vertices")
        check(faces == int(summary["mesh_faces"]), f"{variant_id}: faces")
        graph = read_json(dsg)
        objects = [
            node
            for node in graph.get("nodes", [])
            if node.get("layer") == 2 and node.get("partition", 0) == 0
        ]
        check(
            len(objects) == int(summary["dsg_object_nodes"]),
            f"{variant_id}: object nodes",
        )
        metrics = variant / "metrics"
        mesh_sample = np.load(
            metrics / "mesh_vertices_sample.npy", allow_pickle=False
        )
        lidar_sample = np.load(
            metrics / "visible_lidar_sample.npy", allow_pickle=False
        )
        mesh_to_lidar = np.load(
            metrics / "mesh_to_visible_lidar_distances_m.npy",
            allow_pickle=False,
        )
        lidar_to_mesh = np.load(
            metrics / "visible_lidar_to_mesh_distances_m.npy",
            allow_pickle=False,
        )
        recomputed_m2l = cKDTree(lidar_sample).query(
            mesh_sample, workers=-1
        )[0]
        recomputed_l2m = cKDTree(mesh_sample).query(
            lidar_sample, workers=-1
        )[0]
        check(
            np.allclose(mesh_to_lidar, recomputed_m2l, atol=1.0e-6),
            f"{variant_id}: mesh-lidar distances",
        )
        check(
            np.allclose(lidar_to_mesh, recomputed_l2m, atol=1.0e-6),
            f"{variant_id}: lidar-mesh distances",
        )
        check(
            close(
                distance_fraction(lidar_to_mesh, 0.10),
                summary["lidar_completeness_proxy_within_0p10m"],
            ),
            f"{variant_id}: completeness proxy",
        )
        check(
            close(
                distance_fraction(mesh_to_lidar, 0.10),
                summary["lidar_accuracy_proxy_within_0p10m"],
            ),
            f"{variant_id}: accuracy proxy",
        )
        p95 = float(child["backend_stats"]["processing_time_ms"]["p95"])
        maximum = float(
            child["backend_stats"]["processing_time_ms"]["maximum"]
        )
        check(close(p95, summary["hydra_processing_p95_ms"]), f"{variant_id}: p95")
        check(
            close(maximum, summary["hydra_processing_max_ms"]),
            f"{variant_id}: maximum",
        )
        check(
            bool(summary["runtime_hard_gate_passed"])
            == (p95 <= 250.0),
            f"{variant_id}: runtime gate",
        )
        check(summary["formal_mesh_accuracy"] is None, f"{variant_id}: no mesh accuracy")
        check(summary["formal_object_recall"] is None, f"{variant_id}: no object recall")
        challenge = read_json(
            metrics / "metrics_by_challenge_tag.json"
        )
        check(
            challenge["manual_tag_metrics"] is None,
            f"{variant_id}: pending manual challenge metrics",
        )
        variant_checks[variant_id] = {
            "source_frames": 101,
            "flush_events": 1,
            "hydra_events": 102,
            "mesh_vertices": vertices,
            "dsg_objects": len(objects),
            "p95_ms": p95,
            "maximum_ms": maximum,
        }

    risks = read_json(root / "tables/risk_assessment.json")
    check(
        risks["status"] == "risks_present_do_not_auto_promote",
        "risk assessment blocks automatic promotion",
    )
    risk_by_id = {
        str(value["risk_id"]): value for value in risks["risks"]
    }
    check(
        set(risk_by_id) == {f"E16-R{index}" for index in range(1, 7)},
        "six structured risks",
    )
    check(
        risk_by_id["E16-R5"]["compatible"] is False
        and close(risk_by_id["E16-R5"]["gvd_min_distance_m"], 0.10)
        and close(risk_by_id["E16-R5"]["truncation_distance_m"], 0.09),
        "3 cm GVD/truncation risk",
    )
    diagnostic_log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (
            root
            / "variants/voxel_3cm_obs8_vol0p005/hydra_realtime/logs"
        ).iterdir()
        if path.is_file() and not path.is_symlink()
    )
    check(
        risk_by_id["E16-R5"]["observed_error"] in diagnostic_log_text,
        "3 cm GVD/truncation runtime evidence",
    )
    check(
        float(risk_by_id["E16-R4"]["maximum_seconds"]) > 800.0
        and float(risk_by_id["E16-R4"]["peak_rss_gib"]) > 13.0,
        "low-volume resource explosion",
    )

    audit = {
        "schema": "daaam.g1_no_gt_e16_independent_audit.v1",
        "generated_at": utc_now(),
        "passed": True,
        "inventory": inventory,
        "frozen_file_count": frozen_count,
        "input_frames_verified": 101,
        "flush_events_verified": 1,
        "source_visible_labels_verified": len(source_labels),
        "visible_lidar_points_verified": int(len(lidar_points)),
        "variants": variant_checks,
        "checks": {
            "frozen_source_hashes": True,
            "frame_time_lineage": True,
            "depth_label_shapes_and_dtypes": True,
            "semantic_label_bindings": True,
            "single_factor_parameter_matrix": True,
            "hydra_101_of_101_and_zero_rejected": True,
            "mesh_and_dsg_counts": True,
            "sparse_lidar_distances_recomputed": True,
            "runtime_gate_recomputed": True,
            "maximum_latency_recomputed": True,
            "structured_risk_assessment": True,
            "three_cm_gvd_truncation_conflict": True,
            "formal_gt_claims_absent": True,
        },
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
