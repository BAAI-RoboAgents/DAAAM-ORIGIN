#!/usr/bin/env python3
"""Run the GT-free E16 Hydra geometry/object parameter experiment."""

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

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.quality import analyze_ascii_ply_mesh  # noqa: E402
from daaam.realtime.semantic_labels import persist_semantic_label  # noqa: E402


EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments/g1_20260724_473_573_v1_1"
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e16_safe035_hydra_20260730"
)
DEFAULT_E13_RUN = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e13_e14_safety_ablation_20260730"
)
DEFAULT_E14_RUN = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e14_safe035_unique_obs8_seed0_20260730"
)
DEFAULT_E15_RUN = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e15_safe035_increment_20260730"
)
DEFAULT_GEOMETRY_RUN = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e13_e12fed_mapmemory_20260729"
)
DEFAULT_PREPARED = (
    EXPERIMENT_ROOT
    / "shared_artifacts/prepared_stereo_473_573_v1_v2"
)
DEFAULT_DEPTH = (
    EXPERIMENT_ROOT
    / "shared_artifacts/e13_metric_depth_473_573_20260729"
)
DEFAULT_LIDAR = (
    EXPERIMENT_ROOT
    / "shared_artifacts/lidar_camera_473_573"
)
INVENTORY_EXCLUDES = {
    "artifact_inventory.csv",
    "artifact_inventory.jsonl",
    "inventory_summary.json",
    "COMPLETION.json",
}
DISTANCE_THRESHOLDS_M = (0.05, 0.10, 0.20, 0.30)
RUNTIME_HARD_GATE_MS = 250.0
FRAME_COUNT = 101
FLUSH_EVENT_COUNT = 1
HYDRA_EVENT_COUNT = FRAME_COUNT + FLUSH_EVENT_COUNT
FLUSH_AFTER_LAST_FRAME_SECONDS = 10.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--e13-run", type=Path, default=DEFAULT_E13_RUN)
    parser.add_argument("--e14-run", type=Path, default=DEFAULT_E14_RUN)
    parser.add_argument("--e15-run", type=Path, default=DEFAULT_E15_RUN)
    parser.add_argument("--geometry-run", type=Path, default=DEFAULT_GEOMETRY_RUN)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--depth", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--lidar", type=Path, default=DEFAULT_LIDAR)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reseal-existing", action="store_true")
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frozen_reference(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
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
    write_json(
        root / "inventory_summary.json",
        {
            "schema": "daaam.g1_no_gt_e16_inventory.v1",
            "generated_at": utc_now(),
            "file_count": len(rows),
            "total_bytes": sum(int(row["size_bytes"]) for row in rows),
            "inventory_root_sha256": root_hash,
        },
    )
    audited = status == "complete_independently_audited"
    write_json(
        root / "COMPLETION.json",
        {
            "schema": "daaam.g1_no_gt_e16_completion.v1",
            "status": status,
            "generated_at": utc_now(),
            "artifact_inventory_root_sha256": root_hash,
            "artifact_inventory_file_count": len(rows),
            "formal_claims_permitted": False,
            "formal_object_recall_available": False,
            "absolute_mesh_ground_truth_available": False,
            "independent_audit": "passed" if audited else "pending",
        },
    )


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return value


def variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "voxel_12cm_obs8_vol0p005",
            "axis": "voxel",
            "voxel_size_m": 0.12,
            "truncation_distance_m": 0.36,
            "grid_size_m": 0.12,
            "minimum_observations": 8,
            "minimum_volume_m3": 0.005,
            "diagnostic_only": False,
        },
        {
            "variant_id": "voxel_5cm_obs8_vol0p005",
            "axis": "voxel",
            "voxel_size_m": 0.05,
            "truncation_distance_m": 0.15,
            "grid_size_m": 0.05,
            "minimum_observations": 8,
            "minimum_volume_m3": 0.005,
            "diagnostic_only": False,
        },
        {
            "variant_id": "voxel_5cm_obs4_vol0p005",
            "axis": "minimum_observations",
            "voxel_size_m": 0.05,
            "truncation_distance_m": 0.15,
            "grid_size_m": 0.05,
            "minimum_observations": 4,
            "minimum_volume_m3": 0.005,
            "diagnostic_only": False,
        },
        {
            "variant_id": "voxel_5cm_obs8_vol0p0001",
            "axis": "minimum_volume",
            "voxel_size_m": 0.05,
            "truncation_distance_m": 0.15,
            "grid_size_m": 0.05,
            "minimum_observations": 8,
            "minimum_volume_m3": 0.0001,
            "diagnostic_only": False,
        },
        {
            "variant_id": "voxel_3cm_obs8_vol0p005",
            "axis": "voxel",
            "voxel_size_m": 0.03,
            "truncation_distance_m": 0.09,
            "grid_size_m": 0.03,
            "minimum_observations": 8,
            "minimum_volume_m3": 0.005,
            "diagnostic_only": True,
        },
    ]


def build_variant_config(
    base: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    active = config["active_window"]
    active["detach_object_extraction"] = False
    active["extraction_worker"] = {
        "num_workers": 1,
        "poll_time_us": 1000,
        "verbosity": 0,
    }
    active["volumetric_map"]["voxel_size"] = float(spec["voxel_size_m"])
    active["volumetric_map"]["truncation_distance"] = float(
        spec["truncation_distance_m"]
    )
    active["object_detector"]["grid_size"] = float(spec["grid_size_m"])
    active["tracker"]["min_num_observations"] = int(
        spec["minimum_observations"]
    )
    active["object_extractor"]["min_object_volume"] = float(
        spec["minimum_volume_m3"]
    )
    return config


def parse_time_v(path: Path) -> dict[str, Any]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()

    def number(key: str) -> float | None:
        value = values.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    return {
        "maximum_resident_set_size_kib": number(
            "Maximum resident set size (kbytes)"
        ),
        "user_time_seconds": number("User time (seconds)"),
        "system_time_seconds": number("System time (seconds)"),
        "cpu_percent": values.get("Percent of CPU this job got"),
        "elapsed_wall_clock": values.get(
            "Elapsed (wall clock) time (h:mm:ss or m:ss)"
        ),
        "major_page_faults": number(
            "Major (requiring I/O) page faults"
        ),
        "minor_page_faults": number(
            "Minor (reclaiming a frame) page faults"
        ),
        "file_system_inputs": number("File system inputs"),
        "file_system_outputs": number("File system outputs"),
        "exit_status": number("Exit status"),
    }


def load_ascii_ply_vertices(path: Path) -> np.ndarray:
    vertex_count: int | None = None
    with path.open("r", encoding="ascii") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"incomplete PLY header: {path}")
            fields = line.split()
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
            if fields and fields[0] == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"missing PLY vertex count: {path}")
        vertices = np.empty((vertex_count, 3), dtype=np.float32)
        for index in range(vertex_count):
            fields = stream.readline().split()
            if len(fields) < 3:
                raise ValueError(f"incomplete PLY vertex {index}: {path}")
            vertices[index] = [float(value) for value in fields[:3]]
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"non-finite PLY vertices: {path}")
    return vertices


def deterministic_sample(array: np.ndarray, maximum: int) -> np.ndarray:
    if len(array) <= maximum:
        return array
    indices = np.linspace(0, len(array) - 1, maximum, dtype=np.int64)
    return array[indices]


def distance_summary(distances: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": int(len(distances)),
        "mean_m": float(np.mean(distances)),
        "p50_m": float(np.percentile(distances, 50)),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "maximum_m": float(np.max(distances)),
        "within_threshold_fraction": {},
    }
    for threshold in DISTANCE_THRESHOLDS_M:
        result["within_threshold_fraction"][f"{threshold:.2f}"] = float(
            np.mean(distances <= threshold)
        )
    return result


def sparse_lidar_agreement(
    mesh_vertices: np.ndarray,
    lidar_points: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    mesh_sample = deterministic_sample(mesh_vertices, 500_000)
    lidar_sample = deterministic_sample(lidar_points, 250_000)
    lidar_tree = cKDTree(lidar_sample)
    mesh_tree = cKDTree(mesh_sample)
    mesh_to_lidar = lidar_tree.query(mesh_sample, workers=-1)[0].astype(
        np.float32
    )
    lidar_to_mesh = mesh_tree.query(lidar_sample, workers=-1)[0].astype(
        np.float32
    )
    np.save(output / "mesh_vertices_sample.npy", mesh_sample, allow_pickle=False)
    np.save(output / "visible_lidar_sample.npy", lidar_sample, allow_pickle=False)
    np.save(
        output / "mesh_to_visible_lidar_distances_m.npy",
        mesh_to_lidar,
        allow_pickle=False,
    )
    np.save(
        output / "visible_lidar_to_mesh_distances_m.npy",
        lidar_to_mesh,
        allow_pickle=False,
    )
    return {
        "schema": "daaam.g1_no_gt_e16_sparse_lidar_agreement.v1",
        "reference_scope": (
            "LiDAR returns projected inside the left rectified camera image; "
            "2 cm voxel-deduplicated in map coordinates"
        ),
        "mesh_sampling": (
            "all vertices when <=500000, otherwise deterministic linear-index sample"
        ),
        "lidar_sampling": (
            "all reference points when <=250000, otherwise deterministic "
            "linear-index sample"
        ),
        "mesh_vertices_total": int(len(mesh_vertices)),
        "mesh_vertices_sampled": int(len(mesh_sample)),
        "visible_lidar_points_total": int(len(lidar_points)),
        "visible_lidar_points_sampled": int(len(lidar_sample)),
        "accuracy_proxy_mesh_to_sparse_lidar": distance_summary(mesh_to_lidar),
        "completeness_proxy_sparse_lidar_to_mesh": distance_summary(
            lidar_to_mesh
        ),
        "formal_mesh_accuracy": None,
        "formal_mesh_completeness": None,
        "limitation": (
            "nearest-vertex symmetric distance is a sparse cross-sensor "
            "agreement proxy, not point-to-surface GT; unobserved backsides "
            "and occlusions are not scored"
        ),
    }


def object_metrics(
    dsg_path: Path,
    source_labels: set[int],
    named_labels: set[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    graph = read_json(dsg_path)
    objects = [
        node
        for node in graph.get("nodes", [])
        if node.get("layer") == 2 and node.get("partition", 0) == 0
    ]
    rows = []
    for node in objects:
        attributes = node.get("attributes") or {}
        mesh = attributes.get("mesh") or {}
        label = int(attributes.get("semantic_label", -1))
        box = attributes.get("bounding_box") or {}
        dimensions = box.get("dimensions") or [None, None, None]
        rows.append(
            {
                "node_id": int(node["id"]),
                "semantic_label": label,
                "mesh_points": len(mesh.get("points") or []),
                "mesh_faces": len(mesh.get("faces") or []),
                "position_json": json.dumps(attributes.get("position")),
                "dimensions_json": json.dumps(dimensions),
                "source_label_present": label in source_labels,
                "named_label_present": label in named_labels,
            }
        )
    object_labels = {int(row["semantic_label"]) for row in rows}
    positive_object_labels = {value for value in object_labels if value > 0}
    source_overlap = source_labels & positive_object_labels
    named_overlap = named_labels & positive_object_labels
    return (
        {
            "schema": "daaam.g1_no_gt_e16_object_metrics.v1",
            "dsg_object_nodes": len(objects),
            "dsg_object_nodes_with_mesh": sum(
                int(row["mesh_points"]) > 0 for row in rows
            ),
            "unique_positive_dsg_semantic_labels": len(
                positive_object_labels
            ),
            "source_visible_semantic_labels": len(source_labels),
            "source_labels_surviving_to_dsg": len(source_overlap),
            "semantic_label_survival_proxy": (
                len(source_overlap) / len(source_labels)
                if source_labels
                else None
            ),
            "named_source_labels": len(named_labels),
            "named_labels_surviving_to_dsg": len(named_overlap),
            "named_label_survival_proxy": (
                len(named_overlap) / len(named_labels)
                if named_labels
                else None
            ),
            "missing_source_labels": sorted(source_labels - source_overlap),
            "missing_named_labels": sorted(named_labels - named_overlap),
            "duplicate_object_node_count": (
                len(objects) - len(positive_object_labels)
            ),
            "formal_object_recall": None,
            "evaluation_basis": (
                "upstream E13 IDs are tested outputs, not object GT; survival "
                "is a pipeline proxy and cannot be called recall"
            ),
        },
        rows,
    )


def read_timing_stream(path: Path) -> list[dict[str, float]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            elapsed = float(row["elapsed(s)"])
            if elapsed >= 0.0 and np.isfinite(elapsed):
                rows.append(
                    {
                        "sensor_time_ns": int(row["timestamp(ns)"]),
                        "elapsed_ms": elapsed * 1000.0,
                    }
                )
    return rows


def distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"samples": 0}
    return {
        "samples": int(len(array)),
        "mean_ms": float(np.mean(array)),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "maximum_ms": float(np.max(array)),
    }


def block_bootstrap_mean(
    values: Sequence[float],
    *,
    block_length: int = 5,
    repetitions: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"samples": 0}
    rng = np.random.default_rng(seed)
    blocks = [
        array[start : min(start + block_length, len(array))]
        for start in range(0, len(array), block_length)
    ]
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        sample = np.concatenate([blocks[value] for value in selected])[: len(array)]
        means[index] = np.mean(sample)
    return {
        "samples": int(len(array)),
        "block_length": block_length,
        "repetitions": repetitions,
        "seed": seed,
        "mean_ms": float(np.mean(array)),
        "ci95_low_ms": float(np.percentile(means, 2.5)),
        "ci95_high_ms": float(np.percentile(means, 97.5)),
    }


def grouped_timing_metrics(
    rows: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
    challenges: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame_by_time = {
        int(frame["sensor_time_ns"]): frame for frame in frames
    }
    split_values: dict[str, list[float]] = {}
    tag_values: dict[str, list[float]] = {}
    unmatched = 0
    for row in rows:
        frame = frame_by_time.get(int(row["sensor_time_ns"]))
        if frame is None:
            unmatched += 1
            continue
        if bool(frame.get("is_flush_event", False)):
            continue
        source = int(frame["source_frame_index"])
        challenge = challenges[source]
        split = str(challenge["split"])
        split_values.setdefault(split, []).append(float(row["elapsed_ms"]))
        for tag in challenge.get("automatic_tags", []):
            tag_values.setdefault(str(tag), []).append(float(row["elapsed_ms"]))
    by_split = {
        key: {
            **distribution(values),
            "block_bootstrap_mean": block_bootstrap_mean(values),
        }
        for key, values in sorted(split_values.items())
    }
    by_tag = {
        key: {
            **distribution(values),
            "block_bootstrap_mean": block_bootstrap_mean(values),
        }
        for key, values in sorted(tag_values.items())
    }
    manual_statuses = sorted(
        {
            str(frame["manual_tags"]["status"])
            for frame in challenges.values()
        }
    )
    return (
        {
            "schema": "daaam.g1_no_gt_e16_metrics_by_split.v1",
            "metric": "Hydra active_window/all elapsed time",
            "unmatched_timing_rows": unmatched,
            "splits": by_split,
        },
        {
            "schema": "daaam.g1_no_gt_e16_metrics_by_challenge_tag.v1",
            "metric": "Hydra active_window/all elapsed time",
            "automatic_tags": by_tag,
            "manual_tag_statuses": manual_statuses,
            "manual_tag_metrics": None,
            "limitation": (
                "manual plant/thin/occlusion/density tags remain pending human "
                "review; a global fused mesh cannot be decomposed into per-frame "
                "geometry accuracy without a separately frozen surface GT"
            ),
        },
    )


def write_visible_lidar_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("end_header\n")
        for point in points:
            stream.write(
                f"{float(point[0]):.7g} {float(point[1]):.7g} "
                f"{float(point[2]):.7g}\n"
            )


def voxel_deduplicate(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    quantized = np.rint(points / voxel_size_m).astype(np.int64)
    _, indices = np.unique(quantized, axis=0, return_index=True)
    return points[np.sort(indices)].astype(np.float32, copy=False)


def prepare_inputs(
    args: argparse.Namespace,
    output: Path,
    paths: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], str, set[int], set[int]]:
    input_root = output / "shared_input"
    static_depth = input_root / "static_depth"
    label_frames = input_root / "label_frames"
    static_depth.mkdir(parents=True, exist_ok=True)
    label_frames.mkdir(parents=True, exist_ok=True)
    frames_source = read_jsonl(paths["e12_frames"])
    if len(frames_source) != FRAME_COUNT:
        raise ValueError(f"expected {FRAME_COUNT} E12 frames")
    poses = np.loadtxt(paths["poses"], dtype=np.float64).reshape(-1, 4, 4)
    if len(poses) != FRAME_COUNT:
        raise ValueError(f"expected {FRAME_COUNT} poses")
    camera = read_json(paths["camera_info"])
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("camera intrinsics must be 3x3")

    source_manifest = []
    frames = []
    for index, source in enumerate(frames_source):
        source_frame = int(source["source_frame_index"])
        rgb = Path(source["rgb_path"]).resolve()
        depth_npy = (
            args.geometry_run.resolve()
            / "geometry_input/scaled_depth_meter"
            / f"{index:08d}.npy"
        )
        entity_dir = (
            args.e13_run.resolve()
            / "variants/safe_merge_0p35m/frames"
            / f"{index:08d}"
        )
        entity_map = entity_dir / "entity_id_map.png"
        entity_frame = entity_dir / "frame.json"
        lidar_npz = (
            args.lidar.resolve()
            / "correspondences"
            / f"{source_frame:06d}.npz"
        )
        required = (rgb, depth_npy, entity_map, entity_frame, lidar_npz)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        metadata = read_json(entity_frame)
        if int(metadata["frame_index"]) != index:
            raise ValueError(f"E13 frame mismatch: {entity_frame}")
        if int(metadata["source_frame_index"]) != source_frame:
            raise ValueError(f"E13 source frame mismatch: {entity_frame}")
        if int(metadata["sensor_time_ns"]) != int(source["sensor_time_ns"]):
            raise ValueError(f"E13 sensor time mismatch: {entity_frame}")
        source_manifest.append(
            {
                "frame_index": index,
                "source_frame_index": source_frame,
                "sensor_time_ns": int(source["sensor_time_ns"]),
                "rgb": frozen_reference(rgb),
                "scaled_depth_meter": frozen_reference(depth_npy),
                "entity_id_map": frozen_reference(entity_map),
                "entity_frame": frozen_reference(entity_frame),
                "visible_lidar_correspondence": frozen_reference(lidar_npz),
            }
        )
        frames.append(
            {
                "frame_index": index,
                "source_frame_index": source_frame,
                "sensor_time_ns": int(source["sensor_time_ns"]),
                "rgb_path": str(rgb),
                "world_T_camera": poses[index].tolist(),
                "intrinsics": intrinsics.tolist(),
            }
        )
    write_jsonl(input_root / "source_frame_manifest.jsonl", source_manifest)
    label_configuration = {
        "schema": "daaam.g1_no_gt_e16_label_configuration.v1",
        "E11": "conf=0.3, area=300, IoU=0.5",
        "E12": "BotSort buffer=10",
        "E13": "safe_merge_0p35m",
        "frame_manifest_sha256": sha256_file(
            input_root / "source_frame_manifest.jsonl"
        ),
        "flush_policy": (
            "one zero-depth/zero-label event 10.1 seconds after the final "
            "source frame; excluded from source/split/challenge metrics"
        ),
    }
    label_configuration_sha256 = canonical_sha256(label_configuration)
    label_configuration["sha256"] = label_configuration_sha256
    write_json(input_root / "label_configuration.json", label_configuration)

    source_labels: set[int] = set()
    depth_rows = []
    label_rows = []
    for record in source_manifest:
        index = int(record["frame_index"])
        depth = np.load(
            record["scaled_depth_meter"]["path"], allow_pickle=False
        ).astype(np.float32, copy=False)
        labels = cv2.imread(
            record["entity_id_map"]["path"], cv2.IMREAD_UNCHANGED
        )
        rgb = cv2.imread(record["rgb"]["path"], cv2.IMREAD_COLOR)
        if rgb is None or labels is None:
            raise FileNotFoundError(f"failed to decode frame {index}")
        if depth.shape != rgb.shape[:2] or labels.shape != depth.shape:
            raise ValueError(f"RGB/depth/label shape mismatch at frame {index}")
        valid = np.isfinite(depth) & (depth >= 0.1) & (depth <= 8.0)
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
        depth_mm[valid] = np.rint(depth[valid] * 1000.0).astype(np.uint16)
        depth_path = static_depth / f"{index:08d}.png"
        if not cv2.imwrite(str(depth_path), depth_mm):
            raise OSError(depth_path)
        label_record = persist_semantic_label(
            label_frames,
            index,
            labels,
            sensor_time_ns=int(record["sensor_time_ns"]),
            run_configuration_sha256=label_configuration_sha256,
        )
        source_labels.update(
            int(value) for value in np.unique(labels) if int(value) > 0
        )
        depth_rows.append(
            {
                "frame_index": index,
                "source_frame_index": int(record["source_frame_index"]),
                "valid_pixels": int(np.count_nonzero(valid)),
                "valid_fraction": float(np.mean(valid)),
                "minimum_valid_m": float(np.min(depth[valid])),
                "maximum_valid_m": float(np.max(depth[valid])),
                "static_depth_path": str(depth_path.resolve()),
                "static_depth_sha256": sha256_file(depth_path),
                "storage": "uint16 millimetres; zero invalid",
                "dynamic_isolation": (
                    "not available: all valid frozen metric depth retained"
                ),
            }
        )
        label_rows.append(
            {
                "frame_index": index,
                "source_frame_index": int(record["source_frame_index"]),
                "sensor_time_ns": int(record["sensor_time_ns"]),
                "nonzero_pixels": int(label_record["nonzero_pixels"]),
                "minimum_label": int(label_record["minimum_label"]),
                "maximum_label": int(label_record["maximum_label"]),
                "image_path": str(label_record["path"]),
                "image_sha256": str(label_record["sha256"]),
                "metadata_path": str(label_record["metadata_path"]),
            }
        )
    flush_index = FRAME_COUNT
    flush_sensor_time_ns = int(frames[-1]["sensor_time_ns"]) + int(
        FLUSH_AFTER_LAST_FRAME_SECONDS * 1.0e9
    )
    flush_shape = (int(camera["height"]), int(camera["width"]))
    flush_depth = np.zeros(flush_shape, dtype=np.uint16)
    flush_labels = np.zeros(flush_shape, dtype=np.uint16)
    flush_depth_path = static_depth / f"{flush_index:08d}.png"
    if not cv2.imwrite(str(flush_depth_path), flush_depth):
        raise OSError(flush_depth_path)
    flush_label_record = persist_semantic_label(
        label_frames,
        flush_index,
        flush_labels,
        sensor_time_ns=flush_sensor_time_ns,
        run_configuration_sha256=label_configuration_sha256,
    )
    depth_rows.append(
        {
            "frame_index": flush_index,
            "source_frame_index": None,
            "valid_pixels": 0,
            "valid_fraction": 0.0,
            "minimum_valid_m": None,
            "maximum_valid_m": None,
            "static_depth_path": str(flush_depth_path.resolve()),
            "static_depth_sha256": sha256_file(flush_depth_path),
            "storage": "uint16 millimetres; all zero",
            "dynamic_isolation": "not applicable: synthetic flush event",
            "is_flush_event": True,
        }
    )
    label_rows.append(
        {
            "frame_index": flush_index,
            "source_frame_index": None,
            "sensor_time_ns": flush_sensor_time_ns,
            "nonzero_pixels": 0,
            "minimum_label": int(flush_label_record["minimum_label"]),
            "maximum_label": int(flush_label_record["maximum_label"]),
            "image_path": str(flush_label_record["path"]),
            "image_sha256": str(flush_label_record["sha256"]),
            "metadata_path": str(flush_label_record["metadata_path"]),
            "is_flush_event": True,
        }
    )
    frames.append(
        {
            "frame_index": flush_index,
            "source_frame_index": None,
            "sensor_time_ns": flush_sensor_time_ns,
            "rgb_path": frames[-1]["rgb_path"],
            "world_T_camera": frames[-1]["world_T_camera"],
            "intrinsics": intrinsics.tolist(),
            "is_flush_event": True,
            "flush_after_last_source_frame_seconds": (
                FLUSH_AFTER_LAST_FRAME_SECONDS
            ),
        }
    )
    write_jsonl(input_root / "static_depth_manifest.jsonl", depth_rows)
    write_jsonl(input_root / "semantic_label_manifest.jsonl", label_rows)
    write_jsonl(input_root / "frames.jsonl", frames)

    lidar_points = []
    for record in source_manifest:
        with np.load(
            record["visible_lidar_correspondence"]["path"],
            allow_pickle=False,
        ) as archive:
            points = archive["map_points_m"].astype(np.float32, copy=False)
        points = points[np.all(np.isfinite(points), axis=1)]
        lidar_points.append(points)
    lidar_raw = np.concatenate(lidar_points, axis=0)
    lidar_reference = voxel_deduplicate(lidar_raw, 0.02)
    np.save(
        input_root / "visible_lidar_reference_2cm.npy",
        lidar_reference,
        allow_pickle=False,
    )
    write_visible_lidar_ply(
        input_root / "visible_lidar_reference_2cm.ply",
        lidar_reference,
    )
    write_json(
        input_root / "visible_lidar_reference.json",
        {
            "schema": "daaam.g1_no_gt_e16_visible_lidar_reference.v1",
            "coordinate_frame": "map",
            "source": (
                "per-frame LiDAR points projected inside the rectified left "
                "camera image"
            ),
            "source_frame_range": [473, 573],
            "raw_point_count_with_repetition": int(len(lidar_raw)),
            "finite_point_count_with_repetition": int(len(lidar_raw)),
            "voxel_deduplication_m": 0.02,
            "deduplicated_point_count": int(len(lidar_reference)),
            "npy_sha256": sha256_file(
                input_root / "visible_lidar_reference_2cm.npy"
            ),
            "formal_surface_ground_truth": False,
        },
    )
    final_labels = read_jsonl(paths["e14_labels"])
    named_labels = {
        int(row["entity_ordinal"])
        for row in final_labels
        if int(row["entity_ordinal"]) in source_labels
    }
    write_json(
        input_root / "semantic_source_summary.json",
        {
            "source_entity_count": 169,
            "source_labels_visible_in_id_maps": len(source_labels),
            "source_visible_labels": sorted(source_labels),
            "E14_named_entity_count": len(final_labels),
            "E14_named_labels_visible_in_id_maps": len(named_labels),
            "E14_named_visible_labels": sorted(named_labels),
        },
    )
    return frames, label_configuration_sha256, source_labels, named_labels


def load_prepared_inputs(
    output: Path,
) -> tuple[list[dict[str, Any]], str, set[int], set[int]]:
    input_root = output / "shared_input"
    frames = read_jsonl(input_root / "frames.jsonl")
    label_configuration = read_json(
        input_root / "label_configuration.json"
    )
    semantic = read_json(input_root / "semantic_source_summary.json")
    return (
        frames,
        str(label_configuration["sha256"]),
        {int(value) for value in semantic["source_visible_labels"]},
        {int(value) for value in semantic["E14_named_visible_labels"]},
    )


def render_lidar_reference(points: np.ndarray, output: Path) -> None:
    sample = deterministic_sample(points, 150_000)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for axis, dims, title in (
        (axes[0], (0, 1), "Visible LiDAR reference: X/Y"),
        (axes[1], (0, 2), "Visible LiDAR reference: X/Z"),
    ):
        axis.scatter(
            sample[:, dims[0]],
            sample[:, dims[1]],
            s=0.3,
            alpha=0.5,
            color="#3182bd",
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("XYZ"[dims[0]] + " (m)")
        axis.set_ylabel("XYZ"[dims[1]] + " (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"Camera-visible LiDAR proxy reference — {len(points):,} points"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze_variant(
    variant: Path,
    spec: Mapping[str, Any],
    source_labels: set[int],
    named_labels: set[int],
    lidar_points: np.ndarray,
    frames: Sequence[Mapping[str, Any]],
    challenges: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    metrics_dir = variant / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    child = read_json(variant / "hydra_postpass_report.json")
    mesh_path = variant / "hydra_realtime/backend/mesh.ply"
    dsg_path = variant / "hydra_realtime/backend/dsg.json"
    mesh_quality = analyze_ascii_ply_mesh(mesh_path)
    write_json(metrics_dir / "mesh_quality.json", mesh_quality)
    mesh_vertices = load_ascii_ply_vertices(mesh_path)
    lidar_metrics = sparse_lidar_agreement(
        mesh_vertices, lidar_points, metrics_dir
    )
    write_json(metrics_dir / "sparse_lidar_agreement.json", lidar_metrics)
    objects, object_rows = object_metrics(
        dsg_path, source_labels, named_labels
    )
    write_json(metrics_dir / "object_metrics.json", objects)
    write_jsonl(metrics_dir / "object_nodes.jsonl", object_rows)
    write_csv(metrics_dir / "object_nodes.csv", object_rows)
    timing_rows = read_timing_stream(
        variant
        / "hydra_realtime/timing/active_window_all_timing_raw.csv"
    )
    timing = distribution([row["elapsed_ms"] for row in timing_rows])
    timing["block_bootstrap_mean"] = block_bootstrap_mean(
        [row["elapsed_ms"] for row in timing_rows]
    )
    write_json(metrics_dir / "active_window_timing.json", timing)
    by_split, by_challenge = grouped_timing_metrics(
        timing_rows, frames, challenges
    )
    write_json(metrics_dir / "metrics_by_split.json", by_split)
    write_json(
        metrics_dir / "metrics_by_challenge_tag.json", by_challenge
    )
    resource = parse_time_v(variant / "resource_usage_time_v.txt")
    write_json(metrics_dir / "resource_metrics.json", resource)
    backend = child["backend_stats"]
    processing = backend.get("processing_time_ms") or {}
    p95_ms = float(processing.get("p95", np.inf))
    summary = {
        "schema": "daaam.g1_no_gt_e16_variant_summary.v1",
        **dict(spec),
        "status": "complete",
        "frames_replayed": int(child["frames_replayed"]),
        "label_coverage": float(child["label_coverage"]),
        "mesh_vertices": int(mesh_quality["vertices"]),
        "mesh_faces": int(mesh_quality["faces"]),
        "mesh_surface_area_m2": float(mesh_quality["surface_area_m2"]),
        "mesh_connected_components": int(
            mesh_quality["connected_components"]
        ),
        "mesh_largest_component_area_ratio": float(
            mesh_quality["largest_component_area_ratio"]
        ),
        "mesh_tiny_component_area_ratio": float(
            mesh_quality["tiny_component_area_ratio"]
        ),
        "lidar_completeness_proxy_within_0p10m": float(
            lidar_metrics["completeness_proxy_sparse_lidar_to_mesh"][
                "within_threshold_fraction"
            ]["0.10"]
        ),
        "lidar_completeness_proxy_p50_m": float(
            lidar_metrics["completeness_proxy_sparse_lidar_to_mesh"][
                "p50_m"
            ]
        ),
        "lidar_accuracy_proxy_within_0p10m": float(
            lidar_metrics["accuracy_proxy_mesh_to_sparse_lidar"][
                "within_threshold_fraction"
            ]["0.10"]
        ),
        "lidar_accuracy_proxy_p50_m": float(
            lidar_metrics["accuracy_proxy_mesh_to_sparse_lidar"]["p50_m"]
        ),
        "dsg_object_nodes": int(objects["dsg_object_nodes"]),
        "unique_dsg_semantic_labels": int(
            objects["unique_positive_dsg_semantic_labels"]
        ),
        "semantic_label_survival_proxy": float(
            objects["semantic_label_survival_proxy"]
        ),
        "named_label_survival_proxy": float(
            objects["named_label_survival_proxy"]
        ),
        "hydra_processing_p50_ms": float(processing.get("p50", np.nan)),
        "hydra_processing_p95_ms": p95_ms,
        "hydra_processing_max_ms": float(
            processing.get("maximum", np.nan)
        ),
        "wall_elapsed_seconds": float(child["elapsed_seconds"]),
        "peak_rss_mib": (
            float(resource["maximum_resident_set_size_kib"]) / 1024.0
            if resource["maximum_resident_set_size_kib"] is not None
            else None
        ),
        "output_bytes": sum(
            path.stat().st_size
            for path in (variant / "hydra_realtime").rglob("*")
            if path.is_file()
        ),
        "runtime_hard_gate_ms": RUNTIME_HARD_GATE_MS,
        "runtime_hard_gate_passed": p95_ms <= RUNTIME_HARD_GATE_MS,
        "winner_eligible": (
            not bool(spec["diagnostic_only"])
            and p95_ms <= RUNTIME_HARD_GATE_MS
        ),
        "formal_mesh_accuracy": None,
        "formal_mesh_completeness": None,
        "formal_object_recall": None,
    }
    write_json(variant / "variant_summary.json", summary)
    return summary


def run_variant(
    output: Path,
    spec: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
    label_configuration_sha256: str,
    timeout_seconds: int,
) -> None:
    variant = output / "variants" / str(spec["variant_id"])
    variant.mkdir(parents=True, exist_ok=True)
    attempt_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"_pid{os.getpid()}"
    )
    process_evidence = (
        output.parent
        / f"{output.name}_process_evidence"
        / str(spec["variant_id"])
        / attempt_id
    )
    process_evidence.mkdir(parents=True, exist_ok=True)
    scratch_output = (
        output.parent
        / f"{output.name}_hydra_scratch"
        / str(spec["variant_id"])
        / attempt_id
        / "hydra_realtime"
    )
    scratch_output.parent.mkdir(parents=True, exist_ok=False)
    promoted_output = variant / "hydra_realtime"
    if promoted_output.exists():
        raise FileExistsError(
            "refuse to overwrite incomplete Hydra output: "
            f"{promoted_output}"
        )
    plan = {
        "schema": "daaam.hydra_semantic_postpass_plan.v1",
        "run_dir": str((output / "shared_input").resolve()),
        "output_dir": str(scratch_output.resolve()),
        "semantic_label_dir": str(
            (output / "shared_input/label_frames").resolve()
        ),
        "label_run_configuration_sha256": label_configuration_sha256,
        "hydra_config_path": str(
            (output / "configs" / f"{spec['variant_id']}.yaml").resolve()
        ),
        "labelspace_path": str(
            (REPOSITORY_ROOT / "config/labels_pseudo.yaml").resolve()
        ),
        "labelspace_colors": str(
            (REPOSITORY_ROOT / "config/labels_pseudo.csv").resolve()
        ),
        "maximum_depth_m": 8.0,
        "frames": [
            {
                key: frame[key]
                for key in (
                    "frame_index",
                    "sensor_time_ns",
                    "rgb_path",
                    "world_T_camera",
                    "intrinsics",
                )
            }
            for frame in frames
        ],
    }
    write_json(process_evidence / "hydra_postpass_plan.json", plan)
    process_report = process_evidence / "hydra_postpass_report.json"
    command = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(process_evidence / "resource_usage_time_v.txt"),
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/run_hydra_semantic_postpass.py"),
        "--plan",
        str(process_evidence / "hydra_postpass_plan.json"),
        "--report",
        str(process_report),
    ]
    started = time.monotonic()
    status = "failed"
    return_code: int | None = None
    failure_reason: str | None = None
    try:
        with (process_evidence / "stdout.log").open(
            "w", encoding="utf-8"
        ) as stdout:
            with (process_evidence / "stderr.log").open(
                "w", encoding="utf-8"
            ) as stderr:
                result = subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout_seconds,
                    check=False,
                )
        return_code = result.returncode
        status = "complete" if return_code == 0 else "failed"
        if return_code:
            failure_reason = f"subprocess exit code {return_code}"
        elif not scratch_output.is_dir():
            status = "failed"
            failure_reason = "Hydra completed without its isolated output"
        elif not process_report.is_file():
            status = "failed"
            failure_reason = "Hydra completed without its process report"
        else:
            shutil.move(str(scratch_output), str(promoted_output))
    except subprocess.TimeoutExpired:
        status = "timeout"
        failure_reason = f"timeout after {timeout_seconds} seconds"
    finally:
        variant.mkdir(parents=True, exist_ok=True)
        write_json(variant / "hydra_postpass_plan.json", plan)
        if process_report.is_file():
            shutil.copy2(
                process_report,
                variant / "hydra_postpass_report.json",
            )
        if process_evidence.joinpath("resource_usage_time_v.txt").is_file():
            shutil.copy2(
                process_evidence / "resource_usage_time_v.txt",
                variant / "resource_usage_time_v.txt",
            )
        for name in ("stdout.log", "stderr.log"):
            if process_evidence.joinpath(name).is_file():
                shutil.copy2(
                    process_evidence / name,
                    variant / name,
                )
        write_json(
            variant / "execution.json",
            {
                "schema": "daaam.g1_no_gt_e16_execution.v1",
                "variant_id": spec["variant_id"],
                "status": status,
                "command": command,
                "started_at_monotonic": started,
                "elapsed_seconds": time.monotonic() - started,
                "return_code": return_code,
                "failure_reason": failure_reason,
                "process_evidence_directory": str(process_evidence),
                "isolated_hydra_output": str(scratch_output),
                "promoted_hydra_output": (
                    str(promoted_output) if status == "complete" else None
                ),
            },
        )
    if status != "complete":
        raise RuntimeError(
            f"Hydra variant {spec['variant_id']} failed: {failure_reason}"
        )


def make_visualizations(
    output: Path,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    visualizations = output / "visualizations"
    visualizations.mkdir(parents=True, exist_ok=True)
    labels = [
        str(row["variant_id"])
        .replace("voxel_", "")
        .replace("_vol", "\nvol")
        .replace("_obs", "\nobs")
        for row in summaries
    ]
    x = np.arange(len(summaries))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    values = [
        float(row["lidar_completeness_proxy_within_0p10m"])
        for row in summaries
    ]
    axes[0, 0].bar(x, values, color="#3182bd")
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_ylabel("Visible LiDAR points within 0.10 m")
    axes[0, 0].set_title("Sparse-LiDAR completeness proxy")
    values = [
        float(row["lidar_accuracy_proxy_within_0p10m"])
        for row in summaries
    ]
    axes[0, 1].bar(x, values, color="#6baed6")
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].set_ylabel("Mesh vertices within 0.10 m")
    axes[0, 1].set_title("Sparse-LiDAR accuracy proxy")
    values = [float(row["semantic_label_survival_proxy"]) for row in summaries]
    axes[1, 0].bar(x, values, color="#31a354")
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].set_ylabel("E13 visible labels surviving to DSG")
    axes[1, 0].set_title("Semantic-label survival proxy (not recall)")
    p95 = [float(row["hydra_processing_p95_ms"]) for row in summaries]
    axes[1, 1].bar(x, p95, color="#fd8d3c")
    axes[1, 1].axhline(
        RUNTIME_HARD_GATE_MS,
        color="#de2d26",
        linestyle="--",
        label="250 ms hard gate",
    )
    axes[1, 1].set_ylabel("Hydra process-frame P95 (ms)")
    axes[1, 1].set_title("Runtime hard gate")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("E16 parameter frontier — NO HUMAN GEOMETRY/OBJECT GT")
    fig.savefig(visualizations / "01_e16_frontier.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for row in summaries:
        metrics = (
            output
            / "variants"
            / str(row["variant_id"])
            / "metrics"
        )
        for axis, filename, title in (
            (
                axes[0],
                "visible_lidar_to_mesh_distances_m.npy",
                "Visible LiDAR → nearest mesh vertex",
            ),
            (
                axes[1],
                "mesh_to_visible_lidar_distances_m.npy",
                "Mesh vertex → nearest visible LiDAR",
            ),
        ):
            distances = np.load(metrics / filename, allow_pickle=False)
            clipped = np.sort(np.minimum(distances, 0.5))
            axis.plot(
                clipped,
                np.linspace(0, 1, len(clipped), endpoint=True),
                label=str(row["variant_id"]),
                linewidth=1.4,
            )
            axis.set_title(title)
            axis.set_xlabel("Nearest-neighbour distance (m), clipped at 0.5")
            axis.set_ylabel("Empirical CDF")
            axis.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    fig.suptitle("Sparse cross-sensor agreement proxy — not surface GT")
    fig.savefig(visualizations / "02_sparse_lidar_distance_cdf.png", dpi=180)
    plt.close(fig)

    preview_paths = [
        output
        / "variants"
        / str(row["variant_id"])
        / "hydra_map_preview.png"
        for row in summaries
    ]
    images = [plt.imread(path) for path in preview_paths]
    fig, axes = plt.subplots(
        len(images),
        1,
        figsize=(18, 5 * len(images)),
        constrained_layout=True,
    )
    if len(images) == 1:
        axes = [axes]
    for axis, image, row in zip(axes, images, summaries):
        axis.imshow(image)
        axis.set_title(str(row["variant_id"]))
        axis.axis("off")
    fig.suptitle("E16 Hydra map previews (fixed projections)")
    fig.savefig(visualizations / "03_map_preview_contact_sheet.jpg", dpi=130)
    plt.close(fig)


def failure_rows(
    summaries: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        variant_id = str(summary["variant_id"])
        if not summary["runtime_hard_gate_passed"]:
            rows.append(
                {
                    "variant_id": variant_id,
                    "primary_cause": "F-COMP",
                    "symptom": "Hydra process-frame P95 exceeds 250 ms",
                    "value": summary["hydra_processing_p95_ms"],
                    "evidence": (
                        f"variants/{variant_id}/hydra_postpass_report.json"
                    ),
                }
            )
        objects = read_json(
            output
            / "variants"
            / variant_id
            / "metrics/object_metrics.json"
        )
        for label in objects["missing_source_labels"]:
            rows.append(
                {
                    "variant_id": variant_id,
                    "primary_cause": "F-HYDRA",
                    "symptom": "visible E13 semantic label has no DSG object node",
                    "value": int(label),
                    "evidence": (
                        f"variants/{variant_id}/metrics/object_metrics.json"
                    ),
                }
            )
    return rows


def build_report(
    summaries: Sequence[Mapping[str, Any]],
    source_labels: set[int],
    named_labels: set[int],
) -> str:
    rows = []
    for value in summaries:
        rows.append(
            "| {variant_id} | {mesh_vertices:,} | {comp:.1%} | {acc:.1%} | "
            "{objects} / {labels} | {survival:.1%} | {p50:.1f} | {p95:.1f} | "
            "{maximum:,.1f} | {wall:.1f} | {rss:.0f} | {gate} |".format(
                variant_id=value["variant_id"],
                mesh_vertices=int(value["mesh_vertices"]),
                comp=float(
                    value["lidar_completeness_proxy_within_0p10m"]
                ),
                acc=float(value["lidar_accuracy_proxy_within_0p10m"]),
                objects=int(value["dsg_object_nodes"]),
                labels=int(value["unique_dsg_semantic_labels"]),
                survival=float(value["semantic_label_survival_proxy"]),
                p50=float(value["hydra_processing_p50_ms"]),
                p95=float(value["hydra_processing_p95_ms"]),
                maximum=float(value["hydra_processing_max_ms"]),
                wall=float(value["wall_elapsed_seconds"]),
                rss=float(value["peak_rss_mib"]),
                gate=(
                    "通过"
                    if value["runtime_hard_gate_passed"]
                    else "失败"
                ),
            )
        )
    baseline = next(
        value
        for value in summaries
        if value["variant_id"] == "voxel_5cm_obs8_vol0p005"
    )
    eligible = [
        value
        for value in summaries
        if value["winner_eligible"]
    ]
    if not eligible:
        raise ValueError("no non-diagnostic candidate passed the runtime gate")
    best_completeness = max(
        eligible,
        key=lambda value: value[
            "lidar_completeness_proxy_within_0p10m"
        ],
    )
    best_survival = max(
        eligible,
        key=lambda value: value["semantic_label_survival_proxy"],
    )
    diagnostic = next(
        value
        for value in summaries
        if value["variant_id"] == "voxel_3cm_obs8_vol0p005"
    )
    observation_variant = next(
        value
        for value in summaries
        if value["variant_id"] == "voxel_5cm_obs4_vol0p005"
    )
    volume_variant = next(
        value
        for value in summaries
        if value["variant_id"] == "voxel_5cm_obs8_vol0p0001"
    )
    return f"""# E16 Hydra 体素与 object 参数实验

## 结论

本轮冻结 E11 `conf=0.3, area=300, IoU=0.5`、E12 `buffer=10`、
E13 `safe_merge=0.35 m` 和 101 帧公制深度/`map_T_camera`，在相同输入上运行
5 个 Hydra 候选。体素对照以 5 cm 配置为公共基线，只联动修改体素尺寸、
三倍截断带和 object detection grid；`obs=4` 与 `min_volume=0.0001 m³`
均为相对 5 cm baseline 的单因素对照。

片段本身时长 9.9785 s，小于冻结的 tracker `temporal_window=10.0 s`。因此在
101 个真实数据帧后加入 1 个明确标记的零深度/零标签 flush 事件，时间为末帧后
10.1 s，使对象到期和提取在所有候选中一致发生。flush 不是数据帧，排除在
split/challenge 几何统计之外，但纳入运行时资源与最大延迟，因为对象收尾本身有成本。
为避免 Khronos 默认的并发对象提取共享同一 `mesh_integrator_`，所有候选统一冻结
`extraction_worker.num_workers=1`、`detach_object_extraction=false`，并使用
`patches/khronos_object_worker_pool_deterministic.patch` 修复 worker queue 的
join/publish 顺序。该修复不改变候选间的参数差异。

结果支持“存在质量/资源前沿”，但**不能给出正式 winner**。原因是没有人工表面 GT 和
object `should_have_mesh` GT：下表几何值是相机可见 LiDAR 的稀疏最近邻一致性代理，
object 值是上游 E13 标签进入 DSG 的生存代理，不是 mesh accuracy/completeness 或
object recall。

## 汇总

| variant | mesh vertices | LiDAR→mesh ≤10 cm | mesh→LiDAR ≤10 cm | DSG nodes / labels | label survival | P50 ms | P95 ms | max ms | wall s | peak RSS MiB | 250 ms P95 门 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

5 cm baseline 的 P95 为 {float(baseline["hydra_processing_p95_ms"]):.1f} ms，
标签生存代理为 {float(baseline["semantic_label_survival_proxy"]):.1%}。
在可晋级候选中，LiDAR→mesh 10 cm 覆盖最高的是
`{best_completeness["variant_id"]}`（
{float(best_completeness["lidar_completeness_proxy_within_0p10m"]):.1%}），
标签生存最高的是 `{best_survival["variant_id"]}`（
{float(best_survival["semantic_label_survival_proxy"]):.1%}）。这些是工程代理的
Pareto 观察，不替代正式晋级。

`obs=4` 在几何完全不变的条件下把 object node 从
{int(baseline["dsg_object_nodes"])} 增至
{int(observation_variant["dsg_object_nodes"])}，是本轮最有效的 object 生存代理改善，
但仍未通过 P95 门。把 `minimum_volume` 降到 `0.0001 m³` 只得到
{int(volume_variant["dsg_object_nodes"])} 个 node，最大单事件延迟却达到
{float(volume_variant["hydra_processing_max_ms"]) / 1000.0:.1f} s、墙钟达到
{float(volume_variant["wall_elapsed_seconds"]):.1f} s、峰值 RSS 达到
{float(volume_variant["peak_rss_mib"]) / 1024.0:.2f} GiB，因此是明确的工程劣化。

3 cm 按规约始终为 diagnostic；本次 P95
{float(diagnostic["hydra_processing_p95_ms"]):.1f} ms，
实时硬门{"通过" if diagnostic["runtime_hard_gate_passed"] else "失败"}，
且冻结配置的 GVD `min_distance_m=0.10 m` 大于 TSDF
`truncation_distance=0.09 m`。Hydra 已在原始日志中逐帧报告该冲突，因此
3 cm 结果只用于资源/失败诊断，不能用于几何质量晋级。

12 cm 是唯一通过预注册 P95 硬门的候选，但它的最大单事件延迟仍为
{float(best_completeness["hydra_processing_max_ms"]) / 1000.0:.1f} s。该最大值来自
末尾对象到期/提取的 flush 事件；因为 102 个事件中只有 1 个此类收尾事件，P95 不会
反映它。因此“通过 P95 门”只表示常规逐帧路径满足门限，不表示当前对象收尾可以在线
运行。部署前需把对象提取移出实时关键路径、增量化或施加独立 deadline。

## 对象口径

- E13 共 169 entities，其中 {len(source_labels)} 个正标签实际出现在 101 张 ID 图；
  未出现的 7 个实体没有像素输入，不能要求 Hydra 重建。
- E14 的 87 个命名实体中，{len(named_labels)} 个同时出现在 ID 图；所有 object
  代理均以这个可观察集合为分母。
- 一个 semantic label 进入 DSG 只证明 Hydra 产生了对应 object node，不证明它与
  真实物体一一对应。E11 漏检、粘连以及 E13 over-split 仍会传入本轮。

## 几何口径与限制

LiDAR reference 仅汇集 473–573 中投影落在左目图像内的返回点，在 map 坐标按 2 cm
去重。`LiDAR→mesh` 表示可见 LiDAR 点到最近 mesh vertex 的距离；
`mesh→LiDAR` 反向度量会受 LiDAR 稀疏性惩罚。它们不是 point-to-triangle 距离，
也不评价物体背面、遮挡面或 LiDAR 未返回区域。

输入深度保留 0.1–8 m 的全部有效缩放深度。由于没有可靠 dynamic GT，本轮没有执行
dynamic mask 隔离；动态残影风险必须记为限制，不能把 mesh 面积增加自动解释为质量提升。

## 完整性、切分与挑战标签

每个 variant 均要求 101 个 source frames 加 1 个 flush event 的 102/102 exact
label-frame 绑定、0 rejected frames 和完整 mesh/DSG。`metrics_by_split.json`
与 `metrics_by_challenge_tag.json` 对
`active_window/all` 原始逐帧计时按 split/自动标签报告，并用长度 5、5000 次的时间块
bootstrap 给出均值置信区间。人工 `plant_boundary/thin_structure/occluded/
semantic_dense` 标签仍是 `pending_human_review`，因此未伪造分组几何结论。

## 证据位置

- `PRE_REGISTRATION.json`：候选、门限、单因素策略及禁止声明。
- `FROZEN_INPUTS.json`、`shared_input/*manifest*`：逐帧 RGB/深度/标签/LiDAR/pose lineage。
- `configs/`：5 份实际执行配置和参数矩阵。
- `variants/*/hydra_realtime/`：原始 Hydra mesh、DSG、timing 与日志。
- `variants/*/metrics/`：mesh 拓扑、对象节点、双向 LiDAR 距离数组、资源和分组指标。
- `tables/risk_assessment.json`：定稿延迟、资源、阈值和 3 cm 配置冲突的结构化风险。
- `visualizations/`：固定投影视图、参数前沿和双向距离 CDF。
- `failure_cases/`：超时门失败及输入标签未进入 DSG 的逐项记录。
- `artifact_inventory.*`、`COMPLETION.json`：文件级封存。
- 同级 `*_process_evidence/` 与 `*_hydra_scratch/`：隔离 attempt 的原始进程证据；
  失败/中止的环境预飞不进入正式候选比较，成功结果已复制并晋级到 `variants/`。
"""


def build_risk_assessment(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {str(value["variant_id"]): value for value in summaries}
    baseline_12cm = by_id["voxel_12cm_obs8_vol0p005"]
    baseline_5cm = by_id["voxel_5cm_obs8_vol0p005"]
    observations_4 = by_id["voxel_5cm_obs4_vol0p005"]
    minimum_volume = by_id["voxel_5cm_obs8_vol0p0001"]
    diagnostic = by_id["voxel_3cm_obs8_vol0p005"]
    return {
        "schema": "daaam.g1_no_gt_e16_risk_assessment.v1",
        "status": "risks_present_do_not_auto_promote",
        "formal_winner": None,
        "interim_engineering_fallback": (
            "voxel_12cm_obs8_vol0p005"
            if baseline_12cm["runtime_hard_gate_passed"]
            else None
        ),
        "risks": [
            {
                "risk_id": "E16-R1",
                "name": "synchronous_object_finalization_stall",
                "severity": "critical_for_online_use",
                "maximum_seconds_by_variant": {
                    variant_id: float(value["hydra_processing_max_ms"])
                    / 1000.0
                    for variant_id, value in by_id.items()
                },
                "mitigation": (
                    "move final drain off the online main thread; retain one "
                    "deterministic worker, bounded queue and backpressure"
                ),
                "evidence": "variants/*/hydra_realtime/timing/",
            },
            {
                "risk_id": "E16-R2",
                "name": "resolution_runtime_tradeoff",
                "severity": "high",
                "baseline_12cm_p95_ms": baseline_12cm[
                    "hydra_processing_p95_ms"
                ],
                "candidate_5cm_p95_ms": baseline_5cm[
                    "hydra_processing_p95_ms"
                ],
                "baseline_12cm_label_survival": baseline_12cm[
                    "semantic_label_survival_proxy"
                ],
                "candidate_5cm_label_survival": baseline_5cm[
                    "semantic_label_survival_proxy"
                ],
                "evidence": "tables/variant_summary.json",
            },
            {
                "risk_id": "E16-R3",
                "name": "low_observation_false_entity_persistence",
                "severity": "unknown_without_object_gt",
                "baseline_objects": baseline_5cm["dsg_object_nodes"],
                "obs4_objects": observations_4["dsg_object_nodes"],
                "formal_precision_available": False,
                "evidence": (
                    "variants/voxel_5cm_obs4_vol0p005/metrics/"
                    "object_metrics.json"
                ),
            },
            {
                "risk_id": "E16-R4",
                "name": "low_minimum_volume_resource_explosion",
                "severity": "critical_for_online_use",
                "maximum_seconds": float(
                    minimum_volume["hydra_processing_max_ms"]
                )
                / 1000.0,
                "peak_rss_gib": float(minimum_volume["peak_rss_mib"])
                / 1024.0,
                "objects": minimum_volume["dsg_object_nodes"],
                "obs4_objects": observations_4["dsg_object_nodes"],
                "evidence": (
                    "variants/voxel_5cm_obs8_vol0p0001/metrics/"
                    "resource_metrics.json"
                ),
            },
            {
                "risk_id": "E16-R5",
                "name": "three_cm_gvd_truncation_conflict",
                "severity": "invalid_for_quality_promotion",
                "diagnostic_only": True,
                "gvd_min_distance_m": 0.10,
                "truncation_distance_m": diagnostic[
                    "truncation_distance_m"
                ],
                "compatible": False,
                "observed_error": (
                    "GVD integrator min distance must be less than "
                    "truncation distance"
                ),
                "evidence": (
                    "variants/voxel_3cm_obs8_vol0p005/"
                    "hydra_realtime/logs/"
                ),
            },
            {
                "risk_id": "E16-R6",
                "name": "upstream_semantic_and_dynamic_error_persistence",
                "severity": "unknown_without_reviewed_gt",
                "dynamic_gt_available": False,
                "object_should_have_mesh_gt_available": False,
                "formal_object_recall_available": False,
                "evidence": "PRE_REGISTRATION.json",
            },
        ],
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if args.reseal_existing:
        if not output.is_dir():
            raise FileNotFoundError(output)
        audit = output / "INDEPENDENT_AUDIT.json"
        passed = audit.is_file() and read_json(audit).get("passed") is True
        seal_output(
            output,
            status=(
                "complete_independently_audited"
                if passed
                else "complete_pending_independent_audit"
            ),
        )
        return 0
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"refuse to overwrite existing output; use --resume: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    for directory in (
        "configs",
        "failure_cases",
        "source_snapshot",
        "tables",
        "variants",
        "visualizations",
    ):
        (output / directory).mkdir(exist_ok=True)
    shutil.copy2(
        Path(__file__),
        output / "source_snapshot/run_g1_no_gt_e16_hydra.py",
    )

    paths = {
        "e13_completion": args.e13_run.resolve() / "COMPLETION.json",
        "e14_completion": args.e14_run.resolve() / "COMPLETION.json",
        "e15_completion": args.e15_run.resolve() / "COMPLETION.json",
        "e12_frames": (
            args.geometry_run.resolve() / "input_manifests/e12_frames.jsonl"
        ),
        "poses": args.depth.resolve() / "pose/poses.txt",
        "camera_info": args.depth.resolve() / "camera_info.json",
        "e14_labels": (
            args.e14_run.resolve() / "tables/final_labels.jsonl"
        ),
        "challenge_tags": EXPERIMENT_ROOT / "manifests/challenge_tags.json",
        "splits": EXPERIMENT_ROOT / "manifests/splits.json",
        "thresholds": EXPERIMENT_ROOT / "protocol/thresholds.yaml",
        "base_hydra_config": REPOSITORY_ROOT / "config/hydra_g1_high_quality.yaml",
        "reference_12cm_config": REPOSITORY_ROOT / "config/hydra_g1_8m_12cm.yaml",
        "reference_3cm_config": (
            REPOSITORY_ROOT / "config/hydra_g1_high_quality_3cm.yaml"
        ),
        "labelspace": REPOSITORY_ROOT / "config/labels_pseudo.yaml",
        "labelspace_colors": REPOSITORY_ROOT / "config/labels_pseudo.csv",
        "postpass_script": (
            REPOSITORY_ROOT / "scripts/run_hydra_semantic_postpass.py"
        ),
        "preview_script": (
            REPOSITORY_ROOT / "scripts/render_hydra_map_preview.py"
        ),
        "khronos_worker_patch": (
            REPOSITORY_ROOT
            / "patches/khronos_object_worker_pool_deterministic.patch"
        ),
        "khronos_worker_source": (
            REPOSITORY_ROOT
            / ".repro/ros2_ws/src/khronos/khronos/src/active_window"
            / "object_extraction/object_worker_pool.cpp"
        ),
        "khronos_library": (
            REPOSITORY_ROOT
            / ".repro/ros2_ws/install/khronos/lib/libkhronos.so"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing E16 inputs: " + ", ".join(missing))
    for key in ("e13_completion", "e14_completion", "e15_completion"):
        completion = read_json(paths[key])
        if completion["status"] != "complete_independently_audited":
            raise ValueError(f"upstream is not independently audited: {paths[key]}")

    specs = variant_specs()
    preregistration_path = output / "PRE_REGISTRATION.json"
    if not preregistration_path.exists():
        write_json(
            preregistration_path,
            {
                "schema": "daaam.g1_no_gt_e16_preregistration.v1",
                "created_at": utc_now(),
                "source_frame_range": [473, 573],
                "frame_count": FRAME_COUNT,
                "hydra_event_count": HYDRA_EVENT_COUNT,
                "frozen_chain": {
                    "E11": "conf=0.3, area=300, IoU=0.5",
                    "E12": "BotSort buffer=10",
                    "E13": "safe_merge=0.35m",
                    "E14": "unique_safe, observations=8, seed=0",
                    "depth": "calibration-only scale applied to metric depth",
                    "pose": "map_T_camera",
                },
                "variants": specs,
                "voxel_isolation_rule": (
                    "all variants derive from hydra_g1_high_quality.yaml; "
                    "voxel variants only change voxel_size, the fixed 3x "
                    "truncation distance, and object detector grid_size"
                ),
                "deterministic_object_extraction": {
                    "detach_object_extraction": False,
                    "worker_count": 1,
                    "patch": (
                        "patches/khronos_object_worker_pool_deterministic.patch"
                    ),
                    "reason": (
                        "MeshObjectExtractor owns a shared mesh_integrator and "
                        "the upstream worker join did not drain queued requests"
                    ),
                },
                "runtime_hard_gate_ms": RUNTIME_HARD_GATE_MS,
                "distance_thresholds_m": DISTANCE_THRESHOLDS_M,
                "statistics": {
                    "time_block_length": 5,
                    "bootstrap_repetitions": 5000,
                    "seed": 0,
                },
                "three_cm_policy": (
                    "diagnostic only and never selected as a formal winner"
                ),
                "object_finalization_policy": {
                    "tracker_temporal_window_seconds": 10.0,
                    "source_sequence_duration_seconds": 9.978507575,
                    "flush_event_count": FLUSH_EVENT_COUNT,
                    "flush_after_last_frame_seconds": (
                        FLUSH_AFTER_LAST_FRAME_SECONDS
                    ),
                    "flush_depth": "all zero",
                    "flush_labels": "all zero",
                    "source_metrics_include_flush": False,
                    "runtime_metrics_include_flush": True,
                },
                "formal_claims_permitted": False,
                "unavailable_metrics": (
                    "absolute mesh completeness/accuracy and object recall "
                    "pending reviewed surface/object GT"
                ),
            },
        )
        write_json(
            output / "invocation.json",
            {
                "schema": "daaam.g1_no_gt_e16_invocation.v1",
                "argv": sys.argv,
                "cwd": os.getcwd(),
                "python": sys.executable,
                "python_version": platform.python_version(),
                "git_commit": _git_value("rev-parse", "HEAD"),
                "git_branch": _git_value("branch", "--show-current"),
                "started_at": utc_now(),
            },
        )
        write_json(
            output / "FROZEN_INPUTS.json",
            {
                "schema": "daaam.g1_no_gt_e16_frozen_inputs.v1",
                "captured_at": utc_now(),
                **{
                    key: frozen_reference(path)
                    for key, path in paths.items()
                },
            },
        )

    base_config = load_yaml(paths["base_hydra_config"])
    config_rows = []
    for spec in specs:
        config = build_variant_config(base_config, spec)
        config_path = output / "configs" / f"{spec['variant_id']}.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        config_rows.append(
            {
                **spec,
                "gvd_min_distance_m": float(
                    config["frontend"]["freespace_places"]["gvd"][
                        "min_distance_m"
                    ]
                ),
                "gvd_truncation_compatible": (
                    float(
                        config["frontend"]["freespace_places"]["gvd"][
                            "min_distance_m"
                        ]
                    )
                    < float(spec["truncation_distance_m"])
                ),
                "config_path": str(config_path.resolve()),
                "config_sha256": sha256_file(config_path),
                "base_config_sha256": sha256_file(paths["base_hydra_config"]),
            }
        )
    write_json(output / "tables/parameter_matrix.json", config_rows)
    write_csv(output / "tables/parameter_matrix.csv", config_rows)

    prepared_marker = output / "shared_input/PREPARATION_COMPLETE.json"
    if not prepared_marker.exists():
        frames, label_sha, source_labels, named_labels = prepare_inputs(
            args, output, paths
        )
        write_json(
            prepared_marker,
            {
                "schema": "daaam.g1_no_gt_e16_input_preparation.v1",
                "status": "complete",
                "frames": len(frames),
                "source_frames": FRAME_COUNT,
                "flush_events": FLUSH_EVENT_COUNT,
                "label_configuration_sha256": label_sha,
                "source_visible_labels": len(source_labels),
                "named_visible_labels": len(named_labels),
                "completed_at": utc_now(),
            },
        )
    else:
        frames, label_sha, source_labels, named_labels = load_prepared_inputs(
            output
        )
    if args.prepare_only:
        print(output)
        return 0

    challenges_value = read_json(paths["challenge_tags"])
    challenges = {
        int(record["source_index"]): record
        for record in challenges_value["frames"]
    }
    lidar_points = np.load(
        output / "shared_input/visible_lidar_reference_2cm.npy",
        allow_pickle=False,
    )
    render_lidar_reference(
        lidar_points, output / "visualizations/00_visible_lidar_reference.png"
    )

    summaries = []
    for spec in specs:
        variant = output / "variants" / str(spec["variant_id"])
        summary_path = variant / "variant_summary.json"
        if not (variant / "hydra_postpass_report.json").is_file():
            run_variant(
                output,
                spec,
                frames,
                label_sha,
                args.timeout_seconds,
            )
        if not (variant / "hydra_map_preview.png").is_file():
            subprocess.run(
                [
                    sys.executable,
                    str(paths["preview_script"]),
                    "--run-dir",
                    str(variant),
                    "--output",
                    str(variant / "hydra_map_preview.png"),
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                stdout=(variant / "preview_stdout.log").open(
                    "w", encoding="utf-8"
                ),
                stderr=(variant / "preview_stderr.log").open(
                    "w", encoding="utf-8"
                ),
            )
        if not summary_path.is_file():
            summary = analyze_variant(
                variant,
                spec,
                source_labels,
                named_labels,
                lidar_points,
                frames,
                challenges,
            )
        else:
            summary = read_json(summary_path)
        summaries.append(summary)

    write_json(output / "tables/variant_summary.json", summaries)
    write_csv(output / "tables/variant_summary.csv", summaries)
    failures = failure_rows(summaries, output)
    write_jsonl(output / "failure_cases/failure_cases.jsonl", failures)
    write_csv(output / "failure_cases/failure_cases.csv", failures)
    write_json(
        output / "failure_cases/failure_summary.json",
        {
            "failure_count": len(failures),
            "F_COMP": sum(row["primary_cause"] == "F-COMP" for row in failures),
            "F_HYDRA": sum(row["primary_cause"] == "F-HYDRA" for row in failures),
        },
    )
    make_visualizations(output, summaries)
    write_json(
        output / "tables/risk_assessment.json",
        build_risk_assessment(summaries),
    )
    (output / "REPORT.md").write_text(
        build_report(summaries, source_labels, named_labels),
        encoding="utf-8",
    )
    write_json(
        output / "RUN_SUMMARY.json",
        {
            "schema": "daaam.g1_no_gt_e16_run_summary.v1",
            "status": "complete_pending_independent_audit",
            "frames": FRAME_COUNT,
            "hydra_events": HYDRA_EVENT_COUNT,
            "variants_complete": len(summaries),
            "variants": summaries,
            "formal_winner": None,
            "reason_no_formal_winner": (
                "reviewed surface geometry and should-have-mesh object GT absent"
            ),
            "finished_at": utc_now(),
        },
    )
    seal_output(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
