#!/usr/bin/env python3
"""Read-only, no-GT audit for the G1 semantic mapping pipeline.

This utility intentionally does not re-run or mutate any upstream stage.  It
joins the persisted raw, rectification, geometry, temporal, semantic, runtime,
MapMemory and Hydra evidence into one auditable bundle.

Accuracy language is deliberately conservative:

* ``exact`` means an artifact/count/timing was measured directly.
* ``proxy`` means a diagnostic is internally consistent but has no independent
  ground truth.
* ``unavailable`` means the requested accuracy cannot be established without
  annotations or an independent reference.

The script hashes a curated set of contract/report artifacts.  It does not
perform a full-file inventory or hash every image/depth/mask file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import platform
import shlex
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCHEMA = "daaam.g1_semantic_pipeline_no_gt_audit.v1"
NO_GT_REASON = (
    "人工 GT、双人裁决及独立 held-out 本阶段被明确跳过；因此不能估计语义/几何/"
    "轨迹真实准确率。直接计数与运行时测量可为 exact，质量解释仅为 proxy，"
    "需要 GT 的精确率、召回率、IoU、ATE 等为 unavailable。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate a read-only, no-GT audit of the G1 semantic pipeline."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/home/user/datasets/g1_20260724"),
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("output/g1_20260724_v1_v2_rectified_prepared"),
    )
    parser.add_argument(
        "--geometry-dir",
        type=Path,
        default=Path("output/g1_20260724_v1_v2_geometry"),
    )
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=Path("output/g1_20260724_v1_v2_semantic_map"),
    )
    parser.add_argument(
        "--best-combination",
        type=Path,
        default=Path(
            "output/g1_20260724_v1_v2_optimal_combination_653_953_final/"
            "best_combination.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/g1_20260724_v1_v2_no_gt_audit"),
    )
    parser.add_argument(
        "--skip-lidar-scan",
        action="store_true",
        help="Skip streaming LiDAR zero/non-finite point statistics.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow replacing files owned by this audit inside an existing output dir.",
    )
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=844,
        help="Expected selected/geometry/semantic frame count.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required file is missing: {path}")
    return path


def require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"required directory is missing: {path}")
    return path


def load_json(path: Path) -> Any:
    with require_file(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stat_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
    }


def sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(sanitize(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def percentile_summary(values: Iterable[Any]) -> dict[str, Any]:
    clean = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if clean.size == 0:
        return {
            "count": 0,
            "mean": None,
            "minimum": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "maximum": None,
        }
    p05, p25, p50, p75, p95 = np.percentile(clean, [5, 25, 50, 75, 95])
    return {
        "count": int(clean.size),
        "mean": float(np.mean(clean)),
        "minimum": float(np.min(clean)),
        "p05": float(p05),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "maximum": float(np.max(clean)),
    }


def accuracy_unavailable(target: str) -> dict[str, str]:
    return {
        "target": target,
        "evidence_level": "unavailable",
        "status": "unavailable",
        "reason": NO_GT_REASON,
    }


def proxy_assessment(
    status: str, claim: str, basis: Sequence[str], limitations: Sequence[str]
) -> dict[str, Any]:
    return {
        "status": status,
        "evidence_level": "proxy",
        "claim": claim,
        "basis": list(basis),
        "limitations": list(limitations),
        "accuracy": accuracy_unavailable(claim),
    }


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def ensure_output_dir(
    output_dir: Path, input_dirs: Sequence[Path], allow_existing: bool
) -> None:
    for input_dir in input_dirs:
        if output_dir == input_dir or is_within(output_dir, input_dir):
            raise ValueError(
                f"output directory must not be inside an upstream input: {output_dir}"
            )
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"output directory is non-empty: {output_dir}; "
            "use --allow-existing-output to replace this audit's files"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def raw_manifest_scan(
    raw_dir: Path, scan_lidar: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = require_file(raw_dir / "manifest.jsonl")
    rows: list[dict[str, Any]] = []
    missing_paths: list[dict[str, Any]] = []
    lidar_load_errors: list[dict[str, Any]] = []
    image_presence = Counter()
    lidar_presence = Counter()
    pose_presence = Counter()
    error_frames = 0
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            tick = int(record.get("tick", line_number - 1))
            images = {
                str(item.get("camera")): item
                for item in record.get("images", [])
                if item.get("camera") is not None
            }
            cam0 = images.get("cam0")
            cam1 = images.get("cam1")
            for camera in ("cam0", "cam1"):
                if camera in images:
                    image_presence[camera] += 1
            stereo_delta_ms = None
            if cam0 and cam1:
                stereo_delta_ms = abs(
                    int(cam0["sensor_time_ns"]) - int(cam1["sensor_time_ns"])
                ) / 1.0e6

            required_paths: list[tuple[str, str]] = []
            for camera, item in images.items():
                if item.get("path"):
                    required_paths.append((f"image:{camera}", str(item["path"])))
            lidar_items = record.get("lidar", [])
            if lidar_items:
                lidar_presence[str(lidar_items[0].get("lidar", "lidar0"))] += 1
                if lidar_items[0].get("path"):
                    required_paths.append(("lidar", str(lidar_items[0]["path"])))
            missing_for_frame = 0
            for kind, relative_path in required_paths:
                if not (raw_dir / relative_path).is_file():
                    missing_for_frame += 1
                    missing_paths.append(
                        {"tick": tick, "kind": kind, "path": relative_path}
                    )

            poses = record.get("poses", {}).get("values", {})
            for pose_name in ("head_camera", "map", "lidar", "left_eef", "right_eef"):
                if pose_name in poses:
                    pose_presence[pose_name] += 1
            record_errors = list(record.get("errors", []))
            if record_errors:
                error_frames += 1

            lidar_zero_ratio = None
            lidar_nonfinite_ratio = None
            lidar_point_count = None
            if scan_lidar and lidar_items and lidar_items[0].get("path"):
                lidar_path = raw_dir / str(lidar_items[0]["path"])
                if lidar_path.is_file():
                    try:
                        points = np.load(lidar_path, mmap_mode="r")
                        xyz = np.asarray(points)
                        if xyz.ndim != 2 or xyz.shape[1] < 3:
                            raise ValueError(f"unexpected LiDAR shape {xyz.shape}")
                        xyz = xyz[:, :3]
                        lidar_point_count = int(xyz.shape[0])
                        finite = np.isfinite(xyz).all(axis=1)
                        zero = np.all(np.abs(xyz) <= 1.0e-12, axis=1)
                        lidar_zero_ratio = float(np.mean(zero)) if zero.size else None
                        lidar_nonfinite_ratio = (
                            float(np.mean(~finite)) if finite.size else None
                        )
                    except Exception as exc:  # retain evidence and continue
                        lidar_load_errors.append(
                            {"tick": tick, "path": str(lidar_path), "error": repr(exc)}
                        )

            sync = record.get("sync", {})
            rows.append(
                {
                    "tick": tick,
                    "wall_time_ns": record.get("wall_time_ns"),
                    "cam0_sensor_time_ns": (
                        int(cam0["sensor_time_ns"]) if cam0 else None
                    ),
                    "cam1_sensor_time_ns": (
                        int(cam1["sensor_time_ns"]) if cam1 else None
                    ),
                    "stereo_delta_ms": stereo_delta_ms,
                    "source_timestamp_span_ms": record.get("source_timestamp_span_ms"),
                    "camera_lidar_timestamp_span_ms": record.get(
                        "camera_lidar_timestamp_span_ms"
                    ),
                    "sync_max_abs_skew_ms": (
                        max(
                            (
                                abs(float(value))
                                for value in sync.get("relative_skew_ms", {}).values()
                            ),
                            default=None,
                        )
                    ),
                    "error_count": len(record_errors),
                    "missing_required_files": missing_for_frame,
                    "has_cam0": cam0 is not None,
                    "has_cam1": cam1 is not None,
                    "has_lidar": bool(lidar_items),
                    "has_map_pose": "map" in poses,
                    "has_head_camera_pose": "head_camera" in poses,
                    "lidar_point_count": lidar_point_count,
                    "lidar_zero_ratio": lidar_zero_ratio,
                    "lidar_nonfinite_ratio": lidar_nonfinite_ratio,
                }
            )

    ticks = [row["tick"] for row in rows]
    expected_ticks = set(range(min(ticks), max(ticks) + 1)) if ticks else set()
    missing_ticks = sorted(expected_ticks.difference(ticks))
    stereo_outliers = [
        row for row in rows if row["stereo_delta_ms"] is not None and row["stereo_delta_ms"] > 10.0
    ]
    window_rows = [row for row in rows if 653 <= row["tick"] <= 953]
    summary = {
        "manifest_records": len(rows),
        "tick_minimum": min(ticks) if ticks else None,
        "tick_maximum": max(ticks) if ticks else None,
        "unique_ticks": len(set(ticks)),
        "missing_ticks": missing_ticks,
        "duplicate_tick_count": len(ticks) - len(set(ticks)),
        "image_presence": dict(image_presence),
        "lidar_presence": dict(lidar_presence),
        "pose_presence": dict(pose_presence),
        "frames_with_recorded_errors": error_frames,
        "missing_required_file_count": len(missing_paths),
        "missing_required_files": missing_paths[:100],
        "stereo_delta_ms": percentile_summary(
            row["stereo_delta_ms"] for row in rows
        ),
        "stereo_delta_over_10ms_count": len(stereo_outliers),
        "stereo_delta_over_10ms_ticks": [
            row["tick"] for row in stereo_outliers
        ],
        "window_653_953": {
            "records": len(window_rows),
            "stereo_delta_ms": percentile_summary(
                row["stereo_delta_ms"] for row in window_rows
            ),
            "stereo_delta_over_10ms_count": sum(
                1
                for row in window_rows
                if row["stereo_delta_ms"] is not None
                and row["stereo_delta_ms"] > 10.0
            ),
            "largest_stereo_delta_frames": sorted(
                (
                    {
                        "tick": row["tick"],
                        "stereo_delta_ms": row["stereo_delta_ms"],
                        "source_timestamp_span_ms": row[
                            "source_timestamp_span_ms"
                        ],
                    }
                    for row in window_rows
                    if row["stereo_delta_ms"] is not None
                ),
                key=lambda item: item["stereo_delta_ms"],
                reverse=True,
            )[:20],
            "requested_anomaly_ticks": [
                row for row in window_rows if row["tick"] in (774, 948)
            ],
        },
        "lidar_scan": {
            "status": "exact" if scan_lidar else "unavailable",
            "frames_scanned": sum(
                row["lidar_zero_ratio"] is not None for row in rows
            ),
            "load_error_count": len(lidar_load_errors),
            "load_errors": lidar_load_errors[:100],
            "zero_point_ratio": percentile_summary(
                row["lidar_zero_ratio"] for row in rows
            ),
            "nonfinite_point_ratio": percentile_summary(
                row["lidar_nonfinite_ratio"] for row in rows
            ),
        },
        "measurement_evidence": "exact",
        "accuracy": accuracy_unavailable("原始数据传感器真实性与场景覆盖准确率"),
    }
    return rows, summary


def temporal_pair_map(report: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(pair["reference_frame"]): pair
        for pair in report.get("pairs", [])
        if int(pair.get("neighbor_offset", 1)) == 1
    }


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(require_file(path)), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode {path}")
    return image


def semantic_frame_scan(
    semantic_dir: Path, frame_count: int
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], Counter[int]]:
    label_dir = require_dir(semantic_dir / "semantic_sidecar" / "label_frames")
    dynamic_dir = require_dir(semantic_dir / "dynamic_masks")
    unknown_dir = require_dir(semantic_dir / "unknown_masks")
    static_depth_dir = require_dir(semantic_dir / "static_depth")
    rows: dict[int, dict[str, Any]] = {}
    label_histogram: Counter[int] = Counter()
    shapes = Counter()
    for frame_idx in range(frame_count):
        stem = f"{frame_idx:08d}"
        metadata = load_json(label_dir / f"{stem}.json")
        label = read_image(label_dir / f"{stem}.png")
        dynamic = read_image(dynamic_dir / f"{stem}.png")
        unknown = read_image(unknown_dir / f"{stem}.png")
        static_depth = read_image(static_depth_dir / f"{stem}.png")
        if label.ndim != 2:
            raise ValueError(f"label image must be 2D: {stem}, {label.shape}")
        for name, image in (
            ("dynamic", dynamic),
            ("unknown", unknown),
            ("static_depth", static_depth),
        ):
            if image.shape[:2] != label.shape:
                raise ValueError(
                    f"{name} shape mismatch at {stem}: {image.shape} vs {label.shape}"
                )
        shapes[tuple(label.shape)] += 1
        unique, counts = np.unique(label, return_counts=True)
        for semantic_id, count in zip(unique.tolist(), counts.tolist()):
            label_histogram[int(semantic_id)] += int(count)
        nonzero_label_mask = label != 0
        dynamic_mask = dynamic != 0
        unknown_mask = unknown != 0
        static_valid = static_depth != 0
        nonzero_counts = counts[unique != 0]
        if nonzero_counts.size:
            probabilities = nonzero_counts.astype(np.float64)
            probabilities /= probabilities.sum()
            entropy = float(-np.sum(probabilities * np.log2(probabilities)))
        else:
            entropy = 0.0
        rows[frame_idx] = {
            "static_depth_valid_ratio": float(np.mean(static_valid)),
            "dynamic_mask_ratio": float(np.mean(dynamic_mask)),
            "unknown_mask_ratio": float(np.mean(unknown_mask)),
            "dynamic_unknown_overlap_ratio": float(
                np.mean(dynamic_mask & unknown_mask)
            ),
            "label_nonzero_ratio": float(np.mean(nonzero_label_mask)),
            "label_nonzero_pixels": int(np.count_nonzero(nonzero_label_mask)),
            "label_unique_nonzero": int(np.count_nonzero(unique != 0)),
            "label_maximum": int(label.max()) if label.size else 0,
            "label_entropy_bits_nonzero": entropy,
            "label_metadata_nonzero_pixels": metadata.get("nonzero_pixels"),
            "label_metadata_sha256_matches": (
                metadata.get("image_sha256") == sha256_file(label_dir / f"{stem}.png")
            ),
        }

    total_pixels = sum(label_histogram.values())
    nonzero_pixels = total_pixels - label_histogram.get(0, 0)
    top_labels = [
        {
            "semantic_id": semantic_id,
            "pixels": pixels,
            "ratio_of_all_pixels": pixels / total_pixels if total_pixels else None,
            "ratio_of_labeled_pixels": (
                pixels / nonzero_pixels if nonzero_pixels and semantic_id != 0 else None
            ),
        }
        for semantic_id, pixels in label_histogram.most_common(31)
    ]
    summary = {
        "frames_scanned": len(rows),
        "image_shapes": [
            {"shape": list(shape), "frames": count}
            for shape, count in sorted(shapes.items())
        ],
        "static_depth_valid_ratio": percentile_summary(
            row["static_depth_valid_ratio"] for row in rows.values()
        ),
        "dynamic_mask_ratio": percentile_summary(
            row["dynamic_mask_ratio"] for row in rows.values()
        ),
        "unknown_mask_ratio": percentile_summary(
            row["unknown_mask_ratio"] for row in rows.values()
        ),
        "label_nonzero_ratio": percentile_summary(
            row["label_nonzero_ratio"] for row in rows.values()
        ),
        "label_unique_nonzero_per_frame": percentile_summary(
            row["label_unique_nonzero"] for row in rows.values()
        ),
        "frames_without_nonzero_labels": [
            frame_idx
            for frame_idx, row in rows.items()
            if row["label_nonzero_pixels"] == 0
        ],
        "label_png_metadata_sha256_mismatch_count": sum(
            not row["label_metadata_sha256_matches"] for row in rows.values()
        ),
        "global_unique_label_count_including_zero": len(label_histogram),
        "global_unique_nonzero_label_count": len(
            [label for label in label_histogram if label != 0]
        ),
        "global_nonzero_label_pixels": nonzero_pixels,
        "top_label_pixel_counts": top_labels,
        "measurement_evidence": "exact",
        "semantic_accuracy": accuracy_unavailable(
            "逐像素语义标签的 IoU、precision、recall 与类别真实性"
        ),
    }
    return rows, summary, label_histogram


def sqlite_map_memory_summary(database_path: Path) -> dict[str, Any]:
    # Ordinary mode=ro may still create/touch a WAL shared-memory sidecar.
    # This completed run has an empty WAL, so immutable=1 provides a strictly
    # read-only audit view of the durable database file.
    uri = f"file:{database_path.as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        table_names = [
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        table_counts = {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in table_names
        }

        def grouped(query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
            return [dict(row) for row in connection.execute(query, params).fetchall()]

        geometry_confidences = [
            row[0]
            for row in connection.execute(
                "SELECT geometry_confidence FROM entities WHERE deleted_ns IS NULL"
            ).fetchall()
        ]
        observations_per_entity = [
            row[0]
            for row in connection.execute(
                "SELECT COUNT(*) FROM entity_observations GROUP BY entity_id"
            ).fetchall()
        ]
        metadata_rows = grouped("SELECT key, value FROM metadata ORDER BY key")
        metadata: dict[str, Any] = {}
        for row in metadata_rows:
            try:
                metadata[row["key"]] = json.loads(row["value"])
            except (TypeError, json.JSONDecodeError):
                metadata[row["key"]] = row["value"]

        result = {
            "open_mode": "sqlite URI mode=ro&immutable=1 + PRAGMA query_only=ON",
            "integrity_check": [row[0] for row in integrity],
            "tables": table_counts,
            "active_entities": int(
                connection.execute(
                    "SELECT COUNT(*) FROM entities WHERE deleted_ns IS NULL"
                ).fetchone()[0]
            ),
            "deleted_entities": int(
                connection.execute(
                    "SELECT COUNT(*) FROM entities WHERE deleted_ns IS NOT NULL"
                ).fetchone()[0]
            ),
            "unknown_named_active_entities": int(
                connection.execute(
                    "SELECT COUNT(*) FROM entities "
                    "WHERE deleted_ns IS NULL AND lower(canonical_name)='unknown'"
                ).fetchone()[0]
            ),
            "active_entity_name_counts_top30": grouped(
                "SELECT canonical_name, COUNT(*) AS count FROM entities "
                "WHERE deleted_ns IS NULL GROUP BY canonical_name "
                "ORDER BY count DESC, canonical_name LIMIT 30"
            ),
            "geometry_confidence": percentile_summary(geometry_confidences),
            "entity_observations": {
                "total": table_counts.get("entity_observations", 0),
                "distinct_entities": int(
                    connection.execute(
                        "SELECT COUNT(DISTINCT entity_id) FROM entity_observations"
                    ).fetchone()[0]
                ),
                "per_entity": percentile_summary(observations_per_entity),
            },
            "semantic_operations_by_status": grouped(
                "SELECT status, COUNT(*) AS count FROM semantic_operations "
                "GROUP BY status ORDER BY status"
            ),
            "semantic_operations_by_source": grouped(
                "SELECT source, COUNT(*) AS count FROM semantic_operations "
                "GROUP BY source ORDER BY count DESC, source"
            ),
            "semantic_delivery_by_status": grouped(
                "SELECT status, COUNT(*) AS count, "
                "SUM(CASE WHEN last_error IS NOT NULL THEN 1 ELSE 0 END) AS with_error "
                "FROM semantic_deliveries GROUP BY status ORDER BY status"
            ),
            "audit_actions": grouped(
                "SELECT action, COUNT(*) AS count FROM audit_log "
                "GROUP BY action ORDER BY count DESC, action"
            ),
            "sessions": grouped(
                "SELECT status, COUNT(*) AS count FROM sessions GROUP BY status"
            ),
            "revisions": {
                "count": table_counts.get("revisions", 0),
                "minimum": connection.execute(
                    "SELECT MIN(revision) FROM revisions"
                ).fetchone()[0],
                "maximum": connection.execute(
                    "SELECT MAX(revision) FROM revisions"
                ).fetchone()[0],
            },
            "metadata": metadata,
            "measurement_evidence": "exact",
            "entity_accuracy": accuracy_unavailable(
                "MapMemory entity precision/recall、over-merge 与 over-split"
            ),
        }
        return result
    finally:
        connection.close()


def hydra_dsg_summary(dsg_path: Path) -> dict[str, Any]:
    dsg = load_json(dsg_path)
    layer_names = dsg.get("layer_names", {})
    lookup = {
        (int(value["layer"]), int(value["partition"])): name
        for name, value in layer_names.items()
    }
    layer_counts: Counter[str] = Counter()
    attribute_types: Counter[str] = Counter()
    object_names: Counter[str] = Counter()
    object_semantic_labels: Counter[int] = Counter()
    object_mesh_vertices: list[int] = []
    for node in dsg.get("nodes", []):
        key = (int(node.get("layer", -1)), int(node.get("partition", 0)))
        layer_counts[lookup.get(key, f"layer_{key[0]}_partition_{key[1]}")] += 1
        attributes = node.get("attributes", {})
        attribute_type = str(attributes.get("type", "unknown"))
        attribute_types[attribute_type] += 1
        if "ObjectAttributes" in attribute_type:
            name = str(attributes.get("name", "")).strip()
            if name:
                object_names[name] += 1
            if attributes.get("semantic_label") is not None:
                object_semantic_labels[int(attributes["semantic_label"])] += 1
            mesh = attributes.get("mesh", {})
            points = mesh.get("points", []) if isinstance(mesh, dict) else []
            object_mesh_vertices.append(len(points))
    return {
        "nodes": len(dsg.get("nodes", [])),
        "edges": len(dsg.get("edges", [])),
        "layer_counts": dict(sorted(layer_counts.items())),
        "attribute_type_counts": dict(sorted(attribute_types.items())),
        "object_nodes": sum(
            count
            for attribute_type, count in attribute_types.items()
            if "ObjectAttributes" in attribute_type
        ),
        "named_object_nodes": sum(object_names.values()),
        "unique_object_names": len(object_names),
        "unique_object_semantic_labels": len(object_semantic_labels),
        "object_mesh_vertices": percentile_summary(object_mesh_vertices),
        "top_object_names": [
            {"name": name, "count": count}
            for name, count in object_names.most_common(20)
        ],
        "measurement_evidence": "exact",
        "object_accuracy": accuracy_unavailable(
            "Hydra object 节点的类别、实例及 mesh 绑定准确率"
        ),
    }


def parse_ply_header(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"format": None, "elements": {}}
    with require_file(path).open("rb") as stream:
        for raw_line in stream:
            line = raw_line.decode("ascii", errors="replace").strip()
            if line.startswith("format "):
                result["format"] = line.removeprefix("format ")
            elif line.startswith("element "):
                _, name, count = line.split()
                result["elements"][name] = int(count)
            elif line == "end_header":
                break
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        ""
                        if row.get(field) is None
                        else json.dumps(row[field], ensure_ascii=False)
                        if isinstance(row.get(field), (list, dict))
                        else row.get(field)
                    )
                    for field in fields
                }
            )


def make_per_frame_rows(
    tick_frames: Sequence[Mapping[str, Any]],
    depth_stats: Sequence[Mapping[str, Any]],
    temporal_before: Mapping[int, Mapping[str, Any]],
    temporal_after: Mapping[int, Mapping[str, Any]],
    filter_stats: Sequence[Mapping[str, Any]],
    semantic_stats: Mapping[int, Mapping[str, Any]],
    raw_by_tick: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    depth_by_frame = {int(item["frame_idx"]): item for item in depth_stats}
    filter_by_frame = {int(item["frame"]): item for item in filter_stats}
    rows: list[dict[str, Any]] = []
    previous_sensor_time_ns: int | None = None
    previous_source_idx: int | None = None
    first_sensor_time_ns = int(tick_frames[0]["sensor_time_ns"])
    for frame in tick_frames:
        frame_idx = int(frame["idx"])
        source_idx = int(frame["source_idx"])
        sensor_time_ns = int(frame["sensor_time_ns"])
        depth = depth_by_frame.get(frame_idx, {})
        before = temporal_before.get(frame_idx, {})
        after = temporal_after.get(frame_idx, {})
        filtered = filter_by_frame.get(frame_idx, {})
        semantic = semantic_stats.get(frame_idx, {})
        raw = raw_by_tick.get(source_idx, {})
        rows.append(
            {
                "frame_idx": frame_idx,
                "source_idx": source_idx,
                "source_frame_idx": frame.get("source_frame_idx"),
                "sensor_time_ns": sensor_time_ns,
                "elapsed_s": (sensor_time_ns - first_sensor_time_ns) / 1.0e9,
                "selection_reason": frame.get("selection_reason"),
                "stereo_delta_ms": frame.get("stereo_delta_ms"),
                "raw_stereo_delta_ms": raw.get("stereo_delta_ms"),
                "selected_gap_from_previous_s": (
                    (sensor_time_ns - previous_sensor_time_ns) / 1.0e9
                    if previous_sensor_time_ns is not None
                    else None
                ),
                "source_gap_from_previous": (
                    source_idx - previous_source_idx
                    if previous_source_idx is not None
                    else None
                ),
                "raw_lidar_zero_ratio": raw.get("lidar_zero_ratio"),
                "raw_lidar_nonfinite_ratio": raw.get("lidar_nonfinite_ratio"),
                "depth_valid_ratio": depth.get("valid_ratio"),
                "depth_median_m": depth.get("median_depth_m"),
                "depth_left_right_consistency": depth.get(
                    "left_right_consistency"
                ),
                "depth_mean_confidence": depth.get("mean_confidence"),
                "depth_occlusion_ratio": depth.get("occlusion_ratio"),
                "depth_raw_positive_disparity_ratio": depth.get(
                    "raw_positive_disparity_ratio"
                ),
                "depth_raw_visible_ratio": depth.get("raw_visible_depth_ratio"),
                "depth_raw_within_5m_ratio": depth.get(
                    "raw_depth_coverage_ratio", {}
                ).get("within_5m"),
                "depth_raw_within_10m_ratio": depth.get(
                    "raw_depth_coverage_ratio", {}
                ).get("within_10m"),
                "depth_raw_within_30m_ratio": depth.get(
                    "raw_depth_coverage_ratio", {}
                ).get("within_30m"),
                "depth_end_to_end_s": depth.get("timing_seconds", {}).get(
                    "end_to_end_wall"
                ),
                "temporal_before_agreement": before.get("agreement_rate"),
                "temporal_before_comparable_samples": before.get(
                    "comparable_samples"
                ),
                "temporal_before_median_abs_error_m": before.get(
                    "median_absolute_depth_error_m"
                ),
                "temporal_before_p95_abs_error_m": before.get(
                    "p95_absolute_depth_error_m"
                ),
                "temporal_before_pose_translation_m": before.get(
                    "pose_translation_m"
                ),
                "temporal_before_pose_rotation_deg": before.get(
                    "pose_rotation_deg"
                ),
                "temporal_after_agreement": after.get("agreement_rate"),
                "temporal_after_comparable_samples": after.get(
                    "comparable_samples"
                ),
                "temporal_after_median_abs_error_m": after.get(
                    "median_absolute_depth_error_m"
                ),
                "temporal_after_p95_abs_error_m": after.get(
                    "p95_absolute_depth_error_m"
                ),
                "filter_judged_ratio": filtered.get(
                    "low_resolution_judged_ratio"
                ),
                "filter_supported_ratio": filtered.get(
                    "low_resolution_supported_ratio"
                ),
                "filter_input_valid_ratio": filtered.get("input_valid_ratio"),
                "filter_output_valid_ratio": filtered.get("output_valid_ratio"),
                "filter_rejected_valid_ratio": filtered.get(
                    "rejected_valid_ratio"
                ),
                **semantic,
            }
        )
        previous_sensor_time_ns = sensor_time_ns
        previous_source_idx = source_idx
    return rows


def build_failure_cases(
    raw_rows: Sequence[Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
    trajectory: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    map_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(
        case_id: str,
        stage: str,
        category: str,
        severity: str,
        metrics: Mapping[str, Any],
        interpretation: str,
        frame_idx: int | None = None,
        source_idx: int | None = None,
    ) -> None:
        if case_id in seen:
            return
        seen.add(case_id)
        cases.append(
            {
                "case_id": case_id,
                "stage": stage,
                "category": category,
                "severity": severity,
                "frame_idx": frame_idx,
                "source_idx": source_idx,
                "metrics": dict(metrics),
                "measurement_evidence": "exact",
                "interpretation_evidence": "proxy",
                "interpretation": interpretation,
                "accuracy": accuracy_unavailable(
                    f"{stage}/{category} 是否对应真实场景错误"
                ),
            }
        )

    for row in raw_rows:
        delta = row.get("stereo_delta_ms")
        if delta is not None and float(delta) > 10.0:
            add(
                f"raw-stereo-{int(row['tick']):04d}",
                "raw_sync",
                "stereo_delta_over_10ms",
                "high" if float(delta) >= 50.0 else "medium",
                {
                    "stereo_delta_ms": delta,
                    "source_timestamp_span_ms": row.get(
                        "source_timestamp_span_ms"
                    ),
                },
                "超过后续针孔双目链使用的 10 ms 同步门；该 raw tick 未必进入物化/选择集。",
                source_idx=int(row["tick"]),
            )

    gap_candidates = sorted(
        (row for row in frame_rows if row.get("selected_gap_from_previous_s") is not None),
        key=lambda row: float(row["selected_gap_from_previous_s"]),
        reverse=True,
    )[:20]
    for row in gap_candidates:
        if float(row["selected_gap_from_previous_s"]) < 0.2:
            continue
        add(
            f"selected-gap-{int(row['frame_idx']):04d}",
            "selection",
            "large_selected_time_gap",
            "medium",
            {
                "gap_s": row["selected_gap_from_previous_s"],
                "source_gap": row["source_gap_from_previous"],
                "selection_reason": row["selection_reason"],
            },
            "选择后时间空洞会减弱相邻帧时序支持；这是覆盖 proxy，不是真实运动误差。",
            int(row["frame_idx"]),
            int(row["source_idx"]),
        )

    for row in sorted(
        frame_rows, key=lambda item: float(item.get("depth_valid_ratio") or 1.0)
    )[:15]:
        add(
            f"depth-low-valid-{int(row['frame_idx']):04d}",
            "depth",
            "low_valid_ratio",
            "medium",
            {
                "valid_ratio": row.get("depth_valid_ratio"),
                "median_depth_m": row.get("depth_median_m"),
                "left_right_consistency": row.get(
                    "depth_left_right_consistency"
                ),
            },
            "该帧深度有效覆盖位于全序列低端；没有 LiDAR/深度 GT，不能判定数值深度准确性。",
            int(row["frame_idx"]),
            int(row["source_idx"]),
        )

    temporal_candidates = sorted(
        (
            row
            for row in frame_rows
            if row.get("temporal_after_agreement") is not None
        ),
        key=lambda row: float(row["temporal_after_agreement"]),
    )
    for row in temporal_candidates:
        if (
            float(row["temporal_after_agreement"]) >= 0.7
            and temporal_candidates.index(row) >= 15
        ):
            continue
        add(
            f"temporal-low-{int(row['frame_idx']):04d}",
            "temporal_validation",
            "low_adjacent_agreement",
            "high" if float(row["temporal_after_agreement"]) < 0.7 else "medium",
            {
                "before_agreement": row.get("temporal_before_agreement"),
                "after_agreement": row.get("temporal_after_agreement"),
                "after_median_abs_error_m": row.get(
                    "temporal_after_median_abs_error_m"
                ),
                "pose_rotation_deg": row.get(
                    "temporal_before_pose_rotation_deg"
                ),
            },
            "重投影时序一致性偏低，可能由运动、遮挡、pose 或 depth 共同造成；仅为失效定位 proxy。",
            int(row["frame_idx"]),
            int(row["source_idx"]),
        )

    for row in sorted(
        frame_rows,
        key=lambda item: float(item.get("filter_rejected_valid_ratio") or 0.0),
        reverse=True,
    )[:15]:
        add(
            f"filter-high-reject-{int(row['frame_idx']):04d}",
            "temporal_filter",
            "high_rejected_valid_ratio",
            "medium",
            {
                "input_valid_ratio": row.get("filter_input_valid_ratio"),
                "output_valid_ratio": row.get("filter_output_valid_ratio"),
                "rejected_valid_ratio": row.get("filter_rejected_valid_ratio"),
            },
            "过滤删除比例位于高端；删除是否正确需要独立深度 GT。",
            int(row["frame_idx"]),
            int(row["source_idx"]),
        )

    for row in frame_rows:
        unknown_ratio = float(row.get("unknown_mask_ratio") or 0.0)
        no_labels = int(row.get("label_nonzero_pixels") or 0) == 0
        if unknown_ratio > 0.6:
            add(
                f"mask-unknown-{int(row['frame_idx']):04d}",
                "dynamic_semantic",
                "unknown_ratio_over_quality_gate",
                "high",
                {
                    "unknown_mask_ratio": unknown_ratio,
                    "dynamic_mask_ratio": row.get("dynamic_mask_ratio"),
                    "label_nonzero_ratio": row.get("label_nonzero_ratio"),
                },
                "unknown mask 超过 0.6 质量阈值；单帧异常不等价于总体 gate 失败。",
                int(row["frame_idx"]),
                int(row["source_idx"]),
            )
        if no_labels:
            add(
                f"semantic-empty-{int(row['frame_idx']):04d}",
                "semantic_labels",
                "no_nonzero_labels",
                "medium",
                {
                    "label_nonzero_pixels": 0,
                    "dynamic_mask_ratio": row.get("dynamic_mask_ratio"),
                    "unknown_mask_ratio": row.get("unknown_mask_ratio"),
                },
                "postpass 中该帧无非零语义标签；可能是无候选、传播失效或可见内容原因。",
                int(row["frame_idx"]),
                int(row["source_idx"]),
            )

    if not bool(trajectory.get("optimizer_success")):
        add(
            "e6-local-optimizer-fallback",
            "E6",
            "optimizer_nonconvergence",
            "high",
            {
                "optimizer_success": trajectory.get("optimizer_success"),
                "optimizer_message": trajectory.get("optimizer_message"),
                "optimizer_nfev": trajectory.get("optimizer_nfev"),
                "candidate_visual_residual_before_m": trajectory.get(
                    "visual_residual_median_before_m"
                ),
                "candidate_visual_residual_after_m": trajectory.get(
                    "visual_residual_median_after_m"
                ),
            },
            "候选优化未收敛，按 E6 写回规则必须 fallback；后续 source 模式确实未写回。",
        )

    for result in quality_report.get("results", []):
        if result.get("status") == "FAIL":
            add(
                f"quality-{result.get('code', 'unknown')}",
                str(result.get("stage", "quality")),
                str(result.get("code", "quality_failure")),
                "high" if result.get("hard") else "medium",
                result.get("metrics", {}),
                str(result.get("message", "quality gate failure")),
            )

    if int(map_metrics.get("connected_components", 0)) > 1000:
        add(
            "hydra-high-component-count",
            "hydra_mesh",
            "connected_components_over_nominal_1000",
            "medium",
            {
                "connected_components": map_metrics.get("connected_components"),
                "significant_connected_components": map_metrics.get(
                    "significant_connected_components"
                ),
                "tiny_component_area_ratio": map_metrics.get(
                    "tiny_component_area_ratio"
                ),
                "largest_component_area_ratio": map_metrics.get(
                    "largest_component_area_ratio"
                ),
            },
            "原始连通分量略高，但显著分量/微小面积指标通过；几何真实性仍无 GT。",
        )

    severity_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        cases,
        key=lambda case: (
            severity_order.get(case["severity"], 99),
            case["stage"],
            case["case_id"],
        ),
    )


def setup_plot() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_depth(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    x = [row["frame_idx"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(x, [row["depth_valid_ratio"] for row in rows], lw=1.1)
    axes[0].set_ylabel("valid ratio")
    axes[0].set_title("Depth coverage (direct measurement; accuracy unavailable)")
    axes[1].plot(
        x,
        [row["depth_left_right_consistency"] for row in rows],
        color="#2a9d8f",
        lw=1.1,
        label="left-right consistency",
    )
    axes[1].plot(
        x,
        [row["depth_mean_confidence"] for row in rows],
        color="#e9c46a",
        lw=0.9,
        label="mean confidence",
    )
    axes[1].legend(loc="lower right")
    axes[1].set_ylabel("ratio")
    axes[2].plot(
        x,
        [row["depth_median_m"] for row in rows],
        color="#264653",
        lw=1.0,
    )
    axes[2].set_ylabel("median depth [m]")
    axes[2].set_xlabel("selected frame")
    save_figure(fig, output)


def plot_temporal(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    filtered = [
        row for row in rows if row.get("temporal_after_agreement") is not None
    ]
    x = [row["frame_idx"] for row in filtered]
    before = np.asarray(
        [row["temporal_before_agreement"] for row in filtered], dtype=float
    )
    after = np.asarray(
        [row["temporal_after_agreement"] for row in filtered], dtype=float
    )
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(x, before, lw=0.9, alpha=0.75, label="before filter")
    axes[0].plot(x, after, lw=1.0, label="after filter")
    axes[0].axhline(0.7, color="#d62828", ls="--", lw=1, label="diagnostic 0.70")
    axes[0].set_ylabel("adjacent agreement")
    axes[0].legend(ncol=3)
    axes[0].set_title("Temporal reprojection proxy (not depth accuracy)")
    axes[1].plot(x, after - before, color="#2a9d8f", lw=0.9)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("after - before")
    axes[1].set_xlabel("reference frame")
    save_figure(fig, output)


def plot_filter(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    x = [row["frame_idx"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(
        x,
        [row["filter_input_valid_ratio"] for row in rows],
        label="input valid",
        lw=0.9,
    )
    axes[0].plot(
        x,
        [row["filter_output_valid_ratio"] for row in rows],
        label="output valid",
        lw=0.9,
    )
    axes[0].legend()
    axes[0].set_ylabel("ratio")
    axes[0].set_title("Temporal filter effect (deletion correctness unavailable)")
    axes[1].plot(
        x,
        [row["filter_rejected_valid_ratio"] for row in rows],
        color="#e76f51",
        lw=0.9,
        label="rejected / input valid",
    )
    axes[1].plot(
        x,
        [row["filter_supported_ratio"] for row in rows],
        color="#2a9d8f",
        lw=0.8,
        alpha=0.8,
        label="low-res supported",
    )
    axes[1].legend()
    axes[1].set_ylabel("ratio")
    axes[1].set_xlabel("selected frame")
    save_figure(fig, output)


def plot_masks(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    x = [row["frame_idx"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(
        x,
        [row["dynamic_mask_ratio"] for row in rows],
        label="dynamic",
        lw=0.8,
    )
    axes[0].plot(
        x,
        [row["unknown_mask_ratio"] for row in rows],
        label="unknown",
        lw=0.8,
    )
    axes[0].plot(
        x,
        [row["static_depth_valid_ratio"] for row in rows],
        label="static depth valid",
        lw=0.9,
    )
    axes[0].legend(ncol=3)
    axes[0].set_ylabel("pixel ratio")
    axes[0].set_title("Dynamic isolation and semantic coverage")
    axes[1].plot(
        x,
        [row["label_nonzero_ratio"] for row in rows],
        color="#6a4c93",
        lw=0.9,
        label="nonzero label pixels",
    )
    axes[1].plot(
        x,
        [
            min(float(row["label_unique_nonzero"]) / 100.0, 1.0)
            for row in rows
        ],
        color="#ff9f1c",
        lw=0.7,
        alpha=0.7,
        label="unique labels / 100 (clipped)",
    )
    axes[1].legend()
    axes[1].set_ylabel("ratio / scaled count")
    axes[1].set_xlabel("selected frame")
    save_figure(fig, output)


def plot_runtime(
    quality_report: Mapping[str, Any],
    realtime_metrics: Mapping[str, Any],
    output: Path,
) -> None:
    runtime_result = next(
        (
            item
            for item in quality_report.get("results", [])
            if item.get("stage") == "runtime"
        ),
        {},
    )
    metrics = runtime_result.get("metrics", {})
    thresholds = runtime_result.get("thresholds", {})
    p95 = metrics.get("stage_p95_ms", {})
    limits = thresholds.get("stage_p95_limits_ms", {})
    stages = sorted(set(p95).intersection(limits))
    x = np.arange(len(stages))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    axes[0].bar(x - 0.2, [p95[s] for s in stages], width=0.4, label="service p95")
    axes[0].bar(
        x + 0.2,
        [limits[s] for s in stages],
        width=0.4,
        label="limit",
        color="#adb5bd",
    )
    axes[0].set_xticks(x, stages, rotation=25, ha="right")
    axes[0].set_ylabel("milliseconds")
    axes[0].legend()
    axes[0].set_title(
        f"Runtime quality gate: {runtime_result.get('status', 'unavailable')}"
    )
    runtime_stages = realtime_metrics.get("stages", {})
    stage_names = sorted(runtime_stages)
    axes[1].bar(
        np.arange(len(stage_names)),
        [runtime_stages[s].get("processed", 0) for s in stage_names],
        color="#457b9d",
    )
    axes[1].set_xticks(
        np.arange(len(stage_names)), stage_names, rotation=25, ha="right"
    )
    axes[1].set_ylabel("persisted samples")
    axes[1].set_title("Runtime evidence coverage (resume run metrics are partial)")
    save_figure(fig, output)


def plot_funnel(
    summary: Mapping[str, Any], quality_report: Mapping[str, Any], output: Path
) -> None:
    frame_names = ["raw", "prepared", "selected", "depth", "filtered", "postpass"]
    frame_values = [
        summary["raw"]["manifest_records"],
        summary["prepared"]["materialized_frames"],
        summary["selection"]["selected_frames"],
        summary["depth"]["processed_frames"],
        summary["filter"]["frame_count"],
        summary["semantic_postpass"]["frames_replayed"],
    ]
    semantic_names = ["MapMemory entities", "Hydra objects", "mesh-bound", "DSG ops"]
    semantic_values = [
        summary["map_memory"]["active_entities"],
        summary["hydra"]["dsg"]["object_nodes"],
        summary["hydra"]["commit"]["verified_entity_count"],
        summary["hydra"]["commit"]["verified_operation_count"],
    ]
    results = quality_report.get("results", [])
    quality_names = [item.get("stage", "?") for item in results]
    quality_values = [
        1 if item.get("status") == "PASS" else 0 for item in results
    ]
    quality_colors = [
        "#2a9d8f" if value else "#e63946" for value in quality_values
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].barh(frame_names[::-1], frame_values[::-1], color="#457b9d")
    axes[0].set_title("Frame evidence funnel")
    axes[0].set_xlabel("count")
    axes[1].barh(semantic_names[::-1], semantic_values[::-1], color="#6a4c93")
    axes[1].set_title("Semantic artifact counts")
    axes[1].set_xlabel("count (not one identity domain)")
    axes[2].barh(quality_names[::-1], quality_values[::-1], color=quality_colors[::-1])
    axes[2].set_xlim(0, 1.05)
    axes[2].set_xticks([0, 1], ["FAIL", "PASS"])
    axes[2].set_title("Internal quality gates")
    save_figure(fig, output)


def plot_failure_heatmap(
    rows: Sequence[Mapping[str, Any]], output: Path
) -> None:
    signals = {
        "1-depth valid": np.asarray(
            [1.0 - float(row["depth_valid_ratio"]) for row in rows]
        ),
        "1-temporal after": np.asarray(
            [
                1.0 - float(row["temporal_after_agreement"])
                if row.get("temporal_after_agreement") is not None
                else np.nan
                for row in rows
            ]
        ),
        "filter rejection": np.asarray(
            [float(row["filter_rejected_valid_ratio"]) for row in rows]
        ),
        "unknown mask": np.asarray(
            [float(row["unknown_mask_ratio"]) for row in rows]
        ),
        "1-label coverage": np.asarray(
            [1.0 - float(row["label_nonzero_ratio"]) for row in rows]
        ),
    }
    matrix = []
    for values in signals.values():
        finite = values[np.isfinite(values)]
        p95 = float(np.percentile(finite, 95)) if finite.size else 1.0
        scale = p95 if p95 > 1.0e-12 else 1.0
        matrix.append(np.clip(np.nan_to_num(values, nan=0.0) / scale, 0.0, 1.0))
    fig, ax = plt.subplots(figsize=(14, 4.8))
    image = ax.imshow(np.vstack(matrix), aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(signals)), list(signals))
    ax.set_xlabel("selected frame")
    ax.set_title("Failure-localization proxy (each row normalized by its p95)")
    fig.colorbar(image, ax=ax, label="normalized anomaly intensity")
    save_figure(fig, output)


def plot_lidar(raw_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    available = [
        row for row in raw_rows if row.get("lidar_zero_ratio") is not None
    ]
    fig, ax = plt.subplots(figsize=(13, 5))
    if available:
        ax.plot(
            [row["tick"] for row in available],
            [row["lidar_zero_ratio"] for row in available],
            lw=0.8,
            color="#264653",
        )
        ax.set_ylabel("all-zero XYZ point ratio")
    else:
        ax.text(
            0.5,
            0.5,
            "LiDAR scan unavailable (--skip-lidar-scan)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("raw tick")
    ax.set_title("Raw LiDAR zero-point ratio (format/data-quality measurement)")
    save_figure(fig, output)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "unavailable"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(
    output_dir: Path,
    summary: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    plot_files: Sequence[str],
) -> str:
    raw = summary["raw"]
    depth = summary["depth"]
    temporal = summary["temporal"]
    filt = summary["filter"]
    semantic = summary["semantics"]
    runtime = summary["runtime"]
    hydra = summary["hydra"]
    e6 = summary["experiments"]["E6"]
    e7 = summary["experiments"]["E7"]
    e8 = summary["experiments"]["E8"]
    requested_anomalies = raw["window_653_953"]["requested_anomaly_ticks"]
    largest_gap = summary["selection"]["largest_selected_time_gaps"][0]
    report = f"""# G1 语义地图全流程无 GT 只读审计

生成时间：`{provenance["generated_at_utc"]}`  
审计 schema：`{SCHEMA}`  
帧域：raw `1113` 帧；进入 geometry/semantic 的 selected `844` 帧。

## 结论边界

{NO_GT_REASON}

本报告中的 `PASS/FAIL` 只表示已有内部 gate 或 proxy 是否满足，不能替代 L0–L6 GT、
双人标注/裁决或独立 held-out。语义 IoU/precision/recall、实体 P/R/F1、深度绝对误差、
轨迹 ATE/RPE 和地图真实完整度均为 **unavailable**。

## 关键结果

{md_table(
    ["环节", "直接证据", "结果", "准确性结论"],
    [
        ["raw", f'{raw["manifest_records"]} records', f'missing files={raw["missing_required_file_count"]}', "unavailable"],
        ["prepared", f'{summary["prepared"]["materialized_frames"]} frames', f'joint valid={summary["prepared"]["joint_valid_area_ratio"]:.4f}', "proxy"],
        ["selection", f'{summary["selection"]["selected_frames"]} frames', f'max gap={largest_gap["gap_s"]:.3f}s', "proxy"],
        ["depth", f'{depth["processed_frames"]} frames', f'mean valid={depth["valid_ratio"]["mean"]:.4f}', "unavailable"],
        ["temporal", "843 adjacent pairs", f'{temporal["before_weighted_agreement"]:.4f} → {temporal["after_weighted_agreement"]:.4f}', "proxy"],
        ["filter", f'{filt["frame_count"]} frames', f'rejected pixels={filt["rejected_pixels"]}', "unavailable"],
        ["semantics", f'{semantic["frames_scanned"]} label/mask frames', f'nonzero label frames={summary["semantic_postpass"]["nonzero_label_frames"]}', "unavailable"],
        ["runtime", f'{runtime["persisted_runtime_frame_samples"]} resumed-frame samples', f'quality_passed={runtime["quality_passed"]}', "exact timing / partial coverage"],
        ["Hydra", f'{hydra["mesh"]["vertices"]} vertices, {hydra["mesh"]["faces"]} faces', f'{hydra["dsg"]["object_nodes"]} object nodes', "unavailable"],
    ],
)}

## 原始数据、同步与选择覆盖

- `quality_report.json` 和逐行 `manifest.jsonl` 都给出 1113 个记录；cam0/cam1/LiDAR、
  map/head pose 的存在性与磁盘路径逐帧核验，缺失文件数为
  `{raw["missing_required_file_count"]}`。
- raw 双目 delta > 10 ms 共 `{raw["stereo_delta_over_10ms_count"]}` 帧；653–953
  区间内为 `{raw["window_653_953"]["stereo_delta_over_10ms_count"]}` 帧。
- source 774/948 的直接证据：

{md_table(
    ["source tick", "stereo delta [ms]", "source span [ms]", "进入 prepared/selected"],
    [
        [
            item["tick"],
            item["stereo_delta_ms"],
            item["source_timestamp_span_ms"],
            item["tick"] in summary["prepared"]["source_indices"],
        ]
        for item in requested_anomalies
    ],
)}

- selected 最大相邻时间空洞为 `{largest_gap["gap_s"]:.6f} s`，发生在 frame
  `{largest_gap["previous_frame_idx"]} → {largest_gap["frame_idx"]}`，raw source
  `{largest_gap["previous_source_idx"]} → {largest_gap["source_idx"]}`。这影响时序覆盖，
  但不能单独证明 pose/depth 错误。
- LiDAR 零点采用逐 `.npy` mmap 流式统计：状态
  `{raw["lidar_scan"]["status"]}`，扫描 `{raw["lidar_scan"]["frames_scanned"]}` 帧，
  全零点比例中位数 `{raw["lidar_scan"]["zero_point_ratio"]["p50"]}`。

## 深度、时序与过滤

- FoundationStereo：`{depth["processed_frames"]}/844`，failed
  `{depth["failed_frames"]}`；valid ratio p05/p50/p95 =
  `{depth["valid_ratio"]["p05"]:.4f}/{depth["valid_ratio"]["p50"]:.4f}/{depth["valid_ratio"]["p95"]:.4f}`。
- 左右一致性为模型内交叉检查，不是 GT；均值
  `{depth["left_right_consistency"]["mean"]:.4f}`。
- 相邻重投影 agreement 加权值由
  `{temporal["before_weighted_agreement"]:.6f}` 提升到
  `{temporal["after_weighted_agreement"]:.6f}`；这是 pose+depth+遮挡共同作用的 proxy。
- filter 后有效率 p50 `{filt["output_valid_ratio"]["p50"]:.4f}`，相对输入有效像素
  删除率 p50 `{filt["rejected_valid_ratio"]["p50"]:.4f}`。删除正确率无 GT，unavailable。

## E6 / E7 / E8

{md_table(
    ["实验", "proxy 状态", "直接证据", "严格解释"],
    [
        ["E6", e6["status"], e6["claim"], "局部优化未收敛；source fallback 未写回，pose accuracy unavailable"],
        ["E7", e7["status"], e7["claim"], "582 retrieval / 80 dense / 0 verified；无 loop GT"],
        ["E8", e8["status"], e8["claim"], "0 selected loop 且 source 轨迹变化为 0；不得宣称 global gain"],
    ],
)}

## 动态 mask、标签与 MapMemory

- 844 帧 dynamic/unknown/static-depth/uint16 label 全像素扫描。unknown ratio
  p50/p95 = `{semantic["unknown_mask_ratio"]["p50"]:.4f}/`
  `{semantic["unknown_mask_ratio"]["p95"]:.4f}`；label nonzero ratio
  p50/p95 = `{semantic["label_nonzero_ratio"]["p50"]:.4f}/`
  `{semantic["label_nonzero_ratio"]["p95"]:.4f}`。
- exact-label postpass `{summary["semantic_postpass"]["frames_replayed"]}/`
  `{summary["semantic_postpass"]["frames_expected"]}`，覆盖
  `{summary["semantic_postpass"]["label_coverage"]:.3f}`；无非零标签的帧共
  `{len(semantic["frames_without_nonzero_labels"])}`。
- MapMemory 以 SQLite `mode=ro&immutable=1` + `query_only` 打开，integrity check =
  `{summary["map_memory"]["integrity_check"]}`；active entities
  `{summary["map_memory"]["active_entities"]}`，unknown named
  `{summary["map_memory"]["unknown_named_active_entities"]}`，observations
  `{summary["map_memory"]["entity_observations"]["total"]}`。
- 这些 entity/label 是自动输出，无法据此计算 instance accuracy、over-merge 或
  over-split。

## Runtime、Hydra mesh / DSG / postpass

- overall quality gate：`{summary["quality"]["passed"]}`；硬失败
  `{summary["quality"]["hard_failures"]}`。唯一已持久化硬失败是 global service
  P95 `{runtime["global_service_p95_ms"]:.3f} ms` > 250 ms。
- 运行从 frame 600 checkpoint 恢复；runtime histogram 只覆盖本次 dispatch 的
  `{runtime["persisted_runtime_frame_samples"]}` 帧，不可误写成 844 帧完整 runtime 分布。
- mesh：`{hydra["mesh"]["vertices"]}` vertices /
  `{hydra["mesh"]["faces"]}` faces，显著连通分量
  `{hydra["mesh"]["significant_connected_components"]}`，tiny area ratio
  `{hydra["mesh"]["tiny_component_area_ratio"]:.6f}`。
- DSG：`{hydra["dsg"]["nodes"]}` nodes / `{hydra["dsg"]["edges"]}` edges /
  `{hydra["dsg"]["object_nodes"]}` object nodes。commit 验证 entity
  `{hydra["commit"]["verified_entity_count"]}`、operation
  `{hydra["commit"]["verified_operation_count"]}`、rejected
  `{hydra["commit"]["verified_rejected_operation_count"]}`。
- postpass 全量 replay 成功不等价于语义准确；它只精确证明标签持久化、重放和最终
  DSG commit 的内部闭环。

## 失败案例与误差传播

共保存 `{len(failures)}` 个结构化 failure cases，其中 high
`{sum(case["severity"] == "high" for case in failures)}`、medium
`{sum(case["severity"] == "medium" for case in failures)}`。完整记录见
`failure_cases.jsonl`。

典型传播链：

1. raw stereo delta/选择空洞影响左右匹配和可比较的时序邻居；
2. depth 低覆盖或左右不一致进入 temporal reprojection；
3. temporal filter 删除不一致像素，随后 dynamic/unknown mask 再削减静态融合深度；
4. label 空帧、entity merge 与 mesh binding 影响最终 Hydra object/DSG；
5. global object rebuild 提高 runtime P95，导致功能完成但 quality gate 失败。

这条链是基于产物相关性的定位 proxy，不是因果或真实误差定量证明。

## 可视化

"""
    for plot in plot_files:
        report += f"![{plot}]({plot})\n\n"
    report += """## 证据文件

- `per_frame_metrics.csv`：844 帧 tick/depth/temporal/filter/mask/label 联表。
- `raw_frame_metrics.csv`：1113 帧 raw 同步、完整性与 LiDAR 流式统计。
- `stage_summary.json`：各阶段聚合、E6/E7/E8、runtime、Hydra、质量边界。
- `failure_cases.jsonl`：结构化失败案例。
- `map_memory_summary.json`：只读 SQLite 表计数及分布。
- `semantic_label_histogram.csv`：全量 label pixel histogram。
- `runtime_stage_metrics.csv`：持久化 runtime stage 指标。
- `provenance.json`、`command.txt`：命令、环境、精选输入 SHA-256 与只读稳定性。

注意：`provenance.json` 只哈希精选 contract/report/DB/DSG/mesh 文件，不是上游全文件
hash inventory；没有重新哈希每一张 RGB/depth/mask。
"""
    return report


def markdown_to_html(markdown_text: str, plot_files: Sequence[str]) -> str:
    # Purpose-built readable HTML; the Markdown remains the authoritative text.
    escaped = html.escape(markdown_text)
    escaped = escaped.replace("\n", "<br>\n")
    images = "\n".join(
        f'<figure><img src="{html.escape(name)}" alt="{html.escape(name)}">'
        f"<figcaption>{html.escape(name)}</figcaption></figure>"
        for name in plot_files
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>G1 semantic pipeline no-GT audit</title>
<style>
body {{ max-width: 1180px; margin: 2rem auto; padding: 0 1.2rem;
       font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       color: #17212b; line-height: 1.55; }}
.notice {{ background: #fff3cd; border-left: 5px solid #e9c46a; padding: 1rem; }}
pre {{ white-space: pre-wrap; background: #f6f8fa; padding: 1rem;
       border-radius: 8px; font-family: inherit; }}
figure {{ margin: 2.2rem 0; }}
img {{ width: 100%; height: auto; border: 1px solid #d8dee4; border-radius: 8px; }}
figcaption {{ color: #57606a; text-align: center; }}
</style>
</head>
<body>
<h1>G1 语义地图全流程无 GT 只读审计</h1>
<div class="notice">{html.escape(NO_GT_REASON)}</div>
<h2>完整 Markdown 报告</h2>
<pre>{escaped}</pre>
<h2>可视化</h2>
{images}
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    cwd = Path.cwd().resolve()
    raw_dir = require_dir(resolve(args.raw_dir))
    prepared_dir = require_dir(resolve(args.prepared_dir))
    geometry_dir = require_dir(resolve(args.geometry_dir))
    semantic_dir = require_dir(resolve(args.semantic_dir))
    best_combination = require_file(resolve(args.best_combination))
    output_dir = resolve(args.output_dir)
    ensure_output_dir(
        output_dir,
        [raw_dir, prepared_dir, geometry_dir, semantic_dir],
        args.allow_existing_output,
    )

    paths = {
        "raw_manifest": raw_dir / "manifest.json",
        "raw_manifest_jsonl": raw_dir / "manifest.jsonl",
        "raw_quality": raw_dir / "quality_report.json",
        "prepared_report": prepared_dir / "rectification_materialization_report.json",
        "prepared_tick": prepared_dir / "tick_index.json",
        "prepared_integrity": prepared_dir / "image_integrity.json",
        "selected_report": geometry_dir / "02_selected" / "keyframe_selection_report.json",
        "selected_tick": geometry_dir / "02_selected" / "tick_index.json",
        "depth_report": geometry_dir / "03_geometry" / "fast_foundation_stereo_run.json",
        "temporal_before": geometry_dir
        / "04_temporal_input"
        / "temporal_depth_consistency_report.json",
        "trajectory": geometry_dir
        / "05_rgbd_window_graph"
        / "trajectory_refinement.json",
        "loops": geometry_dir / "06_loop_closures" / "loop_closure_report.json",
        "global_pose": geometry_dir
        / "07_global_pose_graph"
        / "global_pose_graph_report.json",
        "filter": geometry_dir
        / "08_temporal_depth_filtered"
        / "temporal_depth_filter_report.json",
        "final_tick": geometry_dir
        / "08_temporal_depth_filtered"
        / "tick_index.json",
        "temporal_after": geometry_dir
        / "09_temporal_validation"
        / "temporal_depth_consistency_report.json",
        "mapping_run": geometry_dir / "mapping_run.json",
        "semantic_run": semantic_dir / "realtime_run_report.json",
        "semantic_metrics": semantic_dir / "realtime_metrics.json",
        "quality_context": semantic_dir / "quality_context.json",
        "quality_report": semantic_dir / "quality_report.json",
        "semantic_postpass": semantic_dir / "hydra_semantic_postpass.json",
        "semantic_commit": semantic_dir
        / "hydra_realtime"
        / "backend"
        / "semantic_dsg_commit.json",
        "map_memory": semantic_dir / "map_memory.sqlite3",
        "hydra_dsg": semantic_dir / "hydra_realtime" / "backend" / "dsg.json",
        "hydra_dsg_mesh": semantic_dir
        / "hydra_realtime"
        / "backend"
        / "dsg_with_mesh.json",
        "hydra_mesh": semantic_dir / "hydra_realtime" / "backend" / "mesh.ply",
        "best_combination": best_combination,
    }
    for path in paths.values():
        require_file(path)

    curated_paths = list(paths.values())
    calibration_dir = raw_dir / "calibrations" / "000000"
    curated_paths.extend(
        path
        for path in sorted(calibration_dir.glob("*"))
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}
    )
    wal_path = semantic_dir / "map_memory.sqlite3-wal"
    if wal_path.is_file():
        curated_paths.append(wal_path)
    script_path = Path(__file__).resolve()
    curated_paths.append(script_path)
    before_signatures = {str(path): stat_signature(path) for path in curated_paths}

    load_json(paths["raw_quality"])
    prepared_report = load_json(paths["prepared_report"])
    prepared_tick = load_json(paths["prepared_tick"])
    selected_report = load_json(paths["selected_report"])
    final_tick = load_json(paths["final_tick"])
    depth_report = load_json(paths["depth_report"])
    temporal_before_report = load_json(paths["temporal_before"])
    temporal_after_report = load_json(paths["temporal_after"])
    filter_report = load_json(paths["filter"])
    trajectory = load_json(paths["trajectory"])
    loops = load_json(paths["loops"])
    global_pose = load_json(paths["global_pose"])
    realtime_run = load_json(paths["semantic_run"])
    realtime_metrics = load_json(paths["semantic_metrics"])
    quality_report = load_json(paths["quality_report"])
    semantic_postpass = load_json(paths["semantic_postpass"])
    semantic_commit = load_json(paths["semantic_commit"])

    tick_frames = final_tick.get("frames", [])
    expected = int(args.expected_frames)
    contracts = {
        "expected_frames": expected,
        "selected_tick_frames": len(tick_frames),
        "depth_frame_stats": len(depth_report.get("frame_stats", [])),
        "filter_per_frame": len(filter_report.get("per_frame", [])),
        "semantic_frames_completed": realtime_run.get("frames_completed"),
        "postpass_frames_replayed": semantic_postpass.get("frames_replayed"),
    }
    mismatches = {
        key: value
        for key, value in contracts.items()
        if key != "expected_frames" and int(value) != expected
    }
    if mismatches:
        raise ValueError(f"844-frame contract mismatch: {mismatches}")
    frame_indices = [int(frame["idx"]) for frame in tick_frames]
    if frame_indices != list(range(expected)):
        raise ValueError("final tick index is not contiguous 0..expected-1")

    raw_rows, raw_summary = raw_manifest_scan(
        raw_dir, scan_lidar=not args.skip_lidar_scan
    )
    raw_by_tick = {int(row["tick"]): row for row in raw_rows}
    semantic_rows, semantic_summary, label_histogram = semantic_frame_scan(
        semantic_dir, expected
    )
    map_memory = sqlite_map_memory_summary(paths["map_memory"])
    dsg_summary = hydra_dsg_summary(paths["hydra_dsg"])
    ply_header = parse_ply_header(paths["hydra_mesh"])

    frame_rows = make_per_frame_rows(
        tick_frames,
        depth_report["frame_stats"],
        temporal_pair_map(temporal_before_report),
        temporal_pair_map(temporal_after_report),
        filter_report["per_frame"],
        semantic_rows,
        raw_by_tick,
    )
    per_frame_fields = list(frame_rows[0].keys())
    write_csv(output_dir / "per_frame_metrics.csv", frame_rows, per_frame_fields)
    write_csv(
        output_dir / "raw_frame_metrics.csv",
        raw_rows,
        list(raw_rows[0].keys()),
    )
    write_csv(
        output_dir / "semantic_label_histogram.csv",
        [
            {
                "semantic_id": label,
                "pixels": pixels,
                "ratio_of_all_pixels": pixels / sum(label_histogram.values()),
                "ratio_of_labeled_pixels": (
                    pixels
                    / (
                        sum(label_histogram.values())
                        - label_histogram.get(0, 0)
                    )
                    if label != 0
                    else None
                ),
            }
            for label, pixels in sorted(label_histogram.items())
        ],
        [
            "semantic_id",
            "pixels",
            "ratio_of_all_pixels",
            "ratio_of_labeled_pixels",
        ],
    )

    runtime_rows = []
    for stage, metrics in sorted(realtime_metrics.get("stages", {}).items()):
        latency = metrics.get("latency", {})
        runtime_rows.append(
            {
                "stage": stage,
                "processed": metrics.get("processed"),
                "errors": metrics.get("errors"),
                "throughput_hz": metrics.get("throughput_hz"),
                "queue_high_water": metrics.get("queue_high_water"),
                "queue_wait_p50_ms": latency.get("queue_wait_ms", {}).get("p50"),
                "queue_wait_p95_ms": latency.get("queue_wait_ms", {}).get("p95"),
                "service_p50_ms": latency.get("service_ms", {}).get("p50"),
                "service_p95_ms": latency.get("service_ms", {}).get("p95"),
                "service_p99_ms": latency.get("service_ms", {}).get("p99"),
                "end_to_end_p50_ms": latency.get("end_to_end_ms", {}).get("p50"),
                "end_to_end_p95_ms": latency.get("end_to_end_ms", {}).get("p95"),
            }
        )
    write_csv(
        output_dir / "runtime_stage_metrics.csv",
        runtime_rows,
        list(runtime_rows[0].keys()),
    )
    write_json(output_dir / "map_memory_summary.json", map_memory)

    selected_gaps = []
    for previous, current in zip(frame_rows, frame_rows[1:]):
        selected_gaps.append(
            {
                "previous_frame_idx": previous["frame_idx"],
                "frame_idx": current["frame_idx"],
                "previous_source_idx": previous["source_idx"],
                "source_idx": current["source_idx"],
                "gap_s": current["selected_gap_from_previous_s"],
                "source_gap": current["source_gap_from_previous"],
                "selection_reason": current["selection_reason"],
            }
        )
    selected_gaps.sort(key=lambda row: float(row["gap_s"]), reverse=True)

    local_candidate_improved = (
        trajectory.get("visual_residual_median_after_m") is not None
        and trajectory.get("visual_residual_median_before_m") is not None
        and trajectory["visual_residual_median_after_m"]
        < trajectory["visual_residual_median_before_m"]
    )
    e6 = proxy_assessment(
        "PASS_WITH_FALLBACK",
        "未收敛的局部 RGB-D optimizer 没有写回；global source 模式保持原始轨迹。",
        [
            f"optimizer_success={trajectory.get('optimizer_success')}",
            f"candidate_residual_improved={local_candidate_improved}",
            f"global_mode={global_pose.get('optimization', {}).get('mode')}",
            "position_change_from_source_m.max="
            f"{global_pose.get('position_change_from_source_m', {}).get('max')}",
        ],
        [
            "没有独立高精度轨迹 GT。",
            "候选残差下降不能覆盖 optimizer 未收敛。",
            "结论只验证安全 fallback，不验证 source pose 正确。",
        ],
    )
    e7 = proxy_assessment(
        "PASS_DIAGNOSTIC",
        f"{loops.get('retrieved_count')} 个 retrieval 候选中，"
        f"{loops.get('dense_tested_count')} 个进入 dense verification，"
        f"{loops.get('verified_count')} 个被接受。",
        [
            f"retrieved_count={loops.get('retrieved_count')}",
            f"dense_tested_count={loops.get('dense_tested_count')}",
            f"dense_verified_count={loops.get('dense_verified_count')}",
            f"verified_count={loops.get('verified_count')}",
        ],
        [
            "没有 true-loop/false-loop GT。",
            "0 accepted 与“此窗口无真实闭环”一致，但不能量化 false rejection。",
        ],
    )
    e8 = proxy_assessment(
        "PASS_NO_GLOBAL_GAIN_CLAIMED",
        "0 个 verified loop 被选入，global pose graph 使用 source 模式且轨迹变化为 0。",
        [
            f"selected_verified_loops={len(global_pose.get('selected_verified_loops', []))}",
            f"optimization.mode={global_pose.get('optimization', {}).get('mode')}",
            "position_change_from_source_m.max="
            f"{global_pose.get('position_change_from_source_m', {}).get('max')}",
        ],
        [
            "没有闭环 GT 或全局轨迹 GT。",
            "该结果证明没有虚报/写回 global gain，不证明 source trajectory 准确。",
        ],
    )

    map_metrics = realtime_run.get("map_metrics", {})
    runtime_quality = next(
        (
            result
            for result in quality_report.get("results", [])
            if result.get("stage") == "runtime"
        ),
        {},
    )
    prepared_source_indices = [
        int(frame["source_idx"]) for frame in prepared_tick.get("frames", [])
    ]
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "no_gt_policy": {
            "ground_truth_available": False,
            "manual_annotation_skipped": True,
            "held_out_skipped": True,
            "reason": NO_GT_REASON,
            "allowed_evidence_levels": ["exact", "proxy", "unavailable"],
        },
        "contracts": contracts,
        "raw": raw_summary,
        "prepared": {
            "source_frames": prepared_report.get("source_frames"),
            "matched_frames_before_pose_coverage": prepared_report.get(
                "matched_frames_before_pose_coverage"
            ),
            "materialized_frames": prepared_report.get("materialized_frames"),
            "skipped_cam0_sync": prepared_report.get("skipped_cam0_sync"),
            "skipped_cam1_sync": prepared_report.get("skipped_cam1_sync"),
            "maximum_matched_stereo_delta_ms": prepared_report.get(
                "maximum_matched_stereo_delta_ms"
            ),
            "joint_valid_area_ratio": prepared_report.get("valid_area", {}).get(
                "joint_valid_area_ratio"
            ),
            "source_indices": prepared_source_indices,
            "best_combination": {
                "path": str(best_combination),
                "sha256": sha256_file(best_combination),
                "embedded_report_sha256": prepared_report.get(
                    "calibration", {}
                ).get("rectification_report_sha256"),
                "hash_matches": sha256_file(best_combination)
                == prepared_report.get("calibration", {}).get(
                    "rectification_report_sha256"
                ),
            },
            "measurement_evidence": "exact",
            "rectification_accuracy": accuracy_unavailable(
                "校正后的真实极线/重投影准确率"
            ),
        },
        "selection": {
            "source_frames": selected_report.get("source_frame_count"),
            "selected_frames": selected_report.get("selected_frame_count"),
            "reduction_ratio": selected_report.get("reduction_ratio"),
            "selection_reasons": selected_report.get("selection_reasons"),
            "largest_selected_time_gaps": selected_gaps[:30],
            "gap_seconds": percentile_summary(
                row["selected_gap_from_previous_s"] for row in frame_rows
            ),
            "source_index_gap": percentile_summary(
                row["source_gap_from_previous"] for row in frame_rows
            ),
            "measurement_evidence": "exact",
            "coverage_accuracy": accuracy_unavailable(
                "关键帧选择对真实场景覆盖和事件召回的准确率"
            ),
        },
        "depth": {
            "processed_frames": depth_report.get("processed"),
            "failed_frames": depth_report.get("failed"),
            "skipped_frames": depth_report.get("skipped"),
            "elapsed_seconds": depth_report.get("elapsed_seconds"),
            "valid_ratio": percentile_summary(
                row["depth_valid_ratio"] for row in frame_rows
            ),
            "median_depth_m": percentile_summary(
                row["depth_median_m"] for row in frame_rows
            ),
            "left_right_consistency": percentile_summary(
                row["depth_left_right_consistency"] for row in frame_rows
            ),
            "mean_confidence": percentile_summary(
                row["depth_mean_confidence"] for row in frame_rows
            ),
            "occlusion_ratio": percentile_summary(
                row["depth_occlusion_ratio"] for row in frame_rows
            ),
            "end_to_end_seconds": percentile_summary(
                row["depth_end_to_end_s"] for row in frame_rows
            ),
            "measurement_evidence": "exact",
            "diagnostic_interpretation": "proxy",
            "depth_accuracy": accuracy_unavailable(
                "深度绝对/相对误差及边界正确率"
            ),
        },
        "temporal": {
            "before_weighted_agreement": temporal_before_report.get(
                "overall_agreement_rate_weighted"
            ),
            "after_weighted_agreement": temporal_after_report.get(
                "overall_agreement_rate_weighted"
            ),
            "agreement_delta": (
                temporal_after_report.get("overall_agreement_rate_weighted")
                - temporal_before_report.get("overall_agreement_rate_weighted")
            ),
            "before_comparable_samples": temporal_before_report.get(
                "overall_comparable_samples"
            ),
            "after_comparable_samples": temporal_after_report.get(
                "overall_comparable_samples"
            ),
            "before_adjacent_agreement": percentile_summary(
                row["temporal_before_agreement"] for row in frame_rows
            ),
            "after_adjacent_agreement": percentile_summary(
                row["temporal_after_agreement"] for row in frame_rows
            ),
            "before_median_abs_error_m": percentile_summary(
                row["temporal_before_median_abs_error_m"] for row in frame_rows
            ),
            "after_median_abs_error_m": percentile_summary(
                row["temporal_after_median_abs_error_m"] for row in frame_rows
            ),
            "post_filter_gate": temporal_after_report.get("pre_hydra_gate"),
            "measurement_evidence": "exact",
            "interpretation_evidence": "proxy",
            "accuracy": accuracy_unavailable(
                "时序重投影相对真实场景的一致性准确率"
            ),
        },
        "filter": {
            "frame_count": filter_report.get("frame_count"),
            "rejected_pixels": filter_report.get("rejected_pixels"),
            "input_valid_ratio": percentile_summary(
                row["filter_input_valid_ratio"] for row in frame_rows
            ),
            "output_valid_ratio": percentile_summary(
                row["filter_output_valid_ratio"] for row in frame_rows
            ),
            "rejected_valid_ratio": percentile_summary(
                row["filter_rejected_valid_ratio"] for row in frame_rows
            ),
            "settings": {
                key: filter_report.get(key)
                for key in (
                    "neighbor_offsets",
                    "filter_scale",
                    "min_judged_neighbors",
                    "min_support_ratio",
                    "require_temporal_support",
                    "insufficient_evidence_policy",
                    "absolute_tolerance_m",
                    "relative_tolerance",
                )
            },
            "measurement_evidence": "exact",
            "filter_accuracy": accuracy_unavailable(
                "时序过滤像素的 true-positive/false-positive 删除率"
            ),
        },
        "semantics": semantic_summary,
        "semantic_postpass": semantic_postpass,
        "map_memory": map_memory,
        "runtime": {
            "quality_passed": realtime_run.get("quality_passed"),
            "hard_quality_failures": realtime_run.get("hard_quality_failures"),
            "frames_requested": realtime_run.get("frames_requested"),
            "frames_resumed_from": realtime_run.get("frames_resumed_from"),
            "persisted_runtime_frame_samples": realtime_run.get(
                "frames_dispatched"
            ),
            "frames_completed": realtime_run.get("frames_completed"),
            "dropped_frames": realtime_run.get("dropped_frames"),
            "metrics_elapsed_seconds": realtime_metrics.get("elapsed_seconds"),
            "stage_metrics": realtime_metrics.get("stages"),
            "global_service_p95_ms": runtime_quality.get("metrics", {})
            .get("stage_p95_ms", {})
            .get("global"),
            "runtime_quality_result": runtime_quality,
            "measurement_evidence": "exact",
            "coverage_limitation": (
                "realtime_metrics only preserves the resumed 244-frame dispatch; "
                "it is not a full 844-frame per-frame runtime trace"
            ),
        },
        "hydra": {
            "mesh": {
                "path": str(paths["hydra_mesh"]),
                "ply_header": ply_header,
                "vertices": map_metrics.get("vertices"),
                "faces": map_metrics.get("faces"),
                "surface_area_m2": map_metrics.get("surface_area_m2"),
                "connected_components": map_metrics.get("connected_components"),
                "significant_connected_components": map_metrics.get(
                    "significant_connected_components"
                ),
                "largest_component_area_ratio": map_metrics.get(
                    "largest_component_area_ratio"
                ),
                "tiny_component_area_ratio": map_metrics.get(
                    "tiny_component_area_ratio"
                ),
                "bounds": map_metrics.get("bounds"),
                "measurement_evidence": "exact",
                "geometry_accuracy": accuracy_unavailable(
                    "mesh 表面到真实场景的距离、完整度及拓扑准确率"
                ),
            },
            "dsg": dsg_summary,
            "postpass": semantic_postpass,
            "commit": semantic_commit,
            "measurement_evidence": "exact",
        },
        "quality": quality_report,
        "experiments": {"E6": e6, "E7": e7, "E8": e8},
        "pipeline_funnel": {
            "raw_records": len(raw_rows),
            "prepared_frames": prepared_report.get("materialized_frames"),
            "selected_frames": len(tick_frames),
            "depth_frames": depth_report.get("processed"),
            "filtered_frames": filter_report.get("frame_count"),
            "postpass_frames": semantic_postpass.get("frames_replayed"),
            "nonzero_label_frames": semantic_postpass.get(
                "nonzero_label_frames"
            ),
            "map_memory_active_entities": map_memory.get("active_entities"),
            "hydra_object_nodes": dsg_summary.get("object_nodes"),
            "mesh_bound_verified_entities": semantic_commit.get(
                "verified_entity_count"
            ),
            "durable_verified_operations": semantic_commit.get(
                "verified_operation_count"
            ),
            "warning": (
                "frame, entity, Hydra object and operation counts are different "
                "identity domains and must not be interpreted as one recall funnel"
            ),
        },
    }

    failures = build_failure_cases(
        raw_rows, frame_rows, trajectory, quality_report, map_metrics
    )
    with (output_dir / "failure_cases.jsonl").open("w", encoding="utf-8") as stream:
        for failure in failures:
            stream.write(
                json.dumps(sanitize(failure), ensure_ascii=False, allow_nan=False)
                + "\n"
            )
    summary["failure_cases"] = {
        "count": len(failures),
        "by_severity": dict(Counter(case["severity"] for case in failures)),
        "by_stage": dict(Counter(case["stage"] for case in failures)),
        "artifact": str(output_dir / "failure_cases.jsonl"),
    }
    write_json(output_dir / "stage_summary.json", summary)

    setup_plot()
    plot_files = [
        "01_depth_quality_timeseries.png",
        "02_temporal_before_after.png",
        "03_temporal_filter_effect.png",
        "04_masks_semantics.png",
        "05_runtime_quality.png",
        "06_pipeline_funnel_quality.png",
        "07_failure_localization_heatmap.png",
        "08_raw_lidar_zero_ratio.png",
    ]
    plot_depth(frame_rows, output_dir / plot_files[0])
    plot_temporal(frame_rows, output_dir / plot_files[1])
    plot_filter(frame_rows, output_dir / plot_files[2])
    plot_masks(frame_rows, output_dir / plot_files[3])
    plot_runtime(quality_report, realtime_metrics, output_dir / plot_files[4])
    plot_funnel(summary, quality_report, output_dir / plot_files[5])
    plot_failure_heatmap(frame_rows, output_dir / plot_files[6])
    plot_lidar(raw_rows, output_dir / plot_files[7])

    after_signatures = {str(path): stat_signature(path) for path in curated_paths}
    curated_hashes = [
        {
            "path": str(path),
            "size_bytes": before_signatures[str(path)]["size_bytes"],
            "sha256": sha256_file(path),
            "signature_before": before_signatures[str(path)],
            "signature_after": after_signatures[str(path)],
            "unchanged_during_audit": before_signatures[str(path)]
            == after_signatures[str(path)],
        }
        for path in curated_paths
    ]
    command = " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
    provenance = {
        "schema": f"{SCHEMA}.provenance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(cwd),
        "command": command,
        "argv": [sys.executable, *sys.argv],
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "matplotlib": matplotlib.__version__,
            "sqlite": sqlite3.sqlite_version,
        },
        "inputs": {
            "raw_dir": str(raw_dir),
            "prepared_dir": str(prepared_dir),
            "geometry_dir": str(geometry_dir),
            "semantic_dir": str(semantic_dir),
            "best_combination": str(best_combination),
        },
        "output_dir": str(output_dir),
        "curated_input_hashes": curated_hashes,
        "hash_scope": (
            "Curated contracts/reports/calibrations/SQLite/DSG/mesh only. "
            "This is deliberately not a complete input-file hash inventory."
        ),
        "all_curated_inputs_unchanged_during_audit": all(
            item["unchanged_during_audit"] for item in curated_hashes
        ),
        "read_only_guarantees": {
            "upstream_output_target_overlap_rejected": True,
            "map_memory_open_mode": "mode=ro&immutable=1 + query_only",
            "upstream_pipeline_commands_executed": False,
            "upstream_artifacts_overwritten": False,
        },
        "no_gt_policy": summary["no_gt_policy"],
    }
    write_json(output_dir / "provenance.json", provenance)
    with (output_dir / "command.txt").open("w", encoding="utf-8") as stream:
        stream.write(command + "\n")

    report = render_report(output_dir, summary, failures, provenance, plot_files)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "report.html").write_text(
        markdown_to_html(report, plot_files), encoding="utf-8"
    )

    # Hash the completed evidence bundle (excluding this self-referential index).
    evidence_files = sorted(
        path for path in output_dir.iterdir() if path.name != "evidence_manifest.json"
    )
    evidence_manifest = {
        "schema": f"{SCHEMA}.evidence_manifest",
        "generated_at_utc": provenance["generated_at_utc"],
        "files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in evidence_files
            if path.is_file()
        ],
    }
    write_json(output_dir / "evidence_manifest.json", evidence_manifest)

    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "per_frame_rows": len(frame_rows),
                "raw_rows": len(raw_rows),
                "failure_cases": len(failures),
                "png_count": len(plot_files),
                "quality_passed": quality_report.get("passed"),
                "no_gt_accuracy": "unavailable",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
