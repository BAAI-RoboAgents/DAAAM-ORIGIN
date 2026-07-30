#!/usr/bin/env python3
"""Replay E13 MapMemory entity merging on frozen E12/E11 geometry evidence.

This is deliberately a GT-free diagnostic.  It exercises the production
MapMemory.observe_entity implementation and retains every intermediate needed
to audit geometry, merge decisions, entity versions, and ID visualizations.
It does not report entity precision/recall/F1 or correctness of over-merges.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
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

from daaam.memory.store import MapMemory, MapMemoryConfig  # noqa: E402
from daaam.realtime.masked_geometry import backproject_masked_depth  # noqa: E402


EXPERIMENT_ROOT = (
    REPOSITORY_ROOT / "experiments/g1_20260724_473_573_v1_1"
)
DEFAULT_E12 = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e12_e11fed_botsort_20260729"
)
DEFAULT_DEPTH = (
    EXPERIMENT_ROOT
    / "shared_artifacts/e13_metric_depth_473_573_20260729"
)
DEFAULT_PREPARED = (
    EXPERIMENT_ROOT
    / "shared_artifacts/prepared_stereo_473_573_v1_v2"
)
DEFAULT_SCALE = (
    EXPERIMENT_ROOT
    / "shared_artifacts/scale_473_487_proposal_v1_1.json"
)
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e13_e12fed_mapmemory_20260729"
)
PROTOCOL = REPOSITORY_ROOT / "docs/g1_semantic_map_experiments_v1_1.md"
DIAGNOSTIC_PROTOCOL = (
    REPOSITORY_ROOT / "docs/g1_semantic_map_diagnostic_no_gt_stage.md"
)
MAP_MEMORY_SOURCE = REPOSITORY_ROOT / "src/daaam/memory/store.py"
GEOMETRY_SOURCE = REPOSITORY_ROOT / "src/daaam/realtime/masked_geometry.py"
THRESHOLDS_M = (0.20, 0.35, 0.50)
MINIMUM_VALID_DEPTH_RATIO = 0.25
MAXIMUM_POINTS = 20_000
SESSION_ID = "g1-e13-source-473-573"
INVENTORY_EXCLUDES = {
    "artifact_inventory.csv",
    "artifact_inventory.jsonl",
    "inventory_summary.json",
    "COMPLETION.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GT-free E13 MapMemory diagnostics on E12 baseline tracks."
    )
    parser.add_argument("--e12-run", type=Path, default=DEFAULT_E12)
    parser.add_argument("--depth-run", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--scale-proposal", type=Path, default=DEFAULT_SCALE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-frames", type=int)
    parser.add_argument(
        "--reseal-existing",
        action="store_true",
        help="Only rebuild the artifact inventory and completion seal.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write {path}")


def variant_id(threshold_m: float, association_policy: str = "legacy") -> str:
    base = f"merge_{threshold_m:.2f}m".replace(".", "p")
    return base if association_policy == "legacy" else f"{association_policy}_{base}"


def stable_color(ordinal: int) -> tuple[int, int, int]:
    hue = (ordinal * 47 + 11) % 180
    hsv = np.uint8([[[hue, 220, 245]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def inventory_rows(root: Path, *, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
                "absolute_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def inventory_root(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                f"{row['relative_path']}\t{row['size_bytes']}\t{row['sha256']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def seal_output(root: Path, *, status: str = "complete_pending_independent_audit") -> None:
    rows = inventory_rows(root, excluded=INVENTORY_EXCLUDES)
    root_hash = inventory_root(rows)
    write_jsonl(root / "artifact_inventory.jsonl", rows)
    write_csv(root / "artifact_inventory.csv", rows)
    summary = {
        "schema": "daaam.artifact_inventory.v1",
        "generated_at": utc_now(),
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "inventory_root_sha256": root_hash,
        "excluded_self_referential_files": sorted(INVENTORY_EXCLUDES),
    }
    write_json(root / "inventory_summary.json", summary)
    completion = {
        "schema": "daaam.g1_no_gt_e13_completion.v1",
        "status": status,
        "generated_at": utc_now(),
        "artifact_inventory_root_sha256": root_hash,
        "artifact_inventory_file_count": len(rows),
        "formal_claims_permitted": False,
        "independent_audit": (
            "passed"
            if status == "complete_independently_audited"
            else "pending"
        ),
    }
    write_json(root / "COMPLETION.json", completion)


def hash_reference(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_preregistration(
    output: Path,
    e12: Path,
    depth: Path,
    prepared: Path,
    scale_path: Path,
    frame_count: int,
) -> None:
    scale = json.loads(scale_path.read_text(encoding="utf-8"))
    e12_completion = json.loads((e12 / "COMPLETION.json").read_text(encoding="utf-8"))
    preregistration = {
        "schema": "daaam.g1_no_gt_e13_preregistration.v1",
        "registered_at": utc_now(),
        "stage": "E13 entity merge",
        "status": "diagnostic_gt_free_upstream_coupled",
        "hypothesis": (
            "MapMemory center-distance merging reduces E12 track fragments without "
            "obvious proxy conflict signatures."
        ),
        "formal_protocol_candidates_m": list(THRESHOLDS_M),
        "production_implementation": "MapMemory.observe_entity",
        "input_contract": {
            "tracks": "frozen E12 baseline, itself E11-fed rather than GT",
            "geometry": "E11 mask + scaled stereo depth + world_T_camera",
            "semantic_label": "unknown for every observation",
            "entity_type": "object",
            "valid_depth_ratio_minimum": MINIMUM_VALID_DEPTH_RATIO,
            "maximum_backprojected_points": MAXIMUM_POINTS,
            "position_estimator": "component-wise median world point",
            "dimension_estimator": "world-point q05-q95, minimum 0.05 m",
            "depth_scale": scale["depth_scale"],
            "depth_scale_status": scale["status"],
            "frame_count": frame_count,
        },
        "frozen_inputs": {
            "e12_inventory_root_sha256": e12_completion["inventory"][
                "manifest_root_sha256"
            ],
            "e12_completion": hash_reference(e12 / "COMPLETION.json"),
            "e12_baseline_observations": hash_reference(
                e12 / "variants/baseline/track_observations.jsonl"
            ),
            "depth_run": hash_reference(depth / "fast_foundation_stereo_run.json"),
            "depth_camera": hash_reference(depth / "camera_info.json"),
            "depth_tick_index": hash_reference(depth / "tick_index.json"),
            "prepared_tick_index": hash_reference(prepared / "tick_index.json"),
            "prepared_poses": hash_reference(prepared / "pose/poses.txt"),
            "scale_proposal": hash_reference(scale_path),
            "map_memory_source": hash_reference(MAP_MEMORY_SOURCE),
            "geometry_source": hash_reference(GEOMETRY_SOURCE),
            "formal_protocol": hash_reference(PROTOCOL),
            "diagnostic_protocol": hash_reference(DIAGNOSTIC_PROTOCOL),
        },
        "reported_diagnostic_proxies": [
            "entity count",
            "new-track merge count",
            "local-track reassociation count",
            "multi-track entity count",
            "same-frame multi-track entity collisions",
            "track-to-multiple-entity count",
            "within-entity spatial spread",
            "entity dimensions",
        ],
        "forbidden_formal_claims": [
            "entity precision/recall/F1",
            "true over-merge rate",
            "true over-split rate",
            "best or winning threshold",
        ],
        "known_limitations": [
            "No reviewed GT track fragments or GT 3D centers are available.",
            "E12 baseline tracks are estimated, not oracle tracks.",
            "The metric scale is proposal_not_frozen and validated only on frames 473-487.",
            "All pre-DAM labels are unknown, so semantic-name gating cannot separate objects.",
            "Proxy conflicts are review queues, not correctness labels.",
        ],
    }
    write_json(output / "PRE_REGISTRATION.json", preregistration)


def save_depth_inventory(depth: Path, output: Path) -> dict[str, Any]:
    rows = inventory_rows(depth)
    write_jsonl(output / "input_manifests/depth_inventory.jsonl", rows)
    write_csv(output / "input_manifests/depth_inventory.csv", rows)
    summary = {
        "schema": "daaam.g1_no_gt_e13_depth_inventory.v1",
        "depth_root": str(depth.resolve()),
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "inventory_root_sha256": inventory_root(rows),
    }
    write_json(output / "input_manifests/depth_inventory_summary.json", summary)
    return summary


def load_inputs(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray, float]:
    frames = read_jsonl(args.e12_run / "input_frames.jsonl")
    observations = read_jsonl(
        args.e12_run / "variants/baseline/track_observations.jsonl"
    )
    if args.maximum_frames is not None:
        frames = frames[: args.maximum_frames]
        accepted = {int(row["frame_index"]) for row in frames}
        observations = [
            row for row in observations if int(row["frame_index"]) in accepted
        ]
    camera = json.loads((args.depth_run / "camera_info.json").read_text())
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    poses_flat = np.loadtxt(args.prepared / "pose/poses.txt", dtype=np.float64)
    poses = poses_flat.reshape((-1, 4, 4))
    scale = json.loads(args.scale_proposal.read_text())["depth_scale"]
    if len(poses) < len(frames):
        raise ValueError("pose rows do not cover all selected frames")
    return frames, observations, intrinsics, poses, float(scale)


def build_geometry(
    output: Path,
    depth_root: Path,
    frames: Sequence[dict[str, Any]],
    observations: Sequence[dict[str, Any]],
    intrinsics: np.ndarray,
    poses: np.ndarray,
    scale: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    observations_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        observations_by_frame[int(row["frame_index"])].append(row)
    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for frame in frames:
        frame_index = int(frame["frame_index"])
        stem = f"{frame_index:08d}"
        source_depth = depth_root / "depth" / f"{stem}.png"
        depth_mm = cv2.imread(str(source_depth), cv2.IMREAD_UNCHANGED)
        if depth_mm is None or depth_mm.ndim != 2:
            raise ValueError(f"invalid depth image {source_depth}")
        raw_depth_m = depth_mm.astype(np.float32) / 1000.0
        scaled_depth_m = raw_depth_m * np.float32(scale)
        scaled_path = output / "geometry_input/scaled_depth_meter" / f"{stem}.npy"
        scaled_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(scaled_path, scaled_depth_m, allow_pickle=False)
        rgb = cv2.imread(str(frame["rgb_path"]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"invalid RGB image {frame['rgb_path']}")
        overlay = rgb.copy()
        frame_valid = 0
        frame_rejected = 0
        for observation in sorted(
            observations_by_frame[frame_index],
            key=lambda item: (int(item["track_id"]), int(item["e11_instance_id"])),
        ):
            mask_path = Path(observation["source_mask_path"])
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_image is None or mask_image.shape != depth_mm.shape:
                raise ValueError(f"invalid mask {mask_path}")
            mask = mask_image > 0
            mask_area = int(np.count_nonzero(mask))
            valid_mask = mask & np.isfinite(scaled_depth_m) & (scaled_depth_m > 0)
            valid_depth_count = int(np.count_nonzero(valid_mask))
            valid_ratio = (
                valid_depth_count / mask_area if mask_area else 0.0
            )
            raw_values = raw_depth_m[valid_mask]
            base = {
                "schema": "daaam.g1_no_gt_e13_geometry_observation.v1",
                "frame_index": frame_index,
                "source_frame_index": int(frame["source_frame_index"]),
                "sensor_time_ns": int(frame["sensor_time_ns"]),
                "track_id": int(observation["track_id"]),
                "local_entity_id": f"botsort:{int(observation['track_id'])}",
                "e11_instance_id": int(observation["e11_instance_id"]),
                "model_confidence": float(observation["model_confidence"]),
                "e11_box_xyxy": observation["e11_box_xyxy"],
                "mask_area_px": mask_area,
                "valid_depth_pixel_count": valid_depth_count,
                "valid_depth_ratio": valid_ratio,
                "minimum_valid_depth_ratio": MINIMUM_VALID_DEPTH_RATIO,
                "source_mask_path": str(mask_path.resolve()),
                "source_mask_sha256": observation["source_mask_sha256"],
                "source_depth_path": str(source_depth.resolve()),
                "source_depth_sha256": sha256_file(source_depth),
                "scaled_depth_path": str(scaled_path.resolve()),
                "scaled_depth_scale": scale,
                "raw_depth_median_m": (
                    float(np.median(raw_values)) if len(raw_values) else None
                ),
                "scaled_depth_median_m": (
                    float(np.median(raw_values) * scale) if len(raw_values) else None
                ),
                "intrinsics": intrinsics.tolist(),
                "world_T_camera": poses[frame_index].tolist(),
            }
            box = [int(round(value)) for value in observation["track_box_xyxy"]]
            if mask_area == 0:
                rejected_rows.append(
                    {**base, "accepted": False, "rejection_reason": "empty_mask"}
                )
                frame_rejected += 1
                color = (0, 0, 255)
            elif valid_ratio < MINIMUM_VALID_DEPTH_RATIO:
                rejected_rows.append(
                    {
                        **base,
                        "accepted": False,
                        "rejection_reason": "insufficient_valid_depth_ratio",
                    }
                )
                frame_rejected += 1
                color = (0, 0, 255)
            else:
                geometry = backproject_masked_depth(
                    mask,
                    scaled_depth_m,
                    intrinsics,
                    poses[frame_index],
                    maximum_points=MAXIMUM_POINTS,
                )
                npz_path = (
                    output
                    / "geometry_input/track_geometry"
                    / stem
                    / (
                        f"track_{int(observation['track_id']):04d}_"
                        f"e11_{int(observation['e11_instance_id']):04d}.npz"
                    )
                )
                npz_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    npz_path,
                    points_world_m=geometry.points_world_m,
                    pixel_yx=geometry.pixel_yx,
                    position_m=geometry.position_m,
                    dimensions_m=geometry.dimensions_m,
                    valid_pixel_count=np.asarray(geometry.valid_pixel_count),
                    frame_index=np.asarray(frame_index),
                    track_id=np.asarray(int(observation["track_id"])),
                    e11_instance_id=np.asarray(int(observation["e11_instance_id"])),
                )
                accepted = {
                    **base,
                    "accepted": True,
                    "rejection_reason": None,
                    "sampled_point_count": int(len(geometry.points_world_m)),
                    "position_world_m": geometry.position_m.tolist(),
                    "dimensions_world_m": geometry.dimensions_m.tolist(),
                    "geometry_npz_path": str(npz_path.resolve()),
                    "geometry_npz_sha256": sha256_file(npz_path),
                }
                valid_rows.append(accepted)
                frame_valid += 1
                color = (0, 220, 0)
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, 1)
            cv2.putText(
                overlay,
                f"T{observation['track_id']} d={valid_ratio:.2f}",
                (max(0, box[0]), max(14, box[1] + 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        overlay_path = output / "geometry_input/frame_overlays" / f"{stem}.jpg"
        require_image(overlay_path, overlay)
        frame_rows.append(
            {
                "schema": "daaam.g1_no_gt_e13_geometry_frame.v1",
                "frame_index": frame_index,
                "source_frame_index": int(frame["source_frame_index"]),
                "sensor_time_ns": int(frame["sensor_time_ns"]),
                "input_track_observations": len(observations_by_frame[frame_index]),
                "geometry_accepted": frame_valid,
                "geometry_rejected": frame_rejected,
                "source_depth_path": str(source_depth.resolve()),
                "source_depth_sha256": sha256_file(source_depth),
                "scaled_depth_path": str(scaled_path.resolve()),
                "scaled_depth_sha256": sha256_file(scaled_path),
                "geometry_overlay_path": str(overlay_path.resolve()),
                "geometry_overlay_sha256": sha256_file(overlay_path),
            }
        )
        print(
            f"geometry frame {frame_index + 1}/{len(frames)}: "
            f"accepted={frame_valid} rejected={frame_rejected}",
            flush=True,
        )
    write_jsonl(output / "tables/geometry_observations.jsonl", valid_rows)
    write_csv(output / "tables/geometry_observations.csv", valid_rows)
    write_jsonl(output / "tables/geometry_rejections.jsonl", rejected_rows)
    write_csv(output / "tables/geometry_rejections.csv", rejected_rows)
    write_jsonl(output / "tables/geometry_frames.jsonl", frame_rows)
    write_csv(output / "tables/geometry_frames.csv", frame_rows)
    write_json(
        output / "geometry_input/SUMMARY.json",
        {
            "schema": "daaam.g1_no_gt_e13_geometry_summary.v1",
            "input_observations": len(observations),
            "accepted_observations": len(valid_rows),
            "rejected_observations": len(rejected_rows),
            "accepted_fraction": (
                len(valid_rows) / len(observations) if observations else None
            ),
            "minimum_valid_depth_ratio": MINIMUM_VALID_DEPTH_RATIO,
            "maximum_points": MAXIMUM_POINTS,
            "depth_scale": scale,
            "depth_scale_status": "proposal_not_frozen",
        },
    )
    return valid_rows, rejected_rows, frame_rows


def sqlite_entities(database: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM entities WHERE deleted_ns IS NULL ORDER BY created_ns, entity_id"
            )
        ]


def export_database(database: Path, export_root: Path) -> dict[str, Any]:
    export_root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        table_names = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
        ]
        counts = {}
        for table in table_names:
            rows = [
                dict(row)
                for row in connection.execute(f'SELECT * FROM "{table}"')
            ]
            counts[table] = len(rows)
            write_jsonl(export_root / f"{table}.jsonl", rows)
            write_csv(export_root / f"{table}.csv", rows)
    summary = {
        "schema": "daaam.g1_no_gt_e13_database_export.v1",
        "database_path": str(database.resolve()),
        "database_sha256": sha256_file(database),
        "sqlite_integrity_check": integrity,
        "table_row_counts": counts,
    }
    write_json(export_root / "SUMMARY.json", summary)
    return summary


def candidates_before(
    memory: MapMemory,
    position: np.ndarray,
    threshold_m: float,
) -> list[dict[str, Any]]:
    candidates = []
    for entity in memory.list_entities():
        if entity["entity_type"] != "object" or entity["canonical_name"].casefold() != "unknown":
            continue
        center = np.asarray(entity["position_m"], dtype=np.float64)
        distance = float(np.linalg.norm(position - center))
        candidates.append(
            {
                "entity_id": entity["entity_id"],
                "position_m": center.tolist(),
                "distance_m": distance,
                "within_threshold": distance <= threshold_m,
            }
        )
    return sorted(candidates, key=lambda item: (item["distance_m"], item["entity_id"]))


def event_action(
    prior: str | None, entity_id: str, created: bool
) -> str:
    if prior is None and created:
        return "created_new"
    if prior is None and not created:
        return "new_track_merged"
    if prior == entity_id:
        return "local_track_continued"
    if created:
        return "local_track_reassociated_new"
    return "local_track_reassociated_existing"


def run_variant(
    output: Path,
    threshold_m: float,
    frames: Sequence[dict[str, Any]],
    observations: Sequence[dict[str, Any]],
    *,
    association_policy: str = "legacy",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identifier = variant_id(threshold_m, association_policy)
    root = output / "variants" / identifier
    database = root / "map_memory.sqlite3"
    memory = MapMemory(
        database,
        MapMemoryConfig(
            entity_merge_distance_m=threshold_m,
            entity_association_policy=association_policy,
        ),
    )
    memory.create_session(
        SESSION_ID, int(frames[0]["sensor_time_ns"]), canonical=True
    )
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_frame[int(row["frame_index"])].append(row)
    local_mapping: dict[str, str] = {}
    entity_ordinals: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    for frame in frames:
        frame_index = int(frame["frame_index"])
        rgb = cv2.imread(str(frame["rgb_path"]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"cannot load {frame['rgb_path']}")
        id_map = np.zeros(rgb.shape[:2], dtype=np.uint16)
        confidence_map = np.full(rgb.shape[:2], -1.0, dtype=np.float32)
        frame_events = []
        for geometry in sorted(
            by_frame[frame_index],
            key=lambda item: (int(item["track_id"]), int(item["e11_instance_id"])),
        ):
            position = np.asarray(geometry["position_world_m"], dtype=np.float64)
            dimensions = np.asarray(
                geometry["dimensions_world_m"], dtype=np.float64
            )
            local_id = str(geometry["local_entity_id"])
            prior = local_mapping.get(local_id)
            before = candidates_before(memory, position, threshold_m)
            started = time.perf_counter_ns()
            entity_id, created = memory.observe_entity(
                SESSION_ID,
                local_id,
                position,
                sensor_time_ns=int(geometry["sensor_time_ns"]),
                semantic_label="unknown",
                dimensions_m=dimensions,
                confidence=float(geometry["model_confidence"]),
                entity_type="object",
            )
            latency_ms = (time.perf_counter_ns() - started) / 1.0e6
            latencies_ms.append(latency_ms)
            local_mapping[local_id] = entity_id
            if entity_id not in entity_ordinals:
                entity_ordinals[entity_id] = len(entity_ordinals) + 1
            ordinal = entity_ordinals[entity_id]
            after = memory.get_entity(entity_id)
            nearest = before[0] if before else None
            action = event_action(prior, entity_id, created)
            event = {
                "schema": "daaam.g1_no_gt_e13_merge_event.v1",
                "variant_id": identifier,
                "threshold_m": threshold_m,
                "association_policy": association_policy,
                "event_index": len(events),
                "frame_index": frame_index,
                "source_frame_index": int(geometry["source_frame_index"]),
                "sensor_time_ns": int(geometry["sensor_time_ns"]),
                "track_id": int(geometry["track_id"]),
                "local_entity_id": local_id,
                "e11_instance_id": int(geometry["e11_instance_id"]),
                "entity_id": entity_id,
                "entity_ordinal": ordinal,
                "prior_entity_id": prior,
                "created": bool(created),
                "action": action,
                "observation_position_m": geometry["position_world_m"],
                "observation_dimensions_m": geometry["dimensions_world_m"],
                "model_confidence": float(geometry["model_confidence"]),
                "candidate_count_before": len(before),
                "within_threshold_candidate_count_before": sum(
                    int(item["within_threshold"]) for item in before
                ),
                "nearest_candidate_entity_id_before": (
                    None if nearest is None else nearest["entity_id"]
                ),
                "nearest_candidate_distance_m_before": (
                    None if nearest is None else nearest["distance_m"]
                ),
                "selected_distance_m_before": next(
                    (
                        item["distance_m"]
                        for item in before
                        if item["entity_id"] == entity_id
                    ),
                    None,
                ),
                "entity_position_after_m": after["position_m"],
                "entity_dimensions_after_m": after["dimensions_m"],
                "entity_observation_count_after": after["temporal_history"][
                    "observation_count"
                ],
                "observe_latency_ms": latency_ms,
                "geometry_npz_path": geometry["geometry_npz_path"],
                "geometry_npz_sha256": geometry["geometry_npz_sha256"],
                "source_mask_path": geometry["source_mask_path"],
                "source_mask_sha256": geometry["source_mask_sha256"],
            }
            events.append(event)
            frame_events.append(event)
            mask = cv2.imread(
                geometry["source_mask_path"], cv2.IMREAD_GRAYSCALE
            )
            selected = (mask > 0) & (
                float(geometry["model_confidence"]) > confidence_map
            )
            id_map[selected] = ordinal
            confidence_map[selected] = float(geometry["model_confidence"])
        overlay = rgb.copy()
        for ordinal in sorted(set(int(value) for value in np.unique(id_map)) - {0}):
            mask = id_map == ordinal
            color = stable_color(ordinal)
            overlay[mask] = (
                0.55 * overlay[mask] + 0.45 * np.asarray(color)
            ).astype(np.uint8)
            ys, xs = np.nonzero(mask)
            if len(xs):
                cv2.putText(
                    overlay,
                    f"E{ordinal}",
                    (int(np.median(xs)), int(np.median(ys))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        frame_root = root / "frames" / f"{frame_index:08d}"
        id_path = frame_root / "entity_id_map.png"
        overlay_path = frame_root / "entity_overlay.jpg"
        require_image(id_path, id_map)
        require_image(overlay_path, overlay)
        frame_record = {
            "schema": "daaam.g1_no_gt_e13_frame.v1",
            "variant_id": identifier,
            "threshold_m": threshold_m,
            "frame_index": frame_index,
            "source_frame_index": int(frame["source_frame_index"]),
            "sensor_time_ns": int(frame["sensor_time_ns"]),
            "geometry_observation_count": len(by_frame[frame_index]),
            "entity_ids_visible": sorted(
                {event["entity_id"] for event in frame_events}
            ),
            "entity_ordinals_visible": sorted(
                {int(event["entity_ordinal"]) for event in frame_events}
            ),
            "events": frame_events,
            "entity_id_map_path": str(id_path.resolve()),
            "entity_id_map_sha256": sha256_file(id_path),
            "entity_overlay_path": str(overlay_path.resolve()),
            "entity_overlay_sha256": sha256_file(overlay_path),
        }
        write_json(frame_root / "frame.json", frame_record)
        frame_summaries.append(
            {
                key: value
                for key, value in frame_record.items()
                if key not in {"events", "entity_ids_visible"}
            }
        )
    entities = memory.list_entities()
    memory.close()
    database_summary = export_database(database, root / "database_export")
    write_jsonl(root / "merge_events.jsonl", events)
    write_csv(root / "merge_events.csv", events)
    write_jsonl(root / "frame_summary.jsonl", frame_summaries)
    write_csv(root / "frame_summary.csv", frame_summaries)
    membership = build_membership(events, entities, threshold_m, entity_ordinals)
    write_json(root / "entity_membership.json", membership)
    write_csv(root / "entity_membership.csv", membership)
    track_timelines = build_track_timelines(events)
    write_json(root / "track_entity_timelines.json", track_timelines)
    write_csv(root / "track_entity_timelines.csv", track_timelines)
    write_merge_graph(root, membership, events)
    failure_cases = build_failure_cases(identifier, threshold_m, membership, track_timelines)
    write_jsonl(root / "failure_cases.jsonl", failure_cases)
    write_csv(root / "failure_cases.csv", failure_cases)
    same_frame_collisions = sum(
        int(row["same_frame_multi_track_collision_count"]) for row in membership
    )
    multi_track_entities = sum(int(row["unique_track_count"] > 1) for row in membership)
    split_tracks = sum(int(row["unique_entity_count"] > 1) for row in track_timelines)
    reassignments = sum(
        int("reassociated" in row["action"]) for row in events
    )
    summary = {
        "schema": "daaam.g1_no_gt_e13_variant_summary.v1",
        "variant_id": identifier,
        "threshold_m": threshold_m,
        "association_policy": association_policy,
        "input_geometry_observations": len(observations),
        "unique_e12_track_ids": len({row["track_id"] for row in observations}),
        "entity_count": len(entities),
        "created_entity_count": sum(int(row["created"]) for row in events),
        "new_track_merge_event_count": sum(
            int(row["action"] == "new_track_merged") for row in events
        ),
        "local_track_reassignment_count": reassignments,
        "multi_track_entity_count_proxy": multi_track_entities,
        "same_frame_multi_track_collision_count_proxy": same_frame_collisions,
        "track_to_multiple_entity_count_proxy": split_tracks,
        "maximum_tracks_per_entity": max(
            (int(row["unique_track_count"]) for row in membership), default=0
        ),
        "observations_per_entity_mean": (
            len(events) / len(entities) if entities else None
        ),
        "maximum_entity_observation_spread_m": max(
            (float(row["maximum_observation_distance_to_final_center_m"]) for row in membership),
            default=0.0,
        ),
        "observe_latency_ms_mean": (
            float(np.mean(latencies_ms)) if latencies_ms else None
        ),
        "observe_latency_ms_p50": (
            float(np.percentile(latencies_ms, 50)) if latencies_ms else None
        ),
        "observe_latency_ms_p95": (
            float(np.percentile(latencies_ms, 95)) if latencies_ms else None
        ),
        "sqlite_integrity_check": database_summary["sqlite_integrity_check"],
        "map_memory_database_path": str(database.resolve()),
        "map_memory_database_sha256": database_summary["database_sha256"],
        "formal_entity_precision": None,
        "formal_entity_recall": None,
        "formal_entity_f1": None,
        "formal_over_merge_rate": None,
        "formal_over_split_rate": None,
        "evaluation_basis": "GT-free structural proxies; not correctness metrics",
    }
    write_json(root / "SUMMARY.json", summary)
    print(
        f"{identifier}: observations={len(events)} entities={len(entities)} "
        f"new-track-merges={summary['new_track_merge_event_count']} "
        f"reassignments={reassignments}",
        flush=True,
    )
    return summary, events


def build_membership(
    events: Sequence[dict[str, Any]],
    entities: Sequence[dict[str, Any]],
    threshold_m: float,
    ordinals: Mapping[str, int],
) -> list[dict[str, Any]]:
    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_entity[event["entity_id"]].append(event)
    rows = []
    for entity in entities:
        entity_id = str(entity["entity_id"])
        members = by_entity[entity_id]
        final_center = np.asarray(entity["position_m"], dtype=np.float64)
        positions = np.asarray(
            [row["observation_position_m"] for row in members], dtype=np.float64
        )
        distances = np.linalg.norm(positions - final_center, axis=1)
        per_frame_tracks: dict[int, set[int]] = defaultdict(set)
        for row in members:
            per_frame_tracks[int(row["frame_index"])].add(int(row["track_id"]))
        collision_count = sum(
            len(tracks) - 1 for tracks in per_frame_tracks.values() if len(tracks) > 1
        )
        unique_tracks = sorted({int(row["track_id"]) for row in members})
        rows.append(
            {
                "schema": "daaam.g1_no_gt_e13_entity_membership.v1",
                "entity_id": entity_id,
                "entity_ordinal": ordinals[entity_id],
                "threshold_m": threshold_m,
                "canonical_name": entity["canonical_name"],
                "observation_count": len(members),
                "unique_track_count": len(unique_tracks),
                "track_ids_json": json.dumps(unique_tracks),
                "first_frame_index": min(int(row["frame_index"]) for row in members),
                "last_frame_index": max(int(row["frame_index"]) for row in members),
                "final_position_m_json": json.dumps(entity["position_m"]),
                "final_dimensions_m_json": json.dumps(entity["dimensions_m"]),
                "maximum_dimension_m": max(entity["dimensions_m"]),
                "median_observation_distance_to_final_center_m": float(
                    np.median(distances)
                ),
                "p95_observation_distance_to_final_center_m": float(
                    np.percentile(distances, 95)
                ),
                "maximum_observation_distance_to_final_center_m": float(
                    np.max(distances)
                ),
                "spread_exceeds_threshold_proxy": bool(
                    float(np.max(distances)) > threshold_m
                ),
                "same_frame_multi_track_collision_count": collision_count,
            }
        )
    return sorted(rows, key=lambda row: int(row["entity_ordinal"]))


def build_track_timelines(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_track[int(event["track_id"])].append(event)
    rows = []
    for track_id, members in sorted(by_track.items()):
        members = sorted(members, key=lambda row: row["event_index"])
        ordered_entities = []
        for member in members:
            if not ordered_entities or ordered_entities[-1] != member["entity_ordinal"]:
                ordered_entities.append(member["entity_ordinal"])
        rows.append(
            {
                "schema": "daaam.g1_no_gt_e13_track_timeline.v1",
                "track_id": track_id,
                "observation_count": len(members),
                "first_frame_index": min(int(row["frame_index"]) for row in members),
                "last_frame_index": max(int(row["frame_index"]) for row in members),
                "unique_entity_count": len(set(ordered_entities)),
                "entity_ordinal_sequence_json": json.dumps(ordered_entities),
                "reassignment_count": sum(
                    int("reassociated" in row["action"]) for row in members
                ),
            }
        )
    return rows


def write_merge_graph(
    root: Path,
    membership: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
) -> None:
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for event in events:
        edge_counts[(int(event["track_id"]), int(event["entity_ordinal"]))] += 1
    graph = {
        "schema": "daaam.g1_no_gt_e13_merge_graph.v1",
        "entity_nodes": [
            {
                "entity_ordinal": row["entity_ordinal"],
                "observation_count": row["observation_count"],
                "track_count": row["unique_track_count"],
            }
            for row in membership
        ],
        "track_nodes": sorted({int(event["track_id"]) for event in events}),
        "edges": [
            {
                "track_id": track_id,
                "entity_ordinal": entity_ordinal,
                "observation_count": count,
            }
            for (track_id, entity_ordinal), count in sorted(edge_counts.items())
        ],
    }
    write_json(root / "merge_graph.json", graph)
    lines = ["graph E13 {", "  rankdir=LR;"]
    for row in membership:
        lines.append(
            f'  e{row["entity_ordinal"]} [shape=box,label="E{row["entity_ordinal"]} '
            f'({row["observation_count"]} obs)"];'
        )
    for track_id in graph["track_nodes"]:
        lines.append(f'  t{track_id} [shape=ellipse,label="T{track_id}"];')
    for edge in graph["edges"]:
        lines.append(
            f'  t{edge["track_id"]} -- e{edge["entity_ordinal"]} '
            f'[label="{edge["observation_count"]}"];'
        )
    lines.append("}")
    (root / "merge_graph.dot").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_failure_cases(
    variant: str,
    threshold_m: float,
    membership: Sequence[dict[str, Any]],
    timelines: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures = []
    for row in membership:
        signatures = []
        if int(row["unique_track_count"]) > 1:
            signatures.append("multi_track_entity_review")
        if int(row["same_frame_multi_track_collision_count"]) > 0:
            signatures.append("same_frame_multi_track_collision_review")
        if bool(row["spread_exceeds_threshold_proxy"]):
            signatures.append("within_entity_spread_exceeds_threshold_review")
        if float(row["maximum_dimension_m"]) > 3.0:
            signatures.append("large_geometry_extent_review")
        for signature in signatures:
            failures.append(
                {
                    "schema": "daaam.g1_no_gt_e13_failure_proxy.v1",
                    "variant_id": variant,
                    "threshold_m": threshold_m,
                    "failure_signature": signature,
                    "entity_ordinal": row["entity_ordinal"],
                    "track_id": None,
                    "details_json": json.dumps(row, ensure_ascii=False),
                    "correctness_label": None,
                    "requires_human_review": True,
                }
            )
    for row in timelines:
        if int(row["unique_entity_count"]) > 1:
            failures.append(
                {
                    "schema": "daaam.g1_no_gt_e13_failure_proxy.v1",
                    "variant_id": variant,
                    "threshold_m": threshold_m,
                    "failure_signature": "local_track_reassociated_review",
                    "entity_ordinal": None,
                    "track_id": row["track_id"],
                    "details_json": json.dumps(row, ensure_ascii=False),
                    "correctness_label": None,
                    "requires_human_review": True,
                }
            )
    return failures


def create_visualizations(
    output: Path,
    summaries: Sequence[dict[str, Any]],
    all_events: Mapping[str, Sequence[dict[str, Any]]],
) -> None:
    labels = [f"{row['threshold_m']:.2f} m" for row in summaries]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].bar(x - 0.18, [row["entity_count"] for row in summaries], 0.36, label="entities")
    axes[0].bar(
        x + 0.18,
        [row["new_track_merge_event_count"] for row in summaries],
        0.36,
        label="new-track merges",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Entity count and merge events")
    axes[0].legend()
    axes[1].bar(
        x - 0.24,
        [row["multi_track_entity_count_proxy"] for row in summaries],
        0.24,
        label="multi-track entities",
    )
    axes[1].bar(
        x,
        [row["same_frame_multi_track_collision_count_proxy"] for row in summaries],
        0.24,
        label="same-frame collisions",
    )
    axes[1].bar(
        x + 0.24,
        [row["track_to_multiple_entity_count_proxy"] for row in summaries],
        0.24,
        label="split/reassociated tracks",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_title("GT-free review proxies (not errors)")
    axes[1].legend(fontsize=8)
    figure_path = output / "visualizations/01_threshold_comparison.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(summaries), figsize=(6 * len(summaries), 5), constrained_layout=True)
    if len(summaries) == 1:
        axes = [axes]
    for axis, summary in zip(axes, summaries):
        events = all_events[summary["variant_id"]]
        latest: dict[int, np.ndarray] = {}
        for event in events:
            latest[int(event["entity_ordinal"])] = np.asarray(
                event["entity_position_after_m"], dtype=float
            )
        for ordinal, point in sorted(latest.items()):
            axis.scatter(point[0], point[1], s=18)
            axis.text(point[0], point[1], str(ordinal), fontsize=6)
        axis.set_title(f"{summary['threshold_m']:.2f} m final entity centers")
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world y (m)")
        axis.axis("equal")
        axis.grid(alpha=0.25)
    fig.savefig(output / "visualizations/02_topdown_entity_centers.png", dpi=180)
    plt.close(fig)


def write_report(
    output: Path,
    frames: Sequence[dict[str, Any]],
    geometry_summary: Mapping[str, Any],
    summaries: Sequence[dict[str, Any]],
) -> None:
    table = "\n".join(
        (
            f"| {row['threshold_m']:.2f} m | {row['entity_count']} | "
            f"{row['new_track_merge_event_count']} | "
            f"{row['local_track_reassignment_count']} | "
            f"{row['multi_track_entity_count_proxy']} | "
            f"{row['same_frame_multi_track_collision_count_proxy']} | "
            f"{row['track_to_multiple_entity_count_proxy']} | "
            f"{row['observe_latency_ms_p95']:.3f} |"
        )
        for row in summaries
    )
    report = f"""# E13 实体合并诊断报告（E12-fed、无 GT）

## 结论边界

本实验已经用生产实现 `MapMemory.observe_entity()` 完整重放 {len(frames)} 帧，
并比较正式协议规定的 0.20/0.35/0.50 m 三个中心距离门限。输入不是 GT track
fragment/GT 3D center，而是 E11 FastSAM mask、E12 baseline BotSort track、
双目深度和尺度提案，因此只能作结构性诊断，不能计算实体 P/R/F1，也不能把
multi-track entity 等代理信号称为真实 over-merge。

尺度文件状态为 `proposal_not_frozen`；所有 pre-DAM 名称均为 `unknown`，因此
MapMemory 的同名门在本实验中不能区分语义类别。`SCREENING_RESULT.json`
不会宣布最佳候选。

## 数据流和实际模块

1. 读取 E12 baseline 的 track observation，并通过路径和 SHA-256 回链 E11 mask。
2. 读取 Fast-FoundationStereo uint16 毫米深度，乘以尺度
   `{geometry_summary['depth_scale']:.12f}`。
3. mask 内有效深度至少占 25% 才进入 3D；调用
   `backproject_masked_depth()` 保留像素—深度对应，最多确定性采样 20,000 点。
4. 用每帧 `world_T_camera` 变换到 map/world 系；位置取 3D 点逐分量中位数，
   尺寸取 q05–q95 且每维下限 0.05 m。
5. 三个候选各自创建独立 SQLite；依次调用 `MapMemory.observe_entity()`。
   已映射 local track 若距其 entity 中位中心超门限会先解除映射；随后在全部
   `unknown/object` entity 中选门限内最近者，否则新建 entity。entity 几何随
   历史观察按逐分量中位数更新。

几何输入共 {geometry_summary['input_observations']} 条，接受
{geometry_summary['accepted_observations']} 条，拒绝
{geometry_summary['rejected_observations']} 条。每条接受观察保存点云 NPZ，
每条拒绝观察保存明确原因。

## 候选结果

| 门限 | entity 数 | 新 track 合入已有 entity | local track 重关联 | multi-track entity 代理 | 同帧冲突代理 | track 多 entity 代理 | observe P95 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

这些列回答“系统做了什么”和“哪些位置值得复核”，不回答“合并是否正确”。
门限增大通常减少 entity 数并增加合并机会；是否为改善必须由封存 GT 或人工逐例
复核判定。

## 证据位置

- `geometry_input/`：逐帧尺度化深度、track mask 世界系点云和有效深度 overlay。
- `tables/geometry_observations.*`、`geometry_rejections.*`：3D 输入与拒绝账本。
- `variants/*/map_memory.sqlite3`：每个候选的原生 MapMemory 数据库。
- `variants/*/database_export/`：所有 SQLite 表的 JSONL/CSV 导出。
- `variants/*/merge_events.*`：逐观察候选距离、动作、前后实体中心和源证据。
- `variants/*/entity_membership.*`、`track_entity_timelines.*`：实体/track 双向归属。
- `variants/*/merge_graph.json|dot`：实体合并图。
- `variants/*/frames/*/entity_id_map.png` 与 `entity_overlay.jpg`：逐帧 uint16 实体图和可视化。
- `variants/*/failure_cases.*`：仅供复核的代理失败签名，correctness_label 均为空。
- `visualizations/`：阈值对比与 world XY 实体中心。

## 正式结论仍缺什么

需要按 V1.1 协议提供隔离的 GT track fragments、GT 3D centers 和人工裁决，
才可计算 entity precision/recall/F1、真实 over-merge/over-split 并选择阈值。
在此之前，本实验不得用于冻结生产门限。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.reseal_existing:
        if not args.output.exists():
            raise FileNotFoundError(args.output)
        status = (
            "complete_independently_audited"
            if (args.output / "INDEPENDENT_AUDIT.json").exists()
            and json.loads((args.output / "INDEPENDENT_AUDIT.json").read_text()).get("passed")
            else "complete_pending_independent_audit"
        )
        seal_output(args.output, status=status)
        return 0
    if args.output.exists():
        raise FileExistsError(
            f"output already exists; refuse overwrite: {args.output}"
        )
    args.output.mkdir(parents=True)
    started = utc_now()
    frames, observations, intrinsics, poses, scale = load_inputs(args)
    if not frames:
        raise ValueError("no input frames")
    invocation = {
        "schema": "daaam.g1_no_gt_e13_invocation.v1",
        "started_at": started,
        "argv": sys.argv,
        "repository_root": str(REPOSITORY_ROOT),
        "e12_run": str(args.e12_run.resolve()),
        "depth_run": str(args.depth_run.resolve()),
        "prepared": str(args.prepared.resolve()),
        "scale_proposal": str(args.scale_proposal.resolve()),
        "output": str(args.output.resolve()),
        "maximum_frames": args.maximum_frames,
    }
    write_json(args.output / "invocation.json", invocation)
    write_preregistration(
        args.output,
        args.e12_run,
        args.depth_run,
        args.prepared,
        args.scale_proposal,
        len(frames),
    )
    save_depth_inventory(args.depth_run, args.output)
    write_jsonl(args.output / "input_manifests/e12_frames.jsonl", frames)
    write_csv(args.output / "input_manifests/e12_frames.csv", frames)
    geometry, rejected, geometry_frames = build_geometry(
        args.output,
        args.depth_run,
        frames,
        observations,
        intrinsics,
        poses,
        scale,
    )
    summaries = []
    all_events: dict[str, Sequence[dict[str, Any]]] = {}
    for threshold in THRESHOLDS_M:
        summary, events = run_variant(args.output, threshold, frames, geometry)
        summaries.append(summary)
        all_events[summary["variant_id"]] = events
    write_json(args.output / "tables/variant_summary.json", summaries)
    write_csv(args.output / "tables/variant_summary.csv", summaries)
    failures = []
    for summary in summaries:
        failures.extend(
            read_jsonl(
                args.output
                / "variants"
                / summary["variant_id"]
                / "failure_cases.jsonl"
            )
        )
    write_jsonl(args.output / "failure_cases/failure_cases.jsonl", failures)
    write_csv(args.output / "failure_cases/failure_cases.csv", failures)
    screening = {
        "schema": "daaam.g1_no_gt_e13_screening.v1",
        "status": "diagnostic_only_no_winner",
        "winner": None,
        "candidates_m": list(THRESHOLDS_M),
        "reason": (
            "No GT entity identities/centers; structural proxies cannot determine "
            "whether a merge is correct."
        ),
        "formal_metrics": {
            "entity_precision": None,
            "entity_recall": None,
            "entity_f1": None,
            "over_merge_rate": None,
            "over_split_rate": None,
        },
    }
    write_json(args.output / "SCREENING_RESULT.json", screening)
    create_visualizations(args.output, summaries, all_events)
    geometry_summary = json.loads(
        (args.output / "geometry_input/SUMMARY.json").read_text()
    )
    write_report(args.output, frames, geometry_summary, summaries)
    write_json(
        args.output / "RUN_SUMMARY.json",
        {
            "schema": "daaam.g1_no_gt_e13_run_summary.v1",
            "status": "complete_pending_independent_audit",
            "started_at": started,
            "completed_at": utc_now(),
            "frame_count": len(frames),
            "source_frame_range": [
                min(int(row["source_frame_index"]) for row in frames),
                max(int(row["source_frame_index"]) for row in frames),
            ],
            "geometry": geometry_summary,
            "variant_summaries": summaries,
            "failure_proxy_count": len(failures),
            "formal_claims_permitted": False,
        },
    )
    seal_output(args.output)
    print(f"complete: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
