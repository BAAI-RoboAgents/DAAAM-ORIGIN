#!/usr/bin/env python3
"""Independently verify and replay the retained GT-free E13 evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.memory.store import MapMemory, MapMemoryConfig  # noqa: E402
from daaam.realtime.masked_geometry import backproject_masked_depth  # noqa: E402


DEFAULT_RUN = (
    REPOSITORY_ROOT
    / "experiments/g1_20260724_473_573_v1_1/runs/"
    "diagnostic_gt_free_e13_e12fed_mapmemory_20260729"
)
SESSION_ID = "g1-e13-source-473-573"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def action(prior: str | None, current: str, created: bool) -> str:
    if prior is None and created:
        return "created_new"
    if prior is None and not created:
        return "new_track_merged"
    if prior == current:
        return "local_track_continued"
    if created:
        return "local_track_reassociated_new"
    return "local_track_reassociated_existing"


def verify_inventory(root: Path) -> dict[str, Any]:
    rows = read_jsonl(root / "artifact_inventory.jsonl")
    excluded = {
        "artifact_inventory.csv",
        "artifact_inventory.jsonl",
        "inventory_summary.json",
        "COMPLETION.json",
    }
    retained_paths = {str(row["relative_path"]) for row in rows}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }
    if retained_paths != actual_paths:
        missing = sorted(retained_paths - actual_paths)
        unregistered = sorted(actual_paths - retained_paths)
        raise AssertionError(
            f"inventory path-set mismatch: missing={missing[:5]} "
            f"unregistered={unregistered[:5]}"
        )
    digest = hashlib.sha256()
    total = 0
    for row in rows:
        path = root / row["relative_path"]
        if not path.is_file():
            raise AssertionError(f"missing inventory item: {path}")
        size = path.stat().st_size
        file_hash = sha256_file(path)
        if size != int(row["size_bytes"]) or file_hash != row["sha256"]:
            raise AssertionError(f"inventory mismatch: {path}")
        total += size
        digest.update(
            (
                f"{row['relative_path']}\t{row['size_bytes']}\t{row['sha256']}\n"
            ).encode("utf-8")
        )
    summary = json.loads((root / "inventory_summary.json").read_text())
    if digest.hexdigest() != summary["inventory_root_sha256"]:
        raise AssertionError("inventory root mismatch")
    if len(rows) != int(summary["file_count"]) or total != int(
        summary["total_size_bytes"]
    ):
        raise AssertionError("inventory count/size mismatch")
    return {
        "file_count": len(rows),
        "total_size_bytes": total,
        "inventory_root_sha256": digest.hexdigest(),
        "all_file_hashes_verified": True,
        "path_set_exact": True,
    }


def verify_upstream(root: Path) -> dict[str, Any]:
    prereg = json.loads((root / "PRE_REGISTRATION.json").read_text())
    checks = {}
    for name, reference in prereg["frozen_inputs"].items():
        if not isinstance(reference, Mapping) or "path" not in reference:
            continue
        path = Path(reference["path"])
        passed = (
            path.is_file()
            and path.stat().st_size == int(reference["size_bytes"])
            and sha256_file(path) == reference["sha256"]
        )
        checks[name] = passed
        if not passed:
            raise AssertionError(f"upstream reference mismatch: {name}")
    return {"checked_reference_count": len(checks), "checks": checks}


def verify_geometry(root: Path) -> dict[str, Any]:
    accepted = read_jsonl(root / "tables/geometry_observations.jsonl")
    rejected = read_jsonl(root / "tables/geometry_rejections.jsonl")
    frame_rows = read_jsonl(root / "tables/geometry_frames.jsonl")
    summary = json.loads((root / "geometry_input/SUMMARY.json").read_text())
    if len(accepted) != summary["accepted_observations"]:
        raise AssertionError("accepted geometry count mismatch")
    if len(rejected) != summary["rejected_observations"]:
        raise AssertionError("rejected geometry count mismatch")
    if len(accepted) + len(rejected) != summary["input_observations"]:
        raise AssertionError("geometry accounting mismatch")
    file_hash_cache: dict[Path, str] = {}

    def cached_hash(path: Path) -> str:
        if path not in file_hash_cache:
            file_hash_cache[path] = sha256_file(path)
        return file_hash_cache[path]

    recomputed_points = 0
    for index, row in enumerate(accepted):
        mask_path = Path(row["source_mask_path"])
        depth_path = Path(row["source_depth_path"])
        scaled_path = Path(row["scaled_depth_path"])
        npz_path = Path(row["geometry_npz_path"])
        if cached_hash(mask_path) != row["source_mask_sha256"]:
            raise AssertionError(f"mask hash mismatch at accepted row {index}")
        if cached_hash(depth_path) != row["source_depth_sha256"]:
            raise AssertionError(f"depth hash mismatch at accepted row {index}")
        if cached_hash(npz_path) != row["geometry_npz_sha256"]:
            raise AssertionError(f"NPZ hash mismatch at accepted row {index}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
        scaled_depth = np.load(scaled_path, allow_pickle=False)
        valid_count = int(
            np.count_nonzero(mask & np.isfinite(scaled_depth) & (scaled_depth > 0))
        )
        ratio = valid_count / int(np.count_nonzero(mask))
        if valid_count != row["valid_depth_pixel_count"] or not np.isclose(
            ratio, row["valid_depth_ratio"], atol=0.0, rtol=1e-14
        ):
            raise AssertionError(f"valid depth mismatch at accepted row {index}")
        geometry = backproject_masked_depth(
            mask,
            scaled_depth,
            np.asarray(row["intrinsics"], dtype=np.float64),
            np.asarray(row["world_T_camera"], dtype=np.float64),
            maximum_points=20_000,
        )
        with np.load(npz_path, allow_pickle=False) as retained:
            if not np.array_equal(retained["pixel_yx"], geometry.pixel_yx):
                raise AssertionError(f"pixel correspondence mismatch at row {index}")
            if not np.array_equal(
                retained["points_world_m"], geometry.points_world_m
            ):
                raise AssertionError(f"world points mismatch at row {index}")
            if not np.allclose(
                retained["position_m"], geometry.position_m, atol=1e-12, rtol=0
            ):
                raise AssertionError(f"position mismatch at row {index}")
            if not np.allclose(
                retained["dimensions_m"], geometry.dimensions_m, atol=1e-12, rtol=0
            ):
                raise AssertionError(f"dimensions mismatch at row {index}")
        if not np.allclose(
            geometry.position_m,
            np.asarray(row["position_world_m"]),
            atol=1e-12,
            rtol=0,
        ):
            raise AssertionError(f"JSON position mismatch at row {index}")
        recomputed_points += len(geometry.points_world_m)
    for index, row in enumerate(rejected):
        mask = cv2.imread(row["source_mask_path"], cv2.IMREAD_GRAYSCALE) > 0
        scaled_depth = np.load(row["scaled_depth_path"], allow_pickle=False)
        valid_count = int(
            np.count_nonzero(mask & np.isfinite(scaled_depth) & (scaled_depth > 0))
        )
        mask_count = int(np.count_nonzero(mask))
        ratio = valid_count / mask_count if mask_count else 0.0
        if row["rejection_reason"] == "insufficient_valid_depth_ratio":
            if not ratio < float(row["minimum_valid_depth_ratio"]):
                raise AssertionError(f"invalid rejection at row {index}")
        elif row["rejection_reason"] == "empty_mask":
            if mask_count != 0:
                raise AssertionError(f"invalid empty-mask rejection at row {index}")
        else:
            raise AssertionError(f"unknown rejection reason at row {index}")
    if len(frame_rows) != 101:
        raise AssertionError("geometry frame count is not 101")
    return {
        "accepted_rows_fully_recomputed": len(accepted),
        "rejected_rows_fully_recomputed": len(rejected),
        "world_points_fully_recomputed": recomputed_points,
        "frame_rows": len(frame_rows),
        "exact_pixel_and_float32_point_match": True,
    }


def verify_id_maps(
    variant_root: Path, events: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_frame[int(event["frame_index"])].append(event)
    checked = 0
    total_pixels = 0
    for frame_dir in sorted((variant_root / "frames").iterdir()):
        if not frame_dir.is_dir():
            continue
        frame = json.loads((frame_dir / "frame.json").read_text())
        actual = cv2.imread(
            str(frame_dir / "entity_id_map.png"), cv2.IMREAD_UNCHANGED
        )
        expected = np.zeros(actual.shape, dtype=np.uint16)
        confidence = np.full(actual.shape, -1.0, dtype=np.float32)
        for event in by_frame[int(frame["frame_index"])]:
            mask = cv2.imread(
                event["source_mask_path"], cv2.IMREAD_GRAYSCALE
            ) > 0
            selected = mask & (float(event["model_confidence"]) > confidence)
            expected[selected] = int(event["entity_ordinal"])
            confidence[selected] = float(event["model_confidence"])
        if actual.dtype != np.uint16 or not np.array_equal(actual, expected):
            raise AssertionError(f"ID map mismatch: {frame_dir}")
        checked += 1
        total_pixels += actual.size
    return {
        "id_maps_exactly_reconstructed": checked,
        "pixels_exactly_compared": total_pixels,
    }


def verify_database_and_replay(
    root: Path,
    variant: Mapping[str, Any],
    geometry: Sequence[dict[str, Any]],
    frames: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    identifier = variant["variant_id"]
    variant_root = root / "variants" / identifier
    database = variant_root / "map_memory.sqlite3"
    events = read_jsonl(variant_root / "merge_events.jsonl")
    if len(events) != len(geometry):
        raise AssertionError(f"{identifier}: event count mismatch")
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        entity_count = connection.execute(
            "SELECT COUNT(*) FROM entities WHERE deleted_ns IS NULL"
        ).fetchone()[0]
        observation_count = connection.execute(
            "SELECT COUNT(*) FROM entity_observations"
        ).fetchone()[0]
        audit_count = connection.execute(
            "SELECT COUNT(*) FROM audit_log WHERE action='local_track_reassociated'"
        ).fetchone()[0]
        db_observations = [
            dict(row)
            for row in connection.execute(
                """
                SELECT session_id, local_entity_id, entity_id, sensor_time_ns,
                       position_m, dimensions_m, confidence
                FROM entity_observations ORDER BY observation_id
                """
            )
        ]
    if integrity != "ok":
        raise AssertionError(f"{identifier}: SQLite integrity failed")
    if entity_count != variant["entity_count"]:
        raise AssertionError(f"{identifier}: entity count mismatch")
    if observation_count != len(events):
        raise AssertionError(f"{identifier}: observation count mismatch")
    if audit_count != variant["local_track_reassignment_count"]:
        raise AssertionError(f"{identifier}: reassociation audit count mismatch")
    for event, db_row in zip(events, db_observations):
        if (
            event["local_entity_id"] != db_row["local_entity_id"]
            or event["entity_id"] != db_row["entity_id"]
            or event["sensor_time_ns"] != db_row["sensor_time_ns"]
            or not np.allclose(
                event["observation_position_m"],
                json.loads(db_row["position_m"]),
                atol=1e-12,
                rtol=0,
            )
        ):
            raise AssertionError(f"{identifier}: DB/event linkage mismatch")
    geometry_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in geometry:
        geometry_by_frame[int(row["frame_index"])].append(row)
    with tempfile.TemporaryDirectory(prefix=f"daaam-{identifier}-replay-") as temp:
        replay = MapMemory(
            Path(temp) / "replay.sqlite3",
            MapMemoryConfig(entity_merge_distance_m=float(variant["threshold_m"])),
        )
        replay.create_session(
            SESSION_ID, int(frames[0]["sensor_time_ns"]), canonical=True
        )
        local_mapping: dict[str, str] = {}
        ordinals: dict[str, int] = {}
        replay_index = 0
        for frame in frames:
            for row in sorted(
                geometry_by_frame[int(frame["frame_index"])],
                key=lambda item: (
                    int(item["track_id"]),
                    int(item["e11_instance_id"]),
                ),
            ):
                prior = local_mapping.get(row["local_entity_id"])
                entity_id, created = replay.observe_entity(
                    SESSION_ID,
                    row["local_entity_id"],
                    np.asarray(row["position_world_m"], dtype=np.float64),
                    sensor_time_ns=int(row["sensor_time_ns"]),
                    semantic_label="unknown",
                    dimensions_m=np.asarray(
                        row["dimensions_world_m"], dtype=np.float64
                    ),
                    confidence=float(row["model_confidence"]),
                    entity_type="object",
                )
                local_mapping[row["local_entity_id"]] = entity_id
                if entity_id not in ordinals:
                    ordinals[entity_id] = len(ordinals) + 1
                retained = events[replay_index]
                if bool(created) != retained["created"]:
                    raise AssertionError(f"{identifier}: replay created flag mismatch")
                if ordinals[entity_id] != retained["entity_ordinal"]:
                    raise AssertionError(f"{identifier}: replay ordinal mismatch")
                if action(prior, entity_id, created) != retained["action"]:
                    raise AssertionError(f"{identifier}: replay action mismatch")
                replay_index += 1
        replay_entity_count = len(replay.list_entities())
        replay.close()
    if replay_entity_count != entity_count or replay_index != len(events):
        raise AssertionError(f"{identifier}: replay terminal mismatch")
    membership = json.loads((variant_root / "entity_membership.json").read_text())
    if sum(int(row["observation_count"]) for row in membership) != len(events):
        raise AssertionError(f"{identifier}: membership accounting mismatch")
    id_map_result = verify_id_maps(variant_root, events)
    return {
        "sqlite_integrity_check": integrity,
        "entities_verified": entity_count,
        "observations_verified": observation_count,
        "reassociation_audit_rows_verified": audit_count,
        "database_event_links_verified": len(events),
        "independent_replay_events_exact_action_and_ordinal": replay_index,
        "independent_replay_entity_count": replay_entity_count,
        **id_map_result,
    }


def verify_metric_contract(root: Path) -> dict[str, Any]:
    summaries = json.loads((root / "tables/variant_summary.json").read_text())
    screening = json.loads((root / "SCREENING_RESULT.json").read_text())
    expected = [0.20, 0.35, 0.50]
    if [row["threshold_m"] for row in summaries] != expected:
        raise AssertionError("formal candidate list mismatch")
    null_fields = [
        "formal_entity_precision",
        "formal_entity_recall",
        "formal_entity_f1",
        "formal_over_merge_rate",
        "formal_over_split_rate",
    ]
    for row in summaries:
        if any(row[field] is not None for field in null_fields):
            raise AssertionError("formal metric was populated without GT")
    if screening["winner"] is not None or screening["status"] != "diagnostic_only_no_winner":
        raise AssertionError("GT-free run improperly selected a winner")
    return {
        "formal_candidates_exact": expected,
        "formal_metrics_null": True,
        "winner_null": True,
        "diagnostic_boundary_preserved": True,
    }


def main() -> int:
    args = parse_args()
    root = args.run.resolve()
    if (root / "INDEPENDENT_AUDIT.json").exists():
        raise FileExistsError("audit already exists; refuse overwrite")
    started = utc_now()
    inventory = verify_inventory(root)
    upstream = verify_upstream(root)
    geometry_result = verify_geometry(root)
    geometry = read_jsonl(root / "tables/geometry_observations.jsonl")
    frames = read_jsonl(root / "input_manifests/e12_frames.jsonl")
    variants = json.loads((root / "tables/variant_summary.json").read_text())
    variant_results = {}
    for variant in variants:
        print(f"verifying {variant['variant_id']} ...", flush=True)
        variant_results[variant["variant_id"]] = verify_database_and_replay(
            root, variant, geometry, frames
        )
    metric_contract = verify_metric_contract(root)
    audit = {
        "schema": "daaam.g1_no_gt_e13_independent_audit.v1",
        "passed": True,
        "started_at": started,
        "completed_at": utc_now(),
        "auditor_script": str(Path(__file__).resolve()),
        "auditor_script_sha256": sha256_file(Path(__file__)),
        "scope": (
            "Full artifact hashes, upstream references, all geometry, all SQLite "
            "links, full MapMemory replay, every entity-ID map, and metric boundary."
        ),
        "inventory_before_audit_append": inventory,
        "upstream_references": upstream,
        "geometry": geometry_result,
        "variants": variant_results,
        "metric_contract": metric_contract,
        "correctness_claim": (
            "Evidence integrity passed; entity merge correctness remains unavailable "
            "without reviewed GT."
        ),
    }
    write_json(root / "INDEPENDENT_AUDIT.json", audit)
    print(json.dumps(audit, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
