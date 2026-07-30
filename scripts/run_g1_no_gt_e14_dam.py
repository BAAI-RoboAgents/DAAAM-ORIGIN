#!/usr/bin/env python3
"""Run E14 DAM observation-threshold diagnostics on frozen E13 entities.

The formal E14 protocol requires reviewed GT entity crops/observations.  Those
are unavailable, so this collector is explicitly E13-fed and GT-free.  It uses
the production DAMAgentPanoptic model/query path, preserves every prompt mask,
crop, response, correction operation, and cloned MapMemory database, and never
reports name accuracy or a winning threshold.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.memory.store import MapMemory, MapMemoryConfig  # noqa: E402
from daaam.query_manager.dam.services import DAMAgentPanoptic  # noqa: E402
from daaam.realtime.contracts import SemanticCorrection  # noqa: E402


EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments/g1_20260724_473_573_v1_1"
DEFAULT_E13 = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e13_e12fed_mapmemory_20260729"
)
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e14_e13fed_dam_20260729"
)
E13_VARIANT = "merge_0p50m"
THRESHOLDS = (3, 5, 8)
SEEDS = (0, 1, 2)
MODEL_LINK = REPOSITORY_ROOT / "checkpoints/dam/DAM-3B"
MODEL_INVENTORY = (
    EXPERIMENT_ROOT / "diagnostic_no_gt/model_inventory_full"
)
PROTOCOL = REPOSITORY_ROOT / "docs/g1_semantic_map_experiments_v1_1.md"
DIAGNOSTIC_PROTOCOL = (
    REPOSITORY_ROOT / "docs/g1_semantic_map_diagnostic_no_gt_stage.md"
)
REALTIME_CONFIG = REPOSITORY_ROOT / "config/pipeline_config_realtime.yaml"
DAM_AGENT_SOURCE = REPOSITORY_ROOT / "src/daaam/query_manager/dam/services.py"
DAM_WORKER_SOURCE = (
    REPOSITORY_ROOT / "src/daaam/grounding/workers/dam_grounding.py"
)
SEMANTIC_SOURCE = REPOSITORY_ROOT / "src/daaam/realtime/semantic.py"
MEMORY_SOURCE = REPOSITORY_ROOT / "src/daaam/memory/store.py"
SESSION_ID = "g1-e13-source-473-573"
QUERY = "Describe what you see in this region."
TEMPERATURE = 0.2
TOP_P = 0.9
MAX_NEW_TOKENS = 512
MULTI_IMAGE_MIN_MASKS = 16
AUTOMATIC_CONFIDENCE = 0.5
INVENTORY_EXCLUDES = {
    "artifact_inventory.csv",
    "artifact_inventory.jsonl",
    "inventory_summary.json",
    "COMPLETION.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e13-run", type=Path, default=DEFAULT_E13)
    parser.add_argument("--e13-variant", default=E13_VARIANT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prompt-policy",
        choices=("legacy", "unique_safe"),
        default="legacy",
        help=(
            "legacy counts every track observation; unique_safe counts one "
            "entity observation per frame, selects one mask, and rejects "
            "entities after a same-frame multi-track collision"
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(THRESHOLDS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--maximum-prompt-records",
        type=int,
        help="Smoke-test only: truncate each threshold after this many prompt records.",
    )
    parser.add_argument(
        "--reseal-existing",
        action="store_true",
        help="Only rebuild the final inventory/completion seal.",
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
        raise OSError(path)


def hash_reference(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def inventory_rows(root: Path, excluded: set[str] | None = None) -> list[dict[str, Any]]:
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
            f"{row['relative_path']}\t{row['size_bytes']}\t{row['sha256']}\n".encode()
        )
    return digest.hexdigest()


def seal_output(root: Path, status: str = "complete_pending_independent_audit") -> None:
    rows = inventory_rows(root, INVENTORY_EXCLUDES)
    root_hash = inventory_root(rows)
    write_jsonl(root / "artifact_inventory.jsonl", rows)
    write_csv(root / "artifact_inventory.csv", rows)
    write_json(
        root / "inventory_summary.json",
        {
            "schema": "daaam.artifact_inventory.v1",
            "generated_at": utc_now(),
            "file_count": len(rows),
            "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
            "inventory_root_sha256": root_hash,
            "excluded_self_referential_files": sorted(INVENTORY_EXCLUDES),
        },
    )
    write_json(
        root / "COMPLETION.json",
        {
            "schema": "daaam.g1_no_gt_e14_completion.v1",
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
        },
    )


def nvidia_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,driver_version,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(value: str) -> str:
    return " ".join(str(value).split()).strip()


def normalized_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def jaccard(first: str, second: str) -> float:
    left = normalized_tokens(first)
    right = normalized_tokens(second)
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def numeric_summary(values: Iterable[float | int]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def sqlite_backup(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as source_connection:
        with sqlite3.connect(target) as target_connection:
            source_connection.backup(target_connection)


def export_database(database: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        tables = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name
                """
            )
        ]
        counts = {}
        for table in tables:
            rows = [
                dict(row)
                for row in connection.execute(f'SELECT * FROM "{table}"')
            ]
            counts[table] = len(rows)
            write_jsonl(output / f"{table}.jsonl", rows)
            write_csv(output / f"{table}.csv", rows)
    summary = {
        "schema": "daaam.g1_no_gt_e14_database_export.v1",
        "database_path": str(database.resolve()),
        "database_sha256": sha256_file(database),
        "sqlite_integrity_check": integrity,
        "table_row_counts": counts,
    }
    write_json(output / "SUMMARY.json", summary)
    return summary


def model_identity() -> dict[str, Any]:
    snapshot = MODEL_LINK.resolve()
    historical_summary = json.loads(
        (MODEL_INVENTORY / "inventory_summary.json").read_text()
    )
    return {
        "configured_path": str(MODEL_LINK.relative_to(REPOSITORY_ROOT)),
        "configured_path_absolute": str(MODEL_LINK.resolve()),
        "snapshot_id": snapshot.name,
        "snapshot_path": str(snapshot),
        "snapshot_file_count": 20,
        "snapshot_bytes_hashed": 7_120_506_885,
        "historical_model_inventory": hash_reference(
            MODEL_INVENTORY / "artifact_inventory.jsonl"
        ),
        "historical_model_inventory_summary": hash_reference(
            MODEL_INVENTORY / "inventory_summary.json"
        ),
        "historical_inventory_record": historical_summary,
    }


def write_preregistration(
    output: Path,
    e13: Path,
    e13_variant: str,
    prompt_policy: str,
    thresholds: Sequence[int],
    seeds: Sequence[int],
    maximum_prompt_records: int | None,
) -> None:
    completion = json.loads((e13 / "COMPLETION.json").read_text())
    source_variant = e13 / "variants" / e13_variant
    preregistration = {
        "schema": "daaam.g1_no_gt_e14_preregistration.v1",
        "registered_at": utc_now(),
        "stage": "E14 DAM observation threshold",
        "status": "diagnostic_gt_free_e13_fed",
        "formal_protocol_candidates": [3, 5, 8],
        "executed_thresholds": list(thresholds),
        "seeds": list(seeds),
        "smoke_truncation": maximum_prompt_records,
        "fixed_upstream": {
            "e13_variant": e13_variant,
            "selection_reason": (
                "explicit invocation; no formal E13 winner without independent GT"
            ),
            "e13_inventory_root_sha256": completion[
                "artifact_inventory_root_sha256"
            ],
            "e13_completion": hash_reference(e13 / "COMPLETION.json"),
            "e13_merge_events": hash_reference(source_variant / "merge_events.jsonl"),
            "e13_membership": hash_reference(source_variant / "entity_membership.json"),
            "e13_map_memory": hash_reference(source_variant / "map_memory.sqlite3"),
            "e13_frame_manifest": hash_reference(
                e13 / "input_manifests/e12_frames.jsonl"
            ),
        },
        "prompt_contract": {
            "policy": prompt_policy,
            "counter_domain": "MapMemory entity, not frame and not BotSort track",
            "counted_observations": (
                "legacy: every accepted entity-track observation; unique_safe: "
                "one accepted observation per entity per segmentation frame"
            ),
            "trigger": (
                "after all accepted observations in the current frame increment "
                "entity counts, prompt every current-frame track whose entity count "
                "meets N and whose entity revision has not been prompted"
            ),
            "same_frame_duplicate_entity_masks": (
                "legacy retains duplicates; unique_safe rejects colliding entities "
                "and otherwise selects one largest-area mask per entity"
            ),
            "map_revision": 0,
            "semantic_id": "E13 entity ordinal",
        },
        "dam_contract": {
            "implementation": "DAMAgentPanoptic.query_multi_image_multi_mask",
            "model": model_identity(),
            "query": QUERY,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "conv_mode": "v1",
            "prompt_mode": "focal_prompt",
            "full_image_description": False,
            "multi_image_min_n_masks": MULTI_IMAGE_MIN_MASKS,
            "batching": (
                "ordered prompt records aggregate until >=16 masks; final underfilled "
                "batch is explicitly drained"
            ),
            "seed_reset": "python, numpy, torch CPU and all CUDA generators per cell",
            "clip_feature_extraction": (
                "omitted: it does not condition DAM generation and is outside "
                "the E14 naming/description isolation"
            ),
        },
        "correction_contract": {
            "database": (
                f"fresh SQLite backup of frozen E13 {e13_variant} per cell"
            ),
            "automatic_confidence": AUTOMATIC_CONFIDENCE,
            "operation_id": (
                "SHA256(source|request_id|entity_id|map_revision|label.casefold())"
            ),
            "batch_application": (
                "enqueue responses in model order, then apply pending corrections "
                "after each DAM batch; same-entity pending operations obey production "
                "supersession rules"
            ),
            "hydra_delivery": "not executed in isolated E14; E17 owns mesh binding",
        },
        "reported_metrics": {
            "exact": [
                "eligible entities",
                "prompt records/masks/batches",
                "response and correction counts",
                "description coverage",
                "first-description frame/time delay",
                "DAM latency",
                "pending/superseded/applied operation counts",
            ],
            "proxy": [
                "cross-seed exact agreement and token Jaccard",
                "cross-threshold token Jaccard",
                "same-entity duplicate-mask description conflict",
                "empty/unknown/long response review queues",
            ],
            "unavailable": [
                "name accuracy",
                "description correctness",
                "true early misnaming rate",
            ],
        },
        "forbidden_claims": [
            "best threshold",
            "name accuracy",
            "semantic correctness",
            "formal E14 isolated result",
        ],
        "sources": {
            "formal_protocol": hash_reference(PROTOCOL),
            "diagnostic_protocol": hash_reference(DIAGNOSTIC_PROTOCOL),
            "realtime_config": hash_reference(REALTIME_CONFIG),
            "dam_agent_source": hash_reference(DAM_AGENT_SOURCE),
            "dam_worker_source": hash_reference(DAM_WORKER_SOURCE),
            "semantic_adapter_source": hash_reference(SEMANTIC_SOURCE),
            "map_memory_source": hash_reference(MEMORY_SOURCE),
            "collector": hash_reference(Path(__file__)),
        },
    }
    write_json(output / "PRE_REGISTRATION.json", preregistration)


def prompt_records_for_threshold(
    threshold: int,
    events: Sequence[dict[str, Any]],
    frames: Mapping[int, dict[str, Any]],
    maximum_prompt_records: int | None,
    prompt_policy: str = "legacy",
) -> list[dict[str, Any]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_frame[int(event["frame_index"])].append(event)
    first_by_entity = {}
    for event in events:
        first_by_entity.setdefault(event["entity_id"], event)
    counts: dict[str, int] = defaultdict(int)
    prompted: set[str] = set()
    colliding_entities: set[str] = set()
    records = []
    request_index = 0
    for frame_index in sorted(by_frame):
        current = sorted(
            by_frame[frame_index],
            key=lambda row: (int(row["track_id"]), int(row["e11_instance_id"])),
        )
        current_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in current:
            current_by_entity[str(event["entity_id"])].append(event)
        if prompt_policy == "legacy":
            for event in current:
                counts[event["entity_id"]] += 1
            selected = [
                event
                for event in current
                if counts[event["entity_id"]] >= threshold
                and event["entity_id"] not in prompted
            ]
        elif prompt_policy == "unique_safe":
            for entity_id, entity_events in current_by_entity.items():
                counts[entity_id] += 1
                if len({int(event["track_id"]) for event in entity_events}) > 1:
                    colliding_entities.add(entity_id)
            selected = []
            for entity_id, entity_events in current_by_entity.items():
                if (
                    counts[entity_id] < threshold
                    or entity_id in prompted
                    or entity_id in colliding_entities
                ):
                    continue
                selected.append(
                    max(
                        entity_events,
                        key=lambda event: (
                            int(
                                np.count_nonzero(
                                    cv2.imread(
                                        event["source_mask_path"],
                                        cv2.IMREAD_GRAYSCALE,
                                    )
                                )
                            ),
                            -int(event["track_id"]),
                        ),
                    )
                )
        else:
            raise ValueError(f"unsupported E14 prompt policy: {prompt_policy}")
        if not selected:
            continue
        if maximum_prompt_records is not None and len(records) >= maximum_prompt_records:
            break
        frame = frames[frame_index]
        entities_by_track = {
            int(event["track_id"]): event["entity_id"] for event in selected
        }
        material = "|".join(
            [
                SESSION_ID,
                str(frame["sensor_time_ns"]),
                "0",
                *(entities_by_track[key] for key in sorted(entities_by_track)),
            ]
        )
        request_id = hashlib.sha256(material.encode()).hexdigest()
        requests = []
        for event in selected:
            first = first_by_entity[event["entity_id"]]
            requests.append(
                {
                    "request_index": request_index,
                    "entity_id": event["entity_id"],
                    "entity_ordinal": int(event["entity_ordinal"]),
                    "semantic_id": int(event["entity_ordinal"]),
                    "track_id": int(event["track_id"]),
                    "e11_instance_id": int(event["e11_instance_id"]),
                    "observation_count_at_prompt": counts[event["entity_id"]],
                    "first_frame_index": int(first["frame_index"]),
                    "first_source_frame_index": int(first["source_frame_index"]),
                    "first_sensor_time_ns": int(first["sensor_time_ns"]),
                    "delay_frames": frame_index - int(first["frame_index"]),
                    "delay_seconds": (
                        int(frame["sensor_time_ns"]) - int(first["sensor_time_ns"])
                    )
                    / 1.0e9,
                    "source_mask_path": event["source_mask_path"],
                    "source_mask_sha256": event["source_mask_sha256"],
                    "model_confidence": float(event["model_confidence"]),
                }
            )
            request_index += 1
        records.append(
            {
                "schema": "daaam.g1_no_gt_e14_prompt_record.v1",
                "threshold_observations": threshold,
                "prompt_policy": prompt_policy,
                "prompt_record_index": len(records),
                "request_id": request_id,
                "frame_index": frame_index,
                "source_frame_index": int(frame["source_frame_index"]),
                "sensor_time_ns": int(frame["sensor_time_ns"]),
                "map_revision": 0,
                "rgb_path": frame["rgb_path"],
                "rgb_sha256": frame["rgb_sha256"],
                "requests": requests,
            }
        )
        prompted.update(event["entity_id"] for event in selected)
    return records


def materialize_prompt_inputs(
    output: Path, threshold: int, records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    threshold_root = output / "prompt_inputs" / f"obs_{threshold:02d}"
    record_rows = []
    request_rows = []
    for record in records:
        record_index = int(record["prompt_record_index"])
        record_root = threshold_root / "records" / f"record_{record_index:04d}"
        rgb_bgr = cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise ValueError(record["rgb_path"])
        overlay = rgb_bgr.copy()
        entity_counts: dict[str, int] = defaultdict(int)
        serialized_requests = []
        for local_index, request in enumerate(record["requests"]):
            mask = cv2.imread(
                request["source_mask_path"], cv2.IMREAD_GRAYSCALE
            )
            if mask is None or mask.shape != rgb_bgr.shape[:2]:
                raise ValueError(request["source_mask_path"])
            binary = mask > 0
            ys, xs = np.nonzero(binary)
            if not len(xs):
                raise ValueError("empty E14 prompt mask")
            x1, y1, x2, y2 = (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            )
            mask_path = record_root / "masks" / f"request_{local_index:03d}.png"
            require_image(mask_path, (binary.astype(np.uint8) * 255))
            margin_x = max(2, int(round((x2 - x1) * 0.10)))
            margin_y = max(2, int(round((y2 - y1) * 0.10)))
            sx1, sy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
            sx2, sy2 = min(rgb_bgr.shape[1], x2 + margin_x), min(
                rgb_bgr.shape[0], y2 + margin_y
            )
            crop = rgb_bgr[sy1:sy2, sx1:sx2].copy()
            crop_mask = binary[sy1:sy2, sx1:sx2]
            dimmed = (crop.astype(np.float32) * 0.25).astype(np.uint8)
            dimmed[crop_mask] = crop[crop_mask]
            crop_path = record_root / "crops" / f"request_{local_index:03d}.jpg"
            require_image(crop_path, dimmed)
            color = (
                int((request["entity_ordinal"] * 67) % 255),
                int((request["entity_ordinal"] * 113) % 255),
                int((request["entity_ordinal"] * 173) % 255),
            )
            overlay[binary] = (
                0.65 * overlay[binary] + 0.35 * np.asarray(color)
            ).astype(np.uint8)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                overlay,
                f"E{request['entity_ordinal']} T{request['track_id']}",
                (x1, max(14, y1 + 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
            entity_counts[request["entity_id"]] += 1
            serialized = {
                **request,
                "prompt_local_index": local_index,
                "mask_area_px": int(np.count_nonzero(binary)),
                "bbox_xyxy": [x1, y1, x2, y2],
                "materialized_mask_path": str(mask_path.resolve()),
                "materialized_mask_sha256": sha256_file(mask_path),
                "crop_preview_path": str(crop_path.resolve()),
                "crop_preview_sha256": sha256_file(crop_path),
                "crop_preview_bbox_xyxy": [sx1, sy1, sx2, sy2],
            }
            serialized_requests.append(serialized)
            request_rows.append(
                {
                    "schema": "daaam.g1_no_gt_e14_prompt_request.v1",
                    "threshold_observations": threshold,
                    "prompt_record_index": record_index,
                    "request_id": record["request_id"],
                    "frame_index": record["frame_index"],
                    "source_frame_index": record["source_frame_index"],
                    "sensor_time_ns": record["sensor_time_ns"],
                    "rgb_path": record["rgb_path"],
                    "rgb_sha256": record["rgb_sha256"],
                    **serialized,
                }
            )
        overlay_path = record_root / "prompt_overlay.jpg"
        require_image(overlay_path, overlay)
        serialized_record = {
            **record,
            "requests": serialized_requests,
            "unique_entity_count": len(entity_counts),
            "same_entity_extra_mask_count": sum(
                value - 1 for value in entity_counts.values() if value > 1
            ),
            "prompt_overlay_path": str(overlay_path.resolve()),
            "prompt_overlay_sha256": sha256_file(overlay_path),
        }
        write_json(record_root / "record.json", serialized_record)
        record_rows.append(serialized_record)
    write_jsonl(threshold_root / "prompt_records.jsonl", record_rows)
    write_jsonl(threshold_root / "prompt_requests.jsonl", request_rows)
    write_csv(threshold_root / "prompt_requests.csv", request_rows)
    unique_entities = {row["entity_id"] for row in request_rows}
    write_json(
        threshold_root / "SUMMARY.json",
        {
            "schema": "daaam.g1_no_gt_e14_prompt_input_summary.v1",
            "threshold_observations": threshold,
            "prompt_record_count": len(record_rows),
            "prompt_mask_request_count": len(request_rows),
            "unique_eligible_entity_count": len(unique_entities),
            "same_entity_extra_mask_request_count": len(request_rows)
            - len(unique_entities),
        },
    )
    return record_rows, request_rows


def make_batches(records: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    batches = []
    pending = []
    total_masks = 0
    for record in records:
        pending.append(record)
        total_masks += len(record["requests"])
        if total_masks >= MULTI_IMAGE_MIN_MASKS:
            batches.append(pending)
            pending = []
            total_masks = 0
    if pending:
        batches.append(pending)
    return batches


def apply_responses_to_memory(
    source_database: Path,
    cell_root: Path,
    response_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    database = cell_root / "map_memory.sqlite3"
    sqlite_backup(source_database, database)
    memory = MapMemory(
        database, MapMemoryConfig(entity_merge_distance_m=0.50)
    )
    receipts = []
    applied_rows = []
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in response_rows:
        by_batch[int(row["batch_index"])].append(row)
    for batch_index in sorted(by_batch):
        for row in by_batch[batch_index]:
            label = normalize_label(row["description"])
            if not label or label.casefold() == "unknown":
                receipts.append(
                    {
                        "response_index": row["response_index"],
                        "operation_id": None,
                        "enqueue_status": "skipped",
                        "enqueue_reason": "empty_or_unknown",
                        "duplicate": False,
                    }
                )
                continue
            source = "dam:checkpoints/dam/DAM-3B"
            material = "|".join(
                [
                    source,
                    row["request_id"],
                    row["entity_id"],
                    "0",
                    label.casefold(),
                ]
            )
            operation_id = hashlib.sha256(material.encode()).hexdigest()
            receipt = memory.enqueue_correction(
                SemanticCorrection(
                    operation_id=operation_id,
                    entity_id=row["entity_id"],
                    sensor_time_ns=int(row["sensor_time_ns"]),
                    map_revision=0,
                    label=label,
                    confidence=AUTOMATIC_CONFIDENCE,
                    source=source,
                )
            )
            receipts.append(
                {
                    "response_index": row["response_index"],
                    "operation_id": operation_id,
                    "enqueue_status": receipt.status,
                    "enqueue_reason": receipt.reason,
                    "duplicate": receipt.duplicate,
                }
            )
        for receipt in memory.apply_pending_corrections(limit=10_000):
            applied_rows.append(
                {
                    "batch_index": batch_index,
                    "operation_id": receipt.operation_id,
                    "status": receipt.status,
                    "reason": receipt.reason,
                }
            )
    for receipt in memory.apply_pending_corrections(limit=10_000):
        applied_rows.append(
            {
                "batch_index": None,
                "operation_id": receipt.operation_id,
                "status": receipt.status,
                "reason": receipt.reason,
            }
        )
    final_entities = memory.list_entities()
    memory.close()
    export = export_database(database, cell_root / "database_export")
    write_jsonl(cell_root / "correction_receipts.jsonl", receipts)
    write_csv(cell_root / "correction_receipts.csv", receipts)
    write_jsonl(cell_root / "applied_corrections.jsonl", applied_rows)
    write_csv(cell_root / "applied_corrections.csv", applied_rows)
    write_json(cell_root / "final_entities.json", final_entities)
    return export, receipts, applied_rows


def run_cell(
    agent: DAMAgentPanoptic,
    output: Path,
    source_database: Path,
    threshold: int,
    seed: int,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    cell_root = output / "cells" / f"obs_{threshold:02d}" / f"seed_{seed}"
    cell_root.mkdir(parents=True)
    set_seed(seed)
    batches = make_batches(records)
    response_rows = []
    batch_rows = []
    console_path = cell_root / "model_console.log"
    response_index = 0
    with console_path.open("w", encoding="utf-8") as console:
        for batch_index, batch in enumerate(batches):
            image_mask_pairs = []
            for record in batch:
                rgb_bgr = cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR)
                if rgb_bgr is None:
                    raise ValueError(record["rgb_path"])
                image = PILImage.fromarray(
                    cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                )
                masks = [
                    PILImage.fromarray(
                        cv2.imread(
                            request["materialized_mask_path"],
                            cv2.IMREAD_GRAYSCALE,
                        )
                    )
                    for request in record["requests"]
                ]
                image_mask_pairs.append((image, masks))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pre_gpu = nvidia_snapshot()
            started = time.perf_counter()
            with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
                descriptions = agent.query_multi_image_multi_mask(
                    image_mask_pairs=image_mask_pairs,
                    query=QUERY,
                    batch_size=None,
                    auto_batch=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    num_beams=1,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_s = time.perf_counter() - started
            post_gpu = nvidia_snapshot()
            if len(descriptions) != len(batch):
                raise RuntimeError("DAM image response count mismatch")
            batch_mask_count = sum(len(record["requests"]) for record in batch)
            batch_record = {
                "schema": "daaam.g1_no_gt_e14_dam_batch.v1",
                "threshold_observations": threshold,
                "seed": seed,
                "batch_index": batch_index,
                "prompt_record_indices": [
                    record["prompt_record_index"] for record in batch
                ],
                "image_count": len(batch),
                "mask_count": batch_mask_count,
                "latency_s": latency_s,
                "latency_ms_per_mask": latency_s * 1000.0 / batch_mask_count,
                "pre_gpu": pre_gpu,
                "post_gpu": post_gpu,
            }
            batch_rows.append(batch_record)
            write_json(cell_root / "batches" / f"batch_{batch_index:03d}.json", batch_record)
            for record, record_descriptions in zip(batch, descriptions, strict=True):
                if len(record_descriptions) != len(record["requests"]):
                    raise RuntimeError("DAM mask response count mismatch")
                for request, raw_description in zip(
                    record["requests"], record_descriptions, strict=True
                ):
                    description = normalize_label(raw_description)
                    response_path = (
                        cell_root / "responses" / f"response_{response_index:04d}.txt"
                    )
                    response_path.parent.mkdir(parents=True, exist_ok=True)
                    response_path.write_text(description + "\n", encoding="utf-8")
                    row = {
                        "schema": "daaam.g1_no_gt_e14_dam_response.v1",
                        "threshold_observations": threshold,
                        "seed": seed,
                        "response_index": response_index,
                        "batch_index": batch_index,
                        "prompt_record_index": record["prompt_record_index"],
                        "request_index": request["request_index"],
                        "request_id": record["request_id"],
                        "frame_index": record["frame_index"],
                        "source_frame_index": record["source_frame_index"],
                        "sensor_time_ns": record["sensor_time_ns"],
                        "entity_id": request["entity_id"],
                        "entity_ordinal": request["entity_ordinal"],
                        "semantic_id": request["semantic_id"],
                        "track_id": request["track_id"],
                        "e11_instance_id": request["e11_instance_id"],
                        "observation_count_at_prompt": request[
                            "observation_count_at_prompt"
                        ],
                        "delay_frames": request["delay_frames"],
                        "delay_seconds": request["delay_seconds"],
                        "mask_area_px": request["mask_area_px"],
                        "source_mask_path": request["source_mask_path"],
                        "source_mask_sha256": request["source_mask_sha256"],
                        "materialized_mask_path": request[
                            "materialized_mask_path"
                        ],
                        "materialized_mask_sha256": request[
                            "materialized_mask_sha256"
                        ],
                        "crop_preview_path": request["crop_preview_path"],
                        "crop_preview_sha256": request["crop_preview_sha256"],
                        "description": description,
                        "description_character_count": len(description),
                        "description_token_proxy_count": len(
                            re.findall(r"[A-Za-z0-9]+", description)
                        ),
                        "empty": not bool(description),
                        "unknown": description.casefold() == "unknown",
                        "response_path": str(response_path.resolve()),
                        "response_sha256": sha256_file(response_path),
                    }
                    response_rows.append(row)
                    write_json(
                        cell_root
                        / "responses"
                        / f"response_{response_index:04d}.json",
                        row,
                    )
                    response_index += 1
            print(
                f"obs={threshold} seed={seed} batch={batch_index + 1}/{len(batches)} "
                f"images={len(batch)} masks={batch_mask_count} "
                f"latency={latency_s:.3f}s",
                flush=True,
            )
    write_jsonl(cell_root / "responses.jsonl", response_rows)
    write_csv(cell_root / "responses.csv", response_rows)
    write_jsonl(cell_root / "batches.jsonl", batch_rows)
    write_csv(cell_root / "batches.csv", batch_rows)
    export, receipts, applied_rows = apply_responses_to_memory(
        source_database, cell_root, response_rows
    )
    final_entities = json.loads((cell_root / "final_entities.json").read_text())
    entity_names = {
        row["entity_id"]: row["canonical_name"] for row in final_entities
    }
    unique_eligible = sorted({row["entity_id"] for row in response_rows})
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for row in response_rows:
        duplicate_groups[row["entity_id"]].append(row["description"])
    duplicate_entities = {
        entity: labels for entity, labels in duplicate_groups.items() if len(labels) > 1
    }
    conflicting_duplicate_entities = {
        entity: labels
        for entity, labels in duplicate_entities.items()
        if len({normalize_label(label).casefold() for label in labels}) > 1
    }
    summary = {
        "schema": "daaam.g1_no_gt_e14_cell_summary.v1",
        "threshold_observations": threshold,
        "seed": seed,
        "total_e13_entities": len(final_entities),
        "eligible_entity_count": len(unique_eligible),
        "prompt_record_count": len(records),
        "prompt_mask_request_count": len(response_rows),
        "same_entity_extra_mask_request_count": len(response_rows)
        - len(unique_eligible),
        "dam_batch_call_count": len(batch_rows),
        "response_count": len(response_rows),
        "nonempty_response_count": sum(not row["empty"] for row in response_rows),
        "unknown_response_count": sum(row["unknown"] for row in response_rows),
        "responded_entity_count": len(
            {row["entity_id"] for row in response_rows if not row["empty"]}
        ),
        "eligible_description_coverage": (
            len({row["entity_id"] for row in response_rows if not row["empty"]})
            / len(unique_eligible)
            if unique_eligible
            else None
        ),
        "all_entity_description_coverage": (
            len({row["entity_id"] for row in response_rows if not row["empty"]})
            / len(final_entities)
        ),
        "duplicate_prompt_entity_count": len(duplicate_entities),
        "conflicting_duplicate_description_entity_count_proxy": len(
            conflicting_duplicate_entities
        ),
        "first_description_delay_frames": numeric_summary(
            min(
                row["delay_frames"]
                for row in response_rows
                if row["entity_id"] == entity_id
            )
            for entity_id in unique_eligible
        ),
        "first_description_delay_seconds": numeric_summary(
            min(
                row["delay_seconds"]
                for row in response_rows
                if row["entity_id"] == entity_id
            )
            for entity_id in unique_eligible
        ),
        "description_character_count": numeric_summary(
            row["description_character_count"] for row in response_rows
        ),
        "description_token_proxy_count": numeric_summary(
            row["description_token_proxy_count"] for row in response_rows
        ),
        "batch_latency_s": numeric_summary(row["latency_s"] for row in batch_rows),
        "latency_ms_per_mask": numeric_summary(
            row["latency_ms_per_mask"] for row in batch_rows
        ),
        "correction_receipt_count": len(receipts),
        "correction_applied_count": len(applied_rows),
        "final_named_eligible_entity_count": sum(
            normalize_label(entity_names[entity]).casefold() != "unknown"
            for entity in unique_eligible
        ),
        "sqlite_integrity_check": export["sqlite_integrity_check"],
        "database_sha256": export["database_sha256"],
        "formal_name_accuracy": None,
        "formal_description_correctness": None,
        "evaluation_basis": "exact engineering counts + GT-free consistency proxies",
    }
    write_json(cell_root / "SUMMARY.json", summary)
    write_json(
        cell_root / "duplicate_entity_responses.json",
        {
            "duplicate_entities": duplicate_entities,
            "conflicting_duplicate_entities_proxy": conflicting_duplicate_entities,
        },
    )
    return summary


def final_label_rows(output: Path, summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        cell_root = (
            output
            / "cells"
            / f"obs_{summary['threshold_observations']:02d}"
            / f"seed_{summary['seed']}"
        )
        responses = read_jsonl(cell_root / "responses.jsonl")
        entities = json.loads((cell_root / "final_entities.json").read_text())
        eligible = {row["entity_id"] for row in responses}
        for entity in entities:
            if entity["entity_id"] not in eligible:
                continue
            entity_responses = [
                row for row in responses if row["entity_id"] == entity["entity_id"]
            ]
            rows.append(
                {
                    "schema": "daaam.g1_no_gt_e14_final_label.v1",
                    "threshold_observations": summary["threshold_observations"],
                    "seed": summary["seed"],
                    "entity_id": entity["entity_id"],
                    "entity_ordinal": entity_responses[0]["entity_ordinal"],
                    "response_count": len(entity_responses),
                    "final_label": entity["canonical_name"],
                    "final_label_normalized": normalize_label(
                        entity["canonical_name"]
                    ).casefold(),
                    "response_labels_json": json.dumps(
                        [row["description"] for row in entity_responses],
                        ensure_ascii=False,
                    ),
                }
            )
    return rows


def consistency_analysis(
    output: Path,
    summaries: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    thresholds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = final_label_rows(output, summaries)
    write_jsonl(output / "tables/final_labels.jsonl", labels)
    write_csv(output / "tables/final_labels.csv", labels)
    lookup = {
        (row["threshold_observations"], row["seed"], row["entity_id"]): row
        for row in labels
    }
    seed_rows = []
    for threshold in thresholds:
        entity_sets = [
            {
                row["entity_id"]
                for row in labels
                if row["threshold_observations"] == threshold
                and row["seed"] == seed
            }
            for seed in seeds
        ]
        shared = set.intersection(*entity_sets) if entity_sets else set()
        for entity_id in sorted(shared):
            selected = [lookup[(threshold, seed, entity_id)] for seed in seeds]
            pair_scores = [
                jaccard(selected[i]["final_label"], selected[j]["final_label"])
                for i in range(len(selected))
                for j in range(i + 1, len(selected))
            ]
            seed_rows.append(
                {
                    "schema": "daaam.g1_no_gt_e14_seed_consistency.v1",
                    "threshold_observations": threshold,
                    "entity_id": entity_id,
                    "entity_ordinal": selected[0]["entity_ordinal"],
                    "seed_count": len(selected),
                    "exact_all_seed_agreement": len(
                        {row["final_label_normalized"] for row in selected}
                    )
                    == 1,
                    "mean_pairwise_token_jaccard": (
                        float(np.mean(pair_scores)) if pair_scores else 1.0
                    ),
                    "minimum_pairwise_token_jaccard": (
                        min(pair_scores) if pair_scores else 1.0
                    ),
                    "labels_json": json.dumps(
                        {str(seed): row["final_label"] for seed, row in zip(seeds, selected)},
                        ensure_ascii=False,
                    ),
                    "correctness_label": None,
                }
            )
    threshold_rows = []
    for seed in seeds:
        entity_sets = [
            {
                row["entity_id"]
                for row in labels
                if row["threshold_observations"] == threshold
                and row["seed"] == seed
            }
            for threshold in thresholds
        ]
        shared = set.intersection(*entity_sets) if entity_sets else set()
        for entity_id in sorted(shared):
            selected = [lookup[(threshold, seed, entity_id)] for threshold in thresholds]
            pair_scores = [
                jaccard(selected[i]["final_label"], selected[j]["final_label"])
                for i in range(len(selected))
                for j in range(i + 1, len(selected))
            ]
            threshold_rows.append(
                {
                    "schema": "daaam.g1_no_gt_e14_threshold_consistency.v1",
                    "seed": seed,
                    "entity_id": entity_id,
                    "entity_ordinal": selected[0]["entity_ordinal"],
                    "threshold_count": len(selected),
                    "exact_all_threshold_agreement": len(
                        {row["final_label_normalized"] for row in selected}
                    )
                    == 1,
                    "mean_pairwise_token_jaccard": (
                        float(np.mean(pair_scores)) if pair_scores else 1.0
                    ),
                    "minimum_pairwise_token_jaccard": (
                        min(pair_scores) if pair_scores else 1.0
                    ),
                    "labels_json": json.dumps(
                        {
                            str(threshold): row["final_label"]
                            for threshold, row in zip(thresholds, selected)
                        },
                        ensure_ascii=False,
                    ),
                    "correctness_label": None,
                }
            )
    write_jsonl(output / "tables/seed_consistency.jsonl", seed_rows)
    write_csv(output / "tables/seed_consistency.csv", seed_rows)
    write_jsonl(output / "tables/threshold_consistency.jsonl", threshold_rows)
    write_csv(output / "tables/threshold_consistency.csv", threshold_rows)
    return seed_rows, threshold_rows


def aggregate_summaries(
    summaries: Sequence[dict[str, Any]],
    seed_rows: Sequence[dict[str, Any]],
    threshold_rows: Sequence[dict[str, Any]],
    thresholds: Sequence[int],
) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        cells = [
            row for row in summaries if row["threshold_observations"] == threshold
        ]
        consistency = [
            row
            for row in seed_rows
            if row["threshold_observations"] == threshold
        ]
        rows.append(
            {
                "schema": "daaam.g1_no_gt_e14_threshold_summary.v1",
                "threshold_observations": threshold,
                "seed_count": len(cells),
                "eligible_entity_count": cells[0]["eligible_entity_count"],
                "prompt_record_count": cells[0]["prompt_record_count"],
                "prompt_mask_request_count": cells[0][
                    "prompt_mask_request_count"
                ],
                "same_entity_extra_mask_request_count": cells[0][
                    "same_entity_extra_mask_request_count"
                ],
                "mean_eligible_description_coverage": float(
                    np.mean(
                        [row["eligible_description_coverage"] for row in cells]
                    )
                ),
                "mean_all_entity_description_coverage": float(
                    np.mean([row["all_entity_description_coverage"] for row in cells])
                ),
                "mean_first_description_delay_frames": float(
                    np.mean(
                        [
                            row["first_description_delay_frames"]["mean"]
                            for row in cells
                        ]
                    )
                ),
                "mean_first_description_delay_seconds": float(
                    np.mean(
                        [
                            row["first_description_delay_seconds"]["mean"]
                            for row in cells
                        ]
                    )
                ),
                "mean_dam_batch_calls": float(
                    np.mean([row["dam_batch_call_count"] for row in cells])
                ),
                "mean_total_dam_latency_s": float(
                    np.mean(
                        [
                            row["batch_latency_s"]["mean"]
                            * row["batch_latency_s"]["count"]
                            for row in cells
                        ]
                    )
                ),
                "mean_conflicting_duplicate_entities_proxy": float(
                    np.mean(
                        [
                            row[
                                "conflicting_duplicate_description_entity_count_proxy"
                            ]
                            for row in cells
                        ]
                    )
                ),
                "cross_seed_exact_agreement_fraction_proxy": (
                    float(
                        np.mean(
                            [
                                row["exact_all_seed_agreement"]
                                for row in consistency
                            ]
                        )
                    )
                    if consistency
                    else None
                ),
                "cross_seed_mean_token_jaccard_proxy": (
                    float(
                        np.mean(
                            [
                                row["mean_pairwise_token_jaccard"]
                                for row in consistency
                            ]
                        )
                    )
                    if consistency
                    else None
                ),
                "formal_name_accuracy": None,
                "winner": None,
            }
        )
    return rows


def create_visualizations(
    output: Path,
    threshold_summaries: Sequence[dict[str, Any]],
    final_labels: Sequence[dict[str, Any]],
) -> None:
    labels = [str(row["threshold_observations"]) for row in threshold_summaries]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(x, [row["eligible_entity_count"] for row in threshold_summaries])
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("minimum observations")
    axes[0].set_title("Entities receiving DAM descriptions")
    axes[1].bar(
        x,
        [
            row["mean_first_description_delay_frames"]
            for row in threshold_summaries
        ],
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("minimum observations")
    axes[1].set_title("Mean first-description delay (frames)")
    axes[2].bar(
        x - 0.18,
        [
            row["cross_seed_mean_token_jaccard_proxy"]
            for row in threshold_summaries
        ],
        0.36,
        label="token Jaccard",
    )
    axes[2].bar(
        x + 0.18,
        [
            row["cross_seed_exact_agreement_fraction_proxy"]
            for row in threshold_summaries
        ],
        0.36,
        label="exact",
    )
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("minimum observations")
    axes[2].set_title("Cross-seed consistency (proxy)")
    axes[2].legend()
    figure_root = output / "visualizations"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_root / "01_threshold_tradeoffs.png", dpi=180)
    plt.close(fig)

    available_cells = sorted(
        {
            (int(row["threshold_observations"]), int(row["seed"]))
            for row in final_labels
        },
        key=lambda cell: (
            cell != (5, 0),
            abs(cell[0] - 5),
            cell[1],
        ),
    )
    if not available_cells:
        return
    example_threshold, example_seed = available_cells[0]
    candidates = [
        row
        for row in final_labels
        if row["threshold_observations"] == example_threshold
        and row["seed"] == example_seed
    ]
    candidates.sort(key=lambda row: row["entity_ordinal"])
    selected = candidates[:9]
    tiles = []
    for row in selected:
        response = next(
            item
            for item in read_jsonl(
                output
                / "cells"
                / f"obs_{example_threshold:02d}"
                / f"seed_{example_seed}"
                / "responses.jsonl"
            )
            if item["entity_id"] == row["entity_id"]
        )
        image = cv2.imread(response["crop_preview_path"], cv2.IMREAD_COLOR)
        tile = np.full((280, 420, 3), 245, dtype=np.uint8)
        if image is not None:
            scale = min(400 / image.shape[1], 170 / image.shape[0])
            resized = cv2.resize(
                image,
                (
                    max(1, int(image.shape[1] * scale)),
                    max(1, int(image.shape[0] * scale)),
                ),
            )
            x0 = (420 - resized.shape[1]) // 2
            tile[8 : 8 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        cv2.putText(
            tile,
            (
                f"E{row['entity_ordinal']} "
                f"obs={example_threshold} seed={example_seed}"
            ),
            (8, 198),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        words = row["final_label"].split()
        lines = []
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > 52:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            lines.append(line)
        for index, line in enumerate(lines[:4]):
            cv2.putText(
                tile,
                line,
                (8, 220 + index * 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        tiles.append(tile)
    if tiles:
        while len(tiles) < 9:
            tiles.append(np.full_like(tiles[0], 255))
        require_image(
            figure_root / "02_prompt_response_examples.jpg",
            np.vstack(
                [np.hstack(tiles[index : index + 3]) for index in range(0, 9, 3)]
            ),
        )


def failure_cases(
    output: Path,
    summaries: Sequence[dict[str, Any]],
    seed_rows: Sequence[dict[str, Any]],
    threshold_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        if summary["conflicting_duplicate_description_entity_count_proxy"]:
            rows.append(
                {
                    "schema": "daaam.g1_no_gt_e14_failure_proxy.v1",
                    "failure_signature": "same_entity_duplicate_prompt_description_conflict",
                    "threshold_observations": summary["threshold_observations"],
                    "seed": summary["seed"],
                    "entity_id": None,
                    "details_json": json.dumps(summary, ensure_ascii=False),
                    "correctness_label": None,
                    "requires_human_review": True,
                }
            )
    for row in seed_rows:
        if row["minimum_pairwise_token_jaccard"] < 0.25:
            rows.append(
                {
                    "schema": "daaam.g1_no_gt_e14_failure_proxy.v1",
                    "failure_signature": "low_cross_seed_description_overlap",
                    "threshold_observations": row["threshold_observations"],
                    "seed": None,
                    "entity_id": row["entity_id"],
                    "details_json": json.dumps(row, ensure_ascii=False),
                    "correctness_label": None,
                    "requires_human_review": True,
                }
            )
    for row in threshold_rows:
        if row["minimum_pairwise_token_jaccard"] < 0.25:
            rows.append(
                {
                    "schema": "daaam.g1_no_gt_e14_failure_proxy.v1",
                    "failure_signature": "low_cross_threshold_description_overlap",
                    "threshold_observations": None,
                    "seed": row["seed"],
                    "entity_id": row["entity_id"],
                    "details_json": json.dumps(row, ensure_ascii=False),
                    "correctness_label": None,
                    "requires_human_review": True,
                }
            )
    write_jsonl(output / "failure_cases/failure_cases.jsonl", rows)
    write_csv(output / "failure_cases/failure_cases.csv", rows)
    return rows


def write_report(
    output: Path,
    threshold_summaries: Sequence[dict[str, Any]],
    cell_count: int,
    failures: Sequence[dict[str, Any]],
    *,
    e13_variant: str = E13_VARIANT,
    prompt_policy: str = "legacy",
) -> None:
    table = "\n".join(
        (
            f"| {row['threshold_observations']} | {row['eligible_entity_count']} | "
            f"{row['prompt_mask_request_count']} | "
            f"{row['mean_first_description_delay_frames']:.2f} | "
            f"{row['mean_all_entity_description_coverage']:.1%} | "
            f"{row['mean_dam_batch_calls']:.1f} | "
            f"{row['cross_seed_exact_agreement_fraction_proxy']:.1%} | "
            f"{row['cross_seed_mean_token_jaccard_proxy']:.3f} |"
        )
        for row in threshold_summaries
    )
    executed_thresholds = "/".join(
        str(row["threshold_observations"]) for row in threshold_summaries
    )
    report = f"""# E14 DAM 观察门限诊断（E13-fed、无 GT）

## 结论边界

本实验固定使用 E13 `{e13_variant}`，E14 prompt policy 为
`{prompt_policy}`，比较 MapMemory entity 的真实分割观察门限
{executed_thresholds}，共运行 {cell_count} 个 DAM 单元。选择某个距离
或安全策略不代表 E13 已选出 winner。正式 E14 要求 GT entity
crops/observations；当前没有人工名称 GT，所以名称准确率、描述正确率和真实早期错命名率
均不可得，不能选择 E14 winner。

## 实际原理

门限按 MapMemory entity 累计观察计数，不按单一 BotSort track。
只有来自真实 E11 分割、通过深度资格并成功形成 E13 observation 的记录计数。
`legacy` 让一帧内多个 track 分别计数并保留重复 mask；`unique_safe` 每个 entity
每个分割帧只计一次、只选择一个最大面积 mask，并在发现同帧不同 track 冲突后拒绝
该 entity。这一策略与当前分支修订后的准实时 adapter 契约一致。

每个触发记录以完整左目校正图和独立二值 mask 调用生产
`DAMAgentPanoptic.query_multi_image_multi_mask()`，固定问题
`{QUERY}`、temperature={TEMPERATURE}、top-p={TOP_P}、max tokens={MAX_NEW_TOKENS}。
prompt record 按顺序累计到至少 {MULTI_IMAGE_MIN_MASKS} 个 mask 后执行一次 DAM，
末尾不足的 batch 显式 drain。每个 cell 从 E13 SQLite 新副本开始，DAM response 按生产
operation-id 和 supersession 规则写入 MapMemory；Hydra mesh 绑定留给 E17。

## 结果

| observation 门限 | 有描述资格 entity | DAM mask 请求 | 平均首描述延迟/帧 | 全部 E13 entity 覆盖 | DAM batch calls/seed | 跨 seed 完全一致 | 跨 seed token Jaccard |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

coverage、调用数和延迟是精确工程量；跨 seed/threshold 文本重合只是稳定性代理，
不代表文本正确。failure queue 共 {len(failures)} 条代理复核项，全部保留
`correctness_label=null`。

## 证据

- `prompt_inputs/obs_*/`：触发记录、原 mask 的像素副本、焦点 crop、overlay 与逐项哈希。
- `cells/obs_*/seed_*/responses.*`：逐 mask 原始 DAM 文本、source entity/track/frame。
- `cells/obs_*/seed_*/batches.*`：DAM 调用边界、图像/mask 数、延迟和 GPU 快照。
- `cells/obs_*/seed_*/map_memory.sqlite3`：每个单元独立的原生 MapMemory。
- `cells/obs_*/seed_*/database_export/`：operations、deliveries、versions、audit 等全表导出。
- `tables/final_labels.*`：每个 entity 的最终有效 label 与同实体全部 response。
- `tables/seed_consistency.*`、`threshold_consistency.*`：文本稳定性代理。
- `failure_cases/`：重复 prompt 冲突、低跨 seed/threshold 重合的人工复核队列。
- `visualizations/`：覆盖/延迟/稳定性和 prompt-response 示例。

## 尚不能回答

没有人工 GT 就无法判断较早触发的描述是否正确，也无法判断当前门限未命名的
entity 是否本应被命名。正式筛选必须补充固定 GT entity crops/observations、
人工名称/描述裁决及独立 held-out，再计算名称准确率和真实缺失率。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.reseal_existing:
        if not args.output.exists():
            raise FileNotFoundError(args.output)
        audited = (
            args.output / "INDEPENDENT_AUDIT.json"
        ).is_file() and json.loads(
            (args.output / "INDEPENDENT_AUDIT.json").read_text()
        ).get(
            "passed"
        )
        seal_output(
            args.output,
            "complete_independently_audited"
            if audited
            else "complete_pending_independent_audit",
        )
        return 0
    thresholds = tuple(args.thresholds)
    seeds = tuple(args.seeds)
    if any(value not in THRESHOLDS for value in thresholds):
        raise ValueError("thresholds must be chosen from 3, 5, 8")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = utc_now()
    write_json(
        args.output / "invocation.json",
        {
            "schema": "daaam.g1_no_gt_e14_invocation.v1",
            "started_at": started,
            "argv": sys.argv,
            "python": sys.executable,
            "python_version": sys.version,
            "thresholds": list(thresholds),
            "seeds": list(seeds),
            "maximum_prompt_records": args.maximum_prompt_records,
            "e13_run": str(args.e13_run.resolve()),
            "e13_variant": args.e13_variant,
            "prompt_policy": args.prompt_policy,
            "output": str(args.output.resolve()),
            "environment": {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            },
        },
    )
    write_preregistration(
        args.output,
        args.e13_run,
        args.e13_variant,
        args.prompt_policy,
        thresholds,
        seeds,
        args.maximum_prompt_records,
    )
    source_variant = args.e13_run / "variants" / args.e13_variant
    events = read_jsonl(source_variant / "merge_events.jsonl")
    frames_list = read_jsonl(
        args.e13_run / "input_manifests/e12_frames.jsonl"
    )
    frames = {int(row["frame_index"]): row for row in frames_list}
    records_by_threshold = {}
    for threshold in thresholds:
        records = prompt_records_for_threshold(
            threshold,
            events,
            frames,
            args.maximum_prompt_records,
            args.prompt_policy,
        )
        records_by_threshold[threshold], _ = materialize_prompt_inputs(
            args.output, threshold, records
        )
    startup_log = args.output / "model_startup.log"
    pre_model_gpu = nvidia_snapshot()
    model_started = time.perf_counter()
    with startup_log.open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            agent = DAMAgentPanoptic(
                model_path=str(MODEL_LINK),
                conv_mode="v1",
                prompt_mode="focal_prompt",
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_s = time.perf_counter() - model_started
    write_json(
        args.output / "model_runtime.json",
        {
            "schema": "daaam.g1_no_gt_e14_model_runtime.v1",
            "model_load_s": model_load_s,
            "pre_model_gpu": pre_model_gpu,
            "post_model_gpu": nvidia_snapshot(),
            "model": model_identity(),
        },
    )
    summaries = []
    for threshold in thresholds:
        for seed in seeds:
            summaries.append(
                run_cell(
                    agent,
                    args.output,
                    source_variant / "map_memory.sqlite3",
                    threshold,
                    seed,
                    records_by_threshold[threshold],
                )
            )
    write_json(args.output / "tables/cell_summary.json", summaries)
    write_csv(args.output / "tables/cell_summary.csv", summaries)
    seed_rows, threshold_rows = consistency_analysis(
        args.output, summaries, seeds, thresholds
    )
    threshold_summaries = aggregate_summaries(
        summaries, seed_rows, threshold_rows, thresholds
    )
    write_json(
        args.output / "tables/threshold_summary.json", threshold_summaries
    )
    write_csv(
        args.output / "tables/threshold_summary.csv", threshold_summaries
    )
    final_labels = read_jsonl(args.output / "tables/final_labels.jsonl")
    create_visualizations(args.output, threshold_summaries, final_labels)
    failures = failure_cases(
        args.output, summaries, seed_rows, threshold_rows
    )
    write_json(
        args.output / "SCREENING_RESULT.json",
        {
            "schema": "daaam.g1_no_gt_e14_screening.v1",
            "status": "diagnostic_only_no_winner",
            "winner": None,
            "candidates": list(thresholds),
            "reason": (
                "No reviewed GT entity names/descriptions; coverage and consistency "
                "cannot determine semantic correctness."
            ),
            "formal_metrics": {
                "name_accuracy": None,
                "description_correctness": None,
                "true_early_misnaming_rate": None,
            },
        },
    )
    write_report(
        args.output,
        threshold_summaries,
        len(summaries),
        failures,
        e13_variant=args.e13_variant,
        prompt_policy=args.prompt_policy,
    )
    write_json(
        args.output / "RUN_SUMMARY.json",
        {
            "schema": "daaam.g1_no_gt_e14_run_summary.v1",
            "status": "complete_pending_independent_audit",
            "started_at": started,
            "completed_at": utc_now(),
            "thresholds": list(thresholds),
            "seeds": list(seeds),
            "cell_count": len(summaries),
            "source_e13_variant": args.e13_variant,
            "prompt_policy": args.prompt_policy,
            "threshold_summaries": threshold_summaries,
            "failure_proxy_count": len(failures),
            "formal_claims_permitted": False,
        },
    )
    seal_output(args.output)
    print(f"complete: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
