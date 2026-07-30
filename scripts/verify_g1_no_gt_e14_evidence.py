#!/usr/bin/env python3
"""Independently verify the frozen E14 DAM diagnostic evidence.

This verifier does not call DAM again and does not assess semantic correctness.
It reconstructs the E14 trigger stream directly from the frozen E13 events,
checks every prompt mask against its E11 source, accounts for every model
response and MapMemory correction, recomputes the consistency proxies, and
validates the pre-audit artifact inventory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping

import cv2


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments/g1_20260724_473_573_v1_1"
DEFAULT_RUN = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e14_e13fed_dam_20260729"
)
SESSION_ID = "g1-e13-source-473-573"
THRESHOLDS = (3, 5, 8)
SEEDS = (0, 1, 2)
INVENTORY_EXCLUDES = {
    "artifact_inventory.csv",
    "artifact_inventory.jsonl",
    "inventory_summary.json",
    "COMPLETION.json",
}


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
    )


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", text.casefold()))


def jaccard(left: str, right: str) -> float:
    a, b = tokens(left), tokens(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 1.0


def operation_id(response: Mapping[str, Any]) -> str:
    label = " ".join(str(response["description"]).split())
    material = "|".join(
        [
            "dam:checkpoints/dam/DAM-3B",
            str(response["request_id"]),
            str(response["entity_id"]),
            "0",
            label.casefold(),
        ]
    )
    return hashlib.sha256(material.encode()).hexdigest()


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


def verify_reference(reference: Mapping[str, Any], label: str) -> None:
    path = Path(str(reference["path"]))
    check(path.is_file(), f"missing reference: {label}: {path}")
    check(path.stat().st_size == int(reference["size_bytes"]), f"size: {label}")
    check(sha256_file(path) == reference["sha256"], f"sha256: {label}")


def reconstruct_records(
    threshold: int,
    events: list[dict[str, Any]],
    frames: Mapping[int, dict[str, Any]],
    prompt_policy: str = "legacy",
) -> list[dict[str, Any]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    first_by_entity: dict[str, dict[str, Any]] = {}
    for event in events:
        by_frame[int(event["frame_index"])].append(event)
        first_by_entity.setdefault(str(event["entity_id"]), event)
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
                counts[str(event["entity_id"])] += 1
            selected = [
                event
                for event in current
                if counts[str(event["entity_id"])] >= threshold
                and str(event["entity_id"]) not in prompted
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
                                (
                                    cv2.imread(
                                        str(event["source_mask_path"]),
                                        cv2.IMREAD_GRAYSCALE,
                                    )
                                    > 0
                                ).sum()
                            ),
                            -int(event["track_id"]),
                        ),
                    )
                )
        else:
            raise ValueError(f"unsupported prompt policy: {prompt_policy}")
        if not selected:
            continue
        frame = frames[frame_index]
        entities_by_track = {
            int(event["track_id"]): str(event["entity_id"])
            for event in selected
        }
        material = "|".join(
            [
                SESSION_ID,
                str(frame["sensor_time_ns"]),
                "0",
                *(entities_by_track[key] for key in sorted(entities_by_track)),
            ]
        )
        requests = []
        for event in selected:
            entity_id = str(event["entity_id"])
            first = first_by_entity[entity_id]
            requests.append(
                {
                    "request_index": request_index,
                    "entity_id": entity_id,
                    "entity_ordinal": int(event["entity_ordinal"]),
                    "track_id": int(event["track_id"]),
                    "e11_instance_id": int(event["e11_instance_id"]),
                    "observation_count_at_prompt": counts[entity_id],
                    "first_frame_index": int(first["frame_index"]),
                    "delay_frames": frame_index - int(first["frame_index"]),
                    "delay_seconds": (
                        int(frame["sensor_time_ns"])
                        - int(first["sensor_time_ns"])
                    )
                    / 1.0e9,
                    "source_mask_path": str(event["source_mask_path"]),
                    "source_mask_sha256": str(event["source_mask_sha256"]),
                }
            )
            request_index += 1
        records.append(
            {
                "prompt_record_index": len(records),
                "request_id": hashlib.sha256(material.encode()).hexdigest(),
                "frame_index": frame_index,
                "source_frame_index": int(frame["source_frame_index"]),
                "sensor_time_ns": int(frame["sensor_time_ns"]),
                "rgb_path": str(frame["rgb_path"]),
                "rgb_sha256": str(frame["rgb_sha256"]),
                "requests": requests,
            }
        )
        prompted.update(str(event["entity_id"]) for event in selected)
    return records


def make_expected_batches(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batches = []
    pending = []
    mask_count = 0
    for record in records:
        pending.append(int(record["prompt_record_index"]))
        mask_count += len(record["requests"])
        if mask_count >= 16:
            batches.append(
                {
                    "prompt_record_indices": pending,
                    "image_count": len(pending),
                    "mask_count": mask_count,
                }
            )
            pending = []
            mask_count = 0
    if pending:
        batches.append(
            {
                "prompt_record_indices": pending,
                "image_count": len(pending),
                "mask_count": mask_count,
            }
        )
    return batches


def verify_inventory(root: Path) -> dict[str, Any]:
    recorded = read_jsonl(root / "artifact_inventory.jsonl")
    recorded_by_path = {row["relative_path"]: row for row in recorded}
    actual_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in INVENTORY_EXCLUDES
    )
    check(actual_paths == sorted(recorded_by_path), "inventory path set")
    for relative in actual_paths:
        path = root / relative
        row = recorded_by_path[relative]
        check(path.stat().st_size == int(row["size_bytes"]), f"inventory size: {relative}")
        check(sha256_file(path) == row["sha256"], f"inventory hash: {relative}")
    summary = read_json(root / "inventory_summary.json")
    check(len(recorded) == int(summary["file_count"]), "inventory file count")
    check(
        sum(int(row["size_bytes"]) for row in recorded)
        == int(summary["total_size_bytes"]),
        "inventory bytes",
    )
    check(
        inventory_root(recorded) == summary["inventory_root_sha256"],
        "inventory root",
    )
    return {
        "file_count": len(recorded),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in recorded),
        "inventory_root_sha256": inventory_root(recorded),
    }


def verify_prompt_inputs(
    root: Path,
    threshold: int,
    expected: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_root = root / "prompt_inputs" / f"obs_{threshold:02d}"
    actual = read_jsonl(prompt_root / "prompt_records.jsonl")
    flat = read_jsonl(prompt_root / "prompt_requests.jsonl")
    check(len(actual) == len(expected), f"obs {threshold}: record count")
    flat_index = 0
    for expected_record, actual_record in zip(expected, actual, strict=True):
        for key in (
            "prompt_record_index",
            "request_id",
            "frame_index",
            "source_frame_index",
            "sensor_time_ns",
            "rgb_path",
            "rgb_sha256",
        ):
            check(
                actual_record[key] == expected_record[key],
                f"obs {threshold}: record {key}",
            )
        rgb_path = Path(actual_record["rgb_path"])
        check(sha256_file(rgb_path) == actual_record["rgb_sha256"], "rgb hash")
        check(
            len(actual_record["requests"]) == len(expected_record["requests"]),
            f"obs {threshold}: request count in record",
        )
        for expected_request, actual_request in zip(
            expected_record["requests"],
            actual_record["requests"],
            strict=True,
        ):
            for key in (
                "request_index",
                "entity_id",
                "entity_ordinal",
                "track_id",
                "e11_instance_id",
                "observation_count_at_prompt",
                "first_frame_index",
                "delay_frames",
                "source_mask_path",
                "source_mask_sha256",
            ):
                check(
                    actual_request[key] == expected_request[key],
                    f"obs {threshold}: request {key}",
                )
            check(
                close(actual_request["delay_seconds"], expected_request["delay_seconds"]),
                f"obs {threshold}: delay seconds",
            )
            flat_row = flat[flat_index]
            check(
                flat_row["request_index"] == actual_request["request_index"],
                f"obs {threshold}: flat request order",
            )
            source = Path(actual_request["source_mask_path"])
            materialized = Path(actual_request["materialized_mask_path"])
            check(sha256_file(source) == actual_request["source_mask_sha256"], "source mask hash")
            check(
                sha256_file(materialized)
                == actual_request["materialized_mask_sha256"],
                "materialized mask hash",
            )
            source_mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
            copied_mask = cv2.imread(str(materialized), cv2.IMREAD_GRAYSCALE)
            check(source_mask is not None and copied_mask is not None, "read mask")
            source_binary = source_mask > 0
            copied_binary = copied_mask > 0
            check(
                source_binary.shape == copied_binary.shape
                and bool((source_binary == copied_binary).all()),
                "mask pixel identity",
            )
            ys, xs = source_binary.nonzero()
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            check(bbox == actual_request["bbox_xyxy"], "mask bbox")
            check(
                int(source_binary.sum()) == int(actual_request["mask_area_px"]),
                "mask area",
            )
            for path_key, sha_key in (
                ("crop_preview_path", "crop_preview_sha256"),
            ):
                path = Path(actual_request[path_key])
                check(path.is_file(), path_key)
                check(sha256_file(path) == actual_request[sha_key], sha_key)
            flat_index += 1
        overlay = Path(actual_record["prompt_overlay_path"])
        check(overlay.is_file(), "prompt overlay")
        check(
            sha256_file(overlay) == actual_record["prompt_overlay_sha256"],
            "prompt overlay hash",
        )
    check(flat_index == len(flat), f"obs {threshold}: flat request count")
    unique = {row["entity_id"] for row in flat}
    summary = read_json(prompt_root / "SUMMARY.json")
    check(summary["prompt_record_count"] == len(actual), "prompt summary records")
    check(summary["prompt_mask_request_count"] == len(flat), "prompt summary masks")
    check(summary["unique_eligible_entity_count"] == len(unique), "prompt summary entities")
    check(
        summary["same_entity_extra_mask_request_count"] == len(flat) - len(unique),
        "prompt summary duplicates",
    )
    return {
        "prompt_record_count": len(actual),
        "prompt_mask_request_count": len(flat),
        "eligible_entity_count": len(unique),
        "same_entity_extra_mask_request_count": len(flat) - len(unique),
        "mask_pixel_checks": len(flat),
    }


def verify_cell(
    root: Path,
    threshold: int,
    seed: int,
    expected_records: list[dict[str, Any]],
    expected_entity_count: int = 89,
) -> dict[str, Any]:
    cell = root / "cells" / f"obs_{threshold:02d}" / f"seed_{seed}"
    prompt_requests = read_jsonl(
        root / "prompt_inputs" / f"obs_{threshold:02d}" / "prompt_requests.jsonl"
    )
    responses = read_jsonl(cell / "responses.jsonl")
    batches = read_jsonl(cell / "batches.jsonl")
    receipts = read_jsonl(cell / "correction_receipts.jsonl")
    check(len(responses) == len(prompt_requests), f"cell {threshold}/{seed}: responses")
    check(len(receipts) == len(responses), f"cell {threshold}/{seed}: receipts")
    for response, prompt in zip(responses, prompt_requests, strict=True):
        for key in (
            "request_index",
            "request_id",
            "frame_index",
            "source_frame_index",
            "sensor_time_ns",
            "entity_id",
            "entity_ordinal",
            "track_id",
            "e11_instance_id",
            "observation_count_at_prompt",
            "delay_frames",
            "mask_area_px",
            "source_mask_sha256",
            "materialized_mask_sha256",
            "crop_preview_sha256",
        ):
            check(response[key] == prompt[key], f"cell {threshold}/{seed}: response {key}")
        check(close(response["delay_seconds"], prompt["delay_seconds"]), "response delay")
        text_path = Path(response["response_path"])
        check(text_path.read_text(encoding="utf-8") == response["description"] + "\n", "response text")
        check(sha256_file(text_path) == response["response_sha256"], "response hash")
        check(
            len(response["description"]) == response["description_character_count"],
            "response character count",
        )
        check(
            len(re.findall(r"[A-Za-z0-9]+", response["description"]))
            == response["description_token_proxy_count"],
            "response token count",
        )
    expected_batches = make_expected_batches(expected_records)
    check(len(batches) == len(expected_batches), f"cell {threshold}/{seed}: batch count")
    for index, (actual, expected) in enumerate(
        zip(batches, expected_batches, strict=True)
    ):
        for key in ("prompt_record_indices", "image_count", "mask_count"):
            check(actual[key] == expected[key], f"cell {threshold}/{seed}: batch {key}")
        check(actual["batch_index"] == index, "batch index")
        check(actual["latency_s"] > 0 and actual["latency_ms_per_mask"] > 0, "batch latency")
    with sqlite3.connect(cell / "map_memory.sqlite3") as connection:
        check(connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "sqlite integrity")
        entity_count = connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        operation_rows = [
            dict(zip(("operation_id", "entity_id", "label", "status"), row))
            for row in connection.execute(
                "SELECT operation_id, entity_id, label, status FROM semantic_operations"
            )
        ]
    check(entity_count == expected_entity_count, "cell entity count")
    eligible = {row["entity_id"] for row in responses}
    expected_operations = {
        operation_id(row): row
        for row in responses
        if row["description"].strip()
        and row["description"].strip().casefold() != "unknown"
    }
    actual_operations = {row["operation_id"]: row for row in operation_rows}
    check(set(actual_operations) == set(expected_operations), "operation id set")
    statuses: dict[str, int] = defaultdict(int)
    for operation in operation_rows:
        statuses[operation["status"]] += 1
        expected = expected_operations[operation["operation_id"]]
        check(operation["entity_id"] == expected["entity_id"], "operation entity")
        check(operation["label"] == expected["description"], "operation label")
    check(statuses.get("applied", 0) == len(eligible), "one applied operation per entity")
    check(
        statuses.get("superseded", 0) == len(expected_operations) - len(eligible),
        "superseded operation count",
    )
    final_entities = read_json(cell / "final_entities.json")
    final_by_id = {row["entity_id"]: row for row in final_entities}
    applied_by_entity = {
        row["entity_id"]: row["label"]
        for row in operation_rows
        if row["status"] == "applied"
    }
    check(
        {
            entity_id: final_by_id[entity_id]["canonical_name"]
            for entity_id in eligible
        }
        == applied_by_entity,
        "final entity labels",
    )
    db_summary = read_json(cell / "database_export" / "SUMMARY.json")
    check(db_summary["sqlite_integrity_check"] == "ok", "export integrity")
    check(
        sha256_file(cell / "map_memory.sqlite3") == db_summary["database_sha256"],
        "export database hash",
    )
    summary = read_json(cell / "SUMMARY.json")
    check(summary["response_count"] == len(responses), "cell response summary")
    check(summary["eligible_entity_count"] == len(eligible), "cell entity summary")
    check(summary["dam_batch_call_count"] == len(batches), "cell batch summary")
    check(
        close(summary["eligible_description_coverage"], 1.0),
        "eligible coverage",
    )
    check(summary["formal_name_accuracy"] is None, "formal accuracy null")
    return {
        "threshold_observations": threshold,
        "seed": seed,
        "response_count": len(responses),
        "eligible_entity_count": len(eligible),
        "batch_count": len(batches),
        "total_latency_s": sum(float(row["latency_s"]) for row in batches),
        "operation_status_counts": dict(statuses),
    }


def verify_consistency(
    root: Path,
    seeds: tuple[int, ...] = SEEDS,
    thresholds: tuple[int, ...] = THRESHOLDS,
) -> dict[str, int]:
    final_labels = read_jsonl(root / "tables/final_labels.jsonl")
    lookup = {
        (row["threshold_observations"], row["seed"], row["entity_id"]): row
        for row in final_labels
    }
    seed_rows = read_jsonl(root / "tables/seed_consistency.jsonl")
    for row in seed_rows:
        labels = [
            lookup[(row["threshold_observations"], seed, row["entity_id"])][
                "final_label"
            ]
            for seed in seeds
        ]
        scores = [
            jaccard(labels[i], labels[j])
            for i in range(len(labels))
            for j in range(i + 1, len(labels))
        ] or [1.0]
        check(
            row["exact_all_seed_agreement"]
            == (len({label.casefold() for label in labels}) == 1),
            "seed exact agreement",
        )
        check(
            close(
                row["mean_pairwise_token_jaccard"],
                sum(scores) / len(scores),
            ),
            "seed jaccard mean",
        )
        check(
            close(row["minimum_pairwise_token_jaccard"], min(scores)),
            "seed jaccard min",
        )
        check(row["correctness_label"] is None, "seed correctness null")
    threshold_rows = read_jsonl(root / "tables/threshold_consistency.jsonl")
    for row in threshold_rows:
        labels = [
            lookup[(threshold, row["seed"], row["entity_id"])]["final_label"]
            for threshold in thresholds
        ]
        scores = [
            jaccard(labels[i], labels[j])
            for i in range(len(labels))
            for j in range(i + 1, len(labels))
        ] or [1.0]
        check(
            row["exact_all_threshold_agreement"]
            == (len({label.casefold() for label in labels}) == 1),
            "threshold exact agreement",
        )
        check(
            close(
                row["mean_pairwise_token_jaccard"],
                sum(scores) / len(scores),
            ),
            "threshold jaccard mean",
        )
        check(
            close(row["minimum_pairwise_token_jaccard"], min(scores)),
            "threshold jaccard min",
        )
        check(row["correctness_label"] is None, "threshold correctness null")
    return {
        "final_label_rows": len(final_labels),
        "seed_consistency_rows": len(seed_rows),
        "threshold_consistency_rows": len(threshold_rows),
    }


def main() -> int:
    args = parse_args()
    root = args.run.resolve()
    check(root.is_dir(), f"missing E14 run: {root}")
    check(not (root / "INDEPENDENT_AUDIT.json").exists(), "audit already exists")
    prereg = read_json(root / "PRE_REGISTRATION.json")
    thresholds = tuple(int(value) for value in prereg["executed_thresholds"])
    seeds = tuple(int(value) for value in prereg["seeds"])
    check(bool(thresholds), "at least one threshold")
    check(bool(seeds), "at least one seed")
    check(prereg["smoke_truncation"] is None, "formal run is not truncated")
    prompt_policy = str(prereg["prompt_contract"].get("policy", "legacy"))
    check(prompt_policy in {"legacy", "unique_safe"}, "prompt policy")
    reference_count = 0
    for section in ("fixed_upstream", "sources"):
        for key, value in prereg[section].items():
            if isinstance(value, dict) and {"path", "size_bytes", "sha256"} <= set(value):
                verify_reference(value, f"{section}.{key}")
                reference_count += 1
    model = prereg["dam_contract"]["model"]
    for key in (
        "historical_model_inventory",
        "historical_model_inventory_summary",
    ):
        verify_reference(model[key], f"dam_contract.model.{key}")
        reference_count += 1
    inventory = verify_inventory(root)
    source_variant = Path(
        prereg["fixed_upstream"]["e13_merge_events"]["path"]
    ).parent
    e13 = source_variant.parents[1]
    events = read_jsonl(source_variant / "merge_events.jsonl")
    frame_rows = read_jsonl(e13 / "input_manifests/e12_frames.jsonl")
    frames = {int(row["frame_index"]): row for row in frame_rows}
    with sqlite3.connect(source_variant / "map_memory.sqlite3") as connection:
        expected_entity_count = int(
            connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        )
    prompt_results = []
    cell_results = []
    reconstructed = {}
    for threshold in thresholds:
        expected = reconstruct_records(
            threshold,
            events,
            frames,
            prompt_policy,
        )
        reconstructed[threshold] = expected
        prompt_results.append(verify_prompt_inputs(root, threshold, expected))
        for seed in seeds:
            cell_results.append(
                verify_cell(
                    root,
                    threshold,
                    seed,
                    expected,
                    expected_entity_count,
                )
            )
    consistency = verify_consistency(root, seeds, thresholds)
    screening = read_json(root / "SCREENING_RESULT.json")
    check(screening["winner"] is None, "winner must remain null")
    check(
        screening["formal_metrics"]
        == {
            "name_accuracy": None,
            "description_correctness": None,
            "true_early_misnaming_rate": None,
        },
        "formal metrics remain unavailable",
    )
    checks = {
        "pre_registered_reference_hashes": reference_count,
        "inventory": inventory,
        "prompt_inputs": prompt_results,
        "cells": cell_results,
        "consistency": consistency,
        "response_count_total": sum(row["response_count"] for row in cell_results),
        "mask_pixel_checks": sum(row["mask_pixel_checks"] for row in prompt_results),
        "sqlite_integrity_checks": len(cell_results),
    }
    audit = {
        "schema": "daaam.g1_no_gt_e14_independent_audit.v1",
        "audited_at": utc_now(),
        "auditor": "independent deterministic verifier; no DAM re-inference",
        "verifier": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "passed": True,
        "result": "PASS",
        "checks": checks,
        "semantic_correctness_audited": False,
        "semantic_correctness_reason": (
            "No reviewed human GT names/descriptions; the audit proves evidence "
            "identity, accounting, proxy recomputation, and database application only."
        ),
        "permitted_claims": [
            "exact E13-fed trigger, mask, response, batch, latency, and correction counts",
            "recomputed GT-free text consistency proxies",
        ],
        "forbidden_claims": [
            "name accuracy",
            "description correctness",
            "best observation threshold",
        ],
    }
    write_json(root / "INDEPENDENT_AUDIT.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
