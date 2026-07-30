#!/usr/bin/env python3
"""Independently verify one GT-free E11 FastSAM evidence bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print verification only; do not create INDEPENDENT_AUDIT.json.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def aggregate(
    masks: Iterable[np.ndarray], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(shape, dtype=np.uint16)
    boundary = np.zeros(shape, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for mask in masks:
        mask_u8 = np.asarray(mask > 0, dtype=np.uint8)
        counts += mask_u8.astype(np.uint16)
        eroded = cv2.erode(mask_u8, kernel)
        boundary[(mask_u8 - eroded) > 0] = 255
    union = np.where(counts > 0, 255, 0).astype(np.uint8)
    return union, counts, boundary


def verify_inventory(run: Path) -> dict[str, Any]:
    rows = load_jsonl(run / "artifact_inventory.jsonl")
    root_digest = hashlib.sha256()
    total_bytes = 0
    errors: list[str] = []
    for row in rows:
        path = run / row["relative_path"]
        if not path.is_file():
            errors.append(f"missing inventory path: {row['relative_path']}")
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(row["size_bytes"]):
            errors.append(f"size mismatch: {row['relative_path']}")
        if digest != row["sha256"]:
            errors.append(f"hash mismatch: {row['relative_path']}")
        total_bytes += size
        root_digest.update(
            f"{row['relative_path']}\0{size}\0{digest}\n".encode("utf-8")
        )
    summary = json.loads((run / "inventory_summary.json").read_text())
    observed_root = root_digest.hexdigest()
    if len(rows) != int(summary["file_count"]):
        errors.append("inventory file_count mismatch")
    if total_bytes != int(summary["total_bytes"]):
        errors.append("inventory total_bytes mismatch")
    if observed_root != summary["manifest_root_sha256"]:
        errors.append("inventory root mismatch")
    return {
        "passed": not errors,
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "observed_root_sha256": observed_root,
        "declared_root_sha256": summary["manifest_root_sha256"],
        "errors": errors,
    }


def verify_sources(run: Path) -> dict[str, Any]:
    rows = load_jsonl(run / "source_frames.jsonl")
    errors: list[str] = []
    for row in rows:
        path = Path(row["rgb_path"])
        if not path.is_file():
            errors.append(f"missing source RGB: {path}")
            continue
        if path.stat().st_size != int(row["rgb_size_bytes"]):
            errors.append(f"source size mismatch: {path}")
        if sha256_file(path) != row["rgb_sha256"]:
            errors.append(f"source hash mismatch: {path}")
    indices = [int(row["source_frame_index"]) for row in rows]
    if indices != list(range(473, 574)):
        errors.append("source frame indices are not exactly 473..573")
    return {
        "passed": not errors,
        "frame_count": len(rows),
        "source_frame_range": [min(indices), max(indices)] if indices else None,
        "errors": errors,
    }


def verify_raw_instances(run: Path) -> dict[str, Any]:
    frame_records = sorted(run.glob("raw_profiles/*/frames/*/frame.json"))
    errors: list[str] = []
    instance_count = 0
    binary_mask_count = 0
    binary_boundary_count = 0
    profile_frame_counts: dict[str, int] = {}
    for frame_path in frame_records:
        frame = json.loads(frame_path.read_text())
        profile = str(frame["profile_id"])
        profile_frame_counts[profile] = profile_frame_counts.get(profile, 0) + 1
        instances = list(frame["instances"])
        if len(instances) != int(frame["raw_instance_count"]):
            errors.append(f"raw count mismatch: {frame_path}")
        instance_count += len(instances)
        threshold = float(frame["confidence_threshold"])
        for instance in instances:
            mask_path = Path(instance["mask_path"])
            boundary_path = Path(instance["boundary_path"])
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            boundary = cv2.imread(str(boundary_path), cv2.IMREAD_UNCHANGED)
            if mask is None or boundary is None:
                errors.append(
                    f"unreadable instance {profile}/"
                    f"{frame['frame_index']}/{instance['instance_id']}"
                )
                continue
            if mask.shape != (960, 1280):
                errors.append(f"mask shape mismatch: {mask_path}")
            if boundary.shape != (960, 1280):
                errors.append(f"boundary shape mismatch: {boundary_path}")
            if not np.isin(np.unique(mask), [0, 255]).all():
                errors.append(f"non-binary mask: {mask_path}")
            else:
                binary_mask_count += 1
            if not np.isin(np.unique(boundary), [0, 255]).all():
                errors.append(f"non-binary boundary: {boundary_path}")
            else:
                binary_boundary_count += 1
            if int(np.count_nonzero(mask)) != int(instance["area_px"]):
                errors.append(f"mask area mismatch: {mask_path}")
            if int(np.count_nonzero(boundary)) != int(
                instance["boundary_pixel_count"]
            ):
                errors.append(f"boundary area mismatch: {boundary_path}")
            if float(instance["model_confidence"]) + 1e-6 < threshold:
                errors.append(f"confidence below threshold: {mask_path}")
            if sha256_file(mask_path) != instance["mask_sha256"]:
                errors.append(f"instance mask hash mismatch: {mask_path}")
            if sha256_file(boundary_path) != instance["boundary_sha256"]:
                errors.append(f"instance boundary hash mismatch: {boundary_path}")
    if len(profile_frame_counts) != 5:
        errors.append("expected five inference profiles")
    if any(value != 101 for value in profile_frame_counts.values()):
        errors.append("each inference profile must cover 101 frames")
    completion = json.loads((run / "COMPLETION.json").read_text())
    if instance_count != int(completion["raw_instance_count"]):
        errors.append("raw instance count differs from completion")
    return {
        "passed": not errors,
        "profile_count": len(profile_frame_counts),
        "profile_frame_counts": profile_frame_counts,
        "raw_frame_record_count": len(frame_records),
        "raw_instance_count": instance_count,
        "binary_mask_count": binary_mask_count,
        "binary_boundary_count": binary_boundary_count,
        "errors": errors[:100],
        "error_count": len(errors),
    }


def verify_cell_selections(run: Path) -> dict[str, Any]:
    selections = sorted(run.glob("cells/*/selected_instances/*.json"))
    errors: list[str] = []
    cell_counts: dict[str, int] = {}
    for path in selections:
        selection = json.loads(path.read_text())
        cell = selection["cell"]
        cell_name = str(cell["cell_id"])
        cell_counts[cell_name] = cell_counts.get(cell_name, 0) + 1
        raw_path = Path(selection["raw_frame_record"])
        raw = json.loads(raw_path.read_text())
        expected = [
            int(instance["instance_id"])
            for instance in raw["instances"]
            if int(instance["area_px"]) >= int(selection["minimum_area_px"])
        ]
        if expected != list(selection["kept_instance_ids"]):
            errors.append(f"area selection mismatch: {path}")
        expected_rejected = [
            int(instance["instance_id"])
            for instance in raw["instances"]
            if int(instance["area_px"]) < int(selection["minimum_area_px"])
        ]
        if expected_rejected != list(selection["rejected_instance_ids"]):
            errors.append(f"area rejection mismatch: {path}")
    if len(cell_counts) != 13:
        errors.append("expected thirteen cells including production baseline")
    if any(value != 101 for value in cell_counts.values()):
        errors.append("each cell must cover 101 frames")
    return {
        "passed": not errors,
        "cell_count": len(cell_counts),
        "cell_frame_counts": cell_counts,
        "selection_record_count": len(selections),
        "errors": errors,
    }


def verify_aggregate_samples(run: Path) -> dict[str, Any]:
    summary = json.loads((run / "tables/cell_summary.json").read_text())
    frame_indices = (0, 50, 100)
    errors: list[str] = []
    samples = 0
    for cell in summary:
        cell_name = str(cell["cell_id"])
        for frame_index in frame_indices:
            selection_path = (
                run
                / "cells"
                / cell_name
                / "selected_instances"
                / f"{frame_index:08d}.json"
            )
            selection = json.loads(selection_path.read_text())
            raw = json.loads(Path(selection["raw_frame_record"]).read_text())
            instances = {int(row["instance_id"]): row for row in raw["instances"]}
            masks = [
                cv2.imread(
                    instances[int(instance_id)]["mask_path"],
                    cv2.IMREAD_UNCHANGED,
                )
                for instance_id in selection["kept_instance_ids"]
            ]
            union, overlap, boundary = aggregate(masks, (960, 1280))
            declared_union = cv2.imread(
                str(
                    run
                    / "cells"
                    / cell_name
                    / "union_masks"
                    / f"{frame_index:08d}.png"
                ),
                cv2.IMREAD_UNCHANGED,
            )
            declared_overlap = cv2.imread(
                str(
                    run
                    / "cells"
                    / cell_name
                    / "overlap_counts"
                    / f"{frame_index:08d}.png"
                ),
                cv2.IMREAD_UNCHANGED,
            )
            declared_boundary = cv2.imread(
                str(
                    run
                    / "cells"
                    / cell_name
                    / "boundary_unions"
                    / f"{frame_index:08d}.png"
                ),
                cv2.IMREAD_UNCHANGED,
            )
            if not np.array_equal(union, declared_union):
                errors.append(f"union reconstruction mismatch: {cell_name}/{frame_index}")
            if not np.array_equal(overlap, declared_overlap):
                errors.append(
                    f"overlap reconstruction mismatch: {cell_name}/{frame_index}"
                )
            if not np.array_equal(boundary, declared_boundary):
                errors.append(
                    f"boundary reconstruction mismatch: {cell_name}/{frame_index}"
                )
            samples += 1
    return {
        "passed": not errors,
        "deterministic_sample_rule": "all 13 cells x frames 0,50,100",
        "sample_count": samples,
        "errors": errors,
    }


def verify_metric_contract(run: Path) -> dict[str, Any]:
    forbidden = {
        "mask_ap",
        "mask_ap50",
        "mask_ap75",
        "boundary_f",
        "small_object_recall",
        "accuracy_winner",
    }
    summaries = json.loads((run / "tables/cell_summary.json").read_text())
    errors: list[str] = []
    for row in summaries:
        intersection = forbidden.intersection(key.lower() for key in row)
        if intersection:
            errors.append(
                f"forbidden GT metric in {row['cell_id']}: {sorted(intersection)}"
            )
        if "proxy/provisional" not in str(row["metric_status"]):
            errors.append(f"missing proxy status: {row['cell_id']}")
    pareto = json.loads(
        (run / "tables/diagnostic_pareto.json").read_text()
    )
    if pareto.get("correctness_winner") is not None:
        errors.append("diagnostic Pareto declares a correctness winner")
    return {
        "passed": not errors,
        "cell_summary_count": len(summaries),
        "correctness_winner": pareto.get("correctness_winner"),
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    checks = {
        "inventory": verify_inventory(run),
        "sources": verify_sources(run),
        "raw_instances": verify_raw_instances(run),
        "cell_selections": verify_cell_selections(run),
        "aggregate_samples": verify_aggregate_samples(run),
        "metric_contract": verify_metric_contract(run),
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    result = {
        "schema": "daaam.g1_no_gt_e11_independent_audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "run": str(run),
        "verifier": str(Path(__file__).resolve()),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.version,
        "checks": checks,
        "self_inclusion_contract": (
            "This audit verifies the inventory that existed before the audit "
            "file itself. A subsequent reseal may include this audit and will "
            "therefore have a different manifest root without changing native evidence."
        ),
    }
    if not args.no_write:
        write_json(run / "INDEPENDENT_AUDIT.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
