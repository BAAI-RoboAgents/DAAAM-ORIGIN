#!/usr/bin/env python3
"""Independently verify E11-fed E12 tracking artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    return parser.parse_args()


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
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def check_inventory(run: Path) -> dict[str, Any]:
    errors: list[str] = []
    rows = read_jsonl(run / "artifact_inventory.jsonl")
    declared = json.loads(
        (run / "inventory_summary.json").read_text(encoding="utf-8")
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for row in rows:
        relative = str(row["relative_path"])
        path = run / relative
        if not path.is_file():
            errors.append(f"missing inventory file: {relative}")
            continue
        size = path.stat().st_size
        observed_hash = sha256_file(path)
        if size != int(row["size_bytes"]):
            errors.append(f"size mismatch: {relative}")
        if observed_hash != row["sha256"]:
            errors.append(f"hash mismatch: {relative}")
        total_bytes += size
        digest.update(
            f"{relative}\0{size}\0{observed_hash}\n".encode("utf-8")
        )
    observed_root = digest.hexdigest()
    if len(rows) != int(declared["file_count"]):
        errors.append("inventory file count mismatch")
    if total_bytes != int(declared["total_bytes"]):
        errors.append("inventory byte count mismatch")
    if observed_root != declared["manifest_root_sha256"]:
        errors.append("inventory root mismatch")
    return {
        "passed": not errors,
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "observed_root_sha256": observed_root,
        "declared_root_sha256": declared["manifest_root_sha256"],
        "errors": errors,
    }


def check_upstream(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[str] = []
    prereg = json.loads(
        (run / "PRE_REGISTRATION.json").read_text(encoding="utf-8")
    )
    controlled = prereg["controlled_input"]
    e11 = Path(controlled["e11_run"])
    e11_inventory_path = e11 / "inventory_summary.json"
    if sha256_file(e11_inventory_path) != controlled[
        "e11_inventory_summary_sha256"
    ]:
        errors.append("E11 inventory summary hash mismatch")
    e11_inventory = json.loads(
        e11_inventory_path.read_text(encoding="utf-8")
    )
    if (
        e11_inventory["manifest_root_sha256"]
        != controlled["e11_manifest_root_sha256"]
    ):
        errors.append("E11 manifest root mismatch")
    frames = read_jsonl(run / "input_frames.jsonl")
    if len(frames) != int(controlled["frame_count"]):
        errors.append("input frame count mismatch")
    source_indices = [int(row["source_frame_index"]) for row in frames]
    if source_indices != list(range(473, 574)):
        errors.append("source range is not exactly 473..573")
    for row in frames:
        for path_key, hash_key in (
            ("rgb_path", "rgb_sha256"),
            ("selection_path", "selection_sha256"),
            ("raw_frame_path", "raw_frame_sha256"),
        ):
            path = Path(row[path_key])
            if not path.is_file():
                errors.append(f"missing upstream file: {path}")
            elif sha256_file(path) != row[hash_key]:
                errors.append(f"upstream hash mismatch: {path}")
    return (
        {
            "passed": not errors,
            "e11_run": str(e11),
            "e11_manifest_root_sha256": e11_inventory[
                "manifest_root_sha256"
            ],
            "frame_count": len(frames),
            "source_frame_range": [
                source_indices[0],
                source_indices[-1],
            ],
            "errors": errors,
        },
        frames,
    )


def compare_box(first: list[float], second: list[float]) -> bool:
    return bool(
        np.allclose(
            np.asarray(first, dtype=np.float64),
            np.asarray(second, dtype=np.float64),
            atol=1e-4,
            rtol=0.0,
        )
    )


def check_tracking(
    run: Path, frames: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    errors: list[str] = []
    prereg = json.loads(
        (run / "PRE_REGISTRATION.json").read_text(encoding="utf-8")
    )
    variants = [
        str(row["variant_id"]) for row in prereg["variants"]
    ]
    observations_by_variant: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in variants
    }
    frame_counts: dict[str, int] = defaultdict(int)
    map_count = 0
    overlay_count = 0
    reconstructed_maps = 0
    for frame in frames:
        frame_index = int(frame["frame_index"])
        selection = json.loads(
            Path(frame["selection_path"]).read_text(encoding="utf-8")
        )
        raw_frame = json.loads(
            Path(frame["raw_frame_path"]).read_text(encoding="utf-8")
        )
        kept_ids = [int(value) for value in selection["kept_instance_ids"]]
        by_id = {
            int(instance["instance_id"]): instance
            for instance in raw_frame["instances"]
        }
        instances = [by_id[value] for value in kept_ids]
        records: dict[str, dict[str, Any]] = {}
        needed_masks: set[int] = set()
        for variant in variants:
            path = (
                run
                / "variants"
                / variant
                / "frames"
                / f"{frame_index:08d}"
                / "frame.json"
            )
            if not path.is_file():
                errors.append(f"missing E12 frame: {path}")
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            records[variant] = record
            frame_counts[variant] += 1
            if int(record["source_frame_index"]) != int(
                frame["source_frame_index"]
            ):
                errors.append(f"source mismatch: {variant}/{frame_index}")
            if record["input_e11_instance_ids"] != kept_ids:
                errors.append(
                    f"E11 selected IDs mismatch: {variant}/{frame_index}"
                )
            observations = list(record["track_observations"])
            observations_by_variant[variant].extend(observations)
            local_indices = [
                int(row["detection_local_index"]) for row in observations
            ]
            if len(local_indices) != len(set(local_indices)):
                errors.append(
                    f"duplicate assigned detection: {variant}/{frame_index}"
                )
            assigned = sorted(local_indices)
            unassigned = sorted(
                int(value)
                for value in record["unassigned_detection_local_indices"]
            )
            if assigned != sorted(
                int(value)
                for value in record["assigned_detection_local_indices"]
            ):
                errors.append(
                    f"assigned list mismatch: {variant}/{frame_index}"
                )
            if sorted(assigned + unassigned) != list(range(len(instances))):
                errors.append(
                    f"detection partition mismatch: {variant}/{frame_index}"
                )
            if len(observations) != int(record["tracked_observation_count"]):
                errors.append(
                    f"tracked count mismatch: {variant}/{frame_index}"
                )
            for observation in observations:
                local_index = int(observation["detection_local_index"])
                if not 0 <= local_index < len(instances):
                    errors.append(
                        f"local index outside E11 input: {variant}/{frame_index}"
                    )
                    continue
                instance = instances[local_index]
                if int(observation["e11_instance_id"]) != int(
                    instance["instance_id"]
                ):
                    errors.append(
                        f"instance linkage mismatch: {variant}/{frame_index}"
                    )
                if observation["source_mask_sha256"] != instance["mask_sha256"]:
                    errors.append(
                        f"mask hash metadata mismatch: {variant}/{frame_index}"
                    )
                if sha256_file(Path(observation["source_mask_path"])) != instance[
                    "mask_sha256"
                ]:
                    errors.append(
                        f"source mask content mismatch: {variant}/{frame_index}"
                    )
                if abs(
                    float(observation["model_confidence"])
                    - float(instance["model_confidence"])
                ) > 1e-8:
                    errors.append(
                        f"confidence mismatch: {variant}/{frame_index}"
                    )
                if int(observation["e11_area_px"]) != int(instance["area_px"]):
                    errors.append(f"area mismatch: {variant}/{frame_index}")
                if not compare_box(
                    observation["e11_box_xyxy"], instance["box_xyxy"]
                ):
                    errors.append(f"box mismatch: {variant}/{frame_index}")
                needed_masks.add(local_index)

        mask_cache: dict[int, np.ndarray] = {}
        for local_index in needed_masks:
            mask = cv2.imread(
                str(instances[local_index]["mask_path"]),
                cv2.IMREAD_GRAYSCALE,
            )
            if mask is None:
                errors.append(
                    f"cannot decode E11 mask: {frame_index}/{local_index}"
                )
                continue
            mask_cache[local_index] = mask > 0
        for variant, record in records.items():
            frame_root = (
                run
                / "variants"
                / variant
                / "frames"
                / f"{frame_index:08d}"
            )
            id_map = cv2.imread(
                str(frame_root / "track_id_map.png"),
                cv2.IMREAD_UNCHANGED,
            )
            if (
                id_map is None
                or id_map.dtype != np.uint16
                or id_map.shape != (960, 1280)
            ):
                errors.append(
                    f"invalid exact ID map: {variant}/{frame_index}"
                )
                continue
            map_count += 1
            reconstructed = np.zeros((960, 1280), dtype=np.uint16)
            ordered = sorted(
                record["track_observations"],
                key=lambda row: (
                    -float(row["model_confidence"]),
                    int(row["track_id"]),
                ),
            )
            for observation in ordered:
                local_index = int(observation["detection_local_index"])
                if local_index not in mask_cache:
                    continue
                mask = mask_cache[local_index]
                reconstructed[
                    mask & (reconstructed == 0)
                ] = int(observation["track_id"])
            if not np.array_equal(id_map, reconstructed):
                errors.append(
                    f"track ID map reconstruction mismatch: {variant}/{frame_index}"
                )
            else:
                reconstructed_maps += 1
            overlay = cv2.imread(
                str(frame_root / "track_overlay.png"), cv2.IMREAD_COLOR
            )
            if overlay is None or overlay.shape[:2] != (960, 1280):
                errors.append(
                    f"invalid overlay: {variant}/{frame_index}"
                )
            else:
                overlay_count += 1
    return (
        {
            "passed": not errors,
            "variant_count": len(variants),
            "variant_frame_counts": dict(frame_counts),
            "frame_record_count": sum(frame_counts.values()),
            "track_observation_count": sum(
                len(rows) for rows in observations_by_variant.values()
            ),
            "exact_uint16_track_id_map_count": map_count,
            "exact_map_reconstruction_pass_count": reconstructed_maps,
            "overlay_count": overlay_count,
            "error_count": len(errors),
            "errors": errors[:100],
        },
        observations_by_variant,
    )


def check_lifecycles(
    run: Path, observations_by_variant: Mapping[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    errors: list[str] = []
    lifecycle_count = 0
    variant_counts: dict[str, int] = {}
    summaries = {
        row["variant_id"]: row
        for row in json.loads(
            (run / "tables/variant_summary.json").read_text(encoding="utf-8")
        )
    }
    for variant, observations in observations_by_variant.items():
        lifecycles = json.loads(
            (
                run / "variants" / variant / "track_lifecycles.json"
            ).read_text(encoding="utf-8")
        )
        variant_counts[variant] = len(lifecycles)
        lifecycle_count += len(lifecycles)
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for observation in observations:
            grouped[int(observation["track_id"])].append(observation)
        if set(grouped) != {
            int(row["track_id"]) for row in lifecycles
        }:
            errors.append(f"lifecycle track-ID set mismatch: {variant}")
        for lifecycle in lifecycles:
            track_id = int(lifecycle["track_id"])
            rows = sorted(
                grouped[track_id], key=lambda row: int(row["frame_index"])
            )
            frame_indices = [int(row["frame_index"]) for row in rows]
            if frame_indices != lifecycle["observation_frame_indices"]:
                errors.append(
                    f"lifecycle observation frames mismatch: {variant}/{track_id}"
                )
            if len(rows) != int(lifecycle["observation_count"]):
                errors.append(
                    f"lifecycle observation count mismatch: {variant}/{track_id}"
                )
            expected_lifespan = frame_indices[-1] - frame_indices[0] + 1
            if expected_lifespan != int(lifecycle["lifespan_frames"]):
                errors.append(
                    f"lifecycle lifespan mismatch: {variant}/{track_id}"
                )
        summary = summaries[variant]
        if int(summary["unique_track_count"]) != len(lifecycles):
            errors.append(f"summary unique-track mismatch: {variant}")
        if int(summary["tracked_observation_count"]) != len(observations):
            errors.append(f"summary observation mismatch: {variant}")
        if summary["formal_metrics"] != {
            "HOTA": None,
            "IDF1": None,
            "real_id_switches": None,
            "real_fragmentation": None,
        }:
            errors.append(f"formal metric contract mismatch: {variant}")
        if summary["correctness_winner"] is not None:
            errors.append(f"unexpected correctness winner: {variant}")
    screening = json.loads(
        (run / "SCREENING_RESULT.json").read_text(encoding="utf-8")
    )
    if screening["correctness_winner"] is not None:
        errors.append("screening result contains a correctness winner")
    return {
        "passed": not errors,
        "lifecycle_count": lifecycle_count,
        "variant_lifecycle_counts": variant_counts,
        "correctness_winner": screening["correctness_winner"],
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    inventory = check_inventory(run)
    upstream, frames = check_upstream(run)
    tracking, observations = check_tracking(run, frames)
    lifecycles = check_lifecycles(run, observations)
    passed = all(
        section["passed"]
        for section in (inventory, upstream, tracking, lifecycles)
    )
    audit = {
        "schema": "daaam.g1_no_gt_e12_independent_audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "run": str(run),
        "verifier": str(Path(__file__).resolve()),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.version,
        "checks": {
            "inventory": inventory,
            "upstream_e11": upstream,
            "tracking_records_and_pixel_maps": tracking,
            "lifecycles_and_metric_contract": lifecycles,
        },
        "self_inclusion_contract": (
            "This audit verifies the inventory that existed before the audit "
            "file itself. A subsequent reseal may include this audit."
        ),
    }
    write_json(run / "INDEPENDENT_AUDIT.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
