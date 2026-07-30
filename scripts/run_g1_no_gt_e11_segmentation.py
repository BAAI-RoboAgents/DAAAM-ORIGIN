#!/usr/bin/env python3
"""Collect auditable E11 FastSAM evidence without claiming GT accuracy.

The collector runs FastSAM on one frozen RGB window, persists the unfiltered
instances produced by every unique (confidence, NMS-IoU) inference profile, and
derives the full confidence x minimum-area x NMS-IoU matrix without re-running
identical inference.  All masks remain at source-image resolution.

This is deliberately a GT-free diagnostic.  It reports coverage, overlap,
boundary complexity, temporal stability, latency, and failure signatures, but
never Mask AP, boundary F-score, small-object recall, or a correctness winner.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    REPOSITORY_ROOT
    / "experiments/g1_20260724_473_573_v1_1/shared_artifacts/"
    "prepared_stereo_473_573_v1_v2"
)
DEFAULT_ENGINE = REPOSITORY_ROOT / "checkpoints/fastsam/FastSAM-x-640x480.engine"
DEFAULT_FASTSAM_CONFIG = REPOSITORY_ROOT / "config/fastsam/fastsam_config.yaml"


def parse_numeric_list(value: str, cast: type) -> list[Any]:
    output: list[Any] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            output.append(cast(item))
    if not output:
        raise argparse.ArgumentTypeError("numeric list cannot be empty")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persist raw FastSAM instances and a GT-free E11 matrix."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument(
        "--fastsam-config", type=Path, default=DEFAULT_FASTSAM_CONFIG
    )
    parser.add_argument("--conf-values", default="0.2,0.4")
    parser.add_argument("--area-values", default="150,300,600")
    parser.add_argument("--iou-values", default="0.4,0.6")
    parser.add_argument("--baseline-conf", type=float, default=0.3)
    parser.add_argument("--baseline-area", type=int, default=300)
    parser.add_argument("--baseline-iou", type=float, default=0.5)
    parser.add_argument("--maximum-frames", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.42,
        help="Alpha for colored instance masks in lossless overlays.",
    )
    parser.add_argument(
        "--reseal-existing",
        action="store_true",
        help=(
            "Do not run inference; remove any circular embedded inventory root "
            "from an existing report and rebuild its manifests/completion."
        ),
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
            stream.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )


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
        for row in rows:
            writer.writerow(row)


def require_write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def numeric_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def profile_id(confidence: float, iou: float) -> str:
    return f"conf_{numeric_token(confidence)}__iou_{numeric_token(iou)}"


def cell_id(confidence: float, area: int, iou: float) -> str:
    return (
        f"conf_{numeric_token(confidence)}__area_{area:04d}"
        f"__iou_{numeric_token(iou)}"
    )


def stable_color(index: int) -> tuple[int, int, int]:
    # OpenCV BGR, intentionally bright and deterministic.
    hue = (index * 47 + 13) % 180
    hsv = np.uint8([[[hue, 210, 245]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def contours_for_mask(
    mask: np.ndarray,
) -> tuple[list[np.ndarray], list[dict[str, Any]], np.ndarray, float]:
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    boundary = np.zeros_like(mask_u8)
    if contours:
        cv2.drawContours(boundary, contours, -1, 255, 1, lineType=cv2.LINE_8)
    hierarchy_rows: list[list[int]] = []
    if hierarchy is not None and len(hierarchy):
        hierarchy_rows = hierarchy[0].astype(int).tolist()
    records: list[dict[str, Any]] = []
    perimeter = 0.0
    for index, contour in enumerate(contours):
        points = contour.reshape(-1, 2).astype(int)
        closed = bool(len(points) >= 3)
        contour_perimeter = float(cv2.arcLength(contour, closed))
        perimeter += contour_perimeter
        records.append(
            {
                "contour_index": index,
                "hierarchy": (
                    hierarchy_rows[index]
                    if index < len(hierarchy_rows)
                    else [-1, -1, -1, -1]
                ),
                "closed": closed,
                "perimeter_px": contour_perimeter,
                "vertices_xy": points.tolist(),
            }
        )
    return contours, records, boundary, perimeter


def draw_overlay(
    rgb_bgr: np.ndarray,
    masks: Sequence[np.ndarray],
    detections: Sequence[Sequence[float]],
    instance_ids: Sequence[int],
    *,
    alpha: float,
    title: str,
) -> np.ndarray:
    overlay = rgb_bgr.copy()
    color_layer = rgb_bgr.copy()
    for instance_id in instance_ids:
        mask = np.asarray(masks[instance_id], dtype=bool)
        color_layer[mask] = stable_color(instance_id)
    overlay = cv2.addWeighted(color_layer, alpha, overlay, 1.0 - alpha, 0.0)
    for instance_id in instance_ids:
        x1, y1, x2, y2, confidence, _ = detections[instance_id]
        color = stable_color(instance_id)
        cv2.rectangle(
            overlay,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"{instance_id}:{confidence:.3f}",
            (int(round(x1)), max(18, int(round(y1)) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )
    banner_height = 34
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], banner_height), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def aggregate_masks(
    masks: Sequence[np.ndarray], instance_ids: Sequence[int], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(shape, dtype=np.uint16)
    boundary_union = np.zeros(shape, dtype=np.uint8)
    for instance_id in instance_ids:
        mask = np.asarray(masks[instance_id], dtype=bool)
        counts += mask.astype(np.uint16)
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        boundary_union[(mask.astype(np.uint8) - eroded) > 0] = 255
    union = np.where(counts > 0, 255, 0).astype(np.uint8)
    return union, counts, boundary_union


def jaccard(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    left_bool = left > 0
    right_bool = right > 0
    union = int(np.count_nonzero(left_bool | right_bool))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(left_bool & right_bool) / union)


def summary(values: Iterable[float | int | None]) -> dict[str, Any]:
    clean = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if not clean.size:
        return {
            "count": 0,
            "mean": None,
            "minimum": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": int(clean.size),
        "mean": float(np.mean(clean)),
        "minimum": float(np.min(clean)),
        "p05": float(np.quantile(clean, 0.05)),
        "p50": float(np.quantile(clean, 0.50)),
        "p95": float(np.quantile(clean, 0.95)),
        "maximum": float(np.max(clean)),
    }


def float_or_nan(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def format_metric(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def nvidia_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def read_frames(dataset: Path, maximum_frames: int | None) -> list[dict[str, Any]]:
    tick_index_path = dataset / "tick_index.json"
    tick_index = json.loads(tick_index_path.read_text(encoding="utf-8"))
    frames: list[dict[str, Any]] = []
    for tick in tick_index.get("frames", []):
        image_path = Path(str(tick["cam0"]))
        if not image_path.is_absolute():
            image_path = dataset / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"missing RGB input: {image_path}")
        frames.append(
            {
                "frame_index": int(tick["idx"]),
                "source_frame_index": int(tick["source_idx"]),
                "sensor_time_ns": int(tick["sensor_time_ns"]),
                "timestamp_s": float(tick["timestamp"]),
                "stereo_delta_ms": float(tick["stereo_delta_ms"]),
                "rgb_path": str(image_path.resolve()),
                "rgb_size_bytes": image_path.stat().st_size,
                "rgb_sha256": sha256_file(image_path),
            }
        )
    if maximum_frames is not None:
        if maximum_frames <= 0:
            raise ValueError("--maximum-frames must be positive")
        frames = frames[:maximum_frames]
    if not frames:
        raise ValueError("no frames found")
    return frames


def add_cell_failure_flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if int(row["kept_instance_count"]) == 0:
        flags.append("E11_EMPTY_AFTER_FILTER_PROXY")
    if (
        int(row["tiny_instance_count"]) >= 5
        and float(row["tiny_instance_fraction"]) >= 0.25
    ):
        flags.append("E11_FRAGMENTATION_PROXY")
    if int(row["giant_instance_count"]) > 0:
        flags.append("E11_GIANT_MASK_PROXY")
    if float(row["overlap_over_union_ratio"]) >= 0.25:
        flags.append("E11_HIGH_OVERLAP_PROXY")
    if float(row["boundary_over_union_ratio"]) >= 0.10:
        flags.append("E11_HIGH_BOUNDARY_COMPLEXITY_PROXY")
    return flags


def pareto_frontier(rows: Sequence[dict[str, Any]]) -> list[str]:
    # More coverage/stability and less fragmentation/overlap/latency.  This is
    # explicitly a diagnostic frontier, not a correctness ranking.
    maximize = ("union_ratio_mean", "temporal_union_iou_mean")
    minimize = (
        "tiny_instance_fraction_mean",
        "overlap_over_union_ratio_mean",
        "inference_latency_p95_ms",
    )
    frontier: list[str] = []
    for candidate in rows:
        if any(candidate.get(key) is None for key in (*maximize, *minimize)):
            continue
        dominated = False
        for other in rows:
            if candidate is other:
                continue
            if any(other.get(key) is None for key in (*maximize, *minimize)):
                continue
            no_worse = all(
                float(other[key]) >= float(candidate[key]) for key in maximize
            ) and all(
                float(other[key]) <= float(candidate[key]) for key in minimize
            )
            strictly_better = any(
                float(other[key]) > float(candidate[key]) + 1e-12
                for key in maximize
            ) or any(
                float(other[key]) < float(candidate[key]) - 1e-12
                for key in minimize
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(str(candidate["cell_id"]))
    return frontier


def inventory_tree(root: Path) -> dict[str, Any]:
    excluded = {
        "artifact_inventory.jsonl",
        "artifact_inventory.csv",
        "inventory_summary.json",
        "COMPLETION.json",
    }
    rows: list[dict[str, Any]] = []
    root_digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        file_hash = sha256_file(path)
        size = path.stat().st_size
        row = {
            "relative_path": relative,
            "size_bytes": size,
            "sha256": file_hash,
        }
        rows.append(row)
        root_digest.update(
            f"{relative}\0{size}\0{file_hash}\n".encode("utf-8")
        )
    write_jsonl(root / "artifact_inventory.jsonl", rows)
    write_csv(root / "artifact_inventory.csv", rows)
    value = {
        "schema": "daaam.g1_no_gt_e11_inventory.v1",
        "created_utc": utc_now(),
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "content_hash_complete": True,
        "manifest_root_sha256": root_digest.hexdigest(),
        "root_definition": "SHA-256 over sorted relative_path\\0size\\0sha256 rows",
        "excluded_self_referential_products": sorted(excluded),
    }
    write_json(root / "inventory_summary.json", value)
    return value


def build_screening_result(output: Path) -> dict[str, Any]:
    """Build paired factor effects and explicit GT-free operating examples."""
    summary_path = output / "tables/cell_summary.json"
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    grid = [
        row for row in summaries if not bool(row["is_production_baseline"])
    ]
    baseline = next(
        row for row in summaries if bool(row["is_production_baseline"])
    )
    metrics = [
        "kept_instance_count_mean",
        "union_ratio_mean",
        "temporal_union_iou_mean",
        "tiny_instance_fraction_mean",
        "giant_frame_fraction",
        "overlap_over_union_ratio_mean",
        "boundary_over_union_ratio_mean",
        "inference_latency_p95_ms",
    ]
    confidences = sorted({float(row["confidence"]) for row in grid})
    areas = sorted({int(row["minimum_area_px"]) for row in grid})
    ious = sorted({float(row["nms_iou"]) for row in grid})
    lookup = {
        (
            float(row["confidence"]),
            int(row["minimum_area_px"]),
            float(row["nms_iou"]),
        ): row
        for row in grid
    }
    effects: list[dict[str, Any]] = []

    def add_effect(
        factor: str,
        low_value: float | int,
        high_value: float | int,
        low: Mapping[str, Any],
        high: Mapping[str, Any],
        fixed: Mapping[str, Any],
    ) -> None:
        record: dict[str, Any] = {
            "factor": factor,
            "from_value": low_value,
            "to_value": high_value,
            "from_cell_id": low["cell_id"],
            "to_cell_id": high["cell_id"],
            "fixed_confidence": fixed.get("confidence"),
            "fixed_minimum_area_px": fixed.get("minimum_area_px"),
            "fixed_nms_iou": fixed.get("nms_iou"),
        }
        for metric in metrics:
            record[f"delta_{metric}"] = (
                float(high[metric]) - float(low[metric])
            )
        effects.append(record)

    for confidence in confidences:
        for iou in ious:
            add_effect(
                "minimum_area_px",
                areas[0],
                areas[-1],
                lookup[(confidence, areas[0], iou)],
                lookup[(confidence, areas[-1], iou)],
                {"confidence": confidence, "nms_iou": iou},
            )
    for area in areas:
        for iou in ious:
            add_effect(
                "confidence",
                confidences[0],
                confidences[-1],
                lookup[(confidences[0], area, iou)],
                lookup[(confidences[-1], area, iou)],
                {"minimum_area_px": area, "nms_iou": iou},
            )
    for confidence in confidences:
        for area in areas:
            add_effect(
                "nms_iou",
                ious[0],
                ious[-1],
                lookup[(confidence, area, ious[0])],
                lookup[(confidence, area, ious[-1])],
                {"confidence": confidence, "minimum_area_px": area},
            )
    write_json(output / "tables/factor_effects.json", effects)
    write_csv(output / "tables/factor_effects.csv", effects)

    area_effects = [
        row for row in effects if row["factor"] == "minimum_area_px"
    ]
    confidence_effects = [
        row for row in effects if row["factor"] == "confidence"
    ]
    iou_effects = [row for row in effects if row["factor"] == "nms_iou"]
    coverage = max(
        grid,
        key=lambda row: (
            float(row["union_ratio_mean"]),
            float(row["temporal_union_iou_mean"]),
        ),
    )
    near_coverage = [
        row
        for row in grid
        if float(row["confidence"]) == float(coverage["confidence"])
        and float(row["nms_iou"]) == float(coverage["nms_iou"])
        and (
            float(coverage["union_ratio_mean"])
            - float(row["union_ratio_mean"])
            <= 0.001
        )
    ]
    practical_coverage = min(
        near_coverage,
        key=lambda row: (
            float(row["tiny_instance_fraction_mean"]),
            float(row["overlap_over_union_ratio_mean"]),
            -int(row["minimum_area_px"]),
        ),
    )
    cleanliness = min(
        grid,
        key=lambda row: (
            float(row["overlap_over_union_ratio_mean"]),
            float(row["tiny_instance_fraction_mean"]),
            float(row["inference_latency_p95_ms"]),
        ),
    )
    existing_result = output / "SCREENING_RESULT.json"
    created_utc = utc_now()
    if existing_result.is_file():
        previous = json.loads(existing_result.read_text(encoding="utf-8"))
        created_utc = str(previous.get("created_utc") or created_utc)

    def cell_extract(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "cell_id": row["cell_id"],
            "confidence": row["confidence"],
            "minimum_area_px": row["minimum_area_px"],
            "nms_iou": row["nms_iou"],
            **{metric: row[metric] for metric in metrics},
        }

    frontier = [
        str(row["cell_id"])
        for row in grid
        if bool(row["diagnostic_pareto_frontier"])
    ]
    result = {
        "schema": "daaam.g1_no_gt_e11_screening_result.v1",
        "created_utc": created_utc,
        "status": "complete / diagnostic_gt_free / no_accuracy_claim",
        "matrix": {
            "confidence": confidences,
            "minimum_area_px": areas,
            "nms_iou": ious,
            "full_factorial_cells": len(grid),
        },
        "correctness_winner": None,
        "correctness_winner_reason": (
            "Reviewed instance GT is unavailable; proxy metrics cannot establish "
            "segmentation accuracy or small-object recall."
        ),
        "diagnostic_pareto": {
            "cell_count": len(frontier),
            "cell_ids": frontier,
            "interpretation": (
                "All listed cells are non-dominated under the preregistered "
                "five proxy objectives; this is not an accuracy ranking."
            ),
        },
        "paired_factor_effects": {
            "definition": "to_value minus from_value with other factors fixed",
            "table_json": "tables/factor_effects.json",
            "table_csv": "tables/factor_effects.csv",
            "minimum_area_px": {
                "comparison": [areas[0], areas[-1]],
                "pair_count": len(area_effects),
                "kept_instances_per_frame_delta_range": [
                    min(
                        row["delta_kept_instance_count_mean"]
                        for row in area_effects
                    ),
                    max(
                        row["delta_kept_instance_count_mean"]
                        for row in area_effects
                    ),
                ],
                "maximum_absolute_union_ratio_delta": max(
                    abs(row["delta_union_ratio_mean"])
                    for row in area_effects
                ),
                "interpretation": (
                    "Raising area removes small masks with negligible union "
                    "coverage change in this window; small-object recall remains "
                    "unknown without GT."
                ),
            },
            "confidence": {
                "comparison": [confidences[0], confidences[-1]],
                "pair_count": len(confidence_effects),
                "mean_deltas": {
                    metric: statistics.fmean(
                        row[f"delta_{metric}"] for row in confidence_effects
                    )
                    for metric in metrics
                },
                "interpretation": (
                    "Higher confidence reduces mask count, coverage, temporal "
                    "union IoU, overlap, and runtime in this diagnostic window."
                ),
            },
            "nms_iou": {
                "comparison": [ious[0], ious[-1]],
                "pair_count": len(iou_effects),
                "mean_deltas": {
                    metric: statistics.fmean(
                        row[f"delta_{metric}"] for row in iou_effects
                    )
                    for metric in metrics
                },
                "interpretation": (
                    "Higher NMS IoU retains more overlapping masks and raises "
                    "coverage/stability proxies at a modest runtime cost."
                ),
            },
        },
        "operating_examples_not_accuracy_winners": {
            "production_baseline": cell_extract(baseline),
            "maximum_coverage_proxy": cell_extract(coverage),
            "near_maximum_coverage_with_area_cleanup": cell_extract(
                practical_coverage
            ),
            "minimum_overlap_proxy": cell_extract(cleanliness),
        },
        "unavailable_without_reviewed_gt": [
            "Mask AP50",
            "Mask AP75",
            "boundary F",
            "small-object recall",
            "correct merge/split labels",
        ],
    }
    write_json(existing_result, result)
    return result


def save_heatmaps(
    output: Path, summaries: Sequence[dict[str, Any]], confs: Sequence[float],
    areas: Sequence[int], ious: Sequence[float]
) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("union_ratio_mean", "mean union coverage"),
        ("temporal_union_iou_mean", "mean adjacent union IoU"),
        ("tiny_instance_fraction_mean", "mean tiny-instance fraction"),
        ("overlap_over_union_ratio_mean", "mean overlap / union"),
    ]
    figure, axes = plt.subplots(
        len(ious), len(metrics), figsize=(17, 4.4 * len(ious)), squeeze=False
    )
    lookup = {
        (
            float(row["confidence"]),
            int(row["minimum_area_px"]),
            float(row["nms_iou"]),
        ): row
        for row in summaries
        if not bool(row["is_production_baseline"])
    }
    for row_index, iou in enumerate(ious):
        for column_index, (metric, title) in enumerate(metrics):
            matrix = np.full((len(confs), len(areas)), np.nan, dtype=np.float64)
            for ci, confidence in enumerate(confs):
                for ai, area in enumerate(areas):
                    row = lookup[(confidence, area, iou)]
                    matrix[ci, ai] = float_or_nan(row[metric])
            axis = axes[row_index, column_index]
            image = axis.imshow(matrix, aspect="auto", cmap="viridis")
            axis.set_xticks(range(len(areas)), [str(value) for value in areas])
            axis.set_yticks(
                range(len(confs)), [f"{value:g}" for value in confs]
            )
            axis.set_xlabel("minimum area (px)")
            axis.set_ylabel("confidence")
            axis.set_title(f"IoU={iou:g} | {title}")
            for ci in range(len(confs)):
                for ai in range(len(areas)):
                    axis.text(
                        ai,
                        ci,
                        f"{matrix[ci, ai]:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        "E11 GT-free parameter diagnostics — coverage/stability are not accuracy",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output / "visualizations/01_parameter_heatmaps.png", dpi=180)
    plt.close(figure)


def save_pareto_plot(
    output: Path, summaries: Sequence[dict[str, Any]], frontier: set[str]
) -> None:
    import matplotlib.pyplot as plt

    grid = [row for row in summaries if not bool(row["is_production_baseline"])]
    figure, axis = plt.subplots(figsize=(10, 7))
    for row in grid:
        is_frontier = str(row["cell_id"]) in frontier
        axis.scatter(
            float(row["union_ratio_mean"]),
            float_or_nan(row["temporal_union_iou_mean"]),
            s=40 + 240 * float(row["overlap_over_union_ratio_mean"]),
            c=("tab:red" if is_frontier else "tab:blue"),
            marker=("D" if is_frontier else "o"),
            alpha=0.82,
        )
        axis.annotate(
            (
                f"c={float(row['confidence']):g},"
                f"a={int(row['minimum_area_px'])},"
                f"i={float(row['nms_iou']):g}"
            ),
            (
                float(row["union_ratio_mean"]),
                float_or_nan(row["temporal_union_iou_mean"]),
            ),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axis.set_xlabel("mean union coverage (maximize only as diagnostic)")
    axis.set_ylabel("mean adjacent union IoU (camera-motion confounded)")
    axis.set_title(
        "E11 diagnostic frontier (red diamonds); marker size = overlap proxy"
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "visualizations/02_diagnostic_pareto.png", dpi=180)
    plt.close(figure)


def save_representative_montage(
    output: Path,
    frames: Sequence[dict[str, Any]],
    cells: Sequence[dict[str, Any]],
) -> None:
    # First/middle/last x baseline and two grid extremes.
    positions = sorted({0, len(frames) // 2, len(frames) - 1})
    baseline = next(row for row in cells if bool(row["is_production_baseline"]))
    grid = [row for row in cells if not bool(row["is_production_baseline"])]
    candidates = [
        baseline,
        min(
            grid,
            key=lambda row: (
                float(row["confidence"]),
                int(row["minimum_area_px"]),
                float(row["nms_iou"]),
            ),
        ),
        max(
            grid,
            key=lambda row: (
                float(row["confidence"]),
                int(row["minimum_area_px"]),
                float(row["nms_iou"]),
            ),
        ),
    ]
    tiles: list[list[np.ndarray]] = []
    target_width = 640
    for position in positions:
        frame_index = int(frames[position]["frame_index"])
        row_tiles: list[np.ndarray] = []
        for cell in candidates:
            path = (
                output
                / "cells"
                / str(cell["cell_id"])
                / "overlays"
                / f"{frame_index:08d}.png"
            )
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            target_height = int(round(image.shape[0] * target_width / image.shape[1]))
            row_tiles.append(
                cv2.resize(
                    image, (target_width, target_height), interpolation=cv2.INTER_AREA
                )
            )
        tiles.append(row_tiles)
    montage = cv2.vconcat([cv2.hconcat(row) for row in tiles])
    require_write_image(
        output / "visualizations/03_representative_overlays.png", montage
    )


def render_report(
    output: Path,
    frames: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    frontier: Sequence[str],
    raw_instance_count: int,
    failure_count: int,
    inventory: Mapping[str, Any],
) -> None:
    baseline = next(row for row in summaries if bool(row["is_production_baseline"]))
    lines = [
        "# E11 FastSAM 无人工 GT 隔离证据",
        "",
        f"- 状态：`complete / diagnostic_gt_free / no_accuracy_claim`",
        f"- 冻结输入：{len(frames)} 帧，source "
        f"{frames[0]['source_frame_index']}–{frames[-1]['source_frame_index']}",
        f"- 原始实例：{raw_instance_count}",
        f"- 参数矩阵：{sum(not bool(row['is_production_baseline']) for row in summaries)} "
        "个全因子单元 + 1 个生产基线",
        f"- 诊断失败标记：{failure_count}",
        "- 证据文件数与总字节数：见 `inventory_summary.json`（避免报告内容"
        "对自身封存字节数产生循环依赖）",
        "- inventory 根哈希：见 `inventory_summary.json`；报告本身在根哈希"
        "覆盖范围内，因此不在报告中内嵌会产生自引用的根值",
        "",
        "> 本实验没有人工实例真值。coverage、边界复杂度、相邻帧 IoU、碎片和巨型",
        "> mask 都是代理诊断，不能解释为 Mask AP、boundary F 或 small recall。",
        "",
        "## 生产基线",
        "",
        "| conf | area | NMS IoU | masks/frame mean | union mean | temporal IoU mean | "
        "overlap mean | inference P95 ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {baseline['confidence']:.3g} | {baseline['minimum_area_px']} | "
            f"{baseline['nms_iou']:.3g} | {baseline['kept_instance_count_mean']:.3f} | "
            f"{format_metric(baseline['union_ratio_mean'])} | "
            f"{format_metric(baseline['temporal_union_iou_mean'])} | "
            f"{format_metric(baseline['overlap_over_union_ratio_mean'])} | "
            f"{format_metric(baseline['inference_latency_p95_ms'], 3)} |"
        ),
        "",
        "## 全因子矩阵",
        "",
        "| cell | masks/frame | union | temporal IoU | tiny frac | giant frame frac | "
        "overlap | P95 ms | Pareto |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    frontier_set = set(frontier)
    for row in summaries:
        if bool(row["is_production_baseline"]):
            continue
        lines.append(
            f"| `{row['cell_id']}` | {row['kept_instance_count_mean']:.3f} | "
            f"{format_metric(row['union_ratio_mean'])} | "
            f"{format_metric(row['temporal_union_iou_mean'])} | "
            f"{format_metric(row['tiny_instance_fraction_mean'])} | "
            f"{format_metric(row['giant_frame_fraction'])} | "
            f"{format_metric(row['overlap_over_union_ratio_mean'])} | "
            f"{format_metric(row['inference_latency_p95_ms'], 3)} | "
            f"{'yes' if row['cell_id'] in frontier_set else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## 证据导航",
            "",
            "- `raw_profiles/`：未做面积过滤的原始 mask、边界、轮廓、置信度和 overlay；",
            "- `cells/`：每个参数单元的筛选 ID、union、overlap、boundary 和 overlay；",
            "- `tables/`：实例级、逐帧和单元汇总 CSV/JSONL；",
            "- `SCREENING_RESULT.json`：配对主效应、诊断 Pareto 与候选工作点；",
            "- `failure_cases/`：空结果、碎片、巨型 mask、高重叠和高边界复杂度索引；",
            "- `visualizations/`：矩阵热力图、诊断 Pareto 和代表帧 overlay；",
            "- `artifact_inventory.*`：所有证据文件的路径、大小和 SHA-256。",
            "",
            "## 不能声明",
            "",
            "- 不得声明 Mask AP50/AP75、boundary F、small-object recall；",
            "- 不得把诊断 Pareto 前沿称为准确率最优；",
            "- 不得把低碎片数自动解释成更少漏检；",
            "- NMS IoU 是 FastSAM 后处理阈值，不是预测 mask 对 GT 的 IoU。",
            "",
        ]
    )
    report = "\n".join(lines)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    escaped = html.escape(report)
    (output / "REPORT.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>E11 evidence</title>"
        "<style>body{max-width:1200px;margin:2rem auto;font:15px/1.5 sans-serif}"
        "pre{white-space:pre-wrap}</style><pre>"
        + escaped
        + "</pre>",
        encoding="utf-8",
    )


def reseal_existing(output: Path) -> None:
    if not output.is_dir():
        raise FileNotFoundError(output)
    report_path = output / "REPORT.md"
    report_html_path = output / "REPORT.html"
    completion_path = output / "COMPLETION.json"
    if not report_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError("existing E11 report/completion is incomplete")
    neutral_line = (
        "- inventory 根哈希：见 `inventory_summary.json`；报告本身在根哈希"
        "覆盖范围内，因此不在报告中内嵌会产生自引用的根值"
    )
    report = report_path.read_text(encoding="utf-8")
    report = re.sub(
        r"- inventory 根哈希：`[0-9a-f]{64}`",
        neutral_line,
        report,
    )
    report = re.sub(
        r"- 证据文件：\d+，\d+(?:\.\d+)? GiB",
        "- 证据文件数与总字节数：见 `inventory_summary.json`（避免报告内容"
        "对自身封存字节数产生循环依赖）",
        report,
    )
    screening_line = (
        "- `SCREENING_RESULT.json`：配对主效应、诊断 Pareto 与候选工作点；"
    )
    if screening_line not in report:
        report = report.replace(
            "- `failure_cases/`：空结果、碎片、巨型 mask、高重叠和高边界复杂度索引；",
            screening_line
            + "\n- `failure_cases/`：空结果、碎片、巨型 mask、高重叠和高边界复杂度索引；",
        )
    report_path.write_text(report, encoding="utf-8")
    if report_html_path.is_file():
        escaped = html.escape(report)
        report_html_path.write_text(
            "<!doctype html><meta charset='utf-8'><title>E11 evidence</title>"
            "<style>body{max-width:1200px;margin:2rem auto;"
            "font:15px/1.5 sans-serif}pre{white-space:pre-wrap}</style><pre>"
            + escaped
            + "</pre>",
            encoding="utf-8",
        )
    previous_completion = json.loads(
        completion_path.read_text(encoding="utf-8")
    )
    previous_inventory = previous_completion.get("inventory")
    screening_result = build_screening_result(output)
    inventory = inventory_tree(output)
    history = list(previous_completion.get("inventory_history") or [])
    if previous_inventory:
        history.append(
            {
                "superseded_utc": utc_now(),
                "reason": (
                    "Report embedded a pre-final inventory root; remove the "
                    "circular value and reseal without changing native masks."
                ),
                "inventory": previous_inventory,
            }
        )
    previous_completion.update(
        {
            "resealed_utc": utc_now(),
            "reseal_script": str(Path(__file__).resolve()),
            "reseal_script_sha256": sha256_file(Path(__file__).resolve()),
            "inventory_history": history,
            "inventory": inventory,
            "inventory_jsonl_sha256": sha256_file(
                output / "artifact_inventory.jsonl"
            ),
            "inventory_csv_sha256": sha256_file(
                output / "artifact_inventory.csv"
            ),
            "inventory_summary_sha256": sha256_file(
                output / "inventory_summary.json"
            ),
            "screening_result": "SCREENING_RESULT.json",
            "screening_result_sha256": sha256_file(
                output / "SCREENING_RESULT.json"
            ),
            "correctness_winner": screening_result["correctness_winner"],
        }
    )
    write_json(completion_path, previous_completion)
    print(
        json.dumps(
            {
                "status": "resealed",
                "output": str(output),
                "inventory": inventory,
                "completion_sha256": sha256_file(completion_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    args = parse_args()
    started_utc = utc_now()
    started = time.perf_counter()

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    engine = args.engine.resolve()
    fastsam_config = args.fastsam_config.resolve()
    confs = sorted(set(parse_numeric_list(args.conf_values, float)))
    areas = sorted(set(parse_numeric_list(args.area_values, int)))
    ious = sorted(set(parse_numeric_list(args.iou_values, float)))

    if args.reseal_existing:
        reseal_existing(output)
        return
    if output.exists():
        raise FileExistsError(f"refusing to reuse E11 output directory: {output}")
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    if not engine.is_file():
        raise FileNotFoundError(engine)
    if not fastsam_config.is_file():
        raise FileNotFoundError(fastsam_config)
    if not (0.0 < args.overlay_alpha < 1.0):
        raise ValueError("--overlay-alpha must be between zero and one")
    if any(value <= 0 for value in areas):
        raise ValueError("minimum-area values must be positive")
    if any(not 0.0 <= value <= 1.0 for value in confs + ious):
        raise ValueError("confidence and IoU values must be in [0, 1]")

    output.mkdir(parents=True)
    (output / "tables").mkdir()
    (output / "failure_cases").mkdir()
    (output / "visualizations").mkdir()

    frames = read_frames(dataset, args.maximum_frames)
    baseline_tuple = (
        float(args.baseline_conf),
        int(args.baseline_area),
        float(args.baseline_iou),
    )
    inference_profiles = sorted(
        {(confidence, iou) for confidence in confs for iou in ious}
        | {(baseline_tuple[0], baseline_tuple[2])}
    )
    matrix_cells = [
        {
            "cell_id": cell_id(confidence, area, iou),
            "confidence": confidence,
            "minimum_area_px": area,
            "nms_iou": iou,
            "profile_id": profile_id(confidence, iou),
            "is_production_baseline": False,
        }
        for confidence in confs
        for area in areas
        for iou in ious
    ]
    baseline_cell = {
        "cell_id": cell_id(*baseline_tuple),
        "confidence": baseline_tuple[0],
        "minimum_area_px": baseline_tuple[1],
        "nms_iou": baseline_tuple[2],
        "profile_id": profile_id(baseline_tuple[0], baseline_tuple[2]),
        "is_production_baseline": True,
    }
    # Baseline first so every matrix cell can compute same-frame agreement
    # against the production reference without a second pass over image data.
    cells = [baseline_cell] + matrix_cells

    preregistration = {
        "schema": "daaam.g1_no_gt_e11_preregistration.v1",
        "created_utc": utc_now(),
        "authority": "diagnostic_gt_free",
        "experiment": "E11 isolated FastSAM",
        "input": {
            "dataset": str(dataset),
            "tick_index": str(dataset / "tick_index.json"),
            "tick_index_sha256": sha256_file(dataset / "tick_index.json"),
            "frame_count": len(frames),
            "first_source_frame": frames[0]["source_frame_index"],
            "last_source_frame": frames[-1]["source_frame_index"],
            "rgb_contract": "prepared monocular-undistorted + stereo-rectified cam0",
        },
        "matrix": {
            "confidence": confs,
            "minimum_area_px": areas,
            "nms_iou": ious,
            "full_factorial_cells": len(matrix_cells),
            "production_baseline": {
                "confidence": baseline_tuple[0],
                "minimum_area_px": baseline_tuple[1],
                "nms_iou": baseline_tuple[2],
            },
            "inference_reuse_rule": (
                "Run each confidence x NMS-IoU profile once with minimum area 0; "
                "derive area cells by exact pixel-area filtering."
            ),
        },
        "fixed_diagnostic_thresholds": {
            "tiny_instance": (
                "area <= max(600 px, 0.001 * image pixels)"
            ),
            "giant_instance": "area / image pixels >= 0.25",
            "fragmentation_flag": "tiny count >= 5 and tiny fraction >= 0.25",
            "high_overlap_flag": "pixels covered by >=2 masks / union >= 0.25",
            "high_boundary_flag": "boundary pixels / union >= 0.10",
        },
        "pareto_objectives": {
            "maximize": ["mean union coverage", "mean adjacent-frame union IoU"],
            "minimize": [
                "mean tiny-instance fraction",
                "mean overlap/union",
                "inference P95 latency",
            ],
            "interpretation": "diagnostic frontier only; no accuracy winner",
        },
        "unavailable_without_reviewed_gt": [
            "Mask AP50",
            "Mask AP75",
            "boundary F",
            "small-object recall",
            "correct merge/split labels",
        ],
    }
    write_json(output / "PRE_REGISTRATION.json", preregistration)
    write_jsonl(output / "source_frames.jsonl", frames)
    write_csv(output / "source_frames.csv", frames)

    invocation = {
        "schema": "daaam.g1_no_gt_e11_invocation.v1",
        "created_utc": utc_now(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "engine": str(engine),
        "engine_size_bytes": engine.stat().st_size,
        "engine_sha256": sha256_file(engine),
        "fastsam_config": str(fastsam_config),
        "fastsam_config_sha256": sha256_file(fastsam_config),
        "device": args.device,
        "imgsz_hw": [args.image_height, args.image_width],
        "nvidia_before": nvidia_snapshot(),
    }
    write_json(output / "invocation.json", invocation)

    # Import after immutable invocation products exist so startup failures remain
    # visible.  Minimum area zero is essential: these are the raw profile masks.
    from daaam.utils.segmentation import UniversalSegmenter

    segmenter = UniversalSegmenter(
        model_checkpoint_path=str(engine),
        model_config_path=str(fastsam_config),
        device=args.device,
        min_mask_region_area=0,
        imgsz=(args.image_height, args.image_width),
    )

    # Warm-up is excluded from latency, but retained as its own evidence.
    warmup_bgr = cv2.imread(frames[0]["rgb_path"], cv2.IMREAD_COLOR)
    if warmup_bgr is None:
        raise FileNotFoundError(frames[0]["rgb_path"])
    warmup_rgb = cv2.cvtColor(warmup_bgr, cv2.COLOR_BGR2RGB)
    warmup_started = time.perf_counter()
    warmup_dets, warmup_masks = segmenter(
        warmup_rgb,
        fastsam_conf=baseline_tuple[0],
        fastsam_iou=baseline_tuple[2],
        fastsam_retina_masks=True,
        fastsam_imgsz=(args.image_height, args.image_width),
    )
    write_json(
        output / "warmup.json",
        {
            "elapsed_ms": (time.perf_counter() - warmup_started) * 1000.0,
            "detections": len(warmup_dets),
            "masks": len(warmup_masks),
            "excluded_from_latency": True,
        },
    )

    raw_instance_rows: list[dict[str, Any]] = []
    profile_frame_rows: list[dict[str, Any]] = []
    cell_frame_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    profile_latencies: dict[str, list[float]] = {
        profile_id(confidence, iou): []
        for confidence, iou in inference_profiles
    }
    cell_previous_union: dict[str, np.ndarray | None] = {
        str(cell["cell_id"]): None for cell in cells
    }
    baseline_unions_by_frame: dict[int, np.ndarray] = {}

    for frame_position, frame in enumerate(frames):
        frame_index = int(frame["frame_index"])
        source_index = int(frame["source_frame_index"])
        bgr = cv2.imread(frame["rgb_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(frame["rgb_path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_shape = bgr.shape[:2]
        image_pixels = int(image_shape[0] * image_shape[1])
        tiny_threshold = max(600, int(math.ceil(image_pixels * 0.001)))

        profile_results: dict[
            str, tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]]]
        ] = {}
        for confidence, iou in inference_profiles:
            current_profile_id = profile_id(confidence, iou)
            profile_root = output / "raw_profiles" / current_profile_id
            started_inference = time.perf_counter()
            detections, masks = segmenter(
                rgb,
                fastsam_conf=confidence,
                fastsam_iou=iou,
                fastsam_retina_masks=True,
                fastsam_imgsz=(args.image_height, args.image_width),
            )
            latency_ms = (time.perf_counter() - started_inference) * 1000.0
            profile_latencies[current_profile_id].append(latency_ms)

            normalized_masks: list[np.ndarray] = []
            instance_records: list[dict[str, Any]] = []
            frame_instance_dir = (
                profile_root / "frames" / f"{frame_index:08d}" / "instances"
            )
            for instance_id, (detection, mask) in enumerate(
                zip(detections.tolist(), masks, strict=True)
            ):
                raw_mask = np.asarray(mask, dtype=bool)
                native_shape = list(raw_mask.shape)
                resized_to_source = False
                if raw_mask.shape != image_shape:
                    raw_mask = (
                        cv2.resize(
                            raw_mask.astype(np.uint8),
                            (image_shape[1], image_shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        > 0
                    )
                    resized_to_source = True
                normalized_masks.append(raw_mask)
                _, contour_records, boundary, perimeter = contours_for_mask(
                    raw_mask
                )
                components, _, component_stats, _ = cv2.connectedComponentsWithStats(
                    raw_mask.astype(np.uint8), connectivity=8
                )
                component_areas = (
                    component_stats[1:, cv2.CC_STAT_AREA].astype(int).tolist()
                    if components > 1
                    else []
                )
                area_px = int(np.count_nonzero(raw_mask))
                x1, y1, x2, y2, score, class_id = detection
                mask_path = (
                    frame_instance_dir / f"instance_{instance_id:04d}_mask.png"
                )
                boundary_path = (
                    frame_instance_dir / f"instance_{instance_id:04d}_boundary.png"
                )
                require_write_image(
                    mask_path, np.where(raw_mask, 255, 0).astype(np.uint8)
                )
                require_write_image(boundary_path, boundary)
                record = {
                    "schema": "daaam.g1_no_gt_e11_raw_instance.v1",
                    "profile_id": current_profile_id,
                    "confidence_threshold": confidence,
                    "nms_iou_threshold": iou,
                    "frame_index": frame_index,
                    "source_frame_index": source_index,
                    "sensor_time_ns": frame["sensor_time_ns"],
                    "instance_id": instance_id,
                    "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "model_confidence": float(score),
                    "model_class_id": int(class_id),
                    "area_px": area_px,
                    "area_ratio": area_px / image_pixels,
                    "perimeter_px": perimeter,
                    "boundary_pixel_count": int(np.count_nonzero(boundary)),
                    "component_count": int(max(0, components - 1)),
                    "component_areas_px": component_areas,
                    "native_mask_shape": native_shape,
                    "source_mask_shape": list(raw_mask.shape),
                    "resized_to_source": resized_to_source,
                    "mask_path": str(mask_path),
                    "mask_sha256": sha256_file(mask_path),
                    "boundary_path": str(boundary_path),
                    "boundary_sha256": sha256_file(boundary_path),
                    "contours": contour_records,
                    "metric_status": "exact artifact geometry; no correctness GT",
                }
                instance_records.append(record)
                raw_instance_rows.append(
                    {
                        key: value
                        for key, value in record.items()
                        if key
                        not in {
                            "contours",
                            "component_areas_px",
                            "box_xyxy",
                            "native_mask_shape",
                            "source_mask_shape",
                        }
                    }
                    | {
                        "box_xyxy_json": json.dumps(record["box_xyxy"]),
                        "component_areas_px_json": json.dumps(component_areas),
                    }
                )
            raw_ids = list(range(len(normalized_masks)))
            union, overlap_count, boundary_union = aggregate_masks(
                normalized_masks, raw_ids, image_shape
            )
            profile_frame_dir = (
                profile_root / "frames" / f"{frame_index:08d}"
            )
            require_write_image(profile_frame_dir / "union_mask.png", union)
            require_write_image(
                profile_frame_dir / "overlap_count.png", overlap_count
            )
            require_write_image(
                profile_frame_dir / "boundary_union.png", boundary_union
            )
            require_write_image(
                profile_frame_dir / "raw_overlay.png",
                draw_overlay(
                    bgr,
                    normalized_masks,
                    detections,
                    raw_ids,
                    alpha=args.overlay_alpha,
                    title=(
                        f"RAW {current_profile_id} | source={source_index} | "
                        f"instances={len(raw_ids)}"
                    ),
                ),
            )
            frame_record = {
                "schema": "daaam.g1_no_gt_e11_raw_frame.v1",
                "profile_id": current_profile_id,
                "confidence_threshold": confidence,
                "nms_iou_threshold": iou,
                "frame_index": frame_index,
                "source_frame_index": source_index,
                "sensor_time_ns": frame["sensor_time_ns"],
                "rgb_path": frame["rgb_path"],
                "rgb_sha256": frame["rgb_sha256"],
                "image_shape": list(image_shape),
                "inference_latency_ms": latency_ms,
                "raw_instance_count": len(raw_ids),
                "union_pixels": int(np.count_nonzero(union)),
                "union_ratio": float(np.count_nonzero(union) / image_pixels),
                "overlap_pixels": int(np.count_nonzero(overlap_count >= 2)),
                "boundary_pixels": int(np.count_nonzero(boundary_union)),
                "instances": instance_records,
                "metric_status": "exact artifact statistics; no correctness GT",
            }
            write_json(profile_frame_dir / "frame.json", frame_record)
            profile_frame_rows.append(
                {
                    key: value
                    for key, value in frame_record.items()
                    if key not in {"instances", "image_shape"}
                }
            )
            profile_results[current_profile_id] = (
                np.asarray(detections, dtype=np.float64),
                normalized_masks,
                instance_records,
            )

        # Derive every area cell from exact raw profile instances.
        for cell in cells:
            current_cell_id = str(cell["cell_id"])
            current_profile_id = str(cell["profile_id"])
            detections, masks, instances = profile_results[current_profile_id]
            kept_ids = [
                int(record["instance_id"])
                for record in instances
                if int(record["area_px"]) >= int(cell["minimum_area_px"])
            ]
            cell_started = time.perf_counter()
            union, overlap_count, boundary_union = aggregate_masks(
                masks, kept_ids, image_shape
            )
            union_pixels = int(np.count_nonzero(union))
            overlap_pixels = int(np.count_nonzero(overlap_count >= 2))
            boundary_pixels = int(np.count_nonzero(boundary_union))
            tiny_count = sum(
                int(instances[index]["area_px"]) <= tiny_threshold
                for index in kept_ids
            )
            giant_count = sum(
                float(instances[index]["area_ratio"]) >= 0.25
                for index in kept_ids
            )
            confidences = [
                float(instances[index]["model_confidence"]) for index in kept_ids
            ]
            current_previous = cell_previous_union[current_cell_id]
            temporal_iou = jaccard(current_previous, union)
            cell_previous_union[current_cell_id] = union.copy()
            if bool(cell["is_production_baseline"]):
                baseline_unions_by_frame[frame_index] = union.copy()
            baseline_agreement = (
                1.0
                if bool(cell["is_production_baseline"])
                else jaccard(baseline_unions_by_frame.get(frame_index), union)
            )
            cell_root = output / "cells" / current_cell_id
            overlay_path = cell_root / "overlays" / f"{frame_index:08d}.png"
            union_path = cell_root / "union_masks" / f"{frame_index:08d}.png"
            overlap_path = (
                cell_root / "overlap_counts" / f"{frame_index:08d}.png"
            )
            boundary_path = (
                cell_root / "boundary_unions" / f"{frame_index:08d}.png"
            )
            selected_path = (
                cell_root / "selected_instances" / f"{frame_index:08d}.json"
            )
            require_write_image(union_path, union)
            require_write_image(overlap_path, overlap_count)
            require_write_image(boundary_path, boundary_union)
            require_write_image(
                overlay_path,
                draw_overlay(
                    bgr,
                    masks,
                    detections,
                    kept_ids,
                    alpha=args.overlay_alpha,
                    title=(
                        f"{current_cell_id} | source={source_index} | "
                        f"kept={len(kept_ids)}/{len(instances)}"
                    ),
                ),
            )
            postprocess_ms = (time.perf_counter() - cell_started) * 1000.0
            raw_sum_area = sum(int(instances[index]["area_px"]) for index in kept_ids)
            row = {
                "schema": "daaam.g1_no_gt_e11_cell_frame.v1",
                **cell,
                "frame_index": frame_index,
                "source_frame_index": source_index,
                "sensor_time_ns": frame["sensor_time_ns"],
                "raw_instance_count": len(instances),
                "kept_instance_count": len(kept_ids),
                "rejected_by_area_count": len(instances) - len(kept_ids),
                "tiny_threshold_px": tiny_threshold,
                "tiny_instance_count": tiny_count,
                "tiny_instance_fraction": (
                    tiny_count / len(kept_ids) if kept_ids else 0.0
                ),
                "giant_instance_count": giant_count,
                "union_pixels": union_pixels,
                "union_ratio": union_pixels / image_pixels,
                "summed_instance_area_px": raw_sum_area,
                "area_redundancy_ratio": (
                    raw_sum_area / union_pixels if union_pixels else 0.0
                ),
                "overlap_pixels": overlap_pixels,
                "overlap_over_union_ratio": (
                    overlap_pixels / union_pixels if union_pixels else 0.0
                ),
                "boundary_pixels": boundary_pixels,
                "boundary_over_union_ratio": (
                    boundary_pixels / union_pixels if union_pixels else 0.0
                ),
                "confidence_mean": (
                    float(np.mean(confidences)) if confidences else None
                ),
                "confidence_minimum": min(confidences) if confidences else None,
                "confidence_maximum": max(confidences) if confidences else None,
                "temporal_union_iou_previous": temporal_iou,
                "production_baseline_union_iou": baseline_agreement,
                "inference_latency_ms": profile_latencies[current_profile_id][-1],
                "cell_postprocess_ms": postprocess_ms,
                "union_mask_path": str(union_path),
                "overlap_count_path": str(overlap_path),
                "boundary_union_path": str(boundary_path),
                "overlay_path": str(overlay_path),
                "selected_instances_path": str(selected_path),
                "metric_status": "proxy/provisional (no reviewed human GT)",
            }
            flags = add_cell_failure_flags(row)
            row["failure_flags"] = flags
            write_json(
                selected_path,
                {
                    "schema": "daaam.g1_no_gt_e11_cell_selection.v1",
                    "cell": cell,
                    "frame_index": frame_index,
                    "source_frame_index": source_index,
                    "profile_id": current_profile_id,
                    "raw_frame_record": str(
                        output
                        / "raw_profiles"
                        / current_profile_id
                        / "frames"
                        / f"{frame_index:08d}"
                        / "frame.json"
                    ),
                    "kept_instance_ids": kept_ids,
                    "rejected_instance_ids": [
                        int(record["instance_id"])
                        for record in instances
                        if int(record["instance_id"]) not in set(kept_ids)
                    ],
                    "minimum_area_px": int(cell["minimum_area_px"]),
                    "failure_flags": flags,
                },
            )
            cell_frame_rows.append(row)
            for flag in flags:
                failure_rows.append(
                    {
                        "schema": "daaam.g1_no_gt_e11_failure.v1",
                        "failure_code": flag,
                        "cell_id": current_cell_id,
                        "frame_index": frame_index,
                        "source_frame_index": source_index,
                        "observed": {
                            "kept_instance_count": len(kept_ids),
                            "tiny_instance_count": tiny_count,
                            "tiny_instance_fraction": row[
                                "tiny_instance_fraction"
                            ],
                            "giant_instance_count": giant_count,
                            "overlap_over_union_ratio": row[
                                "overlap_over_union_ratio"
                            ],
                            "boundary_over_union_ratio": row[
                                "boundary_over_union_ratio"
                            ],
                        },
                        "overlay_path": str(overlay_path),
                        "selected_instances_path": str(selected_path),
                        "metric_status": "diagnostic proxy; not a correctness label",
                    }
                )
        print(
            json.dumps(
                {
                    "progress": f"{frame_position + 1}/{len(frames)}",
                    "frame_index": frame_index,
                    "source_frame_index": source_index,
                    "raw_instances_total": len(raw_instance_rows),
                    "failure_flags_total": len(failure_rows),
                    "elapsed_s": round(time.perf_counter() - started, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_jsonl(output / "tables/raw_instances.jsonl", raw_instance_rows)
    write_csv(output / "tables/raw_instances.csv", raw_instance_rows)
    write_jsonl(output / "tables/profile_per_frame.jsonl", profile_frame_rows)
    write_csv(output / "tables/profile_per_frame.csv", profile_frame_rows)
    flattened_cell_rows: list[dict[str, Any]] = []
    for row in cell_frame_rows:
        flat = dict(row)
        flat["failure_flags"] = json.dumps(
            row["failure_flags"], ensure_ascii=False
        )
        flattened_cell_rows.append(flat)
    write_jsonl(output / "tables/cell_per_frame.jsonl", cell_frame_rows)
    write_csv(output / "tables/cell_per_frame.csv", flattened_cell_rows)
    write_jsonl(output / "failure_cases/failure_cases.jsonl", failure_rows)
    failure_csv_rows = [
        dict(row)
        | {"observed": json.dumps(row["observed"], ensure_ascii=False)}
        for row in failure_rows
    ]
    write_csv(output / "failure_cases/failure_cases.csv", failure_csv_rows)

    cell_summaries: list[dict[str, Any]] = []
    for cell in cells:
        rows = [row for row in cell_frame_rows if row["cell_id"] == cell["cell_id"]]
        latency = summary(row["inference_latency_ms"] for row in rows)
        kept = summary(row["kept_instance_count"] for row in rows)
        union = summary(row["union_ratio"] for row in rows)
        temporal = summary(row["temporal_union_iou_previous"] for row in rows)
        tiny = summary(row["tiny_instance_fraction"] for row in rows)
        overlap = summary(row["overlap_over_union_ratio"] for row in rows)
        boundary = summary(row["boundary_over_union_ratio"] for row in rows)
        baseline_agreement = summary(
            row["production_baseline_union_iou"] for row in rows
        )
        failures_for_cell = [
            row for row in failure_rows if row["cell_id"] == cell["cell_id"]
        ]
        record = {
            "schema": "daaam.g1_no_gt_e11_cell_summary.v1",
            **cell,
            "frame_count": len(rows),
            "raw_instance_count": sum(int(row["raw_instance_count"]) for row in rows),
            "kept_instance_count": sum(
                int(row["kept_instance_count"]) for row in rows
            ),
            "kept_instance_count_mean": kept["mean"],
            "empty_frame_count": sum(
                int(row["kept_instance_count"]) == 0 for row in rows
            ),
            "empty_frame_fraction": sum(
                int(row["kept_instance_count"]) == 0 for row in rows
            )
            / len(rows),
            "union_ratio_mean": union["mean"],
            "union_ratio_p05": union["p05"],
            "union_ratio_p50": union["p50"],
            "union_ratio_p95": union["p95"],
            "temporal_union_iou_mean": temporal["mean"],
            "temporal_union_iou_p05": temporal["p05"],
            "temporal_union_iou_p50": temporal["p50"],
            "tiny_instance_fraction_mean": tiny["mean"],
            "giant_frame_fraction": sum(
                int(row["giant_instance_count"]) > 0 for row in rows
            )
            / len(rows),
            "overlap_over_union_ratio_mean": overlap["mean"],
            "boundary_over_union_ratio_mean": boundary["mean"],
            "production_baseline_union_iou_mean": baseline_agreement["mean"],
            "inference_latency_mean_ms": latency["mean"],
            "inference_latency_p50_ms": latency["p50"],
            "inference_latency_p95_ms": latency["p95"],
            "inference_latency_maximum_ms": latency["maximum"],
            "cell_postprocess_p95_ms": summary(
                row["cell_postprocess_ms"] for row in rows
            )["p95"],
            "failure_flag_count": len(failures_for_cell),
            "metric_status": "proxy/provisional (no reviewed human GT)",
        }
        cell_summaries.append(record)

    frontier = pareto_frontier(
        [row for row in cell_summaries if not bool(row["is_production_baseline"])]
    )
    for row in cell_summaries:
        row["diagnostic_pareto_frontier"] = row["cell_id"] in set(frontier)
    write_json(output / "tables/cell_summary.json", cell_summaries)
    write_csv(output / "tables/cell_summary.csv", cell_summaries)
    write_json(
        output / "tables/diagnostic_pareto.json",
        {
            "schema": "daaam.g1_no_gt_e11_pareto.v1",
            "frontier_cell_ids": frontier,
            "objectives": preregistration["pareto_objectives"],
            "correctness_winner": None,
            "reason": "reviewed instance GT is unavailable",
        },
    )
    screening_result = build_screening_result(output)

    save_heatmaps(output, cell_summaries, confs, areas, ious)
    save_pareto_plot(output, cell_summaries, set(frontier))
    save_representative_montage(output, frames, cell_summaries)

    # Report first, then hash all evidence products.  The report references the
    # inventory file, but deliberately does not embed its root to avoid a
    # circular hash dependency.
    preliminary_inventory = {
        "file_count": 0,
        "total_bytes": 0,
        "manifest_root_sha256": "pending",
    }
    render_report(
        output,
        frames,
        cell_summaries,
        frontier,
        len(raw_instance_rows),
        len(failure_rows),
        preliminary_inventory,
    )
    inventory = inventory_tree(output)

    completion = {
        "schema": "daaam.g1_no_gt_e11_completion.v1",
        "status": "complete",
        "authority": "diagnostic_gt_free",
        "accuracy_claim": False,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "elapsed_s": time.perf_counter() - started,
        "frame_count": len(frames),
        "source_frame_range": [
            frames[0]["source_frame_index"],
            frames[-1]["source_frame_index"],
        ],
        "unique_inference_profiles": len(inference_profiles),
        "full_factorial_cells": len(matrix_cells),
        "production_baseline_cells": 1,
        "raw_instance_count": len(raw_instance_rows),
        "cell_frame_records": len(cell_frame_rows),
        "failure_flag_records": len(failure_rows),
        "diagnostic_pareto_frontier": frontier,
        "screening_result": "SCREENING_RESULT.json",
        "screening_result_sha256": sha256_file(
            output / "SCREENING_RESULT.json"
        ),
        "correctness_winner": screening_result["correctness_winner"],
        "inventory": inventory,
        "inventory_jsonl_sha256": sha256_file(
            output / "artifact_inventory.jsonl"
        ),
        "inventory_csv_sha256": sha256_file(output / "artifact_inventory.csv"),
        "inventory_summary_sha256": sha256_file(
            output / "inventory_summary.json"
        ),
        "resource_usage": {
            "maximum_resident_set_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "nvidia_after": nvidia_snapshot(),
        },
        "unavailable_metrics": preregistration[
            "unavailable_without_reviewed_gt"
        ],
    }
    write_json(output / "COMPLETION.json", completion)
    print(json.dumps(completion, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
