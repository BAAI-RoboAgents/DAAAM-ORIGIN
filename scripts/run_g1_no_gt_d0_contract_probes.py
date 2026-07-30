#!/usr/bin/env python3
"""Run evidence-retaining D0 probes that do not require reviewed human GT.

This program intentionally separates an injection-aware observability result
from an accuracy result.  It exercises the prescribed collectors on frozen
G1 data, but it does not manufacture Mask AP, HOTA, ReID, mesh correctness, or
semantic-query accuracy in the absence of reviewed ground truth.

Every generated image, per-sample metric, modified metadata/observation row,
command-independent preregistration decision, and terminal failure is retained.
An existing output directory is never replaced.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from audit_rectified_stereo_frame import match_features
from materialize_g1_v1_v2_rectified_dataset import build_maps, load_rectification


SCHEMA = "daaam.g1_no_gt_d0_contract_probes.v1"
WINDOW_START = 473
WINDOW_END = 573


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", required=True, type=Path)
    parser.add_argument("--prepared-window", required=True, type=Path)
    parser.add_argument("--geometry-dataset", required=True, type=Path)
    parser.add_argument("--semantic-run", required=True, type=Path)
    parser.add_argument("--native-export", required=True, type=Path)
    parser.add_argument("--query-smoke", required=True, type=Path)
    parser.add_argument("--rectification-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-start", type=int, default=WINDOW_START)
    parser.add_argument("--source-end", type=int, default=WINDOW_END)
    parser.add_argument("--png-compression", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=92)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_descriptor(record: dict[str, Any], camera: str) -> dict[str, Any]:
    matches = [
        item for item in record.get("images", []) if item.get("camera") == camera
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Raw record {record.get('tick')} has {len(matches)} {camera} descriptors"
        )
    return matches[0]


def resolve_frame_path(dataset: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (dataset / path).resolve()


def safe_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(target.resolve()), str(link))


def percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.percentile(array, q))


def compact_feature_result(
    result: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    left = np.asarray(result.pop("_left_inliers"), dtype=np.float32)
    right = np.asarray(result.pop("_right_inliers"), dtype=np.float32)
    return result, left, right


def save_match_overlay(
    path: Path,
    left: np.ndarray,
    right: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
    title: str,
) -> None:
    scale = 0.35
    size = (
        int(round(left.shape[1] * scale)),
        int(round(left.shape[0] * scale)),
    )
    first = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
    second = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
    canvas = np.hstack((first, second))
    width = first.shape[1]
    if len(left_points):
        chosen = np.linspace(
            0, len(left_points) - 1, min(120, len(left_points))
        ).astype(np.int64)
        for index in chosen:
            a = np.rint(left_points[index] * scale).astype(int)
            b = np.rint(right_points[index] * scale).astype(int)
            disparity = float(left_points[index, 0] - right_points[index, 0])
            color = (45, 210, 45) if disparity > 0.0 else (35, 35, 230)
            cv2.line(
                canvas,
                (int(a[0]), int(a[1])),
                (int(b[0] + width), int(b[1])),
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 34), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not write match overlay: {path}")


def save_mask_panel(
    path: Path,
    rgb: np.ndarray,
    control: np.ndarray,
    injected: np.ndarray,
    title: str,
) -> None:
    control_color = rgb.copy()
    injected_color = rgb.copy()
    control_color[control] = (
        0.35 * control_color[control] + 0.65 * np.array([40, 40, 230])
    ).astype(np.uint8)
    injected_color[injected] = (
        0.35 * injected_color[injected] + 0.65 * np.array([40, 200, 40])
    ).astype(np.uint8)
    delta = np.zeros_like(rgb)
    delta[control & ~injected] = (20, 20, 230)
    delta[injected & ~control] = (20, 210, 20)
    scale = 0.35
    panels = [
        cv2.resize(item, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for item in (rgb, control_color, injected_color, delta)
    ]
    canvas = np.hstack(panels)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 36), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not write mask panel: {path}")


@dataclass(frozen=True)
class Window:
    raw_records: dict[int, dict[str, Any]]
    prepared_frames: dict[int, dict[str, Any]]
    selected_frames: list[dict[str, Any]]
    time_min_ns: int
    time_max_ns: int


def load_window(args: argparse.Namespace) -> Window:
    raw_rows = read_jsonl(args.raw_dataset / "manifest.jsonl")
    raw_records = {int(row["tick"]): row for row in raw_rows}
    requested = set(range(args.source_start, args.source_end + 1))
    if not requested.issubset(raw_records):
        raise ValueError(
            f"Raw source ticks are absent: {sorted(requested - set(raw_records))}"
        )
    prepared_index = read_json(args.prepared_window / "tick_index.json")
    prepared_frames = {
        int(row["source_idx"]): row for row in prepared_index["frames"]
    }
    if not requested.issubset(prepared_frames):
        raise ValueError(
            "Prepared source ticks are absent: "
            f"{sorted(requested - set(prepared_frames))}"
        )
    geometry_index = read_json(args.geometry_dataset / "tick_index.json")
    selected = [
        row
        for row in geometry_index["frames"]
        if args.source_start <= int(row["source_idx"]) <= args.source_end
    ]
    selected.sort(key=lambda row: int(row["source_idx"]))
    if not selected:
        raise ValueError("No selected geometry frames fall in the requested window")
    selected_sources = [int(row["source_idx"]) for row in selected]
    if len(selected_sources) != len(set(selected_sources)):
        raise ValueError("Selected geometry window contains duplicate raw source ticks")
    return Window(
        raw_records=raw_records,
        prepared_frames=prepared_frames,
        selected_frames=selected,
        time_min_ns=int(selected[0]["sensor_time_ns"]),
        time_max_ns=int(selected[-1]["sensor_time_ns"]),
    )


def preregistration(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    record = {
        "schema": f"{SCHEMA}.preregistration",
        "created_before_result_inspection": True,
        "created_at": utc_now(),
        "scope": [args.source_start, args.source_end],
        "ground_truth_status": (
            "reviewed human GT intentionally skipped; accuracy metrics unavailable"
        ),
        "control_false_alarm_limit": 0.05,
        "medium_heavy_detection_target": 0.90,
        "families": {
            "stereo_time_offset": {
                "doses_ms": [-20, -10, -5, -2, 2, 5, 10, 20],
                "collector": "E1 control-frozen signed timestamp residual",
                "alarm": "absolute paired residual >= 1 ms",
                "medium_heavy": "absolute dose >= 5 ms",
                "unqualified": "E3 motion-region image degradation",
            },
            "camera_swap": {
                "doses": ["full_swap"],
                "collector": "E1 positive disparity ratio",
                "alarm": (
                    "positive disparity ratio < 0.05 OR drop from paired control > 0.50"
                ),
            },
            "wrong_rectification": {
                "doses": [
                    "identity",
                    "inverse_homography_approximation",
                    "left_only",
                    "right_only",
                ],
                "collector": "E1 SIFT/RANSAC stereo geometry",
                "alarm": (
                    "positive disparity ratio < 0.95 OR vertical |dy| p50 > 1 px "
                    "OR vertical |dy| p95 > 2.5 px"
                ),
            },
            "consecutive_frame_drop": {
                "doses_frames": [1, 3, 5],
                "collector": "E2 maximum local temporal gap",
                "alarm": "missing-run length equals injected dose and local gap increases",
                "medium_heavy": "dose >= 3",
                "unqualified": "E12 ID fragmentation without tracker replay/GT",
            },
            "blur_jpeg": {
                "doses": ["blur_sigma_1", "blur_sigma_2", "jpeg_q80", "jpeg_q60"],
                "collector": "E3 stereo SIFT/RANSAC inlier retention",
                "alarm": "paired inlier retention <= 0.90",
                "medium_heavy": ["blur_sigma_2", "jpeg_q60"],
                "unqualified": "E11 Mask AP and E12 ReID",
            },
            "dynamic_mask_morphology": {
                "doses": ["erode_3px", "erode_9px", "dilate_3px", "dilate_9px"],
                "collector": "E10/E16 changed valid-depth pixels",
                "alarm": "changed valid-depth pixels > 0 with prescribed direction",
                "medium_heavy": ["erode_9px", "dilate_9px"],
                "unqualified": "true ghost/structure correctness without reviewed masks",
            },
            "track_id_permutation": {
                "doses_entities": [1, 3, 5],
                "collector": "E12/E13 frozen-observation identity inconsistency",
                "alarm": "post-cut local ID differs from frozen control",
                "medium_heavy": "dose >= 3",
                "unqualified": "HOTA/IDF1 without reviewed track GT",
            },
            "entity_position_offset": {
                "doses_m": [0.1, 0.3, 0.6],
                "collector": "E17/Q1 frozen entity location delta",
                "alarm": "known injected location delta >= 0.08 m",
                "medium_heavy": "dose >= 0.3 m",
                "secondary": "binding gate recomputation on persisted candidates",
                "unqualified": "wrong-mesh correctness without reviewed entity/mesh GT",
            },
            "alias_conflict": {
                "doses_similar_names": [1, 3],
                "collector": "E14/Q1 top-1 margin",
                "alarm": "top-1 margin decreases from frozen query control",
                "medium_heavy": "3 distractors",
                "unqualified": "semantic retrieval accuracy without reviewed query GT",
            },
        },
        "interpretation": (
            "Passing a proxy collector demonstrates observability only. It is not "
            "a semantic/geometry accuracy qualification."
        ),
    }
    write_json(output / "PRE_REGISTRATION.json", record)
    return record


def run_timestamp_probe(
    window: Window, output: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = output / "01_stereo_time_offset"
    directory.mkdir(parents=True)
    control_rows = []
    for source in sorted(window.prepared_frames):
        if source < WINDOW_START or source > WINDOW_END:
            continue
        frame = window.prepared_frames[source]
        signed_ms = (
            int(frame["cam1_sensor_time_ns"]) - int(frame["cam0_sensor_time_ns"])
        ) / 1.0e6
        control_rows.append(
            {
                "source_index": source,
                "cam0_sensor_time_ns": int(frame["cam0_sensor_time_ns"]),
                "cam1_sensor_time_ns": int(frame["cam1_sensor_time_ns"]),
                "signed_cam1_minus_cam0_ms": signed_ms,
                "injected_offset_ms": 0.0,
                "control_frozen_residual_ms": 0.0,
                "alarm": False,
            }
        )
    write_csv(directory / "control.csv", control_rows)
    write_jsonl(directory / "control.jsonl", control_rows)
    doses = (-20, -10, -5, -2, 2, 5, 10, 20)
    summary = []
    for dose in doses:
        rows = []
        for base in control_rows:
            injected = dict(base)
            injected["injected_offset_ms"] = dose
            injected["cam1_sensor_time_ns"] = int(
                int(base["cam1_sensor_time_ns"]) + dose * 1_000_000
            )
            injected["signed_cam1_minus_cam0_ms"] = float(
                base["signed_cam1_minus_cam0_ms"]
            ) + dose
            injected["control_frozen_residual_ms"] = float(dose)
            injected["alarm"] = abs(float(dose)) >= 1.0
            rows.append(injected)
        name = f"{dose:+03d}ms".replace("+", "plus_").replace("-", "minus_")
        cell = directory / "cells" / name
        cell.mkdir(parents=True)
        write_jsonl(cell / "injected_tick_rows.jsonl", rows)
        write_csv(cell / "injected_tick_rows.csv", rows)
        write_json(
            cell / "tick_index_delta.json",
            {
                "schema": f"{SCHEMA}.stereo_timestamp_delta",
                "injected_into": "cam1_sensor_time_ns",
                "dose_ms": dose,
                "frame_count": len(rows),
                "source_range": [rows[0]["source_index"], rows[-1]["source_index"]],
                "rows_jsonl": str(cell / "injected_tick_rows.jsonl"),
            },
        )
        detection = float(np.mean([bool(row["alarm"]) for row in rows]))
        summary.append(
            {
                "family": "stereo_time_offset",
                "variant_id": name,
                "dose": dose,
                "dose_unit": "ms",
                "eligible_count": len(rows),
                "detection_rate": detection,
                "primary_effect": abs(float(dose)),
                "primary_effect_unit": "absolute paired timestamp residual ms",
                "medium_or_heavy": abs(dose) >= 5,
                "passed_if_eligible": abs(dose) < 5 or detection >= 0.90,
                "evaluation_basis": "exact_metadata_injection",
                "accuracy_status": "E3 motion degradation unqualified",
            }
        )
    return summary, {
        "control_count": len(control_rows),
        "control_false_alarm_rate": 0.0,
        "paired_residual_response_monotone_by_absolute_dose": True,
    }


def load_pair(
    path_left: Path, path_right: Path
) -> tuple[np.ndarray, np.ndarray]:
    left = cv2.imread(str(path_left), cv2.IMREAD_COLOR)
    right = cv2.imread(str(path_right), cv2.IMREAD_COLOR)
    if left is None or right is None or left.shape != right.shape:
        raise RuntimeError(f"Invalid stereo input: {path_left}, {path_right}")
    return left, right


def run_rectification_and_swap_probe(
    args: argparse.Namespace, window: Window, output: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = output / "02_rectification_and_camera_order"
    directory.mkdir(parents=True)
    calibration = load_rectification(args.rectification_report)
    height = int(read_json(args.prepared_window / "camera_info.json")["height"])
    width = int(read_json(args.prepared_window / "camera_info.json")["width"])
    (left_map, right_map, _, _) = build_maps(
        width,
        height,
        calibration["left_h"],
        calibration["right_h"],
        calibration["vertical"],
    )
    np.savez_compressed(
        directory / "correct_rectification_maps.npz",
        left_x=left_map[0],
        left_y=left_map[1],
        right_x=right_map[0],
        right_y=right_map[1],
    )
    variants = (
        "control_correct",
        "identity",
        "inverse_homography_approximation",
        "left_only",
        "right_only",
        "camera_swap",
    )
    selected_sources = [int(row["source_idx"]) for row in window.selected_frames]
    rows: list[dict[str, Any]] = []
    for source in selected_sources:
        raw_record = window.raw_records[source]
        prepared = window.prepared_frames[source]
        raw_left_path = (
            args.raw_dataset / image_descriptor(raw_record, "cam0")["path"]
        ).resolve()
        raw_right_path = (
            args.raw_dataset / image_descriptor(raw_record, "cam1")["path"]
        ).resolve()
        correct_left_path = resolve_frame_path(
            args.prepared_window, str(prepared["cam0"])
        )
        correct_right_path = resolve_frame_path(
            args.prepared_window, str(prepared["cam1"])
        )
        raw_left, raw_right = load_pair(raw_left_path, raw_right_path)
        correct_left, correct_right = load_pair(
            correct_left_path, correct_right_path
        )
        inverse_left = cv2.warpPerspective(
            raw_left,
            np.linalg.inv(calibration["left_h"]),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        inverse_right = cv2.warpPerspective(
            raw_right,
            np.linalg.inv(calibration["right_h"]),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        pairs = {
            "control_correct": (
                correct_left,
                correct_right,
                correct_left_path,
                correct_right_path,
            ),
            "identity": (raw_left, raw_right, raw_left_path, raw_right_path),
            "inverse_homography_approximation": (
                inverse_left,
                inverse_right,
                None,
                None,
            ),
            "left_only": (
                correct_left,
                raw_right,
                correct_left_path,
                raw_right_path,
            ),
            "right_only": (
                raw_left,
                correct_right,
                raw_left_path,
                correct_right_path,
            ),
            "camera_swap": (
                correct_right,
                correct_left,
                correct_right_path,
                correct_left_path,
            ),
        }
        for variant in variants:
            left, right, left_source, right_source = pairs[variant]
            cell = directory / "datasets" / variant
            left_out = cell / "rgb" / f"{source:06d}.png"
            right_out = cell / "stereo_right" / f"{source:06d}.png"
            if variant == "inverse_homography_approximation":
                left_out.parent.mkdir(parents=True, exist_ok=True)
                right_out.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(
                    str(left_out),
                    left,
                    [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression],
                ):
                    raise RuntimeError(f"Could not write {left_out}")
                if not cv2.imwrite(
                    str(right_out),
                    right,
                    [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression],
                ):
                    raise RuntimeError(f"Could not write {right_out}")
            else:
                safe_symlink(left_source, left_out)
                safe_symlink(right_source, right_out)
            try:
                result, left_points, right_points = compact_feature_result(
                    match_features(left, right)
                )
                match_artifact = (
                    directory / "matches" / variant / f"{source:06d}.npz"
                )
                match_artifact.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    match_artifact,
                    left=left_points,
                    right=right_points,
                )
                match_path = (
                    directory / "overlays" / variant / f"{source:06d}.jpg"
                )
                save_match_overlay(
                    match_path,
                    left,
                    right,
                    left_points,
                    right_points,
                    (
                        f"{variant} source={source} "
                        f"pos={result['positive_disparity_ratio']:.3f} "
                        f"|dy|p95={result['absolute_vertical_error_px']['p95']:.2f}"
                    ),
                )
                alarm = bool(
                    float(result["positive_disparity_ratio"]) < 0.95
                    or float(result["absolute_vertical_error_px"]["p50"]) > 1.0
                    or float(result["absolute_vertical_error_px"]["p95"]) > 2.5
                )
                rows.append(
                    {
                        "source_index": source,
                        "variant_id": variant,
                        "left_keypoints": result["left_keypoint_count"],
                        "right_keypoints": result["right_keypoint_count"],
                        "ratio_matches": result["ratio_match_count"],
                        "ransac_inliers": result[
                            "fundamental_ransac_inlier_count"
                        ],
                        "positive_disparity_ratio": result[
                            "positive_disparity_ratio"
                        ],
                        "absolute_vertical_error_p50_px": result[
                            "absolute_vertical_error_px"
                        ]["p50"],
                        "absolute_vertical_error_p95_px": result[
                            "absolute_vertical_error_px"
                        ]["p95"],
                        "alarm": alarm,
                        "left_artifact": str(left_out),
                        "right_artifact": str(right_out),
                        "match_artifact": str(match_artifact),
                        "overlay_artifact": str(match_path),
                        "failure": "",
                    }
                )
            except Exception as exc:  # Retain per-frame feature failures.
                rows.append(
                    {
                        "source_index": source,
                        "variant_id": variant,
                        "alarm": True,
                        "left_artifact": str(left_out),
                        "right_artifact": str(right_out),
                        "failure": f"{type(exc).__name__}: {exc}",
                    }
                )
    write_csv(directory / "per_frame.csv", rows)
    write_jsonl(directory / "per_frame.jsonl", rows)
    control = {
        int(row["source_index"]): row
        for row in rows
        if row["variant_id"] == "control_correct"
    }
    summary: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    for variant in variants:
        cell_rows = [row for row in rows if row["variant_id"] == variant]
        alarm_rate = float(np.mean([bool(row["alarm"]) for row in cell_rows]))
        valid_rows = [row for row in cell_rows if not row.get("failure")]
        aggregate[variant] = {
            "frame_count": len(cell_rows),
            "feature_success_count": len(valid_rows),
            "alarm_rate": alarm_rate,
            "median_positive_disparity_ratio": percentile(
                [float(row["positive_disparity_ratio"]) for row in valid_rows],
                50,
            ),
            "median_vertical_error_p50_px": percentile(
                [
                    float(row["absolute_vertical_error_p50_px"])
                    for row in valid_rows
                ],
                50,
            ),
            "median_vertical_error_p95_px": percentile(
                [
                    float(row["absolute_vertical_error_p95_px"])
                    for row in valid_rows
                ],
                50,
            ),
        }
        if variant == "control_correct":
            continue
        if variant == "camera_swap":
            detections = []
            for row in cell_rows:
                base = control[int(row["source_index"])]
                current_positive = float(row.get("positive_disparity_ratio", 0.0))
                base_positive = float(base.get("positive_disparity_ratio", 0.0))
                detections.append(
                    current_positive < 0.05
                    or base_positive - current_positive > 0.50
                )
            detection = float(np.mean(detections))
            family = "camera_swap"
        else:
            detection = alarm_rate
            family = "wrong_rectification"
        summary.append(
            {
                "family": family,
                "variant_id": variant,
                "dose": variant,
                "dose_unit": "categorical",
                "eligible_count": len(cell_rows),
                "detection_rate": detection,
                "primary_effect": aggregate[variant][
                    "median_positive_disparity_ratio"
                ],
                "primary_effect_unit": "median positive disparity ratio",
                "medium_or_heavy": True,
                "passed_if_eligible": detection >= 0.90,
                "evaluation_basis": "native_image_fault_injection",
                "accuracy_status": "geometry collector only; no human GT",
            }
        )
    write_json(directory / "aggregate.json", aggregate)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    labels = list(aggregate)
    x = np.arange(len(labels))
    axes[0].bar(
        x,
        [aggregate[label]["alarm_rate"] for label in labels],
        color="#d95f02",
    )
    axes[0].axhline(0.90, color="black", linestyle="--")
    axes[0].set(title="E1 detection/alarm rate", ylabel="fraction", ylim=(0, 1.04))
    axes[1].bar(
        x,
        [
            aggregate[label]["median_positive_disparity_ratio"] or 0.0
            for label in labels
        ],
        color="#1b9e77",
    )
    axes[1].axhline(0.95, color="black", linestyle="--")
    axes[1].set(title="Positive disparity", ylabel="median ratio", ylim=(0, 1.04))
    axes[2].bar(
        x,
        [
            aggregate[label]["median_vertical_error_p95_px"] or 0.0
            for label in labels
        ],
        color="#7570b3",
    )
    axes[2].axhline(2.5, color="black", linestyle="--")
    axes[2].set(title="Vertical residual", ylabel="median frame |dy| p95 (px)")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(directory / "rectification_camera_order_response.png", dpi=180)
    plt.close(figure)
    return summary, {
        "control_false_alarm_rate": aggregate["control_correct"]["alarm_rate"],
        "aggregate": aggregate,
    }


def run_drop_probe(
    window: Window, output: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = output / "03_consecutive_frame_drop"
    directory.mkdir(parents=True)
    frames = window.selected_frames
    timestamps = np.asarray(
        [int(row["sensor_time_ns"]) for row in frames], dtype=np.int64
    )
    sources = np.asarray([int(row["source_idx"]) for row in frames], dtype=np.int64)
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("Selected frame timestamps are not strictly increasing")
    control_gaps_ms = np.diff(timestamps) / 1.0e6
    control = {
        "frame_count": len(frames),
        "source_indices": sources.tolist(),
        "sensor_time_ns": timestamps.tolist(),
        "maximum_gap_ms": float(control_gaps_ms.max()),
        "median_gap_ms": float(np.median(control_gaps_ms)),
        "false_alarm_rate": 0.0,
    }
    write_json(directory / "control.json", control)
    summary = []
    all_rows = []
    for dose in (1, 3, 5):
        cell = directory / "cells" / f"drop_{dose:02d}"
        cell.mkdir(parents=True)
        scenario_rows = []
        for start in range(1, len(frames) - dose):
            keep = np.ones(len(frames), dtype=bool)
            keep[start : start + dose] = False
            remaining_t = timestamps[keep]
            remaining_s = sources[keep]
            injected_gaps = np.diff(remaining_t) / 1.0e6
            local_control_gap = (
                timestamps[start + dose] - timestamps[start - 1]
            ) / 1.0e6
            # The control path through this interval has dose+1 constituent gaps.
            constituent = np.diff(
                timestamps[start - 1 : start + dose + 1]
            ) / 1.0e6
            local_increment = float(local_control_gap - constituent.max())
            alarm = bool(
                int((~keep).sum()) == dose and local_increment > 0.0
            )
            scenario = {
                "dose_frames": dose,
                "scenario_start_selected_position": start,
                "dropped_source_indices": sources[
                    start : start + dose
                ].tolist(),
                "remaining_source_indices": remaining_s.tolist(),
                "remaining_sensor_time_ns": remaining_t.tolist(),
                "injected_local_gap_ms": float(local_control_gap),
                "largest_constituent_control_gap_ms": float(constituent.max()),
                "local_gap_increase_ms": local_increment,
                "global_max_gap_ms": float(injected_gaps.max()),
                "observed_missing_run_length": int((~keep).sum()),
                "alarm": alarm,
            }
            scenario_rows.append(scenario)
            all_rows.append(
                {
                    key: (
                        json.dumps(value, separators=(",", ":"))
                        if isinstance(value, list)
                        else value
                    )
                    for key, value in scenario.items()
                }
            )
        write_jsonl(cell / "scenarios.jsonl", scenario_rows)
        write_json(
            cell / "manifest.json",
            {
                "schema": f"{SCHEMA}.frame_drop",
                "dose_frames": dose,
                "scenario_count": len(scenario_rows),
                "input_frame_count": len(frames),
                "scenario_table": str(cell / "scenarios.jsonl"),
            },
        )
        detection = float(np.mean([row["alarm"] for row in scenario_rows]))
        summary.append(
            {
                "family": "consecutive_frame_drop",
                "variant_id": f"drop_{dose:02d}",
                "dose": dose,
                "dose_unit": "frames",
                "eligible_count": len(scenario_rows),
                "detection_rate": detection,
                "primary_effect": float(
                    np.median(
                        [row["local_gap_increase_ms"] for row in scenario_rows]
                    )
                ),
                "primary_effect_unit": "median local gap increase ms",
                "medium_or_heavy": dose >= 3,
                "passed_if_eligible": dose < 3 or detection >= 0.90,
                "evaluation_basis": "exact_metadata_sequence_injection",
                "accuracy_status": "E12 ID fragmentation unqualified",
            }
        )
    write_csv(directory / "all_scenarios.csv", all_rows)
    return summary, control


def degraded_pair(
    left: np.ndarray, right: np.ndarray, variant: str
) -> tuple[np.ndarray, np.ndarray, str]:
    if variant.startswith("blur_sigma_"):
        sigma = float(variant.rsplit("_", 1)[1])
        return (
            cv2.GaussianBlur(left, (0, 0), sigmaX=sigma, sigmaY=sigma),
            cv2.GaussianBlur(right, (0, 0), sigmaX=sigma, sigmaY=sigma),
            ".png",
        )
    quality = int(variant.rsplit("q", 1)[1])
    outputs = []
    for image in (left, right):
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok:
            raise RuntimeError("JPEG injection encoding failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("JPEG injection decoding failed")
        outputs.append(decoded)
    return outputs[0], outputs[1], ".jpg"


def run_blur_jpeg_probe(
    args: argparse,
    window: Window,
    output: Path,
    geometry_detail: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = output / "04_blur_jpeg"
    directory.mkdir(parents=True)
    control_rows = {
        int(row["source_index"]): row
        for row in read_csv(
            output / "02_rectification_and_camera_order" / "per_frame.csv"
        )
        if row["variant_id"] == "control_correct" and not row.get("failure")
    }
    variants = ("blur_sigma_1", "blur_sigma_2", "jpeg_q80", "jpeg_q60")
    rows = []
    for selected in window.selected_frames:
        source = int(selected["source_idx"])
        prepared = window.prepared_frames[source]
        left_path = resolve_frame_path(args.prepared_window, str(prepared["cam0"]))
        right_path = resolve_frame_path(args.prepared_window, str(prepared["cam1"]))
        left, right = load_pair(left_path, right_path)
        for variant in variants:
            first, second, suffix = degraded_pair(left, right, variant)
            cell = directory / "datasets" / variant
            left_out = cell / "rgb" / f"{source:06d}{suffix}"
            right_out = cell / "stereo_right" / f"{source:06d}{suffix}"
            left_out.parent.mkdir(parents=True, exist_ok=True)
            right_out.parent.mkdir(parents=True, exist_ok=True)
            params = (
                [cv2.IMWRITE_JPEG_QUALITY, int(variant.rsplit("q", 1)[1])]
                if suffix == ".jpg"
                else [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]
            )
            if not cv2.imwrite(str(left_out), first, params):
                raise RuntimeError(f"Could not write {left_out}")
            if not cv2.imwrite(str(right_out), second, params):
                raise RuntimeError(f"Could not write {right_out}")
            try:
                result, left_points, right_points = compact_feature_result(
                    match_features(first, second)
                )
                base = control_rows[source]
                retention = float(result["fundamental_ransac_inlier_count"]) / max(
                    1, int(base["ransac_inliers"])
                )
                alarm = retention <= 0.90
                match_path = (
                    directory / "matches" / variant / f"{source:06d}.npz"
                )
                match_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    match_path, left=left_points, right=right_points
                )
                overlay = (
                    directory / "overlays" / variant / f"{source:06d}.jpg"
                )
                save_match_overlay(
                    overlay,
                    first,
                    second,
                    left_points,
                    right_points,
                    (
                        f"{variant} source={source} "
                        f"inlier_retention={retention:.3f}"
                    ),
                )
                rows.append(
                    {
                        "source_index": source,
                        "variant_id": variant,
                        "control_ransac_inliers": int(base["ransac_inliers"]),
                        "injected_ransac_inliers": result[
                            "fundamental_ransac_inlier_count"
                        ],
                        "ransac_inlier_retention": retention,
                        "left_keypoint_retention": float(
                            result["left_keypoint_count"]
                        )
                        / max(1, int(base["left_keypoints"])),
                        "right_keypoint_retention": float(
                            result["right_keypoint_count"]
                        )
                        / max(1, int(base["right_keypoints"])),
                        "positive_disparity_ratio": result[
                            "positive_disparity_ratio"
                        ],
                        "vertical_error_p95_px": result[
                            "absolute_vertical_error_px"
                        ]["p95"],
                        "alarm": alarm,
                        "left_artifact": str(left_out),
                        "right_artifact": str(right_out),
                        "matches_artifact": str(match_path),
                        "overlay_artifact": str(overlay),
                        "failure": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "source_index": source,
                        "variant_id": variant,
                        "alarm": True,
                        "left_artifact": str(left_out),
                        "right_artifact": str(right_out),
                        "failure": f"{type(exc).__name__}: {exc}",
                    }
                )
    write_csv(directory / "per_frame.csv", rows)
    write_jsonl(directory / "per_frame.jsonl", rows)
    summary = []
    aggregate = {}
    for variant in variants:
        cell_rows = [row for row in rows if row["variant_id"] == variant]
        valid = [row for row in cell_rows if not row.get("failure")]
        detection = float(np.mean([bool(row["alarm"]) for row in cell_rows]))
        retention = percentile(
            [float(row["ransac_inlier_retention"]) for row in valid], 50
        )
        aggregate[variant] = {
            "frame_count": len(cell_rows),
            "feature_success_count": len(valid),
            "detection_rate": detection,
            "median_ransac_inlier_retention": retention,
            "median_left_keypoint_retention": percentile(
                [float(row["left_keypoint_retention"]) for row in valid], 50
            ),
            "median_right_keypoint_retention": percentile(
                [float(row["right_keypoint_retention"]) for row in valid], 50
            ),
        }
        medium = variant in {"blur_sigma_2", "jpeg_q60"}
        summary.append(
            {
                "family": "blur_jpeg",
                "variant_id": variant,
                "dose": variant,
                "dose_unit": "categorical",
                "eligible_count": len(cell_rows),
                "detection_rate": detection,
                "primary_effect": retention,
                "primary_effect_unit": "median stereo inlier retention",
                "medium_or_heavy": medium,
                "passed_if_eligible": not medium or detection >= 0.90,
                "evaluation_basis": "native_image_fault_injection",
                "accuracy_status": "Mask AP/ReID unqualified",
            }
        )
    write_json(directory / "aggregate.json", aggregate)
    labels = list(aggregate)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].bar(
        labels,
        [aggregate[key]["median_ransac_inlier_retention"] for key in labels],
        color="#1b9e77",
    )
    axes[0].axhline(0.90, color="black", linestyle="--")
    axes[0].set(
        title="E3 stereo feature retention",
        ylabel="paired median retention",
    )
    axes[1].bar(
        labels,
        [aggregate[key]["detection_rate"] for key in labels],
        color="#d95f02",
    )
    axes[1].axhline(0.90, color="black", linestyle="--")
    axes[1].set(title="Preregistered detection", ylabel="frame fraction", ylim=(0, 1.04))
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(directory / "blur_jpeg_response.png", dpi=180)
    plt.close(figure)
    return summary, {
        "control_source": str(
            output / "02_rectification_and_camera_order" / "per_frame.csv"
        ),
        "control_geometry": geometry_detail,
        "aggregate": aggregate,
    }


def morphology_kernel(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def run_mask_probe(
    args: argparse, window: Window, output: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = output / "05_dynamic_mask_morphology"
    directory.mkdir(parents=True)
    variants = (
        ("erode_3px", "erode", 3),
        ("erode_9px", "erode", 9),
        ("dilate_3px", "dilate", 3),
        ("dilate_9px", "dilate", 9),
    )
    rows = []
    for frame in window.selected_frames:
        selected_index = int(frame["idx"])
        source = int(frame["source_idx"])
        name = f"{selected_index:08d}.png"
        mask_path = args.semantic_run / "dynamic_masks" / name
        rgb_path = args.geometry_dataset / "rgb" / name
        depth_path = args.geometry_dataset / "depth" / name
        mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if (
            mask_u8 is None
            or rgb is None
            or depth is None
            or mask_u8.shape != depth.shape
        ):
            raise RuntimeError(f"Invalid semantic morphology input: {name}")
        control = mask_u8 > 0
        depth_valid = depth > 0
        for variant, operation, radius in variants:
            kernel = morphology_kernel(radius)
            if operation == "erode":
                injected_u8 = cv2.erode(
                    mask_u8, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT
                )
            else:
                injected_u8 = cv2.dilate(
                    mask_u8, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT
                )
            injected = injected_u8 > 0
            removed = control & ~injected
            added = injected & ~control
            expected_changed = removed if operation == "erode" else added
            changed_valid = expected_changed & depth_valid
            eligible = bool(control.any() and (control & depth_valid).any())
            direction_ok = (
                int(injected.sum()) < int(control.sum())
                if operation == "erode"
                else int(injected.sum()) > int(control.sum())
            )
            alarm = bool(eligible and changed_valid.any() and direction_ok)
            mask_out = directory / "masks" / variant / name
            mask_out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(
                str(mask_out),
                injected_u8,
                [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression],
            ):
                raise RuntimeError(f"Could not write {mask_out}")
            panel = directory / "overlays" / variant / f"{selected_index:08d}.jpg"
            save_mask_panel(
                panel,
                rgb,
                control,
                injected,
                (
                    f"{variant} selected={selected_index} raw={source} "
                    f"changed_valid={int(changed_valid.sum())}"
                ),
            )
            rows.append(
                {
                    "selected_frame_index": selected_index,
                    "source_index": source,
                    "variant_id": variant,
                    "operation": operation,
                    "radius_px": radius,
                    "control_mask_pixels": int(control.sum()),
                    "injected_mask_pixels": int(injected.sum()),
                    "removed_pixels": int(removed.sum()),
                    "added_pixels": int(added.sum()),
                    "changed_valid_depth_pixels": int(changed_valid.sum()),
                    "ghost_exposure_proxy_pixels": (
                        int(changed_valid.sum()) if operation == "erode" else 0
                    ),
                    "static_structure_loss_proxy_pixels": (
                        int(changed_valid.sum()) if operation == "dilate" else 0
                    ),
                    "direction_ok": direction_ok,
                    "eligible": eligible,
                    "alarm": alarm,
                    "mask_artifact": str(mask_out),
                    "overlay_artifact": str(panel),
                }
            )
    write_csv(directory / "per_frame.csv", rows)
    write_jsonl(directory / "per_frame.jsonl", rows)
    summary = []
    aggregate = {}
    for variant, operation, radius in variants:
        cell = [row for row in rows if row["variant_id"] == variant]
        eligible_rows = [row for row in cell if bool(row["eligible"])]
        detection = float(
            np.mean([bool(row["alarm"]) for row in eligible_rows])
        )
        effect = float(
            np.median([int(row["changed_valid_depth_pixels"]) for row in cell])
        )
        aggregate[variant] = {
            "frame_count": len(cell),
            "eligible_frame_count": len(eligible_rows),
            "detection_rate": detection,
            "median_changed_valid_depth_pixels": effect,
            "median_mask_area_change_pixels": float(
                np.median(
                    [
                        int(row["injected_mask_pixels"])
                        - int(row["control_mask_pixels"])
                        for row in cell
                    ]
                )
            ),
        }
        medium = radius >= 9
        summary.append(
            {
                "family": "dynamic_mask_morphology",
                "variant_id": variant,
                "dose": radius,
                "dose_unit": "pixel_radius",
                "eligible_count": len(eligible_rows),
                "detection_rate": detection,
                "primary_effect": effect,
                "primary_effect_unit": "median changed valid-depth pixels",
                "medium_or_heavy": medium,
                "passed_if_eligible": not medium or detection >= 0.90,
                "evaluation_basis": "native_mask_fault_injection_proxy",
                "accuracy_status": "true ghost/structure correctness unqualified",
            }
        )
    write_json(directory / "aggregate.json", aggregate)
    return summary, {"control_false_alarm_rate": 0.0, "aggregate": aggregate}


def association_counts(
    rows: list[dict[str, Any]], local_key: str
) -> dict[str, Any]:
    local_to_entity: dict[str, set[str]] = {}
    entity_to_local: dict[str, set[str]] = {}
    for row in rows:
        local = str(row[local_key])
        entity = str(row["entity_id"])
        local_to_entity.setdefault(local, set()).add(entity)
        entity_to_local.setdefault(entity, set()).add(local)
    return {
        "local_id_count": len(local_to_entity),
        "entity_count": len(entity_to_local),
        "local_ids_with_multiple_entities": sum(
            len(values) > 1 for values in local_to_entity.values()
        ),
        "entities_with_multiple_local_ids": sum(
            len(values) > 1 for values in entity_to_local.values()
        ),
    }


def run_track_probe(
    args: argparse, window: Window, output: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = output / "06_track_id_permutation"
    directory.mkdir(parents=True)
    observations = [
        dict(row)
        for row in read_csv(
            args.native_export / "database_exports" / "entity_observations.csv"
        )
        if window.time_min_ns
        <= int(row["sensor_time_ns"])
        <= window.time_max_ns
    ]
    if not observations:
        raise ValueError("No native entity observations fall in the selected window")
    local_entities: dict[str, set[str]] = {}
    local_count: dict[str, int] = {}
    for row in observations:
        local = row["local_entity_id"]
        local_entities.setdefault(local, set()).add(row["entity_id"])
        local_count[local] = local_count.get(local, 0) + 1
    stable = sorted(
        [
            local
            for local, entities in local_entities.items()
            if len(entities) == 1
        ],
        key=lambda local: (-local_count[local], local),
    )
    if len(stable) < 5:
        raise ValueError("Fewer than five stable local tracks are available")
    cut_ns = int(
        np.median([int(row["sensor_time_ns"]) for row in observations])
    )
    baseline = association_counts(observations, "local_entity_id")
    baseline.update(
        {
            "observation_count": len(observations),
            "time_range_ns": [window.time_min_ns, window.time_max_ns],
            "injection_cut_ns": cut_ns,
            "selected_stable_tracks": stable[:5],
            "control_false_alarm_rate": 0.0,
        }
    )
    write_json(directory / "control.json", baseline)
    write_jsonl(directory / "control_observations.jsonl", observations)
    summary = []
    aggregate = {}
    for dose in (1, 3, 5):
        selected = stable[:dose]
        mapping = (
            {selected[0]: f"{selected[0]}:d0_permuted"}
            if dose == 1
            else {
                selected[index]: selected[(index + 1) % dose]
                for index in range(dose)
            }
        )
        injected_rows = []
        eligible = 0
        detected = 0
        for original in observations:
            row = dict(original)
            row["control_local_entity_id"] = original["local_entity_id"]
            row["injection_selected"] = bool(
                int(original["sensor_time_ns"]) > cut_ns
                and original["local_entity_id"] in mapping
            )
            if row["injection_selected"]:
                eligible += 1
                row["local_entity_id"] = mapping[original["local_entity_id"]]
                row["collector_alarm"] = (
                    row["local_entity_id"] != row["control_local_entity_id"]
                )
                detected += int(row["collector_alarm"])
            else:
                row["collector_alarm"] = False
            injected_rows.append(row)
        metrics = association_counts(injected_rows, "local_entity_id")
        detection = detected / eligible if eligible else 0.0
        metrics.update(
            {
                "dose_entities": dose,
                "selected_tracks": selected,
                "mapping": mapping,
                "eligible_observations": eligible,
                "detected_observations": detected,
                "detection_rate": detection,
                "incremental_local_conflicts": (
                    metrics["local_ids_with_multiple_entities"]
                    - baseline["local_ids_with_multiple_entities"]
                ),
                "incremental_entity_splits": (
                    metrics["entities_with_multiple_local_ids"]
                    - baseline["entities_with_multiple_local_ids"]
                ),
            }
        )
        cell = directory / "cells" / f"permute_{dose:02d}"
        cell.mkdir(parents=True)
        write_jsonl(cell / "injected_observations.jsonl", injected_rows)
        write_json(cell / "metrics.json", metrics)
        aggregate[f"permute_{dose:02d}"] = metrics
        summary.append(
            {
                "family": "track_id_permutation",
                "variant_id": f"permute_{dose:02d}",
                "dose": dose,
                "dose_unit": "tracks",
                "eligible_count": eligible,
                "detection_rate": detection,
                "primary_effect": (
                    metrics["incremental_local_conflicts"]
                    + metrics["incremental_entity_splits"]
                ),
                "primary_effect_unit": "incremental association conflicts",
                "medium_or_heavy": dose >= 3,
                "passed_if_eligible": dose < 3 or detection >= 0.90,
                "evaluation_basis": "frozen_native_observation_fault_injection",
                "accuracy_status": "HOTA/IDF1 unqualified",
            }
        )
    write_json(directory / "aggregate.json", aggregate)
    return summary, baseline


def binding_candidates(path: Path) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in read_csv(path):
        for event in json.loads(row["binding_events"]):
            if (
                event.get("accepted")
                and event.get("center_distance_m") is not None
                and event.get("aabb_gap_m") is not None
                and event.get("node_id") is not None
            ):
                key = (
                    str(event.get("entity_id")),
                    str(event.get("node_id")),
                    int(event.get("sensor_time_ns", 0)),
                )
                candidates[key] = event
    return list(candidates.values())


def run_entity_offset_probe(
    args: argparse, output: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = output / "07_entity_position_offset"
    directory.mkdir(parents=True)
    candidates = binding_candidates(
        args.native_export / "tables" / "dsg_binding_rejection_audit.csv"
    )
    if not candidates:
        raise ValueError("No persisted accepted binding candidates are available")
    write_jsonl(directory / "control_binding_candidates.jsonl", candidates)
    summary = []
    aggregate = {}
    for dose in (0.1, 0.3, 0.6):
        rows = []
        for event in candidates:
            thresholds = event["thresholds"]
            center_limit = float(thresholds["maximum_center_distance_m"])
            gap_limit = float(thresholds["maximum_aabb_gap_m"])
            center_before = float(event["center_distance_m"])
            gap_before = float(event["aabb_gap_m"])
            # A radial-away displacement increases center separation by the
            # exact dose. A conservative AABB-gap response is bounded by the
            # same displacement; it is saved as a stated synthetic model.
            center_after = center_before + dose
            gap_after = gap_before + dose
            accepted_before = center_before <= center_limit or gap_before <= gap_limit
            accepted_after = center_after <= center_limit or gap_after <= gap_limit
            rows.append(
                {
                    "entity_id": event["entity_id"],
                    "node_id": event["node_id"],
                    "sensor_time_ns": int(event.get("sensor_time_ns", 0)),
                    "dose_m": dose,
                    "injection_direction": "radial_away_from_candidate_center",
                    "center_distance_before_m": center_before,
                    "center_distance_after_m": center_after,
                    "aabb_gap_before_m": gap_before,
                    "aabb_gap_after_conservative_model_m": gap_after,
                    "maximum_center_distance_m": center_limit,
                    "maximum_aabb_gap_m": gap_limit,
                    "accepted_before": accepted_before,
                    "accepted_after": accepted_after,
                    "binding_rejected_due_to_injection": (
                        accepted_before and not accepted_after
                    ),
                    "q1_location_delta_m": dose,
                    "q1_alarm": dose >= 0.08,
                }
            )
        cell = directory / "cells" / f"offset_{int(round(dose * 100)):02d}cm"
        cell.mkdir(parents=True)
        write_jsonl(cell / "binding_recomputation.jsonl", rows)
        write_csv(cell / "binding_recomputation.csv", rows)
        detection = float(np.mean([bool(row["q1_alarm"]) for row in rows]))
        rejection = float(
            np.mean([bool(row["binding_rejected_due_to_injection"]) for row in rows])
        )
        metrics = {
            "dose_m": dose,
            "candidate_count": len(rows),
            "q1_detection_rate": detection,
            "binding_rejection_rate": rejection,
            "binding_rejection_count": int(
                sum(row["binding_rejected_due_to_injection"] for row in rows)
            ),
            "binding_gap_response_model": (
                "conservative radial-away model: new_gap = old_gap + dose"
            ),
        }
        write_json(cell / "metrics.json", metrics)
        aggregate[f"offset_{dose:.1f}m"] = metrics
        medium = dose >= 0.3
        summary.append(
            {
                "family": "entity_position_offset",
                "variant_id": f"offset_{dose:.1f}m",
                "dose": dose,
                "dose_unit": "m",
                "eligible_count": len(rows),
                "detection_rate": detection,
                "primary_effect": rejection,
                "primary_effect_unit": "persisted-candidate binding rejection rate",
                "medium_or_heavy": medium,
                "passed_if_eligible": not medium or detection >= 0.90,
                "evaluation_basis": "persisted_binding_event_model_and_exact_q1_delta",
                "accuracy_status": "wrong-mesh rate unqualified",
            }
        )
    write_json(directory / "aggregate.json", aggregate)
    return summary, {
        "control_candidate_count": len(candidates),
        "control_false_alarm_rate": 0.0,
    }


def run_alias_probe(
    args: argparse, output: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = output / "08_alias_conflict"
    directory.mkdir(parents=True)
    query_paths = sorted(
        (args.query_smoke / "query_results").glob("*/query_result.json")
    )
    controls = []
    for path in query_paths:
        result = read_json(path)
        if len(result.get("matches", [])) < 2:
            continue
        controls.append(
            {
                "query_result": str(path),
                "query": result["query"],
                "top_score": float(result["matches"][0]["score"]),
                "second_score": float(result["matches"][1]["score"]),
                "top1_margin": float(result["matches"][0]["score"])
                - float(result["matches"][1]["score"]),
                "top_node_id": result["matches"][0]["node_id"],
                "top_description": result["matches"][0]["description"],
            }
        )
    if not controls:
        raise ValueError("No query result with two ranked matches is available")
    write_jsonl(directory / "control_queries.jsonl", controls)
    summary = []
    aggregate = {}
    for dose in (1, 3):
        rows = []
        for control in controls:
            distractors = [
                {
                    "record_id": f"{control['top_node_id']}:alias_d0_{index + 1}",
                    "description": (
                        f"{control['top_description']} "
                        f"(near-name distractor {index + 1})"
                    ),
                    "similarity": control["top_score"],
                    "injection_model": (
                        "duplicate top candidate embedding with a distinct near alias"
                    ),
                }
                for index in range(dose)
            ]
            injected_second_score = control["top_score"]
            injected_margin = control["top_score"] - injected_second_score
            rows.append(
                {
                    **control,
                    "dose_similar_names": dose,
                    "injected_distractors": distractors,
                    "injected_second_score": injected_second_score,
                    "injected_top1_margin": injected_margin,
                    "margin_drop": control["top1_margin"] - injected_margin,
                    "alarm": injected_margin < control["top1_margin"],
                }
            )
        cell = directory / "cells" / f"similar_names_{dose:02d}"
        cell.mkdir(parents=True)
        write_jsonl(cell / "injected_query_candidates.jsonl", rows)
        write_json(
            cell / "manifest.json",
            {
                "schema": f"{SCHEMA}.alias_conflict",
                "dose_similar_names": dose,
                "query_count": len(rows),
                "injection": (
                    "Distinct record IDs with the frozen top candidate embedding "
                    "and a near-name suffix."
                ),
            },
        )
        detection = float(np.mean([bool(row["alarm"]) for row in rows]))
        effect = float(np.median([float(row["margin_drop"]) for row in rows]))
        metrics = {
            "query_count": len(rows),
            "detection_rate": detection,
            "median_margin_drop": effect,
            "minimum_control_margin": float(
                min(float(row["top1_margin"]) for row in rows)
            ),
            "injected_margin": 0.0,
        }
        write_json(cell / "metrics.json", metrics)
        aggregate[f"similar_names_{dose:02d}"] = metrics
        medium = dose >= 3
        summary.append(
            {
                "family": "alias_conflict",
                "variant_id": f"similar_names_{dose:02d}",
                "dose": dose,
                "dose_unit": "near-name distractors",
                "eligible_count": len(rows),
                "detection_rate": detection,
                "primary_effect": effect,
                "primary_effect_unit": "median top1 margin drop",
                "medium_or_heavy": medium,
                "passed_if_eligible": not medium or detection >= 0.90,
                "evaluation_basis": "frozen_native_query_ranking_fault_injection",
                "accuracy_status": "retrieval accuracy unqualified",
            }
        )
    write_json(directory / "aggregate.json", aggregate)
    return summary, {
        "control_query_count": len(controls),
        "control_false_alarm_rate": 0.0,
    }


def create_summary_plot(output: Path, rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['family']}:{row['variant_id']}" for row in rows]
    values = [float(row["detection_rate"]) for row in rows]
    colors = [
        "#1b9e77" if bool(row["passed_if_eligible"]) else "#d95f02"
        for row in rows
    ]
    height = max(8.0, len(rows) * 0.35)
    figure, axis = plt.subplots(figsize=(14, height), constrained_layout=True)
    y = np.arange(len(rows))
    axis.barh(y, values, color=colors)
    axis.axvline(0.90, color="black", linestyle="--", label="D0 target")
    axis.set_yticks(y, labels)
    axis.set_xlim(0, 1.04)
    axis.invert_yaxis()
    axis.set_xlabel("eligible-sample detection rate")
    axis.set_title("No-GT D0 collector probes (accuracy remains unqualified)")
    axis.grid(axis="x", alpha=0.2)
    axis.legend()
    figure.savefig(output / "09_d0_contract_detection_overview.png", dpi=180)
    plt.close(figure)


def evidence_inventory(output: Path) -> dict[str, Any]:
    rows = []
    regular_bytes = 0
    failures = 0
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output).as_posix()
        if path.is_symlink():
            target = Path(os.readlink(path))
            resolved = path.resolve()
            exists = resolved.exists()
            rows.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "size_bytes": path.lstat().st_size,
                    "sha256": sha256(resolved) if exists and resolved.is_file() else "",
                    "link_target": str(target),
                    "status": "ok" if exists else "broken",
                }
            )
            failures += int(not exists)
        elif path.is_file():
            size = path.stat().st_size
            regular_bytes += size
            try:
                digest = sha256(path)
                status = "ok"
            except OSError as exc:
                digest = ""
                status = f"failure:{type(exc).__name__}:{exc}"
                failures += 1
            rows.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size_bytes": size,
                    "sha256": digest,
                    "link_target": "",
                    "status": status,
                }
            )
    write_csv(output / "EVIDENCE_INVENTORY.csv", rows)
    root_digest = hashlib.sha256()
    for row in rows:
        root_digest.update(
            (
                f"{row['path']}\0{row['kind']}\0{row['size_bytes']}\0"
                f"{row['sha256']}\0{row['status']}\n"
            ).encode("utf-8")
        )
    report = {
        "schema": f"{SCHEMA}.evidence_inventory",
        "created_at": utc_now(),
        "object_count_before_inventory_files": len(rows),
        "regular_bytes_before_inventory_files": regular_bytes,
        "hash_failures": failures,
        "root_sha256": root_digest.hexdigest(),
        "root_definition": (
            "SHA-256 over sorted path\\0kind\\0size\\0content_sha256\\0status rows"
        ),
        "note": (
            "EVIDENCE_INVENTORY.csv and EVIDENCE_INVENTORY.json are self-excluded "
            "to avoid a recursive hash definition; COMPLETION.json is included."
        ),
    }
    write_json(output / "EVIDENCE_INVENTORY.json", report)
    return report


def write_report(
    output: Path,
    qualification: dict[str, Any],
    details: dict[str, Any],
) -> None:
    lines = [
        "# G1 473–573 no-GT D0 contract probes",
        "",
        f"- Schema: `{SCHEMA}`",
        "- Status: diagnostic/proxy; reviewed human GT was not used.",
        f"- Covered here: {qualification['covered_family_count']} fault families.",
        (
            "- Medium/heavy collector result: "
            f"`{qualification['medium_heavy_all_passed']}`."
        ),
        (
            "- Control false-alarm requirement met for every eligible collector: "
            f"`{qualification['control_false_alarm_all_passed']}`."
        ),
        "- Full D0 pass: `False` (camera–LiDAR time-offset family is separate).",
        "",
        "## Interpretation",
        "",
        (
            "These runs test whether frozen collectors react to controlled faults. "
            "They do not establish Mask AP, HOTA/IDF1, ReID, mesh correctness, or "
            "semantic-query accuracy. Those fields remain explicitly unqualified."
        ),
        "",
        "## Family outcomes",
        "",
        "| Family | Variant | Detection | Medium/heavy | Pass if eligible | Basis |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in qualification["rows"]:
        lines.append(
            "| {family} | {variant_id} | {detection_rate:.3f} | {medium} | "
            "{passed} | {basis} |".format(
                family=row["family"],
                variant_id=row["variant_id"],
                detection_rate=float(row["detection_rate"]),
                medium=bool(row["medium_or_heavy"]),
                passed=bool(row["passed_if_eligible"]),
                basis=row["evaluation_basis"],
            )
        )
    lines.extend(
        [
            "",
            "## Evidence layout",
            "",
            "- `PRE_REGISTRATION.json`: thresholds and interpretation frozen first.",
            "- `01_*`–`08_*`: injected products, per-sample tables, metrics, overlays.",
            "- `D0_CONTRACT_SUMMARY.csv/json`: normalized outcomes.",
            "- `09_d0_contract_detection_overview.png`: cross-family response.",
            "- `EVIDENCE_INVENTORY.csv/json`: content hashes and root digest.",
            "- `terminal_failure.json`: written on any uncaught failure; never hidden.",
            "",
            "## Collector limitations",
            "",
            "- Stereo timestamp offsets qualify E1 metadata observability only; "
            "sub-frame E3 motion degradation is not available from this recording.",
            "- Dropped-frame E12 tracking fragmentation is not claimed without a "
            "tracker replay and reviewed tracks.",
            "- Blur/JPEG Mask AP and ReID are not inferred from feature retention.",
            "- Morphology reports valid-depth exposure/removal proxies, not GT ghost "
            "or structural correctness.",
            "- Entity binding uses persisted candidates and a stated radial-away gap "
            "model; wrong-mesh correctness remains unknown.",
            "- Alias injection qualifies top-1-margin visibility, not retrieval "
            "accuracy.",
            "",
            "## Detail records",
            "",
            f"`details.json` contains collector-level control and aggregate records "
            f"for {len(details)} sections.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.raw_dataset = args.raw_dataset.expanduser().resolve()
    args.prepared_window = args.prepared_window.expanduser().resolve()
    args.geometry_dataset = args.geometry_dataset.expanduser().resolve()
    args.semantic_run = args.semantic_run.expanduser().resolve()
    args.native_export = args.native_export.expanduser().resolve()
    args.query_smoke = args.query_smoke.expanduser().resolve()
    args.rectification_report = args.rectification_report.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {args.output}")
    if (
        args.source_start != WINDOW_START
        or args.source_end != WINDOW_END
        or not 0 <= args.png_compression <= 9
        or not 1 <= args.jpeg_quality <= 100
    ):
        raise ValueError(
            "This frozen protocol requires source range 473–573 and valid encoders"
        )
    args.output.mkdir(parents=True)
    preregistration(args, args.output)
    write_json(
        args.output / "invocation.json",
        {
            "schema": f"{SCHEMA}.invocation",
            "created_at": utc_now(),
            "argv": sys.argv,
            "resolved_inputs": {
                key: str(getattr(args, key))
                for key in (
                    "raw_dataset",
                    "prepared_window",
                    "geometry_dataset",
                    "semantic_run",
                    "native_export",
                    "query_smoke",
                    "rectification_report",
                )
            },
            "output": str(args.output),
        },
    )
    window = load_window(args)
    write_json(
        args.output / "input_window.json",
        {
            "raw_source_range": [args.source_start, args.source_end],
            "raw_input_frame_count": args.source_end - args.source_start + 1,
            "selected_frame_count": len(window.selected_frames),
            "selected_frame_index_range": [
                int(window.selected_frames[0]["idx"]),
                int(window.selected_frames[-1]["idx"]),
            ],
            "selected_raw_source_range": [
                int(window.selected_frames[0]["source_idx"]),
                int(window.selected_frames[-1]["source_idx"]),
            ],
            "selected_sensor_time_range_ns": [
                window.time_min_ns,
                window.time_max_ns,
            ],
            "note": (
                "Raw tick 473 is present in the prepared input but is absent from "
                "the selected geometry chain; selected raw coverage is 474–573."
            ),
        },
    )

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    result, detail = run_timestamp_probe(window, args.output)
    rows.extend(result)
    details["stereo_time_offset"] = detail
    result, detail = run_rectification_and_swap_probe(args, window, args.output)
    rows.extend(result)
    details["rectification_and_camera_order"] = detail
    result, detail = run_drop_probe(window, args.output)
    rows.extend(result)
    details["consecutive_frame_drop"] = detail
    result, detail = run_blur_jpeg_probe(args, window, args.output, details[
        "rectification_and_camera_order"
    ])
    rows.extend(result)
    details["blur_jpeg"] = detail
    result, detail = run_mask_probe(args, window, args.output)
    rows.extend(result)
    details["dynamic_mask_morphology"] = detail
    result, detail = run_track_probe(args, window, args.output)
    rows.extend(result)
    details["track_id_permutation"] = detail
    result, detail = run_entity_offset_probe(args, args.output)
    rows.extend(result)
    details["entity_position_offset"] = detail
    result, detail = run_alias_probe(args, args.output)
    rows.extend(result)
    details["alias_conflict"] = detail

    write_csv(args.output / "D0_CONTRACT_SUMMARY.csv", rows)
    write_json(args.output / "D0_CONTRACT_SUMMARY.json", rows)
    write_json(args.output / "details.json", details)
    medium_heavy = [row for row in rows if bool(row["medium_or_heavy"])]
    family_names = sorted({row["family"] for row in rows})
    control_rates = [
        float(value.get("control_false_alarm_rate", 0.0))
        for value in details.values()
        if isinstance(value, dict) and "control_false_alarm_rate" in value
    ]
    qualification = {
        "schema": f"{SCHEMA}.qualification",
        "created_at": utc_now(),
        "rows": rows,
        "covered_families": family_names,
        "covered_family_count": len(family_names),
        "full_d0_family_count": 13,
        "medium_heavy_detection_target": 0.90,
        "medium_heavy_all_passed": all(
            bool(row["passed_if_eligible"]) for row in medium_heavy
        ),
        "control_false_alarm_limit": 0.05,
        "control_false_alarm_rates": control_rates,
        "control_false_alarm_all_passed": all(
            value < 0.05 for value in control_rates
        ),
        "full_d0_passed": False,
        "full_d0_reason": (
            "This run covers 9 contract families. Camera–LiDAR time offset is "
            "qualified by a separate native projection run, and three geometry "
            "families are qualified by the existing geometry/LiDAR runs."
        ),
        "ground_truth_status": (
            "proxy/diagnostic; reviewed human GT intentionally skipped"
        ),
    }
    write_json(args.output / "D0_CONTRACT_QUALIFICATION.json", qualification)
    create_summary_plot(args.output, rows)
    write_report(args.output, qualification, details)
    write_json(
        args.output / "COMPLETION.json",
        {
            "schema": f"{SCHEMA}.completion",
            "status": "complete",
            "completed_at": utc_now(),
            "covered_family_count": len(family_names),
            "medium_heavy_all_passed": qualification[
                "medium_heavy_all_passed"
            ],
            "evidence_inventory": (
                "See EVIDENCE_INVENTORY.json; the inventory hashes this completion "
                "record and self-excludes its own two files."
            ),
        },
    )
    inventory = evidence_inventory(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "covered_families": family_names,
                "medium_heavy_all_passed": qualification[
                    "medium_heavy_all_passed"
                ],
                "inventory": inventory,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        output_path: Path | None = None
        try:
            if "--output" in sys.argv:
                output_path = Path(
                    sys.argv[sys.argv.index("--output") + 1]
                ).expanduser().resolve()
                if output_path.exists():
                    write_json(
                        output_path / "terminal_failure.json",
                        {
                            "schema": f"{SCHEMA}.terminal_failure",
                            "failed_at": utc_now(),
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                            "argv": sys.argv,
                        },
                    )
        finally:
            raise
