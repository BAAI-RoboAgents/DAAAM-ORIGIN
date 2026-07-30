#!/usr/bin/env python3
"""Freeze and audit the P0/P1 inputs for the G1 473-573 V1.1 protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tomllib
from typing import Any

import cv2
import numpy as np
import yaml


SCHEMA = "daaam.g1_semantic_map_experiment.v1_1"
SPLITS = {
    "calibration": {"full": [473, 487], "core": [473, 487]},
    "development": {"full": [488, 527], "core": [493, 522]},
    "stress": {"full": [528, 557], "core": [533, 552]},
    "held_out": {"full": [558, 573], "core": [563, 573]},
}
ANCHOR_COUNTS = {
    "calibration": 4,
    "development": 10,
    "stress": 7,
    "held_out": 4,
}
FAILURE_TAXONOMY = {
    "F-INPUT": "frame/time/camera-order/hash/lineage mismatch",
    "F-RECT": "vertical residual, negative disparity, or insufficient valid area",
    "F-DEPTH": "outlier, hole, scale bias, or overconfidence",
    "F-POSE": "pose/reprojection/optimization/temporal failure",
    "F-KEY": "missed event or excessive temporal gap",
    "F-MOTION": "static false positive, dynamic miss, or excessive unknown",
    "F-SEG": "missed/merged/fragmented instance or poor boundary",
    "F-TRACK": "ID switch, fragmentation, or incorrect continuation",
    "F-ENTITY": "over-merge, over-split, incorrect name, or pending entity",
    "F-HYDRA": "missing/noisy mesh or missing object node",
    "F-BIND": "incorrect mesh binding, unexplained rejection, or bad relation",
    "F-QUERY": "miss, false accept, bad evidence, or bad rejection reason",
    "F-COMP": "deadline miss, OOM, or queue backlog",
    "F-OBS": "known injection not detected by metrics or alarms",
    "F-UNRESOLVED": "GT uncertainty or current experiment cannot disambiguate",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("/home/user/datasets/g1_20260724")
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=root / "experiments/g1_20260724_473_573_v1_1",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=root / "docs/g1_semantic_map_experiments_v1_1.md",
    )
    parser.add_argument("--prepared-dataset", required=True, type=Path)
    parser.add_argument("--geometry-audit", required=True, type=Path)
    parser.add_argument("--lidar-projection", required=True, type=Path)
    parser.add_argument("--calibration-report", required=True, type=Path)
    parser.add_argument("--start", type=int, default=473)
    parser.add_argument("--end", type=int, default=573)
    parser.add_argument("--anchor-seed", type=int, default=20260724)
    parser.add_argument(
        "--model-artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Model/repository artifact to freeze; repeat as needed.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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


def command(
    argv: list[str], cwd: Path, timeout: float = 60.0
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "stdout": "", "stderr": repr(error)}
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def load_records(dataset: Path, start: int, end: int) -> list[dict[str, Any]]:
    records = []
    with (dataset / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if start <= int(record["tick"]) <= end:
                records.append(record)
    return records


def split_for(frame: int) -> tuple[str, bool, bool]:
    for name, bounds in SPLITS.items():
        if bounds["full"][0] <= frame <= bounds["full"][1]:
            in_core = bounds["core"][0] <= frame <= bounds["core"][1]
            return name, in_core, not in_core
    raise ValueError(f"frame {frame} is outside the frozen splits")


def image_metrics(path: Path) -> tuple[list[int], float, float]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return (
        [int(image.shape[1]), int(image.shape[0])],
        float(np.mean(image)),
        float(cv2.Laplacian(image, cv2.CV_64F).var()),
    )


def yaw_degrees(quaternion: list[float]) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    return math.degrees(
        math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def select_anchor_proposal(
    frames_by_split: dict[str, list[int]], seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    proposal = []
    for split, count in ANCHOR_COUNTS.items():
        frames = frames_by_split[split]
        bins = np.array_split(np.asarray(frames, dtype=np.int64), count)
        selected = []
        for values in bins:
            choices = [int(value) for value in values]
            selected.append(rng.choice(choices))
        for frame in sorted(selected):
            proposal.append(
                {
                    "source_index": frame,
                    "split": split,
                    "status": "proposal_pending_human_challenge_review",
                    "selection_basis": (
                        "input-only temporal stratum; no candidate output used"
                    ),
                }
            )
    return proposal


def parse_model_artifacts(values: list[str], root: Path) -> dict[str, Path]:
    artifacts = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --model-artifact: {value}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        artifacts[name] = path.resolve()
    return artifacts


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    record = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if relative_to is not None:
        record["relative_path"] = str(path.resolve().relative_to(relative_to))
    return record


def calibration_checks(
    dataset: Path, calibration_report: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calibration_root = dataset / "calibrations/000000"
    v1_path = calibration_root / "New Calibration.yaml"
    v2_path = calibration_root / "New Calibration_V2.yaml"
    v1 = yaml.safe_load(v1_path.read_text(encoding="utf-8"))
    v2 = tomllib.loads(v2_path.read_text(encoding="utf-8"))
    combination = json.loads(calibration_report.read_text(encoding="utf-8"))
    rotations = {
        "v1.cam1.T_cn_cnm1": np.asarray(
            v1["cam1"]["T_cn_cnm1"], dtype=np.float64
        )[:3, :3],
        "v2.stereo.rotation": np.asarray(
            v2["stereo"]["rotation"], dtype=np.float64
        ),
        "v2.rectification.R1": np.asarray(
            v2["rectification"]["R1"], dtype=np.float64
        ),
        "v2.rectification.R2": np.asarray(
            v2["rectification"]["R2"], dtype=np.float64
        ),
        "combination.rotation_cam1_from_cam0": np.asarray(
            combination["selected_candidate"]["rotation_cam1_from_cam0"],
            dtype=np.float64,
        ),
        "combination.rectification_left_R": np.asarray(
            combination["selected_candidate"]["rectification_left_R"],
            dtype=np.float64,
        ),
        "combination.rectification_right_R": np.asarray(
            combination["selected_candidate"]["rectification_right_R"],
            dtype=np.float64,
        ),
    }
    checks = []
    rotation_records = []
    for name, rotation in rotations.items():
        orthogonality_error = float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3))
        )
        determinant = float(np.linalg.det(rotation))
        passed = (
            rotation.shape == (3, 3)
            and np.isfinite(rotation).all()
            and orthogonality_error <= 1.0e-6
            and abs(determinant - 1.0) <= 1.0e-6
        )
        checks.append(
            {
                "name": f"coordinate_rotation:{name}",
                "passed": passed,
                "failure_code": "F-INPUT",
                "detail": {
                    "from_to": name,
                    "orthogonality_error": orthogonality_error,
                    "determinant": determinant,
                    "unit": "dimensionless",
                },
            }
        )
        rotation_records.append(
            {
                "name": name,
                "orthogonality_error": orthogonality_error,
                "determinant": determinant,
            }
        )
    combination_contract = combination.get("experiment") or {}
    checks.append(
        {
            "name": "calibration_input_pixel_contract",
            "passed": combination_contract.get("input_pixel_model")
            == "already_monocular_undistorted_pinhole_D_zero",
            "failure_code": "F-INPUT",
            "detail": combination_contract.get("input_pixel_model"),
        }
    )
    checks.append(
        {
            "name": "calibration_acceptance",
            "passed": bool(
                (combination.get("acceptance") or {}).get("passed")
            ),
            "failure_code": "F-RECT",
            "detail": combination.get("acceptance"),
        }
    )
    return checks, rotation_records


def write_protocol_files(
    workspace: Path,
    plan: Path,
    plan_hash: str,
    anchor_seed: int,
) -> None:
    protocol = workspace / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    (protocol / "frozen_plan.md").write_text(
        "# G1 473–573 V1.1 frozen protocol\n\n"
        f"- Normative plan: `{plan.resolve()}`\n"
        f"- Normative plan SHA-256: `{plan_hash}`\n"
        "- cam0 is left; cam1 is right.\n"
        "- Input pixels are already monocular-undistorted; distortion is not "
        "applied again.\n"
        "- Stereo input uses the frozen V1/V2 combination learned outside "
        "473–573.\n"
        "- Splits and cores are exactly those in V1.1 §2.2.\n"
        f"- Anchor proposal seed: `{anchor_seed}`; proposal is not GT until "
        "human review locks challenge coverage.\n"
        "- Held-out GT/query contents remain unavailable until P8.\n"
        "- GT not reviewed => no semantic/DSG ranking.\n",
        encoding="utf-8",
    )
    thresholds = {
        "schema": SCHEMA,
        "status": "engineering_frozen__gt_dependent_pending_pilot",
        "stereo": {
            "dy_p50_max_px": 1.0,
            "frame_p95_p50_max_px": 3.0,
            "positive_disparity_ratio_min": 0.95,
            "strict_frame_pass_ratio_min": 0.90,
            "joint_valid_area_ratio_min": 0.75,
        },
        "depth": {
            "lidar_median_absrel_max": 0.12,
            "lr_consistency_mean_min": 0.80,
            "scale_source_frames": [473, 487],
        },
        "temporal": {"agreement_mean_min": 0.85},
        "runtime": {"global_service_p95_max_ms": 250.0},
        "gt_dependent": {
            "status": "pending_reviewed_gt_pilot_before_candidate_open",
            "metrics": [
                "Mask AP",
                "boundary F",
                "HOTA",
                "IDF1",
                "entity recall",
                "binding",
                "query recall",
            ],
        },
    }
    (protocol / "thresholds.yaml").write_text(
        yaml.safe_dump(thresholds, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (protocol / "failure_taxonomy.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": SCHEMA,
                "codes": FAILURE_TAXONOMY,
                "record_contract": [
                    "primary_cause",
                    "secondary_causes",
                    "first_module",
                    "downstream_symptoms",
                    "evidence_links",
                ],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_json(
        protocol / "query_set.json",
        {
            "schema": SCHEMA,
            "status": "not_generated_pending_reviewed_L5_GT",
            "expected_count": 48,
            "sealed_held_out": True,
            "queries": [],
            "reason": "V1.1 forbids query generation before L5 GT is frozen.",
        },
    )


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = args.dataset.expanduser().resolve()
    workspace = args.workspace.expanduser().resolve()
    plan = args.plan.expanduser().resolve()
    prepared = args.prepared_dataset.expanduser().resolve()
    geometry_audit_path = args.geometry_audit.expanduser().resolve()
    lidar_manifest_path = args.lidar_projection.expanduser().resolve()
    calibration_report = args.calibration_report.expanduser().resolve()
    if [args.start, args.end] != [473, 573]:
        raise ValueError("V1.1 protocol is frozen to source frames 473-573")
    sentinel = workspace / "manifests/source_frames.json"
    if sentinel.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen experiment: {sentinel}"
        )
    for path in (
        dataset / "manifest.json",
        dataset / "manifest.jsonl",
        dataset / "quality_report.json",
        plan,
        prepared / "tick_index.json",
        prepared / "camera_info.json",
        prepared / "image_integrity.json",
        prepared / "rectification_materialization_report.json",
        geometry_audit_path,
        lidar_manifest_path,
        calibration_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    for directory in (
        "protocol",
        "manifests",
        "ground_truth/annotations",
        "ground_truth/adjudication",
        "shared_artifacts",
        "runs",
        "comparisons",
        "final",
    ):
        (workspace / directory).mkdir(parents=True, exist_ok=True)

    records = load_records(dataset, args.start, args.end)
    expected_ticks = list(range(args.start, args.end + 1))
    actual_ticks = [int(record["tick"]) for record in records]
    checks: list[dict[str, Any]] = []

    def add(
        name: str,
        passed: bool,
        detail: Any,
        failure_code: str = "F-INPUT",
    ) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "failure_code": failure_code,
                "detail": detail,
            }
        )

    add(
        "source_frame_uniqueness_and_contiguity",
        actual_ticks == expected_ticks,
        {"expected": [args.start, args.end, 101], "actual_count": len(actual_ticks)},
    )
    quality = json.loads((dataset / "quality_report.json").read_text())
    add(
        "source_alignment_report",
        bool((quality.get("alignment") or {}).get("ok")),
        quality.get("alignment"),
    )

    source_rows = []
    input_files: dict[str, dict[str, Any]] = {}
    frame_metrics = []
    camera_times = {"cam0": [], "cam1": [], "lidar0": []}
    quaternion_errors = []
    frames_by_split = {name: [] for name in SPLITS}
    for record in records:
        tick = int(record["tick"])
        split, in_core, is_buffer = split_for(tick)
        frames_by_split[split].append(tick)
        images = {item["camera"]: item for item in record.get("images", [])}
        camera_order_ok = (
            [item.get("camera") for item in record.get("images", [])]
            == ["cam0", "cam1"]
            and images.get("cam0", {}).get("sensor") == "HEAD_LEFT_CAMERA"
            and images.get("cam1", {}).get("sensor") == "HEAD_RIGHT_CAMERA"
        )
        add(f"camera_order:{tick}", camera_order_ok, record.get("images"))
        image_rows = {}
        per_frame_brightness = None
        per_frame_sharpness = None
        for camera in ("cam0", "cam1"):
            descriptor = images.get(camera) or {}
            path = dataset / str(descriptor.get("path", ""))
            exists = path.is_file()
            add(f"image_exists:{tick}:{camera}", exists, str(path))
            if not exists:
                continue
            resolution, brightness, sharpness = image_metrics(path)
            add(
                f"image_resolution:{tick}:{camera}",
                resolution == [1280, 960],
                resolution,
            )
            file_info = file_record(path, relative_to=dataset)
            input_files[file_info["relative_path"]] = file_info
            image_rows[camera] = {
                **descriptor,
                "resolution": resolution,
                "sha256": file_info["sha256"],
            }
            camera_times[camera].append(int(descriptor["sensor_time_ns"]))
            if camera == "cam0":
                per_frame_brightness = brightness
                per_frame_sharpness = sharpness
        lidar_descriptors = record.get("lidar") or []
        lidar = lidar_descriptors[0] if lidar_descriptors else {}
        lidar_path = dataset / str(lidar.get("path", ""))
        lidar_exists = lidar_path.is_file()
        add(f"lidar_exists:{tick}", lidar_exists, str(lidar_path))
        lidar_shape = None
        lidar_finite_ratio = None
        lidar_info = None
        if lidar_exists:
            points = np.load(lidar_path, mmap_mode="r", allow_pickle=False)
            lidar_shape = [int(value) for value in points.shape]
            lidar_finite_ratio = float(np.isfinite(points).all(axis=1).mean())
            add(
                f"lidar_shape:{tick}",
                points.ndim == 2
                and points.shape[1] == 3
                and points.shape[0] == int(lidar.get("points", -1)),
                {"shape": lidar_shape, "manifest_points": lidar.get("points")},
            )
            add(
                f"lidar_finite:{tick}",
                lidar_finite_ratio >= 0.999,
                lidar_finite_ratio,
            )
            lidar_info = file_record(lidar_path, relative_to=dataset)
            input_files[lidar_info["relative_path"]] = lidar_info
            camera_times["lidar0"].append(int(lidar["sensor_time_ns"]))
        poses = ((record.get("poses") or {}).get("values") or {})
        pose_contract = {}
        for pose_name, pose in poses.items():
            quaternion = pose.get("orientation_xyzw") or []
            norm = (
                float(np.linalg.norm(np.asarray(quaternion, dtype=np.float64)))
                if len(quaternion) == 4
                else float("nan")
            )
            explicit_frames = bool(
                pose.get("target_frame") and pose.get("source_frame")
            )
            passed = explicit_frames and abs(norm - 1.0) <= 1.0e-6
            if not passed:
                quaternion_errors.append(
                    {"frame": tick, "pose": pose_name, "norm": norm}
                )
            pose_contract[pose_name] = {
                "target_frame": pose.get("target_frame"),
                "source_frame": pose.get("source_frame"),
                "translation_unit": "m",
                "quaternion_order": "xyzw",
                "quaternion_norm": norm,
            }
        cam0_time = int(images["cam0"]["sensor_time_ns"])
        cam1_time = int(images["cam1"]["sensor_time_ns"])
        stereo_delta_ms = abs(cam0_time - cam1_time) / 1.0e6
        map_quaternion = poses["map"]["orientation_xyzw"]
        frame_metrics.append(
            {
                "source_index": tick,
                "brightness": per_frame_brightness,
                "sharpness": per_frame_sharpness,
                "stereo_delta_ms": stereo_delta_ms,
                "yaw_deg": yaw_degrees(map_quaternion),
            }
        )
        source_rows.append(
            {
                "source_index": tick,
                "split": split,
                "in_statistical_core": in_core,
                "warmup_buffer": is_buffer,
                "sensor_time_ns": int(lidar.get("sensor_time_ns", 0)),
                "stereo_delta_ms": stereo_delta_ms,
                "images": image_rows,
                "lidar": {
                    **lidar,
                    "shape": lidar_shape,
                    "finite_ratio": lidar_finite_ratio,
                    "sha256": lidar_info["sha256"] if lidar_info else None,
                },
                "pose_contract": pose_contract,
                "errors": record.get("errors"),
            }
        )
    add(
        "pose_frame_and_quaternion_contract",
        not quaternion_errors,
        quaternion_errors,
    )
    for sensor, timestamps in camera_times.items():
        add(
            f"monotonic_timestamps:{sensor}",
            len(timestamps) == 101
            and all(a < b for a, b in zip(timestamps, timestamps[1:])),
            {
                "count": len(timestamps),
                "first": timestamps[0] if timestamps else None,
                "last": timestamps[-1] if timestamps else None,
                "unit": "ns",
            },
        )

    calibration_check_rows, rotation_records = calibration_checks(
        dataset, calibration_report
    )
    checks.extend(calibration_check_rows)

    prepared_index_path = prepared / "tick_index.json"
    prepared_index = json.loads(prepared_index_path.read_text())
    prepared_frames = prepared_index.get("frames") or []
    prepared_sources = [int(frame["source_idx"]) for frame in prepared_frames]
    add(
        "prepared_source_lineage",
        prepared_sources == expected_ticks,
        {
            "expected_count": 101,
            "actual_count": len(prepared_sources),
            "missing": sorted(set(expected_ticks) - set(prepared_sources)),
            "duplicates": len(prepared_sources) - len(set(prepared_sources)),
        },
    )
    add(
        "prepared_pixel_contract",
        prepared_index.get("source_projection_model")
        == "already_monocular_undistorted_pinhole"
        and prepared_index.get("projection_model") == "pinhole",
        {
            "source_projection_model": prepared_index.get(
                "source_projection_model"
            ),
            "projection_model": prepared_index.get("projection_model"),
        },
    )
    geometry_audit = json.loads(geometry_audit_path.read_text())
    geometry_frames = geometry_audit.get("frames") or []
    geometry_sources = [int(frame["source_index"]) for frame in geometry_frames]
    aggregate = geometry_audit.get("aggregate") or {}
    frame_medians = [
        float(frame["geometry"]["absolute_vertical_error_px"]["p50"])
        for frame in geometry_frames
    ]
    frame_p95s = [
        float(frame["geometry"]["absolute_vertical_error_px"]["p95"])
        for frame in geometry_frames
    ]
    positive = [
        float(frame["geometry"]["positive_disparity_ratio"])
        for frame in geometry_frames
    ]
    strict_ratio = (
        sum(
            float(frame["geometry"]["absolute_vertical_error_px"]["p50"])
            <= 1.0
            and float(frame["geometry"]["absolute_vertical_error_px"]["p95"])
            <= 3.0
            and float(frame["geometry"]["positive_disparity_ratio"]) >= 0.95
            for frame in geometry_frames
        )
        / len(geometry_frames)
        if geometry_frames
        else 0.0
    )
    materialization_report = json.loads(
        (prepared / "rectification_materialization_report.json").read_text()
    )
    source_stage = Path(materialization_report["source"]).resolve()
    source_stage_manifest = source_stage / "manifest.jsonl"
    source_stage_map_pose = source_stage / "state/000000/map_pose.jsonl"
    source_manifest_matches = (
        source_stage_manifest.is_file()
        and sha256(source_stage_manifest)
        == sha256(dataset / "manifest.jsonl")
    )
    add(
        "map_pose_stage_source_lineage",
        source_manifest_matches and source_stage_map_pose.is_file(),
        {
            "source_stage": str(source_stage),
            "source_manifest_matches_raw": source_manifest_matches,
            "derived_map_pose": str(source_stage_map_pose),
            "derived_map_pose_exists": source_stage_map_pose.is_file(),
        },
    )
    if source_stage_map_pose.is_file():
        stage_map_poses = {
            int(value["tick"]): value
            for value in (
                json.loads(line)
                for line in source_stage_map_pose.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
            if args.start <= int(value["tick"]) <= args.end
        }
        map_pose_mismatches = []
        for record in records:
            tick = int(record["tick"])
            expected_pose = (
                ((record.get("poses") or {}).get("values") or {}).get("map")
                or {}
            )
            staged_pose = (stage_map_poses.get(tick) or {}).get("pose") or {}
            if (
                expected_pose.get("target_frame")
                != staged_pose.get("target_frame")
                or expected_pose.get("source_frame")
                != staged_pose.get("source_frame")
                or not np.array_equal(
                    np.asarray(expected_pose.get("position", [])),
                    np.asarray(staged_pose.get("position", [])),
                )
                or not np.array_equal(
                    np.asarray(expected_pose.get("orientation_xyzw", [])),
                    np.asarray(staged_pose.get("orientation_xyzw", [])),
                )
            ):
                map_pose_mismatches.append(tick)
        add(
            "map_pose_stage_values_match_embedded_manifest",
            len(stage_map_poses) == 101 and not map_pose_mismatches,
            {
                "selected_entries": len(stage_map_poses),
                "mismatches": map_pose_mismatches,
            },
        )
    joint_valid = float(
        (materialization_report.get("valid_area") or {}).get(
            "joint_valid_area_ratio", 0.0
        )
    )
    stereo_gate = {
        "evaluated_frames": len(geometry_frames),
        "skipped_frames": int(aggregate.get("skipped_frame_count", 0)),
        "source_coverage_exact": geometry_sources == expected_ticks,
        "dy_p50_px": float(np.median(frame_medians))
        if frame_medians
        else None,
        "frame_p95_p50_px": float(np.median(frame_p95s))
        if frame_p95s
        else None,
        "positive_disparity_ratio_p50": float(np.median(positive))
        if positive
        else None,
        "strict_frame_pass_ratio": strict_ratio,
        "joint_valid_area_ratio": joint_valid,
    }
    stereo_gate_passed = (
        stereo_gate["source_coverage_exact"]
        and stereo_gate["skipped_frames"] == 0
        and stereo_gate["dy_p50_px"] is not None
        and stereo_gate["dy_p50_px"] <= 1.0
        and stereo_gate["frame_p95_p50_px"] <= 3.0
        and stereo_gate["positive_disparity_ratio_p50"] >= 0.95
        and strict_ratio >= 0.90
        and joint_valid >= 0.75
    )
    add(
        "v1_1_stereo_engineering_gate",
        stereo_gate_passed,
        stereo_gate,
        "F-RECT",
    )

    lidar_manifest = json.loads(lidar_manifest_path.read_text())
    lidar_frames_path = Path(
        (lidar_manifest.get("artifacts") or {}).get("frames_jsonl", "")
    )
    lidar_frame_records = (
        [
            json.loads(line)
            for line in lidar_frames_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if lidar_frames_path.is_file()
        else []
    )
    projected_counts = [
        int(record.get("projected_visible_points", 0))
        for record in lidar_frame_records
    ]
    add(
        "lidar_projection_lineage",
        lidar_manifest.get("source_range") == [473, 573]
        and int(lidar_manifest.get("frame_count", 0)) == 101
        and (
            (lidar_manifest.get("frame_semantics") or {}).get("rgb_pixel_space")
            == "prepared rectified cam0"
        )
        and lidar_manifest.get("prepared_tick_index_sha256")
        == sha256(prepared_index_path),
        {
            "source_range": lidar_manifest.get("source_range"),
            "frame_count": lidar_manifest.get("frame_count"),
            "frame_semantics": lidar_manifest.get("frame_semantics"),
            "prepared_tick_index_sha256": lidar_manifest.get(
                "prepared_tick_index_sha256"
            ),
        },
    )
    add(
        "lidar_projection_coverage",
        len(lidar_frame_records) == 101
        and [int(record["source_index"]) for record in lidar_frame_records]
        == expected_ticks
        and bool(projected_counts)
        and min(projected_counts) > 0,
        {
            "frames": len(lidar_frame_records),
            "minimum_projected_visible_points": (
                min(projected_counts) if projected_counts else None
            ),
            "median_projected_visible_points": (
                float(np.median(projected_counts))
                if projected_counts
                else None
            ),
            "maximum_projected_visible_points": (
                max(projected_counts) if projected_counts else None
            ),
            "occlusion_boundary_status": "pending_manual_not_judgeable_review",
        },
    )

    brightness = np.asarray(
        [row["brightness"] for row in frame_metrics], dtype=np.float64
    )
    sharpness = np.asarray(
        [row["sharpness"] for row in frame_metrics], dtype=np.float64
    )
    yaws = np.unwrap(
        np.radians([row["yaw_deg"] for row in frame_metrics])
    )
    yaw_delta = np.degrees(np.abs(np.diff(yaws, prepend=yaws[0])))
    thresholds = {
        "brightness_p05": float(np.percentile(brightness, 5)),
        "brightness_p95": float(np.percentile(brightness, 95)),
        "sharpness_p05": float(np.percentile(sharpness, 5)),
        "turning_abs_yaw_delta_p90_deg": float(np.percentile(yaw_delta, 90)),
    }
    challenge_rows = []
    for metric, turn in zip(frame_metrics, yaw_delta):
        delta = float(metric["stereo_delta_ms"])
        if delta <= 2.0:
            sync_tag = "sync_2"
        elif delta <= 5.0:
            sync_tag = "sync_5"
        elif delta <= 10.0:
            sync_tag = "sync_10"
        else:
            sync_tag = "sync_over10"
        automatic_tags = [sync_tag]
        if metric["sharpness"] <= thresholds["sharpness_p05"]:
            automatic_tags.append("blur_low")
        if (
            metric["brightness"] <= thresholds["brightness_p05"]
            or metric["brightness"] >= thresholds["brightness_p95"]
        ):
            automatic_tags.append("exposure_edge")
        if turn >= thresholds["turning_abs_yaw_delta_p90_deg"] and turn > 0:
            automatic_tags.append("turning")
        split, in_core, is_buffer = split_for(metric["source_index"])
        challenge_rows.append(
            {
                **metric,
                "abs_yaw_delta_deg": float(turn),
                "split": split,
                "in_statistical_core": in_core,
                "warmup_buffer": is_buffer,
                "automatic_tags": automatic_tags,
                "manual_tags": {
                    "status": "pending_human_review",
                    "required": [
                        "plant_boundary",
                        "thin_structure",
                        "occluded",
                        "semantic_dense",
                    ],
                    "values": [],
                },
            }
        )

    plan_hash = sha256(plan)
    write_protocol_files(workspace, plan, plan_hash, args.anchor_seed)
    write_json(
        workspace / "manifests/source_frames.json",
        {
            "schema": SCHEMA,
            "dataset": str(dataset),
            "source_range": [args.start, args.end],
            "frame_count": len(source_rows),
            "camera_order": {"cam0": "left", "cam1": "right"},
            "depth_unit": "m",
            "png_depth_unit": "mm where applicable and declared per artifact",
            "frames": source_rows,
        },
    )
    write_json(
        workspace / "manifests/splits.json",
        {
            "schema": SCHEMA,
            "status": "frozen",
            "splits": {
                name: {
                    **bounds,
                    "full_frames": list(
                        range(bounds["full"][0], bounds["full"][1] + 1)
                    ),
                    "core_frames": list(
                        range(bounds["core"][0], bounds["core"][1] + 1)
                    ),
                    "buffer_frames": [
                        frame
                        for frame in range(
                            bounds["full"][0], bounds["full"][1] + 1
                        )
                        if not (
                            bounds["core"][0] <= frame <= bounds["core"][1]
                        )
                    ],
                }
                for name, bounds in SPLITS.items()
            },
        },
    )
    write_json(
        workspace / "manifests/challenge_tags.json",
        {
            "schema": SCHEMA,
            "status": "automatic_tags_frozen__manual_tags_pending_review",
            "candidate_outputs_consulted": False,
            "thresholds": thresholds,
            "frames": challenge_rows,
        },
    )

    static_files = [
        dataset / "manifest.json",
        dataset / "manifest.jsonl",
        dataset / "quality_report.json",
        dataset / "timestamps/000000.txt",
        dataset / "poses/dense_global/000000/poses.txt",
        dataset / "poses/dense_global/000000/poses_7d.txt",
        dataset / "poses/dense_global/000000/aux_poses.jsonl",
        dataset / "state/000000/odom.jsonl",
        dataset / "state/000000/map_pose.jsonl",
        dataset / "state/000000/joint_states.jsonl",
        source_stage_map_pose,
        *sorted((dataset / "calibrations/000000").glob("*")),
        prepared / "tick_index.json",
        prepared / "camera_info.json",
        prepared / "image_integrity.json",
        prepared / "rectification_materialization_report.json",
        geometry_audit_path,
        lidar_manifest_path,
        lidar_frames_path,
        calibration_report,
        plan,
    ]
    for path in static_files:
        if path.is_file():
            input_files[str(path.resolve())] = file_record(path)
    model_artifacts = parse_model_artifacts(args.model_artifact, root)
    model_records = {}
    for name, path in model_artifacts.items():
        if path.is_file():
            model_records[name] = file_record(path)
        elif path.is_dir():
            directory_record: dict[str, Any] = {
                "path": str(path),
                "kind": "directory",
                "git_sha": command(
                    ["git", "rev-parse", "HEAD"], path
                )["stdout"],
                "git_status": command(
                    ["git", "status", "--porcelain"], path
                )["stdout"],
            }
            reference = path / "refs/main"
            if reference.is_file():
                revision = reference.read_text(encoding="utf-8").strip()
                tree = path / "trees" / f"{revision}.json"
                directory_record["content_addressed_model"] = {
                    "revision": revision,
                    "revision_ref": file_record(reference),
                    "tree": file_record(tree) if tree.is_file() else None,
                }
            model_records[name] = directory_record
        else:
            model_records[name] = {"path": str(path), "missing": True}
            add(f"model_artifact:{name}", False, str(path))

    git_sha = command(["git", "rev-parse", "HEAD"], root)
    git_status = command(["git", "status", "--porcelain"], root)
    git_diff = command(
        ["git", "diff", "--binary", "HEAD"], root, timeout=120.0
    )
    patch_path = workspace / "manifests/worktree.patch"
    patch_path.write_text(git_diff["stdout"] + "\n", encoding="utf-8")
    python_freeze = command(
        [sys.executable, "-m", "pip", "freeze"], root, timeout=120.0
    )
    nvidia = command(["nvidia-smi", "-q"], root, timeout=30.0)
    artifact_hashes = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": {"path": str(plan), "sha256": plan_hash},
        "repository": {
            "root": str(root),
            "git_sha": git_sha["stdout"],
            "git_status": git_status["stdout"],
            "worktree_patch": str(patch_path),
            "worktree_patch_sha256": sha256(patch_path),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "pip_freeze": python_freeze["stdout"],
            "nvidia_smi": nvidia["stdout"],
        },
        "calibration_rotation_checks": rotation_records,
        "models": model_records,
        "files": sorted(
            input_files.values(), key=lambda record: record["path"]
        ),
    }
    write_json(workspace / "manifests/artifact_hashes.json", artifact_hashes)

    anchors = select_anchor_proposal(frames_by_split, args.anchor_seed)
    write_json(
        workspace / "ground_truth/annotations/anchor_proposal.json",
        {
            "schema": SCHEMA,
            "status": "proposal_not_ground_truth",
            "seed": args.anchor_seed,
            "counts": ANCHOR_COUNTS,
            "anchors": anchors,
            "freeze_blocker": (
                "manual challenge tags and independent held-out reviewer pending"
            ),
        },
    )
    task_path = workspace / "ground_truth/annotations/tasks.jsonl"
    with task_path.open("w", encoding="utf-8") as stream:
        for row in source_rows:
            task = {
                "schema": SCHEMA,
                "source_index": row["source_index"],
                "split": row["split"],
                "layers": {
                    "L0": "automatic_proposal_pending_human_review",
                    "L1": "unlabeled",
                    "L2": (
                        "unlabeled_anchor"
                        if any(
                            anchor["source_index"] == row["source_index"]
                            for anchor in anchors
                        )
                        else "not_selected"
                    ),
                    "L3": "unlabeled",
                    "L4": (
                        "lidar_proposal_pending_occlusion_review"
                        if any(
                            anchor["source_index"] == row["source_index"]
                            for anchor in anchors
                        )
                        else "not_selected"
                    ),
                    "L5": "unlabeled",
                    "L6": "blocked_until_L5_frozen",
                },
                "review_status": "unlabeled",
            }
            stream.write(json.dumps(task, ensure_ascii=False) + "\n")
    held_out_anchors = [
        anchor["source_index"]
        for anchor in anchors
        if anchor["split"] == "held_out"
    ]
    non_held_dual = []
    for split in ("calibration", "development", "stress"):
        candidates = [
            anchor["source_index"]
            for anchor in anchors
            if anchor["split"] == split
        ]
        needed = max(1, math.ceil(len(candidates) * 0.20))
        non_held_dual.extend(candidates[:needed])
    write_json(
        workspace / "ground_truth/annotations/review_assignment.json",
        {
            "schema": SCHEMA,
            "status": "pending_assignment_to_distinct_humans",
            "dual_review_source_indices": sorted(
                held_out_anchors + non_held_dual
            ),
            "held_out_anchor_dual_review_fraction": 1.0,
            "other_anchor_minimum_dual_review_fraction": 0.20,
            "annotator_a": None,
            "annotator_b": None,
            "adjudicator": None,
        },
    )
    write_json(
        workspace / "ground_truth/held_out_seal_manifest.json",
        {
            "schema": SCHEMA,
            "status": "pending_independent_reviewer_seal",
            "split": "held_out",
            "frames": [558, 573],
            "core": [563, 573],
            "content_path": None,
            "content_sha256": None,
            "visible_to_tuning_operator": False,
        },
    )
    write_json(
        workspace / "ground_truth/quality_report.json",
        {
            "schema": SCHEMA,
            "status": "not_reviewed",
            "layers": {f"L{index}": "incomplete" for index in range(7)},
            "reviewed_gt_available": False,
            "dual_annotation_complete": False,
            "adjudication_complete": False,
            "gt_pilot_complete": False,
            "formal_semantic_or_dsg_ranking_allowed": False,
        },
    )

    unresolved = [
        check for check in checks if not check["passed"]
    ]
    unresolved_input = [
        check
        for check in unresolved
        if check["failure_code"] == "F-INPUT"
    ]
    p0_status = "complete" if not unresolved_input else "stopped_f_input"
    p1_blockers = [
        "L0-L6 reviewed GT is absent",
        "manual challenge tags are not reviewed/frozen",
        "25 anchor proposal is not human-locked",
        "dual annotation and third-person adjudication are incomplete",
        "held-out GT/query seal has not been created by an independent reviewer",
        "scale_473_487.json with calibration-only provenance is absent",
        "GT-dependent gates have not been frozen by a GT pilot",
        "48-query set cannot be generated before reviewed L5 GT",
    ]
    dashboard = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "P0" if unresolved_input else "P1",
        "p0_status": p0_status,
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_failed": len(unresolved),
        "unresolved_f_input": unresolved_input,
        "other_failures": [
            check
            for check in unresolved
            if check["failure_code"] != "F-INPUT"
        ],
        "stereo_gate": stereo_gate,
        "p1_status": "blocked_missing_reviewed_ground_truth",
        "p1_blockers": p1_blockers,
        "formal_matrix_execution_allowed": False,
        "semantic_dsg_ranking_allowed": False,
        "checks": checks,
    }
    dashboard_path = workspace / "manifests/p0_integrity_dashboard.json"
    write_json(dashboard_path, dashboard)
    rows_html = "".join(
        "<tr>"
        f"<td>{escape(check['name'])}</td>"
        f"<td>{'PASS' if check['passed'] else 'FAIL'}</td>"
        f"<td>{escape(check['failure_code'])}</td>"
        f"<td><pre>{escape(json.dumps(check['detail'], ensure_ascii=False))}</pre></td>"
        "</tr>"
        for check in checks
    )
    (workspace / "manifests/p0_integrity_dashboard.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>G1 V1.1 P0</title>"
        "<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:6px;vertical-align:top}"
        "pre{white-space:pre-wrap;max-width:800px}</style>"
        f"<h1>G1 473–573 V1.1 — {escape(p0_status)}</h1>"
        f"<p>Passed {dashboard['checks_passed']}; failed "
        f"{dashboard['checks_failed']}.</p>"
        "<table><tr><th>Check</th><th>Status</th><th>Code</th>"
        f"<th>Detail</th></tr>{rows_html}</table>",
        encoding="utf-8",
    )
    decision = {
        "schema": SCHEMA,
        "status": (
            "blocked_at_p0_f_input"
            if unresolved_input
            else "blocked_at_p1_missing_reviewed_gt"
        ),
        "formal_experiments_run": 0,
        "rankings_produced": False,
        "held_out_opened": False,
        "reason": unresolved_input if unresolved_input else p1_blockers,
        "next_permitted_action": (
            "resolve F-INPUT and rerun P0 in a fresh versioned workspace"
            if unresolved_input
            else "complete independent human GT review, adjudication, pilot, and sealing"
        ),
    }
    write_json(workspace / "final/decision.json", decision)
    report_lines = [
        "# G1 473–573 V1.1 strict execution report",
        "",
        f"- P0: `{p0_status}`",
        "- P1: `blocked_missing_reviewed_ground_truth`",
        "- Formal matrix runs: `0`",
        "- Semantic/DSG rankings: `not allowed`",
        "- Held-out GT/query opened: `no`",
        "",
        "The protocol was stopped at its mandatory gate. Automatic masks, prior "
        "full-run artifacts, and candidate outputs were not promoted to GT.",
        "",
        "## P1 blockers",
        "",
        *[f"- {blocker}" for blocker in p1_blockers],
    ]
    (workspace / "final/report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    (workspace / "final/report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>G1 V1.1 report</title>"
        "<h1>G1 473–573 V1.1 strict execution report</h1>"
        f"<p>P0: {escape(p0_status)}. P1: blocked missing reviewed GT.</p>"
        "<p>No formal rankings were produced and held-out GT was not opened.</p>",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "workspace": str(workspace),
                "p0_status": p0_status,
                "p1_status": dashboard["p1_status"],
                "checks_passed": dashboard["checks_passed"],
                "checks_failed": dashboard["checks_failed"],
                "stereo_gate": stereo_gate,
                "decision": decision["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 2 if unresolved_input else 3


if __name__ == "__main__":
    raise SystemExit(main())
