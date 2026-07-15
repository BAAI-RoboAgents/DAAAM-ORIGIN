#!/usr/bin/env python3
"""Run stereo data through validated geometry and DAAAM/Hydra mapping.

FoundationStereo is intentionally launched in a separate Conda environment so
its Torch and CUDA dependencies never need to coexist with the DAAAM runtime.
Every dataset-producing stage preserves the original absolute capture times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREPARE_G1 = REPOSITORY_ROOT / "scripts" / "prepare_g1_pinhole_stereo_dataset.py"
SELECT_KEYFRAMES = REPOSITORY_ROOT / "scripts" / "select_mapping_keyframes.py"
RUN_DEPTH = REPOSITORY_ROOT / "scripts" / "run_foundation_stereo_depth.py"
APPLY_FLOOR_CALIBRATION = REPOSITORY_ROOT / "scripts" / "apply_g1_floor_calibration.py"
CHECK_TEMPORAL_DEPTH = (
    REPOSITORY_ROOT / "scripts" / "diagnose_temporal_depth_consistency.py"
)
REFINE_RGBD_TRAJECTORY = REPOSITORY_ROOT / "scripts" / "refine_rgbd_trajectory.py"
DISCOVER_RGBD_LOOPS = REPOSITORY_ROOT / "scripts" / "discover_rgbd_loop_closures.py"
OPTIMIZE_RGBD_GRAPH = REPOSITORY_ROOT / "scripts" / "optimize_rgbd_pose_graph.py"
FILTER_TEMPORAL_DEPTH = (
    REPOSITORY_ROOT / "scripts" / "filter_temporal_depth_consistency.py"
)
DIAGNOSE_RGBD_FUSION = REPOSITORY_ROOT / "scripts" / "diagnose_rgbd_fusion.py"
RUN_PIPELINE = REPOSITORY_ROOT / "scripts" / "run_pipeline.py"
DEFAULT_FS_ROOT = REPOSITORY_ROOT / "third_party" / "FoundationStereo"
STAGES = (
    "prepare",
    "select",
    "depth",
    "calibrate",
    "temporal",
    "odometry",
    "loops",
    "optimize",
    "filter",
    "validate",
    "fuse",
    "map",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run G1 fish-eye or prepared stereo input through time-safe selection, "
            "FoundationStereo, geometric validation, and DAAAM/Hydra."
        )
    )
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--adapter",
        choices=("g1-fisheye", "prepared-stereo"),
        default="g1-fisheye",
        help="Input adapter. prepared-stereo must already satisfy the time contract.",
    )
    parser.add_argument("--stop-after", choices=STAGES, default="map")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    g1 = parser.add_argument_group("G1 fish-eye preparation")
    g1.add_argument("--sequence", default="000000")
    g1.add_argument("--max-delta-ms", type=float, default=10.0)
    g1.add_argument("--horizontal-fov-deg", type=float, default=100.0)
    g1.add_argument("--down-fov-deg", type=float, default=28.0)
    g1.add_argument("--rectification-roll-deg", type=float, default=0.0)
    g1.add_argument(
        "--camera-quaternion-order", choices=("auto", "xyzw", "wxyz"), default="auto"
    )
    g1.add_argument("--recommended-max-depth-m", type=float, default=3.0)

    selection = parser.add_argument_group("Content-safe keyframe selection")
    selection.add_argument("--soft-translation-m", type=float, default=0.06)
    selection.add_argument("--soft-rotation-deg", type=float, default=5.0)
    selection.add_argument("--hard-translation-m", type=float, default=0.15)
    selection.add_argument("--hard-rotation-deg", type=float, default=12.0)
    selection.add_argument("--max-gap-s", type=float, default=1.5)

    depth = parser.add_argument_group("FoundationStereo depth")
    depth.add_argument("--foundation-stereo-env", default="foundation_stereo")
    depth.add_argument("--foundation-stereo-root", type=Path, default=DEFAULT_FS_ROOT)
    depth.add_argument(
        "--checkpoint",
        type=Path,
        default=os.environ.get("FOUNDATION_STEREO_CHECKPOINT"),
    )
    depth.add_argument("--valid-iters", type=int, default=32)
    depth.add_argument("--max-depth-m", type=float)
    depth.add_argument("--swap-stereo", action="store_true")
    depth.add_argument(
        "--accept-foundation-stereo-noncommercial-license",
        action="store_true",
        help="Required before invoking the NVIDIA research/non-commercial backend.",
    )

    geometry = parser.add_argument_group("Geometry and RGB-D validation")
    geometry.add_argument(
        "--floor-calibration-report",
        type=Path,
        help=(
            "Validated fixed G1 floor/image-frame calibration. Required for a "
            "g1-fisheye run that proceeds beyond nominal depth inference."
        ),
    )
    geometry.add_argument("--geometry-max-depth-m", type=float, default=3.0)
    geometry.add_argument("--local-keyframe-distance-m", type=float, default=0.10)
    geometry.add_argument("--local-max-keyframe-gap", type=int, default=8)
    geometry.add_argument("--local-neighbor-span", type=int, default=6)
    geometry.add_argument("--local-min-inliers", type=int, default=80)
    geometry.add_argument("--local-visual-max-nfev", type=int, default=150)
    geometry.add_argument("--loop-dense-candidate-count", type=int, default=80)
    geometry.add_argument("--max-loop-gravity-residual-deg", type=float, default=8.0)
    geometry.add_argument("--global-iterations", type=int, default=250)

    consistency = parser.add_argument_group("Temporal depth and fusion gates")
    consistency.add_argument("--temporal-filter-neighbor-offsets", default="1,2,3")
    consistency.add_argument("--temporal-filter-scale", type=float, default=0.5)
    consistency.add_argument("--temporal-filter-min-judged", type=int, default=3)
    consistency.add_argument("--temporal-filter-min-support", type=float, default=0.5)
    consistency.add_argument(
        "--final-min-adjacent-agreement", type=float, default=0.85
    )
    consistency.add_argument(
        "--final-max-adjacent-median-error-m", type=float, default=0.035
    )
    consistency.add_argument(
        "--final-min-window-adjacent-agreement", type=float, default=0.80
    )
    consistency.add_argument("--final-window-size-frames", type=int, default=100)
    consistency.add_argument("--fusion-frame-step", type=int, default=10)
    consistency.add_argument("--fusion-pixel-step", type=int, default=8)
    consistency.add_argument("--fusion-voxel-size-m", type=float, default=0.035)
    consistency.add_argument(
        "--accept-direct-fusion-preview",
        action="store_true",
        help=(
            "Confirm that the generated direct RGB-D preview has coherent floor, "
            "walls, and objects before Hydra is allowed to run."
        ),
    )

    mapping = parser.add_argument_group("DAAAM/Hydra mapping")
    mapping.add_argument("--pipeline-config", type=Path)
    mapping.add_argument("--hydra-config-path", type=Path)
    mapping.add_argument("--labelspace-path", type=Path)
    mapping.add_argument("--labelspace-colors", type=Path)
    mapping.add_argument("--depth-lb", type=float, default=0.25)
    mapping.add_argument("--depth-ub", type=float, default=3.0)
    mapping.add_argument("--fps", type=float, default=10.0)
    mapping.add_argument("--target-fps", type=float)
    mapping.add_argument("--query-interval-frames", type=int, default=90)
    mapping.add_argument("--dataset-name")
    mapping.add_argument("--zmq-url", default="none")
    return parser.parse_args()


def stage_enabled(stop_after: str, stage: str) -> bool:
    return STAGES.index(stage) <= STAGES.index(stop_after)


def run_command(
    command: list[str], dry_run: bool, execute_in_dry_run: bool = False
) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run or execute_in_dry_run:
        subprocess.run(command, check=True)


def submodule_commit(path: Path) -> str | None:
    if not path.is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frame_count(dataset: Path) -> int:
    return len(json.loads((dataset / "tick_index.json").read_text())["frames"])


def validate_time_contract(dataset: Path) -> dict[str, Any]:
    """Validate the absolute image-to-pose contract before orchestration."""
    tick_path = dataset / "tick_index.json"
    pose_path = dataset / "pose" / "poses.txt"
    pose_time_path = dataset / "pose" / "pose_timestamps_ns.txt"
    try:
        metadata = json.loads(tick_path.read_text())
        frames = metadata["frames"]
        time_origin_ns = int(metadata["time_origin_ns"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Prepared dataset does not provide a valid absolute time contract: {dataset}"
        ) from error
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"Prepared dataset has no frames: {dataset}")

    pose_rows = [line for line in pose_path.read_text().splitlines() if line.strip()]
    try:
        pose_timestamps = [
            int(line.strip()) for line in pose_time_path.read_text().splitlines() if line.strip()
        ]
    except ValueError as error:
        raise RuntimeError(f"Invalid pose timestamp index: {pose_time_path}") from error
    if len(pose_rows) != len(pose_timestamps) or len(frames) != len(pose_rows):
        raise RuntimeError(
            "Frame, pose, and absolute pose timestamp counts must agree before mapping"
        )
    if any(
        second <= first for first, second in zip(pose_timestamps, pose_timestamps[1:])
    ):
        raise RuntimeError("Absolute pose timestamps must be strictly increasing")

    checked_pose_rows: set[int] = set()
    previous_sensor_time: int | None = None
    for index, frame in enumerate(frames):
        try:
            frame_index = int(frame["idx"])
            sensor_time_ns = int(frame["sensor_time_ns"])
            cam0_time_ns = int(frame["cam0_sensor_time_ns"])
            cam1_time_ns = int(frame["cam1_sensor_time_ns"])
            pose_time_ns = int(frame["pose_sensor_time_ns"])
            pose_row = int(frame["pose_row"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Frame {index} is missing required absolute-time metadata"
            ) from error
        if frame_index != index:
            raise RuntimeError(f"Frame index mismatch at {index}: {frame_index}")
        if "source_idx" not in frame and "source_frame_idx" not in frame:
            raise RuntimeError(f"Frame {index} is missing source image provenance")
        if pose_row < 0 or pose_row >= len(pose_timestamps) or pose_row in checked_pose_rows:
            raise RuntimeError(f"Frame {index} has invalid or duplicate pose_row {pose_row}")
        checked_pose_rows.add(pose_row)
        if sensor_time_ns != cam0_time_ns or pose_time_ns != cam0_time_ns:
            raise RuntimeError(f"Frame {index} camera and pose absolute times disagree")
        if pose_timestamps[pose_row] != pose_time_ns:
            raise RuntimeError(f"Frame {index} pose timestamp does not match pose_row")
        if previous_sensor_time is not None and sensor_time_ns <= previous_sensor_time:
            raise RuntimeError("Frame sensor_time_ns values must be strictly increasing")
        previous_sensor_time = sensor_time_ns
        expected_timestamp = (sensor_time_ns - time_origin_ns) / 1.0e9
        if "timestamp" not in frame or not math.isclose(
            float(frame["timestamp"]), expected_timestamp, abs_tol=1.0e-6
        ):
            raise RuntimeError(
                f"Frame {index} relative timestamp is not derived from sensor_time_ns"
            )
        expected_stereo_delta_ms = abs(cam0_time_ns - cam1_time_ns) / 1.0e6
        if "stereo_delta_ms" in frame and not math.isclose(
            float(frame["stereo_delta_ms"]), expected_stereo_delta_ms, abs_tol=1.0e-6
        ):
            raise RuntimeError(f"Frame {index} stereo_delta_ms disagrees with capture times")

    return {
        "valid": True,
        "frame_count": len(frames),
        "time_origin_ns": time_origin_ns,
        "pose_timestamp_file": str(pose_time_path),
        "checked": [
            "cam0_sensor_time_ns == sensor_time_ns == pose_sensor_time_ns",
            "pose_sensor_time_ns == pose_timestamps_ns[pose_row]",
            "strictly_increasing_sensor_time_ns",
            "relative_timestamp_derived_from_time_origin_ns",
            "stereo_delta_ms_derived_from_cam0_cam1_absolute_times",
        ],
    }


def selection_manifest(report_path: Path) -> dict[str, Any]:
    """Extract the immutable selection settings and outcome for the run manifest."""
    report = json.loads(report_path.read_text())
    return {
        "method": report.get("method"),
        "report": str(report_path),
        "config": report.get("config"),
        "source_frame_count": report.get("source_frame_count"),
        "selected_frame_count": report.get("selected_frame_count"),
        "reduction_ratio": report.get("reduction_ratio"),
        "selection_reasons": report.get("selection_reasons"),
    }


def depth_manifest(report_path: Path) -> dict[str, Any]:
    """Preserve the depth backend inputs when resuming an existing run."""
    report = json.loads(report_path.read_text())
    manifest = {
        key: report[key]
        for key in (
            "checkpoint",
            "foundation_stereo_root",
            "valid_iters",
            "max_depth_m",
            "swap_stereo",
        )
        if key in report
    }
    checkpoint = Path(manifest["checkpoint"]) if "checkpoint" in manifest else None
    if checkpoint is not None and checkpoint.is_file():
        manifest["checkpoint_sha256"] = sha256(checkpoint)
    return manifest


def prepared_ready(dataset: Path) -> bool:
    return (
        (dataset / "tick_index.json").is_file()
        and (dataset / "pose" / "poses.txt").is_file()
        and (dataset / "pose" / "pose_timestamps_ns.txt").is_file()
    )


def selected_ready(dataset: Path) -> bool:
    return prepared_ready(dataset) and (dataset / "keyframe_selection_report.json").is_file()


def depth_ready(dataset: Path) -> bool:
    if not selected_ready(dataset):
        return False
    return len(list((dataset / "depth").glob("*.png"))) == load_frame_count(dataset)


def calibrated_ready(dataset: Path) -> bool:
    return (
        prepared_ready(dataset)
        and depth_ready_without_selection_report(dataset)
        and (dataset / "floor_calibration_application.json").is_file()
    )


def depth_ready_without_selection_report(dataset: Path) -> bool:
    if not prepared_ready(dataset):
        return False
    return len(list((dataset / "depth").glob("*.png"))) == load_frame_count(dataset)


def report_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def trajectory_ready(dataset: Path, report_name: str) -> bool:
    return prepared_ready(dataset) and report_ready(dataset / report_name)


def map_ready(output: Path) -> bool:
    return any(output.glob("out_*/**/dsg_with_mesh.json")) and any(
        output.glob("out_*/**/mesh.ply")
    )


def run_or_resume(
    stage: str,
    command: list[str],
    ready: bool,
    args: argparse.Namespace,
) -> str:
    if args.resume and ready:
        print(f"Resume: stage {stage} is already complete", flush=True)
        return "resumed"
    run_command(command, args.dry_run)
    return "planned" if args.dry_run else "executed"


def require_verified_loop(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    verified_count = int(report.get("verified_count", 0))
    if verified_count < 1:
        raise RuntimeError(
            "No geometrically verified loop closure was found; global optimization "
            "and Hydra mapping are intentionally blocked."
        )
    return {
        "report": str(report_path),
        "verified_count": verified_count,
        "dense_tested_count": int(report.get("dense_tested_count", 0)),
    }


def load_gate_summary(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    gate = report.get("pre_hydra_gate", {})
    if not gate.get("passed", False):
        raise RuntimeError(f"Temporal depth gate did not pass: {report_path}")
    return {
        "report": str(report_path),
        "overall_agreement_rate_weighted": report.get(
            "overall_agreement_rate_weighted"
        ),
        "gate": gate,
        "absolute_time_contract": report.get("absolute_time_contract"),
    }


def load_global_optimization_summary(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    optimization = report.get("optimization", {})
    if optimization.get("success") is False:
        raise RuntimeError(f"Global pose graph did not converge: {report_path}")
    loops = report.get("selected_verified_loops", [])
    if not loops:
        raise RuntimeError(f"Global pose graph contains no verified loop: {report_path}")
    return {
        "report": str(report_path),
        "optimization": optimization,
        "selected_verified_loop_count": len(loops),
        "source_path_length_m": report.get("source_path_length_m"),
        "optimized_path_length_m": report.get("optimized_path_length_m"),
    }


def write_manifest(path: Path, manifest: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def selection_command(args: argparse.Namespace, dataset: Path, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(SELECT_KEYFRAMES),
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--soft-translation-m",
        str(args.soft_translation_m),
        "--soft-rotation-deg",
        str(args.soft_rotation_deg),
        "--hard-translation-m",
        str(args.hard_translation_m),
        "--hard-rotation-deg",
        str(args.hard_rotation_deg),
        "--max-gap-s",
        str(args.max_gap_s),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> None:
    args = parse_args()
    source = args.src.resolve()
    run_dir = args.run_dir.resolve()
    prepared = run_dir / "01_pinhole"
    selected = run_dir / "02_selected"
    geometry = run_dir / "03_geometry"
    temporal_output = run_dir / "04_temporal_input"
    rgbd_odometry = run_dir / "05_rgbd_window_graph"
    loop_output = run_dir / "06_loop_closures"
    optimized = run_dir / "07_global_pose_graph"
    final_dataset = run_dir / "08_temporal_depth_filtered"
    validation_output = run_dir / "09_temporal_validation"
    fusion_output = run_dir / "10_direct_rgbd_fusion"
    map_output = run_dir / "11_daaam"
    manifest: dict[str, Any] = {
        "source": str(source),
        "adapter": args.adapter,
        "run_dir": str(run_dir),
        "stages_completed": [],
        "stage_results": {},
        "foundation_stereo": {
            "root": str(args.foundation_stereo_root.resolve()),
            "submodule_commit": submodule_commit(args.foundation_stereo_root.resolve()),
            "environment": args.foundation_stereo_env,
            "license": "NVIDIA FoundationStereo research/non-commercial",
        },
        "time_contract": {
            "ordering": "sensor_time_ns",
            "pose_match": "cam0_sensor_time_ns == pose_sensor_time_ns",
            "filtered_frames_retimestamped": False,
        },
    }

    def finish(stage: str) -> bool:
        if args.stop_after != stage:
            return False
        manifest["status"] = "planned" if args.dry_run else "complete"
        write_manifest(run_dir / "mapping_run.json", manifest, args.dry_run)
        return True

    def record(stage: str, result: str) -> None:
        manifest["stages_completed"].append(stage)
        manifest["stage_results"][stage] = result
        if not args.dry_run:
            manifest["status"] = "running"
            write_manifest(run_dir / "mapping_run.json", manifest, False)

    if args.adapter == "prepared-stereo":
        prepared = source
        manifest["prepared_dataset"] = str(prepared)
        manifest["stage_results"]["prepare"] = "adapter_input"
    else:
        command = [
            sys.executable,
            str(PREPARE_G1),
            "--src",
            str(source),
            "--output",
            str(prepared),
            "--sequence",
            args.sequence,
            "--max-delta-ms",
            str(args.max_delta_ms),
            "--horizontal-fov-deg",
            str(args.horizontal_fov_deg),
            "--down-fov-deg",
            str(args.down_fov_deg),
            "--rectification-roll-deg",
            str(args.rectification_roll_deg),
            "--camera-quaternion-order",
            args.camera_quaternion_order,
            "--recommended-max-depth-m",
            str(args.recommended_max_depth_m),
        ]
        if args.overwrite:
            command.append("--overwrite")
        result = run_or_resume(
            "prepare", command, prepared_ready(prepared), args
        )
        manifest["prepared_dataset"] = str(prepared)
        record("prepare", result)

    if not args.dry_run and not prepared_ready(prepared):
        raise RuntimeError(f"Prepared dataset does not satisfy the time contract: {prepared}")
    if prepared_ready(prepared):
        manifest["prepared_time_contract"] = validate_time_contract(prepared)
    if finish("prepare"):
        return

    command = selection_command(args, prepared, selected)
    result = run_or_resume("select", command, selected_ready(selected), args)
    manifest["selected_dataset"] = str(selected)
    record("select", result)
    if args.dry_run:
        manifest["keyframe_selection"] = {
            "method": "absolute_time_pose_visual_content_safe",
            "report": str(selected / "keyframe_selection_report.json"),
            "requested_pose_config": {
                "soft_translation_m": args.soft_translation_m,
                "soft_rotation_deg": args.soft_rotation_deg,
                "hard_translation_m": args.hard_translation_m,
                "hard_rotation_deg": args.hard_rotation_deg,
                "max_gap_s": args.max_gap_s,
            },
        }
    else:
        if not selected_ready(selected):
            raise RuntimeError(f"Selected dataset is not available: {selected}")
        manifest["keyframe_selection"] = selection_manifest(
            selected / "keyframe_selection_report.json"
        )
        manifest["selected_time_contract"] = validate_time_contract(selected)
    if finish("select"):
        return

    depth_report = selected / "foundation_stereo_run.json"
    depth_is_ready = depth_ready(selected)
    if args.resume and depth_is_ready:
        result = "resumed"
        print(f"Resume: stage depth is already complete in {selected}", flush=True)
    else:
        if not args.dry_run:
            if not args.accept_foundation_stereo_noncommercial_license:
                raise RuntimeError(
                    "Pass --accept-foundation-stereo-noncommercial-license to run "
                    "the FoundationStereo research/non-commercial backend."
                )
            if args.checkpoint is None:
                raise RuntimeError(
                    "Pass --checkpoint or set FOUNDATION_STEREO_CHECKPOINT before depth inference."
                )
            checkpoint = args.checkpoint.expanduser().resolve()
            if not checkpoint.is_file():
                raise RuntimeError(f"Checkpoint is missing: {checkpoint}")
            if shutil.which("conda") is None:
                raise RuntimeError("conda is required to run FoundationStereo in its own environment")
        else:
            checkpoint = (
                args.checkpoint.expanduser().resolve()
                if args.checkpoint is not None
                else Path("<FOUNDATION_STEREO_CHECKPOINT>")
            )
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            args.foundation_stereo_env,
            "python",
            str(RUN_DEPTH),
            "--dataset",
            str(selected),
            "--fs-root",
            str(args.foundation_stereo_root.resolve()),
            "--checkpoint",
            str(checkpoint),
            "--valid-iters",
            str(args.valid_iters),
        ]
        if args.max_depth_m is not None:
            command.extend(["--max-depth-m", str(args.max_depth_m)])
        if args.swap_stereo:
            command.append("--swap-stereo")
        if args.overwrite:
            command.append("--overwrite")
        run_command(command, args.dry_run)
        result = "planned" if args.dry_run else "executed"
        if not args.dry_run:
            manifest["foundation_stereo"].update(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "valid_iters": args.valid_iters,
                }
            )
    if depth_report.is_file():
        manifest["foundation_stereo"].update(depth_manifest(depth_report))
    record("depth", result)
    if finish("depth"):
        return

    calibration_report = (
        args.floor_calibration_report.expanduser().resolve()
        if args.floor_calibration_report is not None
        else None
    )
    if calibration_report is None and args.adapter == "g1-fisheye":
        raise RuntimeError(
            "A validated --floor-calibration-report is required before G1 trajectory "
            "refinement and mapping. Stop after depth to create one first."
        )
    if calibration_report is not None:
        if not args.dry_run and not calibration_report.is_file():
            raise FileNotFoundError(f"Floor calibration report is missing: {calibration_report}")
        command = [
            sys.executable,
            str(APPLY_FLOOR_CALIBRATION),
            "--dataset",
            str(selected),
            "--calibration-report",
            str(calibration_report),
            "--output",
            str(geometry),
            "--max-depth-m",
            str(args.geometry_max_depth_m),
        ]
        if args.overwrite:
            command.append("--overwrite")
        result = run_or_resume(
            "calibrate", command, calibrated_ready(geometry), args
        )
    else:
        geometry = selected
        result = "not_required_for_prepared_stereo"
    manifest["geometry_dataset"] = str(geometry)
    manifest["floor_calibration_report"] = (
        str(calibration_report) if calibration_report is not None else None
    )
    record("calibrate", result)
    if not args.dry_run:
        validate_time_contract(geometry)
    if finish("calibrate"):
        return

    temporal_report = temporal_output / "temporal_depth_consistency_report.json"
    command = [
        sys.executable,
        str(CHECK_TEMPORAL_DEPTH),
        "--dataset",
        str(geometry),
        "--output-dir",
        str(temporal_output),
        "--frame-step",
        "1",
        "--neighbor-offsets",
        "1",
        "--pixel-step",
        "4",
        "--max-depth-m",
        str(args.geometry_max_depth_m),
        "--forward-only",
        "--require-time-contract",
    ]
    result = run_or_resume(
        "temporal", command, report_ready(temporal_report), args
    )
    manifest["input_temporal_report"] = str(temporal_report)
    record("temporal", result)
    if finish("temporal"):
        return

    command = [
        sys.executable,
        str(REFINE_RGBD_TRAJECTORY),
        "--dataset",
        str(geometry),
        "--output",
        str(rgbd_odometry),
        "--mode",
        "pose-graph-3d",
        "--keyframe-distance-m",
        str(args.local_keyframe_distance_m),
        "--max-keyframe-gap",
        str(args.local_max_keyframe_gap),
        "--local-neighbor-span",
        str(args.local_neighbor_span),
        "--min-inliers",
        str(args.local_min_inliers),
        "--visual-max-nfev",
        str(args.local_visual_max_nfev),
        "--max-depth-m",
        str(args.geometry_max_depth_m),
    ]
    if args.overwrite:
        command.append("--overwrite")
    result = run_or_resume(
        "odometry",
        command,
        trajectory_ready(rgbd_odometry, "trajectory_refinement.json"),
        args,
    )
    manifest["rgbd_odometry_dataset"] = str(rgbd_odometry)
    record("odometry", result)
    if not args.dry_run:
        validate_time_contract(rgbd_odometry)
    if finish("odometry"):
        return

    loop_report = loop_output / "loop_closure_report.json"
    command = [
        sys.executable,
        str(DISCOVER_RGBD_LOOPS),
        "--dataset",
        str(geometry),
        "--output-dir",
        str(loop_output),
        "--keyframe-distance-m",
        str(args.local_keyframe_distance_m),
        "--max-keyframe-gap",
        "10",
        "--dense-candidate-count",
        str(args.loop_dense_candidate_count),
        "--max-depth-m",
        str(args.geometry_max_depth_m),
    ]
    result = run_or_resume("loops", command, report_ready(loop_report), args)
    manifest["loop_closures"] = (
        {"report": str(loop_report), "status": "planned"}
        if args.dry_run
        else require_verified_loop(loop_report)
    )
    record("loops", result)
    if finish("loops"):
        return

    command = [
        sys.executable,
        str(OPTIMIZE_RGBD_GRAPH),
        "--dataset",
        str(geometry),
        "--rgbd-odometry-dataset",
        str(rgbd_odometry),
        "--temporal-report",
        str(temporal_report),
        "--loop-report",
        str(loop_report),
        "--output",
        str(optimized),
        "--robot-translation-sigma-m",
        "0.08",
        "--robot-rotation-sigma-deg",
        "8.0",
        "--rgbd-translation-sigma-m",
        "0.04",
        "--rgbd-rotation-sigma-deg",
        "1.5",
        "--loop-translation-sigma-m",
        "0.04",
        "--loop-rotation-sigma-deg",
        "1.5",
        "--max-loop-gravity-residual-deg",
        str(args.max_loop_gravity_residual_deg),
        "--optimizer-mode",
        "gravity-se3",
        "--iterations",
        str(args.global_iterations),
    ]
    if args.overwrite:
        command.append("--overwrite")
    result = run_or_resume(
        "optimize",
        command,
        trajectory_ready(optimized, "global_pose_graph_report.json"),
        args,
    )
    manifest["optimized_dataset"] = str(optimized)
    record("optimize", result)
    if not args.dry_run:
        validate_time_contract(optimized)
        manifest["global_pose_graph"] = load_global_optimization_summary(
            optimized / "global_pose_graph_report.json"
        )
    if finish("optimize"):
        return

    command = [
        sys.executable,
        str(FILTER_TEMPORAL_DEPTH),
        "--dataset",
        str(optimized),
        "--output",
        str(final_dataset),
        "--neighbor-offsets",
        args.temporal_filter_neighbor_offsets,
        "--filter-scale",
        str(args.temporal_filter_scale),
        "--min-judged-neighbors",
        str(args.temporal_filter_min_judged),
        "--min-support-ratio",
        str(args.temporal_filter_min_support),
        "--max-depth-m",
        str(args.geometry_max_depth_m),
    ]
    if args.overwrite:
        command.append("--overwrite")
    result = run_or_resume(
        "filter",
        command,
        trajectory_ready(final_dataset, "temporal_depth_filter_report.json"),
        args,
    )
    manifest["final_dataset"] = str(final_dataset)
    record("filter", result)
    if not args.dry_run:
        manifest["final_time_contract"] = validate_time_contract(final_dataset)
    if finish("filter"):
        return

    final_temporal_report = (
        validation_output / "temporal_depth_consistency_report.json"
    )
    command = [
        sys.executable,
        str(CHECK_TEMPORAL_DEPTH),
        "--dataset",
        str(final_dataset),
        "--output-dir",
        str(validation_output),
        "--frame-step",
        "1",
        "--neighbor-offsets",
        "1",
        "--pixel-step",
        "4",
        "--max-depth-m",
        str(args.geometry_max_depth_m),
        "--forward-only",
        "--require-time-contract",
        "--window-size-frames",
        str(args.final_window_size_frames),
        "--fail-below-adjacent-agreement-rate",
        str(args.final_min_adjacent_agreement),
        "--fail-above-adjacent-median-error-m",
        str(args.final_max_adjacent_median_error_m),
        "--fail-below-window-adjacent-agreement-rate",
        str(args.final_min_window_adjacent_agreement),
    ]
    result = run_or_resume(
        "validate", command, report_ready(final_temporal_report), args
    )
    manifest["temporal_validation"] = (
        {"report": str(final_temporal_report), "status": "planned"}
        if args.dry_run
        else load_gate_summary(final_temporal_report)
    )
    record("validate", result)
    if finish("validate"):
        return

    fusion_report = fusion_output / "direct_rgbd_fusion_report.json"
    command = [
        sys.executable,
        str(DIAGNOSE_RGBD_FUSION),
        "--dataset",
        str(final_dataset),
        "--output-dir",
        str(fusion_output),
        "--frame-step",
        str(args.fusion_frame_step),
        "--pixel-step",
        str(args.fusion_pixel_step),
        "--max-depth-m",
        str(args.geometry_max_depth_m),
        "--voxel-size-m",
        str(args.fusion_voxel_size_m),
    ]
    result = run_or_resume("fuse", command, report_ready(fusion_report), args)
    if not args.dry_run and not (
        report_ready(fusion_report)
        and (fusion_output / "direct_rgbd_fusion_preview.png").is_file()
    ):
        raise RuntimeError(f"Direct RGB-D fusion outputs are incomplete: {fusion_output}")
    manifest["direct_rgbd_fusion"] = {
        "report": str(fusion_report),
        "preview": str(fusion_output / "direct_rgbd_fusion_preview.png"),
        "manually_accepted": bool(args.accept_direct_fusion_preview),
    }
    record("fuse", result)
    if finish("fuse"):
        return

    if args.hydra_config_path is None:
        raise RuntimeError("--hydra-config-path is required when --stop-after map")
    if not args.dry_run and not args.accept_direct_fusion_preview:
        raise RuntimeError(
            "Hydra is blocked until the direct RGB-D preview is inspected. Re-run "
            "with --resume --accept-direct-fusion-preview after confirming coherent "
            f"geometry: {fusion_output / 'direct_rgbd_fusion_preview.png'}"
        )
    command = [
        sys.executable,
        str(RUN_PIPELINE),
        str(final_dataset),
        "--dataset-type",
        "ImageSequenceDataset",
        "--depth-scale",
        "1000",
        "--depth-lb",
        str(args.depth_lb),
        "--depth-ub",
        str(args.depth_ub),
        "--fps",
        str(args.fps),
        "--query-interval-frames",
        str(args.query_interval_frames),
        "--hydra-config-path",
        str(args.hydra_config_path),
        "--output-dir",
        str(map_output),
        "--dataset-name",
        args.dataset_name or run_dir.name,
        "--zmq-url",
        args.zmq_url,
        "--no-progress",
    ]
    if args.pipeline_config is not None:
        command.extend(["--config", str(args.pipeline_config)])
    if args.labelspace_path is not None:
        command.extend(["--labelspace-path", str(args.labelspace_path)])
    if args.labelspace_colors is not None:
        command.extend(["--labelspace-colors", str(args.labelspace_colors)])
    if args.target_fps is not None:
        command.extend(["--target-fps", str(args.target_fps)])
    else:
        command.append("--no-throttle")
    result = run_or_resume("map", command, map_ready(map_output), args)
    if not args.dry_run and not map_ready(map_output):
        raise RuntimeError(f"Hydra completed without a DSG and mesh: {map_output}")
    manifest["map_input_dataset"] = str(final_dataset)
    manifest["map_output"] = str(map_output)
    record("map", result)

    manifest["status"] = "planned" if args.dry_run else "complete"
    write_manifest(run_dir / "mapping_run.json", manifest, args.dry_run)
    if not args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
