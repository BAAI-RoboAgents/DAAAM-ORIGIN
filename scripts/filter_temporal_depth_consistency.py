#!/usr/bin/env python3
"""Mask depth pixels contradicted by multiple timestamp-aligned neighbors.

RGB frames and their absolute-time pose mapping are retained unchanged. Only
depth samples that can be tested in enough neighboring views and receive no
geometric support are removed, so unknown and occluded regions remain intact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from diagnose_temporal_depth_consistency import load_poses, load_time_contract


DEPTH_EVIDENCE_DIRECTORIES = (
    "depth_confidence",
    "depth_consistency",
    "depth_occlusion",
    "depth_metadata",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask temporally contradicted depth while preserving every frame."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--neighbor-offsets",
        default="1,2",
        help="Comma-separated positive frame offsets checked in both directions.",
    )
    parser.add_argument("--filter-scale", type=float, default=0.5)
    parser.add_argument("--min-depth-m", type=float, default=0.25)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--absolute-tolerance-m", type=float, default=0.05)
    parser.add_argument("--relative-tolerance", type=float, default=0.04)
    parser.add_argument(
        "--min-judged-neighbors",
        type=int,
        default=2,
        help="Do not reject a pixel unless this many neighbors can judge it.",
    )
    parser.add_argument(
        "--min-support-ratio",
        type=float,
        default=0.25,
        help="Minimum agreeing/judged ratio; zero support is always rejected.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sorted_depths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No depth PNG files in {directory}")
    return paths


def prepare_output(source: Path, output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("rgb", "stereo_right"):
        source_directory = source / directory
        if source_directory.exists():
            (output / directory).symlink_to(
                source_directory.resolve(), target_is_directory=True
            )
    (output / "depth").mkdir()
    for directory in DEPTH_EVIDENCE_DIRECTORIES:
        if (source / directory).is_dir():
            (output / directory).mkdir()
    (output / "pose").mkdir()
    shutil.copy2(source / "pose" / "poses.txt", output / "pose" / "poses.txt")
    shutil.copy2(
        source / "pose" / "pose_timestamps_ns.txt",
        output / "pose" / "pose_timestamps_ns.txt",
    )
    for name in (
        "camera_info.json",
        "source_manifest.json",
        "keyframe_selection_report.json",
        "floor_calibration_application.json",
        "foundation_stereo_run.json",
        "foundation_stereo_nominal_run.json",
        "trajectory_refinement.json",
        "global_pose_graph_report.json",
    ):
        source_path = source / name
        if source_path.exists():
            (output / name).symlink_to(source_path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def propagate_depth_evidence(
    source: Path,
    output: Path,
    *,
    frame_index: int,
    sensor_time_ns: int,
    frame_name: str,
    rejected_mask: np.ndarray,
    output_depth_path: Path,
    output_valid_ratio: float,
    rejected_valid_ratio: float,
) -> dict[str, object]:
    """Copy verifiable stereo evidence while applying the temporal reject mask."""
    propagated: list[str] = []
    for directory in ("depth_confidence", "depth_consistency", "depth_occlusion"):
        source_directory = source / directory
        if not source_directory.is_dir():
            continue
        source_path = source_directory / frame_name
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Incomplete {directory} evidence for frame {frame_index}: {source_path}"
            )
        artifact = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if artifact is None or artifact.shape[:2] != rejected_mask.shape:
            raise ValueError(f"Invalid {directory} evidence: {source_path}")
        artifact = artifact.copy()
        if directory in {"depth_confidence", "depth_consistency"}:
            artifact[rejected_mask] = 0
        destination = output / directory / frame_name
        if not cv2.imwrite(str(destination), artifact):
            raise RuntimeError(f"Failed to write propagated evidence: {destination}")
        propagated.append(directory)

    metadata_directory = source / "depth_metadata"
    left_right_verified = False
    if metadata_directory.is_dir():
        source_metadata_path = metadata_directory / f"{frame_index:08d}.json"
        if not source_metadata_path.is_file():
            raise FileNotFoundError(
                f"Incomplete depth_metadata evidence for frame {frame_index}: "
                f"{source_metadata_path}"
            )
        try:
            metadata = json.loads(source_metadata_path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid depth metadata: {source_metadata_path}") from error
        if int(metadata.get("sensor_time_ns", -1)) != sensor_time_ns:
            raise ValueError(
                f"Depth metadata timestamp mismatch for frame {frame_index}"
            )
        left_right_verified = bool(metadata.get("left_right_verified", False))
        metadata.update(
            {
                "frame_idx": frame_index,
                "sensor_time_ns": sensor_time_ns,
                "valid_ratio": output_valid_ratio,
                "temporal_filter": {
                    "method": "multi_neighbor_reprojection_consistency",
                    "rejected_valid_ratio": rejected_valid_ratio,
                    "source_metadata_sha256": sha256_file(source_metadata_path),
                    "output_depth_sha256": sha256_file(output_depth_path),
                },
            }
        )
        destination = output / "depth_metadata" / f"{frame_index:08d}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)
        propagated.append("depth_metadata")
    return {
        "propagated": propagated,
        "left_right_verified": left_right_verified,
    }


def projection_support(
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    target_t_source: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth_m: float,
    max_depth_m: float,
    absolute_tolerance_m: float,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid_source = (
        (source_depth >= min_depth_m) & (source_depth <= max_depth_m)
    )
    points = np.stack(
        (
            (u - cx) * source_depth / fx,
            (v - cy) * source_depth / fy,
            source_depth,
        ),
        axis=-1,
    )
    transformed = (
        points @ target_t_source[:3, :3].T + target_t_source[:3, 3]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_u = fx * transformed[..., 0] / transformed[..., 2] + cx
        projected_v = fy * transformed[..., 1] / transformed[..., 2] + cy
    height, width = source_depth.shape
    in_bounds = (
        valid_source
        & (transformed[..., 2] > 0.0)
        & (projected_u >= 0.0)
        & (projected_u <= width - 1.0)
        & (projected_v >= 0.0)
        & (projected_v <= height - 1.0)
    )
    sampled_target = cv2.remap(
        target_depth,
        projected_u.astype(np.float32),
        projected_v.astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    target_valid = (
        (sampled_target >= min_depth_m) & (sampled_target <= max_depth_m)
    )
    tolerance = absolute_tolerance_m + relative_tolerance * transformed[..., 2]
    delta = sampled_target - transformed[..., 2]
    # A closer target surface occludes the source sample and cannot contradict it.
    judged = in_bounds & target_valid & (delta >= -tolerance)
    supported = judged & (np.abs(delta) <= tolerance)
    return judged, supported


def main() -> None:
    args = parse_args()
    offsets = sorted(
        {int(value) for value in args.neighbor_offsets.split(",") if value.strip()}
    )
    if not offsets or offsets[0] < 1:
        raise ValueError("Neighbor offsets must be positive")
    if not 0.0 < args.filter_scale <= 1.0:
        raise ValueError("--filter-scale must be in (0, 1]")
    if not 0.0 <= args.min_support_ratio <= 1.0:
        raise ValueError("--min-support-ratio must be in [0, 1]")
    if args.min_judged_neighbors < 1:
        raise ValueError("--min-judged-neighbors must be positive")

    source = args.dataset.resolve()
    output = args.output.resolve()
    depths = sorted_depths(source / "depth")
    poses = load_poses(source / "pose" / "poses.txt")
    if len(depths) != len(poses):
        raise ValueError("Depth and pose counts must agree")
    metadata, frames = load_time_contract(source, len(poses))
    camera = json.loads((source / "camera_info.json").read_text())
    width = max(1, int(round(float(camera["width"]) * args.filter_scale)))
    height = max(1, int(round(float(camera["height"]) * args.filter_scale)))
    fx = float(camera["fx"]) * args.filter_scale
    fy = float(camera["fy"]) * args.filter_scale
    cx = float(camera["cx"]) * args.filter_scale
    cy = float(camera["cy"]) * args.filter_scale
    v, u = np.mgrid[0:height, 0:width]

    @lru_cache(maxsize=2 * max(offsets) + 3)
    def scaled_depth(index: int) -> np.ndarray:
        depth_mm = cv2.imread(str(depths[index]), cv2.IMREAD_UNCHANGED)
        if depth_mm is None or depth_mm.dtype != np.uint16:
            raise ValueError(f"Invalid uint16 depth image: {depths[index]}")
        depth_m = depth_mm.astype(np.float32) / 1000.0
        if depth_m.shape != (height, width):
            depth_m = cv2.resize(
                depth_m, (width, height), interpolation=cv2.INTER_NEAREST
            )
        return depth_m

    prepare_output(source, output, args.overwrite)
    input_valid_ratios = []
    output_valid_ratios = []
    rejected_ratios = []
    per_frame = []
    rejected_pixels = 0
    propagated_evidence_directories: set[str] = set()
    left_right_verified_frames = 0
    for index, depth_path in enumerate(depths):
        source_low = scaled_depth(index)
        judged_count = np.zeros(source_low.shape, dtype=np.uint8)
        support_count = np.zeros(source_low.shape, dtype=np.uint8)
        tested_neighbors = []
        for signed_offset in [
            value for offset in offsets for value in (-offset, offset)
        ]:
            neighbor = index + signed_offset
            if not 0 <= neighbor < len(depths):
                continue
            target_t_source = np.linalg.inv(poses[neighbor]) @ poses[index]
            judged, supported = projection_support(
                source_low,
                scaled_depth(neighbor),
                target_t_source,
                u,
                v,
                fx,
                fy,
                cx,
                cy,
                args.min_depth_m,
                args.max_depth_m,
                args.absolute_tolerance_m,
                args.relative_tolerance,
            )
            judged_count += judged
            support_count += supported
            tested_neighbors.append(neighbor)

        support_ratio = support_count / np.maximum(judged_count, 1)
        reject_low = (
            (source_low >= args.min_depth_m)
            & (source_low <= args.max_depth_m)
            & (judged_count >= args.min_judged_neighbors)
            & (support_ratio < args.min_support_ratio)
        )

        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        reject_full = cv2.resize(
            reject_low.astype(np.uint8),
            (depth_mm.shape[1], depth_mm.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        valid_before = depth_mm > 0
        reject_full &= valid_before
        filtered = depth_mm.copy()
        filtered[reject_full] = 0
        output_depth_path = output / "depth" / depth_path.name
        if not cv2.imwrite(str(output_depth_path), filtered):
            raise RuntimeError(f"Failed to write filtered depth {depth_path.name}")

        rejected = int(reject_full.sum())
        rejected_pixels += rejected
        input_valid_ratios.append(float(valid_before.mean()))
        output_valid_ratios.append(float((filtered > 0).mean()))
        rejected_ratios.append(float(rejected / max(int(valid_before.sum()), 1)))
        evidence = propagate_depth_evidence(
            source,
            output,
            frame_index=int(frames[index]["idx"]),
            sensor_time_ns=int(frames[index]["sensor_time_ns"]),
            frame_name=depth_path.name,
            rejected_mask=reject_full,
            output_depth_path=output_depth_path,
            output_valid_ratio=output_valid_ratios[-1],
            rejected_valid_ratio=rejected_ratios[-1],
        )
        propagated_evidence_directories.update(evidence["propagated"])
        left_right_verified_frames += int(evidence["left_right_verified"])
        per_frame.append(
            {
                "frame": index,
                "sensor_time_ns": int(frames[index]["sensor_time_ns"]),
                "tested_neighbors": tested_neighbors,
                "low_resolution_judged_ratio": float((judged_count > 0).mean()),
                "low_resolution_supported_ratio": float((support_count > 0).mean()),
                "input_valid_ratio": input_valid_ratios[-1],
                "output_valid_ratio": output_valid_ratios[-1],
                "rejected_valid_ratio": rejected_ratios[-1],
            }
        )
        if (index + 1) % 50 == 0 or index == len(depths) - 1:
            print(
                f"Filtered {index + 1}/{len(depths)} frames; "
                f"valid={output_valid_ratios[-1]:.3f} "
                f"rejected={rejected_ratios[-1]:.3f}",
                flush=True,
            )

    output_tick = copy.deepcopy(metadata)
    output_tick["depth_filtering"] = {
        "method": "multi_neighbor_reprojection_consistency",
        "report": "temporal_depth_filter_report.json",
        "rgb_frames_preserved": True,
        "absolute_time_contract_preserved": True,
    }
    for index, frame in enumerate(output_tick["frames"]):
        frame["cam0"] = str(output / "rgb" / f"{index:08d}.png")
        frame["cam1"] = str(output / "stereo_right" / f"{index:08d}.png")
    (output / "tick_index.json").write_text(
        json.dumps(output_tick, indent=2) + "\n"
    )
    report = {
        "source_dataset": str(source),
        "output_dataset": str(output),
        "frame_count": len(depths),
        "absolute_time_contract_validated": True,
        "rgb_frames_preserved": True,
        "poses_preserved": True,
        "depth_evidence": {
            "propagated_directories": sorted(propagated_evidence_directories),
            "left_right_verified_frames": left_right_verified_frames,
            "coverage": left_right_verified_frames / max(1, len(depths)),
        },
        "neighbor_offsets": offsets,
        "filter_scale": args.filter_scale,
        "min_judged_neighbors": args.min_judged_neighbors,
        "min_support_ratio": args.min_support_ratio,
        "depth_bounds_m": [args.min_depth_m, args.max_depth_m],
        "absolute_tolerance_m": args.absolute_tolerance_m,
        "relative_tolerance": args.relative_tolerance,
        "rejected_pixels": rejected_pixels,
        "input_valid_ratio_percentiles": np.percentile(
            input_valid_ratios, [0, 5, 50, 95, 100]
        ).tolist(),
        "output_valid_ratio_percentiles": np.percentile(
            output_valid_ratios, [0, 5, 50, 95, 100]
        ).tolist(),
        "rejected_valid_ratio_percentiles": np.percentile(
            rejected_ratios, [0, 5, 50, 95, 100]
        ).tolist(),
        "per_frame": per_frame,
    }
    (output / "temporal_depth_filter_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
