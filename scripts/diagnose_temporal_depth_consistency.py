#!/usr/bin/env python3
"""Measure dense RGB-D agreement after reprojection into nearby frames.

The diagnostic only reads an RGB-D dataset.  For sampled reference frames it
uses the supplied world_T_camera trajectory to project valid depth samples into
nearby images, compares their predicted optical Z values with the neighboring
depth maps, and writes a compact visual record plus a JSON report.  This makes
it possible to distinguish a trajectory problem from depth that changes shape
between otherwise adjacent observations.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose temporal consistency of metric RGB-D depth maps."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--pose-path",
        type=Path,
        default=None,
        help="world_T_camera poses; defaults to dataset/pose/poses.txt.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--frame-step",
        type=int,
        default=75,
        help="Temporal spacing between sampled reference frames.",
    )
    parser.add_argument(
        "--neighbor-offsets",
        default="1,5",
        help="Comma-separated positive frame offsets checked in both directions.",
    )
    parser.add_argument(
        "--pixel-step",
        type=int,
        default=4,
        help="Sampling interval in source pixels.",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.25)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument(
        "--absolute-tolerance-m",
        type=float,
        default=0.04,
        help="Base reprojection agreement tolerance in meters.",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.03,
        help="Additional depth-proportional agreement tolerance.",
    )
    parser.add_argument(
        "--max-panels",
        type=int,
        default=20,
        help="Maximum pairwise image panels saved to disk.",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Measure each positive-offset pair once instead of both directions.",
    )
    parser.add_argument(
        "--require-time-contract",
        action="store_true",
        help=(
            "Require tick_index and pose_timestamps_ns to prove exact absolute-time "
            "image/pose alignment."
        ),
    )
    parser.add_argument(
        "--fail-below-adjacent-agreement-rate",
        type=float,
        default=None,
        help="Exit nonzero when offset-one weighted agreement is below this value.",
    )
    parser.add_argument(
        "--fail-above-adjacent-median-error-m",
        type=float,
        default=None,
        help=(
            "Exit nonzero when the median of offset-one pair median depth errors "
            "exceeds this value."
        ),
    )
    parser.add_argument(
        "--window-size-frames",
        type=int,
        default=100,
        help="Frame span used for local adjacent-consistency summaries.",
    )
    parser.add_argument(
        "--fail-below-window-adjacent-agreement-rate",
        type=float,
        default=None,
        help="Exit nonzero when any local offset-one window is below this value.",
    )
    parser.add_argument(
        "--worst-pair-count",
        type=int,
        default=20,
        help="Number of lowest-agreement pairs summarized in the report.",
    )
    return parser.parse_args()


def sorted_pngs(directory: Path) -> list[Path]:
    paths = sorted(path for path in directory.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images in {directory}")
    return paths


def load_poses(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64).reshape(-1, 4, 4)
    if not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"Expected homogeneous poses in {path}")
    return poses


def load_time_contract(dataset: Path, expected_count: int) -> tuple[dict, list[dict]]:
    """Load and strictly validate the absolute image-to-pose time contract."""
    metadata_path = dataset / "tick_index.json"
    pose_time_path = dataset / "pose" / "pose_timestamps_ns.txt"
    metadata = json.loads(metadata_path.read_text())
    frames = metadata.get("frames")
    if not isinstance(frames, list) or len(frames) != expected_count:
        raise ValueError(
            f"Expected {expected_count} tick_index frames, got "
            f"{len(frames) if isinstance(frames, list) else 'invalid'}"
        )
    pose_timestamps = [
        int(line)
        for line in pose_time_path.read_text().splitlines()
        if line.strip()
    ]
    if len(pose_timestamps) != expected_count:
        raise ValueError(
            f"Expected {expected_count} pose timestamps, got {len(pose_timestamps)}"
        )
    time_origin_ns = int(metadata["time_origin_ns"])
    previous_sensor_time_ns = None
    for index, frame in enumerate(frames):
        pose_row = int(frame["pose_row"])
        sensor_time_ns = int(frame["sensor_time_ns"])
        cam0_time_ns = int(frame["cam0_sensor_time_ns"])
        pose_time_ns = int(frame["pose_sensor_time_ns"])
        if int(frame["idx"]) != index:
            raise ValueError(f"tick_index frame {index} has idx={frame['idx']}")
        if pose_row != index:
            raise ValueError(
                f"Selected pose row must match output row at frame {index}: {pose_row}"
            )
        if sensor_time_ns != cam0_time_ns or sensor_time_ns != pose_time_ns:
            raise ValueError(f"Camera/pose absolute times disagree at frame {index}")
        if pose_timestamps[pose_row] != pose_time_ns:
            raise ValueError(f"pose_timestamps_ns disagrees at frame {index}")
        if previous_sensor_time_ns is not None and sensor_time_ns <= previous_sensor_time_ns:
            raise ValueError("sensor_time_ns must be strictly increasing")
        expected_relative_s = (sensor_time_ns - time_origin_ns) / 1.0e9
        if not math.isclose(
            float(frame["timestamp"]), expected_relative_s, abs_tol=1.0e-6
        ):
            raise ValueError(f"Relative timestamp disagrees at frame {index}")
        previous_sensor_time_ns = sensor_time_ns
    return metadata, frames


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def frame_provenance(frame: dict | None, index: int) -> dict:
    if frame is None:
        return {"frame": index}
    source_index = frame.get("source_frame_idx", frame.get("source_idx"))
    return {
        "frame": index,
        "sensor_time_ns": int(frame["sensor_time_ns"]),
        "source_frame_idx": int(source_index) if source_index is not None else None,
        "source_image_idx": (
            int(frame["source_idx"]) if frame.get("source_idx") is not None else None
        ),
        "selection_reason": frame.get("selection_reason"),
    }


def parse_offsets(value: str) -> list[int]:
    try:
        offsets = sorted({int(item.strip()) for item in value.split(",")})
    except ValueError as error:
        raise ValueError("--neighbor-offsets must contain integer values") from error
    if not offsets or offsets[0] < 1:
        raise ValueError("--neighbor-offsets must contain positive values")
    return offsets


def read_depth(path: Path) -> np.ndarray:
    depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None or depth_mm.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth image: {path}")
    return depth_mm.astype(np.float32) / 1000.0


def percentile_or_none(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if len(values) else None


def color_depth(depth: np.ndarray, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    normalized = np.clip(
        (depth - min_depth_m) / (max_depth_m - min_depth_m), 0.0, 1.0
    )
    colored = cv2.applyColorMap(
        np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    colored[depth <= 0.0] = 0
    return colored


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def make_panel(
    rgb_reference: np.ndarray,
    depth_reference: np.ndarray,
    status: np.ndarray,
    rgb_neighbor: np.ndarray,
    depth_neighbor: np.ndarray,
    reference_index: int,
    neighbor_index: int,
    agreement_rate: float | None,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Create a five-tile panel with a dense agreement-status overlay."""
    scale = 0.5
    output_size = (
        max(1, int(rgb_reference.shape[1] * scale)),
        max(1, int(rgb_reference.shape[0] * scale)),
    )
    reference = cv2.resize(rgb_reference, output_size, interpolation=cv2.INTER_AREA)
    neighbor = cv2.resize(rgb_neighbor, output_size, interpolation=cv2.INTER_AREA)
    reference_depth = cv2.resize(
        color_depth(depth_reference, min_depth_m, max_depth_m),
        output_size,
        interpolation=cv2.INTER_AREA,
    )
    neighbor_depth = cv2.resize(
        color_depth(depth_neighbor, min_depth_m, max_depth_m),
        output_size,
        interpolation=cv2.INTER_AREA,
    )

    # Status is evaluated on the sampled source grid.  Nearest-neighbor
    # expansion keeps disagreement regions legible instead of inventing colors.
    status_full = cv2.resize(
        status, (rgb_reference.shape[1], rgb_reference.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    status_overlay = rgb_reference.copy()
    colors = {
        1: (70, 70, 70),       # projected out of view
        2: (255, 120, 20),     # target lacks valid depth
        3: (30, 185, 45),      # agreement
        4: (35, 35, 235),      # disagreement
    }
    for code, color in colors.items():
        mask = status_full == code
        status_overlay[mask] = (
            0.35 * status_overlay[mask] + 0.65 * np.asarray(color)
        ).astype(np.uint8)
    status_overlay = cv2.resize(
        status_overlay, output_size, interpolation=cv2.INTER_AREA
    )
    agreement_text = "n/a" if agreement_rate is None else f"{agreement_rate * 100.0:.1f}%"
    tiles = (
        add_label(reference, f"reference RGB {reference_index}"),
        add_label(reference_depth, "reference depth"),
        add_label(status_overlay, f"green agree / red reject: {agreement_text}"),
        add_label(neighbor, f"neighbor RGB {neighbor_index}"),
        add_label(neighbor_depth, "neighbor depth"),
    )
    return np.hstack(tiles)


def radial_summary(
    radii: np.ndarray,
    comparable: np.ndarray,
    agreement: np.ndarray,
) -> list[dict]:
    edges = (0.0, 0.4, 0.8, 1.2, float("inf"))
    bins = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = comparable & (radii >= lower) & (radii < upper)
        count = int(selected.sum())
        bins.append(
            {
                "normalized_ray_radius": [lower, upper if np.isfinite(upper) else None],
                "comparable_samples": count,
                "agreement_rate": (
                    float(agreement[selected].mean()) if count else None
                ),
            }
        )
    return bins


def main() -> None:
    args = parse_args()
    if args.frame_step < 1 or args.pixel_step < 1 or args.window_size_frames < 1:
        raise ValueError("Frame, pixel, and window steps must be positive")
    if args.min_depth_m <= 0.0 or args.min_depth_m >= args.max_depth_m:
        raise ValueError("Depth bounds must satisfy 0 < min < max")
    if args.absolute_tolerance_m <= 0.0 or args.relative_tolerance < 0.0:
        raise ValueError("Depth agreement tolerances must be non-negative")
    offsets = parse_offsets(args.neighbor_offsets)

    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    panel_dir = output_dir / "pairs"
    panel_dir.mkdir(parents=True, exist_ok=True)
    pose_path = (args.pose_path or dataset / "pose" / "poses.txt").resolve()
    poses = load_poses(pose_path)
    rgb_paths = sorted_pngs(dataset / "rgb")
    depth_paths = sorted_pngs(dataset / "depth")
    camera = json.loads((dataset / "camera_info.json").read_text())
    fx, fy, cx, cy = (float(camera[key]) for key in ("fx", "fy", "cx", "cy"))

    frame_count = min(len(poses), len(rgb_paths), len(depth_paths))
    time_metadata = None
    time_frames = None
    if args.require_time_contract:
        time_metadata, time_frames = load_time_contract(dataset, frame_count)
    end_frame = frame_count if args.end_frame is None else args.end_frame
    if not 0 <= args.start_frame < end_frame <= frame_count:
        raise ValueError(
            f"Expected 0 <= start < end <= {frame_count}, got "
            f"{args.start_frame}, {end_frame}"
        )
    reference_indices = list(range(args.start_frame, end_frame, args.frame_step))
    if reference_indices[-1] != end_frame - 1:
        reference_indices.append(end_frame - 1)

    first_depth = read_depth(depth_paths[reference_indices[0]])
    height, width = first_depth.shape
    if width != int(camera["width"]) or height != int(camera["height"]):
        raise ValueError("Depth dimensions disagree with camera_info.json")
    sampled_u = np.arange(0, width, args.pixel_step, dtype=np.float32)
    sampled_v = np.arange(0, height, args.pixel_step, dtype=np.float32)
    u, v = np.meshgrid(sampled_u, sampled_v)
    ray_x = (u - cx) / fx
    ray_y = (v - cy) / fy
    radii = np.hypot(ray_x, ray_y)

    pairs = []
    panels_written = 0
    for reference_ordinal, reference_index in enumerate(reference_indices, start=1):
        depth_reference = read_depth(depth_paths[reference_index])
        rgb_reference = cv2.imread(str(rgb_paths[reference_index]), cv2.IMREAD_COLOR)
        if rgb_reference is None or depth_reference.shape != (height, width):
            raise ValueError(f"Invalid reference RGB-D frame {reference_index}")
        z_reference = depth_reference[:: args.pixel_step, :: args.pixel_step]
        z_reference = z_reference[: u.shape[0], : u.shape[1]]
        valid_reference = (z_reference >= args.min_depth_m) & (
            z_reference <= args.max_depth_m
        )

        for offset in offsets:
            directions = (1,) if args.forward_only else (-1, 1)
            for direction in directions:
                neighbor_index = reference_index + direction * offset
                if not args.start_frame <= neighbor_index < end_frame:
                    continue
                depth_neighbor = read_depth(depth_paths[neighbor_index])
                rgb_neighbor = cv2.imread(
                    str(rgb_paths[neighbor_index]), cv2.IMREAD_COLOR
                )
                if rgb_neighbor is None or depth_neighbor.shape != (height, width):
                    raise ValueError(f"Invalid neighbor RGB-D frame {neighbor_index}")

                points_reference = np.stack(
                    (ray_x * z_reference, ray_y * z_reference, z_reference), axis=-1
                )
                relative = np.linalg.inv(poses[neighbor_index]) @ poses[reference_index]
                points_neighbor = (
                    points_reference @ relative[:3, :3].T + relative[:3, 3]
                )
                predicted_z = points_neighbor[..., 2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    projected_u = fx * points_neighbor[..., 0] / predicted_z + cx
                    projected_v = fy * points_neighbor[..., 1] / predicted_z + cy
                in_bounds = (
                    valid_reference
                    & (predicted_z > 0.0)
                    & (projected_u >= 0.0)
                    & (projected_u <= width - 1.0)
                    & (projected_v >= 0.0)
                    & (projected_v <= height - 1.0)
                )
                sampled_neighbor_z = cv2.remap(
                    depth_neighbor,
                    projected_u.astype(np.float32),
                    projected_v.astype(np.float32),
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                neighbor_valid = (sampled_neighbor_z >= args.min_depth_m) & (
                    sampled_neighbor_z <= args.max_depth_m
                )
                comparable = in_bounds & neighbor_valid
                absolute_error = np.abs(sampled_neighbor_z - predicted_z)
                relative_error = absolute_error / np.maximum(predicted_z, 1.0e-6)
                threshold = args.absolute_tolerance_m + args.relative_tolerance * predicted_z
                agreement = comparable & (absolute_error <= threshold)

                status = np.zeros(u.shape, dtype=np.uint8)
                status[valid_reference & ~in_bounds] = 1
                status[in_bounds & ~neighbor_valid] = 2
                status[agreement] = 3
                status[comparable & ~agreement] = 4
                comparable_error = absolute_error[comparable]
                comparable_relative_error = relative_error[comparable]
                comparable_count = int(comparable.sum())
                agreement_rate = (
                    float(agreement[comparable].mean()) if comparable_count else None
                )
                relative_pose = np.linalg.inv(poses[neighbor_index]) @ poses[reference_index]
                reference_metadata = (
                    time_frames[reference_index] if time_frames is not None else None
                )
                neighbor_metadata = (
                    time_frames[neighbor_index] if time_frames is not None else None
                )
                pair = {
                    "reference_frame": reference_index,
                    "neighbor_frame": neighbor_index,
                    "neighbor_offset": neighbor_index - reference_index,
                    "reference": frame_provenance(reference_metadata, reference_index),
                    "neighbor": frame_provenance(neighbor_metadata, neighbor_index),
                    "absolute_time_delta_s": (
                        abs(
                            int(neighbor_metadata["sensor_time_ns"])
                            - int(reference_metadata["sensor_time_ns"])
                        )
                        / 1.0e9
                        if time_frames is not None
                        else None
                    ),
                    "pose_translation_m": float(np.linalg.norm(relative_pose[:3, 3])),
                    "pose_rotation_deg": rotation_angle_deg(relative_pose[:3, :3]),
                    "valid_reference_samples": int(valid_reference.sum()),
                    "projected_in_bounds_samples": int(in_bounds.sum()),
                    "comparable_samples": comparable_count,
                    "agreement_samples": int(agreement.sum()),
                    "agreement_rate": agreement_rate,
                    "median_absolute_depth_error_m": percentile_or_none(
                        comparable_error, 50
                    ),
                    "p95_absolute_depth_error_m": percentile_or_none(
                        comparable_error, 95
                    ),
                    "median_relative_depth_error": percentile_or_none(
                        comparable_relative_error, 50
                    ),
                    "p95_relative_depth_error": percentile_or_none(
                        comparable_relative_error, 95
                    ),
                    "radial_bins": radial_summary(radii, comparable, agreement),
                }
                pairs.append(pair)

                if panels_written < args.max_panels:
                    panel = make_panel(
                        rgb_reference,
                        depth_reference,
                        status,
                        rgb_neighbor,
                        depth_neighbor,
                        reference_index,
                        neighbor_index,
                        agreement_rate,
                        args.min_depth_m,
                        args.max_depth_m,
                    )
                    panel_path = panel_dir / (
                        f"{reference_index:08d}_to_{neighbor_index:08d}.png"
                    )
                    cv2.imwrite(str(panel_path), panel)
                    pair["panel"] = str(panel_path)
                    panels_written += 1
        if reference_ordinal % 25 == 0 or reference_ordinal == len(reference_indices):
            print(
                f"Measured {reference_ordinal}/{len(reference_indices)} reference "
                f"frames; pairs={len(pairs)}",
                flush=True,
            )

    by_offset = {}
    for offset in sorted({abs(pair["neighbor_offset"]) for pair in pairs}):
        selected = [pair for pair in pairs if abs(pair["neighbor_offset"]) == offset]
        comparable = sum(pair["comparable_samples"] for pair in selected)
        agreed = sum(pair["agreement_samples"] for pair in selected)
        by_offset[str(offset)] = {
            "pairs": len(selected),
            "comparable_samples": comparable,
            "agreement_rate_weighted": float(agreed / comparable) if comparable else None,
            "agreement_rate_pair_median": percentile_or_none(
                np.asarray(
                    [
                        pair["agreement_rate"]
                        for pair in selected
                        if pair["agreement_rate"] is not None
                    ],
                    dtype=np.float64,
                ),
                50,
            ),
            "median_absolute_depth_error_pair_median_m": percentile_or_none(
                np.asarray(
                    [
                        pair["median_absolute_depth_error_m"]
                        for pair in selected
                        if pair["median_absolute_depth_error_m"] is not None
                    ],
                    dtype=np.float64,
                ),
                50,
            ),
            "agreement_rate_pair_p05": percentile_or_none(
                np.asarray(
                    [
                        pair["agreement_rate"]
                        for pair in selected
                        if pair["agreement_rate"] is not None
                    ],
                    dtype=np.float64,
                ),
                5,
            ),
            "absolute_time_delta_s_median": percentile_or_none(
                np.asarray(
                    [
                        pair["absolute_time_delta_s"]
                        for pair in selected
                        if pair["absolute_time_delta_s"] is not None
                    ],
                    dtype=np.float64,
                ),
                50,
            ),
            "absolute_time_delta_s_max": (
                max(
                    pair["absolute_time_delta_s"]
                    for pair in selected
                    if pair["absolute_time_delta_s"] is not None
                )
                if any(pair["absolute_time_delta_s"] is not None for pair in selected)
                else None
            ),
        }

    total_comparable = sum(pair["comparable_samples"] for pair in pairs)
    total_agreed = sum(pair["agreement_samples"] for pair in pairs)
    ranked_pairs = sorted(
        (pair for pair in pairs if pair["agreement_rate"] is not None),
        key=lambda pair: pair["agreement_rate"],
    )
    adjacent = by_offset.get("1")
    adjacent_windows = []
    adjacent_pairs = [
        pair for pair in pairs if abs(pair["neighbor_offset"]) == 1
    ]
    for window_start in range(
        args.start_frame, end_frame, args.window_size_frames
    ):
        window_end = min(window_start + args.window_size_frames, end_frame)
        selected = [
            pair
            for pair in adjacent_pairs
            if window_start
            <= min(pair["reference_frame"], pair["neighbor_frame"])
            < window_end
        ]
        if not selected:
            continue
        comparable = sum(pair["comparable_samples"] for pair in selected)
        agreed = sum(pair["agreement_samples"] for pair in selected)
        adjacent_windows.append(
            {
                "frame_range": [window_start, window_end],
                "pairs": len(selected),
                "comparable_samples": comparable,
                "agreement_rate_weighted": (
                    float(agreed / comparable) if comparable else None
                ),
                "median_absolute_depth_error_pair_median_m": percentile_or_none(
                    np.asarray(
                        [
                            pair["median_absolute_depth_error_m"]
                            for pair in selected
                            if pair["median_absolute_depth_error_m"] is not None
                        ],
                        dtype=np.float64,
                    ),
                    50,
                ),
            }
        )
    gate_checks = []
    if args.fail_below_adjacent_agreement_rate is not None:
        actual = adjacent["agreement_rate_weighted"] if adjacent else None
        gate_checks.append(
            {
                "metric": "offset_one_weighted_agreement_rate",
                "operator": ">=",
                "threshold": args.fail_below_adjacent_agreement_rate,
                "actual": actual,
                "passed": actual is not None
                and actual >= args.fail_below_adjacent_agreement_rate,
            }
        )
    if args.fail_above_adjacent_median_error_m is not None:
        actual = adjacent["median_absolute_depth_error_pair_median_m"] if adjacent else None
        gate_checks.append(
            {
                "metric": "offset_one_pair_median_absolute_error_median_m",
                "operator": "<=",
                "threshold": args.fail_above_adjacent_median_error_m,
                "actual": actual,
                "passed": actual is not None
                and actual <= args.fail_above_adjacent_median_error_m,
            }
        )
    if args.fail_below_window_adjacent_agreement_rate is not None:
        valid_windows = [
            window
            for window in adjacent_windows
            if window["agreement_rate_weighted"] is not None
        ]
        worst_window = min(
            valid_windows,
            key=lambda window: window["agreement_rate_weighted"],
            default=None,
        )
        actual = (
            worst_window["agreement_rate_weighted"]
            if worst_window is not None
            else None
        )
        gate_checks.append(
            {
                "metric": "minimum_window_offset_one_weighted_agreement_rate",
                "operator": ">=",
                "threshold": args.fail_below_window_adjacent_agreement_rate,
                "actual": actual,
                "worst_frame_range": (
                    worst_window["frame_range"] if worst_window is not None else None
                ),
                "passed": actual is not None
                and actual >= args.fail_below_window_adjacent_agreement_rate,
            }
        )
    report = {
        "dataset": str(dataset),
        "pose_path": str(pose_path),
        "absolute_time_contract": (
            {
                "validated": True,
                "time_origin_ns": int(time_metadata["time_origin_ns"]),
                "pose_timestamp_file": str(dataset / "pose" / "pose_timestamps_ns.txt"),
            }
            if time_metadata is not None
            else {"validated": False, "required": False}
        ),
        "frames_available": frame_count,
        "reference_frame_range": [args.start_frame, end_frame],
        "reference_frames": reference_indices,
        "neighbor_offsets": offsets,
        "pixel_step": args.pixel_step,
        "depth_bounds_m": [args.min_depth_m, args.max_depth_m],
        "agreement_tolerance": {
            "absolute_m": args.absolute_tolerance_m,
            "relative": args.relative_tolerance,
        },
        "pairs": pairs,
        "summary_by_absolute_offset": by_offset,
        "adjacent_window_summary": adjacent_windows,
        "worst_pairs": ranked_pairs[: args.worst_pair_count],
        "overall_comparable_samples": total_comparable,
        "overall_agreement_rate_weighted": (
            float(total_agreed / total_comparable) if total_comparable else None
        ),
        "pre_hydra_gate": {
            "passed": all(check["passed"] for check in gate_checks),
            "checks": gate_checks,
        },
    }
    report_path = output_dir / "temporal_depth_consistency_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "summary_by_absolute_offset",
        "adjacent_window_summary",
        "overall_comparable_samples",
        "overall_agreement_rate_weighted",
        "pre_hydra_gate",
    )}, indent=2))
    if not report["pre_hydra_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
