#!/usr/bin/env python3
"""Create deterministic depth/pose diagnostic controls with visual differences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "depth_noise",
            "depth_scale",
            "pose_time_offset",
            "pose_translation",
            "pose_yaw",
        ),
    )
    parser.add_argument("--depth-noise-standard-deviation-m", type=float, default=0.05)
    parser.add_argument("--depth-scale", type=float, default=1.10)
    parser.add_argument("--pose-offset-frames", type=int, default=1)
    parser.add_argument("--pose-translation-m", type=float, default=0.05)
    parser.add_argument("--pose-yaw-deg", type=float, default=1.0)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame to perturb (inclusive); other frames remain byte-identical links.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame to perturb (exclusive); defaults to the dataset length.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def link_tree(source: Path, output: Path, *, excluded: set[str]) -> None:
    for path in source.iterdir():
        if path.name in excluded:
            continue
        destination = output / path.name
        if path.is_dir():
            destination.symlink_to(path.resolve(), target_is_directory=True)
        elif path.is_file():
            shutil.copy2(path, destination)


def depth_color(depth_m: np.ndarray) -> np.ndarray:
    valid = depth_m > 0.0
    encoded = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        maximum = max(0.25, float(np.percentile(depth_m[valid], 95)))
        encoded[valid] = np.clip(
            np.rint(depth_m[valid] / maximum * 255),
            1,
            255,
        ).astype(np.uint8)
    colored = cv2.applyColorMap(encoded, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def yaw_rotation(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    rotation = np.eye(4, dtype=np.float64)
    rotation[:3, :3] = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rotation


def pose_rotation_deg(transform: np.ndarray) -> float:
    cosine = np.clip((np.trace(transform[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def draw_pose_trajectory(
    source_poses: np.ndarray,
    output_poses: np.ndarray,
    changed: np.ndarray,
    destination: Path,
) -> None:
    """Save a deterministic top-down source/perturbed trajectory diagnostic."""

    source_xy = source_poses[:, :2, 3]
    output_xy = output_poses[:, :2, 3]
    all_xy = np.vstack((source_xy, output_xy))
    minimum = np.min(all_xy, axis=0)
    maximum = np.max(all_xy, axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    canvas = np.full((960, 1280, 3), 255, dtype=np.uint8)

    def project(points: np.ndarray) -> np.ndarray:
        normalized = (points - minimum) / span
        pixels = np.empty_like(normalized)
        pixels[:, 0] = 70.0 + normalized[:, 0] * 1140.0
        pixels[:, 1] = 890.0 - normalized[:, 1] * 820.0
        return np.rint(pixels).astype(np.int32)

    source_pixels = project(source_xy)
    output_pixels = project(output_xy)
    cv2.polylines(canvas, [source_pixels], False, (90, 90, 90), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [output_pixels], False, (0, 90, 220), 2, cv2.LINE_AA)
    for index in np.flatnonzero(changed):
        cv2.line(
            canvas,
            tuple(source_pixels[index]),
            tuple(output_pixels[index]),
            (20, 160, 20),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "source trajectory (gray) | perturbed trajectory (orange) | injected delta (green)",
        (40, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError(f"Could not write pose trajectory diagnostic: {destination}")


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if (
        args.depth_noise_standard_deviation_m <= 0.0
        or args.depth_scale <= 0.0
        or args.pose_offset_frames == 0
        or args.pose_translation_m <= 0.0
        or args.pose_yaw_deg <= 0.0
    ):
        raise ValueError("perturbation magnitudes are invalid")
    output.mkdir(parents=True)
    modified = {"pose"}
    if args.mode in {"depth_noise", "depth_scale"}:
        modified.add("depth")
    link_tree(dataset, output, excluded=modified)

    source_pose_path = dataset / "pose/poses.txt"
    source_poses = np.loadtxt(source_pose_path).reshape(-1, 4, 4)
    frame_count = len(source_poses)
    start_frame = args.start_frame
    end_frame = frame_count if args.end_frame is None else args.end_frame
    if start_frame < 0 or end_frame > frame_count or start_frame >= end_frame:
        raise ValueError(
            f"Invalid perturbation range [{start_frame}, {end_frame}) for {frame_count} frames"
        )
    perturbation_mask = np.zeros(frame_count, dtype=bool)
    perturbation_mask[start_frame:end_frame] = True

    pose_directory = output / "pose"
    pose_directory.mkdir()
    indices = np.arange(frame_count)
    output_poses = source_poses.copy()
    if args.mode == "pose_time_offset":
        shifted = np.clip(
            indices + args.pose_offset_frames,
            0,
            frame_count - 1,
        )
        indices[perturbation_mask] = shifted[perturbation_mask]
        output_poses[perturbation_mask] = source_poses[indices[perturbation_mask]]
    elif args.mode in {"pose_translation", "pose_yaw"}:
        # Perturb every other eligible frame.  A common rigid transform applied to
        # every pose would preserve relative motion and therefore be invisible to
        # temporal reprojection; alternating eligible frames creates a deterministic,
        # dose-controlled relative-pose fault while retaining unmodified controls.
        injected_indices = np.arange(start_frame, end_frame, 2)
        perturbation_mask[:] = False
        perturbation_mask[injected_indices] = True
        local_delta = np.eye(4, dtype=np.float64)
        if args.mode == "pose_translation":
            local_delta[0, 3] = args.pose_translation_m
        else:
            local_delta = yaw_rotation(args.pose_yaw_deg)
        output_poses[perturbation_mask] = (
            source_poses[perturbation_mask] @ local_delta
        )

    output_pose_path = pose_directory / "poses.txt"
    np.savetxt(output_pose_path, output_poses.reshape(-1, 16), fmt="%.12g")
    for source_sidecar in (dataset / "pose").iterdir():
        if source_sidecar.name == "poses.txt":
            continue
        destination = pose_directory / source_sidecar.name
        if source_sidecar.is_file():
            destination.symlink_to(source_sidecar.resolve())

    frame_records = []
    if args.mode in {"depth_noise", "depth_scale"}:
        source_depths = sorted((dataset / "depth").glob("*.png"))
        if not source_depths:
            raise FileNotFoundError(f"No depth PNG files in {dataset / 'depth'}")
        depth_directory = output / "depth"
        visual_directory = output / "perturbation_visualizations"
        raw_directory = output / "perturbation_raw"
        for directory in (depth_directory, visual_directory, raw_directory):
            directory.mkdir()
        rng = np.random.default_rng(args.seed)
        for index, source_path in enumerate(source_depths):
            destination = depth_directory / source_path.name
            if not perturbation_mask[index]:
                destination.symlink_to(source_path.resolve())
                continue
            raw = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.dtype != np.uint16:
                raise ValueError(f"Invalid depth PNG: {source_path}")
            depth = raw.astype(np.float32) / 1000.0
            valid = depth > 0.0
            if args.mode == "depth_noise":
                delta = np.zeros_like(depth)
                delta[valid] = rng.normal(
                    0.0,
                    args.depth_noise_standard_deviation_m,
                    size=int(np.count_nonzero(valid)),
                )
                perturbed = np.where(valid, np.maximum(0.001, depth + delta), 0.0)
            else:
                perturbed = np.where(valid, depth * args.depth_scale, 0.0)
                delta = perturbed - depth
            encoded = np.clip(
                np.rint(perturbed * 1000.0),
                0,
                65535,
            ).astype(np.uint16)
            if not cv2.imwrite(str(destination), encoded):
                raise RuntimeError(f"Could not write perturbed depth: {destination}")
            np.save(
                raw_directory / f"{index:08d}.npy",
                delta.astype(np.float32),
                allow_pickle=False,
            )
            before = depth_color(depth)
            after = depth_color(perturbed)
            magnitude = cv2.applyColorMap(
                np.clip(
                    np.rint(np.abs(delta) / max(0.01, float(np.percentile(np.abs(delta[valid]), 95)) if np.any(valid) else 0.01) * 255),
                    0,
                    255,
                ).astype(np.uint8),
                cv2.COLORMAP_INFERNO,
            )
            visual = np.hstack((before, after, magnitude))
            cv2.putText(
                visual,
                "original | perturbed | absolute delta",
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            visual_path = visual_directory / f"{index:08d}.jpg"
            if not cv2.imwrite(
                str(visual_path),
                visual,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            ):
                raise RuntimeError(f"Could not write perturbation visual: {visual_path}")
            frame_records.append(
                {
                    "frame_index": index,
                    "valid_pixels": int(np.count_nonzero(valid)),
                    "mean_absolute_delta_m": (
                        float(np.mean(np.abs(delta[valid]))) if np.any(valid) else 0.0
                    ),
                    "p95_absolute_delta_m": (
                        float(np.percentile(np.abs(delta[valid]), 95))
                        if np.any(valid)
                        else 0.0
                    ),
                    "raw_delta": str(raw_directory / f"{index:08d}.npy"),
                    "visualization": str(visual_path),
                    "source_sha256": sha256_file(source_path),
                    "perturbed_sha256": sha256_file(destination),
                }
            )
    else:
        raw_directory = output / "perturbation_raw"
        visual_directory = output / "perturbation_visualizations"
        raw_directory.mkdir()
        visual_directory.mkdir()
        delta_transforms = np.linalg.inv(source_poses) @ output_poses
        np.save(
            raw_directory / "pose_delta_transforms.npy",
            delta_transforms,
            allow_pickle=False,
        )
        draw_pose_trajectory(
            source_poses,
            output_poses,
            perturbation_mask,
            visual_directory / "pose_trajectory_topdown.png",
        )
        with (raw_directory / "pose_deltas.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "frame_index",
                    "source_pose_index",
                    "injected",
                    "translation_delta_m",
                    "rotation_delta_deg",
                ),
            )
            writer.writeheader()
            for index, transform in enumerate(delta_transforms):
                record = {
                    "frame_index": index,
                    "source_pose_index": int(indices[index]),
                    "injected": bool(perturbation_mask[index]),
                    "translation_delta_m": float(
                        np.linalg.norm(transform[:3, 3])
                    ),
                    "rotation_delta_deg": pose_rotation_deg(transform),
                }
                writer.writerow(record)
                if perturbation_mask[index]:
                    frame_records.append(record)

    report = {
        "schema": "daaam.geometry_perturbation.v1",
        "source_dataset": str(dataset),
        "output_dataset": str(output),
        "mode": args.mode,
        "seed": args.seed,
        "parameters": {
            "depth_noise_standard_deviation_m": args.depth_noise_standard_deviation_m,
            "depth_scale": args.depth_scale,
            "pose_offset_frames": args.pose_offset_frames,
            "pose_translation_m": args.pose_translation_m,
            "pose_yaw_deg": args.pose_yaw_deg,
            "start_frame_inclusive": start_frame,
            "end_frame_exclusive": end_frame,
            "pose_spatial_injection_pattern": (
                "every_other_eligible_frame_local_camera_transform"
                if args.mode in {"pose_translation", "pose_yaw"}
                else None
            ),
        },
        "source_pose_sha256": sha256_file(source_pose_path),
        "output_pose_sha256": sha256_file(output_pose_path),
        "perturbed_frame_count": int(np.count_nonzero(perturbation_mask)),
        "pose_source_indices": indices.tolist(),
        "frames": frame_records,
    }
    (output / "geometry_perturbation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
