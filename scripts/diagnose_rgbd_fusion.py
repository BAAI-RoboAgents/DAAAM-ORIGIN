#!/usr/bin/env python3
"""Fuse sampled RGB-D frames outside Hydra and render an alignment preview.

This utility is intentionally simple: it backprojects each sampled depth pixel
with the dataset intrinsics, applies the supplied world_T_camera pose, and
writes a colored point cloud plus orthographic diagnostics.  It separates
pose/depth alignment problems from TSDF and scene-graph processing.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a direct world-space RGB-D fusion diagnostic."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--pose-path",
        type=Path,
        default=None,
        help="Optional world_T_camera pose file; defaults to dataset/pose/poses.txt.",
    )
    parser.add_argument("--frame-step", type=int, default=15)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Inclusive first frame used for the diagnostic.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Exclusive final frame used for the diagnostic; defaults to all frames.",
    )
    parser.add_argument("--pixel-step", type=int, default=10)
    parser.add_argument("--min-depth-m", type=float, default=0.25)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--voxel-size-m", type=float, default=0.04)
    parser.add_argument("--render-sample", type=int, default=180000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_poses(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64).reshape(-1, 4, 4)
    if not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"Expected homogeneous poses in {path}")
    return poses


def sorted_images(directory: Path) -> list[Path]:
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() == ".png")
    if not paths:
        raise FileNotFoundError(f"No PNG images in {directory}")
    return paths


def sample_frame(
    rgb_path: Path,
    depth_path: Path,
    pose: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    pixel_step: int,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    color = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if color is None or depth_mm is None or depth_mm.dtype != np.uint16:
        raise ValueError(f"Invalid RGB-D input: {rgb_path} / {depth_path}")
    if color.shape[:2] != depth_mm.shape:
        raise ValueError(f"RGB/depth dimensions differ: {rgb_path} / {depth_path}")

    v, u = np.mgrid[0 : depth_mm.shape[0] : pixel_step, 0 : depth_mm.shape[1] : pixel_step]
    z = depth_mm[::pixel_step, ::pixel_step].astype(np.float64) / 1000.0
    valid = (z >= min_depth_m) & (z <= max_depth_m)
    z = z[valid]
    u = u[valid]
    v = v[valid]
    if not len(z):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    points_camera = np.column_stack(
        ((u - cx) * z / fx, (v - cy) * z / fy, z)
    )
    points_world = points_camera @ pose[:3, :3].T + pose[:3, 3]
    colors_rgb = color[::pixel_step, ::pixel_step][valid][:, ::-1]
    return points_world.astype(np.float32), colors_rgb.astype(np.uint8)


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size_m <= 0.0 or not len(points):
        return points, colors
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep.sort()
    return points[keep], colors[keep]


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    vertices = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(vertices.tobytes())


def set_equal_aspect(axis, first: np.ndarray, second: np.ndarray) -> None:
    minimum = min(first.min(), second.min())
    maximum = max(first.max(), second.max())
    padding = max(0.05, 0.04 * (maximum - minimum))
    axis.set_xlim(minimum - padding, maximum + padding)
    axis.set_ylim(minimum - padding, maximum + padding)
    axis.set_aspect("equal", adjustable="box")


def render_preview(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    camera_positions: np.ndarray,
    render_sample: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    if len(points) > render_sample:
        indices = rng.choice(len(points), render_sample, replace=False)
        render_points = points[indices]
        render_colors = colors[indices]
    else:
        render_points = points
        render_colors = colors

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    views = ((0, 1, "Top: X / Y"), (0, 2, "Side: X / Z"), (1, 2, "Side: Y / Z"))
    normalized_colors = render_colors.astype(np.float32) / 255.0
    for axis, (first, second, title) in zip(axes, views):
        axis.scatter(
            render_points[:, first],
            render_points[:, second],
            c=normalized_colors,
            s=0.35,
            marker=",",
            linewidths=0,
            rasterized=True,
        )
        axis.plot(
            camera_positions[:, first],
            camera_positions[:, second],
            color="black",
            linewidth=0.7,
            alpha=0.8,
        )
        axis.set_title(title)
        axis.set_xlabel("XYZ"[first] + " (m)")
        axis.set_ylabel("XYZ"[second] + " (m)")
        axis.grid(alpha=0.2)
        set_equal_aspect(axis, render_points[:, first], render_points[:, second])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    if args.frame_step < 1 or args.pixel_step < 1:
        raise ValueError("--frame-step and --pixel-step must be positive")
    if args.min_depth_m <= 0.0 or args.min_depth_m >= args.max_depth_m:
        raise ValueError("Depth bounds must satisfy 0 < min < max")

    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_path = (args.pose_path or dataset / "pose" / "poses.txt").resolve()
    camera = json.loads((dataset / "camera_info.json").read_text())
    poses = load_poses(pose_path)
    rgb_paths = sorted_images(dataset / "rgb")
    depth_paths = sorted_images(dataset / "depth")
    frame_count = min(len(poses), len(rgb_paths), len(depth_paths))
    if frame_count == 0:
        raise ValueError("No aligned RGB-D poses available")

    start_frame = args.start_frame
    end_frame = frame_count if args.end_frame is None else args.end_frame
    if not 0 <= start_frame < end_frame <= frame_count:
        raise ValueError(
            f"Expected 0 <= start < end <= {frame_count}, got "
            f"start={start_frame}, end={end_frame}"
        )
    indices = list(range(start_frame, end_frame, args.frame_step))
    if indices[-1] != end_frame - 1:
        indices.append(end_frame - 1)
    fx, fy, cx, cy = (float(camera[key]) for key in ("fx", "fy", "cx", "cy"))
    point_batches = []
    color_batches = []
    for ordinal, index in enumerate(indices, start=1):
        points, colors = sample_frame(
            rgb_paths[index],
            depth_paths[index],
            poses[index],
            fx,
            fy,
            cx,
            cy,
            args.pixel_step,
            args.min_depth_m,
            args.max_depth_m,
        )
        point_batches.append(points)
        color_batches.append(colors)
        print(f"Sampled {ordinal}/{len(indices)} RGB-D frames", flush=True)

    points = np.concatenate(point_batches)
    colors = np.concatenate(color_batches)
    points, colors = voxel_downsample(points, colors, args.voxel_size_m)
    if not len(points):
        raise RuntimeError("No valid world-space points after depth filtering")

    ply_path = output_dir / "direct_rgbd_fusion.ply"
    preview_path = output_dir / "direct_rgbd_fusion_preview.png"
    write_ply(ply_path, points, colors)
    render_preview(
        preview_path,
        points,
        colors,
        poses[indices, :3, 3],
        args.render_sample,
        args.seed,
    )
    report = {
        "dataset": str(dataset),
        "pose_path": str(pose_path),
        "frames_available": frame_count,
        "frame_range": [start_frame, end_frame],
        "frames_sampled": len(indices),
        "frame_step": args.frame_step,
        "pixel_step": args.pixel_step,
        "depth_bounds_m": [args.min_depth_m, args.max_depth_m],
        "voxel_size_m": args.voxel_size_m,
        "points_after_voxel_downsample": int(len(points)),
        "bounds_min_m": points.min(axis=0).astype(float).tolist(),
        "bounds_max_m": points.max(axis=0).astype(float).tolist(),
        "sampled_camera_path_length_m": float(
            np.linalg.norm(np.diff(poses[indices, :3, 3], axis=0), axis=1).sum()
        ),
        "ply": str(ply_path),
        "preview": str(preview_path),
    }
    report_path = output_dir / "direct_rgbd_fusion_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
