#!/usr/bin/env python3
"""Render an estimated depth map as a color image without modifying the source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma": cv2.COLORMAP_MAGMA,
    "jet": cv2.COLORMAP_JET,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a metric depth map using a fixed color range. Integer depth "
            "PNG files are interpreted as millimeters by default."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset",
        type=Path,
        help="Dataset containing tick_index.json, rgb/, and depth/.",
    )
    source.add_argument("--depth", type=Path, help="Depth image path.")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame idx from tick_index.json when --dataset is used.",
    )
    parser.add_argument("--rgb", type=Path, help="Optional RGB image when --depth is used.")
    parser.add_argument("--output", type=Path, help="Destination PNG path.")
    parser.add_argument(
        "--depth-scale",
        type=float,
        help="Depth units per meter. Defaults to 1000 for integer data and 1 for float data.",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.25)
    parser.add_argument(
        "--max-depth-m",
        type=float,
        help="Upper visualization range in meters. Defaults to dataset metadata or 3.0m.",
    )
    parser.add_argument("--colormap", choices=tuple(COLORMAPS), default="turbo")
    parser.add_argument(
        "--with-context",
        action="store_true",
        help="Include RGB, RGB/depth overlay, and the depth legend for diagnostics.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.55,
        help="Color contribution in the RGB/depth overlay.",
    )
    return parser.parse_args()


def resolve_path(dataset: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else dataset / path


def dataset_inputs(dataset: Path, frame_index: int) -> tuple[Path, Path | None, float | None]:
    dataset = dataset.resolve()
    metadata_path = dataset / "tick_index.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    frame = next(
        (
            item
            for item in metadata.get("frames", [])
            if isinstance(item, dict) and int(item.get("idx", -1)) == frame_index
        ),
        None,
    )
    if frame is None:
        raise ValueError(f"Frame idx {frame_index} is absent from {metadata_path}")
    depth_value = frame.get("depth")
    depth_path = (
        resolve_path(dataset, depth_value)
        if depth_value
        else dataset / "depth" / f"{frame_index:08d}.png"
    )
    rgb_path = resolve_path(dataset, frame["cam0"]) if frame.get("cam0") else None
    max_depth_m = metadata.get("recommended_max_depth_m")
    return depth_path, rgb_path, float(max_depth_m) if max_depth_m is not None else None


def load_depth(path: Path, depth_scale: float | None) -> tuple[np.ndarray, float]:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Unable to read depth image: {path}")
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be single-channel: {path}")
    scale = depth_scale
    if scale is None:
        scale = 1000.0 if np.issubdtype(depth.dtype, np.integer) else 1.0
    if scale <= 0.0:
        raise ValueError("--depth-scale must be positive")
    return depth.astype(np.float32) / scale, scale


def color_depth(
    depth_m: np.ndarray, min_depth_m: float, max_depth_m: float, colormap: int
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    normalized = np.clip(
        (depth_m - min_depth_m) / (max_depth_m - min_depth_m), 0.0, 1.0
    )
    colored = cv2.applyColorMap(
        np.rint(normalized * 255.0).astype(np.uint8), colormap
    )
    colored[~valid] = 0
    return colored, valid


def labeled_panel(image: np.ndarray, label: str) -> np.ndarray:
    label_height = 38
    panel = np.full((image.shape[0] + label_height, image.shape[1], 3), 255, dtype=np.uint8)
    panel[label_height:] = image
    cv2.putText(
        panel,
        label,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (28, 28, 28),
        2,
        cv2.LINE_AA,
    )
    return panel


def depth_legend(
    height: int, min_depth_m: float, max_depth_m: float, colormap: int
) -> np.ndarray:
    width = 120
    legend = np.full((height + 38, width, 3), 255, dtype=np.uint8)
    gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    colorbar = cv2.applyColorMap(np.repeat(gradient, 20, axis=1), colormap)
    legend[38:, 16:36] = colorbar
    cv2.rectangle(legend, (16, 38), (35, height + 37), (30, 30, 30), 1)
    cv2.putText(
        legend,
        "Depth (m)",
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (28, 28, 28),
        1,
        cv2.LINE_AA,
    )
    for fraction in (0.0, 0.5, 1.0):
        y = int(38 + (1.0 - fraction) * (height - 1))
        value = min_depth_m + fraction * (max_depth_m - min_depth_m)
        cv2.line(legend, (37, y), (44, y), (30, 30, 30), 1)
        cv2.putText(
            legend,
            f"{value:.2f}",
            (50, min(height + 31, y + 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (28, 28, 28),
            1,
            cv2.LINE_AA,
        )
    return legend


def main() -> None:
    args = parse_args()
    if args.min_depth_m < 0.0:
        raise ValueError("--min-depth-m must be non-negative")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be in [0, 1]")

    metadata_max_depth_m = None
    if args.dataset is not None:
        depth_path, rgb_path, metadata_max_depth_m = dataset_inputs(args.dataset, args.frame)
    else:
        depth_path = args.depth.resolve()
        rgb_path = args.rgb.resolve() if args.rgb is not None else None
    depth_m, scale = load_depth(depth_path, args.depth_scale)
    max_depth_m = args.max_depth_m or metadata_max_depth_m or 3.0
    if max_depth_m <= args.min_depth_m:
        raise ValueError("--max-depth-m must exceed --min-depth-m")

    colored_depth, valid = color_depth(
        depth_m, args.min_depth_m, max_depth_m, COLORMAPS[args.colormap]
    )
    preview = colored_depth
    if args.with_context:
        panels = [labeled_panel(colored_depth, "Pseudo-color depth")]
        if rgb_path is not None and rgb_path.is_file():
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(f"Unable to read RGB image: {rgb_path}")
            if rgb.shape[:2] != depth_m.shape:
                rgb = cv2.resize(
                    rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_AREA
                )
            overlay = cv2.addWeighted(
                rgb, 1.0 - args.overlay_alpha, colored_depth, args.overlay_alpha, 0.0
            )
            overlay[~valid] = rgb[~valid]
            panels = [
                labeled_panel(rgb, "RGB"),
                panels[0],
                labeled_panel(overlay, "RGB + depth"),
            ]
        panels.append(
            depth_legend(depth_m.shape[0], args.min_depth_m, max_depth_m, COLORMAPS[args.colormap])
        )
        preview = cv2.hconcat(panels)

    if args.output is not None:
        output_path = args.output.resolve()
    elif args.dataset is not None:
        output_path = args.dataset.resolve() / "depth_visualizations" / f"{args.frame:08d}_depth_color.png"
    else:
        output_path = depth_path.with_name(f"{depth_path.stem}_color.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"Failed to write visualization: {output_path}")

    metric_values = depth_m[valid]
    report = {
        "depth": str(depth_path),
        "rgb": str(rgb_path) if rgb_path is not None else None,
        "output": str(output_path),
        "depth_scale": scale,
        "visual_range_m": [args.min_depth_m, max_depth_m],
        "valid_pixel_ratio": float(valid.mean()),
        "valid_depth_min_m": float(metric_values.min()) if len(metric_values) else None,
        "valid_depth_median_m": float(np.median(metric_values)) if len(metric_values) else None,
        "valid_depth_max_m": float(metric_values.max()) if len(metric_values) else None,
        "colormap": args.colormap,
        "with_context": args.with_context,
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
