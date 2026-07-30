#!/usr/bin/env python3
"""Summarize LiDAR-camera ground truth by range, edge, and texture bins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


RANGE_BINS_M = (
    (0.0, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
    (10.0, 65.535),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidar-ground-truth", required=True, type=Path)
    parser.add_argument("--prepared-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--edge-threshold", type=float, default=25.0)
    return parser.parse_args()


def percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def main() -> None:
    args = parse_args()
    gt = args.lidar_ground_truth.resolve()
    prepared = args.prepared_dataset.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    sparse_dir = gt / "sparse_depth_mm"
    mask_dir = gt / "valid_masks"
    rgb_dir = prepared / "rgb"
    if not sparse_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"LiDAR ground truth artifacts missing under {gt}")

    source_to_rgb: dict[int, Path] = {}
    tick_path = prepared / "tick_index.json"
    if tick_path.is_file():
        tick = json.loads(tick_path.read_text())
        for frame in tick.get("frames", []):
            source_index = int(
                frame.get("source_idx", frame.get("source_index", frame.get("tick", -1)))
            )
            cam0 = frame.get("cam0")
            if cam0:
                source_to_rgb[source_index] = Path(cam0)

    frame_rows = []
    bin_counts = {
        f"{lo:g}-{hi:g}m": 0 for lo, hi in RANGE_BINS_M
    }
    edge_counts = {"edge": 0, "non_edge": 0}
    texture_counts = {"low": 0, "high": 0}
    all_depths = []

    for sparse_path in sorted(sparse_dir.glob("*.png")):
        source_index = int(sparse_path.stem)
        sparse = cv2.imread(str(sparse_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_dir / sparse_path.name), cv2.IMREAD_GRAYSCALE)
        rgb_path = source_to_rgb.get(source_index)
        if rgb_path is None:
            for candidate in (
                rgb_dir / f"{source_index:06d}.png",
                rgb_dir / f"{source_index:08d}.png",
            ):
                if candidate.is_file():
                    rgb_path = candidate
                    break
        if sparse is None or mask is None:
            continue
        depth_m = sparse.astype(np.float64) / 1000.0
        valid = mask > 0
        depths = depth_m[valid]
        if depths.size == 0:
            frame_rows.append(
                {
                    "source_index": source_index,
                    "valid_points": 0,
                }
            )
            continue
        all_depths.append(depths)

        rgb = (
            cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb_path is not None and rgb_path.is_file()
            else None
        )
        if rgb is None:
            edge = np.zeros(valid.shape, dtype=bool)
            texture_high = np.zeros(valid.shape, dtype=bool)
        else:
            if rgb.shape[:2] != valid.shape:
                rgb = cv2.resize(rgb, (valid.shape[1], valid.shape[0]), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160) > 0
            edge = edges & valid
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            texture_high = (np.abs(lap) >= args.edge_threshold) & valid
        non_edge = valid & ~edge
        texture_low = valid & ~texture_high

        row_bins = {}
        for lo, hi in RANGE_BINS_M:
            key = f"{lo:g}-{hi:g}m"
            count = int(((depths >= lo) & (depths < hi)).sum())
            row_bins[key] = count
            bin_counts[key] += count
        edge_counts["edge"] += int(edge.sum())
        edge_counts["non_edge"] += int(non_edge.sum())
        texture_counts["high"] += int(texture_high.sum())
        texture_counts["low"] += int(texture_low.sum())
        frame_rows.append(
            {
                "source_index": source_index,
                "valid_points": int(depths.size),
                "depth_m": {
                    "p50": percentile(depths, 50),
                    "p95": percentile(depths, 95),
                    "max": float(depths.max()),
                },
                "range_bins": row_bins,
                "edge_points": int(edge.sum()),
                "non_edge_points": int(non_edge.sum()),
                "high_texture_points": int(texture_high.sum()),
                "low_texture_points": int(texture_low.sum()),
            }
        )

    concat = np.concatenate(all_depths) if all_depths else np.zeros(0, dtype=np.float64)
    report = {
        "schema": "daaam.g1_lidar_ground_truth_summary.v1",
        "lidar_ground_truth": str(gt),
        "prepared_dataset": str(prepared),
        "frame_count": len(frame_rows),
        "total_valid_points": int(concat.size),
        "depth_m": {
            "p50": percentile(concat, 50),
            "p95": percentile(concat, 95),
            "max": float(concat.max()) if concat.size else None,
        },
        "range_bins_m": bin_counts,
        "edge_bins": edge_counts,
        "texture_bins": texture_counts,
        "frames": frame_rows,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
