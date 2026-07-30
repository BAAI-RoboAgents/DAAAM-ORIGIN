#!/usr/bin/env python3
"""Audit recorded rectified stereo geometry over a representative frame batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from audit_rectified_stereo_frame import match_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-indices", required=True, nargs="+", type=int)
    parser.add_argument("--maximum-stereo-delta-ms", type=float, default=5.0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if (
        not args.source_indices
        or len(args.source_indices) != len(set(args.source_indices))
        or min(args.source_indices) < 0
    ):
        raise ValueError("source-indices must be unique non-negative values")
    if args.maximum_stereo_delta_ms <= 0.0:
        raise ValueError("maximum-stereo-delta-ms must be positive")

    manifest = {
        int(record["tick"]): record
        for record in load_jsonl(dataset / "manifest.jsonl")
    }
    frame_reports = []
    for source_index in args.source_indices:
        if source_index not in manifest:
            raise IndexError(f"Source frame {source_index} is absent")
        descriptors = {
            image["camera"]: image for image in manifest[source_index]["images"]
        }
        cam0_descriptor = descriptors["cam0"]
        cam1_descriptor = descriptors["cam1"]
        cam0_time = int(cam0_descriptor["sensor_time_ns"])
        cam1_time = int(cam1_descriptor["sensor_time_ns"])
        stereo_delta_ms = abs(cam0_time - cam1_time) / 1.0e6
        if stereo_delta_ms > args.maximum_stereo_delta_ms:
            raise ValueError(
                f"Source {source_index} stereo delta {stereo_delta_ms:.3f} ms "
                f"exceeds {args.maximum_stereo_delta_ms:.3f} ms"
            )
        cam0_path = (dataset / cam0_descriptor["path"]).resolve()
        cam1_path = (dataset / cam1_descriptor["path"]).resolve()
        cam0 = cv2.imread(str(cam0_path), cv2.IMREAD_COLOR)
        cam1 = cv2.imread(str(cam1_path), cv2.IMREAD_COLOR)
        if cam0 is None or cam1 is None or cam0.shape != cam1.shape:
            raise RuntimeError(f"Invalid stereo pair for source {source_index}")

        geometry = match_features(cam0, cam1)
        geometry.pop("_left_inliers")
        geometry.pop("_right_inliers")
        median_vertical = geometry["absolute_vertical_error_px"]["p50"]
        p95_vertical = geometry["absolute_vertical_error_px"]["p95"]
        positive_ratio = geometry["positive_disparity_ratio"]
        frame_reports.append(
            {
                "source_index": source_index,
                "stereo_delta_ms": stereo_delta_ms,
                "shape_hwc": list(cam0.shape),
                "cam0_path": str(cam0_path),
                "cam1_path": str(cam1_path),
                "geometry": geometry,
                "rectified_geometry_gate": {
                    "median_absolute_vertical_error_below_0_5px": (
                        median_vertical < 0.5
                    ),
                    "p95_absolute_vertical_error_below_1px": p95_vertical < 1.0,
                    "positive_disparity_ratio_above_0_95": positive_ratio > 0.95,
                },
            }
        )
        frame_reports[-1]["rectified_geometry_gate"]["passed"] = all(
            frame_reports[-1]["rectified_geometry_gate"].values()
        )
        print(
            f"{source_index:06d} dt={stereo_delta_ms:.3f}ms "
            f"|dy| p50={median_vertical:.3f}px p95={p95_vertical:.3f}px "
            f"positive={positive_ratio:.3f}",
            flush=True,
        )

    median_vertical_values = np.asarray(
        [
            frame["geometry"]["absolute_vertical_error_px"]["p50"]
            for frame in frame_reports
        ],
        dtype=np.float64,
    )
    p95_vertical_values = np.asarray(
        [
            frame["geometry"]["absolute_vertical_error_px"]["p95"]
            for frame in frame_reports
        ],
        dtype=np.float64,
    )
    positive_values = np.asarray(
        [
            frame["geometry"]["positive_disparity_ratio"]
            for frame in frame_reports
        ],
        dtype=np.float64,
    )
    report = {
        "contract": (
            "Camera-only audit of original recorded 2d_rect pixels; no resize, "
            "crop, rotation, remap, depth inference, or LiDAR input"
        ),
        "dataset": str(dataset),
        "source_indices": args.source_indices,
        "maximum_stereo_delta_ms": args.maximum_stereo_delta_ms,
        "expected_geometry": {
            "disparity_definition": "x_cam0 - x_cam1",
            "median_absolute_vertical_error_below_px": 0.5,
            "p95_absolute_vertical_error_below_px": 1.0,
            "positive_disparity_ratio_above": 0.95,
        },
        "aggregate": {
            "frame_count": len(frame_reports),
            "passed_frame_count": sum(
                frame["rectified_geometry_gate"]["passed"]
                for frame in frame_reports
            ),
            "median_of_frame_median_absolute_vertical_error_px": float(
                np.median(median_vertical_values)
            ),
            "median_of_frame_p95_absolute_vertical_error_px": float(
                np.median(p95_vertical_values)
            ),
            "median_positive_disparity_ratio": float(
                np.median(positive_values)
            ),
            "minimum_positive_disparity_ratio": float(positive_values.min()),
            "maximum_positive_disparity_ratio": float(positive_values.max()),
        },
        "frames": frame_reports,
        "interpretation": (
            "Failure across low-time-delta frames indicates that recorded 2d_rect "
            "pixels do not obey the supplied horizontal, positive-disparity "
            "rectified-stereo model. It does not identify whether image labels, "
            "the calibration export, or the upstream rectification producer is "
            "the source of the mismatch."
        ),
    }

    output.mkdir(parents=True)
    (output / "rectified_stereo_batch_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    x = np.arange(len(frame_reports))
    labels = [str(frame["source_index"]) for frame in frame_reports]
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    axes[0].plot(x, median_vertical_values, "o-", label="median |vertical error|")
    axes[0].plot(x, p95_vertical_values, "s-", label="p95 |vertical error|")
    axes[0].axhline(0.5, color="#31a354", linestyle="--", label="median gate")
    axes[0].axhline(1.0, color="#d7301f", linestyle="--", label="p95 gate")
    axes[0].set_ylabel("pixels")
    axes[0].set_title("Recorded cam0/cam1 epipolar geometry")
    axes[0].legend()
    axes[1].plot(x, positive_values, "o-", color="#756bb1")
    axes[1].axhline(0.95, color="#d7301f", linestyle="--", label="positive gate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("positive disparity ratio")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x, labels, rotation=45)
        axis.grid(alpha=0.25)
    figure.savefig(output / "rectified_stereo_batch_audit.png", dpi=170)
    plt.close(figure)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
