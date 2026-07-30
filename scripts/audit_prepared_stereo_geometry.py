#!/usr/bin/env python3
"""Audit the epipolar geometry of a prepared stereo dataset."""

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
    parser.add_argument("--maximum-stereo-delta-ms", type=float, default=5.0)
    parser.add_argument("--source-indices", nargs="+", type=int)
    parser.add_argument(
        "--save-match-visualizations",
        action="store_true",
        help="Save per-frame inlier correspondence overlays for geometry diagnosis.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    tick_index = load_json(dataset / "tick_index.json")
    if tick_index.get("projection_model") != "pinhole":
        raise ValueError("Prepared dataset does not declare pinhole output")

    reports = []
    skipped = []
    requested_source_indices = (
        set(args.source_indices) if args.source_indices is not None else None
    )
    for frame in tick_index["frames"]:
        if (
            requested_source_indices is not None
            and int(frame["source_idx"]) not in requested_source_indices
        ):
            continue
        stereo_delta_ms = float(frame["stereo_delta_ms"])
        if stereo_delta_ms > args.maximum_stereo_delta_ms:
            skipped.append(
                {
                    "output_index": int(frame["idx"]),
                    "source_index": int(frame["source_idx"]),
                    "stereo_delta_ms": stereo_delta_ms,
                    "reason": "stereo_delta_above_limit",
                }
            )
            continue
        left = cv2.imread(str(frame["cam0"]), cv2.IMREAD_COLOR)
        right = cv2.imread(str(frame["cam1"]), cv2.IMREAD_COLOR)
        if left is None or right is None or left.shape != right.shape:
            raise RuntimeError(f"Invalid prepared frame {frame['idx']}")
        geometry = match_features(left, right)
        left_inliers = geometry.pop("_left_inliers")
        right_inliers = geometry.pop("_right_inliers")
        median_vertical = geometry["absolute_vertical_error_px"]["p50"]
        p95_vertical = geometry["absolute_vertical_error_px"]["p95"]
        positive_ratio = geometry["positive_disparity_ratio"]
        gate = {
            "median_absolute_vertical_error_below_0_5px": median_vertical < 0.5,
            "p95_absolute_vertical_error_below_1px": p95_vertical < 1.0,
            "positive_disparity_ratio_above_0_95": positive_ratio > 0.95,
        }
        gate["passed"] = all(gate.values())
        frame_report = {
                "output_index": int(frame["idx"]),
                "source_index": int(frame["source_idx"]),
                "stereo_delta_ms": stereo_delta_ms,
                "geometry": geometry,
                "rectified_geometry_gate": gate,
            }
        if args.save_match_visualizations:
            visualization_directory = output / "match_visualizations"
            visualization_directory.mkdir(parents=True, exist_ok=True)
            canvas = np.hstack((left, right))
            if len(left_inliers):
                sample_indices = np.linspace(
                    0,
                    len(left_inliers) - 1,
                    min(240, len(left_inliers)),
                ).astype(int)
                width = left.shape[1]
                for match_index in sample_indices:
                    left_point = np.rint(left_inliers[match_index]).astype(int)
                    right_point = np.rint(right_inliers[match_index]).astype(int)
                    color = (
                        (40, 210, 40)
                        if abs(left_point[1] - right_point[1]) < 1
                        else (30, 30, 230)
                    )
                    cv2.line(
                        canvas,
                        tuple(left_point),
                        (int(right_point[0] + width), int(right_point[1])),
                        color,
                        1,
                        cv2.LINE_AA,
                    )
            cv2.putText(
                canvas,
                (
                    f"src={int(frame['source_idx'])} dt={stereo_delta_ms:.3f}ms "
                    f"|dy| p50={median_vertical:.3f}px "
                    f"p95={p95_vertical:.3f}px pass={gate['passed']}"
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            visualization_path = (
                visualization_directory / f"{int(frame['source_idx']):06d}.jpg"
            )
            if not cv2.imwrite(
                str(visualization_path),
                canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            ):
                raise RuntimeError(
                    f"Could not write geometry visualization: {visualization_path}"
                )
            frame_report["match_visualization"] = str(visualization_path)
        reports.append(frame_report)
        print(
            f"{int(frame['source_idx']):06d} dt={stereo_delta_ms:.3f}ms "
            f"|dy| p50={median_vertical:.3f}px p95={p95_vertical:.3f}px "
            f"positive={positive_ratio:.3f} pass={gate['passed']}",
            flush=True,
        )
    if not reports:
        raise ValueError("No prepared frame passed the stereo-delta filter")
    if requested_source_indices is not None:
        found = {report["source_index"] for report in reports} | {
            report["source_index"] for report in skipped
        }
        missing = requested_source_indices - found
        if missing:
            raise ValueError(
                f"Requested source indices are absent from prepared data: {sorted(missing)}"
            )

    median_vertical = np.asarray(
        [
            report["geometry"]["absolute_vertical_error_px"]["p50"]
            for report in reports
        ]
    )
    p95_vertical = np.asarray(
        [
            report["geometry"]["absolute_vertical_error_px"]["p95"]
            for report in reports
        ]
    )
    positive_ratio = np.asarray(
        [report["geometry"]["positive_disparity_ratio"] for report in reports]
    )
    aggregate = {
        "evaluated_frame_count": len(reports),
        "skipped_frame_count": len(skipped),
        "passed_frame_count": sum(
            report["rectified_geometry_gate"]["passed"] for report in reports
        ),
        "median_of_frame_median_absolute_vertical_error_px": float(
            np.median(median_vertical)
        ),
        "median_of_frame_p95_absolute_vertical_error_px": float(
            np.median(p95_vertical)
        ),
        "median_positive_disparity_ratio": float(np.median(positive_ratio)),
    }
    report = {
        "contract": (
            "Camera-only geometry audit of prepared stereo pixels; no LiDAR"
        ),
        "dataset": str(dataset),
        "maximum_stereo_delta_ms": args.maximum_stereo_delta_ms,
        "expected_geometry": {
            "median_absolute_vertical_error_below_px": 0.5,
            "p95_absolute_vertical_error_below_px": 1.0,
            "positive_disparity_ratio_above": 0.95,
        },
        "aggregate": aggregate,
        "frames": reports,
        "skipped": skipped,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "prepared_stereo_geometry_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )

    x = np.arange(len(reports))
    labels = [str(report["source_index"]) for report in reports]
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    axes[0].plot(x, median_vertical, "o-", label="median |vertical error|")
    axes[0].plot(x, p95_vertical, "s-", label="p95 |vertical error|")
    axes[0].axhline(0.5, color="#31a354", linestyle="--", label="median gate")
    axes[0].axhline(1.0, color="#d7301f", linestyle="--", label="p95 gate")
    axes[0].set_ylabel("pixels")
    axes[0].set_title("Prepared virtual-pinhole stereo geometry")
    axes[0].legend()
    axes[1].plot(x, positive_ratio, "o-", color="#756bb1")
    axes[1].axhline(0.95, color="#d7301f", linestyle="--", label="positive gate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("positive disparity ratio")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x, labels, rotation=45)
        axis.grid(alpha=0.25)
    figure.savefig(output / "prepared_stereo_geometry_audit.png", dpi=170)
    plt.close(figure)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
