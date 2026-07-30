#!/usr/bin/env python3
"""Compare both stereo camera orders while retaining every match overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return parser.parse_args()


def render_matches(
    first: np.ndarray,
    second: np.ndarray,
    first_points: np.ndarray,
    second_points: np.ndarray,
    label: str,
) -> np.ndarray:
    canvas = np.hstack((first, second))
    width = first.shape[1]
    if len(first_points):
        indices = np.linspace(
            0, len(first_points) - 1, min(240, len(first_points))
        ).astype(int)
        for index in indices:
            point_a = np.rint(first_points[index]).astype(int)
            point_b = np.rint(second_points[index]).astype(int)
            disparity = float(point_a[0] - point_b[0])
            color = (40, 200, 40) if disparity > 0 else (30, 30, 230)
            cv2.line(
                canvas,
                tuple(point_a),
                (int(point_b[0] + width), int(point_b[1])),
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        canvas,
        label,
        (24, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    output.mkdir(parents=True)
    normal_directory = output / "cam0_reference"
    reverse_directory = output / "cam1_reference"
    normal_directory.mkdir()
    reverse_directory.mkdir()
    index = json.loads((dataset / "tick_index.json").read_text())
    reports = []
    for frame in index["frames"]:
        left = cv2.imread(str(frame["cam0"]), cv2.IMREAD_COLOR)
        right = cv2.imread(str(frame["cam1"]), cv2.IMREAD_COLOR)
        if left is None or right is None or left.shape != right.shape:
            raise RuntimeError(f"Invalid stereo frame: {frame['source_idx']}")
        normal = match_features(left, right)
        normal_left = normal.pop("_left_inliers")
        normal_right = normal.pop("_right_inliers")
        reverse = match_features(right, left)
        reverse_left = reverse.pop("_left_inliers")
        reverse_right = reverse.pop("_right_inliers")
        source_index = int(frame["source_idx"])
        normal_path = normal_directory / f"{source_index:06d}.jpg"
        reverse_path = reverse_directory / f"{source_index:06d}.jpg"
        normal_canvas = render_matches(
            left,
            right,
            normal_left,
            normal_right,
            (
                f"cam0->cam1 source={source_index} "
                f"positive={normal['positive_disparity_ratio']:.3f}"
            ),
        )
        reverse_canvas = render_matches(
            right,
            left,
            reverse_left,
            reverse_right,
            (
                f"cam1->cam0 source={source_index} "
                f"positive={reverse['positive_disparity_ratio']:.3f}"
            ),
        )
        for path, canvas in (
            (normal_path, normal_canvas),
            (reverse_path, reverse_canvas),
        ):
            if not cv2.imwrite(
                str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]
            ):
                raise RuntimeError(f"Could not write {path}")
        reports.append(
            {
                "source_index": source_index,
                "cam0_reference": normal,
                "cam1_reference": reverse,
                "cam0_reference_visualization": str(normal_path),
                "cam1_reference_visualization": str(reverse_path),
            }
        )

    source_indices = np.asarray([record["source_index"] for record in reports])
    normal_positive = np.asarray(
        [
            record["cam0_reference"]["positive_disparity_ratio"]
            for record in reports
        ]
    )
    reverse_positive = np.asarray(
        [
            record["cam1_reference"]["positive_disparity_ratio"]
            for record in reports
        ]
    )
    figure, axis = plt.subplots(figsize=(15, 5), constrained_layout=True)
    axis.plot(source_indices, normal_positive, label="cam0 reference", marker=".")
    axis.plot(source_indices, reverse_positive, label="cam1 reference", marker=".")
    axis.axhline(0.95, color="#31a354", linestyle="--", label="0.95 geometry gate")
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("source frame")
    axis.set_ylabel("positive disparity ratio")
    axis.set_title("Stereo camera-order feature geometry")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output / "camera_order_timeline.png", dpi=180)
    plt.close(figure)

    report = {
        "schema": "daaam.stereo_camera_order_feature_audit.v1",
        "dataset": str(dataset),
        "aggregate": {
            "cam0_reference_median_positive_disparity_ratio": float(
                np.median(normal_positive)
            ),
            "cam1_reference_median_positive_disparity_ratio": float(
                np.median(reverse_positive)
            ),
            "recommended_reference_camera": (
                "cam0"
                if np.median(normal_positive) >= np.median(reverse_positive)
                else "cam1"
            ),
        },
        "frames": reports,
    }
    (output / "camera_order_feature_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
