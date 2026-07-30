#!/usr/bin/env python3
"""Persist every keyframe decision and sequence-level diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dataset", required=True, type=Path)
    parser.add_argument("--selected-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_positions(dataset: Path) -> np.ndarray | None:
    path = dataset / "pose/poses.txt"
    if not path.is_file():
        return None
    matrices = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        values = np.asarray([float(value) for value in line.split()])
        if values.size == 12:
            matrix = np.eye(4)
            matrix[:3, :] = values.reshape(3, 4)
        elif values.size == 16:
            matrix = values.reshape(4, 4)
        else:
            raise ValueError(f"Unsupported pose row in {path}")
        matrices.append(matrix)
    return np.asarray([matrix[:3, 3] for matrix in matrices])


def main() -> None:
    args = parse_args()
    prepared = args.prepared_dataset.resolve()
    selected = args.selected_dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    output.mkdir(parents=True)
    frames_directory = output / "frame_decisions"
    contact_directory = output / "contact_sheets"
    frames_directory.mkdir()
    contact_directory.mkdir()

    prepared_index = load_json(prepared / "tick_index.json")
    selected_index = load_json(selected / "tick_index.json")
    selected_by_source = {
        int(frame["source_idx"]): frame for frame in selected_index["frames"]
    }
    frame_records = []
    thumbnails = []
    for position, frame in enumerate(prepared_index["frames"]):
        source_index = int(frame["source_idx"])
        selected_frame = selected_by_source.get(source_index)
        retained = selected_frame is not None
        reason = (
            str(selected_frame.get("selection_reason", "selected"))
            if selected_frame is not None
            else "not_selected"
        )
        image = cv2.imread(str(frame["cam0"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read keyframe source image: {frame['cam0']}")
        color = (40, 200, 40) if retained else (40, 40, 220)
        cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), color, 10)
        cv2.rectangle(image, (0, 0), (image.shape[1], 68), (0, 0, 0), -1)
        cv2.putText(
            image,
            (
                f"source={source_index} {'KEEP' if retained else 'DROP'} "
                f"reason={reason} dt={float(frame['stereo_delta_ms']):.3f}ms"
            ),
            (24, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
        frame_path = frames_directory / f"{source_index:06d}.jpg"
        if not cv2.imwrite(
            str(frame_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]
        ):
            raise RuntimeError(f"Could not write {frame_path}")
        thumbnails.append(cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA))
        frame_records.append(
            {
                "prepared_output_index": position,
                "source_index": source_index,
                "selected": retained,
                "selection_reason": reason,
                "stereo_delta_ms": float(frame["stereo_delta_ms"]),
                "visualization": str(frame_path),
            }
        )

    for start in range(0, len(thumbnails), 20):
        batch = thumbnails[start : start + 20]
        rows = []
        for row_start in range(0, len(batch), 5):
            row = batch[row_start : row_start + 5]
            while len(row) < 5:
                row.append(np.zeros_like(batch[0]))
            rows.append(np.hstack(row))
        sheet = np.vstack(rows)
        sheet_path = contact_directory / f"{start:04d}-{start + len(batch) - 1:04d}.jpg"
        if not cv2.imwrite(
            str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]
        ):
            raise RuntimeError(f"Could not write {sheet_path}")

    retained_mask = np.asarray([record["selected"] for record in frame_records])
    source_indices = np.asarray([record["source_index"] for record in frame_records])
    stereo_delta = np.asarray([record["stereo_delta_ms"] for record in frame_records])
    figure, axes = plt.subplots(2, 1, figsize=(16, 7), constrained_layout=True)
    axes[0].scatter(
        source_indices[~retained_mask],
        np.zeros(np.count_nonzero(~retained_mask)),
        c="#de2d26",
        label="dropped",
    )
    axes[0].scatter(
        source_indices[retained_mask],
        np.ones(np.count_nonzero(retained_mask)),
        c="#31a354",
        label="retained",
    )
    axes[0].set_yticks([0, 1], ["drop", "keep"])
    axes[0].set_title("Keyframe decision for every prepared frame")
    axes[0].legend()
    axes[1].plot(source_indices, stereo_delta, "o-", markersize=3)
    axes[1].axhline(10.0, color="#de2d26", linestyle="--", label="10 ms input gate")
    axes[1].set_xlabel("source frame")
    axes[1].set_ylabel("stereo delta (ms)")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(output / "selection_timeline.png", dpi=180)
    plt.close(figure)

    positions = load_positions(prepared)
    if positions is not None and len(positions) == len(frame_records):
        figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
        axis.plot(positions[:, 0], positions[:, 1], color="#969696", linewidth=1)
        axis.scatter(
            positions[~retained_mask, 0],
            positions[~retained_mask, 1],
            c="#de2d26",
            s=15,
            label="dropped",
        )
        axis.scatter(
            positions[retained_mask, 0],
            positions[retained_mask, 1],
            c="#31a354",
            s=26,
            label="retained",
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world y (m)")
        axis.set_title("Keyframe selection along trajectory")
        axis.legend()
        axis.grid(alpha=0.2)
        figure.savefig(output / "trajectory_selection.png", dpi=180)
        plt.close(figure)

    with (output / "selection_decisions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=frame_records[0].keys())
        writer.writeheader()
        writer.writerows(frame_records)
    report = {
        "schema": "daaam.keyframe_selection_visualization.v1",
        "prepared_dataset": str(prepared),
        "selected_dataset": str(selected),
        "prepared_frames": len(frame_records),
        "selected_frames": int(np.count_nonzero(retained_mask)),
        "frames": frame_records,
    }
    (output / "visualization_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
