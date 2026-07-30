#!/usr/bin/env python3
"""Create a metadata-only subset of an already prepared stereo dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-indices", required=True, nargs="+", type=int)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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
    tick_index = load_json(dataset / "tick_index.json")
    frames_by_source = {
        int(frame["source_idx"]): frame for frame in tick_index["frames"]
    }
    missing = set(args.source_indices) - set(frames_by_source)
    if missing:
        raise ValueError(f"Source frames are absent: {sorted(missing)}")

    output.mkdir(parents=True)
    selected_frames = []
    lineage_frames = []
    source_pose_rows = (
        dataset / "pose" / "poses.txt"
    ).read_text().splitlines()
    selected_pose_rows = []
    artifact_directories = [
        path
        for path in dataset.iterdir()
        if path.is_dir()
        and path.name not in {"rgb", "stereo_right", "pose"}
    ]
    for output_index, source_index in enumerate(args.source_indices):
        source_frame = frames_by_source[source_index]
        source_pose_row = int(source_frame["pose_row"])
        if not 0 <= source_pose_row < len(source_pose_rows):
            raise ValueError(
                f"Pose row is outside the prepared pose file: {source_pose_row}"
            )
        frame = source_frame.copy()
        frame["idx"] = output_index
        frame["pose_row"] = output_index
        frame["selected_from_prepared_output_index"] = int(source_frame["idx"])
        selected_frames.append(frame)
        selected_pose_rows.append(source_pose_rows[source_pose_row])
        prepared_output_index = int(source_frame["idx"])
        for source_directory in artifact_directories:
            matches = list(source_directory.glob(f"{prepared_output_index:08d}.*"))
            if len(matches) > 1:
                raise ValueError(
                    "Multiple frame artifacts share a stem in "
                    f"{source_directory}: {prepared_output_index:08d}"
                )
            if not matches:
                continue
            destination_directory = output / source_directory.name
            destination_directory.mkdir(parents=True, exist_ok=True)
            destination = destination_directory / (
                f"{output_index:08d}{matches[0].suffix}"
            )
            destination.symlink_to(matches[0].resolve())
        lineage_frames.append(
            {
                "output_index": output_index,
                "source_index": source_index,
                "prepared_output_index": int(source_frame["idx"]),
                "cam0": source_frame["cam0"],
                "cam1": source_frame["cam1"],
                "stereo_delta_ms": float(source_frame["stereo_delta_ms"]),
            }
        )

    subset_index = tick_index.copy()
    subset_index["parent_prepared_dataset"] = str(dataset)
    subset_index["selection_policy"] = "metadata-only absolute image references"
    subset_index["frames"] = selected_frames
    (output / "tick_index.json").write_text(
        json.dumps(subset_index, indent=2) + "\n"
    )
    shutil.copy2(dataset / "camera_info.json", output / "camera_info.json")
    pose_directory = output / "pose"
    pose_directory.mkdir()
    (pose_directory / "poses.txt").write_text(
        "\n".join(selected_pose_rows) + "\n"
    )
    for metadata_name in (
        "pinhole_preparation_report.json",
        "source_manifest.json",
        "fast_foundation_stereo_run.json",
        "foundation_stereo_run.json",
    ):
        source_metadata = dataset / metadata_name
        if source_metadata.is_file():
            shutil.copy2(source_metadata, output / metadata_name)
    report = {
        "contract": (
            "Metadata-only selected-frame view; prepared image pixels are "
            "referenced without another decode, warp, or re-encode"
        ),
        "parent_prepared_dataset": str(dataset),
        "source_indices": args.source_indices,
        "frames": lineage_frames,
    }
    (output / "input_integrity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
