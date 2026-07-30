#!/usr/bin/env python3
"""Freeze a contiguous source-frame RGB-D window as a reindexed, linked dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any


FRAME_DIRECTORIES = (
    "depth",
    "depth_confidence",
    "depth_consistency",
    "depth_metadata",
    "depth_occlusion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-frame-start", required=True, type=int)
    parser.add_argument("--source-frame-end", required=True, type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def source_frame_index(frame: dict[str, Any]) -> int:
    for key in ("source_frame_idx", "source_idx", "source_index", "tick"):
        if key in frame:
            return int(frame[key])
    raise KeyError(f"Frame has no source index: {frame}")


def link_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    output.mkdir(parents=True)

    tick_path = dataset / "tick_index.json"
    tick = json.loads(tick_path.read_text(encoding="utf-8"))
    selected = [
        frame
        for frame in tick["frames"]
        if args.source_frame_start
        <= source_frame_index(frame)
        <= args.source_frame_end
    ]
    if not selected:
        raise RuntimeError("No frames overlap the requested source-frame range")

    pose_path = dataset / "pose/poses.txt"
    timestamp_path = dataset / "pose/pose_timestamps_ns.txt"
    poses = pose_path.read_text(encoding="utf-8").splitlines()
    timestamps = timestamp_path.read_text(encoding="utf-8").splitlines()
    output_poses: list[str] = []
    output_timestamps: list[str] = []
    output_frames: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    for directory in (*FRAME_DIRECTORIES, "rgb", "stereo_right", "pose"):
        (output / directory).mkdir()
    for new_index, source_frame in enumerate(selected):
        old_index = int(source_frame["idx"])
        pose_row = int(source_frame["pose_row"])
        new_frame = dict(source_frame)
        new_frame["idx"] = new_index
        new_frame["pose_row"] = new_index
        sensor_time_ns = int(source_frame["sensor_time_ns"])
        if new_index == 0:
            time_origin_ns = sensor_time_ns
        new_frame["timestamp"] = (sensor_time_ns - time_origin_ns) / 1.0e9

        rgb_source = Path(source_frame["cam0"])
        right_source = Path(source_frame["cam1"])
        rgb_destination = output / "rgb" / f"{new_index:08d}.png"
        right_destination = output / "stereo_right" / f"{new_index:08d}.png"
        link_file(rgb_source, rgb_destination)
        link_file(right_source, right_destination)
        new_frame["cam0"] = str(rgb_destination)
        new_frame["cam1"] = str(right_destination)
        links.extend(
            [
                {
                    "path": str(rgb_destination.relative_to(output)),
                    "target": str(rgb_source.resolve()),
                },
                {
                    "path": str(right_destination.relative_to(output)),
                    "target": str(right_source.resolve()),
                },
            ]
        )

        for directory in FRAME_DIRECTORIES:
            suffix = ".json" if directory == "depth_metadata" else ".png"
            source = dataset / directory / f"{old_index:08d}{suffix}"
            destination = output / directory / f"{new_index:08d}{suffix}"
            link_file(source, destination)
            links.append(
                {
                    "path": str(destination.relative_to(output)),
                    "target": str(source.resolve()),
                }
            )
        output_poses.append(poses[pose_row])
        output_timestamps.append(timestamps[pose_row])
        output_frames.append(new_frame)

    output_pose_path = output / "pose/poses.txt"
    output_timestamp_path = output / "pose/pose_timestamps_ns.txt"
    output_pose_path.write_text("\n".join(output_poses) + "\n", encoding="utf-8")
    output_timestamp_path.write_text(
        "\n".join(output_timestamps) + "\n", encoding="utf-8"
    )

    metadata_links: list[dict[str, Any]] = []
    for source in sorted(dataset.iterdir()):
        if source.name in {
            "tick_index.json",
            "pose",
            "rgb",
            "stereo_right",
            *FRAME_DIRECTORIES,
        }:
            continue
        destination = output / source.name
        if source.is_file() or source.is_symlink():
            destination.symlink_to(source.resolve())
            metadata_links.append(
                {
                    "path": str(destination.relative_to(output)),
                    "target": str(source.resolve()),
                }
            )

    new_tick = dict(tick)
    new_tick["time_origin_ns"] = time_origin_ns
    new_tick["frames"] = output_frames
    new_tick["frame_count"] = len(output_frames)
    new_tick["window_materialization"] = {
        "source_dataset": str(dataset),
        "source_tick_index_sha256": sha256_file(tick_path),
        "requested_source_frame_range_inclusive": [
            args.source_frame_start,
            args.source_frame_end,
        ],
        "source_geometry_frame_indices": [
            int(frame["idx"]) for frame in selected
        ],
        "source_frame_indices": [source_frame_index(frame) for frame in selected],
        "reindexed": True,
        "unmodified_frame_artifacts_linked_read_only": True,
    }
    output_tick_path = output / "tick_index.json"
    write_json(output_tick_path, new_tick)

    manifest = {
        "schema": "daaam.g1_rgbd_window.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset),
        "output_dataset": str(output),
        "requested_source_frame_range_inclusive": [
            args.source_frame_start,
            args.source_frame_end,
        ],
        "frame_count": len(output_frames),
        "source_geometry_frame_range": [
            int(selected[0]["idx"]),
            int(selected[-1]["idx"]),
        ],
        "source_frame_indices": [source_frame_index(frame) for frame in selected],
        "regular_files": {
            "tick_index.json": sha256_file(output_tick_path),
            "pose/poses.txt": sha256_file(output_pose_path),
            "pose/pose_timestamps_ns.txt": sha256_file(output_timestamp_path),
        },
        "frame_artifact_links": links,
        "metadata_links": metadata_links,
        "source_tick_index_sha256": sha256_file(tick_path),
        "source_pose_sha256": sha256_file(pose_path),
        "source_pose_timestamps_sha256": sha256_file(timestamp_path),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output / "window_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "frame_count": len(output_frames),
                "source_geometry_frame_range": manifest[
                    "source_geometry_frame_range"
                ],
                "frame_artifact_link_count": len(links),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
