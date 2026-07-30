#!/usr/bin/env python3
"""Create a prepared stereo subset that only keeps geometry-passing frames."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source-indices", required=True, type=int, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def frame_source_index(frame: dict) -> int:
    for key in ("source_idx", "source_index", "tick", "cam0_source_idx"):
        if key in frame:
            return int(frame[key])
    raise KeyError(f"Frame lacks source index: {frame}")


def copy_named(source_root: Path, output_root: Path, relative_or_abs: str, new_name: str) -> str:
    src = Path(relative_or_abs)
    if not src.is_file():
        src = source_root / relative_or_abs
    if not src.is_file():
        raise FileNotFoundError(relative_or_abs)
    destination = output_root / new_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, destination)
    return str(destination)


def main() -> None:
    args = parse_args()
    source = args.dataset.resolve()
    output = args.output.resolve()
    keep = set(int(index) for index in args.source_indices)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for name in ("rgb", "stereo_right", "depth", "pose"):
        (output / name).mkdir(parents=True, exist_ok=True)

    tick = json.loads((source / "tick_index.json").read_text())
    original_frames = list(tick["frames"])
    poses_path = source / "pose" / "poses.txt"
    stamps_path = source / "pose" / "pose_timestamps_ns.txt"
    poses = poses_path.read_text().splitlines() if poses_path.is_file() else []
    stamps = stamps_path.read_text().splitlines() if stamps_path.is_file() else []

    kept_frames = []
    kept_poses = []
    kept_stamps = []
    for ordinal, frame in enumerate(original_frames):
        source_index = frame_source_index(frame)
        if source_index not in keep:
            continue
        new_idx = len(kept_frames)
        new_frame = dict(frame)
        new_frame["idx"] = new_idx
        new_frame["pose_row"] = new_idx
        if "cam0" in frame:
            new_frame["cam0"] = copy_named(
                source,
                output,
                frame["cam0"],
                f"rgb/{new_idx:08d}.png",
            )
        if "cam1" in frame:
            new_frame["cam1"] = copy_named(
                source,
                output,
                frame["cam1"],
                f"stereo_right/{new_idx:08d}.png",
            )
        depth_candidates = [
            source / "depth" / f"{ordinal:08d}.png",
            source / "depth" / f"{source_index:06d}.png",
            source / "depth" / f"{source_index:08d}.png",
        ]
        for candidate in depth_candidates:
            if candidate.is_file():
                copy_named(source, output, str(candidate), f"depth/{new_idx:08d}.png")
                new_frame["depth"] = str(output / "depth" / f"{new_idx:08d}.png")
                break
        pose_row = int(frame.get("pose_row", ordinal))
        if poses:
            kept_poses.append(poses[pose_row])
        if stamps:
            kept_stamps.append(stamps[pose_row])
        kept_frames.append(new_frame)

    if not kept_frames:
        raise ValueError("No frames matched the requested source indices")
    if poses and len(kept_poses) != len(kept_frames):
        raise ValueError("Pose subset size mismatch")
    if stamps and len(kept_stamps) != len(kept_frames):
        raise ValueError("Timestamp subset size mismatch")

    if kept_poses:
        (output / "pose" / "poses.txt").write_text("\n".join(kept_poses) + "\n")
    if kept_stamps:
        (output / "pose" / "pose_timestamps_ns.txt").write_text(
            "\n".join(kept_stamps) + "\n"
        )

    for name in (
        "camera_info.json",
        "source_manifest.json",
        "pinhole_preparation_report.json",
        "foundation_stereo_run.json",
        "fast_foundation_stereo_run.json",
        "keyframe_selection_report.json",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)

    new_tick = dict(tick)
    new_tick["frames"] = kept_frames
    new_tick["frame_count"] = len(kept_frames)
    new_tick["subset_of"] = str(source)
    new_tick["kept_source_indices"] = [
        frame_source_index(frame) for frame in kept_frames
    ]
    (output / "tick_index.json").write_text(
        json.dumps(new_tick, indent=2, ensure_ascii=False) + "\n"
    )
    subset_report = {
        "schema": "daaam.g1_prepared_stereo_subset.v1",
        "source_dataset": str(source),
        "output_dataset": str(output),
        "requested_source_indices": sorted(keep),
        "kept_frame_count": len(kept_frames),
        "kept_source_indices": new_tick["kept_source_indices"],
    }
    (output / "subset_report.json").write_text(
        json.dumps(subset_report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(subset_report, indent=2))


if __name__ == "__main__":
    main()
