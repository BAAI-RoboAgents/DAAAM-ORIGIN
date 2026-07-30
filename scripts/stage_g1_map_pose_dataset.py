#!/usr/bin/env python3
"""Stage a G1 capture with manifest-embedded map_T_base_link poses."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequence", default="000000")
    args = parser.parse_args()

    src = args.src.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing staging path: {output}")
    output.mkdir(parents=True)

    for name in (
        "2d_rect",
        "3d_raw",
        "calibrations",
        "lidar",
        "poses",
        "timestamps",
        "manifest.json",
        "manifest.jsonl",
        "quality_report.json",
    ):
        source = src / name
        if source.exists():
            (output / name).symlink_to(source, target_is_directory=source.is_dir())

    source_state = src / "state"
    staged_state = output / "state"
    staged_state.mkdir()
    for sequence_dir in source_state.iterdir():
        if sequence_dir.name != args.sequence:
            (staged_state / sequence_dir.name).symlink_to(
                sequence_dir, target_is_directory=True
            )
    staged_sequence = staged_state / args.sequence
    staged_sequence.mkdir()
    for source in (source_state / args.sequence).iterdir():
        (staged_sequence / source.name).symlink_to(
            source, target_is_directory=source.is_dir()
        )

    records = []
    previous_timestamp = None
    with (src / "manifest.jsonl").open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            pose = record["poses"]["values"]["map"]
            if pose["target_frame"] != "map" or pose["source_frame"] != "base_link":
                raise ValueError(f"Invalid map pose direction at line {line_number}")
            timestamp_ns = int(pose["timestamp_ns"])
            if previous_timestamp is not None and timestamp_ns <= previous_timestamp:
                raise ValueError(f"Map pose time is not increasing at line {line_number}")
            previous_timestamp = timestamp_ns
            records.append(
                {
                    "tick": int(record["tick"]),
                    "sensor_time_ns": timestamp_ns,
                    "target_frame": "map",
                    "source_frame": "base_link",
                    "pose": {
                        "target_frame": "map",
                        "source_frame": "base_link",
                        "timestamp_ns": timestamp_ns,
                        "position": pose["position"],
                        "orientation_xyzw": pose["orientation_xyzw"],
                    },
                }
            )

    if not records:
        raise ValueError("No manifest-embedded map poses found")
    map_pose_path = staged_sequence / "map_pose.jsonl"
    with map_pose_path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "source": str(src),
                "output": str(output),
                "map_pose_path": str(map_pose_path),
                "records": len(records),
                "first_timestamp_ns": records[0]["sensor_time_ns"],
                "last_timestamp_ns": records[-1]["sensor_time_ns"],
                "pose_direction": "map_T_base_link",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
