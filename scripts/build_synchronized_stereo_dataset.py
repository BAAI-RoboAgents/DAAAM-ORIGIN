#!/usr/bin/env python3
"""Build a timestamp-filtered RGB-D dataset with global G1 camera poses."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-delta-ms", type=float, default=10.0)
    parser.add_argument(
        "--camera-quaternion-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Storage order used by head_camera orientation values.",
    )
    return parser.parse_args()


def ensure_symlink(link: Path, target: Path):
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() == target:
            return
        raise RuntimeError(f"Existing symlink points elsewhere: {link}")
    if link.exists():
        raise RuntimeError(f"Refusing to replace existing path: {link}")
    link.symlink_to(target)


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def camera_timestamps(records, camera):
    values = []
    for record in records:
        images = {image["camera"]: image for image in record.get("images", [])}
        if camera not in images:
            raise ValueError(f"Missing {camera} at tick {record.get('tick')}")
        values.append(int(images[camera]["sensor_time_ns"]))
    return np.asarray(values, dtype=np.int64)


def monotonic_matches(left_ts, right_ts, threshold_ns):
    """Maximum-cardinality monotonic matching for sorted timestamp streams."""
    left_idx = right_idx = 0
    matches = []
    skipped_left = []
    skipped_right = []
    while left_idx < len(left_ts) and right_idx < len(right_ts):
        delta = int(left_ts[left_idx] - right_ts[right_idx])
        if abs(delta) <= threshold_ns:
            matches.append((left_idx, right_idx, abs(delta)))
            left_idx += 1
            right_idx += 1
        elif delta < 0:
            skipped_left.append(left_idx)
            left_idx += 1
        else:
            skipped_right.append(right_idx)
            right_idx += 1
    skipped_left.extend(range(left_idx, len(left_ts)))
    skipped_right.extend(range(right_idx, len(right_ts)))
    return matches, skipped_left, skipped_right


def unique_pose_samples(timestamps, positions, quaternions):
    """Drop duplicate timestamps while retaining the most recent sample."""
    keep = np.r_[np.diff(timestamps) != 0, True]
    timestamps = timestamps[keep]
    positions = positions[keep]
    quaternions = quaternions[keep]
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("Pose timestamps are not monotonic")
    return timestamps, positions, quaternions


def interpolate_poses(timestamps, positions, quaternions, target_timestamps):
    timestamps, positions, quaternions = unique_pose_samples(
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(quaternions, dtype=np.float64),
    )
    targets = np.asarray(target_timestamps, dtype=np.float64)
    clipped = np.clip(targets, timestamps[0], timestamps[-1])
    interpolated_positions = np.column_stack(
        [np.interp(clipped, timestamps, positions[:, axis]) for axis in range(3)]
    )
    interpolated_rotations = Slerp(
        timestamps, Rotation.from_quat(quaternions)
    )(clipped)
    return interpolated_positions, interpolated_rotations, int(np.sum(clipped != targets))


def compose_global_camera_poses(
    raw_source: Path, target_timestamps, camera_quaternion_order="xyzw"
):
    odom_records = load_jsonl(raw_source / "state/000000/odom.jsonl")
    odom_timestamps = [record["odom"]["timestamp_ns"] for record in odom_records]
    odom_positions = [record["odom"]["position"] for record in odom_records]
    odom_quaternions = [record["odom"]["orientation"] for record in odom_records]

    aux_records = load_jsonl(
        raw_source / "poses/dense_global/000000/aux_poses.jsonl"
    )
    camera_samples = [record["poses"]["head_camera"] for record in aux_records]
    camera_timestamps_ns = [sample["timestamp_ns"] for sample in camera_samples]
    camera_positions = [sample["position"] for sample in camera_samples]
    camera_quaternions = [sample["orientation_xyzw"] for sample in camera_samples]
    if camera_quaternion_order == "wxyz":
        camera_quaternions = [
            [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
            for quaternion in camera_quaternions
        ]
    elif camera_quaternion_order != "xyzw":
        raise ValueError(
            f"Unsupported camera quaternion order: {camera_quaternion_order}"
        )

    world_t_base, world_r_base, odom_clamped = interpolate_poses(
        odom_timestamps, odom_positions, odom_quaternions, target_timestamps
    )
    base_t_camera, base_r_camera, camera_clamped = interpolate_poses(
        camera_timestamps_ns,
        camera_positions,
        camera_quaternions,
        target_timestamps,
    )

    matrices = []
    for t_world_base, r_world_base, t_base_camera, r_base_camera in zip(
        world_t_base, world_r_base, base_t_camera, base_r_camera
    ):
        rotation = r_world_base * r_base_camera
        translation = t_world_base + r_world_base.apply(t_base_camera)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = translation
        matrices.append(matrix)
    return matrices, odom_clamped, camera_clamped


def main():
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    tick_index = json.loads((dataset / "tick_index.json").read_text())
    raw_source = Path(tick_index["source"]).resolve()
    records = load_jsonl(raw_source / "manifest.jsonl")
    left_ts = camera_timestamps(records, "cam0")
    right_ts = camera_timestamps(records, "cam1")
    if len(left_ts) != len(tick_index["frames"]):
        raise ValueError("Capture manifest and prepared dataset counts differ")

    threshold_ns = int(round(args.max_delta_ms * 1e6))
    matches, skipped_left, skipped_right = monotonic_matches(
        left_ts, right_ts, threshold_ns
    )
    reindexed = [(left, right) for left, right, _ in matches if left != right]
    if reindexed:
        raise RuntimeError(
            "Reliable matches change stereo indices, so existing depth cannot be reused: "
            f"{reindexed[:10]}. Rerun FoundationStereo on the matched pairs instead."
        )
    if not matches:
        raise ValueError("No synchronized stereo pairs found")

    for directory in (output / "rgb", output / "depth", output / "pose"):
        directory.mkdir(parents=True, exist_ok=True)

    selected_left = [left for left, _, _ in matches]
    selected_timestamps = left_ts[selected_left]
    global_poses, odom_clamped, camera_clamped = compose_global_camera_poses(
        raw_source,
        selected_timestamps,
        camera_quaternion_order=args.camera_quaternion_order,
    )
    origin_ns = int(selected_timestamps[0])
    output_frames = []
    for output_idx, ((left_idx, right_idx, delta_ns), pose) in enumerate(
        zip(matches, global_poses)
    ):
        source_frame = tick_index["frames"][left_idx]
        rgb_source = Path(source_frame["cam0"])
        depth_source = dataset / "depth" / f"{left_idx:08d}.png"
        if not depth_source.exists():
            raise FileNotFoundError(depth_source)
        ensure_symlink(output / "rgb" / f"{output_idx:08d}.png", rgb_source)
        ensure_symlink(output / "depth" / f"{output_idx:08d}.png", depth_source)
        output_frames.append(
            {
                "idx": output_idx,
                "source_idx": left_idx,
                "cam0_source_idx": left_idx,
                "cam1_source_idx": right_idx,
                "pose_row": output_idx,
                "cam0": str(rgb_source.resolve()),
                "cam1": str(Path(tick_index["frames"][right_idx]["cam1"]).resolve()),
                "depth": str(depth_source.resolve()),
                "timestamp": (int(left_ts[left_idx]) - origin_ns) / 1e9,
                "cam0_sensor_time_ns": int(left_ts[left_idx]),
                "cam1_sensor_time_ns": int(right_ts[right_idx]),
                "sensor_time_ns": int(left_ts[left_idx]),
                "pose_sensor_time_ns": int(left_ts[left_idx]),
                "stereo_delta_ms": delta_ns / 1e6,
            }
        )

    pose_text = "".join(
        " ".join(f"{value:.12g}" for value in pose.reshape(-1)) + "\n"
        for pose in global_poses
    )
    pose_path = output / "pose/poses.txt"
    if pose_path.exists() and pose_path.read_text() != pose_text:
        raise RuntimeError(f"Refusing to replace different pose file: {pose_path}")
    pose_path.write_text(pose_text)
    pose_timestamp_path = output / "pose/pose_timestamps_ns.txt"
    pose_timestamp_text = "".join(
        f"{int(timestamp)}\n" for timestamp in selected_timestamps
    )
    if (
        pose_timestamp_path.exists()
        and pose_timestamp_path.read_text() != pose_timestamp_text
    ):
        raise RuntimeError(
            f"Refusing to replace different pose timestamps: {pose_timestamp_path}"
        )
    pose_timestamp_path.write_text(pose_timestamp_text)
    ensure_symlink(output / "camera_info.json", dataset / "camera_info.json")
    if (dataset / "source_manifest.json").exists():
        ensure_symlink(
            output / "source_manifest.json", dataset / "source_manifest.json"
        )

    filtered_tick_index = {
        key: value for key, value in tick_index.items() if key != "frames"
    }
    filtered_tick_index.update(
        {
            "source_dataset": str(dataset),
            "sync_threshold_ms": args.max_delta_ms,
            "pose_frame": "odom",
            "pose_composition": "odom_T_base_link @ base_link_T_head_camera",
            "time_origin_ns": origin_ns,
            "timebase": {
                "clock": "sensor_time_ns",
                "unit": "ns",
                "timestamp_definition": "(sensor_time_ns - time_origin_ns) / 1e9",
            },
            "pose_time_alignment": {
                "method": "interpolate_odom_and_head_camera_at_cam0_sensor_time_ns",
                "pose_timestamp_file": "pose/pose_timestamps_ns.txt",
                "pose_row_field": "pose_row",
            },
            "frames": output_frames,
        }
    )
    (output / "tick_index.json").write_text(
        json.dumps(filtered_tick_index, indent=2) + "\n"
    )

    skipped_records = []
    for index in skipped_left:
        skipped_records.append(
            {
                "source_idx": index,
                "cam0_sensor_time_ns": int(left_ts[index]),
                "cam1_sensor_time_ns_same_tick": int(right_ts[index]),
                "same_tick_delta_ms": abs(int(left_ts[index] - right_ts[index]))
                / 1e6,
                "reason": "no one-to-one cam1 match within threshold",
            }
        )
    translations = np.asarray([pose[:3, 3] for pose in global_poses])
    report = {
        "source_dataset": str(dataset),
        "raw_source": str(raw_source),
        "threshold_ms": args.max_delta_ms,
        "source_pairs": len(left_ts),
        "matched_pairs": len(matches),
        "skipped_cam0": len(skipped_left),
        "skipped_cam1": len(skipped_right),
        "reindexed_pairs": len(reindexed),
        "max_matched_delta_ms": max(delta for _, _, delta in matches) / 1e6,
        "odom_interpolation_clamped": odom_clamped,
        "camera_pose_interpolation_clamped": camera_clamped,
        "global_translation_first_m": translations[0].tolist(),
        "global_translation_last_m": translations[-1].tolist(),
        "global_translation_path_length_m": float(
            np.linalg.norm(np.diff(translations, axis=0), axis=1).sum()
        ),
        "skipped_frames": skipped_records,
        "skipped_cam1_indices": skipped_right,
    }
    (output / "sync_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Built {output}: matched={len(matches)}/{len(left_ts)}, "
        f"skipped={len(skipped_left)}, max_delta={report['max_matched_delta_ms']:.3f}ms\n"
        f"  global camera translation: {translations[0].round(3).tolist()} -> "
        f"{translations[-1].round(3).tolist()}, "
        f"path={report['global_translation_path_length_m']:.3f}m"
    )


if __name__ == "__main__":
    main()
