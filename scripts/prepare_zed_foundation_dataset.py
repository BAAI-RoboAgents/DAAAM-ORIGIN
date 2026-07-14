#!/usr/bin/env python3
"""Prepare a rectified stereo sequence for FoundationStereo and DAAAM."""

import argparse
import json
import os
from pathlib import Path

import yaml


SUPPORTED_LAYOUTS = {"zed_sequence_v1", "capture4daaam_like"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--sequence", default="000000")
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


def parse_timestamps(lines):
    """Normalize either '<frame> <seconds>' or single-column nanoseconds."""
    tokens = [line.split() for line in lines]
    if all(len(parts) >= 2 for parts in tokens):
        return [float(parts[-1]) for parts in tokens]
    if not all(len(parts) == 1 for parts in tokens):
        raise ValueError("Inconsistent timestamp format")
    values = [float(parts[0]) for parts in tokens]
    if values and values[0] > 1e12:
        origin = values[0]
        return [(value - origin) / 1e9 for value in values]
    return values


def read_calibration(calibration_dir: Path, layout: str):
    cam0_calib = yaml.safe_load(
        (calibration_dir / "calib_cam0_intrinsics.yaml").read_text()
    )
    if layout == "zed_sequence_v1":
        extrinsic = yaml.safe_load(
            (calibration_dir / "calib_cam0_to_cam1.yaml").read_text()
        )["extrinsic_matrix"]
        K = cam0_calib["camera_matrix"]["data"]
        width = int(cam0_calib["image_width"])
        height = int(cam0_calib["image_height"])
        distortion = cam0_calib["distortion_coefficients"]["data"]
        translation = extrinsic["T"]
    else:
        cam1_calib = yaml.safe_load(
            (calibration_dir / "calib_cam1_intrinsics.yaml").read_text()
        )
        cam0 = cam0_calib["intrinsics"]
        cam1 = cam1_calib["intrinsics"]
        K = cam0["K"]
        width = int(cam0["width"])
        height = int(cam0["height"])
        distortion = cam0.get("D", [])
        if cam0["K"] != cam1["K"] or cam0["R"] != cam1["R"]:
            raise ValueError("G1 stereo images do not share rectified intrinsics")
        if not cam0.get("roi", {}).get("do_rectify") or not cam1.get("roi", {}).get("do_rectify"):
            raise ValueError("G1 stereo calibration does not declare rectified images")
        transform = cam1["T"]
        if len(transform) != 12:
            raise ValueError("Expected a 3x4 cam1 transform in calibration")
        translation = [transform[3], transform[7], transform[11]]

    baseline = float(sum(value * value for value in translation) ** 0.5)
    if baseline <= 0:
        raise ValueError(f"Invalid stereo baseline: {baseline}")
    return K, width, height, distortion, baseline


def read_stereo_sync(src: Path, expected_count: int):
    """Read per-frame camera timestamp deltas when capture metadata exists."""
    manifest_jsonl = src / "manifest.jsonl"
    if not manifest_jsonl.exists():
        return [None] * expected_count
    deltas = []
    for line in manifest_jsonl.read_text().splitlines():
        record = json.loads(line)
        stamps = {
            image["camera"]: int(image["sensor_time_ns"])
            for image in record.get("images", [])
        }
        if "cam0" not in stamps or "cam1" not in stamps:
            raise ValueError(f"Missing stereo timestamp at tick {record.get('tick')}")
        deltas.append(abs(stamps["cam0"] - stamps["cam1"]) / 1e6)
    if len(deltas) != expected_count:
        raise ValueError(
            f"manifest.jsonl count mismatch: {len(deltas)} != {expected_count}"
        )
    return deltas


def main():
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    seq = args.sequence

    manifest = json.loads((src / "manifest.json").read_text())
    layout = manifest.get("layout_version") or manifest.get("layout")
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(f"Unsupported layout: {layout}")

    cam0_dir = src / "2d_rect" / "cam0" / seq
    cam1_dir = src / "2d_rect" / "cam1" / seq
    pose_file = src / "poses" / "dense_global" / seq / "poses.txt"
    timestamp_file = src / "timestamps" / f"{seq}.txt"
    calibration_dir = src / "calibrations" / seq

    cam0_files = sorted(cam0_dir.glob("*.png"))
    cam1_files = sorted(cam1_dir.glob("*.png"))
    pose_lines = [line for line in pose_file.read_text().splitlines() if line.strip()]
    timestamp_lines = [
        line for line in timestamp_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    counts = {len(cam0_files), len(cam1_files), len(pose_lines), len(timestamp_lines)}
    if len(counts) != 1 or not cam0_files:
        raise ValueError(
            "Unaligned sequence: "
            f"cam0={len(cam0_files)} cam1={len(cam1_files)} "
            f"poses={len(pose_lines)} timestamps={len(timestamp_lines)}"
        )
    if layout == "capture4daaam_like":
        quality = json.loads((src / "quality_report.json").read_text())
        if not quality.get("alignment", {}).get("ok"):
            raise ValueError("G1 quality report says the sequence is not aligned")

    timestamps = parse_timestamps(timestamp_lines)
    stereo_deltas_ms = read_stereo_sync(src, len(cam0_files))
    K, width, height, distortion, baseline = read_calibration(
        calibration_dir, layout
    )

    rgb_dir = dst / "rgb"
    depth_dir = dst / "depth"
    pose_dir = dst / "pose"
    for directory in (rgb_dir, depth_dir, pose_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frames = []
    for idx, (left, right) in enumerate(zip(cam0_files, cam1_files)):
        if left.stem != right.stem:
            raise ValueError(f"Stereo filename mismatch: {left.name} != {right.name}")
        ensure_symlink(rgb_dir / f"{idx:08d}.png", left)
        frames.append(
            {
                "idx": idx,
                "frame_id": int(left.stem),
                "cam0": str(left.resolve()),
                "cam1": str(right.resolve()),
                "timestamp": timestamps[idx],
                "stereo_delta_ms": stereo_deltas_ms[idx],
            }
        )

    ensure_symlink(pose_dir / "poses.txt", pose_file)
    camera_info = {
        "width": width,
        "height": height,
        "intrinsics": [K[0:3], K[3:6], K[6:9]],
        "distortion": distortion,
        "fx": float(K[0]),
        "fy": float(K[4]),
        "cx": float(K[2]),
        "cy": float(K[5]),
        "baseline": baseline,
    }
    (dst / "camera_info.json").write_text(json.dumps(camera_info, indent=2) + "\n")
    (dst / "tick_index.json").write_text(
        json.dumps(
            {
                "source": str(src),
                "source_layout": layout,
                "sequence": seq,
                "fx": camera_info["fx"],
                "baseline": baseline,
                "width": width,
                "height": height,
                "frames": frames,
            },
            indent=2,
        )
        + "\n"
    )
    (dst / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Prepared {len(frames)} frames at {dst}\n"
        f"  size={width}x{height} fx={camera_info['fx']:.6f} "
        f"baseline={baseline:.9f}m"
    )
    measured_deltas = [value for value in stereo_deltas_ms if value is not None]
    if measured_deltas:
        risky = sum(value > 10.0 for value in measured_deltas)
        print(
            f"  stereo sync: <=10ms={len(measured_deltas) - risky}/"
            f"{len(measured_deltas)}, >10ms={risky}, "
            f"max={max(measured_deltas):.3f}ms"
        )


if __name__ == "__main__":
    main()
