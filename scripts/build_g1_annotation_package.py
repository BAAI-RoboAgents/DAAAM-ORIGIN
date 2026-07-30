#!/usr/bin/env python3
"""Build visual annotation packets for instances, tracks, and DSG bindings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SCHEMA = "daaam.g1_manual_semantic_ground_truth.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--lidar-ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    tasks = [
        json.loads(line)
        for line in args.tasks.read_text().splitlines()
        if line.strip()
    ]
    if not tasks:
        raise ValueError("annotation task list is empty")
    packets = output / "frame_packets"
    packets.mkdir(parents=True)
    annotation_frames = []
    for task in tasks:
        source_index = int(task["source_index"])
        left = cv2.imread(str(task["images"]["cam0"]), cv2.IMREAD_COLOR)
        right = cv2.imread(str(task["images"]["cam1"]), cv2.IMREAD_COLOR)
        overlay = cv2.imread(
            str(
                args.lidar_ground_truth
                / "overlays"
                / f"{source_index:06d}.png"
            ),
            cv2.IMREAD_COLOR,
        )
        if left is None or right is None or overlay is None:
            raise FileNotFoundError(f"annotation image missing at {source_index}")
        height = min(left.shape[0], right.shape[0], overlay.shape[0])
        panels = []
        for name, image in (("LEFT RGB", left), ("LIDAR OVERLAY", overlay), ("RIGHT RGB", right)):
            panel = image[:height].copy()
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 52), (0, 0, 0), -1)
            cv2.putText(
                panel,
                name,
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)
        packet = np.hstack(panels)
        cv2.putText(
            packet,
            f"source={source_index} split={task['split']}",
            (18, packet.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        packet_path = packets / f"{source_index:06d}.jpg"
        if not cv2.imwrite(
            str(packet_path),
            packet,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        ):
            raise RuntimeError(f"Could not write annotation packet: {packet_path}")
        annotation_frames.append(
            {
                "source_index": source_index,
                "split": task["split"],
                "sensor_time_ns": task["sensor_time_ns"],
                "left_image": task["images"]["cam0"],
                "right_image": task["images"]["cam1"],
                "lidar": task["lidar"],
                "visual_packet": str(packet_path),
                "instances": [],
                "review": {
                    "annotator": None,
                    "reviewer": None,
                    "status": "unlabeled",
                    "notes": "",
                },
            }
        )
    template = {
        "schema": SCHEMA,
        "annotation_version": 1,
        "coordinate_system": {
            "mask": "cam0 prepared/original-left pixel coordinates",
            "position": "map frame, meters",
            "time": "absolute sensor_time_ns",
        },
        "instance_contract": {
            "required": [
                "instance_id",
                "track_id",
                "mask_rle",
                "canonical_name",
                "acceptable_synonyms",
                "attributes",
                "dynamic_state",
                "visibility",
                "occlusion",
                "should_have_mesh",
                "expected_dsg_object_id",
            ],
            "dynamic_state_values": ["static", "dynamic", "uncertain"],
            "visibility_range": [0.0, 1.0],
            "occlusion_range": [0.0, 1.0],
        },
        "tracks": [],
        "dsg_bindings": [],
        "frames": annotation_frames,
    }
    (output / "annotations.json").write_text(
        json.dumps(template, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (output / "README.md").write_text(
        "# G1 人工语义真值标注\n\n"
        "逐帧 packet 依次显示左目、运动补偿 LiDAR 投影和右目。mask 必须画在"
        "左目坐标；跨帧同一实体使用同一 track_id。每个实例同时填写标准名、"
        "可接受同义词、属性、动静状态、可见率、遮挡率、是否应有 mesh，以及"
        "预期 DSG object 绑定。标注完成后由第二人审核，并将状态改为 reviewed。\n"
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "frames": len(annotation_frames),
                "packets": str(packets),
                "annotations": str(output / "annotations.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
