#!/usr/bin/env python3
"""Bootstrap non-authoritative instance proposals for annotation assist.

These proposals are never treated as human-reviewed ground truth. They only
seed the annotation package so labelers can accept/edit/reject masks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-package", required=True, type=Path)
    parser.add_argument("--annotation-windows", required=True, type=Path)
    parser.add_argument("--lidar-ground-truth", required=True, type=Path)
    parser.add_argument("--min-area", type=int, default=400)
    parser.add_argument("--max-instances-per-frame", type=int, default=12)
    return parser.parse_args()


def proposals_from_lidar(mask: np.ndarray, min_area: int, limit: int) -> list[dict]:
    binary = (mask > 0).astype(np.uint8) * 255
    # Split sparse lidar into connected components after light dilation.
    kernel = np.ones((5, 5), np.uint8)
    thick = cv2.dilate(binary, kernel, iterations=2)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(thick, connectivity=8)
    instances = []
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        polygon = contour.reshape(-1, 2).tolist()
        x, y, w, h = cv2.boundingRect(contour)
        instances.append(
            {
                "object_id": f"proposal_{label}",
                "canonical_name": "unknown_object",
                "acceptable_synonyms": [],
                "attributes": [],
                "dynamic_state": "unknown",
                "should_have_mesh": True,
                "dsg_object_id": None,
                "bbox_xywh": [int(x), int(y), int(w), int(h)],
                "polygon_xy": polygon,
                "area_px": int(cv2.contourArea(contour)),
                "label_source": "lidar_connected_component_proposal",
                "needs_human_review": True,
            }
        )
    instances.sort(key=lambda item: item["area_px"], reverse=True)
    return instances[:limit]


def main() -> None:
    args = parse_args()
    package_path = args.annotation_package.resolve()
    package = json.loads(package_path.read_text())
    windows = json.loads(args.annotation_windows.read_text())
    dual = set(int(index) for index in windows.get("dual_review_source_indices", []))
    window_by_frame = {}
    for window in windows.get("windows", []):
        for source_index in window.get("source_indices", []):
            window_by_frame[int(source_index)] = window["window_id"]

    mask_dir = args.lidar_ground_truth.resolve() / "valid_masks"
    updated = []
    proposal_frames = 0
    proposal_instances = 0
    for frame in package.get("frames", []):
        source_index = int(frame["source_index"])
        mask = cv2.imread(str(mask_dir / f"{source_index:06d}.png"), cv2.IMREAD_GRAYSCALE)
        instances = []
        if mask is not None:
            instances = proposals_from_lidar(
                mask, args.min_area, args.max_instances_per_frame
            )
        frame = dict(frame)
        frame["instances"] = instances
        frame["annotation_window_id"] = window_by_frame.get(source_index)
        frame["dual_review"] = source_index in dual
        review = dict(frame.get("review") or {})
        review["status"] = "proposal_pending_human_review"
        review["proposal_generator"] = "lidar_connected_component_v1"
        frame["review"] = review
        if instances:
            proposal_frames += 1
            proposal_instances += len(instances)
        updated.append(frame)

    package["frames"] = updated
    package["proposal_bootstrap"] = {
        "generator": "lidar_connected_component_v1",
        "authoritative": False,
        "proposal_frames": proposal_frames,
        "proposal_instances": proposal_instances,
        "note": "Human accept/edit/reject required before formal semantic metrics.",
    }
    package_path.write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n")

    # Mirror proposal status into annotation_tasks.jsonl beside the package.
    tasks_path = package_path.parent.parent / "annotation_tasks.jsonl"
    if tasks_path.is_file():
        tasks = []
        for line in tasks_path.read_text().splitlines():
            if not line.strip():
                continue
            task = json.loads(line)
            source_index = int(task["source_index"])
            task["status"] = "proposal_pending_human_review"
            task["annotation_window_id"] = window_by_frame.get(source_index)
            task["dual_review"] = source_index in dual
            tasks.append(task)
        tasks_path.write_text(
            "".join(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n" for task in tasks)
        )

    print(
        json.dumps(
            {
                "proposal_frames": proposal_frames,
                "proposal_instances": proposal_instances,
                "dual_review_frames": len(dual),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
