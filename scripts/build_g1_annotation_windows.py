#!/usr/bin/env python3
"""Create continuous annotation windows and dual-review subsets for G1 GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--annotation-tasks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--dual-review-fraction", type=float, default=0.10)
    return parser.parse_args()


def contiguous_windows(frames: list[int], window_size: int) -> list[list[int]]:
    if not frames:
        return []
    windows = []
    start = 0
    while start < len(frames):
        chunk = frames[start : start + window_size]
        if len(chunk) < max(8, window_size // 2) and windows:
            windows[-1].extend(chunk)
            break
        windows.append(chunk)
        start += window_size
    return windows


def main() -> None:
    args = parse_args()
    split_manifest = json.loads(args.split_manifest.read_text())
    tasks = [
        json.loads(line)
        for line in args.annotation_tasks.read_text().splitlines()
        if line.strip()
    ]
    by_split: dict[str, list[int]] = {}
    for task in tasks:
        by_split.setdefault(str(task["split"]), []).append(int(task["source_index"]))
    for frames in by_split.values():
        frames.sort()

    windows = []
    dual_review = []
    for split_name, frames in by_split.items():
        for ordinal, window in enumerate(contiguous_windows(frames, args.window_size), start=1):
            window_id = f"{split_name}_w{ordinal:02d}"
            windows.append(
                {
                    "window_id": window_id,
                    "split": split_name,
                    "source_indices": window,
                    "frame_count": len(window),
                    "purpose": "track_consistency_and_instance_identity",
                    "status": "ready_for_human_labeling",
                }
            )
            # First window of each split is dual-reviewed; plus fraction of frames.
            if ordinal == 1:
                dual_review.extend(window)

    unique_frames = sorted(set(int(task["source_index"]) for task in tasks))
    target = max(1, int(round(len(unique_frames) * args.dual_review_fraction)))
    if len(dual_review) < target:
        for source_index in unique_frames:
            if source_index not in dual_review:
                dual_review.append(source_index)
            if len(dual_review) >= target:
                break
    dual_review = sorted(set(dual_review))[: max(target, len(by_split) * 8)]

    content_labels = {
        name: {
            "frames": payload["frames"],
            "count": payload["count"],
            "content_hypothesis": {
                "calibration": "static planes / known scale / dense LiDAR coverage",
                "development": "ordinary indoor motion with texture and loops",
                "stress": "low texture / glare / occlusion / people / fast turn / far range",
                "held_out": "contiguous held-out block; no parameter selection",
            }.get(name, "review_required"),
            "content_review_status": "hypothesis_pending_human_lock",
        }
        for name, payload in split_manifest.get("splits", {}).items()
    }

    report = {
        "schema": "daaam.g1_annotation_windows.v1",
        "annotation_version": 1,
        "window_size": args.window_size,
        "dual_review_fraction": args.dual_review_fraction,
        "windows": windows,
        "dual_review_source_indices": dual_review,
        "dual_review_count": len(dual_review),
        "content_labels": content_labels,
        "required_fields": [
            "instances[].mask",
            "instances[].object_id",
            "instances[].canonical_name",
            "instances[].acceptable_synonyms",
            "instances[].attributes",
            "instances[].dynamic_state",
            "instances[].should_have_mesh",
            "instances[].dsg_object_id",
        ],
        "inter_annotator_protocol": {
            "dual_review_subset_fraction": args.dual_review_fraction,
            "metrics": ["mask_iou", "class_or_description_agreement", "object_id_agreement"],
            "arbitration": "third_pass_then_lock_annotation_version_hash",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "windows": len(windows),
                "dual_review_count": len(dual_review),
                "splits": sorted(by_split),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
