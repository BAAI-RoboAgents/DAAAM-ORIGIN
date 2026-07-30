#!/usr/bin/env python3
"""Render the strongest frozen RGB/semantic-mask observation per E16 label."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-input", type=Path, required=True)
    parser.add_argument("--labels", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def main() -> int:
    args = parse_args()
    shared_input = args.shared_input.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = read_jsonl(shared_input / "frames.jsonl")[:101]
    requested = set(args.labels)
    best: dict[int, tuple[int, int, np.ndarray]] = {}
    for index in range(len(frames)):
        labels = cv2.imread(
            str(shared_input / "label_frames" / f"{index:08d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        if labels is None:
            raise FileNotFoundError(f"semantic labels for frame {index}")
        for label in requested:
            mask = labels == label
            area = int(np.count_nonzero(mask))
            if area and (label not in best or area > best[label][0]):
                best[label] = (area, index, mask)

    index_rows = []
    for label in sorted(requested):
        if label not in best:
            index_rows.append(
                {"semantic_label": label, "status": "not_visible"}
            )
            continue
        area, index, mask = best[label]
        frame = frames[index]
        rgb = cv2.imread(str(frame["rgb_path"]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(frame["rgb_path"])
        overlay = rgb.copy()
        color = np.array([0, 215, 255], dtype=np.float32)
        overlay[mask] = np.clip(
            0.35 * overlay[mask].astype(np.float32) + 0.65 * color,
            0,
            255,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 3)
        rows, columns = np.nonzero(mask)
        margin = 40
        x0 = max(0, int(columns.min()) - margin)
        x1 = min(rgb.shape[1], int(columns.max()) + margin + 1)
        y0 = max(0, int(rows.min()) - margin)
        y1 = min(rgb.shape[0], int(rows.max()) + margin + 1)
        crop = overlay[y0:y1, x0:x1]
        panel = np.hstack(
            (letterbox(overlay, 640, 480), letterbox(crop, 640, 480))
        )
        title = (
            f"E{label:03d} | frame={index} | "
            f"source={frame.get('source_frame_index')} | area={area}px"
        )
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (20, 20, 20), -1)
        cv2.putText(
            panel,
            title,
            (14, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        output = args.output / f"label_{label:03d}.jpg"
        if not cv2.imwrite(str(output), panel):
            raise OSError(output)
        index_rows.append(
            {
                "semantic_label": label,
                "status": "rendered",
                "frame_index": index,
                "source_frame_index": frame.get("source_frame_index"),
                "mask_area_px": area,
                "output": str(output.resolve()),
            }
        )
    (args.output / "index.json").write_text(
        json.dumps(index_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
