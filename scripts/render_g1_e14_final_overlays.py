#!/usr/bin/env python3
"""Render final E14 MapMemory labels on the triggering left RGB frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPOSITORY_ROOT
    / "experiments/g1_20260724_473_573_v1_1"
    / "runs/diagnostic_gt_free_e14_e13fed_dam_20260729"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def color_for(entity_ordinal: int) -> tuple[int, int, int]:
    return (
        int((entity_ordinal * 67) % 205 + 50),
        int((entity_ordinal * 113) % 205 + 50),
        int((entity_ordinal * 173) % 205 + 50),
    )


def wrap_pixels(
    text: str,
    maximum_width: int,
    font_scale: float,
    thickness: int,
) -> list[str]:
    words = text.split()
    if not words:
        return ["<empty>"]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0][0]
        if width <= maximum_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_record(
    record: dict[str, Any],
    final_by_id: Mapping[str, dict[str, Any]],
    output: Path,
    threshold: int,
    seed: int,
) -> dict[str, Any]:
    rgb = cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(record["rgb_path"])
    overlay = rgb.copy()
    entity_requests: dict[str, list[dict[str, Any]]] = {}
    for request in record["requests"]:
        entity_requests.setdefault(request["entity_id"], []).append(request)
        mask = cv2.imread(request["materialized_mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != rgb.shape[:2]:
            raise ValueError(request["materialized_mask_path"])
        binary = mask > 0
        color = np.asarray(color_for(int(request["entity_ordinal"])), dtype=np.float32)
        overlay[binary] = (
            0.58 * overlay[binary].astype(np.float32) + 0.42 * color
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            color_for(int(request["entity_ordinal"])),
            2,
            cv2.LINE_AA,
        )
        x1, y1, _, _ = request["bbox_xyxy"]
        entity_text = f"E{request['entity_ordinal']}"
        text_size = cv2.getTextSize(
            entity_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2
        )[0]
        tx, ty = int(x1), max(20, int(y1) - 4)
        cv2.rectangle(
            overlay,
            (tx - 2, ty - text_size[1] - 4),
            (tx + text_size[0] + 3, ty + 3),
            (15, 15, 15),
            -1,
        )
        cv2.putText(
            overlay,
            entity_text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color_for(int(request["entity_ordinal"])),
            2,
            cv2.LINE_AA,
        )
    title = (
        f"E14 final overlay | obs={threshold} seed={seed} | "
        f"source frame={record['source_frame_index']}"
    )
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 36), (15, 15, 15), -1)
    cv2.putText(
        overlay,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    panel_width = 980
    panel = np.full((rgb.shape[0], panel_width, 3), 248, dtype=np.uint8)
    cv2.putText(
        panel,
        "Final MapMemory descriptions",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        (
            f"obs={threshold}, seed={seed}; E# colors match the RGB masks. "
            "Not human GT."
        ),
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    y = 88
    entities = []
    for entity_id, requests in sorted(
        entity_requests.items(),
        key=lambda item: int(item[1][0]["entity_ordinal"]),
    ):
        request = requests[0]
        entity = final_by_id[entity_id]
        label = " ".join(str(entity["canonical_name"]).split())
        color = color_for(int(request["entity_ordinal"]))
        heading = (
            f"E{request['entity_ordinal']}  masks={len(requests)}  "
            f"count_at_trigger={request['observation_count_at_prompt']}"
        )
        cv2.rectangle(panel, (20, y - 14), (35, y + 1), color, -1)
        cv2.putText(
            panel,
            heading,
            (44, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 23
        lines = wrap_pixels(label, panel_width - 50, 0.44, 1)
        maximum_lines = 5
        rendered_lines = lines[:maximum_lines]
        if len(lines) > maximum_lines:
            rendered_lines[-1] = textwrap.shorten(
                rendered_lines[-1], width=110, placeholder=" ..."
            )
        for line in rendered_lines:
            cv2.putText(
                panel,
                line,
                (28, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (35, 35, 35),
                1,
                cv2.LINE_AA,
            )
            y += 20
        y += 15
        entities.append(
            {
                "entity_id": entity_id,
                "entity_ordinal": int(request["entity_ordinal"]),
                "track_ids": sorted({int(row["track_id"]) for row in requests}),
                "mask_count_in_trigger_frame": len(requests),
                "observation_count_at_prompt": int(
                    request["observation_count_at_prompt"]
                ),
                "final_canonical_name": label,
                "request_indices": [int(row["request_index"]) for row in requests],
                "materialized_mask_paths": [
                    row["materialized_mask_path"] for row in requests
                ],
            }
        )
    pure_path = output / "rgb_overlays" / (
        f"source_{int(record['source_frame_index']):06d}_"
        f"record_{int(record['prompt_record_index']):04d}.jpg"
    )
    annotated_path = output / "annotated_rgb" / pure_path.name
    save_image(pure_path, overlay)
    save_image(annotated_path, np.hstack((overlay, panel)))
    sidecar_path = output / "sidecars" / pure_path.with_suffix(".json").name
    sidecar = {
        "schema": "daaam.g1_no_gt_e14_final_rgb_overlay.v1",
        "threshold_observations": threshold,
        "seed": seed,
        "prompt_record_index": int(record["prompt_record_index"]),
        "request_id": record["request_id"],
        "frame_index": int(record["frame_index"]),
        "source_frame_index": int(record["source_frame_index"]),
        "rgb_path": record["rgb_path"],
        "rgb_sha256": record["rgb_sha256"],
        "rgb_overlay_path": str(pure_path.resolve()),
        "rgb_overlay_sha256": sha256_file(pure_path),
        "annotated_rgb_path": str(annotated_path.resolve()),
        "annotated_rgb_sha256": sha256_file(annotated_path),
        "entities": entities,
        "semantic_correctness": None,
        "interpretation": (
            "Final canonical_name from the selected E14 cell, rendered on the "
            "triggering left/cam0 RGB; this is not a human-GT label."
        ),
    }
    write_json(sidecar_path, sidecar)
    return {**sidecar, "sidecar_path": str(sidecar_path.resolve())}


def create_overview(output: Path, rows: list[dict[str, Any]]) -> Path:
    preferred_sources = [476, 480, 536, 545]
    selected = []
    for source in preferred_sources:
        selected.extend(row for row in rows if row["source_frame_index"] == source)
    selected = selected[:4] or rows[:4]
    tiles = []
    for row in selected:
        image = cv2.imread(row["rgb_overlay_path"], cv2.IMREAD_COLOR)
        resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        cv2.putText(
            resized,
            f"source {row['source_frame_index']} | "
            f"{len(row['entities'])} final entities",
            (10, 465),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(resized)
    while len(tiles) < 4:
        tiles.append(np.full((480, 640, 3), 245, dtype=np.uint8))
    overview = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:4])))
    path = output / "OVERVIEW.jpg"
    save_image(path, overview)
    return path


def main() -> int:
    args = parse_args()
    run = args.run.resolve()
    threshold = int(args.threshold)
    seed = int(args.seed)
    output = (
        run
        / "visualizations"
        / f"final_overlays_obs_{threshold:02d}_seed_{seed}"
    )
    if output.exists():
        raise FileExistsError(output)
    records = read_jsonl(
        run
        / "prompt_inputs"
        / f"obs_{threshold:02d}"
        / "prompt_records.jsonl"
    )
    final_entities = json.loads(
        (
            run
            / "cells"
            / f"obs_{threshold:02d}"
            / f"seed_{seed}"
            / "final_entities.json"
        ).read_text(encoding="utf-8")
    )
    final_by_id = {row["entity_id"]: row for row in final_entities}
    rows = [
        render_record(record, final_by_id, output, threshold, seed)
        for record in records
    ]
    write_jsonl(output / "manifest.jsonl", rows)
    overview = create_overview(output, rows)
    validation = {
        "schema": "daaam.g1_no_gt_e14_final_overlay_validation.v1",
        "passed": True,
        "record_count": len(rows),
        "entity_appearances": sum(len(row["entities"]) for row in rows),
        "mask_appearances": sum(
            sum(entity["mask_count_in_trigger_frame"] for entity in row["entities"])
            for row in rows
        ),
        "all_rgb_hashes_match": all(
            sha256_file(Path(row["rgb_path"])) == row["rgb_sha256"] for row in rows
        ),
        "all_final_labels_present": all(
            entity["final_canonical_name"]
            for row in rows
            for entity in row["entities"]
        ),
        "overview_path": str(overview.resolve()),
        "overview_sha256": sha256_file(overview),
        "scope": (
            "presentation-layer deterministic rendering of audited prompt inputs "
            "and final MapMemory labels; no semantic correctness claim"
        ),
    }
    write_json(output / "VALIDATION.json", validation)
    write_json(
        output / "SUMMARY.json",
        {
            "schema": "daaam.g1_no_gt_e14_final_overlay_summary.v1",
            "threshold_observations": threshold,
            "seed": seed,
            "trigger_frame_count": len(rows),
            "eligible_entity_count": len(
                {
                    entity["entity_id"]
                    for row in rows
                    for entity in row["entities"]
                }
            ),
            "mask_request_count": validation["mask_appearances"],
            "output": str(output.resolve()),
            "note": (
                "The right panel shows the selected cell's final canonical_name. "
                "It is a DAM description, not a human-GT class."
            ),
        },
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
