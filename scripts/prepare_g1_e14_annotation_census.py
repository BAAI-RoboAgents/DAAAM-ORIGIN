#!/usr/bin/env python3
"""Prepare a finite-population visual census of E14 final annotations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
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


def mask_bbox(binary: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(binary)
    if not len(xs):
        raise ValueError("empty mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def wrap_pixels(
    text: str,
    maximum_width: int,
    font_scale: float = 0.46,
    thickness: int = 1,
) -> list[str]:
    words = text.split()
    if not words:
        return ["<empty>"]
    lines = []
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


def fit_tile(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 242, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_CUBIC,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def render_entity(
    census_index: int,
    population_count: int,
    entity: dict[str, Any],
    entity_responses: list[dict[str, Any]],
    rgb_path: str,
    rgb_sha256: str,
    output: Path,
    threshold: int,
    seed: int,
) -> dict[str, Any]:
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    overlay = rgb.copy()
    mask_records = []
    crops = []
    colors = [(40, 220, 255), (255, 170, 40), (120, 255, 80), (220, 80, 255)]
    for index, response in enumerate(entity_responses):
        mask = cv2.imread(
            response["materialized_mask_path"], cv2.IMREAD_GRAYSCALE
        )
        if mask is None or mask.shape != rgb.shape[:2]:
            raise ValueError(response["materialized_mask_path"])
        binary = mask > 0
        bbox = mask_bbox(binary)
        color = np.asarray(colors[index % len(colors)], dtype=np.float32)
        overlay[binary] = (
            0.52 * overlay[binary].astype(np.float32) + 0.48 * color
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
            tuple(int(value) for value in color),
            3,
            cv2.LINE_AA,
        )
        x1, y1, x2, y2 = bbox
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            tuple(int(value) for value in color),
            2,
        )
        cv2.putText(
            overlay,
            f"M{index + 1}",
            (x1, max(22, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            tuple(int(value) for value in color),
            2,
            cv2.LINE_AA,
        )
        margin_x = max(30, int(round((x2 - x1) * 0.55)))
        margin_y = max(30, int(round((y2 - y1) * 0.55)))
        sx1, sy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
        sx2, sy2 = min(rgb.shape[1], x2 + margin_x), min(
            rgb.shape[0], y2 + margin_y
        )
        crops.append(overlay[sy1:sy2, sx1:sx2].copy())
        mask_records.append(
            {
                "mask_index": index + 1,
                "response_index": int(response["response_index"]),
                "track_id": int(response["track_id"]),
                "e11_instance_id": int(response["e11_instance_id"]),
                "mask_area_px": int(response["mask_area_px"]),
                "bbox_xyxy": bbox,
                "materialized_mask_path": response["materialized_mask_path"],
                "materialized_mask_sha256": response[
                    "materialized_mask_sha256"
                ],
                "response_label": response["description"],
                "is_final_label": (
                    " ".join(response["description"].split())
                    == " ".join(entity["final_label"].split())
                ),
            }
        )
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 42), (15, 15, 15), -1)
    cv2.putText(
        overlay,
        (
            f"E14 annotation census {census_index + 1}/{population_count} | "
            f"E{entity['entity_ordinal']} | obs={threshold} seed={seed} | "
            f"source={entity_responses[0]['source_frame_index']}"
        ),
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    panel_width = 1000
    panel = np.full((rgb.shape[0], panel_width, 3), 248, dtype=np.uint8)
    cv2.putText(
        panel,
        (
            f"E{entity['entity_ordinal']} evidence | "
            f"{len(entity_responses)} mask response(s)"
        ),
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )
    if len(crops) == 1:
        tile = fit_tile(crops[0], 900, 430)
        panel[50:480, 50:950] = tile
    else:
        slots = []
        for index in range(4):
            if index < len(crops):
                tile = fit_tile(crops[index], 445, 205)
                cv2.putText(
                    tile,
                    f"M{index + 1}",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    colors[index % len(colors)],
                    2,
                    cv2.LINE_AA,
                )
            else:
                tile = np.full((205, 445, 3), 242, dtype=np.uint8)
            slots.append(tile)
        panel[50:255, 50:495] = slots[0]
        panel[50:255, 505:950] = slots[1]
        panel[265:470, 50:495] = slots[2]
        panel[265:470, 505:950] = slots[3]
    y = 510
    cv2.putText(
        panel,
        "Final MapMemory description:",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )
    y += 29
    for line in wrap_pixels(entity["final_label"], panel_width - 50):
        cv2.putText(
            panel,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
        y += 22
        if y > 770:
            break
    cv2.putText(
        panel,
        (
            "Review: artifact completeness | union mask quality | "
            "main identity | visible attributes"
        ),
        (20, panel.shape[0] - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    panel_path = output / "review_panels" / (
        f"census_{census_index + 1:03d}_E{int(entity['entity_ordinal']):03d}_"
        f"source_{int(entity_responses[0]['source_frame_index']):06d}.jpg"
    )
    save_image(panel_path, np.hstack((overlay, panel)))
    return {
        "schema": "daaam.g1_no_gt_e14_annotation_census_item.v1",
        "census_index": census_index,
        "entity_id": entity["entity_id"],
        "entity_ordinal": int(entity["entity_ordinal"]),
        "final_label": entity["final_label"],
        "response_count": len(entity_responses),
        "source_frame_index": int(entity_responses[0]["source_frame_index"]),
        "frame_index": int(entity_responses[0]["frame_index"]),
        "rgb_path": rgb_path,
        "rgb_sha256": rgb_sha256,
        "mask_records": mask_records,
        "review_panel_path": str(panel_path.resolve()),
        "review_panel_sha256": sha256_file(panel_path),
    }


def main() -> int:
    args = parse_args()
    run = args.run.resolve()
    threshold = int(args.threshold)
    seed = int(args.seed)
    output = run / f"annotation_census_obs_{threshold:02d}_seed_{seed}"
    if output.exists():
        raise FileExistsError(output)
    final_labels_path = run / "tables/final_labels.jsonl"
    responses_path = (
        run
        / "cells"
        / f"obs_{threshold:02d}"
        / f"seed_{seed}"
        / "responses.jsonl"
    )
    records_path = (
        run / "prompt_inputs" / f"obs_{threshold:02d}" / "prompt_records.jsonl"
    )
    final_labels = sorted(
        (
            row
            for row in read_jsonl(final_labels_path)
            if int(row["threshold_observations"]) == threshold
            and int(row["seed"]) == seed
        ),
        key=lambda row: int(row["entity_ordinal"]),
    )
    responses = read_jsonl(responses_path)
    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in responses:
        by_entity[response["entity_id"]].append(response)
    records = read_jsonl(records_path)
    rgb_by_request = {
        int(request["request_index"]): (
            record["rgb_path"],
            record["rgb_sha256"],
        )
        for record in records
        for request in record["requests"]
    }
    write_json(
        output / "PRE_REGISTRATION.json",
        {
            "schema": "daaam.g1_no_gt_e14_annotation_census_preregistration.v1",
            "scope": (
                f"all E14 obs={threshold}, seed={seed} eligible final entities"
            ),
            "finite_population_count": len(final_labels),
            "coverage_target": (
                f"{len(final_labels)}/{len(final_labels)} unique eligible entities"
            ),
            "stopping_rule": (
                "stop only after every population entity has an artifact, union-mask, "
                "main-identity, and visible-attribute verdict"
            ),
            "review_unit": "unique MapMemory entity",
            "evidence_view": (
                "final-label-producing trigger RGB plus every same-frame mask/response "
                "for that entity"
            ),
            "semantic_verdicts": [
                "correct",
                "partially_correct",
                "incorrect",
                "unjudgeable",
            ],
            "mask_verdicts": [
                "acceptable",
                "acceptable_boundary_truncated",
                "partial",
                "oversegmented",
                "wrong_region",
                "unjudgeable",
            ],
            "strict_correctness": (
                "correct / judgeable; correct requires a correct main identity and "
                "no material visible attribute contradiction"
            ),
            "lenient_correctness": (
                "(correct + partially_correct) / judgeable"
            ),
            "formal_accuracy_claim_permitted": False,
            "formal_limitation": (
                "single Codex visual reviewer, no independent human adjudication"
            ),
            "sources": {
                "final_labels": {
                    "path": str(final_labels_path.resolve()),
                    "sha256": sha256_file(final_labels_path),
                },
                "responses": {
                    "path": str(responses_path.resolve()),
                    "sha256": sha256_file(responses_path),
                },
                "prompt_records": {
                    "path": str(records_path.resolve()),
                    "sha256": sha256_file(records_path),
                },
                "preparer": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__)),
                },
            },
        },
    )
    manifest = []
    for index, entity in enumerate(final_labels):
        entity_responses = sorted(
            by_entity[entity["entity_id"]],
            key=lambda row: int(row["response_index"]),
        )
        rgb_path, rgb_sha256 = rgb_by_request[
            int(entity_responses[0]["request_index"])
        ]
        manifest.append(
            render_entity(
                index,
                len(final_labels),
                entity,
                entity_responses,
                rgb_path,
                rgb_sha256,
                output,
                threshold,
                seed,
            )
        )
    write_jsonl(output / "CENSUS_MANIFEST.jsonl", manifest)
    write_jsonl(
        output / "REVIEW_RESULTS_TEMPLATE.jsonl",
        [
            {
                "census_index": row["census_index"],
                "entity_id": row["entity_id"],
                "entity_ordinal": row["entity_ordinal"],
                "artifact_complete": None,
                "mask_acceptable": None,
                "mask_verdict": None,
                "semantic_complete": None,
                "semantic_verdict": None,
                "main_identity_correct": None,
                "review_notes": None,
            }
            for row in manifest
        ],
    )
    write_json(
        output / "PREPARATION_SUMMARY.json",
        {
            "population_count": len(final_labels),
            "panel_count": len(manifest),
            "response_mask_count": sum(row["response_count"] for row in manifest),
            "entity_ordinals": [row["entity_ordinal"] for row in manifest],
            "all_rgb_hashes_match": all(
                sha256_file(Path(row["rgb_path"])) == row["rgb_sha256"]
                for row in manifest
            ),
            "all_panel_hashes_match": all(
                sha256_file(Path(row["review_panel_path"]))
                == row["review_panel_sha256"]
                for row in manifest
            ),
        },
    )
    print((output / "PREPARATION_SUMMARY.json").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
