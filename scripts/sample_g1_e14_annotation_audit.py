#!/usr/bin/env python3
"""Freeze and render a 10% simple-random visual audit of E14 annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPOSITORY_ROOT
    / "experiments/g1_20260724_473_573_v1_1"
    / "runs/diagnostic_gt_free_e14_e13fed_dam_20260729"
)
RANDOM_SEED = 20_260_729
SAMPLE_FRACTION = 0.10


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


def sha256_jsonl_material(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode()
        )
    return digest.hexdigest()


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def wrap_pixels(
    text: str,
    maximum_width: int,
    font_scale: float = 0.48,
    thickness: int = 1,
) -> list[str]:
    words = text.split()
    if not words:
        return ["<empty>"]
    result = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0][0]
        if width <= maximum_width:
            current = candidate
        else:
            result.append(current)
            current = word
    result.append(current)
    return result


def mask_bbox(binary: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(binary)
    if not len(xs):
        raise ValueError("empty mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def render_review_panel(row: dict[str, Any], output: Path) -> dict[str, Any]:
    rgb = cv2.imread(row["rgb_path"], cv2.IMREAD_COLOR)
    mask = cv2.imread(row["materialized_mask_path"], cv2.IMREAD_GRAYSCALE)
    if rgb is None or mask is None or mask.shape != rgb.shape[:2]:
        raise ValueError(row["entity_id"])
    binary = mask > 0
    bbox = mask_bbox(binary)
    x1, y1, x2, y2 = bbox
    overlay = rgb.copy()
    color = np.asarray((40, 220, 255), dtype=np.float32)
    overlay[binary] = (
        0.50 * overlay[binary].astype(np.float32) + 0.50 * color
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (40, 220, 255), 3, cv2.LINE_AA)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 220, 255), 2)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 40), (15, 15, 15), -1)
    cv2.putText(
        overlay,
        (
            f"Random audit sample {row['sample_index'] + 1} | "
            f"E{row['entity_ordinal']} | source={row['source_frame_index']}"
        ),
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    margin_x = max(30, int(round((x2 - x1) * 0.75)))
    margin_y = max(30, int(round((y2 - y1) * 0.75)))
    sx1, sy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    sx2, sy2 = min(rgb.shape[1], x2 + margin_x), min(
        rgb.shape[0], y2 + margin_y
    )
    focus = overlay[sy1:sy2, sx1:sx2]
    focus_scale = min(860 / focus.shape[1], 500 / focus.shape[0])
    focus_resized = cv2.resize(
        focus,
        (
            max(1, int(round(focus.shape[1] * focus_scale))),
            max(1, int(round(focus.shape[0] * focus_scale))),
        ),
        interpolation=cv2.INTER_CUBIC,
    )
    panel_width = 980
    panel = np.full((rgb.shape[0], panel_width, 3), 248, dtype=np.uint8)
    fx = (panel_width - focus_resized.shape[1]) // 2
    panel[20 : 20 + focus_resized.shape[0], fx : fx + focus_resized.shape[1]] = (
        focus_resized
    )
    y = min(560, 40 + focus_resized.shape[0])
    cv2.putText(
        panel,
        (
            f"E{row['entity_ordinal']} | trigger count="
            f"{row['observation_count_at_prompt']} | mask area={row['mask_area_px']} px"
        ),
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    y += 32
    cv2.putText(
        panel,
        "Final MapMemory description:",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    y += 27
    for line in wrap_pixels(row["final_label"], panel_width - 45):
        cv2.putText(
            panel,
            line,
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
        y += 22
        if y > panel.shape[0] - 65:
            break
    cv2.putText(
        panel,
        "Review target: mask coverage + main identity + visible attributes",
        (20, panel.shape[0] - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    path = output / "review_panels" / (
        f"sample_{row['sample_index'] + 1:02d}_"
        f"E{row['entity_ordinal']:03d}_source_{row['source_frame_index']:06d}.jpg"
    )
    save_image(path, np.hstack((overlay, panel)))
    return {
        **row,
        "review_panel_path": str(path.resolve()),
        "review_panel_sha256": sha256_file(path),
        "focus_bbox_xyxy": [sx1, sy1, sx2, sy2],
    }


def main() -> int:
    args = parse_args()
    run = args.run.resolve()
    threshold = int(args.threshold)
    seed = int(args.seed)
    output = (
        run
        / f"annotation_audit_10pct_obs_{threshold:02d}_seed_{seed}"
    )
    if output.exists():
        raise FileExistsError(output)
    final_labels = [
        row
        for row in read_jsonl(run / "tables/final_labels.jsonl")
        if int(row["threshold_observations"]) == threshold
        and int(row["seed"]) == seed
    ]
    responses = read_jsonl(
        run
        / "cells"
        / f"obs_{threshold:02d}"
        / f"seed_{seed}"
        / "responses.jsonl"
    )
    responses_by_entity: dict[str, list[dict[str, Any]]] = {}
    for response in responses:
        responses_by_entity.setdefault(response["entity_id"], []).append(response)
    population = []
    for label_row in sorted(final_labels, key=lambda row: int(row["entity_ordinal"])):
        entity_responses = responses_by_entity[label_row["entity_id"]]
        selected = next(
            (
                row
                for row in entity_responses
                if " ".join(row["description"].split())
                == " ".join(label_row["final_label"].split())
            ),
            entity_responses[-1],
        )
        population.append(
            {
                "entity_id": label_row["entity_id"],
                "entity_ordinal": int(label_row["entity_ordinal"]),
                "final_label": label_row["final_label"],
                "response_count": int(label_row["response_count"]),
                "source_frame_index": int(selected["source_frame_index"]),
                "frame_index": int(selected["frame_index"]),
                "request_index": int(selected["request_index"]),
                "request_id": selected["request_id"],
                "observation_count_at_prompt": int(
                    selected["observation_count_at_prompt"]
                ),
                "mask_area_px": int(selected["mask_area_px"]),
                "rgb_path": str(
                    (
                        run
                        / "prompt_inputs"
                        / f"obs_{threshold:02d}"
                        / "prompt_records.jsonl"
                    ).resolve()
                ),
                "materialized_mask_path": selected["materialized_mask_path"],
                "materialized_mask_sha256": selected[
                    "materialized_mask_sha256"
                ],
                "crop_preview_path": selected["crop_preview_path"],
                "crop_preview_sha256": selected["crop_preview_sha256"],
                "all_response_labels": [
                    row["description"] for row in entity_responses
                ],
            }
        )
    prompt_records = read_jsonl(
        run
        / "prompt_inputs"
        / f"obs_{threshold:02d}"
        / "prompt_records.jsonl"
    )
    rgb_by_request = {
        request["request_index"]: {
            "rgb_path": record["rgb_path"],
            "rgb_sha256": record["rgb_sha256"],
        }
        for record in prompt_records
        for request in record["requests"]
    }
    for row in population:
        row.update(rgb_by_request[row["request_index"]])
    sample_size = math.ceil(len(population) * SAMPLE_FRACTION)
    population_by_id = {row["entity_id"]: row for row in population}
    sampled_ids = random.Random(RANDOM_SEED).sample(
        sorted(population_by_id), sample_size
    )
    population_hash = sha256_jsonl_material(population)
    write_json(
        output / "PRE_REGISTRATION.json",
        {
            "schema": "daaam.g1_no_gt_e14_annotation_audit_preregistration.v1",
            "population": (
                f"E14 obs={threshold}, seed={seed} eligible final MapMemory entities"
            ),
            "sampling_unit": "unique MapMemory entity",
            "population_count": len(population),
            "population_material_sha256": population_hash,
            "sample_fraction_requested": SAMPLE_FRACTION,
            "sample_size_rule": "ceil(population_count * sample_fraction)",
            "sample_size": sample_size,
            "realized_sample_fraction": sample_size / len(population),
            "sampling_method": "simple random sample without replacement",
            "random_seed": RANDOM_SEED,
            "sampled_entity_ids_in_draw_order": sampled_ids,
            "review_dimensions": {
                "artifact_completeness": (
                    "RGB, mask, final label and traceability artifacts all present"
                ),
                "mask_completeness": (
                    "mask covers the visually intended primary region/object"
                ),
                "semantic_completeness": (
                    "description names/describes the primary masked target"
                ),
                "semantic_correctness": (
                    "main identity and asserted visible attributes agree with RGB"
                ),
            },
            "semantic_verdicts": [
                "correct",
                "partially_correct",
                "incorrect",
                "unjudgeable",
            ],
            "strict_accuracy": "correct / judgeable",
            "lenient_accuracy": "(correct + partially_correct) / judgeable",
            "limitations": [
                "single Codex visual reviewer, not dual-human adjudicated GT",
                "one trigger view per sampled entity",
                "small sample; report exact numerator/denominator and uncertainty",
            ],
            "sampler_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
        },
    )
    write_jsonl(output / "POPULATION.jsonl", population)
    sampled_rows = []
    for index, entity_id in enumerate(sampled_ids):
        sampled_rows.append(
            render_review_panel(
                {
                    "schema": "daaam.g1_no_gt_e14_annotation_audit_sample.v1",
                    "sample_index": index,
                    **population_by_id[entity_id],
                },
                output,
            )
        )
    write_jsonl(output / "SAMPLE_MANIFEST.jsonl", sampled_rows)
    write_jsonl(
        output / "REVIEW_RESULTS_TEMPLATE.jsonl",
        [
            {
                "sample_index": row["sample_index"],
                "entity_id": row["entity_id"],
                "entity_ordinal": row["entity_ordinal"],
                "artifact_complete": None,
                "mask_verdict": None,
                "semantic_complete": None,
                "semantic_verdict": None,
                "review_notes": None,
            }
            for row in sampled_rows
        ],
    )
    write_json(
        output / "SAMPLING_SUMMARY.json",
        {
            "population_count": len(population),
            "sample_count": len(sampled_rows),
            "realized_fraction": len(sampled_rows) / len(population),
            "random_seed": RANDOM_SEED,
            "population_material_sha256": population_hash,
            "sample_manifest_sha256": sha256_file(
                output / "SAMPLE_MANIFEST.jsonl"
            ),
            "sampled_entity_ordinals": [
                row["entity_ordinal"] for row in sampled_rows
            ],
        },
    )
    print(
        json.dumps(
            read_jsonl(output / "SAMPLE_MANIFEST.jsonl"),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
