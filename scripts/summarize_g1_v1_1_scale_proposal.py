#!/usr/bin/env python3
"""Add V1.1 provenance, temporal bootstrap, and range coverage to a scale proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--train-report", required=True, type=Path)
    parser.add_argument("--holdout-report", required=True, type=Path)
    parser.add_argument("--prepared-index", required=True, type=Path)
    parser.add_argument("--depth-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def block_bootstrap_medians(
    values: np.ndarray,
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    if samples < 5000:
        raise ValueError("V1.1 requires at least 5000 bootstrap samples")
    if not 1 <= block_length <= len(values):
        raise ValueError("invalid temporal block length")
    rng = np.random.default_rng(seed)
    output = np.empty(samples, dtype=np.float64)
    block_count = int(np.ceil(len(values) / block_length))
    offsets = np.arange(block_length)
    for index in range(samples):
        starts = rng.integers(0, len(values), size=block_count)
        sampled_indices = (
            starts[:, None] + offsets[None, :]
        ).reshape(-1)[: len(values)] % len(values)
        output[index] = np.median(values[sampled_indices])
    return output


def aggregate_range_bins(
    frames: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    totals: dict[tuple[float, float], dict[str, int]] = {}
    for frame in frames:
        for item in frame.get("lidar_range_bins", []):
            bounds = tuple(float(value) for value in item["lidar_range_m"])
            record = totals.setdefault(
                bounds, {"lidar_pixel_count": 0, "raw_depth_count": 0}
            )
            record["lidar_pixel_count"] += int(item["lidar_pixel_count"])
            record["raw_depth_count"] += int(item["raw_depth_count"])
    return [
        {
            "lidar_range_m": list(bounds),
            **counts,
            "raw_depth_coverage_ratio": (
                counts["raw_depth_count"] / counts["lidar_pixel_count"]
                if counts["lidar_pixel_count"]
                else 0.0
            ),
            "scaled_error_status": (
                "pending_collector_support_and_manual_occlusion_exclusion"
            ),
        }
        for bounds, counts in sorted(totals.items())
    ]


def main() -> None:
    args = parse_args()
    paths = {
        "proposal": args.proposal.resolve(),
        "train_report": args.train_report.resolve(),
        "holdout_report": args.holdout_report.resolve(),
        "prepared_index": args.prepared_index.resolve(),
        "depth_run": args.depth_run.resolve(),
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite scale summary: {output}")
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    proposal = load(paths["proposal"])
    train = load(paths["train_report"])
    holdout = load(paths["holdout_report"])
    all_frames = sorted(
        train["frames"] + holdout["frames"],
        key=lambda frame: int(frame["source_index"]),
    )
    source_indices = [int(frame["source_index"]) for frame in all_frames]
    if source_indices != list(range(473, 488)):
        raise ValueError(
            f"scale evidence is not exactly calibration 473-487: {source_indices}"
        )
    per_frame = []
    for frame in all_frames:
        scale = frame.get("candidate_scale_reference_over_stereo_0_5_to_5m")
        if not scale or int(scale.get("sample_count", 0)) <= 0:
            raise ValueError(
                f"missing per-frame scale at {frame['source_index']}"
            )
        per_frame.append(float(scale["median"]))
    values = np.asarray(per_frame, dtype=np.float64)
    bootstrap = block_bootstrap_medians(
        values,
        samples=args.bootstrap_samples,
        block_length=args.block_length,
        seed=args.seed,
    )
    scale = float(proposal["depth_scale"])
    report = {
        "schema": "daaam.g1_v1_1_scale_proposal.v1",
        "status": "proposal_not_frozen",
        "formal_scale_path": None,
        "source_frames": [473, 487],
        "source_indices": source_indices,
        "depth_scale": scale,
        "source_baseline_m": float(proposal["source_baseline_m"]),
        "effective_baseline_m": float(proposal["effective_baseline_m"]),
        "method": proposal["method"],
        "interleaved_acceptance": proposal["acceptance"],
        "holdout_scaled_metrics_0_25_to_5m": proposal["holdout"][
            "scaled_lidar_reference_0_25_to_5m"
        ],
        "per_frame_scale": [
            {"source_index": source_index, "scale": value}
            for source_index, value in zip(source_indices, per_frame)
        ],
        "temporal_block_bootstrap": {
            "statistic": "median per-frame LiDAR/stereo scale",
            "block_length_frames": args.block_length,
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "p02_5": float(np.percentile(bootstrap, 2.5)),
            "p50": float(np.percentile(bootstrap, 50.0)),
            "p97_5": float(np.percentile(bootstrap, 97.5)),
        },
        "distance_bins": aggregate_range_bins(all_frames),
        "input_hashes": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "freeze_blockers": [
            "LiDAR occlusion boundaries are not yet human-marked not-judgeable.",
            "The collector does not yet emit scaled error metrics per distance bin.",
            "The P1 GT pilot and reviewer sign-off are incomplete.",
        ],
        "interpretation": (
            "Calibration-only evidence supports the numeric scale, but this "
            "file is a proposal and must not be substituted for the frozen "
            "scale_473_487.json required by formal V1.1 runs."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "depth_scale": scale,
                "bootstrap_95pct": [
                    report["temporal_block_bootstrap"]["p02_5"],
                    report["temporal_block_bootstrap"]["p97_5"],
                ],
                "holdout_median_absrel": report[
                    "holdout_scaled_metrics_0_25_to_5m"
                ]["median_absolute_relative_error"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
