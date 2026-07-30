#!/usr/bin/env python3
"""Evaluate D0 depth-scale injections against frozen camera-LiDAR correspondences.

This is a sparse, proxy reference check.  It complements temporal E5, which is
not expected to determine absolute metric scale.  All per-frame samples,
summaries, figures, provenance, and hashes are retained.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--d0-root", required=True, type=Path)
    parser.add_argument("--correspondences", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-lidar-depth-m", type=float, default=0.25)
    parser.add_argument("--maximum-lidar-depth-m", type=float, default=5.0)
    parser.add_argument(
        "--minimum-detectable-signed-shift-m", type=float, default=0.02
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth PNG: {path}")
    return depth.astype(np.float32) / 1000.0


def summarize(prediction: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    signed = prediction.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(signed)
    return {
        "count": int(signed.size),
        "mean_signed_stereo_minus_lidar_m": float(np.mean(signed)),
        "median_signed_stereo_minus_lidar_m": float(np.median(signed)),
        "mean_absolute_error_m": float(np.mean(absolute)),
        "median_absolute_error_m": float(np.median(absolute)),
        "p90_absolute_error_m": float(np.percentile(absolute, 90)),
        "median_absolute_relative_error": float(
            np.median(absolute / np.maximum(reference, 1e-6))
        ),
    }


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    d0_root = args.d0_root.resolve()
    correspondences = args.correspondences.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    output.mkdir(parents=True)

    tick_path = dataset / "tick_index.json"
    tick = read_json(tick_path)
    frame_lookup = {
        int(record["idx"]): int(record["source_frame_idx"])
        for record in tick["frames"]
    }
    summary_rows = read_json(d0_root / "d0_variant_summary.json")
    scale_rows = [row for row in summary_rows if row["mode"] == "depth_scale"]
    if len(scale_rows) != 6:
        raise RuntimeError(f"Expected six depth-scale variants, got {len(scale_rows)}")

    variants: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    sample_files: list[Path] = []
    for row in scale_rows:
        variant_id = str(row["variant_id"])
        perturbation_report_path = (
            d0_root / "variants" / variant_id / "geometry_perturbation_report.json"
        )
        perturbation_report = read_json(perturbation_report_path)
        factor = float(perturbation_report["parameters"]["depth_scale"])
        frame_indices = [
            int(record["frame_index"]) for record in perturbation_report["frames"]
        ]
        control_samples: list[np.ndarray] = []
        injected_samples: list[np.ndarray] = []
        lidar_samples: list[np.ndarray] = []
        frame_records: list[dict[str, Any]] = []
        for frame_index in frame_indices:
            source_frame = frame_lookup[frame_index]
            correspondence_path = correspondences / f"{source_frame:06d}.npz"
            if not correspondence_path.exists():
                continue
            sample_files.append(correspondence_path)
            with np.load(correspondence_path) as values:
                u = values["u"].astype(np.int64)
                v = values["v"].astype(np.int64)
                lidar = values["depth_m"].astype(np.float32)
            control_depth_path = dataset / "depth" / f"{frame_index:08d}.png"
            injected_depth_path = (
                d0_root / "variants" / variant_id / "depth" / f"{frame_index:08d}.png"
            )
            control_depth = load_depth(control_depth_path)
            injected_depth = load_depth(injected_depth_path)
            inside = (
                (u >= 0)
                & (u < control_depth.shape[1])
                & (v >= 0)
                & (v < control_depth.shape[0])
                & (lidar >= args.minimum_lidar_depth_m)
                & (lidar <= args.maximum_lidar_depth_m)
            )
            u = u[inside]
            v = v[inside]
            lidar = lidar[inside]
            control = control_depth[v, u]
            injected = injected_depth[v, u]
            valid = (
                np.isfinite(control)
                & np.isfinite(injected)
                & (control > 0.0)
                & (injected > 0.0)
            )
            control = control[valid]
            injected = injected[valid]
            lidar = lidar[valid]
            if not len(lidar):
                continue
            control_samples.append(control)
            injected_samples.append(injected)
            lidar_samples.append(lidar)
            signed_shift = injected.astype(np.float64) - control.astype(np.float64)
            median_shift = float(np.median(signed_shift))
            direction_correct = bool(
                np.sign(median_shift) == np.sign(factor - 1.0)
            )
            detected = bool(
                direction_correct
                and abs(median_shift) >= args.minimum_detectable_signed_shift_m
            )
            frame_record = {
                "variant_id": variant_id,
                "depth_scale": factor,
                "absolute_scale_dose": abs(factor - 1.0),
                "geometry_frame_index": frame_index,
                "source_frame_index": source_frame,
                "sample_count": int(len(lidar)),
                "control_median_signed_error_m": float(
                    np.median(control.astype(np.float64) - lidar.astype(np.float64))
                ),
                "injected_median_signed_error_m": float(
                    np.median(injected.astype(np.float64) - lidar.astype(np.float64))
                ),
                "median_injected_minus_control_m": median_shift,
                "expected_shift_direction": "positive" if factor > 1.0 else "negative",
                "direction_correct": direction_correct,
                "detected": detected,
                "correspondence": str(correspondence_path),
                "control_depth": str(control_depth_path),
                "injected_depth": str(injected_depth_path),
            }
            frame_records.append(frame_record)
            all_frame_rows.append(frame_record)

        control_array = np.concatenate(control_samples)
        injected_array = np.concatenate(injected_samples)
        lidar_array = np.concatenate(lidar_samples)
        control_summary = summarize(control_array, lidar_array)
        injected_summary = summarize(injected_array, lidar_array)
        detected_count = sum(bool(record["detected"]) for record in frame_records)
        aggregate_shift = float(
            np.median(injected_array.astype(np.float64) - control_array.astype(np.float64))
        )
        variant = {
            "variant_id": variant_id,
            "depth_scale": factor,
            "absolute_scale_dose": abs(factor - 1.0),
            "frame_count": len(frame_records),
            "sample_count": int(len(lidar_array)),
            "control": control_summary,
            "injected": injected_summary,
            "aggregate_median_injected_minus_control_m": aggregate_shift,
            "expected_shift_direction": "positive" if factor > 1.0 else "negative",
            "aggregate_direction_correct": bool(
                np.sign(aggregate_shift) == np.sign(factor - 1.0)
            ),
            "detected_frame_count": detected_count,
            "frame_detection_rate": (
                detected_count / len(frame_records) if frame_records else 0.0
            ),
        }
        variants.append(variant)

    frame_csv = output / "per_frame_lidar_scale_response.csv"
    with frame_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_frame_rows[0]))
        writer.writeheader()
        writer.writerows(all_frame_rows)
    with (output / "per_frame_lidar_scale_response.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for record in all_frame_rows:
            stream.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
    write_json(output / "lidar_scale_variant_summary.json", variants)

    direction_groups: dict[str, Any] = {}
    for direction, predicate in (
        ("negative", lambda factor: factor < 1.0),
        ("positive", lambda factor: factor > 1.0),
    ):
        group = sorted(
            [variant for variant in variants if predicate(variant["depth_scale"])],
            key=lambda variant: variant["absolute_scale_dose"],
        )
        shifts = [
            abs(float(variant["aggregate_median_injected_minus_control_m"]))
            for variant in group
        ]
        direction_groups[direction] = {
            "variant_ids": [variant["variant_id"] for variant in group],
            "absolute_signed_shift_m": shifts,
            "monotone_non_decreasing": bool(np.all(np.diff(shifts) >= -1e-12)),
        }
    medium_heavy = [
        variant for variant in variants if variant["absolute_scale_dose"] >= 0.10
    ]
    qualification = {
        "schema": "daaam.no_gt_d0_lidar_scale_qualification.v1",
        "reference_level": "sparse_lidar_proxy_not_human_ground_truth",
        "minimum_detectable_signed_shift_m": args.minimum_detectable_signed_shift_m,
        "medium_heavy_target": 0.90,
        "medium_heavy": [
            {
                "variant_id": variant["variant_id"],
                "frame_detection_rate": variant["frame_detection_rate"],
                "passed": variant["frame_detection_rate"] >= 0.90,
            }
            for variant in medium_heavy
        ],
        "all_medium_heavy_passed": all(
            variant["frame_detection_rate"] >= 0.90 for variant in medium_heavy
        ),
        "direction_groups": direction_groups,
        "both_directions_monotone": all(
            group["monotone_non_decreasing"]
            for group in direction_groups.values()
        ),
        "e4_scale_observability_passed": (
            all(variant["frame_detection_rate"] >= 0.90 for variant in medium_heavy)
            and all(
                group["monotone_non_decreasing"]
                for group in direction_groups.values()
            )
        ),
        "interpretation": (
            "Temporal E5 may improve when all depths are scaled down and therefore "
            "cannot certify absolute scale. Sparse camera-LiDAR signed residual is "
            "the E4 observer used here."
        ),
        "caveats": [
            "LiDAR is sparse and is only a proxy at projected visible pixels.",
            "Stored point clouds have no per-point timestamps; scan deskew is unavailable.",
            "Occlusion boundaries were not manually excluded in this automated diagnostic.",
            "This tests injected signed-shift observability, not absolute depth accuracy.",
        ],
    }
    write_json(output / "qualification.json", qualification)

    factors = [variant["depth_scale"] for variant in variants]
    signed = [
        variant["injected"]["median_signed_stereo_minus_lidar_m"]
        for variant in variants
    ]
    detection = [100.0 * variant["frame_detection_rate"] for variant in variants]
    order = np.argsort(factors)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    axes[0].plot(
        np.asarray(factors)[order],
        np.asarray(signed)[order],
        marker="o",
        color="tab:blue",
    )
    axes[0].set(
        title="Sparse LiDAR signed residual vs injected depth scale",
        xlabel="Injected depth scale",
        ylabel="Median(stereo − LiDAR), m",
    )
    axes[0].grid(alpha=0.25)
    axes[1].bar(
        np.arange(len(order)),
        np.asarray(detection)[order],
        color="tab:green",
    )
    axes[1].axhline(90.0, color="black", linestyle="--")
    axes[1].set(
        title="Per-frame signed-shift detection",
        ylabel="Detection rate (%)",
        xticks=np.arange(len(order)),
        xticklabels=[f"{np.asarray(factors)[index]:.2f}" for index in order],
        ylim=(0, 105),
    )
    axes[1].grid(axis="y", alpha=0.25)
    figure_path = output / "lidar_scale_dose_response.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    source_hashes = {
        str(path): sha256_file(path)
        for path in sorted(set(sample_files))
    }
    provenance = {
        "schema": "daaam.no_gt_d0_lidar_scale_provenance.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "dataset": str(dataset),
        "dataset_tick_index_sha256": sha256_file(tick_path),
        "d0_root": str(d0_root),
        "d0_variant_summary_sha256": sha256_file(
            d0_root / "d0_variant_summary.json"
        ),
        "correspondences": str(correspondences),
        "correspondence_file_count": len(source_hashes),
        "correspondence_sha256": source_hashes,
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output / "provenance.json", provenance)

    report = [
        "# D0 E4：LiDAR 稀疏代理的深度尺度观测资格",
        "",
        (
            f"- 结论：`"
            f"{'PASS' if qualification['e4_scale_observability_passed'] else 'FAIL'}`"
        ),
        f"- reference：`sparse_lidar_proxy_not_human_ground_truth`",
        f"- 逐帧记录：{len(all_frame_rows)}",
        f"- 相机–LiDAR correspondence 文件：{len(source_hashes)}",
        "",
        "> 本实验只验证已知尺度注入能否被 E4 signed residual 发现，不声称绝对深度准确。",
        "",
        "| scale | frames | samples | median signed error m | injected-control m | detection |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in sorted(variants, key=lambda item: item["depth_scale"]):
        report.append(
            f"| {variant['depth_scale']:.2f} | {variant['frame_count']} | "
            f"{variant['sample_count']} | "
            f"{variant['injected']['median_signed_stereo_minus_lidar_m']:.6f} | "
            f"{variant['aggregate_median_injected_minus_control_m']:+.6f} | "
            f"{variant['frame_detection_rate']:.1%} |"
        )
    report.extend(
        [
            "",
            "## 解释",
            "",
            "- E5 的时序一致性对共同深度尺度不具绝对可观测性；缩小深度可能让它看起来更一致。",
            "- E4 的稀疏 LiDAR signed residual 直接提供公制尺度参照，正/负注入应产生同向残差移动。",
            "- 每一对应文件均单独 SHA-256；逐帧数值和全部路径保存在 CSV/JSONL。",
            "",
            "## 限制",
            "",
            *[f"- {item}" for item in qualification["caveats"]],
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "frame_records": len(all_frame_rows),
                "e4_scale_observability_passed": qualification[
                    "e4_scale_observability_passed"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
