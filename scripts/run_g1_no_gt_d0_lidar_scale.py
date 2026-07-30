#!/usr/bin/env python3
"""Qualify D0 depth-scale observability with frozen camera/LiDAR evidence.

The runner never edits its source RGB-D or LiDAR datasets. It evaluates one
nominal control and six already-materialized depth-scale injections, preserves
the native evaluator outputs and per-command resource logs, then computes
paired per-frame dose response with a pre-registered signed-error alarm.

This is diagnostic observability evidence only. Sparse projected LiDAR is not
human-reviewed dense ground truth, and accuracy/ranking claims remain
unavailable.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPOSITORY_ROOT / "scripts/evaluate_stereo_depth_batch_with_lidar.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset", required=True, type=Path)
    parser.add_argument("--control-dataset", required=True, type=Path)
    parser.add_argument("--geometry-d0-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-frame-start", type=int, default=473)
    parser.add_argument("--source-frame-end", type=int, default=573)
    parser.add_argument("--minimum-depth-m", type=float, default=0.25)
    parser.add_argument("--maximum-depth-m", type=float, default=30.0)
    parser.add_argument(
        "--frame-alarm-signed-shift-m",
        type=float,
        default=0.05,
        help="Minimum expected-direction per-frame signed-error shift.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-block-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], *, output: Path, name: str) -> dict[str, Any]:
    logs = output / "command_logs"
    logs.mkdir(exist_ok=True)
    stdout_path = logs / f"{name}.stdout.log"
    stderr_path = logs / f"{name}.stderr.log"
    resource_path = logs / f"{name}.resource_usage.txt"
    started = datetime.now(timezone.utc)
    before = time.monotonic()
    timed_command = ["/usr/bin/time", "-v", "-o", str(resource_path), *command]
    completed = subprocess.run(
        timed_command,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    record = {
        "name": name,
        "argv": command,
        "timed_argv": timed_command,
        "shell_escaped_command": shlex.join(command),
        "cwd": str(REPOSITORY_ROOT),
        "started_utc": started.isoformat(),
        "elapsed_s": time.monotonic() - before,
        "returncode": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "resource_usage": str(resource_path),
    }
    write_json(logs / f"{name}.command.json", record)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with status {completed.returncode}; see {stderr_path}"
        )
    return record


def policy_name(report: dict[str, Any]) -> str:
    filtered = str(report["filtered_policy_name"])
    candidate = f"{filtered}_lidar_reference_0_25_to_5m"
    if candidate in report["aggregate"]["policies"]:
        return candidate
    return filtered


def policy_metrics(report: dict[str, Any]) -> dict[str, float | int]:
    name = policy_name(report)
    policy = report["aggregate"]["policies"][name]
    return {
        "policy": name,
        "count": int(policy["count"]),
        "mean_signed_error_m": float(
            policy["mean_signed_stereo_minus_lidar_m"]
        ),
        "median_signed_error_m": float(
            policy["median_signed_stereo_minus_lidar_m"]
        ),
        "mean_absolute_error_m": float(policy["mean_absolute_error_m"]),
        "median_absolute_error_m": float(policy["median_absolute_error_m"]),
        "p90_absolute_error_m": float(policy["p90_absolute_error_m"]),
    }


def frame_signed_errors(
    report: dict[str, Any],
) -> dict[int, dict[str, float | int]]:
    filtered = str(report["filtered_policy_name"])
    result: dict[int, dict[str, float | int]] = {}
    for frame in report["frames"]:
        error = frame[filtered]["depth_error"]
        result[int(frame["source_index"])] = {
            "selected_count": int(frame[filtered]["selected_count"]),
            "mean_signed_error_m": float(
                error["mean_signed_stereo_minus_lidar_m"]
            ),
            "median_signed_error_m": float(
                error["median_signed_stereo_minus_lidar_m"]
            ),
        }
    return result


def load_correspondence(cell: Path, source_index: int) -> dict[str, np.ndarray]:
    path = cell / "correspondences" / f"{source_index:06d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def frozen_correspondence_metrics(
    control_cell: Path,
    variant_cell: Path,
    source_indices: list[int],
    *,
    minimum_depth_m: float,
    maximum_lidar_depth_m: float = 5.0,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    all_reference: list[np.ndarray] = []
    all_control: list[np.ndarray] = []
    all_variant: list[np.ndarray] = []
    frames: list[dict[str, float | int]] = []
    for source in source_indices:
        control = load_correspondence(control_cell, source)
        variant = load_correspondence(variant_cell, source)
        for field in ("u", "v", "lidar_depth_m"):
            if not np.array_equal(control[field], variant[field]):
                raise RuntimeError(
                    f"Frozen LiDAR correspondence drifted at source {source}: {field}"
                )
        reference = control["lidar_depth_m"].astype(np.float64)
        control_prediction = control["filtered_prediction_m"].astype(np.float64)
        variant_prediction = variant["filtered_prediction_m"].astype(np.float64)
        frozen = (
            control["filtered_valid"].astype(bool)
            & np.isfinite(reference)
            & (reference >= minimum_depth_m)
            & (reference <= maximum_lidar_depth_m)
        )
        if not np.any(frozen):
            raise RuntimeError(f"No frozen eligible LiDAR samples at source {source}")
        if not np.all(np.isfinite(variant_prediction[frozen])):
            raise RuntimeError(
                f"Variant produced non-finite depth on frozen samples at source {source}"
            )
        reference_selected = reference[frozen]
        control_selected = control_prediction[frozen]
        variant_selected = variant_prediction[frozen]
        control_residual = control_selected - reference_selected
        variant_residual = variant_selected - reference_selected
        all_reference.append(reference_selected)
        all_control.append(control_selected)
        all_variant.append(variant_selected)
        frames.append(
            {
                "source_index": source,
                "selected_count": int(frozen.sum()),
                "control_mean_signed_error_m": float(np.mean(control_residual)),
                "variant_mean_signed_error_m": float(np.mean(variant_residual)),
                "control_median_signed_error_m": float(
                    np.median(control_residual)
                ),
                "variant_median_signed_error_m": float(
                    np.median(variant_residual)
                ),
            }
        )
    reference = np.concatenate(all_reference)
    control_prediction = np.concatenate(all_control)
    variant_prediction = np.concatenate(all_variant)
    control_residual = control_prediction - reference
    variant_residual = variant_prediction - reference
    metrics = {
        "count": int(reference.size),
        "control_mean_signed_error_m": float(np.mean(control_residual)),
        "variant_mean_signed_error_m": float(np.mean(variant_residual)),
        "control_median_signed_error_m": float(np.median(control_residual)),
        "variant_median_signed_error_m": float(np.median(variant_residual)),
        "variant_mean_absolute_error_m": float(
            np.mean(np.abs(variant_residual))
        ),
        "variant_median_absolute_error_m": float(
            np.median(np.abs(variant_residual))
        ),
        "variant_p90_absolute_error_m": float(
            np.percentile(np.abs(variant_residual), 90)
        ),
    }
    return metrics, frames


def moving_block_bootstrap(
    values: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    if values.ndim != 1 or not len(values):
        raise ValueError("Bootstrap requires a non-empty vector")
    if block_length < 1 or replicates < 1:
        raise ValueError("Bootstrap settings must be positive")
    sample_size = len(values)
    blocks_needed = int(np.ceil(sample_size / block_length))
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        starts = rng.integers(0, sample_size, size=blocks_needed)
        sample = np.concatenate(
            [
                values[
                    (start + np.arange(block_length, dtype=np.int64))
                    % sample_size
                ]
                for start in starts
            ]
        )[:sample_size]
        estimates[replicate] = float(np.mean(sample))
    return {
        "estimate": float(np.mean(values)),
        "ci95_low": float(np.percentile(estimates, 2.5)),
        "ci95_high": float(np.percentile(estimates, 97.5)),
        "replicates": replicates,
        "block_length_frames": block_length,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    output: Path,
    rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    alarm_threshold_m: float,
) -> list[str]:
    visual = output / "visualizations"
    visual.mkdir()
    paths: list[str] = []

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for branch, color in (("minus", "#2166ac"), ("plus", "#b2182b")):
        selected = sorted(
            [row for row in rows if row["branch"] == branch],
            key=lambda row: float(row["absolute_dose"]),
        )
        doses = [100.0 * float(row["absolute_dose"]) for row in selected]
        axes[0].plot(
            doses,
            [float(row["mean_signed_delta_m"]) for row in selected],
            marker="o",
            color=color,
            label=branch,
        )
        axes[1].plot(
            doses,
            [100.0 * float(row["frame_detection_rate"]) for row in selected],
            marker="s",
            color=color,
            label=branch,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set(
        title="LiDAR signed-error response to injected depth scale",
        xlabel="Absolute scale dose (%)",
        ylabel="Mean paired signed-error shift (m)",
    )
    axes[1].axhline(90.0, color="black", linestyle="--", label="D0 target")
    axes[1].set(
        title="Expected-direction per-frame detection",
        xlabel="Absolute scale dose (%)",
        ylabel="Eligible frames detected (%)",
        ylim=(0.0, 105.0),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    path = visual / "01_lidar_scale_dose_response.png"
    figure.savefig(path, dpi=190)
    plt.close(figure)
    paths.append(str(path))

    variants = [row["variant_id"] for row in rows]
    sources = sorted({int(row["source_index"]) for row in frame_rows})
    matrix = np.full((len(variants), len(sources)), np.nan, dtype=np.float64)
    by_variant_source = {
        (str(row["variant_id"]), int(row["source_index"])): float(
            row["expected_direction_shift_m"]
        )
        for row in frame_rows
    }
    for i, variant in enumerate(variants):
        for j, source in enumerate(sources):
            matrix[i, j] = by_variant_source.get((str(variant), source), np.nan)
    figure, axis = plt.subplots(figsize=(16, 6), constrained_layout=True)
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=max(alarm_threshold_m * 3.0, float(np.nanpercentile(matrix, 95))),
    )
    axis.set_yticks(np.arange(len(variants)), variants)
    tick_positions = np.linspace(0, len(sources) - 1, min(12, len(sources))).astype(int)
    axis.set_xticks(tick_positions, [sources[index] for index in tick_positions])
    axis.set(
        title=(
            "Per-frame expected-direction LiDAR signed shift "
            f"(alarm ≥ {alarm_threshold_m:.3f} m)"
        ),
        xlabel="Raw source tick",
    )
    figure.colorbar(image, ax=axis, label="Expected-direction signed shift (m)")
    path = visual / "02_per_frame_detection_heatmap.png"
    figure.savefig(path, dpi=190)
    plt.close(figure)
    paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    raw_dataset = args.raw_dataset.resolve()
    control_dataset = args.control_dataset.resolve()
    geometry_d0 = args.geometry_d0_run.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if not 0 < args.frame_alarm_signed_shift_m:
        raise ValueError("Frame alarm threshold must be positive")
    output.mkdir(parents=True)

    variants = [
        ("depth_scale_minus_05pct", 0.95),
        ("depth_scale_minus_10pct", 0.90),
        ("depth_scale_minus_20pct", 0.80),
        ("depth_scale_plus_05pct", 1.05),
        ("depth_scale_plus_10pct", 1.10),
        ("depth_scale_plus_20pct", 1.20),
    ]
    preregistration = {
        "schema": "daaam.no_gt_d0_lidar_scale_preregistration.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "diagnostic_gt_free_observability_only",
        "hypothesis": (
            "A multiplicative depth-scale injection causes the camera-minus-LiDAR "
            "signed residual to move in the same direction with ordered dose."
        ),
        "controlled_input": str(control_dataset),
        "single_changed_factor": "filtered uint16 depth multiplied by fixed scale",
        "primary_metric": "paired per-frame mean signed stereo-minus-LiDAR error",
        "guardrails": [
            "eligible raw source frame set is identical",
            "projected LiDAR sample count remains fixed per frame",
            "source RGB-D/LiDAR datasets are read-only",
        ],
        "frame_alarm_rule": {
            "threshold_m": args.frame_alarm_signed_shift_m,
            "detected": (
                "signed_delta * sign(scale - 1) >= threshold_m"
            ),
        },
        "medium_heavy_detection_target": 0.90,
        "ordered_dose_required": True,
        "bootstrap": {
            "method": "circular moving block",
            "block_length_frames": args.bootstrap_block_length,
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
        },
        "evaluation_basis": "proxy",
        "accuracy_claims_allowed": False,
    }
    write_json(output / "preregistration.json", preregistration)

    datasets = [("control", 1.0, control_dataset)]
    datasets.extend(
        (
            variant_id,
            scale,
            geometry_d0 / "variants" / variant_id,
        )
        for variant_id, scale in variants
    )
    commands: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    cell_paths: dict[str, Path] = {}
    for variant_id, scale, dataset in datasets:
        if not dataset.is_dir():
            raise FileNotFoundError(dataset)
        evaluation = output / "cells" / variant_id
        evaluation.parent.mkdir(exist_ok=True)
        command = [
            sys.executable,
            str(EVALUATOR),
            "--raw-dataset",
            str(raw_dataset),
            "--depth-dataset",
            str(dataset),
            "--output",
            str(evaluation),
            "--source-frame-start",
            str(args.source_frame_start),
            "--source-frame-end",
            str(args.source_frame_end),
            "--minimum-depth-m",
            str(args.minimum_depth_m),
            "--maximum-depth-m",
            str(args.maximum_depth_m),
        ]
        commands.append(
            run_command(command, output=output, name=f"evaluate_{variant_id}")
        )
        report = read_json(evaluation / "lidar_batch_evaluation.json")
        report["_injected_scale"] = scale
        reports[variant_id] = report
        cell_paths[variant_id] = evaluation

    control_report = reports["control"]
    control_metrics = policy_metrics(control_report)
    expected_sources = sorted(
        int(source) for source in control_report["source_indices"]
    )
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for variant_id, scale in variants:
        report = reports[variant_id]
        if sorted(int(source) for source in report["source_indices"]) != expected_sources:
            raise RuntimeError(f"Source frame set drifted for {variant_id}")
        metrics, paired_frames = frozen_correspondence_metrics(
            cell_paths["control"],
            cell_paths[variant_id],
            expected_sources,
            minimum_depth_m=args.minimum_depth_m,
        )
        per_frame_expected_shift: list[float] = []
        detections = 0
        direction = 1.0 if scale > 1.0 else -1.0
        for paired_frame in paired_frames:
            source = int(paired_frame["source_index"])
            signed_delta = float(
                paired_frame["variant_mean_signed_error_m"]
            ) - float(paired_frame["control_mean_signed_error_m"])
            expected_shift = direction * signed_delta
            detected = expected_shift >= args.frame_alarm_signed_shift_m
            detections += int(detected)
            per_frame_expected_shift.append(expected_shift)
            frame_rows.append(
                {
                    "variant_id": variant_id,
                    "scale": scale,
                    "source_index": source,
                    "selected_count": int(paired_frame["selected_count"]),
                    "control_mean_signed_error_m": float(
                        paired_frame["control_mean_signed_error_m"]
                    ),
                    "variant_mean_signed_error_m": float(
                        paired_frame["variant_mean_signed_error_m"]
                    ),
                    "signed_delta_m": signed_delta,
                    "expected_direction_shift_m": expected_shift,
                    "detected": detected,
                }
            )
        values = np.asarray(per_frame_expected_shift, dtype=np.float64)
        bootstrap = moving_block_bootstrap(
            values,
            block_length=args.bootstrap_block_length,
            replicates=args.bootstrap_replicates,
            rng=rng,
        )
        mean_signed_delta = float(
            metrics["variant_mean_signed_error_m"]
        ) - float(
            metrics["control_mean_signed_error_m"]
        )
        rows.append(
            {
                "variant_id": variant_id,
                "scale": scale,
                "branch": "plus" if scale > 1.0 else "minus",
                "absolute_dose": abs(scale - 1.0),
                "evaluated_frames": len(expected_sources),
                "selected_lidar_samples": int(metrics["count"]),
                "control_mean_signed_error_m": metrics[
                    "control_mean_signed_error_m"
                ],
                "variant_mean_signed_error_m": metrics[
                    "variant_mean_signed_error_m"
                ],
                "mean_signed_delta_m": mean_signed_delta,
                "expected_direction_mean_shift_m": direction * mean_signed_delta,
                "paired_frame_mean_expected_shift_m": bootstrap["estimate"],
                "paired_frame_mean_expected_shift_ci95_low_m": bootstrap[
                    "ci95_low"
                ],
                "paired_frame_mean_expected_shift_ci95_high_m": bootstrap[
                    "ci95_high"
                ],
                "frame_detection_rate": detections / len(expected_sources),
                "detected_frames": detections,
                "mean_absolute_error_m": metrics[
                    "variant_mean_absolute_error_m"
                ],
                "median_absolute_error_m": metrics[
                    "variant_median_absolute_error_m"
                ],
                "p90_absolute_error_m": metrics[
                    "variant_p90_absolute_error_m"
                ],
            }
        )

    write_csv(output / "d0_lidar_scale_summary.csv", rows)
    write_json(output / "d0_lidar_scale_summary.json", rows)
    write_csv(output / "d0_lidar_scale_per_frame.csv", frame_rows)
    with (output / "d0_lidar_scale_per_frame.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in frame_rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )

    direction_results: dict[str, Any] = {}
    for branch in ("minus", "plus"):
        branch_rows = sorted(
            [row for row in rows if row["branch"] == branch],
            key=lambda row: float(row["absolute_dose"]),
        )
        shifts = np.asarray(
            [
                float(row["expected_direction_mean_shift_m"])
                for row in branch_rows
            ]
        )
        direction_results[branch] = {
            "variant_ids": [row["variant_id"] for row in branch_rows],
            "expected_direction_mean_shift_m": shifts.tolist(),
            "monotone_non_decreasing": bool(np.all(np.diff(shifts) >= -1e-12)),
        }
    medium_heavy = [
        row for row in rows if float(row["absolute_dose"]) >= 0.10 - 1e-12
    ]
    qualification = {
        "schema": "daaam.no_gt_d0_lidar_scale_qualification.v1",
        "scope": "D0 depth-scale injection observed by E4 camera-LiDAR collector",
        "evaluation_basis": "proxy",
        "control_repeat_false_alarm_rate": 0.0,
        "control_repeat_basis": "deterministic same-record self comparison",
        "frame_alarm_threshold_m": args.frame_alarm_signed_shift_m,
        "medium_heavy_detection_target": 0.90,
        "medium_heavy": [
            {
                "variant_id": row["variant_id"],
                "detection_rate": row["frame_detection_rate"],
                "passed": float(row["frame_detection_rate"]) >= 0.90,
            }
            for row in medium_heavy
        ],
        "all_medium_heavy_detected": all(
            float(row["frame_detection_rate"]) >= 0.90
            for row in medium_heavy
        ),
        "ordered_direction": direction_results,
        "both_branches_monotone": all(
            bool(result["monotone_non_decreasing"])
            for result in direction_results.values()
        ),
    }
    qualification["passed"] = bool(
        qualification["all_medium_heavy_detected"]
        and qualification["both_branches_monotone"]
    )
    qualification["authority_limit"] = (
        "This qualifies the E4 proxy collector for known scale faults only; "
        "occlusion-boundary review and dense metric accuracy remain unavailable."
    )
    write_json(output / "d0_lidar_scale_qualification.json", qualification)
    visualizations = plot_results(
        output, rows, frame_rows, args.frame_alarm_signed_shift_m
    )

    report_lines = [
        "# G1 无人工 GT：D0 深度尺度 / LiDAR 观测资格",
        "",
        f"- 状态：`{'PASS' if qualification['passed'] else 'FAIL'}`",
        f"- source 范围：`{args.source_frame_start}–{args.source_frame_end}`",
        f"- eligible selected 帧：`{len(expected_sources)}`",
        f"- control policy：`{control_metrics['policy']}`",
        (
            f"- control mean signed stereo−LiDAR："
            f"`{float(control_metrics['mean_signed_error_m']):.6f} m`"
        ),
        "",
        "> 这里只验证已知尺度故障能否被 E4 proxy collector 发现；不是正式深度准确率。",
        "",
        "| variant | signed Δ m | expected shift m | 95% block-bootstrap CI | detection |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            "| {variant_id} | {mean_signed_delta_m:+.6f} | "
            "{paired_frame_mean_expected_shift_m:.6f} | "
            "[{paired_frame_mean_expected_shift_ci95_low_m:.6f}, "
            "{paired_frame_mean_expected_shift_ci95_high_m:.6f}] | "
            "{frame_detection_rate:.1%} |".format(**row)
        )
    report_lines.extend(
        [
            "",
            "## 证据与限制",
            "",
            "- 每个 cell 保存 76 帧 LiDAR 投影 overlay、完整逐帧/聚合 JSON、summary 图。",
            "- `d0_lidar_scale_per_frame.*` 保存 control/variant 配对残差与报警。",
            "- 每个命令保存 argv、stdout/stderr、返回码、wall time 与 `/usr/bin/time -v`。",
            "- 使用长度 5 的 circular moving-block bootstrap，5000 次。",
            "- LiDAR 无逐点时间且遮挡边界未人工裁决，故 evaluation basis 为 proxy。",
            f"- 图：`{Path(visualizations[0]).relative_to(output)}`、"
            f"`{Path(visualizations[1]).relative_to(output)}`。",
            "",
        ]
    )
    (output / "REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    provenance = {
        "schema": "daaam.no_gt_d0_lidar_scale_run.v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "diagnostic_gt_free_observability_only",
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {
            "raw_dataset": str(raw_dataset),
            "control_dataset": str(control_dataset),
            "geometry_d0_run": str(geometry_d0),
            "control_tick_index_sha256": sha256_file(
                control_dataset / "tick_index.json"
            ),
            "control_pose_sha256": sha256_file(
                control_dataset / "pose/poses.txt"
            ),
        },
        "implementation": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "evaluator": str(EVALUATOR),
            "evaluator_sha256": sha256_file(EVALUATOR),
        },
        "commands": commands,
        "qualification": qualification,
    }
    write_json(output / "provenance.json", provenance)
    print(
        json.dumps(
            {
                "output": str(output),
                "evaluated_frames": len(expected_sources),
                "variants": len(rows),
                "passed": qualification["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
