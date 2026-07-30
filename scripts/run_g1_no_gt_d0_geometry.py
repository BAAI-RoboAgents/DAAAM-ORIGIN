#!/usr/bin/env python3
"""Run read-only D0 geometry fault-injection qualification on an existing RGB-D run.

The source dataset is never modified.  Each injected dataset is materialized in
its own output directory, diagnostics are executed with identical arguments,
and all native deltas, pair records, panels, commands, summaries, figures, and
content hashes are retained.  Results qualify observability only; they are not
ground-truth accuracy or algorithm-ranking evidence.
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
import traceback
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPOSITORY_ROOT / "scripts/build_geometry_perturbation_dataset.py"
DIAGNOSTIC = REPOSITORY_ROOT / "scripts/diagnose_temporal_depth_consistency.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-frame-start", type=int, default=473)
    parser.add_argument("--source-frame-end", type=int, default=573)
    parser.add_argument("--pixel-step", type=int, default=4)
    parser.add_argument("--max-panels", type=int, default=12)
    parser.add_argument("--absolute-tolerance-m", type=float, default=0.04)
    parser.add_argument("--relative-tolerance", type=float, default=0.03)
    parser.add_argument("--max-depth-m", type=float, default=65.535)
    parser.add_argument("--seed", type=int, default=20260728)
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


def command_record(command: list[str], cwd: Path, log_prefix: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    before = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.monotonic() - before
    stdout_path = Path(f"{log_prefix}.stdout.txt")
    stderr_path = Path(f"{log_prefix}.stderr.txt")
    command_path = Path(f"{log_prefix}.command.json")
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    record = {
        "argv": command,
        "shell_escaped_command": shlex.join(command),
        "cwd": str(cwd),
        "started_utc": started.isoformat(),
        "elapsed_s": elapsed,
        "returncode": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    write_json(command_path, record)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {shlex.join(command)}; "
            f"see {record['stderr']}"
        )
    return record


def selected_range(
    dataset: Path, source_frame_start: int, source_frame_end: int
) -> tuple[int, int, list[int], list[int]]:
    tick_index = read_json(dataset / "tick_index.json")
    frames = tick_index["frames"]
    selected: list[int] = []
    selected_raw_sources: list[int] = []
    for record in frames:
        raw_source = record.get("source_idx", record.get("cam0_source_idx"))
        if raw_source is None:
            raise RuntimeError(
                "tick_index is missing raw source_idx/cam0_source_idx; "
                "source_frame_idx is an intermediate prepared-row identifier and "
                "must not be substituted"
            )
        if (
            record.get("source_idx") is not None
            and record.get("cam0_source_idx") is not None
            and int(record["source_idx"]) != int(record["cam0_source_idx"])
        ):
            raise RuntimeError(
                f"Raw source lineage disagrees at selected frame {record.get('idx')}"
            )
        if source_frame_start <= int(raw_source) <= source_frame_end:
            selected.append(int(record["idx"]))
            selected_raw_sources.append(int(raw_source))
    if not selected:
        raise RuntimeError("No selected geometry frames overlap the requested source range")
    expected = list(range(min(selected), max(selected) + 1))
    if selected != expected:
        raise RuntimeError(
            "D0 runner requires a contiguous selected-frame interval; got "
            f"{len(selected)} frames spanning {min(selected)}..{max(selected)}"
        )
    if selected_raw_sources != sorted(selected_raw_sources):
        raise RuntimeError("Raw source indices must be monotonically increasing")
    return min(selected), max(selected) + 1, selected, selected_raw_sources


def summary(report: dict[str, Any]) -> dict[str, float | int]:
    adjacent = report["summary_by_absolute_offset"]["1"]
    return {
        "pairs": int(adjacent["pairs"]),
        "comparable_samples": int(adjacent["comparable_samples"]),
        "agreement_rate_weighted": float(adjacent["agreement_rate_weighted"]),
        "agreement_rate_pair_median": float(
            adjacent["agreement_rate_pair_median"]
        ),
        "median_absolute_depth_error_pair_median_m": float(
            adjacent["median_absolute_depth_error_pair_median_m"]
        ),
        "agreement_rate_pair_p05": float(adjacent["agreement_rate_pair_p05"]),
    }


def pair_map(report: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(pair["reference_frame"]), int(pair["neighbor_frame"])): pair
        for pair in report["pairs"]
        if abs(int(pair["neighbor_offset"])) == 1
    }


def detection_rates(
    control_report: dict[str, Any],
    injected_report: dict[str, Any],
    *,
    agreement_drop_threshold: float = 0.01,
    error_increase_threshold_m: float = 0.005,
) -> dict[str, Any]:
    controls = pair_map(control_report)
    injected = pair_map(injected_report)
    records: list[dict[str, Any]] = []
    for key in sorted(set(controls) & set(injected)):
        control = controls[key]
        fault = injected[key]
        agreement_drop = float(control["agreement_rate"]) - float(
            fault["agreement_rate"]
        )
        error_increase = float(fault["median_absolute_depth_error_m"]) - float(
            control["median_absolute_depth_error_m"]
        )
        detected = (
            agreement_drop >= agreement_drop_threshold
            or error_increase >= error_increase_threshold_m
        )
        records.append(
            {
                "reference_frame": key[0],
                "neighbor_frame": key[1],
                "control_agreement": float(control["agreement_rate"]),
                "injected_agreement": float(fault["agreement_rate"]),
                "agreement_drop": agreement_drop,
                "control_median_error_m": float(
                    control["median_absolute_depth_error_m"]
                ),
                "injected_median_error_m": float(
                    fault["median_absolute_depth_error_m"]
                ),
                "median_error_increase_m": error_increase,
                "detected": detected,
            }
        )
    detected_count = sum(bool(record["detected"]) for record in records)
    return {
        "eligible_pair_count": len(records),
        "detected_pair_count": detected_count,
        "detection_rate": detected_count / len(records) if records else 0.0,
        "thresholds": {
            "agreement_drop": agreement_drop_threshold,
            "median_error_increase_m": error_increase_threshold_m,
            "rule": "agreement_drop >= threshold OR median_error_increase >= threshold",
        },
        "pairs": records,
    }


def ordered_direction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: float(row["dose"]))
    agreement_deterioration = np.asarray(
        [
            float(row["control_agreement_rate_weighted"])
            - float(row["agreement_rate_weighted"])
            for row in rows
        ]
    )
    error_deterioration = np.asarray(
        [
            float(row["median_absolute_depth_error_pair_median_m"])
            - float(row["control_median_error_m"])
            for row in rows
        ]
    )

    def monotone(values: np.ndarray) -> bool:
        return bool(np.all(np.diff(values) >= -1e-12))

    return {
        "ordered_variant_ids": [row["variant_id"] for row in rows],
        "agreement_deterioration": agreement_deterioration.tolist(),
        "median_error_deterioration_m": error_deterioration.tolist(),
        "agreement_direction_monotone": monotone(agreement_deterioration),
        "median_error_direction_monotone": monotone(error_deterioration),
        "at_least_one_primary_metric_monotone": (
            monotone(agreement_deterioration) or monotone(error_deterioration)
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output: Path, rows: list[dict[str, Any]]) -> list[str]:
    visual_dir = output / "visualizations"
    visual_dir.mkdir()
    paths: list[str] = []
    styles = {
        "depth_scale_low": ("tab:blue", "o"),
        "depth_scale_high": ("tab:cyan", "s"),
        "pose_translation": ("tab:orange", "^"),
        "pose_yaw": ("tab:red", "D"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for family, (color, marker) in styles.items():
        family_rows = sorted(
            [row for row in rows if row["dose_family"] == family],
            key=lambda row: float(row["dose"]),
        )
        if not family_rows:
            continue
        doses = [float(row["dose"]) for row in family_rows]
        axes[0].plot(
            doses,
            [float(row["agreement_rate_weighted"]) for row in family_rows],
            marker=marker,
            color=color,
            label=family,
        )
        axes[1].plot(
            doses,
            [
                float(row["median_absolute_depth_error_pair_median_m"])
                for row in family_rows
            ],
            marker=marker,
            color=color,
            label=family,
        )
    axes[0].axhline(
        float(rows[0]["control_agreement_rate_weighted"]),
        color="black",
        linestyle="--",
        label="nominal control",
    )
    axes[1].axhline(
        float(rows[0]["control_median_error_m"]),
        color="black",
        linestyle="--",
        label="nominal control",
    )
    axes[0].set(title="Adjacent temporal agreement dose response", xlabel="Dose")
    axes[1].set(
        title="Adjacent pair median absolute depth error", xlabel="Dose", ylabel="m"
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    path = visual_dir / "01_dose_response.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ordered = sorted(rows, key=lambda row: (row["dose_family"], float(row["dose"])))
    labels = [str(row["variant_id"]) for row in ordered]
    values = [100.0 * float(row["pair_detection_rate"]) for row in ordered]
    bars = axis.bar(
        np.arange(len(ordered)),
        values,
        color=[styles[str(row["dose_family"])][0] for row in ordered],
    )
    axis.axhline(90.0, color="black", linestyle="--", label="D0 medium/heavy target")
    axis.set(
        title="Known-fault pair detection rate",
        ylabel="Detected eligible pairs (%)",
        xticks=np.arange(len(ordered)),
        xticklabels=labels,
        ylim=(0, 105),
    )
    axis.tick_params(axis="x", rotation=50, labelsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 1.5, 102),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    path = visual_dir / "02_pair_detection_rates.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def inventory(output: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output).as_posix()
        if relative in {"artifact_inventory.jsonl", "inventory_summary.json"}:
            continue
        if path.is_symlink():
            records.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": str(path.readlink()),
                    "size_bytes": None,
                    "sha256": None,
                }
            )
        elif path.is_file():
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "target": None,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = output / "artifact_inventory.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    result = {
        "schema": "daaam.d0_geometry_inventory.v1",
        "record_count": len(records),
        "regular_file_count": sum(record["kind"] == "file" for record in records),
        "symlink_count": sum(record["kind"] == "symlink" for record in records),
        "regular_file_bytes": sum(
            int(record["size_bytes"] or 0) for record in records
        ),
        "regular_files_hashed": sum(
            record["kind"] == "file" and bool(record["sha256"]) for record in records
        ),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
    }
    write_json(output / "inventory_summary.json", result)
    return result


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    output.mkdir(parents=True)
    start_frame, end_frame, selected, selected_raw_sources = selected_range(
        dataset, args.source_frame_start, args.source_frame_end
    )
    variants = [
        ("depth_scale_minus_05pct", "depth_scale", 0.05, ["--depth-scale", "0.95"]),
        ("depth_scale_minus_10pct", "depth_scale", 0.10, ["--depth-scale", "0.90"]),
        ("depth_scale_minus_20pct", "depth_scale", 0.20, ["--depth-scale", "0.80"]),
        ("depth_scale_plus_05pct", "depth_scale", 0.05, ["--depth-scale", "1.05"]),
        ("depth_scale_plus_10pct", "depth_scale", 0.10, ["--depth-scale", "1.10"]),
        ("depth_scale_plus_20pct", "depth_scale", 0.20, ["--depth-scale", "1.20"]),
        (
            "pose_translation_02cm",
            "pose_translation",
            0.02,
            ["--pose-translation-m", "0.02"],
        ),
        (
            "pose_translation_05cm",
            "pose_translation",
            0.05,
            ["--pose-translation-m", "0.05"],
        ),
        (
            "pose_translation_10cm",
            "pose_translation",
            0.10,
            ["--pose-translation-m", "0.10"],
        ),
        ("pose_yaw_0p5deg", "pose_yaw", 0.5, ["--pose-yaw-deg", "0.5"]),
        ("pose_yaw_1deg", "pose_yaw", 1.0, ["--pose-yaw-deg", "1.0"]),
        ("pose_yaw_3deg", "pose_yaw", 3.0, ["--pose-yaw-deg", "3.0"]),
    ]
    provenance: dict[str, Any] = {
        "schema": "daaam.no_gt_d0_geometry_run.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "diagnostic_gt_free_observability_only",
        "dataset": str(dataset),
        "dataset_pose_sha256": sha256_file(dataset / "pose/poses.txt"),
        "dataset_tick_index_sha256": sha256_file(dataset / "tick_index.json"),
        "requested_source_frame_range_inclusive": [
            args.source_frame_start,
            args.source_frame_end,
        ],
        "selected_geometry_frame_range_half_open": [start_frame, end_frame],
        "selected_geometry_frame_count": len(selected),
        "selected_raw_source_indices": selected_raw_sources,
        "raw_source_index_field": "source_idx with cam0_source_idx equality check",
        "python": sys.version,
        "platform": platform.platform(),
        "builder": str(BUILDER),
        "builder_sha256": sha256_file(BUILDER),
        "diagnostic": str(DIAGNOSTIC),
        "diagnostic_sha256": sha256_file(DIAGNOSTIC),
        "argv": sys.argv,
        "commands": [],
    }
    write_json(output / "provenance_started.json", provenance)

    common_diagnostic = [
        sys.executable,
        str(DIAGNOSTIC),
        "--start-frame",
        str(start_frame),
        "--end-frame",
        str(end_frame),
        "--frame-step",
        "1",
        "--neighbor-offsets",
        "1",
        "--pixel-step",
        str(args.pixel_step),
        "--min-depth-m",
        "0.25",
        "--max-depth-m",
        str(args.max_depth_m),
        "--absolute-tolerance-m",
        str(args.absolute_tolerance_m),
        "--relative-tolerance",
        str(args.relative_tolerance),
        "--max-panels",
        str(args.max_panels),
        "--forward-only",
        "--require-time-contract",
        "--window-size-frames",
        str(max(2, end_frame - start_frame)),
    ]
    control_dir = output / "control_nominal"
    control_dir.mkdir()
    control_command = common_diagnostic + [
        "--dataset",
        str(dataset),
        "--output-dir",
        str(control_dir / "temporal_diagnostic"),
    ]
    provenance["commands"].append(
        command_record(control_command, REPOSITORY_ROOT, control_dir / "diagnose")
    )
    control_report = read_json(
        control_dir / "temporal_diagnostic/temporal_depth_consistency_report.json"
    )
    control_summary = summary(control_report)
    write_json(control_dir / "control_summary.json", control_summary)

    result_rows: list[dict[str, Any]] = []
    all_detections: dict[str, Any] = {}
    for variant_id, mode, dose, extra in variants:
        variant_dir = output / "variants" / variant_id
        variant_dir.parent.mkdir(exist_ok=True)
        build_command = [
            sys.executable,
            str(BUILDER),
            "--dataset",
            str(dataset),
            "--output",
            str(variant_dir),
            "--mode",
            mode,
            "--start-frame",
            str(start_frame),
            "--end-frame",
            str(end_frame),
            "--seed",
            str(args.seed),
            *extra,
        ]
        provenance["commands"].append(
            command_record(
                build_command,
                REPOSITORY_ROOT,
                output / "variants" / f"{variant_id}.build",
            )
        )
        diagnose_command = common_diagnostic + [
            "--dataset",
            str(variant_dir),
            "--output-dir",
            str(variant_dir / "temporal_diagnostic"),
        ]
        provenance["commands"].append(
            command_record(
                diagnose_command,
                REPOSITORY_ROOT,
                output / "variants" / f"{variant_id}.diagnose",
            )
        )
        variant_report = read_json(
            variant_dir / "temporal_diagnostic/temporal_depth_consistency_report.json"
        )
        variant_summary = summary(variant_report)
        detections = detection_rates(control_report, variant_report)
        all_detections[variant_id] = detections
        dose_family = mode
        if mode == "depth_scale":
            scale = float(extra[-1])
            dose_family = "depth_scale_low" if scale < 1.0 else "depth_scale_high"
        row = {
            "variant_id": variant_id,
            "mode": mode,
            "dose_family": dose_family,
            "dose": dose,
            **variant_summary,
            "control_agreement_rate_weighted": control_summary[
                "agreement_rate_weighted"
            ],
            "control_median_error_m": control_summary[
                "median_absolute_depth_error_pair_median_m"
            ],
            "agreement_delta_vs_control": float(
                variant_summary["agreement_rate_weighted"]
            )
            - float(control_summary["agreement_rate_weighted"]),
            "median_error_delta_vs_control_m": float(
                variant_summary["median_absolute_depth_error_pair_median_m"]
            )
            - float(control_summary["median_absolute_depth_error_pair_median_m"]),
            "pair_detection_rate": detections["detection_rate"],
            "eligible_pair_count": detections["eligible_pair_count"],
            "detected_pair_count": detections["detected_pair_count"],
        }
        result_rows.append(row)
        write_json(variant_dir / "d0_variant_summary.json", row)

    write_csv(output / "d0_variant_summary.csv", result_rows)
    write_json(output / "d0_variant_summary.json", result_rows)
    with (output / "d0_pair_detection.jsonl").open("w", encoding="utf-8") as stream:
        for variant_id, result in all_detections.items():
            for record in result["pairs"]:
                stream.write(
                    json.dumps(
                        {"variant_id": variant_id, **record},
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )

    direction_groups: dict[str, Any] = {}
    for family in sorted({str(row["dose_family"]) for row in result_rows}):
        direction_groups[family] = ordered_direction(
            [row for row in result_rows if row["dose_family"] == family]
        )
    medium_heavy = [
        row
        for row in result_rows
        if (
            row["dose_family"] in {"depth_scale_low", "depth_scale_high"}
            and float(row["dose"]) >= 0.10
        )
        or (
            row["dose_family"] == "pose_translation"
            and float(row["dose"]) >= 0.05
        )
        or (row["dose_family"] == "pose_yaw" and float(row["dose"]) >= 1.0)
    ]
    qualification = {
        "scope": (
            "partial D0: depth scale, alternating local-camera translation, "
            "alternating local-camera yaw; temporal E5 metric only"
        ),
        "control_repeat_false_alarm_rate": 0.0,
        "control_repeat_basis": (
            "deterministic self-comparison of the exact same control pair records"
        ),
        "medium_heavy_detection_target": 0.90,
        "medium_heavy_variants": [
            {
                "variant_id": row["variant_id"],
                "detection_rate": row["pair_detection_rate"],
                "passed": float(row["pair_detection_rate"]) >= 0.90,
            }
            for row in medium_heavy
        ],
        "all_medium_heavy_detection_passed": all(
            float(row["pair_detection_rate"]) >= 0.90 for row in medium_heavy
        ),
        "ordered_direction": direction_groups,
        "all_families_have_a_monotone_primary_metric": all(
            bool(group["at_least_one_primary_metric_monotone"])
            for group in direction_groups.values()
        ),
        "partial_d0_passed": (
            all(float(row["pair_detection_rate"]) >= 0.90 for row in medium_heavy)
            and all(
                bool(group["at_least_one_primary_metric_monotone"])
                for group in direction_groups.values()
            )
        ),
        "not_tested": [
            "stereo time offset",
            "camera-LiDAR time offset",
            "left/right swap",
            "wrong rectification direction",
            "consecutive dropped frames",
            "blur/JPEG",
            "dynamic-mask morphology",
            "track-ID permutation",
            "entity position offset",
            "alias conflict/distractor",
        ],
        "authority_limit": (
            "This qualifies metric sensitivity only. It does not establish depth, "
            "pose, segmentation, tracking, entity, binding, or query accuracy."
        ),
    }
    write_json(output / "d0_partial_qualification.json", qualification)
    plot_paths = make_plots(output, result_rows)

    report_lines = [
        "# G1 无人工 GT：D0 几何观测资格验证",
        "",
        f"- 来源：`{dataset}`（只读，pose SHA-256 `{provenance['dataset_pose_sha256']}`）",
        (
            f"- source 473–573 对应 selected geometry `["
            f"{start_frame}, {end_frame})`，共 {len(selected)} 帧"
        ),
        (
            f"- 实际 raw source：`{selected_raw_sources[0]}–"
            f"{selected_raw_sources[-1]}`；缺失/未选 source tick 保留在 lineage 中"
        ),
        "- 权限：`diagnostic_gt_free_observability_only`",
        (
            f"- 部分 D0 结论：`"
            f"{'PASS' if qualification['partial_d0_passed'] else 'FAIL'}`"
        ),
        "",
        "> 该结论仅说明 E5 时序指标能否响应三类已知注入；不是准确率、算法优选或正式 D0 全通过。",
        "",
        "## Nominal control",
        "",
        (
            f"- weighted agreement："
            f"`{float(control_summary['agreement_rate_weighted']):.6f}`"
        ),
        (
            f"- pair-median absolute depth error："
            f"`{float(control_summary['median_absolute_depth_error_pair_median_m']):.6f} m`"
        ),
        f"- pairs：`{int(control_summary['pairs'])}`",
        "",
        "## 剂量结果",
        "",
        "| variant | agreement | Δagreement | median error m | Δerror m | pair detection |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        report_lines.append(
            "| {variant_id} | {agreement_rate_weighted:.6f} | "
            "{agreement_delta_vs_control:+.6f} | "
            "{median_absolute_depth_error_pair_median_m:.6f} | "
            "{median_error_delta_vs_control_m:+.6f} | "
            "{pair_detection_rate:.1%} |".format(**row)
        )
    report_lines.extend(
        [
            "",
            "## 证据",
            "",
            "- 每个 variant 保留修改后的原生 depth/pose、raw delta、可视化和完整逐对诊断 JSON。",
            "- `d0_pair_detection.jsonl` 保留逐 variant、逐相邻帧的 control/injection 差值和报警结果。",
            "- `*.command.json`、stdout/stderr 保存每个子命令、返回码和耗时。",
            f"- 剂量图：`{Path(plot_paths[0]).relative_to(output)}`、`{Path(plot_paths[1]).relative_to(output)}`。",
            "- `artifact_inventory.jsonl` 对所有普通文件计算 SHA-256，并记录所有符号链接目标。",
            "",
            "## 限制",
            "",
            "- 本次只覆盖 D0 的 depth-scale、pose-translation、pose-yaw 三族及 E5 时序观测。",
            "- control 误报警率来自确定性同输入自比较，不代表跨进程/跨设备随机稳定性。",
            "- 位姿空间注入采用隔帧 local-camera transform；统一刚体变换会保持相对位姿，无法被时序重投影发现。",
            "- 未测试项详见 `d0_partial_qualification.json`；因此不得写成“完整 D0 通过”。",
            "",
        ]
    )
    (output / "REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    provenance["completed_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["control_summary"] = control_summary
    provenance["partial_qualification"] = qualification
    write_json(output / "provenance.json", provenance)
    inventory_result = inventory(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "variant_count": len(result_rows),
                "partial_d0_passed": qualification["partial_d0_passed"],
                "inventory": inventory_result,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        # A failed collector run is itself evidence. Preserve a terminal record
        # when the run directory exists, then re-raise so callers still receive
        # the non-zero status.
        try:
            parsed = parse_args()
            failure_output = parsed.output.resolve()
            if failure_output.is_dir():
                write_json(
                    failure_output / "terminal_failure.json",
                    {
                        "schema": "daaam.no_gt_d0_geometry_terminal_failure.v1",
                        "failed_utc": datetime.now(timezone.utc).isoformat(),
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                        "argv": sys.argv,
                    },
                )
        except BaseException:
            pass
        raise
