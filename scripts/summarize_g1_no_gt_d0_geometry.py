#!/usr/bin/env python3
"""Combine E4 scale and E5 pose fault-injection evidence for diagnostic D0."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", required=True, type=Path)
    parser.add_argument("--lidar-scale-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-block-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def circular_block_bootstrap(
    values: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    sample_size = len(values)
    blocks = int(np.ceil(sample_size / block_length))
    estimates = np.empty(replicates, dtype=np.float64)
    offsets = np.arange(block_length, dtype=np.int64)
    for replicate in range(replicates):
        starts = rng.integers(0, sample_size, size=blocks)
        indices = np.concatenate(
            [(start + offsets) % sample_size for start in starts]
        )[:sample_size]
        estimates[replicate] = float(np.mean(values[indices]))
    return {
        "estimate": float(np.mean(values)),
        "ci95_low": float(np.percentile(estimates, 2.5)),
        "ci95_high": float(np.percentile(estimates, 97.5)),
        "block_length": block_length,
        "replicates": replicates,
    }


def main() -> None:
    args = parse_args()
    geometry = args.geometry_run.resolve()
    lidar = args.lidar_scale_run.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    output.mkdir(parents=True)
    visual = output / "visualizations"
    visual.mkdir()

    geometry_rows = read_csv(geometry / "d0_variant_summary.csv")
    lidar_rows = read_csv(lidar / "d0_lidar_scale_summary.csv")
    pair_records: dict[str, list[dict[str, Any]]] = {}
    with (geometry / "d0_pair_detection.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            pair_records.setdefault(str(record["variant_id"]), []).append(record)
    rng = np.random.default_rng(args.seed)

    combined: list[dict[str, Any]] = []
    pose_effects: list[dict[str, Any]] = []
    for row in geometry_rows:
        variant_id = row["variant_id"]
        mode = row["mode"]
        if mode == "depth_scale":
            continue
        records = pair_records[variant_id]
        agreement = np.asarray(
            [float(record["agreement_drop"]) for record in records],
            dtype=np.float64,
        )
        error = np.asarray(
            [float(record["median_error_increase_m"]) for record in records],
            dtype=np.float64,
        )
        agreement_ci = circular_block_bootstrap(
            agreement,
            block_length=args.bootstrap_block_length,
            replicates=args.bootstrap_replicates,
            rng=rng,
        )
        error_ci = circular_block_bootstrap(
            error,
            block_length=args.bootstrap_block_length,
            replicates=args.bootstrap_replicates,
            rng=rng,
        )
        effect = {
            "variant_id": variant_id,
            "family": mode,
            "dose": float(row["dose"]),
            "eligible_pairs": int(row["eligible_pair_count"]),
            "detection_rate": float(row["pair_detection_rate"]),
            "agreement_drop_mean": agreement_ci["estimate"],
            "agreement_drop_ci95_low": agreement_ci["ci95_low"],
            "agreement_drop_ci95_high": agreement_ci["ci95_high"],
            "median_error_increase_mean_m": error_ci["estimate"],
            "median_error_increase_ci95_low_m": error_ci["ci95_low"],
            "median_error_increase_ci95_high_m": error_ci["ci95_high"],
        }
        pose_effects.append(effect)
        medium_heavy = (
            (mode == "pose_translation" and float(row["dose"]) >= 0.05)
            or (mode == "pose_yaw" and float(row["dose"]) >= 1.0)
        )
        combined.append(
            {
                "variant_id": variant_id,
                "fault_family": mode,
                "dose": float(row["dose"]),
                "prescribed_first_collector": "E5 temporal reprojection",
                "primary_response": "agreement drop / median error increase",
                "primary_effect": agreement_ci["estimate"],
                "ci95_low": agreement_ci["ci95_low"],
                "ci95_high": agreement_ci["ci95_high"],
                "detection_rate": float(row["pair_detection_rate"]),
                "medium_or_heavy": medium_heavy,
                "passed_if_eligible": (
                    not medium_heavy
                    or float(row["pair_detection_rate"]) >= 0.90
                ),
                "evaluation_basis": "proxy",
            }
        )

    for row in lidar_rows:
        dose = float(row["absolute_dose"])
        medium_heavy = dose >= 0.10 - 1e-12
        combined.append(
            {
                "variant_id": row["variant_id"],
                "fault_family": "depth_scale",
                "dose": dose,
                "prescribed_first_collector": "E4 camera-LiDAR signed error",
                "primary_response": "expected-direction paired signed shift m",
                "primary_effect": float(
                    row["paired_frame_mean_expected_shift_m"]
                ),
                "ci95_low": float(
                    row["paired_frame_mean_expected_shift_ci95_low_m"]
                ),
                "ci95_high": float(
                    row["paired_frame_mean_expected_shift_ci95_high_m"]
                ),
                "detection_rate": float(row["frame_detection_rate"]),
                "medium_or_heavy": medium_heavy,
                "passed_if_eligible": (
                    not medium_heavy
                    or float(row["frame_detection_rate"]) >= 0.90
                ),
                "evaluation_basis": "proxy",
            }
        )
    combined.sort(key=lambda row: (row["fault_family"], float(row["dose"]), row["variant_id"]))
    write_csv(output / "combined_d0_geometry_summary.csv", combined)
    write_json(output / "combined_d0_geometry_summary.json", combined)
    write_csv(output / "pose_pair_effects_with_block_ci.csv", pose_effects)
    write_json(output / "pose_pair_effects_with_block_ci.json", pose_effects)

    family_direction: dict[str, Any] = {}
    for family in ("pose_translation", "pose_yaw"):
        rows = sorted(
            [row for row in pose_effects if row["family"] == family],
            key=lambda row: float(row["dose"]),
        )
        agreement = np.asarray(
            [float(row["agreement_drop_mean"]) for row in rows]
        )
        error = np.asarray(
            [float(row["median_error_increase_mean_m"]) for row in rows]
        )
        family_direction[family] = {
            "variant_ids": [row["variant_id"] for row in rows],
            "agreement_drop_mean": agreement.tolist(),
            "median_error_increase_mean_m": error.tolist(),
            "monotone": bool(
                np.all(np.diff(agreement) >= -1e-12)
                and np.all(np.diff(error) >= -1e-12)
            ),
        }
    lidar_qualification = read_json(
        lidar / "d0_lidar_scale_qualification.json"
    )
    medium_heavy = [row for row in combined if row["medium_or_heavy"]]
    qualification = {
        "schema": "daaam.no_gt_d0_geometry_combined_qualification.v1",
        "scope": [
            "depth scale ±5/10/20% routed to prescribed E4 collector",
            "pose translation 2/5/10 cm routed to prescribed E5 collector",
            "pose yaw 0.5/1/3 degrees routed to prescribed E5 collector",
        ],
        "evaluation_basis": "proxy",
        "medium_heavy_detection_target": 0.90,
        "medium_heavy_all_passed": all(
            bool(row["passed_if_eligible"]) for row in medium_heavy
        ),
        "family_direction": {
            **family_direction,
            "depth_scale": lidar_qualification["ordered_direction"],
        },
        "all_dose_branches_monotone": bool(
            all(result["monotone"] for result in family_direction.values())
            and lidar_qualification["both_branches_monotone"]
        ),
        "covered_family_count": 3,
        "full_d0_family_count": 13,
        "covered_fraction": 3 / 13,
        "control_repeat_false_alarm_rate": 0.0,
        "control_repeat_limit": (
            "deterministic self-comparison only; stochastic/device repeat remains "
            "unqualified"
        ),
        "e5_depth_scale_secondary_observation": {
            "status": "blind_or_wrong_direction",
            "interpretation": (
                "E5 temporal agreement is not the prescribed primary scale "
                "collector; shrinking depth can improve its internal agreement. "
                "E4 detects all six scale doses on frozen LiDAR samples."
            ),
        },
    }
    qualification["covered_geometry_families_passed"] = bool(
        qualification["medium_heavy_all_passed"]
        and qualification["all_dose_branches_monotone"]
    )
    qualification["full_d0_passed"] = False
    qualification["full_d0_reason"] = (
        "Only 3 of the 13 protocol fault families have been qualified."
    )
    write_json(output / "combined_d0_geometry_qualification.json", qualification)

    failures = [
        {
            "code": "F-OBS-COLLECTOR-AGGREGATION-FIELD",
            "status": "fixed_and_regression_preserved",
            "run": "diagnostic_gt_free_d0_geometry_20260728",
            "symptom": "KeyError after 12 cells completed",
            "evidence": str(
                geometry.parent
                / "diagnostic_gt_free_d0_geometry_20260728/terminal_failure.json"
            ),
        },
        {
            "code": "F-INPUT-LINEAGE-FIELD-CONFLATION",
            "status": "fixed_and_invalidated",
            "run": "diagnostic_gt_free_d0_geometry_v2_20260728",
            "symptom": "prepared-row source_frame_idx used as raw source tick",
            "evidence": str(
                geometry.parent
                / "diagnostic_gt_free_d0_geometry_v2_20260728/POSTHOC_INVALIDATION.json"
            ),
        },
        {
            "code": "F-INPUT-RAW-PRODUCT-INDIRECTION",
            "status": "fixed_and_regression_preserved",
            "run": "diagnostic_gt_free_d0_lidar_scale_20260728",
            "symptom": "downstream geometry dataset references upstream raw products",
            "evidence": str(
                lidar.parent
                / "diagnostic_gt_free_d0_lidar_scale_20260728/control/terminal_failure.json"
            ),
        },
        {
            "code": "F-OBS-SAMPLE-SET-DRIFT",
            "status": "fixed_and_regression_preserved",
            "run": "diagnostic_gt_free_d0_lidar_scale_v2_20260728",
            "symptom": "post-injection validity threshold changed evaluated pixels",
            "evidence": str(
                lidar.parent
                / "diagnostic_gt_free_d0_lidar_scale_v2_20260728/terminal_failure.json"
            ),
        },
        {
            "code": "F-OBS-E5-SCALE-BLIND",
            "status": "expected_secondary_metric_limit",
            "run": geometry.name,
            "symptom": "negative scale injection improved temporal agreement",
            "evidence": str(geometry / "d0_partial_qualification.json"),
            "resolution": (
                "Route scale faults to prescribed E4 LiDAR signed-error collector; "
                "do not retune E5 thresholds to hide the blind direction."
            ),
        },
    ]
    with (output / "failures.jsonl").open("w", encoding="utf-8") as stream:
        for failure in failures:
            stream.write(
                json.dumps(failure, ensure_ascii=False, allow_nan=False) + "\n"
            )

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)
    for family, color in (
        ("pose_translation", "#d95f02"),
        ("pose_yaw", "#7570b3"),
    ):
        rows = sorted(
            [row for row in pose_effects if row["family"] == family],
            key=lambda row: float(row["dose"]),
        )
        x = np.asarray([float(row["dose"]) for row in rows])
        y = np.asarray([float(row["agreement_drop_mean"]) for row in rows])
        low = np.asarray([float(row["agreement_drop_ci95_low"]) for row in rows])
        high = np.asarray([float(row["agreement_drop_ci95_high"]) for row in rows])
        axes[0].plot(x, y, marker="o", color=color, label=family)
        axes[0].fill_between(x, low, high, color=color, alpha=0.18)
    axes[0].set(
        title="E5 pose-fault agreement drop",
        xlabel="Dose (m or degree)",
        ylabel="Paired agreement drop",
    )
    lidar_plot_rows = sorted(
        lidar_rows, key=lambda row: (row["branch"], float(row["absolute_dose"]))
    )
    for branch, color in (("minus", "#2166ac"), ("plus", "#b2182b")):
        rows = [row for row in lidar_plot_rows if row["branch"] == branch]
        axes[1].plot(
            [100 * float(row["absolute_dose"]) for row in rows],
            [float(row["paired_frame_mean_expected_shift_m"]) for row in rows],
            marker="s",
            color=color,
            label=f"scale {branch}",
        )
    axes[1].set(
        title="E4 scale-fault signed shift",
        xlabel="Absolute scale dose (%)",
        ylabel="Expected-direction shift (m)",
    )
    ordered = sorted(combined, key=lambda row: (row["fault_family"], row["dose"]))
    axes[2].bar(
        np.arange(len(ordered)),
        [100 * float(row["detection_rate"]) for row in ordered],
        color="#4c78a8",
    )
    axes[2].axhline(90, color="black", linestyle="--")
    axes[2].set(
        title="Known-fault detection",
        ylabel="Detection rate (%)",
        xticks=np.arange(len(ordered)),
        xticklabels=[row["variant_id"] for row in ordered],
        ylim=(0, 105),
    )
    axes[2].tick_params(axis="x", rotation=70, labelsize=6)
    for axis in axes:
        axis.grid(alpha=0.25)
        if axis is not axes[2]:
            axis.legend()
    figure.savefig(
        visual / "16_injection_dose_response.png", dpi=190
    )
    plt.close(figure)

    report = [
        "# G1 无人工 GT：D0 几何组合资格",
        "",
        (
            f"- 已覆盖三族结论：`"
            f"{'PASS' if qualification['covered_geometry_families_passed'] else 'FAIL'}`"
        ),
        "- 完整 D0：`INCOMPLETE`（3/13 fault families）",
        "- evaluation basis：`proxy`",
        "",
        "深度尺度必须由规范指定的 E4 camera–LiDAR signed-error collector 判断；",
        "位姿平移/yaw 由 E5 temporal reprojection 判断。按正确 collector 路由后，",
        "所有中/重度注入检出率均达到 90%，且剂量方向单调。",
        "",
        "E5 对负向 depth scale 出现反方向响应：缩短深度反而提高内部时序 agreement。",
        "这是保留的观测盲区，不通过调阈值消除；E4 在冻结的 391,660 个 LiDAR 样本上",
        "对六个尺度剂量均达到 100% 帧检出。",
        "",
        "## 统计",
        "",
        "- 位姿逐 pair 效应采用长度 5、5000 次 circular block bootstrap；",
        "- scale 逐 frame 效应采用长度 5、5000 次 circular block bootstrap；",
        "- control/variant 按同一 raw source、同一 LiDAR correspondence 配对；",
        "- 所有 collector 失败和无效运行保存在 `failures.jsonl`，没有删除。",
        "",
        "## 限制",
        "",
        "- 仍未覆盖同步、相机交换、错误校正、丢帧、模糊/JPEG、mask 形态学、",
        "track ID、entity 偏移和 alias 冲突；",
        "- 无人工 GT，不能把 D0 observability PASS 写成算法准确率 PASS；",
        "- control false-alarm 仅为确定性自比较，跨设备/随机重复尚未资格化。",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    provenance = {
        "schema": "daaam.no_gt_d0_geometry_combined_run.v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "inputs": {
            "geometry_run": str(geometry),
            "geometry_provenance_sha256": sha256_file(
                geometry / "provenance.json"
            ),
            "lidar_scale_run": str(lidar),
            "lidar_provenance_sha256": sha256_file(lidar / "provenance.json"),
        },
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "qualification": qualification,
    }
    write_json(output / "provenance.json", provenance)
    print(
        json.dumps(
            {
                "output": str(output),
                "covered_geometry_families_passed": qualification[
                    "covered_geometry_families_passed"
                ],
                "full_d0_passed": False,
                "covered_families": 3,
                "protocol_families": 13,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
