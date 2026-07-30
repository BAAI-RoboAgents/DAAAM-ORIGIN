#!/usr/bin/env python3
"""Run the preregistered E16 12 cm observation/range factorial experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from daaam.experiments.e16_support import (  # noqa: E402
    read_mesh_extraction_decisions,
    summarize_semantic_support,
    write_jsonl as write_support_jsonl,
)
import run_g1_no_gt_e16_hydra as base  # noqa: E402


EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments/g1_20260724_473_573_v1_1"
DEFAULT_SOURCE_RUN = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e16_safe035_hydra_20260730"
)
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT
    / "runs/diagnostic_gt_free_e16_12cm_obs_range_sweep_20260730"
)
DEFAULT_TARGET_REVIEW = (
    EXPERIMENT_ROOT
    / "comparisons/codex_approx_gt_e16_should_have_mesh_20260730"
    / "TARGET_REVIEW.jsonl"
)
OBSERVATION_THRESHOLDS = (4, 6, 8)
MAXIMUM_RANGES_M = (5.0, 8.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-review", type=Path, default=DEFAULT_TARGET_REVIEW)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def variant_specs() -> list[dict[str, Any]]:
    rows = []
    for maximum_range_m in MAXIMUM_RANGES_M:
        for observations in OBSERVATION_THRESHOLDS:
            range_token = int(maximum_range_m)
            rows.append(
                {
                    "variant_id": (
                        f"voxel_12cm_obs{observations}_range{range_token}m"
                        "_vol0p005"
                    ),
                    "axis": (
                        "minimum_observations"
                        if maximum_range_m == 5.0
                        else "minimum_observations_x_maximum_range"
                    ),
                    "voxel_size_m": 0.12,
                    "truncation_distance_m": 0.36,
                    "grid_size_m": 0.12,
                    "minimum_observations": observations,
                    "maximum_object_range_m": maximum_range_m,
                    "minimum_volume_m3": 0.005,
                    "diagnostic_only": False,
                }
            )
    return rows


def build_config(
    source: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    config = base.build_variant_config(source, spec)
    active = config["active_window"]
    active["object_detector"]["max_range"] = float(
        spec["maximum_object_range_m"]
    )
    # Logging only: this makes every terminal extractor decision auditable.
    active["object_extractor"]["verbosity"] = 5
    return config


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    base.write_csv(path, rows)


def _copy_frozen_shared_input(source_run: Path, output: Path) -> None:
    source = source_run / "shared_input"
    destination = output / "shared_input"
    if destination.exists():
        return
    shutil.copytree(source, destination)


def _support_rows(
    shared_input: Path,
    maximum_range_m: float,
) -> list[dict[str, Any]]:
    frames = base.read_jsonl(shared_input / "frames.jsonl")[: base.FRAME_COUNT]
    label_paths = [
        shared_input / "label_frames" / f"{index:08d}.png"
        for index in range(base.FRAME_COUNT)
    ]
    depth_paths = [
        shared_input / "static_depth" / f"{index:08d}.png"
        for index in range(base.FRAME_COUNT)
    ]
    return summarize_semantic_support(
        label_paths=label_paths,
        depth_paths=depth_paths,
        frames=frames,
        maximum_range_m=maximum_range_m,
        minimum_cluster_pixels=20,
        observation_thresholds=OBSERVATION_THRESHOLDS,
    )


def _decision_rows(
    output: Path,
    spec: Mapping[str, Any],
    support_by_range: Mapping[float, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    variant_id = str(spec["variant_id"])
    variant = output / "variants" / variant_id
    hydra_logs = sorted(
        (variant / "hydra_realtime/logs").glob("*.log.*")
    )
    decisions = read_mesh_extraction_decisions(
        [
            variant / "stdout.log",
            variant / "stderr.log",
            *hydra_logs,
        ]
    )
    write_support_jsonl(
        variant / "metrics/mesh_extraction_decisions.jsonl",
        decisions,
    )
    by_label: dict[int, list[dict[str, Any]]] = {}
    for decision in decisions:
        by_label.setdefault(int(decision["semantic_label"]), []).append(decision)
    nodes = {
        int(row["semantic_label"])
        for row in base.read_jsonl(variant / "metrics/object_nodes.jsonl")
    }
    rows = []
    support_rows = support_by_range[float(spec["maximum_object_range_m"])]
    observations = str(int(spec["minimum_observations"]))
    for support in support_rows:
        label = int(support["semantic_label"])
        allocation = support["allocation_gate_by_minimum_observations"][
            observations
        ]
        terminal = by_label.get(label, [])
        rows.append(
            {
                "schema": "daaam.g1_e16_label_gate_ledger.v1",
                "variant_id": variant_id,
                "semantic_label": label,
                "maximum_object_range_m": float(
                    spec["maximum_object_range_m"]
                ),
                "configured_minimum_observations": int(
                    spec["minimum_observations"]
                ),
                "cluster_observation_count": int(
                    support["cluster_observation_count"]
                ),
                "required_observations": int(
                    allocation[
                        "required_observations_due_to_strict_confidence_gate"
                    ]
                ),
                "predicted_track_confidence": float(
                    allocation["predicted_track_confidence"]
                ),
                "predicted_allocation_gate_pass": bool(
                    allocation[
                        "passes_allocation_confidence_strictly_above_0p5"
                    ]
                ),
                "total_label_pixels": int(support["total_label_pixels"]),
                "total_in_range_depth_pixels": int(
                    support["total_in_range_depth_pixels"]
                ),
                "map_aabb_volume_m3": support["map_aabb_volume_m3"],
                "extractor_terminal_decision_count": len(terminal),
                "extractor_terminal_decisions_json": json.dumps(
                    [
                        {
                            "decision": value["decision"],
                            "detail": value["detail"],
                        }
                        for value in terminal
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "dsg_object_node_present": label in nodes,
                "decision_status": (
                    "object_node_present"
                    if label in nodes
                    else (
                        "extractor_drop_logged"
                        if any(
                            value["decision"] == "dropped"
                            for value in terminal
                        )
                        else "no_terminal_extractor_log"
                    )
                ),
            }
        )
    return rows


def _target_candidate_rows(
    output: Path,
    specs: Sequence[Mapping[str, Any]],
    target_review_path: Path,
) -> list[dict[str, Any]]:
    targets = base.read_jsonl(target_review_path)
    rows = []
    for spec in specs:
        variant_id = str(spec["variant_id"])
        labels = {
            int(row["semantic_label"])
            for row in base.read_jsonl(
                output
                / "variants"
                / variant_id
                / "metrics/object_nodes.jsonl"
            )
        }
        for target in targets:
            candidates = {int(value) for value in target["candidate_labels"]}
            matched = sorted(candidates & labels)
            rows.append(
                {
                    "schema": "daaam.g1_e16_target_candidate_survival.v1",
                    "variant_id": variant_id,
                    "instance_id": target["instance_id"],
                    "instance_name": target["instance_name"],
                    "denominator": target["denominator"],
                    "baseline_codex_verdict": target["verdict"],
                    "candidate_labels_json": json.dumps(
                        sorted(candidates), separators=(",", ":")
                    ),
                    "surviving_candidate_labels_json": json.dumps(
                        matched, separators=(",", ":")
                    ),
                    "any_candidate_survived": bool(matched),
                    "correctness_status": (
                        "pending_codex_visual_review"
                        if matched
                        else "no_candidate_node"
                    ),
                }
            )
    return rows


def _build_report(
    summaries: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
) -> str:
    core_by_variant: dict[str, list[Mapping[str, Any]]] = {}
    for row in target_rows:
        if row["denominator"] == "core":
            core_by_variant.setdefault(str(row["variant_id"]), []).append(row)
    table = []
    for summary in summaries:
        variant_id = str(summary["variant_id"])
        targets = core_by_variant[variant_id]
        target_candidates = sum(
            bool(row["any_candidate_survived"]) for row in targets
        )
        table.append(
            "| {variant} | {obs} | {range:.0f} | {nodes} | {candidate}/19 | "
            "{p95:.1f} | {maximum:,.1f} | {rss:.0f} | {gate} |".format(
                variant=variant_id,
                obs=int(summary["minimum_observations"]),
                range=float(summary["maximum_object_range_m"]),
                nodes=int(summary["dsg_object_nodes"]),
                candidate=target_candidates,
                p95=float(summary["hydra_processing_p95_ms"]),
                maximum=float(summary["hydra_processing_max_ms"]),
                rss=float(summary["peak_rss_mib"]),
                gate=(
                    "通过"
                    if summary["runtime_hard_gate_passed"]
                    else "失败"
                ),
            )
        )

    missing_with_gate = sum(
        not bool(row["dsg_object_node_present"])
        and bool(row["predicted_allocation_gate_pass"])
        for row in gate_rows
    )
    missing_without_gate = sum(
        not bool(row["dsg_object_node_present"])
        and not bool(row["predicted_allocation_gate_pass"])
        for row in gate_rows
    )
    return f"""# E16 12 cm `obs × max_range` 实验

## 协议

冻结 E11–E15、101 个真实帧、1 个 10.1 s flush、12 cm 全局体素、
0.36 m 截断距离、12 cm object grid、`min_object_volume=0.005 m³`。
仅交叉改变：

- `min_num_observations = 4 / 6 / 8`；
- `object_detector.max_range = 5 / 8 m`。

这 6 个候选在运行前一次性登记。`obs=N` 并不表示 N 帧即可通过：Khronos 在
轨迹置信度 `<=0.5` 时拒绝，而 N 次观测恰好得到 0.5，因此实际分别至少需要
5、7、9 次合格观测。

## 自动结果

| variant | obs | range(m) | DSG nodes | 有任一候选标签的核心目标 | P95(ms) | max(ms) | RSS(MiB) | 实时门 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{os.linesep.join(table)}

“有任一候选标签”只用于定位恢复对象，不能当作真实成功率。特别是历史候选中包含
错误身份和部件标签，最终严格/部分/失败必须结合 RGB、mask 和 object mesh 复核。

## 拒绝链证据

六组共生成 {len(gate_rows):,} 条 label×variant 账本。其中无 object node 且未通过
观测置信门的记录为 {missing_without_gate:,} 条；通过观测门但仍无节点的记录为
{missing_with_gate:,} 条，后者需由提取器日志中的小体积、空 mesh、重建置信度或
缺失终止日志继续定位。

- `tables/semantic_support_range5m.jsonl` / `range8m`：冻结输入支持量；
- `variants/*/metrics/mesh_extraction_decisions.jsonl`：Hydra 原生逐标签终止日志；
- `tables/label_gate_ledger.jsonl`：输入支持、理论置信门、日志和 DSG 结果联表；
- `tables/target_candidate_survival.*`：19/20 目标候选标签生存表；
- `variants/*/metrics/`：mesh、LiDAR 一致性、对象、延迟和资源证据。

## 结论边界

本轮没有正式物体/表面 GT。节点数量、候选标签生存和稀疏 LiDAR 最近邻均是诊断
代理；只有后续 Codex 逐物体工程复核或人工 GT 才能给出目标级近似/正式召回。
"""


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ("git", *args),
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    args = parse_args()
    source_run = args.source_run.resolve()
    output = args.output.resolve()
    target_review = args.target_review.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"output exists; pass --resume to continue: {output}"
        )
    completion = base.read_json(source_run / "COMPLETION.json")
    if completion["status"] != "complete_independently_audited":
        raise ValueError("source E16 run is not independently audited")
    if not target_review.is_file():
        raise FileNotFoundError(target_review)

    output.mkdir(parents=True, exist_ok=True)
    for directory in (
        "configs",
        "failure_cases",
        "source_snapshot",
        "tables",
        "variants",
        "visualizations",
    ):
        (output / directory).mkdir(exist_ok=True)
    for source in (
        Path(__file__),
        REPOSITORY_ROOT / "scripts/run_g1_no_gt_e16_hydra.py",
        REPOSITORY_ROOT / "src/daaam/experiments/e16_support.py",
    ):
        shutil.copy2(source, output / "source_snapshot" / source.name)
    _copy_frozen_shared_input(source_run, output)

    specs = variant_specs()
    preregistration = output / "PRE_REGISTRATION.json"
    if not preregistration.exists():
        base.write_json(
            preregistration,
            {
                "schema": "daaam.g1_e16_obs_range_preregistration.v1",
                "created_at": utc_now(),
                "source_frame_range": [473, 573],
                "source_frames": base.FRAME_COUNT,
                "flush_events": base.FLUSH_EVENT_COUNT,
                "frozen_source_run": base.frozen_reference(
                    source_run / "inventory_summary.json"
                ),
                "frozen_target_review": base.frozen_reference(target_review),
                "variants": specs,
                "fixed_parameters": {
                    "voxel_size_m": 0.12,
                    "truncation_distance_m": 0.36,
                    "object_grid_size_m": 0.12,
                    "minimum_object_volume_m3": 0.005,
                    "allocation_confidence": 0.5,
                    "runtime_hard_gate_ms": base.RUNTIME_HARD_GATE_MS,
                },
                "factorial_axes": {
                    "minimum_observations": list(OBSERVATION_THRESHOLDS),
                    "maximum_object_range_m": list(MAXIMUM_RANGES_M),
                },
                "claims": (
                    "candidate selection and causal diagnostics only; no formal "
                    "object recall or surface accuracy"
                ),
            },
        )
        base.write_json(
            output / "invocation.json",
            {
                "schema": "daaam.g1_e16_obs_range_invocation.v1",
                "argv": sys.argv,
                "cwd": os.getcwd(),
                "python": sys.executable,
                "python_version": platform.python_version(),
                "git_commit": _git_value("rev-parse", "HEAD"),
                "git_branch": _git_value("branch", "--show-current"),
                "started_at": utc_now(),
            },
        )

    source_config = yaml.safe_load(
        (
            source_run
            / "configs/voxel_12cm_obs8_vol0p005.yaml"
        ).read_text(encoding="utf-8")
    )
    parameter_rows = []
    for spec in specs:
        config = build_config(source_config, spec)
        config_path = output / "configs" / f"{spec['variant_id']}.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        parameter_rows.append(
            {
                **spec,
                "config_path": str(config_path),
                "config_sha256": base.sha256_file(config_path),
                "source_config_sha256": base.sha256_file(
                    source_run
                    / "configs/voxel_12cm_obs8_vol0p005.yaml"
                ),
                "extractor_verbosity": 5,
            }
        )
    base.write_json(output / "tables/parameter_matrix.json", parameter_rows)
    _write_csv(output / "tables/parameter_matrix.csv", parameter_rows)

    support_by_range = {}
    for maximum_range_m in MAXIMUM_RANGES_M:
        support_path = (
            output
            / "tables"
            / f"semantic_support_range{int(maximum_range_m)}m.jsonl"
        )
        if support_path.is_file():
            support = base.read_jsonl(support_path)
        else:
            support = _support_rows(
                output / "shared_input", maximum_range_m
            )
            write_support_jsonl(support_path, support)
        support_by_range[maximum_range_m] = support

    frames, label_sha, source_labels, named_labels = base.load_prepared_inputs(
        output
    )
    lidar_points = np.load(
        output / "shared_input/visible_lidar_reference_2cm.npy",
        allow_pickle=False,
    )
    challenges_value = base.read_json(
        EXPERIMENT_ROOT / "manifests/challenge_tags.json"
    )
    challenges = {
        int(record["source_index"]): record
        for record in challenges_value["frames"]
    }
    preview_script = REPOSITORY_ROOT / "scripts/render_hydra_map_preview.py"

    summaries = []
    gate_rows = []
    for spec in specs:
        variant = output / "variants" / str(spec["variant_id"])
        if not (variant / "hydra_postpass_report.json").is_file():
            base.run_variant(
                output,
                spec,
                frames,
                label_sha,
                args.timeout_seconds,
            )
        if not (variant / "hydra_map_preview.png").is_file():
            with (variant / "preview_stdout.log").open(
                "w", encoding="utf-8"
            ) as stdout, (variant / "preview_stderr.log").open(
                "w", encoding="utf-8"
            ) as stderr:
                subprocess.run(
                    [
                        sys.executable,
                        str(preview_script),
                        "--run-dir",
                        str(variant),
                        "--output",
                        str(variant / "hydra_map_preview.png"),
                    ],
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    stdout=stdout,
                    stderr=stderr,
                )
        summary_path = variant / "variant_summary.json"
        if summary_path.is_file():
            summary = base.read_json(summary_path)
        else:
            summary = base.analyze_variant(
                variant,
                spec,
                source_labels,
                named_labels,
                lidar_points,
                frames,
                challenges,
            )
        summaries.append(summary)
        gate_rows.extend(
            _decision_rows(output, spec, support_by_range)
        )

    base.write_json(output / "tables/variant_summary.json", summaries)
    _write_csv(output / "tables/variant_summary.csv", summaries)
    write_support_jsonl(output / "tables/label_gate_ledger.jsonl", gate_rows)
    _write_csv(output / "tables/label_gate_ledger.csv", gate_rows)
    target_rows = _target_candidate_rows(output, specs, target_review)
    write_support_jsonl(
        output / "tables/target_candidate_survival.jsonl", target_rows
    )
    _write_csv(output / "tables/target_candidate_survival.csv", target_rows)

    base.make_visualizations(output, summaries)
    (output / "REPORT.md").write_text(
        _build_report(summaries, target_rows, gate_rows),
        encoding="utf-8",
    )
    base.write_json(
        output / "RUN_SUMMARY.json",
        {
            "schema": "daaam.g1_e16_obs_range_run_summary.v1",
            "status": "complete_pending_independent_audit",
            "variants_complete": len(summaries),
            "variants": summaries,
            "formal_winner": None,
            "codex_target_review": "pending",
            "finished_at": utc_now(),
        },
    )
    base.seal_output(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
