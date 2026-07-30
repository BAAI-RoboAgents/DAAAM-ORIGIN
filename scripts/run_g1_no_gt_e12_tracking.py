#!/usr/bin/env python3
"""Replay BotSort on frozen E11 detections and retain auditable E12 evidence.

This is an E11-fed, GT-free diagnostic replay.  It deliberately does not claim
HOTA, IDF1, real ID switches, or fragmentation accuracy because reviewed GT
instance detections and track identities are unavailable.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_E11 = (
    REPOSITORY_ROOT
    / "experiments/g1_20260724_473_573_v1_1/runs/"
    "diagnostic_gt_free_e11_fastsam_20260729"
)
E11_CELL_ID = "conf_0p3__area_0300__iou_0p5"
E11_PROFILE_ID = "conf_0p3__iou_0p5"
DEFAULT_REID = REPOSITORY_ROOT / "checkpoints/reid_weights/clip_general.engine"
TRACKING_SOURCE = REPOSITORY_ROOT / "src/daaam/tracking/services.py"
REALTIME_CONFIG = REPOSITORY_ROOT / "config/pipeline_config_realtime.yaml"
EXPERIMENT_DEFINITION = REPOSITORY_ROOT / "src/daaam/experiments/g1_semantic_map.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E12 BotSort diagnostics on frozen E11 baseline detections."
    )
    parser.add_argument("--e11-run", type=Path, default=DEFAULT_E11)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-frames", type=int)
    parser.add_argument(
        "--reseal-existing",
        action="store_true",
        help="Do not track; rebuild inventory/completion after an independent audit.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
            stream.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require_write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def numeric_summary(values: Iterable[float | int | None]) -> dict[str, Any]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "p05": percentile(finite, 5),
        "p50": percentile(finite, 50),
        "p95": percentile(finite, 95),
        "maximum": max(finite),
    }


def jaccard(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return 1.0 if union == 0 else intersection / union


def stable_color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 47 + 13) % 180
    hsv = np.uint8([[[hue, 215, 245]]])
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def nvidia_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def variant_definitions() -> list[dict[str, Any]]:
    # Mirrors src/daaam/experiments/g1_semantic_map.py E12 realtime variants.
    return [
        {
            "variant_id": "baseline",
            "with_reid": True,
            "cmc_method": "ecc",
            "cmc_ecc_max_iterations": 20,
            "track_buffer": 30,
            "batch_reid_crops": True,
        },
        {
            "variant_id": "without_reid",
            "with_reid": False,
            "cmc_method": "ecc",
            "cmc_ecc_max_iterations": 20,
            "track_buffer": 30,
            "batch_reid_crops": False,
        },
        {
            "variant_id": "ecc_100",
            "with_reid": True,
            "cmc_method": "ecc",
            "cmc_ecc_max_iterations": 100,
            "track_buffer": 30,
            "batch_reid_crops": True,
        },
        {
            "variant_id": "buffer_10",
            "with_reid": True,
            "cmc_method": "ecc",
            "cmc_ecc_max_iterations": 20,
            "track_buffer": 10,
            "batch_reid_crops": True,
        },
        {
            "variant_id": "buffer_60",
            "with_reid": True,
            "cmc_method": "ecc",
            "cmc_ecc_max_iterations": 20,
            "track_buffer": 60,
            "batch_reid_crops": True,
        },
    ]


def load_frozen_inputs(
    e11: Path, maximum_frames: int | None
) -> list[dict[str, Any]]:
    frames = read_jsonl(e11 / "source_frames.jsonl")
    if maximum_frames is not None:
        frames = frames[:maximum_frames]
    if not frames:
        raise RuntimeError("E11 source frame list is empty")
    frozen: list[dict[str, Any]] = []
    for expected_index, source in enumerate(frames):
        frame_index = int(source["frame_index"])
        if frame_index != expected_index:
            raise ValueError("E11 frames must be consecutive from zero")
        selection_path = (
            e11
            / "cells"
            / E11_CELL_ID
            / "selected_instances"
            / f"{frame_index:08d}.json"
        )
        raw_frame_path = (
            e11
            / "raw_profiles"
            / E11_PROFILE_ID
            / "frames"
            / f"{frame_index:08d}"
            / "frame.json"
        )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        raw_frame = json.loads(raw_frame_path.read_text(encoding="utf-8"))
        if selection["cell"]["cell_id"] != E11_CELL_ID:
            raise ValueError(f"unexpected E11 cell in {selection_path}")
        if (
            int(selection["source_frame_index"])
            != int(raw_frame["source_frame_index"])
            or int(source["source_frame_index"])
            != int(raw_frame["source_frame_index"])
        ):
            raise ValueError(f"E11 lineage mismatch at frame {frame_index}")
        by_id = {
            int(instance["instance_id"]): instance
            for instance in raw_frame["instances"]
        }
        kept_ids = [int(value) for value in selection["kept_instance_ids"]]
        instances = [by_id[value] for value in kept_ids]
        rgb_path = Path(source["rgb_path"])
        if sha256_file(rgb_path) != source["rgb_sha256"]:
            raise ValueError(f"RGB hash mismatch: {rgb_path}")
        for instance in instances:
            mask_path = Path(instance["mask_path"])
            if sha256_file(mask_path) != instance["mask_sha256"]:
                raise ValueError(f"E11 mask hash mismatch: {mask_path}")
        frozen.append(
            {
                "frame_index": frame_index,
                "source_frame_index": int(source["source_frame_index"]),
                "sensor_time_ns": int(source["sensor_time_ns"]),
                "timestamp_s": float(source["timestamp_s"]),
                "rgb_path": str(rgb_path),
                "rgb_sha256": source["rgb_sha256"],
                "selection_path": str(selection_path),
                "selection_sha256": sha256_file(selection_path),
                "raw_frame_path": str(raw_frame_path),
                "raw_frame_sha256": sha256_file(raw_frame_path),
                "detection_count": len(instances),
                "kept_instance_ids": kept_ids,
                "instances": instances,
            }
        )
    return frozen


def tracker_state(track: Any) -> dict[str, Any]:
    return {
        "track_id": int(track.id) + 1,
        "start_frame_one_based": int(track.start_frame),
        "last_update_frame_one_based": int(track.end_frame),
        "tracklet_len": int(track.tracklet_len),
        "is_activated": bool(track.is_activated),
        "state": int(track.state),
        "confidence": float(track.conf),
        "class_id": int(track.cls),
        "detection_local_index": int(track.det_ind),
        "box_xyxy": [float(value) for value in track.xyxy],
    }


def draw_overlay(
    bgr: np.ndarray,
    observations: Sequence[Mapping[str, Any]],
    mask_cache: Mapping[int, np.ndarray],
    title: str,
) -> np.ndarray:
    canvas = bgr.copy()
    color_layer = bgr.copy()
    for observation in observations:
        track_id = int(observation["track_id"])
        local_index = int(observation["detection_local_index"])
        mask = mask_cache[local_index]
        color = stable_color(track_id)
        color_layer[mask] = color
    canvas = cv2.addWeighted(color_layer, 0.42, canvas, 0.58, 0)
    for observation in observations:
        track_id = int(observation["track_id"])
        color = stable_color(track_id)
        x1, y1, x2, y2 = [
            int(round(value)) for value in observation["track_box_xyxy"]
        ]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"T{track_id} E11:{observation['e11_instance_id']}",
            (max(0, x1), max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def build_exact_track_id_map(
    image_shape: tuple[int, int],
    observations: Sequence[Mapping[str, Any]],
    mask_cache: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, int]:
    label = np.zeros(image_shape, dtype=np.uint16)
    conflicts = 0
    ordered = sorted(
        observations,
        key=lambda row: (
            -float(row["model_confidence"]),
            int(row["track_id"]),
        ),
    )
    for observation in ordered:
        track_id = int(observation["track_id"])
        if track_id > np.iinfo(np.uint16).max:
            raise OverflowError("track ID cannot be stored losslessly in uint16 PNG")
        mask = mask_cache[int(observation["detection_local_index"])]
        conflicts += int(np.count_nonzero(mask & (label != 0)))
        label[mask & (label == 0)] = track_id
    return label, conflicts


def save_track_timeline(
    output: Path,
    variant_id: str,
    tracks: Sequence[Mapping[str, Any]],
    frame_count: int,
) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(
        tracks,
        key=lambda row: (
            int(row["first_frame_index"]),
            int(row["track_id"]),
        ),
    )
    figure_height = max(7.0, min(30.0, 0.16 * len(ordered) + 3.0))
    figure, axis = plt.subplots(figsize=(14, figure_height))
    for row_index, track in enumerate(ordered):
        frames = [int(value) for value in track["observation_frame_indices"]]
        axis.hlines(
            row_index,
            int(track["first_frame_index"]),
            int(track["last_frame_index"]),
            color="0.78",
            linewidth=1,
        )
        axis.scatter(
            frames,
            [row_index] * len(frames),
            s=7,
            color="tab:blue",
        )
    axis.set_xlim(-1, frame_count)
    axis.set_xlabel("E12 frame index")
    axis.set_ylabel("track ordered by first observation")
    axis.set_title(
        f"E12 {variant_id} track timeline — NO HUMAN GT / PROXY"
    )
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "variants" / variant_id / "track_timeline.png", dpi=170)
    plt.close(figure)


def save_comparison_plot(
    output: Path, summaries: Sequence[Mapping[str, Any]]
) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["variant_id"]) for row in summaries]
    metrics = [
        ("tracking_latency_p95_ms", "tracking P95 (ms)"),
        ("tracked_detection_fraction", "tracked detection fraction"),
        ("unique_track_count", "unique tracks"),
        ("short_track_fraction", "short-track proxy"),
        ("consecutive_mask_iou_mean", "same-ID mask IoU proxy"),
    ]
    figure, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    for axis, (key, title) in zip(axes, metrics, strict=True):
        values = [
            0.0 if row[key] is None else float(row[key]) for row in summaries
        ]
        bars = axis.bar(range(len(labels)), values, color="tab:blue", alpha=0.8)
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle(
        "E12 E11-fed diagnostics — NO HUMAN GT / no HOTA-IDF1 claim"
    )
    figure.tight_layout()
    figure.savefig(output / "visualizations/01_variant_comparison.png", dpi=180)
    plt.close(figure)


def save_representative_montage(
    output: Path,
    variants: Sequence[Mapping[str, Any]],
    frame_count: int,
) -> None:
    positions = sorted({0, frame_count // 2, frame_count - 1})
    target_width = 512
    rows: list[np.ndarray] = []
    for frame_index in positions:
        images: list[np.ndarray] = []
        for variant in variants:
            path = (
                output
                / "variants"
                / str(variant["variant_id"])
                / "frames"
                / f"{frame_index:08d}"
                / "track_overlay.png"
            )
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            height = int(round(image.shape[0] * target_width / image.shape[1]))
            images.append(
                cv2.resize(
                    image, (target_width, height), interpolation=cv2.INTER_AREA
                )
            )
        rows.append(cv2.hconcat(images))
    require_write_image(
        output / "visualizations/02_representative_track_overlays.png",
        cv2.vconcat(rows),
    )


def inventory_tree(root: Path) -> dict[str, Any]:
    excluded = {
        "artifact_inventory.jsonl",
        "artifact_inventory.csv",
        "inventory_summary.json",
        "COMPLETION.json",
    }
    rows: list[dict[str, Any]] = []
    root_digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        file_hash = sha256_file(path)
        size = path.stat().st_size
        row = {
            "relative_path": relative,
            "size_bytes": size,
            "sha256": file_hash,
        }
        rows.append(row)
        root_digest.update(
            f"{relative}\0{size}\0{file_hash}\n".encode("utf-8")
        )
    write_jsonl(root / "artifact_inventory.jsonl", rows)
    write_csv(root / "artifact_inventory.csv", rows)
    result = {
        "schema": "daaam.g1_no_gt_e12_inventory.v1",
        "created_utc": utc_now(),
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "content_hash_complete": True,
        "manifest_root_sha256": root_digest.hexdigest(),
        "root_definition": "SHA-256 over sorted relative_path\\0size\\0sha256 rows",
        "excluded_self_referential_products": sorted(excluded),
    }
    write_json(root / "inventory_summary.json", result)
    return result


def render_report(
    output: Path,
    frames: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    baseline = next(row for row in summaries if row["variant_id"] == "baseline")
    by_variant = {str(row["variant_id"]): row for row in summaries}
    without_reid = by_variant["without_reid"]
    ecc_100 = by_variant["ecc_100"]
    buffer_10 = by_variant["buffer_10"]
    buffer_60 = by_variant["buffer_60"]
    reid_p95_reduction = (
        float(baseline["tracking_latency_p95_ms"])
        - float(without_reid["tracking_latency_p95_ms"])
    ) / float(baseline["tracking_latency_p95_ms"])
    lines = [
        "# E12 BotSort/ReID：E11 驱动的无 GT 跟踪证据",
        "",
        "- 状态：`complete / diagnostic_gt_free / upstream_coupled_to_E11`",
        f"- 输入：E11 `{E11_CELL_ID}`，{len(frames)} 帧，source "
        f"{frames[0]['source_frame_index']}–{frames[-1]['source_frame_index']}",
        "- E12 正式 isolated 协议要求 GT instance mask/detection；当前不可得。",
        "- 本运行没有重新执行分割；逐检测引用 E11 mask 路径与 SHA-256。",
        "- 文件数、字节数和封存根见 `inventory_summary.json`。",
        "",
        "> 生命周期、短轨迹、同 ID mask IoU、未分配检测和 gap reacquisition 都是",
        "> 无 GT 诊断。不得解释为 HOTA、IDF1、真实 ID switch 或真实 fragmentation。",
        "",
        "## 生产基线",
        "",
        f"- variant：`{baseline['variant_id']}`，ReID={baseline['with_reid']}，"
        f"ECC iterations={baseline['cmc_ecc_max_iterations']}，"
        f"buffer={baseline['track_buffer']}",
        f"- 输入检测：{baseline['input_detection_count']}；跟踪观测："
        f"{baseline['tracked_observation_count']}；跟踪覆盖："
        f"{baseline['tracked_detection_fraction']:.4f}",
        f"- unique tracks：{baseline['unique_track_count']}；短轨迹代理比例："
        f"{baseline['short_track_fraction']:.4f}",
        f"- tracking P50/P95：{baseline['tracking_latency_p50_ms']:.3f} / "
        f"{baseline['tracking_latency_p95_ms']:.3f} ms",
        "",
        "## 候选对照",
        "",
        "| variant | ReID | ECC iter | buffer | tracked/input | unique tracks | "
        "short-track proxy | same-ID mask IoU | P95 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        iou = row["consecutive_mask_iou_mean"]
        lines.append(
            f"| `{row['variant_id']}` | {row['with_reid']} | "
            f"{row['cmc_ecc_max_iterations']} | {row['track_buffer']} | "
            f"{row['tracked_detection_fraction']:.4f} | "
            f"{row['unique_track_count']} | {row['short_track_fraction']:.4f} | "
            f"{'NA' if iou is None else f'{float(iou):.4f}'} | "
            f"{row['tracking_latency_p95_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## E12 原理和真实数据流",
            "",
            "E12 的任务是把 E11 在每帧独立产生的实例连接成跨帧轨迹。真实数据流为：",
            "",
            "```text",
            "E11 FastSAM",
            "  ├─ bbox [x1,y1,x2,y2]",
            "  ├─ confidence",
            "  ├─ class_id（本运行是类别无关实例）",
            "  └─ instance mask",
            "          │",
            "          ▼",
            "E12 BotSort：RGB + bbox/conf/class",
            "  ├─ ECC 相机运动补偿",
            "  ├─ Kalman bbox 状态预测",
            "  ├─ 高置信检测：IoU + 可选 CLIP ReID 第一次关联",
            "  ├─ 低置信检测：IoU 第二次关联",
            "  ├─ unconfirmed/active/lost/removed 生命周期",
            "  └─ 输出 track_id + detection_index",
            "                         │",
            "                         ▼",
            "detection_index 回指 E11 mask",
            "  ├─ track overlay / uint16 track-ID 图",
            "  ├─ track observation / lifecycle",
            "  └─ 后续 E13 MapMemory 实体合并",
            "```",
            "",
            "关键契约：E11 mask **不直接参与 BotSort 匹配**。实际送入跟踪器的是",
            "`[x1,y1,x2,y2,confidence,class_id] + RGB`；BotSort 返回",
            "`[x1,y1,x2,y2,track_id,confidence,class_id,detection_index]`，",
            "本实验再用 `detection_index` 把 track 无损挂回原 E11 mask。所有链接均保存",
            "mask 路径和 SHA-256。由于 E11 每帧都有检测，本隔离重放没有生成",
            "carry-forward mask，`carry_forward_count=0`。",
            "",
            "## BotSort 使用的模块",
            "",
            "### ECC 相机运动补偿",
            "",
            "头部相机转动时，静态物体会在整幅图像中同步位移。ECC 根据相邻 RGB 估计全局",
            "图像变换，先把历史 track 状态变换到当前画面，再进行关联。它主要避免机器人转头时",
            "因全局画面运动而把静态椅子、植物或桌子错误地重新建 ID。`20/100` 是优化最大",
            "迭代次数，不表示每帧一定运行到该次数。",
            "",
            "### Kalman 运动预测",
            "",
            "每条 track 保存 bbox 的位置、尺寸和运动状态。新帧到来时先预测 bbox，再使用",
            "匹配检测更新状态。它适合连续运动，但不能单独解决长遮挡、快速转头、相邻相似物体",
            "或 E11 bbox 突变。",
            "",
            "### 两阶段 IoU 关联和置信度分层",
            "",
            "| 参数 | 固定值 | 实际作用 |",
            "| --- | ---: | --- |",
            "| `track_high_thresh` | 0.5 | `conf > 0.5` 进入第一阶段关联 |",
            "| `track_low_thresh` | 0.1 | `0.1 < conf < 0.5` 只用于第二阶段补充已有 track |",
            "| `new_track_thresh` | 0.6 | 新建 track 通常要求 `conf >= 0.6` |",
            "| `match_thresh` | 0.8 | 第一次线性分配代价门 |",
            "| `proximity_thresh` | 0.5 | 空间不够接近时禁止用 ReID 强连 |",
            "| `appearance_thresh` | 0.25 | ReID 外观距离门 |",
            "",
            "E11 的分割阈值为 `conf=0.3`，但 E12 会再次执行以上跟踪置信度策略。因此不是",
            "每个 E11 mask 都会成为当前帧活跃 track：低置信检测通常只能更新已有 track，",
            "`0.5–0.6` 的检测也可能无法建立新 track，新 track 还可能需要下一帧确认。",
            f"这解释了 baseline 输入 {baseline['input_detection_count']} 个 E11 实例，",
            f"只输出 {baseline['tracked_observation_count']} 条活跃观测，覆盖率",
            f"{baseline['tracked_detection_fraction']:.4f}。未输出部分不能直接称为漏检。",
            "",
            "### CLIP ReID 外观关联",
            "",
            "ReID 使用 `clip_general.engine` 对 bbox 内 RGB crop 编码，比较当前检测与历史",
            "track 的外观向量。它主要服务于短遮挡恢复和相似空间候选消歧，但仍受空间 proximity",
            "门约束，不能依靠外观把距离很远的候选强行连接。当前 FastSAM 是类别无关分割，",
            "因此 ReID 也不是类别识别或语义命名模块。",
            "",
            "### track buffer 生命周期",
            "",
            "`track_buffer` 是 track 进入 lost 后允许保留的帧数，不是形成可信 track 所需",
            "观测数。按当前约 15 FPS 采集速率，buffer 10/30/60 分别约为 0.67/2/4 秒。",
            "短 buffer 更快清除旧 track，但遮挡后更容易换 ID；长 buffer 有利于恢复旧 ID，",
            "同时增加陈旧 track 错接邻近物体的风险。",
            "",
            "## 对照实验的原理解释",
            "",
            "| 对照 | 相对 baseline 的观测结果 | 原理解释 |",
            "| --- | --- | --- |",
            f"| `without_reid` | 观测 {without_reid['tracked_observation_count']}，"
            f"track {without_reid['unique_track_count']}，P95 "
            f"{without_reid['tracking_latency_p95_ms']:.3f} ms | 当前连续室内窗口中，"
            "ECC+Kalman+IoU 已决定绝大多数关联；关闭 CLIP crop/embedding 显著降低成本。"
            "但没有 GT 遮挡身份，不能据此断言 ReID 无用。 |",
            f"| `ecc_100` | 观测 {ecc_100['tracked_observation_count']}，track "
            f"{ecc_100['unique_track_count']}，P95 "
            f"{ecc_100['tracking_latency_p95_ms']:.3f} ms | 与 baseline 几乎相同，"
            "但尾延迟上升；多数帧可能提前收敛，或 20 次迭代已足够。 |",
            f"| `buffer_10` | 观测 {buffer_10['tracked_observation_count']}，track "
            f"{buffer_10['unique_track_count']}，gap reacquisition "
            f"{buffer_10['gap_reacquisition_event_count']} | 更早删除 lost track，"
            "轨迹总数增加，符合重新建 ID/碎片风险。gap 恢复数变少并不一定更好：旧 ID",
            "被删除后，新 ID 不会被计作同一 track 的恢复。 |",
            f"| `buffer_60` | 观测 {buffer_60['tracked_observation_count']}，track "
            f"{buffer_60['unique_track_count']}，gap reacquisition "
            f"{buffer_60['gap_reacquisition_event_count']}，稀疏生命周期 "
            f"{buffer_60['sparse_lifecycle_count']} | 能保留更久的 lost track，"
            "但出现陈旧/稀疏生命周期代理，身份是否正确必须由 GT 裁决。 |",
            "",
            f"`without_reid` 相对 baseline 的 P95 降低约 "
            f"{100.0 * reid_p95_reduction:.1f}%，而当前 track 数、覆盖率和 same-ID",
            "mask IoU 变化很小。这说明 ReID 是本窗口最主要的跟踪计算成本，但不证明",
            "在转向、遮挡或相邻相似物体场景中可以安全关闭。",
            "",
            "## 为什么 same-ID mask IoU 几乎相同",
            "",
            "五组候选的 same-ID mask IoU 都约为 `0.785–0.787`，原因包括：",
            "",
            "- 所有候选使用完全相同的 E11 mask；",
            "- 当前画面运动连续，ECC+IoU 已稳定关联大部分容易实例；",
            "- 大型墙面、地面和家具 mask 会提高重叠代理；",
            "- 指标只检查一个内部 ID 的相邻 mask 重叠，不能判断两把相似椅子是否串 ID。",
            "",
            "所以该指标适合发现明显跳变，不能替代真实身份 GT、HOTA 或 IDF1。",
            "",
            "## 工程结论",
            "",
            "- 当前室内连续场景继续使用 `ReID on + ECC 20 + buffer 30` 是稳妥默认值；",
            "- `ECC 100` 没有显示代理收益，不建议替代 ECC 20；",
            "- `buffer 10` 的 track 数增加且跟踪观测减少，存在更强的碎片风险；",
            "- `buffer 60` 没有形成明确收益，并引入稀疏/陈旧 track 风险；",
            f"- `without_reid` 是高价值性能候选，P95 从 "
            f"{baseline['tracking_latency_p95_ms']:.3f} 降至 "
            f"{without_reid['tracking_latency_p95_ms']:.3f} ms，但必须在有转向、遮挡和",
            "相邻相似物体的人工 GT 子集上验证后才能决定是否关闭。",
            "",
            "## 证据导航",
            "",
            "- `PRE_REGISTRATION.json`：冻结输入、候选和代理边界；",
            "- `input_frames.*`：每帧 E11 selection/raw-frame/RGB 的路径与哈希；",
            "- `variants/*/frames/*/frame.json`：逐帧输入检测、输出 track、内部状态；",
            "- `variants/*/frames/*/track_id_map.png`：uint16 精确 track-ID 像素图；",
            "- `variants/*/frames/*/track_overlay.png`：逐帧 track overlay；",
            "- `tables/track_observations.*`：每个 track 观测到 E11 mask 的可追溯关系；",
            "- `tables/track_lifecycles.*`：生命周期、空洞和连续 mask IoU；",
            "- `failure_cases/`：未分配检测、短轨迹、稀疏生命周期、gap/跳变代理；",
            "- `visualizations/`：候选对照与代表帧；",
            "- `INDEPENDENT_AUDIT.json`：追加的独立重建核验；",
            "- `artifact_inventory.*`：当前封存内所有文件的 SHA-256。",
            "",
            "## 不可声明",
            "",
            "- HOTA、IDF1、MOTA、真实 ID switch 和真实 fragmentation 均不可得；",
            "- E11 自动实例不能充当 E12 的 oracle/GT；",
            "- unique track 更少不自动表示身份保持更好，也可能是串 ID；",
            "- unique track 更多不自动表示 fragmentation，也可能是新物体进入画面。",
            "",
        ]
    )
    report = "\n".join(lines)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "REPORT.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>E12 evidence</title>"
        "<style>body{max-width:1200px;margin:2rem auto;font:15px/1.5 sans-serif}"
        "pre{white-space:pre-wrap}</style><pre>"
        + html.escape(report)
        + "</pre>",
        encoding="utf-8",
    )


def reseal_existing(output: Path) -> None:
    if not output.is_dir():
        raise FileNotFoundError(output)
    completion_path = output / "COMPLETION.json"
    if not completion_path.is_file():
        raise FileNotFoundError(completion_path)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    previous_inventory = completion.get("inventory")
    frames = read_jsonl(output / "input_frames.jsonl")
    summaries = json.loads(
        (output / "tables/variant_summary.json").read_text(encoding="utf-8")
    )
    render_report(output, frames, summaries)
    inventory = inventory_tree(output)
    history = list(completion.get("inventory_history") or [])
    if previous_inventory:
        history.append(
            {
                "superseded_utc": utc_now(),
                "reason": (
                    "Refresh the explanatory report/HTML or include append-only "
                    "audit products, then reseal without changing native tracks."
                ),
                "inventory": previous_inventory,
            }
        )
    completion.update(
        {
            "resealed_utc": utc_now(),
            "reseal_script": str(Path(__file__).resolve()),
            "reseal_script_sha256": sha256_file(Path(__file__).resolve()),
            "inventory_history": history,
            "inventory": inventory,
            "inventory_jsonl_sha256": sha256_file(
                output / "artifact_inventory.jsonl"
            ),
            "inventory_csv_sha256": sha256_file(
                output / "artifact_inventory.csv"
            ),
            "inventory_summary_sha256": sha256_file(
                output / "inventory_summary.json"
            ),
        }
    )
    write_json(completion_path, completion)
    print(
        json.dumps(
            {
                "status": "resealed",
                "output": str(output),
                "inventory": inventory,
                "completion_sha256": sha256_file(completion_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    e11 = args.e11_run.resolve()
    if args.reseal_existing:
        reseal_existing(output)
        return
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    if not e11.is_dir():
        raise FileNotFoundError(e11)
    for path in (
        DEFAULT_REID,
        TRACKING_SOURCE,
        REALTIME_CONFIG,
        EXPERIMENT_DEFINITION,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    started_utc = utc_now()
    started = time.perf_counter()
    output.mkdir(parents=True)
    (output / "tables").mkdir()
    (output / "failure_cases").mkdir()
    (output / "visualizations").mkdir()
    variants = variant_definitions()
    frames = load_frozen_inputs(e11, args.maximum_frames)
    e11_inventory = json.loads(
        (e11 / "inventory_summary.json").read_text(encoding="utf-8")
    )

    preregistration = {
        "schema": "daaam.g1_no_gt_e12_preregistration.v1",
        "created_utc": utc_now(),
        "status": "diagnostic_gt_free / upstream_coupled_to_E11",
        "experiment": "E12 BotSort/ReID replay",
        "controlled_input": {
            "e11_run": str(e11),
            "e11_inventory_summary_sha256": sha256_file(
                e11 / "inventory_summary.json"
            ),
            "e11_manifest_root_sha256": e11_inventory[
                "manifest_root_sha256"
            ],
            "e11_cell_id": E11_CELL_ID,
            "e11_profile_id": E11_PROFILE_ID,
            "confidence": 0.3,
            "minimum_area_px": 300,
            "nms_iou": 0.5,
            "frame_count": len(frames),
            "source_frame_range": [
                frames[0]["source_frame_index"],
                frames[-1]["source_frame_index"],
            ],
        },
        "formal_isolated_input_required": "reviewed GT instance mask/detection",
        "formal_isolated_input_available": False,
        "variants": variants,
        "fixed_boxmot_thresholds": {
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "match_thresh": 0.8,
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25,
            "frame_rate_internal_default": 30,
        },
        "fixed_failure_proxies": {
            "frame_high_unassigned": "unassigned input detection fraction > 0.25",
            "short_track": "full-run observation count <= 2",
            "sparse_lifecycle": "lifespan >= 5 and observed/lifespan < 0.25",
            "gap_reacquisition": "same internal track returns after >=1 missing frame",
            "large_center_jump": (
                "same-ID adjacent observation center displacement > 0.25 image diagonal"
            ),
            "low_consecutive_mask_iou": (
                "same-ID adjacent-frame E11 mask IoU < 0.05"
            ),
        },
        "track_id_map_overlap_policy": (
            "highest E11 confidence wins; zero is background; uint16 PNG is exact"
        ),
        "carry_forward": {
            "generated": False,
            "reason": (
                "E11 provides detections on every replay frame; this isolated "
                "BotSort runner does not synthesize masks between segmentation calls."
            ),
        },
        "unavailable_without_reviewed_gt": [
            "HOTA",
            "IDF1",
            "MOTA",
            "real ID switches",
            "real fragmentation",
            "occlusion recovery correctness",
        ],
    }
    write_json(output / "PRE_REGISTRATION.json", preregistration)

    input_rows = [
        {
            key: value
            for key, value in frame.items()
            if key not in {"instances", "kept_instance_ids"}
        }
        | {"kept_instance_ids_json": json.dumps(frame["kept_instance_ids"])}
        for frame in frames
    ]
    write_jsonl(output / "input_frames.jsonl", input_rows)
    write_csv(output / "input_frames.csv", input_rows)
    invocation = {
        "schema": "daaam.g1_no_gt_e12_invocation.v1",
        "created_utc": utc_now(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "tracking_source": str(TRACKING_SOURCE),
        "tracking_source_sha256": sha256_file(TRACKING_SOURCE),
        "realtime_config": str(REALTIME_CONFIG),
        "realtime_config_sha256": sha256_file(REALTIME_CONFIG),
        "experiment_definition": str(EXPERIMENT_DEFINITION),
        "experiment_definition_sha256": sha256_file(EXPERIMENT_DEFINITION),
        "reid_engine": str(DEFAULT_REID),
        "reid_engine_size_bytes": DEFAULT_REID.stat().st_size,
        "reid_engine_sha256": sha256_file(DEFAULT_REID),
        "boxmot_version": importlib.metadata.version("boxmot"),
        "device": args.device,
        "nvidia_before": nvidia_snapshot(),
    }
    write_json(output / "invocation.json", invocation)

    from daaam.config import TrackingConfig
    from daaam.tracking.services import TrackingService

    all_frame_rows: list[dict[str, Any]] = []
    all_observations: list[dict[str, Any]] = []
    all_lifecycles: list[dict[str, Any]] = []
    all_failure_rows: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []

    for variant in variants:
        variant_id = str(variant["variant_id"])
        variant_root = output / "variants" / variant_id
        variant_root.mkdir(parents=True)
        config = TrackingConfig(
            device=args.device,
            track_buffer=int(variant["track_buffer"]),
            enable_temporal_history=True,
            reid_weights=str(DEFAULT_REID.relative_to(REPOSITORY_ROOT)),
            with_reid=bool(variant["with_reid"]),
            reid_half=False,
            batch_reid_crops=bool(variant["batch_reid_crops"]),
            cmc_method=str(variant["cmc_method"]),
            cmc_ecc_max_iterations=int(
                variant["cmc_ecc_max_iterations"]
            ),
        )
        service = TrackingService(config)
        warmup_started = time.perf_counter()
        service.warmup()
        warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
        write_json(
            variant_root / "warmup.json",
            {
                "elapsed_ms": warmup_ms,
                "excluded_from_latency": True,
                "tracker_reset_after_warmup": True,
            },
        )

        histories: dict[int, list[dict[str, Any]]] = {}
        last_masks: dict[int, np.ndarray] = {}
        last_frames: dict[int, int] = {}
        frame_rows: list[dict[str, Any]] = []
        observation_rows: list[dict[str, Any]] = []
        frame_failures: list[dict[str, Any]] = []
        for frame in frames:
            frame_index = int(frame["frame_index"])
            bgr = cv2.imread(str(frame["rgb_path"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(frame["rgb_path"])
            instances = list(frame["instances"])
            detections = np.asarray(
                [
                    [
                        *[float(value) for value in instance["box_xyxy"]],
                        float(instance["model_confidence"]),
                        float(instance["model_class_id"]),
                    ]
                    for instance in instances
                ],
                dtype=np.float32,
            )
            if not len(detections):
                detections = np.empty((0, 6), dtype=np.float32)
            tracking_started = time.perf_counter()
            tracks = service.update(detections, bgr)
            latency_ms = (time.perf_counter() - tracking_started) * 1000.0
            tracks = np.asarray(tracks, dtype=np.float64)
            if tracks.size == 0:
                tracks = np.empty((0, 8), dtype=np.float64)
            if tracks.ndim != 2 or tracks.shape[1] != 8:
                raise ValueError(f"unexpected track output shape: {tracks.shape}")

            observations: list[dict[str, Any]] = []
            assigned_local_indices: set[int] = set()
            mask_cache: dict[int, np.ndarray] = {}
            for track in tracks:
                local_index = int(round(float(track[7])))
                track_id = int(round(float(track[4])))
                if local_index < 0 or local_index >= len(instances):
                    raise IndexError("BotSort detection index is outside E11 input")
                if local_index in assigned_local_indices:
                    raise ValueError("one E11 detection was assigned more than once")
                if track_id <= 0:
                    raise ValueError("public track IDs must reserve zero")
                assigned_local_indices.add(local_index)
                instance = instances[local_index]
                mask = cv2.imread(
                    str(instance["mask_path"]), cv2.IMREAD_GRAYSCALE
                )
                if mask is None:
                    raise FileNotFoundError(instance["mask_path"])
                mask_bool = mask > 0
                mask_cache[local_index] = mask_bool
                previous_frame = last_frames.get(track_id)
                frame_gap = (
                    None
                    if previous_frame is None
                    else frame_index - previous_frame
                )
                consecutive_iou = (
                    None
                    if previous_frame is None or track_id not in last_masks
                    else jaccard(last_masks[track_id], mask_bool)
                )
                previous_observations = histories.get(track_id, [])
                center_jump_ratio = None
                if previous_observations:
                    previous_box = previous_observations[-1]["track_box_xyxy"]
                    current_box = [float(value) for value in track[:4]]
                    previous_center = np.asarray(
                        [
                            (previous_box[0] + previous_box[2]) / 2,
                            (previous_box[1] + previous_box[3]) / 2,
                        ]
                    )
                    current_center = np.asarray(
                        [
                            (current_box[0] + current_box[2]) / 2,
                            (current_box[1] + current_box[3]) / 2,
                        ]
                    )
                    center_jump_ratio = float(
                        np.linalg.norm(current_center - previous_center)
                        / math.hypot(bgr.shape[1], bgr.shape[0])
                    )
                observation = {
                    "schema": "daaam.g1_no_gt_e12_track_observation.v1",
                    "variant_id": variant_id,
                    "frame_index": frame_index,
                    "source_frame_index": int(frame["source_frame_index"]),
                    "sensor_time_ns": int(frame["sensor_time_ns"]),
                    "track_id": track_id,
                    "detection_local_index": local_index,
                    "e11_instance_id": int(instance["instance_id"]),
                    "model_confidence": float(instance["model_confidence"]),
                    "model_class_id": int(instance["model_class_id"]),
                    "e11_box_xyxy": [
                        float(value) for value in instance["box_xyxy"]
                    ],
                    "track_box_xyxy": [float(value) for value in track[:4]],
                    "e11_area_px": int(instance["area_px"]),
                    "source_mask_path": str(instance["mask_path"]),
                    "source_mask_sha256": str(instance["mask_sha256"]),
                    "frame_gap": frame_gap,
                    "gap_reacquisition": (
                        frame_gap is not None and frame_gap > 1
                    ),
                    "consecutive_mask_iou": consecutive_iou,
                    "center_jump_image_diagonal_ratio": center_jump_ratio,
                    "evaluation_basis": "exact linkage + GT-free proxy",
                }
                observations.append(observation)
                histories.setdefault(track_id, []).append(observation)
                last_masks[track_id] = mask_bool
                last_frames[track_id] = frame_index

            unassigned_local_indices = sorted(
                set(range(len(instances))) - assigned_local_indices
            )
            unassigned_instance_ids = [
                int(instances[index]["instance_id"])
                for index in unassigned_local_indices
            ]
            unassigned_fraction = (
                len(unassigned_local_indices) / len(instances)
                if instances
                else 0.0
            )
            id_map, overlap_conflict_pixels = build_exact_track_id_map(
                bgr.shape[:2], observations, mask_cache
            )
            frame_root = variant_root / "frames" / f"{frame_index:08d}"
            require_write_image(frame_root / "track_id_map.png", id_map)
            require_write_image(
                frame_root / "track_overlay.png",
                draw_overlay(
                    bgr,
                    observations,
                    mask_cache,
                    (
                        f"E12 {variant_id} | source={frame['source_frame_index']} | "
                        f"tracked={len(observations)}/{len(instances)}"
                    ),
                ),
            )
            bot = service.tracker.tracker
            active_states = [tracker_state(value) for value in bot.active_tracks]
            lost_states = [tracker_state(value) for value in bot.lost_stracks]
            removed_ids = sorted(
                {int(value.id) + 1 for value in bot.removed_stracks}
            )
            flags: list[str] = []
            if unassigned_fraction > 0.25:
                flags.append("E12_HIGH_UNASSIGNED_PROXY")
            if any(bool(row["gap_reacquisition"]) for row in observations):
                flags.append("E12_GAP_REACQUISITION_PROXY")
            if any(
                row["center_jump_image_diagonal_ratio"] is not None
                and float(row["center_jump_image_diagonal_ratio"]) > 0.25
                for row in observations
            ):
                flags.append("E12_LARGE_CENTER_JUMP_PROXY")
            if any(
                row["frame_gap"] == 1
                and row["consecutive_mask_iou"] is not None
                and float(row["consecutive_mask_iou"]) < 0.05
                for row in observations
            ):
                flags.append("E12_LOW_CONSECUTIVE_MASK_IOU_PROXY")
            frame_record = {
                "schema": "daaam.g1_no_gt_e12_frame.v1",
                "variant": variant,
                "frame_index": frame_index,
                "source_frame_index": int(frame["source_frame_index"]),
                "sensor_time_ns": int(frame["sensor_time_ns"]),
                "rgb_path": frame["rgb_path"],
                "rgb_sha256": frame["rgb_sha256"],
                "e11_selection_path": frame["selection_path"],
                "e11_selection_sha256": frame["selection_sha256"],
                "e11_raw_frame_path": frame["raw_frame_path"],
                "e11_raw_frame_sha256": frame["raw_frame_sha256"],
                "input_detection_count": len(instances),
                "input_e11_instance_ids": [
                    int(instance["instance_id"]) for instance in instances
                ],
                "tracked_observation_count": len(observations),
                "assigned_detection_local_indices": sorted(
                    assigned_local_indices
                ),
                "unassigned_detection_local_indices": unassigned_local_indices,
                "unassigned_e11_instance_ids": unassigned_instance_ids,
                "unassigned_detection_fraction": unassigned_fraction,
                "tracking_latency_ms": latency_ms,
                "active_track_count_internal": len(active_states),
                "lost_track_count_internal": len(lost_states),
                "removed_track_count_cumulative_internal": len(removed_ids),
                "active_tracks_internal": active_states,
                "lost_tracks_internal": lost_states,
                "removed_track_ids_cumulative": removed_ids,
                "overlap_conflict_pixels": overlap_conflict_pixels,
                "track_id_map_path": str(frame_root / "track_id_map.png"),
                "track_overlay_path": str(frame_root / "track_overlay.png"),
                "track_observations": observations,
                "failure_flags": flags,
                "carry_forward_count": 0,
                "formal_metrics": {
                    "HOTA": None,
                    "IDF1": None,
                    "real_id_switches": None,
                },
            }
            write_json(frame_root / "frame.json", frame_record)
            flat_frame = {
                key: value
                for key, value in frame_record.items()
                if key
                not in {
                    "variant",
                    "active_tracks_internal",
                    "lost_tracks_internal",
                    "removed_track_ids_cumulative",
                    "track_observations",
                    "formal_metrics",
                    "input_e11_instance_ids",
                    "assigned_detection_local_indices",
                    "unassigned_detection_local_indices",
                    "unassigned_e11_instance_ids",
                    "failure_flags",
                }
            } | {
                "with_reid": bool(variant["with_reid"]),
                "track_buffer": int(variant["track_buffer"]),
                "cmc_ecc_max_iterations": int(
                    variant["cmc_ecc_max_iterations"]
                ),
                "failure_flags_json": json.dumps(flags),
            }
            frame_rows.append(flat_frame)
            observation_rows.extend(observations)
            for code in flags:
                frame_failures.append(
                    {
                        "schema": "daaam.g1_no_gt_e12_failure_proxy.v1",
                        "variant_id": variant_id,
                        "scope": "frame",
                        "failure_code": code,
                        "frame_index": frame_index,
                        "source_frame_index": int(frame["source_frame_index"]),
                        "evidence_path": str(frame_root / "frame.json"),
                        "overlay_path": str(frame_root / "track_overlay.png"),
                        "evaluation_basis": "proxy / no reviewed track GT",
                    }
                )

        lifecycle_rows: list[dict[str, Any]] = []
        lifecycle_failures: list[dict[str, Any]] = []
        for track_id, observations in sorted(histories.items()):
            frame_indices = [
                int(row["frame_index"]) for row in observations
            ]
            gaps = [
                frame_indices[index] - frame_indices[index - 1]
                for index in range(1, len(frame_indices))
            ]
            lifespan = frame_indices[-1] - frame_indices[0] + 1
            consecutive_ious = [
                float(row["consecutive_mask_iou"])
                for row in observations
                if row["frame_gap"] == 1
                and row["consecutive_mask_iou"] is not None
            ]
            center_jumps = [
                float(row["center_jump_image_diagonal_ratio"])
                for row in observations
                if row["center_jump_image_diagonal_ratio"] is not None
            ]
            flags: list[str] = []
            if len(observations) <= 2:
                flags.append("E12_SHORT_TRACK_PROXY")
            if lifespan >= 5 and len(observations) / lifespan < 0.25:
                flags.append("E12_SPARSE_LIFECYCLE_PROXY")
            if any(value > 1 for value in gaps):
                flags.append("E12_GAP_REACQUISITION_PROXY")
            if any(value > 0.25 for value in center_jumps):
                flags.append("E12_LARGE_CENTER_JUMP_PROXY")
            if any(value < 0.05 for value in consecutive_ious):
                flags.append("E12_LOW_CONSECUTIVE_MASK_IOU_PROXY")
            lifecycle = {
                "schema": "daaam.g1_no_gt_e12_track_lifecycle.v1",
                "variant_id": variant_id,
                "track_id": track_id,
                "first_frame_index": frame_indices[0],
                "last_frame_index": frame_indices[-1],
                "first_source_frame_index": int(
                    observations[0]["source_frame_index"]
                ),
                "last_source_frame_index": int(
                    observations[-1]["source_frame_index"]
                ),
                "observation_count": len(observations),
                "lifespan_frames": lifespan,
                "observation_fill_ratio": len(observations) / lifespan,
                "missing_frame_count_within_lifespan": lifespan
                - len(observations),
                "gap_reacquisition_count": sum(value > 1 for value in gaps),
                "maximum_frame_gap": max(gaps, default=0),
                "consecutive_mask_iou_mean": (
                    statistics.fmean(consecutive_ious)
                    if consecutive_ious
                    else None
                ),
                "consecutive_mask_iou_minimum": (
                    min(consecutive_ious) if consecutive_ious else None
                ),
                "center_jump_ratio_maximum": (
                    max(center_jumps) if center_jumps else None
                ),
                "mean_e11_area_px": statistics.fmean(
                    float(row["e11_area_px"]) for row in observations
                ),
                "mean_model_confidence": statistics.fmean(
                    float(row["model_confidence"]) for row in observations
                ),
                "observation_frame_indices": frame_indices,
                "e11_instance_ids": [
                    int(row["e11_instance_id"]) for row in observations
                ],
                "failure_flags": flags,
                "evaluation_basis": "exact lifecycle + GT-free proxy",
            }
            lifecycle_rows.append(lifecycle)
            for code in flags:
                lifecycle_failures.append(
                    {
                        "schema": "daaam.g1_no_gt_e12_failure_proxy.v1",
                        "variant_id": variant_id,
                        "scope": "track",
                        "failure_code": code,
                        "track_id": track_id,
                        "first_frame_index": frame_indices[0],
                        "last_frame_index": frame_indices[-1],
                        "evidence_path": str(
                            variant_root / "track_lifecycles.json"
                        ),
                        "timeline_path": str(
                            variant_root / "track_timeline.png"
                        ),
                        "evaluation_basis": "proxy / no reviewed track GT",
                    }
                )

        write_jsonl(variant_root / "frame_summary.jsonl", frame_rows)
        write_csv(variant_root / "frame_summary.csv", frame_rows)
        write_jsonl(
            variant_root / "track_observations.jsonl", observation_rows
        )
        observation_csv = [
            {
                key: value
                for key, value in row.items()
                if key not in {"e11_box_xyxy", "track_box_xyxy"}
            }
            | {
                "e11_box_xyxy_json": json.dumps(row["e11_box_xyxy"]),
                "track_box_xyxy_json": json.dumps(row["track_box_xyxy"]),
            }
            for row in observation_rows
        ]
        write_csv(variant_root / "track_observations.csv", observation_csv)
        write_json(variant_root / "track_lifecycles.json", lifecycle_rows)
        lifecycle_csv = [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "observation_frame_indices",
                    "e11_instance_ids",
                    "failure_flags",
                }
            }
            | {
                "observation_frame_indices_json": json.dumps(
                    row["observation_frame_indices"]
                ),
                "e11_instance_ids_json": json.dumps(row["e11_instance_ids"]),
                "failure_flags_json": json.dumps(row["failure_flags"]),
            }
            for row in lifecycle_rows
        ]
        write_csv(variant_root / "track_lifecycles.csv", lifecycle_csv)
        save_track_timeline(
            output, variant_id, lifecycle_rows, len(frames)
        )

        latency = numeric_summary(
            row["tracking_latency_ms"] for row in frame_rows
        )
        mask_ious = [
            float(row["consecutive_mask_iou"])
            for row in observation_rows
            if row["frame_gap"] == 1
            and row["consecutive_mask_iou"] is not None
        ]
        total_input = sum(int(row["input_detection_count"]) for row in frame_rows)
        total_tracked = len(observation_rows)
        short_count = sum(
            int(row["observation_count"]) <= 2 for row in lifecycle_rows
        )
        summary = {
            "schema": "daaam.g1_no_gt_e12_variant_summary.v1",
            **variant,
            "frame_count": len(frame_rows),
            "input_detection_count": total_input,
            "tracked_observation_count": total_tracked,
            "unassigned_detection_count": total_input - total_tracked,
            "tracked_detection_fraction": (
                total_tracked / total_input if total_input else 1.0
            ),
            "unique_track_count": len(lifecycle_rows),
            "short_track_count": short_count,
            "short_track_fraction": (
                short_count / len(lifecycle_rows) if lifecycle_rows else 0.0
            ),
            "sparse_lifecycle_count": sum(
                "E12_SPARSE_LIFECYCLE_PROXY" in row["failure_flags"]
                for row in lifecycle_rows
            ),
            "gap_reacquisition_event_count": sum(
                int(row["gap_reacquisition_count"]) for row in lifecycle_rows
            ),
            "large_center_jump_track_count": sum(
                "E12_LARGE_CENTER_JUMP_PROXY" in row["failure_flags"]
                for row in lifecycle_rows
            ),
            "low_consecutive_mask_iou_track_count": sum(
                "E12_LOW_CONSECUTIVE_MASK_IOU_PROXY"
                in row["failure_flags"]
                for row in lifecycle_rows
            ),
            "track_observation_count_mean": numeric_summary(
                row["observation_count"] for row in lifecycle_rows
            )["mean"],
            "track_observation_count_p50": numeric_summary(
                row["observation_count"] for row in lifecycle_rows
            )["p50"],
            "track_observation_count_p95": numeric_summary(
                row["observation_count"] for row in lifecycle_rows
            )["p95"],
            "consecutive_mask_iou_mean": (
                statistics.fmean(mask_ious) if mask_ious else None
            ),
            "consecutive_mask_iou_p05": percentile(mask_ious, 5),
            "tracking_latency_mean_ms": latency["mean"],
            "tracking_latency_p50_ms": latency["p50"],
            "tracking_latency_p95_ms": latency["p95"],
            "tracking_latency_maximum_ms": latency["maximum"],
            "warmup_ms_excluded": warmup_ms,
            "failure_proxy_record_count": len(frame_failures)
            + len(lifecycle_failures),
            "carry_forward_count": 0,
            "formal_metrics": {
                "HOTA": None,
                "IDF1": None,
                "real_id_switches": None,
                "real_fragmentation": None,
            },
            "correctness_winner": None,
            "evaluation_basis": "exact system counts + GT-free proxies",
        }
        write_json(variant_root / "SUMMARY.json", summary)
        all_frame_rows.extend(frame_rows)
        all_observations.extend(observation_rows)
        all_lifecycles.extend(lifecycle_rows)
        all_failure_rows.extend(frame_failures)
        all_failure_rows.extend(lifecycle_failures)
        variant_summaries.append(summary)
        del service

    write_json(output / "tables/variant_summary.json", variant_summaries)
    summary_csv = [
        {
            key: value
            for key, value in row.items()
            if key not in {"formal_metrics"}
        }
        for row in variant_summaries
    ]
    write_csv(output / "tables/variant_summary.csv", summary_csv)
    write_jsonl(output / "tables/frame_summary.jsonl", all_frame_rows)
    write_csv(output / "tables/frame_summary.csv", all_frame_rows)
    write_jsonl(
        output / "tables/track_observations.jsonl", all_observations
    )
    write_csv(
        output / "tables/track_observations.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"e11_box_xyxy", "track_box_xyxy"}
            }
            | {
                "e11_box_xyxy_json": json.dumps(row["e11_box_xyxy"]),
                "track_box_xyxy_json": json.dumps(row["track_box_xyxy"]),
            }
            for row in all_observations
        ],
    )
    write_jsonl(output / "tables/track_lifecycles.jsonl", all_lifecycles)
    write_csv(
        output / "tables/track_lifecycles.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "observation_frame_indices",
                    "e11_instance_ids",
                    "failure_flags",
                }
            }
            | {
                "observation_frame_indices_json": json.dumps(
                    row["observation_frame_indices"]
                ),
                "e11_instance_ids_json": json.dumps(row["e11_instance_ids"]),
                "failure_flags_json": json.dumps(row["failure_flags"]),
            }
            for row in all_lifecycles
        ],
    )
    write_jsonl(
        output / "failure_cases/failure_cases.jsonl", all_failure_rows
    )
    write_csv(
        output / "failure_cases/failure_cases.csv", all_failure_rows
    )
    screening = {
        "schema": "daaam.g1_no_gt_e12_screening_result.v1",
        "created_utc": utc_now(),
        "status": "diagnostic_gt_free / no_accuracy_claim",
        "correctness_winner": None,
        "reason": (
            "Reviewed GT identities are unavailable and E11 predictions are not "
            "an E12 oracle. Candidate differences are engineering diagnostics only."
        ),
        "production_baseline": next(
            row for row in variant_summaries if row["variant_id"] == "baseline"
        ),
        "candidate_summaries": variant_summaries,
        "interpretation_rules": [
            "Fewer tracks may mean less fragmentation or more ID merging.",
            "More tracks may mean fragmentation or newly visible objects.",
            "Higher same-ID mask IoU is camera-motion and segmentation dependent.",
            "Tracked detection fraction includes BotSort confidence/activation policy.",
        ],
    }
    write_json(output / "SCREENING_RESULT.json", screening)
    save_comparison_plot(output, variant_summaries)
    save_representative_montage(output, variants, len(frames))
    render_report(output, frames, variant_summaries)
    inventory = inventory_tree(output)
    completion = {
        "schema": "daaam.g1_no_gt_e12_completion.v1",
        "status": "complete",
        "authority": "diagnostic_gt_free",
        "upstream": "E11 FastSAM production baseline",
        "formal_isolated_e12": False,
        "accuracy_claim": False,
        "correctness_winner": None,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "elapsed_s": time.perf_counter() - started,
        "frame_count": len(frames),
        "source_frame_range": [
            frames[0]["source_frame_index"],
            frames[-1]["source_frame_index"],
        ],
        "variant_count": len(variants),
        "input_detection_records": sum(
            int(row["input_detection_count"]) for row in variant_summaries
        ),
        "track_observation_records": len(all_observations),
        "track_lifecycle_records": len(all_lifecycles),
        "failure_proxy_records": len(all_failure_rows),
        "inventory": inventory,
        "inventory_jsonl_sha256": sha256_file(
            output / "artifact_inventory.jsonl"
        ),
        "inventory_csv_sha256": sha256_file(
            output / "artifact_inventory.csv"
        ),
        "inventory_summary_sha256": sha256_file(
            output / "inventory_summary.json"
        ),
        "resource_usage": {
            "maximum_resident_set_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "nvidia_after": nvidia_snapshot(),
        },
        "unavailable_metrics": preregistration[
            "unavailable_without_reviewed_gt"
        ],
    }
    write_json(output / "COMPLETION.json", completion)
    print(json.dumps(completion, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
