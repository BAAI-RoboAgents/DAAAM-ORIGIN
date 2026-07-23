#!/usr/bin/env python3
"""Compare FoundationStereo and Fast-FoundationStereo depth products.

The capture has no dense depth ground truth.  This report therefore separates
dense inter-model agreement from a sparse, time-skewed LiDAR proxy evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation", required=True, type=Path)
    parser.add_argument("--fast", required=True, type=Path)
    parser.add_argument("--fast-smoke-run", type=Path)
    parser.add_argument("--raw-dataset", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    parser.add_argument("--preview-out", type=Path)
    parser.add_argument("--lidar-step", type=int, default=10)
    parser.add_argument("--minimum-depth-m", type=float, default=0.25)
    parser.add_argument("--maximum-depth-m", type=float, default=5.0)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def summarize(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def dense_agreement(
    foundation: Path,
    fast: Path,
    frames: list[dict[str, Any]],
    shared_maximum_depth_m: float,
) -> dict[str, Any]:
    pixels = foundation_valid = fast_valid = fast_valid_shared = overlap = union = 0
    sum_abs_m = sum_squared_m2 = sum_absrel = sum_signed_m = 0.0
    delta_counts = {"delta_1.05": 0, "delta_1.10": 0, "delta_1.25": 0}
    per_frame_mae_m: list[float] = []
    per_frame_absrel: list[float] = []
    per_frame_overlap_ratio: list[float] = []
    for position, frame in enumerate(frames, start=1):
        name = f"{int(frame['idx']):08d}.png"
        foundation_mm = cv2.imread(
            str(foundation / "depth" / name), cv2.IMREAD_UNCHANGED
        )
        fast_mm = cv2.imread(str(fast / "depth" / name), cv2.IMREAD_UNCHANGED)
        if foundation_mm is None or fast_mm is None:
            raise FileNotFoundError(f"missing depth product for {name}")
        if foundation_mm.shape != fast_mm.shape or foundation_mm.dtype != np.uint16 or fast_mm.dtype != np.uint16:
            raise ValueError(f"incompatible depth products for {name}")
        foundation_mask = foundation_mm > 0
        fast_mask = fast_mm > 0
        fast_shared_mask = fast_mask & (
            fast_mm <= round(shared_maximum_depth_m * 1000.0)
        )
        common = foundation_mask & fast_shared_mask
        either = foundation_mask | fast_shared_mask
        count = int(common.sum())
        pixels += foundation_mm.size
        foundation_valid += int(foundation_mask.sum())
        fast_valid += int(fast_mask.sum())
        fast_valid_shared += int(fast_shared_mask.sum())
        overlap += count
        union += int(either.sum())
        if count:
            foundation_values = foundation_mm[common].astype(np.float64)
            fast_values = fast_mm[common].astype(np.float64)
            signed_m = (fast_values - foundation_values) / 1000.0
            absolute_m = np.abs(signed_m)
            relative = np.abs(fast_values - foundation_values) / foundation_values
            minimum = np.minimum(foundation_values, fast_values)
            maximum = np.maximum(foundation_values, fast_values)
            sum_abs_m += float(absolute_m.sum())
            sum_squared_m2 += float(np.square(signed_m).sum())
            sum_absrel += float(relative.sum())
            sum_signed_m += float(signed_m.sum())
            for threshold, key in (
                (1.05, "delta_1.05"),
                (1.10, "delta_1.10"),
                (1.25, "delta_1.25"),
            ):
                delta_counts[key] += int((maximum < threshold * minimum).sum())
            per_frame_mae_m.append(float(absolute_m.mean()))
            per_frame_absrel.append(float(relative.mean()))
            per_frame_overlap_ratio.append(count / foundation_mm.size)
        if position % 100 == 0:
            print(f"dense agreement {position}/{len(frames)}", flush=True)
    if not overlap:
        raise ValueError("the depth products have no overlapping valid pixels")
    return {
        "interpretation": (
            "Fast-FoundationStereo difference with FoundationStereo as a pseudo-reference; "
            "not absolute accuracy. Error metrics and overlap use the shared depth cap."
        ),
        "shared_maximum_depth_m": shared_maximum_depth_m,
        "frames": len(frames),
        "pixels": pixels,
        "foundation_valid_pixels": foundation_valid,
        "fast_valid_pixels": fast_valid,
        "fast_valid_pixels_within_shared_cap": fast_valid_shared,
        "fast_extra_pixels_beyond_shared_cap": fast_valid - fast_valid_shared,
        "overlap_valid_pixels": overlap,
        "union_valid_pixels": union,
        "foundation_valid_ratio": foundation_valid / pixels,
        "fast_valid_ratio": fast_valid / pixels,
        "fast_valid_ratio_within_shared_cap": fast_valid_shared / pixels,
        "fast_extra_ratio_beyond_shared_cap": (fast_valid - fast_valid_shared)
        / pixels,
        "overlap_ratio": overlap / pixels,
        "intersection_over_union": overlap / union,
        "fast_coverage_on_foundation_valid": overlap / foundation_valid,
        "foundation_coverage_on_fast_valid_within_shared_cap": overlap
        / fast_valid_shared,
        "depth_mae_m": sum_abs_m / overlap,
        "depth_rmse_m": math.sqrt(sum_squared_m2 / overlap),
        "depth_absrel": sum_absrel / overlap,
        "mean_signed_fast_minus_foundation_m": sum_signed_m / overlap,
        **{key: value / overlap for key, value in delta_counts.items()},
        "per_frame_mae_m": summarize(per_frame_mae_m),
        "per_frame_absrel": summarize(per_frame_absrel),
        "per_frame_overlap_ratio": summarize(per_frame_overlap_ratio),
    }


def transform_from_pose(position: list[float], quaternion_xyzw: list[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    transform[:3, 3] = position
    return transform


def odom_transform(record: dict[str, Any]) -> np.ndarray:
    pose = record["odom"]["pose"]["pose"]
    position = pose["position"]
    orientation = pose["orientation"]
    return transform_from_pose(
        [position["x"], position["y"], position["z"]],
        [orientation["x"], orientation["y"], orientation["z"], orientation["w"]],
    )


class ErrorAccumulator:
    def __init__(self) -> None:
        self.reference = 0
        self.valid = 0
        self.absolute_errors: list[np.ndarray] = []
        self.relative_errors: list[np.ndarray] = []
        self.ratios: list[np.ndarray] = []
        self.signed_errors: list[np.ndarray] = []

    def add(self, prediction: np.ndarray, reference: np.ndarray) -> None:
        self.reference += int(reference.size)
        valid = np.isfinite(prediction) & (prediction > 0.0)
        self.valid += int(valid.sum())
        if not np.any(valid):
            return
        predicted = prediction[valid].astype(np.float64)
        target = reference[valid].astype(np.float64)
        signed = predicted - target
        self.absolute_errors.append(np.abs(signed))
        self.relative_errors.append(np.abs(signed) / target)
        self.ratios.append(np.maximum(predicted / target, target / predicted))
        self.signed_errors.append(signed)

    def report(self) -> dict[str, Any]:
        if not self.absolute_errors:
            return {"reference_points": self.reference, "valid_predictions": self.valid}
        absolute = np.concatenate(self.absolute_errors)
        relative = np.concatenate(self.relative_errors)
        ratios = np.concatenate(self.ratios)
        signed = np.concatenate(self.signed_errors)
        return {
            "reference_points": self.reference,
            "valid_predictions": self.valid,
            "prediction_coverage": self.valid / self.reference if self.reference else None,
            "depth_mae_m": float(absolute.mean()),
            "depth_rmse_m": float(np.sqrt(np.square(signed).mean())),
            "depth_median_absolute_error_m": float(np.median(absolute)),
            "depth_absrel": float(relative.mean()),
            "mean_signed_prediction_minus_lidar_m": float(signed.mean()),
            "delta_1.05": float((ratios < 1.05).mean()),
            "delta_1.10": float((ratios < 1.10).mean()),
            "delta_1.25": float((ratios < 1.25).mean()),
        }


def lidar_projection(
    foundation: Path,
    fast: Path,
    raw: Path,
    tick_index: dict[str, Any],
    *,
    frame_step: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> dict[str, Any]:
    if frame_step <= 0:
        raise ValueError("lidar-step must be positive")
    manifest_records = load_jsonl(raw / "manifest.jsonl")
    odom_records = load_jsonl(raw / "state" / "000000" / "odom.jsonl")
    camera_poses = np.loadtxt(foundation / "pose" / "poses.txt").reshape(-1, 4, 4)
    frames = tick_index["frames"][::frame_step]
    width = int(tick_index["width"])
    height = int(tick_index["height"])
    fx = float(tick_index["fx"])
    fy = float(tick_index["fy"])
    cx = float(tick_index["cx"])
    cy = float(tick_index["cy"])
    all_points = {
        "foundation": ErrorAccumulator(),
        "fast": ErrorAccumulator(),
    }
    paired_points = {
        "foundation": ErrorAccumulator(),
        "fast": ErrorAccumulator(),
    }
    projected_points = 0
    unique_pixels = 0
    camera_lidar_skew_ms: list[float] = []
    per_frame: list[dict[str, Any]] = []
    for position, frame in enumerate(frames, start=1):
        source_index = int(frame["source_idx"])
        record = manifest_records[source_index]
        odom = odom_records[source_index]
        lidar_descriptor = record["lidar"][0]
        lidar_pose = record["poses"]["values"]["lidar"]
        odom_T_base = odom_transform(odom)
        base_T_lidar = transform_from_pose(
            lidar_pose["position"], lidar_pose["orientation_xyzw"]
        )
        odom_T_camera = camera_poses[int(frame["pose_row"])]
        camera_T_lidar = np.linalg.inv(odom_T_camera) @ odom_T_base @ base_T_lidar
        points = np.load(raw / lidar_descriptor["path"], allow_pickle=False)
        valid_source = np.isfinite(points).all(axis=1) & (
            np.linalg.norm(points, axis=1) > 0.01
        )
        points = points[valid_source]
        camera_points = (
            points @ camera_T_lidar[:3, :3].T + camera_T_lidar[:3, 3]
        )
        z = camera_points[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * camera_points[:, 0] / z + cx
            v = fy * camera_points[:, 1] / z + cy
        inside = (
            (z >= minimum_depth_m)
            & (z <= maximum_depth_m)
            & (u >= 0.0)
            & (u <= width - 1)
            & (v >= 0.0)
            & (v <= height - 1)
        )
        projected_points += int(inside.sum())
        u_pixel = np.clip(np.rint(u[inside]).astype(np.int64), 0, width - 1)
        v_pixel = np.clip(np.rint(v[inside]).astype(np.int64), 0, height - 1)
        reference = z[inside]
        flat = v_pixel * width + u_pixel
        depth_order = np.argsort(reference)
        _, first = np.unique(flat[depth_order], return_index=True)
        selected = depth_order[first]
        u_pixel = u_pixel[selected]
        v_pixel = v_pixel[selected]
        reference = reference[selected]
        unique_pixels += int(reference.size)

        name = f"{int(frame['idx']):08d}.png"
        foundation_depth = cv2.imread(
            str(foundation / "depth" / name), cv2.IMREAD_UNCHANGED
        ).astype(np.float32) / 1000.0
        fast_depth = cv2.imread(
            str(fast / "depth" / name), cv2.IMREAD_UNCHANGED
        ).astype(np.float32) / 1000.0
        foundation_prediction = foundation_depth[v_pixel, u_pixel]
        fast_prediction = fast_depth[v_pixel, u_pixel]
        all_points["foundation"].add(foundation_prediction, reference)
        all_points["fast"].add(fast_prediction, reference)
        paired = (foundation_prediction > 0.0) & (fast_prediction > 0.0)
        paired_reference = reference[paired]
        paired_points["foundation"].add(
            foundation_prediction[paired], paired_reference
        )
        paired_points["fast"].add(fast_prediction[paired], paired_reference)
        skew = record.get("sync", {}).get("relative_skew_ms", {}).get(
            "camera:HEAD_LEFT_CAMERA"
        )
        if skew is not None:
            camera_lidar_skew_ms.append(float(skew))
        per_frame.append(
            {
                "frame_idx": int(frame["idx"]),
                "source_idx": source_index,
                "projected_points": int(inside.sum()),
                "unique_projected_pixels": int(reference.size),
                "paired_valid_predictions": int(paired.sum()),
                "camera_minus_lidar_time_ms": float(skew) if skew is not None else None,
            }
        )
        if position % 25 == 0 or position == len(frames):
            print(f"lidar proxy {position}/{len(frames)}", flush=True)
    return {
        "interpretation": (
            "Sparse proxy only, not dense ground truth: the LiDAR and camera have "
            "different viewpoints and up to tens of milliseconds of capture skew."
        ),
        "sampling": {
            "frame_step": frame_step,
            "frames_evaluated": len(frames),
            "minimum_depth_m": minimum_depth_m,
            "maximum_depth_m": maximum_depth_m,
            "pixel_sampling": "nearest projected pixel with per-pixel nearest-LiDAR z-buffer",
        },
        "projected_points_before_z_buffer": projected_points,
        "unique_projected_pixels": unique_pixels,
        "camera_minus_lidar_time_ms": summarize(camera_lidar_skew_ms),
        "all_available_prediction_points": {
            name: accumulator.report() for name, accumulator in all_points.items()
        },
        "paired_valid_points": {
            name: accumulator.report() for name, accumulator in paired_points.items()
        },
        "frames": per_frame,
    }


def timing_comparison(
    foundation_run: dict[str, Any],
    fast_run: dict[str, Any],
    fast_smoke_run: dict[str, Any] | None,
) -> dict[str, Any]:
    foundation_model_per_frame = float(
        foundation_run["inference_seconds_per_processed_frame"]
    )
    fast_dual = fast_run["timing"]["dual_model_wall"]
    fast_model_per_frame = float(fast_dual["mean_seconds"])
    foundation_elapsed = float(foundation_run["elapsed_seconds"])
    fast_elapsed = float(fast_run["elapsed_seconds"])
    foundation_checkpoint = Path(foundation_run["checkpoint"])
    preflight_gpu_lines = fast_run.get("gpu_snapshots", {}).get(
        "preflight_before_model_load", {}
    ).get("gpu", {}).get("stdout_lines", [])
    preflight_compute = fast_run.get("gpu_snapshots", {}).get(
        "preflight_before_model_load", {}
    ).get("compute_processes", {}).get("stdout_lines", [])
    result = {
        "comparison_scope": "full-resolution, FP16, left-right confidence (two model calls per output frame)",
        "foundation": {
            "iterations": int(foundation_run["valid_iters"]),
            "maximum_depth_m": float(foundation_run["maximum_depth_m"]),
            "frames": int(foundation_run["processed"]),
            "model_seconds_per_frame_mean": foundation_model_per_frame,
            "model_frames_per_second": 1.0 / foundation_model_per_frame,
            "model_seconds_total": float(foundation_run["inference_seconds"]),
            "end_to_end_seconds_total": foundation_elapsed,
            "peak_cuda_memory_bytes": int(foundation_run["peak_cuda_memory_bytes"]),
            "checkpoint_size_bytes": foundation_checkpoint.stat().st_size,
        },
        "fast": {
            "iterations": int(fast_run["settings"]["iterations"]),
            "maximum_depth_m": float(fast_run["settings"]["maximum_depth_m"]),
            "frames": int(fast_run["processed"]),
            "model_seconds_per_frame_mean": fast_model_per_frame,
            "model_seconds_per_frame_p50": float(fast_dual["p50_seconds"]),
            "model_seconds_per_frame_p95": float(fast_dual["p95_seconds"]),
            "model_frames_per_second": 1.0 / fast_model_per_frame,
            "model_seconds_total": fast_model_per_frame * int(fast_dual["count"]),
            "end_to_end_seconds_total": fast_elapsed,
            "warmup_seconds_excluded": float(fast_run["warmup_seconds"]),
            "peak_cuda_memory_bytes": int(fast_run["peak_cuda_memory_bytes"]),
            "parameter_count": int(fast_run["model"]["parameter_count"]),
            "checkpoint_size_bytes": int(
                fast_run["artifacts"]["checkpoint_size_bytes"]
            ),
        },
        "speedup": {
            "contended_full_run_model_mean": foundation_model_per_frame
            / fast_model_per_frame,
            "contended_full_run_end_to_end_total": foundation_elapsed
            / fast_elapsed,
        },
        "contention": {
            "fast_full_run_contaminated": True,
            "preflight_gpu": preflight_gpu_lines,
            "preflight_compute_processes": preflight_compute,
            "reason": (
                "Fast full-run preflight recorded 100% GPU utilization and an external "
                "FoundationStereo compute process."
            ),
        },
        "caveat": (
            "The methods use their intended quality settings (Foundation 32 iterations, "
            "Fast 8 iterations). Foundation output is capped at 3 m while Fast is capped "
            "at 5 m, so timing remains comparable but coverage is not directly comparable "
            "outside the shared 0-3 m range."
        ),
    }
    if fast_smoke_run is not None:
        smoke_dual = fast_smoke_run["timing"]["dual_model_wall"]
        smoke_seconds = float(smoke_dual["mean_seconds"])
        result["fast"]["exploratory_warmed_single_frame_seconds"] = smoke_seconds
        result["speedup"]["exploratory_warmed_single_frame_model"] = (
            foundation_model_per_frame / smoke_seconds
        )
        result["contention"]["smoke_caveat"] = (
            "The earlier smoke measurement contains one warmed full-resolution frame; "
            "it is useful context, not a report-grade repeated benchmark."
        )
    return result


def depth_color(depth_m: np.ndarray, minimum_m: float, maximum_m: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (depth_m[valid] - minimum_m) * 255.0 / (maximum_m - minimum_m),
        0,
        255,
    ).astype(np.uint8)
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def label_image(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def write_preview(
    foundation: Path,
    fast: Path,
    frames: list[dict[str, Any]],
    output: Path,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> None:
    indices = np.linspace(0, len(frames) - 1, num=min(4, len(frames)), dtype=int)
    rows: list[np.ndarray] = []
    target_width = 420
    for index in indices:
        frame = frames[int(index)]
        name = f"{int(frame['idx']):08d}.png"
        rgb = cv2.imread(str(foundation / "rgb" / name), cv2.IMREAD_COLOR)
        foundation_depth = cv2.imread(
            str(foundation / "depth" / name), cv2.IMREAD_UNCHANGED
        ).astype(np.float32) / 1000.0
        fast_depth = cv2.imread(
            str(fast / "depth" / name), cv2.IMREAD_UNCHANGED
        ).astype(np.float32) / 1000.0
        common = (foundation_depth > 0.0) & (fast_depth > 0.0)
        difference = np.zeros(foundation_depth.shape, dtype=np.float32)
        difference[common] = np.abs(fast_depth[common] - foundation_depth[common])
        difference_color = depth_color(difference, 0.0, 0.30)
        tiles = [
            label_image(rgb, f"frame {frame['idx']} RGB"),
            label_image(
                depth_color(foundation_depth, minimum_depth_m, maximum_depth_m),
                "FoundationStereo depth",
            ),
            label_image(
                depth_color(fast_depth, minimum_depth_m, maximum_depth_m),
                "Fast-FoundationStereo depth",
            ),
            label_image(difference_color, "absolute difference (0-0.30 m)"),
        ]
        resized = [
            cv2.resize(
                tile,
                (target_width, round(tile.shape[0] * target_width / tile.shape[1])),
                interpolation=cv2.INTER_AREA,
            )
            for tile in tiles
        ]
        rows.append(np.concatenate(resized, axis=1))
    preview = np.concatenate(rows, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), preview):
        raise RuntimeError(f"failed to write preview: {output}")


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    timing = report["timing"]
    agreement = report["dense_inter_model_agreement"]
    lidar = report["sparse_lidar_proxy"]
    paired = lidar["paired_valid_points"]
    foundation_lidar = paired["foundation"]
    fast_lidar = paired["fast"]
    return f"""# FoundationStereo 与 Fast-FoundationStereo 对比

## 结论

- Fast 在受 GPU 争用的 1575 帧完整运行中，模型推理仍加速 **{timing['speedup']['contended_full_run_model_mean']:.2f}×**，端到端加速 **{timing['speedup']['contended_full_run_end_to_end_total']:.2f}×**；此前暖机后的单帧探索性计时对应 **{timing['speedup'].get('exploratory_warmed_single_frame_model', float('nan')):.2f}×**。
- Fast 的 0–5 m 有效覆盖率为 **{percent(agreement['fast_valid_ratio'])}**；限制到与 Foundation 相同的 0–3 m 后为 **{percent(agreement['fast_valid_ratio_within_shared_cap'])}**，FoundationStereo 为 **{percent(agreement['foundation_valid_ratio'])}**。
- 两者共同有效区域内，Fast 相对 FoundationStereo 的深度 MAE 为 **{agreement['depth_mae_m']:.4f} m**、AbsRel 为 **{percent(agreement['depth_absrel'])}**，δ1.05 为 **{percent(agreement['delta_1.05'])}**。
- 在两种方法均有预测的同一批稀疏 LiDAR 投影点上，FoundationStereo / Fast 的 MAE 分别为 **{foundation_lidar['depth_mae_m']:.4f} m / {fast_lidar['depth_mae_m']:.4f} m**，δ1.25 分别为 **{percent(foundation_lidar['delta_1.25'])} / {percent(fast_lidar['delta_1.25'])}**。

## 耗时与资源

| 指标 | FoundationStereo | Fast-FoundationStereo |
|---|---:|---:|
| 迭代数 | {timing['foundation']['iterations']} | {timing['fast']['iterations']} |
| 最大输出深度 | {timing['foundation']['maximum_depth_m']:.1f} m | {timing['fast']['maximum_depth_m']:.1f} m |
| 双向模型耗时/帧 | {timing['foundation']['model_seconds_per_frame_mean']:.4f} s | {timing['fast']['model_seconds_per_frame_mean']:.4f} s |
| 模型吞吐 | {timing['foundation']['model_frames_per_second']:.3f} 帧/s | {timing['fast']['model_frames_per_second']:.3f} 帧/s |
| 全序列端到端耗时 | {timing['foundation']['end_to_end_seconds_total']:.1f} s | {timing['fast']['end_to_end_seconds_total']:.1f} s |
| 峰值 PyTorch CUDA 显存 | {timing['foundation']['peak_cuda_memory_bytes'] / 2**30:.2f} GiB | {timing['fast']['peak_cuda_memory_bytes'] / 2**30:.2f} GiB |
| checkpoint 文件大小 | {timing['foundation']['checkpoint_size_bytes'] / 2**20:.1f} MiB | {timing['fast']['checkpoint_size_bytes'] / 2**20:.1f} MiB |

## 稠密结果一致性

| 指标 | 结果 |
|---|---:|
| Foundation 有效覆盖率 | {percent(agreement['foundation_valid_ratio'])} |
| Fast 有效覆盖率（0–5 m） | {percent(agreement['fast_valid_ratio'])} |
| Fast 有效覆盖率（共同 0–3 m） | {percent(agreement['fast_valid_ratio_within_shared_cap'])} |
| Fast 额外 3–5 m 覆盖率 | {percent(agreement['fast_extra_ratio_beyond_shared_cap'])} |
| 共同有效覆盖率 | {percent(agreement['overlap_ratio'])} |
| 深度 MAE | {agreement['depth_mae_m']:.4f} m |
| 深度 RMSE | {agreement['depth_rmse_m']:.4f} m |
| AbsRel | {percent(agreement['depth_absrel'])} |
| δ1.05 / δ1.10 / δ1.25 | {percent(agreement['delta_1.05'])} / {percent(agreement['delta_1.10'])} / {percent(agreement['delta_1.25'])} |
| Fast − Foundation 平均偏差 | {agreement['mean_signed_fast_minus_foundation_m']:.4f} m |

## 稀疏 LiDAR 代理评估

共采样 {lidar['sampling']['frames_evaluated']} 帧、{lidar['unique_projected_pixels']} 个去重投影像素；相机相对 LiDAR 的时间差绝对值可能达到数十毫秒。下表只统计两种方法同时有效的相同点：

| 指标 | FoundationStereo | Fast-FoundationStereo |
|---|---:|---:|
| 有效点数 | {foundation_lidar['valid_predictions']} | {fast_lidar['valid_predictions']} |
| MAE | {foundation_lidar['depth_mae_m']:.4f} m | {fast_lidar['depth_mae_m']:.4f} m |
| RMSE | {foundation_lidar['depth_rmse_m']:.4f} m | {fast_lidar['depth_rmse_m']:.4f} m |
| 中位绝对误差 | {foundation_lidar['depth_median_absolute_error_m']:.4f} m | {fast_lidar['depth_median_absolute_error_m']:.4f} m |
| AbsRel | {percent(foundation_lidar['depth_absrel'])} | {percent(fast_lidar['depth_absrel'])} |
| δ1.25 | {percent(foundation_lidar['delta_1.25'])} | {percent(fast_lidar['delta_1.25'])} |

## 解释边界

原始采集明确关闭了深度传感器，没有稠密 GT。模型间 MAE 衡量的是一致性，不是绝对精度；LiDAR 指标是稀疏代理，受视点差、遮挡、像素量化和相机/LiDAR 时间偏差影响。Foundation 输出上限是 3 m，Fast 输出上限是 5 m，因此覆盖率不可直接横比；配对误差只在两者同时有效的共同区域统计。因而应主要用指标比较两种方法的相对趋势，不能当作公开数据集式的绝对精度排名。
"""


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    foundation = args.foundation.resolve()
    fast = args.fast.resolve()
    raw = args.raw_dataset.resolve()
    tick_index = load_json(foundation / "tick_index.json")
    foundation_run = load_json(foundation / "foundation_stereo_run.json")
    fast_run = load_json(fast / "fast_foundation_stereo_run.json")
    fast_smoke_run = (
        load_json(args.fast_smoke_run.resolve())
        if args.fast_smoke_run is not None
        else None
    )
    frames = tick_index["frames"]
    if int(fast_run["processed"]) != len(frames):
        raise ValueError("Fast-FoundationStereo run is incomplete")
    report = {
        "schema": "daaam.stereo_depth_method_comparison.v1",
        "status": "complete",
        "inputs": {
            "foundation": str(foundation),
            "fast": str(fast),
            "raw_dataset": str(raw),
            "frames": len(frames),
            "width": int(tick_index["width"]),
            "height": int(tick_index["height"]),
            "dense_ground_truth_available": False,
        },
        "timing": timing_comparison(foundation_run, fast_run, fast_smoke_run),
        "dense_inter_model_agreement": dense_agreement(
            foundation,
            fast,
            frames,
            min(
                float(foundation_run["maximum_depth_m"]),
                float(fast_run["settings"]["maximum_depth_m"]),
            ),
        ),
        "sparse_lidar_proxy": lidar_projection(
            foundation,
            fast,
            raw,
            tick_index,
            frame_step=args.lidar_step,
            minimum_depth_m=args.minimum_depth_m,
            maximum_depth_m=args.maximum_depth_m,
        ),
    }
    report["evaluation_elapsed_seconds"] = time.perf_counter() - started
    if args.preview_out is not None:
        write_preview(
            foundation,
            fast,
            frames,
            args.preview_out.resolve(),
            args.minimum_depth_m,
            args.maximum_depth_m,
        )
        report["preview"] = str(args.preview_out.resolve())
    atomic_write(args.json_out.resolve(), json.dumps(report, indent=2, allow_nan=False) + "\n")
    atomic_write(args.markdown_out.resolve(), render_markdown(report))
    print(f"wrote {args.json_out.resolve()}", flush=True)
    print(f"wrote {args.markdown_out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
