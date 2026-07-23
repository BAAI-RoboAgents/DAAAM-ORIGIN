#!/usr/bin/env python3
"""Run pinned Fast-FoundationStereo on a prepared pinhole stereo dataset.

The output layout mirrors ``run_foundation_stereo_depth.py`` so the two runs
can be compared without changing image geometry or confidence thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.depth.confidence import (  # noqa: E402
    compute_left_right_confidence,
    disparity_to_metric_depth,
)


OFFICIAL_REPOSITORY_COMMIT = "a290ba04c1b3ad1ec41a33974a157b2917b624d4"
OFFICIAL_CHECKPOINT_ID = "20-30-48"
OFFICIAL_CHECKPOINT_SHA256 = (
    "98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692"
)
OFFICIAL_CHECKPOINT_SIZE = 62_078_956
OFFICIAL_PARAMETER_COUNT = 15_415_241
OFFICIAL_CONFIG_SHA256 = (
    "d45afe99b176454d5aff416edf16c8da6a99579f8f374b927f37907442a7d6bc"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=416)
    parser.add_argument(
        "--volume-builder", choices=("triton", "pytorch1"), default="triton"
    )
    parser.add_argument(
        "--confidence-mode", choices=("left-right", "validity"), default="left-right"
    )
    parser.add_argument("--lr-absolute-tolerance-px", type=float, default=0.75)
    parser.add_argument("--lr-relative-tolerance", type=float, default=0.03)
    parser.add_argument("--max-depth-m", type=float)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(repo: Path, checkpoint: Path) -> dict[str, Any]:
    repo = repo.resolve()
    checkpoint = checkpoint.resolve()
    required = (
        repo / "core" / "foundation_stereo.py",
        repo / "core" / "utils" / "utils.py",
        repo / "Utils.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Fast-FoundationStereo files missing: " + ", ".join(missing))
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if commit != OFFICIAL_REPOSITORY_COMMIT or dirty:
        raise RuntimeError(
            "Fast-FoundationStereo checkout must be clean and pinned to "
            f"{OFFICIAL_REPOSITORY_COMMIT}; commit={commit}, dirty={bool(dirty)}"
        )
    if checkpoint.stat().st_size != OFFICIAL_CHECKPOINT_SIZE:
        raise RuntimeError("Fast-FoundationStereo checkpoint size mismatch")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != OFFICIAL_CHECKPOINT_SHA256:
        raise RuntimeError("Fast-FoundationStereo checkpoint SHA-256 mismatch")
    config = checkpoint.parent / "cfg.yaml"
    if not config.is_file() or sha256_file(config) != OFFICIAL_CONFIG_SHA256:
        raise RuntimeError("Fast-FoundationStereo cfg.yaml SHA-256 mismatch")
    return {
        "repository": str(repo),
        "repository_commit": commit,
        "repository_clean": True,
        "checkpoint_id": OFFICIAL_CHECKPOINT_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "config": str(config),
        "config_sha256": OFFICIAL_CONFIG_SHA256,
        "verified": True,
    }


def activate_repo(repo: Path) -> Any:
    conflicts = sorted(
        name for name in sys.modules if name == "core" or name.startswith("core.")
    )
    if conflicts:
        raise RuntimeError(f"top-level core package already imported: {conflicts[:5]}")
    repo_text = str(repo.resolve())
    sys.path.insert(0, repo_text)
    importlib.invalidate_caches()
    module = importlib.import_module("core.foundation_stereo")
    if Path(module.__file__).resolve().is_relative_to(repo.resolve()) is False:
        raise RuntimeError(f"Fast-FoundationStereo import escaped repository: {module.__file__}")
    return module


def set_model_arg(model: Any, name: str, value: Any) -> None:
    try:
        setattr(model.args, name, value)
    except (AttributeError, TypeError):
        model.args[name] = value


def get_model_arg(model: Any, name: str, default: Any) -> Any:
    getter = getattr(model.args, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(model.args, name, default)


def load_model(repo: Path, checkpoint: Path, max_disp: int, iters: int) -> tuple[Any, dict]:
    module = activate_repo(repo)
    started = time.perf_counter()
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(model, module.FastFoundationStereo):
        raise TypeError(f"unexpected checkpoint object: {type(model)!r}")
    missing = object()
    normalize_raw = get_model_arg(model, "normalize", missing)
    normalize_missing = normalize_raw is missing
    normalize = True if normalize_missing else bool(normalize_raw)
    set_model_arg(model, "normalize", normalize)
    set_model_arg(model, "max_disp", max_disp)
    set_model_arg(model, "valid_iters", iters)
    set_model_arg(model, "mixed_precision", True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != OFFICIAL_PARAMETER_COUNT:
        raise RuntimeError(
            f"parameter count mismatch: {parameter_count} != {OFFICIAL_PARAMETER_COUNT}"
        )
    model.cuda().eval()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    return model, {
        "model_load_seconds": time.perf_counter() - started,
        "parameter_count": parameter_count,
        "normalize_was_missing": normalize_missing,
        "normalize_effective": normalize,
    }


def prepare_pair(left_bgr: np.ndarray, right_bgr: np.ndarray, padder_type: Any):
    left_rgb = np.ascontiguousarray(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB))
    right_rgb = np.ascontiguousarray(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB))
    left = torch.from_numpy(left_rgb).cuda().float()[None].permute(0, 3, 1, 2).contiguous()
    right = torch.from_numpy(right_rgb).cuda().float()[None].permute(0, 3, 1, 2).contiguous()
    padder = padder_type(left.shape, divis_by=32, force_square=False)
    left, right = padder.pad(left, right)
    return left.contiguous(), right.contiguous(), padder


def infer_once(model: Any, left: torch.Tensor, right: torch.Tensor, iters: int, builder: str):
    with torch.inference_mode(), torch.amp.autocast(
        "cuda", enabled=True, dtype=torch.float16
    ):
        return model.forward(
            left,
            right,
            iters=iters,
            test_mode=True,
            optimize_build_volume=builder,
        )


def timed_inference(
    model: Any,
    left: torch.Tensor,
    right: torch.Tensor,
    iters: int,
    builder: str,
):
    torch.cuda.synchronize()
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    event_start.record()
    output = infer_once(model, left, right, iters, builder)
    event_end.record()
    torch.cuda.synchronize()
    return output, {
        "wall_seconds": time.perf_counter() - wall_started,
        "cuda_event_seconds": event_start.elapsed_time(event_end) / 1000.0,
    }


def percentile_summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_seconds": float(array.mean()),
        "p50_seconds": float(np.percentile(array, 50)),
        "p95_seconds": float(np.percentile(array, 95)),
        "p99_seconds": float(np.percentile(array, 99)),
        "minimum_seconds": float(array.min()),
        "maximum_seconds": float(array.max()),
    }


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def capture_gpu_snapshot() -> dict[str, Any]:
    """Capture lightweight contention evidence without making it a run blocker."""
    commands = {
        "gpu": [
            "nvidia-smi",
            "--query-gpu=timestamp,name,driver_version,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        "compute_processes": [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    }
    result: dict[str, Any] = {}
    for name, command in commands.items():
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        result[name] = {
            "returncode": completed.returncode,
            "stdout_lines": [
                line.strip() for line in completed.stdout.splitlines() if line.strip()
            ],
            "stderr": completed.stderr.strip(),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.iters <= 0 or args.max_disp <= 0 or args.max_disp % 32:
        raise ValueError("iters must be positive and max-disp a positive multiple of 32")
    if args.start_frame < 0 or args.warmup < 0:
        raise ValueError("start-frame and warmup must be non-negative")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("max-frames must be positive")
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    metadata = json.loads((dataset / "tick_index.json").read_text())
    if metadata.get("projection_model") != "pinhole":
        raise ValueError("Fast-FoundationStereo requires a pinhole-rectified dataset")
    width = int(metadata["width"])
    height = int(metadata["height"])
    fx = float(metadata["fx"])
    baseline = float(metadata["baseline"])
    max_depth_m = float(
        args.max_depth_m
        if args.max_depth_m is not None
        else metadata.get("recommended_max_depth_m", 5.0)
    )
    if args.max_disp > math.ceil(width / 32) * 32:
        raise ValueError("max-disp exceeds padded image width")
    frames = metadata["frames"][args.start_frame :]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise ValueError("no frames selected")

    gpu_preflight = capture_gpu_snapshot()
    artifacts = verify_artifacts(args.repo, args.checkpoint)
    model, model_info = load_model(
        args.repo.resolve(), args.checkpoint.resolve(), args.max_disp, args.iters
    )
    from core.utils.utils import InputPadder

    directories = {
        "depth": output / "depth",
        "confidence": output / "depth_confidence",
        "consistency": output / "depth_consistency",
        "occlusion": output / "depth_occlusion",
        "metadata": output / "depth_metadata",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for directory in directories.values():
            suffix = ".json" if directory == directories["metadata"] else ".png"
            for path in directory.glob(f"*{suffix}"):
                path.unlink()

    first_left = cv2.imread(frames[0]["cam0"], cv2.IMREAD_COLOR)
    first_right = cv2.imread(frames[0]["cam1"], cv2.IMREAD_COLOR)
    if first_left is None or first_right is None or first_left.shape != first_right.shape:
        raise RuntimeError("invalid first stereo pair")
    warm_left, warm_right, _ = prepare_pair(first_left, first_right, InputPadder)
    warmup_started = time.perf_counter()
    for _ in range(args.warmup):
        infer_once(model, warm_left, warm_right, args.iters, args.volume_builder)
        if args.confidence_mode == "left-right":
            infer_once(
                model,
                torch.flip(warm_right, dims=[3]),
                torch.flip(warm_left, dims=[3]),
                args.iters,
                args.volume_builder,
            )
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started
    del warm_left, warm_right
    torch.cuda.reset_peak_memory_stats()

    print(
        f"Fast-FoundationStereo: frames={len(frames)} size={width}x{height} "
        f"iters={args.iters} dmax={args.max_disp} fp16 builder={args.volume_builder} "
        f"confidence={args.confidence_mode} max_depth={max_depth_m:.3f}m",
        flush=True,
    )
    run_started = time.perf_counter()
    processed = skipped = failed = 0
    frame_stats: list[dict[str, Any]] = []
    timing: dict[str, list[float]] = {
        "preprocess_wall": [],
        "left_model_wall": [],
        "left_model_cuda_event": [],
        "right_model_wall": [],
        "right_model_cuda_event": [],
        "dual_model_wall": [],
        "dual_model_cuda_event": [],
        "postprocess_and_write_wall": [],
        "end_to_end_wall": [],
    }
    for position, frame in enumerate(frames, start=1):
        frame_idx = int(frame["idx"])
        paths = {
            name: directory
            / f"{frame_idx:08d}{'.json' if name == 'metadata' else '.png'}"
            for name, directory in directories.items()
        }
        if all(path.exists() for path in paths.values()) and not args.overwrite:
            skipped += 1
            try:
                frame_stats.append(json.loads(paths["metadata"].read_text()))
            except (OSError, json.JSONDecodeError):
                pass
            continue
        frame_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        left_bgr = cv2.imread(frame["cam0"], cv2.IMREAD_COLOR)
        right_bgr = cv2.imread(frame["cam1"], cv2.IMREAD_COLOR)
        if (
            left_bgr is None
            or right_bgr is None
            or left_bgr.shape != right_bgr.shape
            or left_bgr.shape[:2] != (height, width)
        ):
            print(f"[{frame_idx}] invalid stereo pair", flush=True)
            failed += 1
            continue
        left, right, padder = prepare_pair(left_bgr, right_bgr, InputPadder)
        torch.cuda.synchronize()
        timing["preprocess_wall"].append(time.perf_counter() - preprocess_started)
        try:
            left_tensor, left_timing = timed_inference(
                model, left, right, args.iters, args.volume_builder
            )
            right_tensor = None
            right_timing = {"wall_seconds": 0.0, "cuda_event_seconds": 0.0}
            if args.confidence_mode == "left-right":
                right_tensor, right_timing = timed_inference(
                    model,
                    torch.flip(right, dims=[3]),
                    torch.flip(left, dims=[3]),
                    args.iters,
                    args.volume_builder,
                )
        except RuntimeError as error:
            print(f"[{frame_idx}] CUDA/model error: {error}", flush=True)
            torch.cuda.empty_cache()
            failed += 1
            continue
        timing["left_model_wall"].append(left_timing["wall_seconds"])
        timing["left_model_cuda_event"].append(left_timing["cuda_event_seconds"])
        if right_tensor is not None:
            timing["right_model_wall"].append(right_timing["wall_seconds"])
            timing["right_model_cuda_event"].append(right_timing["cuda_event_seconds"])
        timing["dual_model_wall"].append(
            left_timing["wall_seconds"] + right_timing["wall_seconds"]
        )
        timing["dual_model_cuda_event"].append(
            left_timing["cuda_event_seconds"] + right_timing["cuda_event_seconds"]
        )

        post_started = time.perf_counter()
        left_disp = padder.unpad(left_tensor.float()).cpu().numpy().squeeze()
        left_disp = np.clip(left_disp, 0.0, None).astype(np.float32, copy=False)
        if right_tensor is not None:
            right_disp = (
                padder.unpad(torch.flip(right_tensor, dims=[3]).float())
                .cpu()
                .numpy()
                .squeeze()
            )
            right_disp = np.clip(right_disp, 0.0, None).astype(np.float32, copy=False)
            stereo = compute_left_right_confidence(
                left_disp,
                right_disp,
                absolute_tolerance_px=args.lr_absolute_tolerance_px,
                relative_tolerance=args.lr_relative_tolerance,
            )
            confidence = stereo.confidence
            consistent = stereo.consistent_mask
            occluded = stereo.occlusion_mask
            confidence_metrics = stereo.metrics
        else:
            consistent = np.isfinite(left_disp) & (left_disp > 0.0)
            occluded = np.zeros_like(consistent)
            confidence = consistent.astype(np.float32)
            confidence_metrics = {
                "left_right_consistency": None,
                "occlusion_ratio": 0.0,
            }
        depth = disparity_to_metric_depth(
            left_disp,
            focal_length_px=fx,
            baseline_m=baseline,
            maximum_depth_m=max_depth_m,
            valid_mask=consistent,
        )
        products = {
            "depth": np.rint(depth * 1000.0).astype(np.uint16),
            "confidence": np.rint(np.clip(confidence, 0.0, 1.0) * 255.0).astype(np.uint8),
            "consistency": consistent.astype(np.uint8) * 255,
            "occlusion": occluded.astype(np.uint8) * 255,
        }
        if not all(cv2.imwrite(str(paths[name]), product) for name, product in products.items()):
            print(f"[{frame_idx}] failed to write depth products", flush=True)
            failed += 1
            continue
        valid = depth > 0.0
        report = {
            "frame_idx": frame_idx,
            "sensor_time_ns": int(frame["sensor_time_ns"]),
            "valid_ratio": float(valid.mean()),
            "median_depth_m": float(np.median(depth[valid])) if np.any(valid) else None,
            "left_right_consistency": confidence_metrics.get("left_right_consistency"),
            "confidence_mode": args.confidence_mode,
            "left_right_verified": args.confidence_mode == "left-right",
            "occlusion_ratio": float(occluded.mean()),
            "mean_confidence": float(np.mean(confidence[valid])) if np.any(valid) else 0.0,
            "timing_seconds": {
                "preprocess_wall": timing["preprocess_wall"][-1],
                "left_model_wall": left_timing["wall_seconds"],
                "left_model_cuda_event": left_timing["cuda_event_seconds"],
                "right_model_wall": right_timing["wall_seconds"] if right_tensor is not None else None,
                "right_model_cuda_event": right_timing["cuda_event_seconds"] if right_tensor is not None else None,
                "dual_model_wall": timing["dual_model_wall"][-1],
                "dual_model_cuda_event": timing["dual_model_cuda_event"][-1],
            },
        }
        post_seconds = time.perf_counter() - post_started
        end_to_end_seconds = time.perf_counter() - frame_started
        report["timing_seconds"]["postprocess_and_write_wall"] = post_seconds
        report["timing_seconds"]["end_to_end_wall"] = end_to_end_seconds
        atomic_write_json(paths["metadata"], report)
        timing["postprocess_and_write_wall"].append(post_seconds)
        timing["end_to_end_wall"].append(end_to_end_seconds)
        frame_stats.append(report)
        processed += 1
        if processed % 25 == 0 or position == len(frames):
            print(
                f"{position}/{len(frames)} processed={processed} "
                f"valid={report['valid_ratio']:.3f} "
                f"LR={report['left_right_consistency']} "
                f"model={timing['dual_model_wall'][-1]:.3f}s "
                f"elapsed={time.perf_counter() - run_started:.1f}s",
                flush=True,
            )

    elapsed_seconds = time.perf_counter() - run_started
    valid_frame_stats = [item for item in frame_stats if "valid_ratio" in item]
    result = {
        "status": "complete" if failed == 0 else "failed",
        "dataset": str(dataset),
        "output": str(output),
        "artifacts": artifacts,
        "model": model_info,
        "settings": {
            "width": width,
            "height": height,
            "fx": fx,
            "baseline_m": baseline,
            "maximum_depth_m": max_depth_m,
            "iterations": args.iters,
            "maximum_disparity_px": args.max_disp,
            "represented_minimum_depth_m": fx * baseline / args.max_disp,
            "precision": "fp16",
            "volume_builder": args.volume_builder,
            "confidence_mode": args.confidence_mode,
            "left_right_inferences_per_frame": 2 if args.confidence_mode == "left-right" else 1,
            "lr_absolute_tolerance_px": args.lr_absolute_tolerance_px,
            "lr_relative_tolerance": args.lr_relative_tolerance,
        },
        "frames_requested": len(frames),
        "start_frame": args.start_frame,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "warmup_iterations": args.warmup,
        "warmup_seconds": warmup_seconds,
        "elapsed_seconds": elapsed_seconds,
        "gpu_snapshots": {
            "preflight_before_model_load": gpu_preflight,
            "postflight_with_model_resident": capture_gpu_snapshot(),
        },
        "timing": {name: percentile_summary(values) for name, values in timing.items()},
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "aggregate": {
            "mean_valid_ratio": float(np.mean([item["valid_ratio"] for item in valid_frame_stats])) if valid_frame_stats else None,
            "mean_left_right_consistency": float(np.mean([item["left_right_consistency"] for item in valid_frame_stats if item.get("left_right_consistency") is not None])) if any(item.get("left_right_consistency") is not None for item in valid_frame_stats) else None,
            "mean_confidence": float(np.mean([item["mean_confidence"] for item in valid_frame_stats])) if valid_frame_stats else None,
        },
        "frame_stats": valid_frame_stats,
    }
    atomic_write_json(output / "fast_foundation_stereo_run.json", result)
    atomic_write_json(output / "tick_index.json", metadata)
    print(json.dumps({key: value for key, value in result.items() if key != "frame_stats"}, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
