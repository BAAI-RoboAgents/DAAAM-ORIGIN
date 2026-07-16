#!/usr/bin/env python3
"""Run FoundationStereo profiles and save metric depth plus confidence products."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOUNDATION_STEREO_ROOT = REPOSITORY_ROOT / "third_party" / "FoundationStereo"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.depth.confidence import (  # noqa: E402
    compute_left_right_confidence,
    disparity_to_metric_depth,
)


PROFILE_DEFAULTS = {
    "online": {"valid_iters": 8, "scale": 0.15, "precision": "fp16"},
    "refine": {"valid_iters": 32, "scale": 1.0, "precision": "fp16"},
    "custom": {"valid_iters": 32, "scale": 1.0, "precision": "fp16"},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--fs-root",
        default=DEFAULT_FOUNDATION_STEREO_ROOT,
        type=Path,
        help="FoundationStereo checkout; defaults to the pinned project submodule.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("FOUNDATION_STEREO_CHECKPOINT"),
        type=Path,
        help="Model checkpoint, or set FOUNDATION_STEREO_CHECKPOINT.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_DEFAULTS),
        default="refine",
        help="online is lower latency; refine preserves the full-resolution baseline.",
    )
    parser.add_argument("--valid-iters", type=int)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--confidence-mode",
        choices=("left-right", "validity"),
        default="left-right",
        help="left-right performs a second inference; validity is the low-latency fallback.",
    )
    parser.add_argument("--lr-absolute-tolerance-px", type=float, default=0.75)
    parser.add_argument("--lr-relative-tolerance", type=float, default=0.03)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--max-depth-m",
        type=float,
        help=(
            "Maximum output depth. Defaults to recommended_max_depth_m from "
            "tick_index.json, or 20m for legacy datasets."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-non-pinhole",
        action="store_true",
        help="Bypass the projection-model safety check.",
    )
    parser.add_argument(
        "--swap-stereo",
        action="store_true",
        help="Use cam1 as the left image and cam0 as the right image.",
    )
    return parser.parse_args()


def resolve_inference_profile(args: argparse.Namespace | SimpleNamespace) -> dict[str, Any]:
    profile = str(args.profile)
    if profile not in PROFILE_DEFAULTS:
        raise ValueError(f"Unknown FoundationStereo profile: {profile}")
    defaults = PROFILE_DEFAULTS[profile]
    valid_iters = args.valid_iters if args.valid_iters is not None else defaults["valid_iters"]
    scale = args.scale if args.scale is not None else defaults["scale"]
    precision = args.precision if args.precision is not None else defaults["precision"]
    if valid_iters not in (8, 16, 32):
        raise ValueError("valid_iters must be one of 8, 16, or 32")
    if not 0.0 < float(scale) <= 1.0:
        raise ValueError("scale must be in (0, 1]")
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16, or bf16")
    if args.lr_absolute_tolerance_px <= 0 or args.lr_relative_tolerance < 0:
        raise ValueError("left/right confidence tolerances are invalid")
    return {
        "name": profile,
        "valid_iters": int(valid_iters),
        "scale": float(scale),
        "precision": precision,
        "torch_compile": bool(args.torch_compile),
        "confidence_mode": str(args.confidence_mode),
        "left_right_inferences_per_frame": (
            2 if args.confidence_mode == "left-right" else 1
        ),
        "lr_absolute_tolerance_px": float(args.lr_absolute_tolerance_px),
        "lr_relative_tolerance": float(args.lr_relative_tolerance),
    }


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, dict, list, dict]:
    fs_root = args.fs_root.resolve()
    if not fs_root.is_dir() or not (fs_root / "core" / "foundation_stereo.py").exists():
        raise ValueError(
            f"FoundationStereo checkout is unavailable at {fs_root}. Initialize "
            "third_party/FoundationStereo or pass --fs-root."
        )
    if args.checkpoint is None:
        raise ValueError(
            "Pass --checkpoint or set FOUNDATION_STEREO_CHECKPOINT before depth inference."
        )
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ValueError(f"FoundationStereo checkpoint is missing: {checkpoint_path}")
    if not (checkpoint_path.parent / "cfg.yaml").is_file():
        raise ValueError(f"FoundationStereo cfg.yaml is missing beside {checkpoint_path}")
    dataset = args.dataset.resolve()
    metadata_path = dataset / "tick_index.json"
    if not metadata_path.is_file():
        raise ValueError(f"Dataset metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    projection_model = metadata.get("projection_model")
    is_g1_capture = metadata.get("source_layout") == "capture4daaam_like"
    if not args.allow_non_pinhole and (
        projection_model not in (None, "pinhole")
        or (is_g1_capture and projection_model != "pinhole")
    ):
        raise ValueError(
            "FoundationStereo requires rectified pinhole input. Convert the "
            "G1 fisheye sequence with prepare_g1_pinhole_stereo_dataset.py, "
            "or pass --allow-non-pinhole only if the metadata is known to be wrong."
        )
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    frames = metadata["frames"][args.start_frame :]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise ValueError("No FoundationStereo frames were selected")
    profile = resolve_inference_profile(args)
    return fs_root, checkpoint_path, metadata, frames, profile


def _resize_disparity_to_original(
    disparity: np.ndarray, original_size: tuple[int, int], scale: float
) -> np.ndarray:
    width, height = original_size
    if disparity.shape == (height, width) and scale == 1.0:
        return disparity.astype(np.float32, copy=False)
    resized = cv2.resize(disparity, (width, height), interpolation=cv2.INTER_LINEAR)
    return (resized / scale).astype(np.float32)


def main():
    args = parse_args()
    fs_root, checkpoint_path, metadata, frames, profile = validate_inputs(args)
    dataset = args.dataset.resolve()
    fx = float(metadata["fx"])
    baseline = float(metadata["baseline"])
    max_depth_m = (
        args.max_depth_m
        if args.max_depth_m is not None
        else float(metadata.get("recommended_max_depth_m", 20.0))
    )
    if max_depth_m <= 0.0 or max_depth_m > np.iinfo(np.uint16).max / 1000.0:
        raise ValueError(f"Invalid maximum depth: {max_depth_m}m")
    plan = {
        "dataset": str(dataset),
        "foundation_stereo_root": str(fs_root),
        "checkpoint": str(checkpoint_path),
        "frames_requested": len(frames),
        "start_frame": args.start_frame,
        "profile": profile,
        "fx": fx,
        "baseline_m": baseline,
        "maximum_depth_m": max_depth_m,
        "projection_model": metadata.get("projection_model"),
        "swap_stereo": args.swap_stereo,
    }
    if args.dry_run:
        print(json.dumps({**plan, "status": "planned"}, indent=2), flush=True)
        return

    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(fs_root))
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    output_directories = {
        "depth": dataset / "depth",
        "confidence": dataset / "depth_confidence",
        "consistency": dataset / "depth_consistency",
        "occlusion": dataset / "depth_occlusion",
        "metadata": dataset / "depth_metadata",
    }
    for directory in output_directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(checkpoint_path.parent / "cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    torch.autograd.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    model = FoundationStereo(cfg)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()
    if profile["torch_compile"]:
        model = torch.compile(model, mode="reduce-overhead")

    autocast_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[profile["precision"]]
    autocast_enabled = profile["precision"] != "fp32"

    print(
        f"FoundationStereo: frames={len(frames)} fx={fx:.6f} "
        f"baseline={baseline:.9f} profile={profile['name']} "
        f"iters={profile['valid_iters']} scale={profile['scale']:.3f} "
        f"precision={profile['precision']} confidence={profile['confidence_mode']} "
        f"swap_stereo={args.swap_stereo} max_depth={max_depth_m:.3f}m",
        flush=True,
    )
    started = time.time()
    inference_seconds = 0.0
    processed = skipped = failed = 0
    frame_stats = []
    left_key, right_key = ("cam1", "cam0") if args.swap_stereo else ("cam0", "cam1")
    for index, frame in enumerate(frames, start=1):
        frame_index = int(frame["idx"])
        paths = {
            name: directory
            / f"{frame_index:08d}{'.json' if name == 'metadata' else '.png'}"
            for name, directory in output_directories.items()
        }
        if all(path.exists() for path in paths.values()) and not args.overwrite:
            skipped += 1
            continue
        left_bgr = cv2.imread(frame[left_key], cv2.IMREAD_COLOR)
        right_bgr = cv2.imread(frame[right_key], cv2.IMREAD_COLOR)
        if left_bgr is None or right_bgr is None or left_bgr.shape != right_bgr.shape:
            print(f"[{frame_index}] invalid stereo pair", flush=True)
            failed += 1
            continue

        original_height, original_width = left_bgr.shape[:2]
        scale = profile["scale"]
        if scale != 1.0:
            scaled_size = (
                max(32, int(round(original_width * scale))),
                max(32, int(round(original_height * scale))),
            )
            left_bgr = cv2.resize(left_bgr, scaled_size, interpolation=cv2.INTER_AREA)
            right_bgr = cv2.resize(right_bgr, scaled_size, interpolation=cv2.INTER_AREA)
            actual_scale = left_bgr.shape[1] / original_width
        else:
            actual_scale = 1.0
        left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        image0 = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
        image1 = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(image0.shape, divis_by=32, force_square=False)
        image0, image1 = padder.pad(image0, image1)
        inference_started = time.time()
        try:
            with torch.amp.autocast(
                "cuda", dtype=autocast_dtype, enabled=autocast_enabled
            ):
                left_disparity_tensor = model.forward(
                    image0,
                    image1,
                    iters=profile["valid_iters"],
                    test_mode=True,
                )
                right_disparity_tensor = None
                if profile["confidence_mode"] == "left-right":
                    right_disparity_tensor = model.forward(
                        torch.flip(image1, dims=[3]),
                        torch.flip(image0, dims=[3]),
                        iters=profile["valid_iters"],
                        test_mode=True,
                    )
                    right_disparity_tensor = torch.flip(
                        right_disparity_tensor, dims=[3]
                    )
        except RuntimeError as error:
            print(f"[{frame_index}] CUDA error: {error}", flush=True)
            torch.cuda.empty_cache()
            failed += 1
            continue
        inference_seconds += time.time() - inference_started

        left_disparity = (
            padder.unpad(left_disparity_tensor.float()).cpu().numpy().squeeze()
        )
        left_disparity = _resize_disparity_to_original(
            left_disparity, (original_width, original_height), actual_scale
        )
        if right_disparity_tensor is not None:
            right_disparity = (
                padder.unpad(right_disparity_tensor.float()).cpu().numpy().squeeze()
            )
            right_disparity = _resize_disparity_to_original(
                right_disparity, (original_width, original_height), actual_scale
            )
            stereo_confidence = compute_left_right_confidence(
                left_disparity,
                right_disparity,
                absolute_tolerance_px=profile["lr_absolute_tolerance_px"],
                relative_tolerance=profile["lr_relative_tolerance"],
            )
            confidence = stereo_confidence.confidence
            consistent = stereo_confidence.consistent_mask
            occluded = stereo_confidence.occlusion_mask
            confidence_metrics = stereo_confidence.metrics
        else:
            consistent = np.isfinite(left_disparity) & (left_disparity > 0.0)
            occluded = np.zeros_like(consistent)
            confidence = consistent.astype(np.float32)
            confidence_metrics = {
                "left_right_consistency": None,
                "consistent_pixels": int(consistent.sum()),
                "occlusion_pixels": 0,
                "confidence_mode": "validity_fallback",
            }

        depth = disparity_to_metric_depth(
            left_disparity,
            focal_length_px=fx,
            baseline_m=baseline,
            maximum_depth_m=max_depth_m,
            valid_mask=consistent,
        )
        depth_mm = np.rint(depth * 1000.0).astype(np.uint16)
        confidence_u8 = np.rint(np.clip(confidence, 0.0, 1.0) * 255.0).astype(np.uint8)
        consistency_u8 = (
            consistent.astype(np.uint8) * 255
            if profile["confidence_mode"] == "left-right"
            else np.zeros_like(consistent, dtype=np.uint8)
        )
        occlusion_u8 = occluded.astype(np.uint8) * 255
        products = {
            "depth": depth_mm,
            "confidence": confidence_u8,
            "consistency": consistency_u8,
            "occlusion": occlusion_u8,
        }
        if not all(
            cv2.imwrite(str(paths[name]), product)
            for name, product in products.items()
        ):
            print(f"[{frame_index}] failed to write one or more depth products", flush=True)
            failed += 1
            continue
        processed += 1
        valid_depth = depth > 0.0
        frame_report = {
            "frame_idx": frame_index,
            "sensor_time_ns": int(frame["sensor_time_ns"]),
            "valid_ratio": float(valid_depth.mean()),
            "median_depth_m": (
                float(np.median(depth[valid_depth])) if np.any(valid_depth) else None
            ),
            "left_right_consistency": confidence_metrics.get(
                "left_right_consistency"
            ),
            "confidence_mode": profile["confidence_mode"],
            "left_right_verified": profile["confidence_mode"] == "left-right",
            "occlusion_ratio": float(occluded.mean()),
            "mean_confidence": (
                float(np.mean(confidence[valid_depth])) if np.any(valid_depth) else 0.0
            ),
        }
        temporary_metadata = paths["metadata"].with_suffix(".json.tmp")
        temporary_metadata.write_text(
            json.dumps(frame_report, indent=2, allow_nan=False) + "\n"
        )
        temporary_metadata.replace(paths["metadata"])
        frame_stats.append(frame_report)

        if processed % 25 == 0 or index == len(frames):
            elapsed = time.time() - started
            print(
                f"{index}/{len(frames)} processed={processed} "
                f"valid={frame_report['valid_ratio']:.3f} "
                f"consistency={frame_report['left_right_consistency']} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        if processed % 50 == 0:
            torch.cuda.empty_cache()

    elapsed_seconds = time.time() - started
    result = {
        **plan,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "valid_iters": profile["valid_iters"],
        "scale": profile["scale"],
        "precision": profile["precision"],
        "confidence_mode": profile["confidence_mode"],
        "left_camera": "cam1" if args.swap_stereo else "cam0",
        "right_camera": "cam0" if args.swap_stereo else "cam1",
        "stereo_pairs_over_10ms": sum(
            frame.get("stereo_delta_ms") is not None
            and frame["stereo_delta_ms"] > 10.0
            for frame in frames
        ),
        "elapsed_seconds": elapsed_seconds,
        "inference_seconds": inference_seconds,
        "inference_seconds_per_processed_frame": (
            inference_seconds / processed if processed else None
        ),
        "effective_inference_hz": (
            processed / inference_seconds if inference_seconds > 0 else None
        ),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "aggregate": {
            "mean_valid_ratio": (
                float(np.mean([item["valid_ratio"] for item in frame_stats]))
                if frame_stats
                else None
            ),
            "mean_left_right_consistency": (
                float(
                    np.mean(
                        [
                            item["left_right_consistency"]
                            for item in frame_stats
                            if item["left_right_consistency"] is not None
                        ]
                    )
                )
                if any(
                    item["left_right_consistency"] is not None
                    for item in frame_stats
                )
                else None
            ),
            "mean_confidence": (
                float(np.mean([item["mean_confidence"] for item in frame_stats]))
                if frame_stats
                else None
            ),
        },
        "frame_stats": frame_stats,
    }
    (dataset / "foundation_stereo_run.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
