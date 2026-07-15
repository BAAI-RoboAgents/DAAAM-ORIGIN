#!/usr/bin/env python3
"""Run FoundationStereo on a prepared stereo sequence and save metric depth."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOUNDATION_STEREO_ROOT = REPOSITORY_ROOT / "third_party" / "FoundationStereo"


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
    parser.add_argument("--valid-iters", type=int, default=32)
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


def main():
    args = parse_args()
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
    sys.path.insert(0, str(fs_root))
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    dataset = args.dataset.resolve()
    metadata = json.loads((dataset / "tick_index.json").read_text())
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
    frames = metadata["frames"][args.start_frame :]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    fx = float(metadata["fx"])
    baseline = float(metadata["baseline"])
    max_depth_m = (
        args.max_depth_m
        if args.max_depth_m is not None
        else float(metadata.get("recommended_max_depth_m", 20.0))
    )
    if max_depth_m <= 0.0 or max_depth_m > np.iinfo(np.uint16).max / 1000.0:
        raise ValueError(f"Invalid maximum depth: {max_depth_m}m")
    depth_dir = dataset / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(checkpoint_path.parent / "cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    torch.autograd.set_grad_enabled(False)
    model = FoundationStereo(cfg)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()

    print(
        f"FoundationStereo: frames={len(frames)} fx={fx:.6f} "
        f"baseline={baseline:.9f} swap_stereo={args.swap_stereo} "
        f"max_depth={max_depth_m:.3f}m "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )
    started = time.time()
    processed = skipped = failed = 0
    for index, frame in enumerate(frames, start=1):
        output_path = depth_dir / f"{int(frame['idx']):08d}.png"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        left_key, right_key = ("cam1", "cam0") if args.swap_stereo else ("cam0", "cam1")
        left = cv2.imread(frame[left_key], cv2.IMREAD_COLOR)
        right = cv2.imread(frame[right_key], cv2.IMREAD_COLOR)
        if left is None or right is None or left.shape != right.shape:
            print(f"[{frame['idx']}] invalid stereo pair", flush=True)
            failed += 1
            continue

        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
        image0 = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
        image1 = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(image0.shape, divis_by=32, force_square=False)
        image0, image1 = padder.pad(image0, image1)
        try:
            with torch.amp.autocast("cuda"):
                disparity = model.forward(
                    image0, image1, iters=args.valid_iters, test_mode=True
                )
        except RuntimeError as error:
            print(f"[{frame['idx']}] CUDA error: {error}", flush=True)
            torch.cuda.empty_cache()
            failed += 1
            continue

        disparity = padder.unpad(disparity.float()).cpu().numpy().squeeze()
        valid = np.isfinite(disparity) & (disparity > 0)
        depth = np.zeros_like(disparity, dtype=np.float32)
        depth[valid] = fx * baseline / disparity[valid]
        valid &= np.isfinite(depth) & (depth > 0) & (depth <= max_depth_m)
        depth[~valid] = 0
        depth_mm = np.rint(depth * 1000.0).astype(np.uint16)
        if not cv2.imwrite(str(output_path), depth_mm):
            print(f"[{frame['idx']}] failed to write {output_path}", flush=True)
            failed += 1
            continue
        processed += 1

        if processed % 25 == 0 or index == len(frames):
            values = depth[valid]
            elapsed = time.time() - started
            print(
                f"{index}/{len(frames)} processed={processed} "
                f"valid={valid.mean():.3f} median={np.median(values):.3f}m "
                f"p95={np.percentile(values, 95):.3f}m elapsed={elapsed:.1f}s",
                flush=True,
            )
        if processed % 50 == 0:
            torch.cuda.empty_cache()

    result = {
        "dataset": str(dataset),
        "foundation_stereo_root": str(fs_root),
        "checkpoint": str(checkpoint_path),
        "frames_requested": len(frames),
        "start_frame": args.start_frame,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "fx": fx,
        "baseline_m": baseline,
        "max_depth_m": max_depth_m,
        "valid_iters": args.valid_iters,
        "projection_model": projection_model,
        "swap_stereo": args.swap_stereo,
        "left_camera": "cam1" if args.swap_stereo else "cam0",
        "right_camera": "cam0" if args.swap_stereo else "cam1",
        "stereo_pairs_over_10ms": sum(
            frame.get("stereo_delta_ms") is not None
            and frame["stereo_delta_ms"] > 10.0
            for frame in frames
        ),
        "elapsed_seconds": time.time() - started,
    }
    (dataset / "foundation_stereo_run.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
