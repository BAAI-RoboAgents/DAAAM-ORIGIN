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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--fs-root", default="/home/user/Code/FoundationStereo", type=Path)
    parser.add_argument(
        "--checkpoint",
        default="/home/user/Code/FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth",
        type=Path,
    )
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-depth-m", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--swap-stereo",
        action="store_true",
        help="Use cam1 as the left image and cam0 as the right image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.fs_root.resolve()))
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    dataset = args.dataset.resolve()
    metadata = json.loads((dataset / "tick_index.json").read_text())
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    frames = metadata["frames"][args.start_frame :]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    fx = float(metadata["fx"])
    baseline = float(metadata["baseline"])
    depth_dir = dataset / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.checkpoint.parent / "cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    torch.autograd.set_grad_enabled(False)
    model = FoundationStereo(cfg)
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()

    print(
        f"FoundationStereo: frames={len(frames)} fx={fx:.6f} "
        f"baseline={baseline:.9f} swap_stereo={args.swap_stereo} "
        f"checkpoint={args.checkpoint}",
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
        valid &= np.isfinite(depth) & (depth > 0) & (depth <= args.max_depth_m)
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
        "foundation_stereo_root": str(args.fs_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "frames_requested": len(frames),
        "start_frame": args.start_frame,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "fx": fx,
        "baseline_m": baseline,
        "max_depth_m": args.max_depth_m,
        "valid_iters": args.valid_iters,
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
