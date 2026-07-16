#!/usr/bin/env python3
"""Persistent FoundationStereo JSON-lines worker for the realtime scheduler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from daaam.depth.confidence import (  # noqa: E402
    compute_left_right_confidence,
    disparity_to_metric_depth,
)
from run_foundation_stereo_depth import resolve_inference_profile  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=("online", "refine", "custom"), default="online")
    parser.add_argument("--valid-iters", type=int)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--confidence-mode", choices=("left-right", "validity"), default="left-right"
    )
    parser.add_argument("--lr-absolute-tolerance-px", type=float, default=0.75)
    parser.add_argument("--lr-relative-tolerance", type=float, default=0.03)
    return parser.parse_args()


class InferenceWorker:
    def __init__(self, args) -> None:
        import torch
        from omegaconf import OmegaConf

        fs_root = args.fs_root.resolve()
        checkpoint_path = args.checkpoint.resolve()
        if not (fs_root / "core" / "foundation_stereo.py").is_file():
            raise ValueError(f"invalid FoundationStereo root: {fs_root}")
        if not checkpoint_path.is_file():
            raise ValueError(f"missing checkpoint: {checkpoint_path}")
        self.profile = resolve_inference_profile(args)
        sys.path.insert(0, str(fs_root))
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder

        self.torch = torch
        self.InputPadder = InputPadder
        cfg = OmegaConf.load(checkpoint_path.parent / "cfg.yaml")
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        torch.autograd.set_grad_enabled(False)
        torch.set_float32_matmul_precision("high")
        model = FoundationStereo(cfg)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        self.model = model.cuda().eval()
        if self.profile["torch_compile"]:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        self.autocast_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[self.profile["precision"]]

    def _infer_pair(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        confidence_mode: str,
    ):
        torch = self.torch
        image0 = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
        image1 = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
        padder = self.InputPadder(image0.shape, divis_by=32, force_square=False)
        image0, image1 = padder.pad(image0, image1)
        with torch.amp.autocast(
            "cuda",
            dtype=self.autocast_dtype,
            enabled=self.profile["precision"] != "fp32",
        ):
            left_tensor = self.model.forward(
                image0,
                image1,
                iters=self.profile["valid_iters"],
                test_mode=True,
            )
            right_tensor = None
            if confidence_mode == "left-right":
                right_tensor = self.model.forward(
                    torch.flip(image1, dims=[3]),
                    torch.flip(image0, dims=[3]),
                    iters=self.profile["valid_iters"],
                    test_mode=True,
                )
                right_tensor = torch.flip(right_tensor, dims=[3])
        left_disparity = padder.unpad(left_tensor.float()).cpu().numpy().squeeze()
        right_disparity = (
            padder.unpad(right_tensor.float()).cpu().numpy().squeeze()
            if right_tensor is not None
            else None
        )
        return left_disparity, right_disparity

    def infer(self, request: dict) -> dict:
        started = time.time()
        confidence_mode = str(
            request.get("confidence_mode", self.profile["confidence_mode"])
        )
        if confidence_mode not in {"left-right", "validity"}:
            raise ValueError(f"unsupported confidence mode: {confidence_mode}")
        left_bgr = cv2.imread(str(request["left_path"]), cv2.IMREAD_COLOR)
        right_bgr = cv2.imread(str(request["right_path"]), cv2.IMREAD_COLOR)
        if left_bgr is None or right_bgr is None or left_bgr.shape != right_bgr.shape:
            raise ValueError("invalid stereo image request")
        original_height, original_width = left_bgr.shape[:2]
        scale = self.profile["scale"]
        if scale != 1.0:
            size = (
                max(32, int(round(original_width * scale))),
                max(32, int(round(original_height * scale))),
            )
            left_bgr = cv2.resize(left_bgr, size, interpolation=cv2.INTER_AREA)
            right_bgr = cv2.resize(right_bgr, size, interpolation=cv2.INTER_AREA)
            actual_scale = left_bgr.shape[1] / original_width
        else:
            actual_scale = 1.0
        left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        left_disparity, right_disparity = self._infer_pair(
            left,
            right,
            confidence_mode=confidence_mode,
        )

        def original_disparity(disparity):
            if actual_scale == 1.0:
                return disparity.astype(np.float32)
            return (
                cv2.resize(
                    disparity,
                    (original_width, original_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                / actual_scale
            ).astype(np.float32)

        left_disparity = original_disparity(left_disparity)
        if right_disparity is not None:
            confidence_result = compute_left_right_confidence(
                left_disparity,
                original_disparity(right_disparity),
                absolute_tolerance_px=self.profile["lr_absolute_tolerance_px"],
                relative_tolerance=self.profile["lr_relative_tolerance"],
            )
            valid = confidence_result.consistent_mask
            confidence = confidence_result.confidence
            occlusion = confidence_result.occlusion_mask
            consistency = confidence_result.metrics["left_right_consistency"]
        else:
            valid = np.isfinite(left_disparity) & (left_disparity > 0.0)
            confidence = valid.astype(np.float32)
            occlusion = np.zeros_like(valid)
            consistency = None
        depth = disparity_to_metric_depth(
            left_disparity,
            focal_length_px=float(request["fx"]),
            baseline_m=float(request["baseline_m"]),
            maximum_depth_m=float(request["maximum_depth_m"]),
            valid_mask=valid,
        )
        output_dir = Path(request["output_dir"])
        frame_name = f"{int(request['frame_idx']):08d}.png"
        paths = {
            "depth_path": output_dir / "depth" / frame_name,
            "confidence_path": output_dir / "depth_confidence" / frame_name,
            "consistency_path": output_dir / "depth_consistency" / frame_name,
            "occlusion_path": output_dir / "depth_occlusion" / frame_name,
            "metadata_path": output_dir
            / "depth_metadata"
            / f"{int(request['frame_idx']):08d}.json",
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        products = {
            "depth_path": np.rint(depth * 1000.0).astype(np.uint16),
            "confidence_path": np.rint(confidence * 255.0).astype(np.uint8),
            "consistency_path": (
                valid.astype(np.uint8) * 255
                if confidence_mode == "left-right"
                else np.zeros_like(valid, dtype=np.uint8)
            ),
            "occlusion_path": occlusion.astype(np.uint8) * 255,
        }
        for key, product in products.items():
            if not cv2.imwrite(str(paths[key]), product):
                raise IOError(f"failed to write {paths[key]}")
        valid_depth = depth > 0.0
        metadata = {
            "frame_idx": int(request["frame_idx"]),
            "sensor_time_ns": int(request["sensor_time_ns"]),
            "confidence_mode": confidence_mode,
            "left_right_verified": confidence_mode == "left-right",
            "valid_ratio": float(valid_depth.mean()),
            "left_right_consistency": consistency,
            "elapsed_seconds": time.time() - started,
            "cuda_memory_allocated_bytes": int(
                self.torch.cuda.memory_allocated()
            ),
            "peak_cuda_memory_bytes": int(
                self.torch.cuda.max_memory_allocated()
            ),
            "peak_worker_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        }
        temporary_metadata = paths["metadata_path"].with_suffix(".json.tmp")
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        temporary_metadata.replace(paths["metadata_path"])
        return {
            "status": "ok",
            "request_id": request["request_id"],
            **metadata,
            **{key: str(path) for key, path in paths.items()},
        }


def emit(message: dict) -> None:
    print(json.dumps(message, allow_nan=False), flush=True)


def main() -> None:
    args = parse_args()
    worker = InferenceWorker(args)
    emit({"status": "ready", "profile": worker.profile})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                emit({"status": "stopped"})
                break
            if request.get("command") != "infer":
                raise ValueError("unknown worker command")
            emit(worker.infer(request))
        except Exception as error:
            if "torch" in locals() and "out of memory" in str(error).lower():
                worker.torch.cuda.empty_cache()
            emit(
                {
                    "status": "error",
                    "request_id": request.get("request_id") if "request" in locals() else None,
                    "error": repr(error),
                }
            )


if __name__ == "__main__":
    main()
