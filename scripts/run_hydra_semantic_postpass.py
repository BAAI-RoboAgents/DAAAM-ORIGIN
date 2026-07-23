#!/usr/bin/env python3
"""Run one semantic Hydra reconstruction in an isolated process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from daaam.mapping.backends import HydraStaticMapBackend  # noqa: E402
from daaam.realtime.contracts import FrameValue  # noqa: E402
from run_realtime_mapping import (  # noqa: E402
    ReplayFrame,
    rebuild_static_map_with_semantics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _frame(record: dict) -> ReplayFrame:
    placeholder = Path(record["rgb_path"])
    return ReplayFrame(
        frame_index=int(record["frame_index"]),
        sensor_time_ns=int(record["sensor_time_ns"]),
        rgb_path=placeholder,
        right_path=placeholder,
        depth_path=placeholder,
        confidence_path=placeholder,
        consistency_path=placeholder,
        depth_metadata_path=placeholder,
        world_T_camera=np.asarray(record["world_T_camera"], dtype=np.float64),
        intrinsics=np.asarray(record["intrinsics"], dtype=np.float64),
        value=FrameValue.ROUTINE,
    )


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.resolve().read_text())
    if plan.get("schema") != "daaam.hydra_semantic_postpass_plan.v1":
        raise ValueError("unsupported Hydra semantic postpass plan")
    frames = [_frame(record) for record in plan["frames"]]
    if not frames:
        raise ValueError("Hydra semantic postpass plan contains no frames")
    completed_indices = {frame.frame_index for frame in frames}
    first_rgb = cv2.imread(str(frames[0].rgb_path), cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise FileNotFoundError(frames[0].rgb_path)

    backend = HydraStaticMapBackend(
        plan["hydra_config_path"],
        plan["output_dir"],
        labelspace_path=plan.get("labelspace_path"),
        labelspace_colors=plan.get("labelspace_colors"),
        maximum_depth_m=float(plan["maximum_depth_m"]),
    )
    started = time.monotonic()
    try:
        backend.initialize(
            first_rgb.shape[1],
            first_rgb.shape[0],
            frames[0].intrinsics,
        )
        report = rebuild_static_map_with_semantics(
            backend,
            Path(plan["run_dir"]),
            frames,
            completed_indices,
            Path(plan["semantic_label_dir"]),
            plan["label_run_configuration_sha256"],
        )
        backend.finalize()
        report["backend_stats"] = backend.stats()
        report["isolated_process"] = True
        report["elapsed_seconds"] = time.monotonic() - started
        report["postpass_pid"] = os.getpid()
    finally:
        backend.close(finalize=False)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.report)


if __name__ == "__main__":
    main()
