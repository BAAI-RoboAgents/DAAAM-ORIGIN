#!/usr/bin/env python3
"""Strict Unitree head-RGB-D to Hydra semantic-map orchestration.

Stages:
  audit   - read-only source validation and an explicit missing-data report
  prepare - RGB-D conversion plus local_T_body @ body_T_camera pose composition
  map     - dynamic isolation, Hydra fusion, FastSAM/BotSort, and DAM postpass
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from prepare_unitree_head_rgbd_dataset import (  # noqa: E402
    audit_dataset,
    prepare_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--sequence", default="000000")
    parser.add_argument(
        "--stop-after",
        choices=("audit", "prepare", "map"),
        default="audit",
    )
    parser.add_argument("--camera-calibration", type=Path)
    parser.add_argument("--time-contract", type=Path)
    parser.add_argument(
        "--rate-hz",
        type=float,
        help=(
            "Required for map. This is a scheduling limit, not a quality knob; "
            "the wrapper never selects or benchmarks a rate automatically."
        ),
    )
    parser.add_argument(
        "--accept-local-world-frame",
        action="store_true",
        help=(
            "Explicitly accept the recorded local_T_body trajectory as the map "
            "world frame. The capture does not contain map_T_body."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--hydra-config",
        type=Path,
        default=REPOSITORY_ROOT / "config/hydra_g1_8m_12cm.yaml",
    )
    parser.add_argument(
        "--semantic-config",
        type=Path,
        default=REPOSITORY_ROOT / "config/pipeline_config_realtime.yaml",
    )
    parser.add_argument(
        "--quality-config",
        type=Path,
        default=REPOSITORY_ROOT / "config/unitree_rgbd_quality_gates.yaml",
    )
    parser.add_argument("--hydra-labelspace-path", type=Path)
    parser.add_argument("--hydra-labelspace-colors", type=Path)
    parser.add_argument("--segmentation-rate-hz", type=float, default=5.0)
    parser.add_argument("--semantic-frontend-rate-hz", type=float, default=10.0)
    parser.add_argument("--semantic-minimum-observations", type=int, default=5)
    parser.add_argument("--entity-merge-distance-m", type=float, default=0.50)
    parser.add_argument(
        "--object-binding-maximum-center-distance-m",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--object-binding-maximum-aabb-gap-m",
        type=float,
        default=0.15,
    )
    return parser.parse_args()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require_mapping_arguments(args: argparse.Namespace) -> None:
    missing = []
    if args.camera_calibration is None:
        missing.append("--camera-calibration")
    if args.time_contract is None:
        missing.append("--time-contract")
    if args.rate_hz is None:
        missing.append("--rate-hz")
    if not args.accept_local_world_frame:
        missing.append("--accept-local-world-frame")
    if missing:
        raise ValueError(
            "Unitree map stage requires explicit, non-inferred inputs: "
            + ", ".join(missing)
        )
    if args.rate_hz <= 0.0:
        raise ValueError("--rate-hz must be positive")


def main() -> None:
    args = parse_args()
    source = args.src.resolve()
    run_dir = args.run_dir.resolve()
    audit_path = run_dir / "00_audit/unitree_data_audit.json"
    prepared = run_dir / "01_prepared_head_rgbd"
    map_run_dir = run_dir / "02_semantic_map"

    audit = audit_dataset(
        source,
        args.sequence,
        camera_calibration_path=args.camera_calibration,
        time_contract_path=args.time_contract,
    )
    write_json(audit_path, audit)
    if args.stop_after == "audit":
        print(json.dumps(audit, indent=2, allow_nan=False))
        return
    if not audit["mapping_ready"]:
        blockers = "; ".join(
            f"{item['code']}: {item['message']}"
            for item in audit["summary"]["hard_blockers"]
        )
        raise RuntimeError(
            f"Unitree pipeline is blocked; see {audit_path}: {blockers}"
        )
    if args.camera_calibration is None or args.time_contract is None:
        raise AssertionError("mapping-ready audit requires both explicit contracts")

    preparation = prepare_dataset(
        source,
        prepared,
        args.sequence,
        camera_calibration_path=args.camera_calibration,
        time_contract_path=args.time_contract,
    )
    if args.stop_after == "prepare":
        print(json.dumps(preparation, indent=2, allow_nan=False))
        return

    require_mapping_arguments(args)
    metadata = json.loads((prepared / "tick_index.json").read_text())
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/run_realtime_mapping.py"),
        "--dataset",
        str(prepared),
        "--run-dir",
        str(map_run_dir),
        "--depth-backend",
        "precomputed",
        "--maximum-depth-m",
        str(float(metadata["recommended_max_depth_m"])),
        "--rate-hz",
        str(args.rate_hz),
        "--stop-after",
        "global",
        "--static-map-backend",
        "hydra",
        "--hydra-config-path",
        str(args.hydra_config.resolve()),
        "--semantic-mode",
        "dam",
        "--semantic-config",
        str(args.semantic_config.resolve()),
        "--quality-config",
        str(args.quality_config.resolve()),
        "--segmentation-rate-hz",
        str(args.segmentation_rate_hz),
        "--semantic-frontend-rate-hz",
        str(args.semantic_frontend_rate_hz),
        "--semantic-minimum-observations",
        str(args.semantic_minimum_observations),
        "--entity-merge-distance-m",
        str(args.entity_merge_distance_m),
        "--object-binding-maximum-center-distance-m",
        str(args.object_binding_maximum_center_distance_m),
        "--object-binding-maximum-aabb-gap-m",
        str(args.object_binding_maximum_aabb_gap_m),
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.resume:
        command.append("--resume")
    if args.max_frames is not None:
        command.extend(("--max-frames", str(args.max_frames)))
    if args.hydra_labelspace_path is not None:
        command.extend(
            (
                "--hydra-labelspace-path",
                str(args.hydra_labelspace_path.resolve()),
            )
        )
    if args.hydra_labelspace_colors is not None:
        command.extend(
            (
                "--hydra-labelspace-colors",
                str(args.hydra_labelspace_colors.resolve()),
            )
        )
    orchestration = {
        "schema": "unitree_mapping_orchestration_v1",
        "source": str(source),
        "audit": str(audit_path),
        "prepared_dataset": str(prepared),
        "map_run_dir": str(map_run_dir),
        "world_frame": "local",
        "local_world_frame_explicitly_accepted": True,
        "command": command,
    }
    write_json(run_dir / "unitree_mapping_run.json", orchestration)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


if __name__ == "__main__":
    main()
