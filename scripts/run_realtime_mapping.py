#!/usr/bin/env python3
"""Replay an absolute-time RGB-D sequence through the bounded realtime map core."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from run_stereo_mapping import validate_time_contract  # noqa: E402
from run_foundation_stereo_depth import resolve_inference_profile  # noqa: E402
from daaam.depth import (  # noqa: E402
    DepthPropagationConfig,
    SubprocessDepthBackend,
    propagate_depth,
)
from daaam.mapping.dynamic_layer import (  # noqa: E402
    DynamicLayer,
    DynamicLayerConfig,
    ObjectObservation,
)
from daaam.mapping.backends import HydraStaticMapBackend  # noqa: E402
from daaam.mapping.fusion import isolate_static_depth  # noqa: E402
from daaam.mapping.motion import MotionConfig, estimate_motion_masks  # noqa: E402
from daaam.mapping.paths import PathObservation, PathRepository  # noqa: E402
from daaam.mapping.submaps import SubmapManager  # noqa: E402
from daaam.memory import MapMemory  # noqa: E402
from daaam.quality import QualityGateConfig, QualityGateRunner  # noqa: E402
from daaam.realtime.checkpoint import RealtimeCheckpoint  # noqa: E402
from daaam.realtime.contracts import (  # noqa: E402
    FrameValue,
    MessageKey,
    PoseEstimate,
    RealtimeEnvelope,
)
from daaam.realtime.manifest import (  # noqa: E402
    build_run_manifest,
    sha256_file,
    write_run_manifest,
)
from daaam.realtime.scheduler import MultiRateScheduler, StageSpec  # noqa: E402
from daaam.realtime.gpu import SharedGpuCoordinator  # noqa: E402
from daaam.slam.backend import PoseBackendConfig, PoseInputValidator  # noqa: E402


STAGES = ("pose", "depth", "dynamic", "fusion", "global")
SEMANTIC_STAGE = "semantic_frontend"
LEGACY_STAGE_ALIASES = {"tracking": "dynamic"}


@dataclass(frozen=True)
class ReplayFrame:
    frame_index: int
    sensor_time_ns: int
    rgb_path: Path
    right_path: Path
    depth_path: Path
    confidence_path: Path
    consistency_path: Path
    depth_metadata_path: Path
    world_T_camera: np.ndarray
    intrinsics: np.ndarray
    value: FrameValue


@dataclass(frozen=True)
class DepthFrame:
    source: ReplayFrame
    rgb_image: np.ndarray
    depth_m: np.ndarray
    confidence: np.ndarray
    has_stereo_confidence: bool


@dataclass(frozen=True)
class TrackedFrame:
    source: ReplayFrame
    rgb_image: np.ndarray
    depth_m: np.ndarray
    confidence: np.ndarray
    dynamic_mask: np.ndarray
    unknown_mask: np.ndarray
    motion_metrics: dict


@dataclass(frozen=True)
class FusionFrame:
    tracked: TrackedFrame
    static_depth_m: np.ndarray
    static_confidence: np.ndarray
    fusion_metrics: dict


class FaultInjector:
    def __init__(
        self,
        stage: Optional[str],
        *,
        delay_ms: float,
        every: int,
        error_every: int,
    ) -> None:
        self.stage = stage
        self.delay_ms = delay_ms
        self.every = every
        self.error_every = error_every
        self.counts: dict[str, int] = {}

    def wrap(self, stage: str, handler):
        def wrapped(envelope):
            count = self.counts.get(stage, 0) + 1
            self.counts[stage] = count
            if self.stage == stage:
                if self.every > 0 and count % self.every == 0 and self.delay_ms > 0:
                    time.sleep(self.delay_ms / 1000.0)
                if self.error_every > 0 and count % self.error_every == 0:
                    raise RuntimeError(f"injected_{stage}_failure")
            return handler(envelope)

        return wrapped


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--stop-after",
        choices=STAGES + tuple(LEGACY_STAGE_ALIASES),
        default="fusion",
    )
    parser.add_argument(
        "--depth-backend",
        choices=("precomputed", "foundation-worker"),
        default="precomputed",
    )
    parser.add_argument("--foundation-stereo-env", default="foundation_stereo")
    parser.add_argument("--foundation-stereo-python", type=Path)
    parser.add_argument(
        "--foundation-stereo-root",
        type=Path,
        default=REPOSITORY_ROOT / "third_party" / "FoundationStereo",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--depth-profile", choices=("online", "refine", "custom"), default="online"
    )
    parser.add_argument("--depth-valid-iters", type=int)
    parser.add_argument("--depth-scale", type=float)
    parser.add_argument("--depth-precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument(
        "--depth-confidence-mode",
        choices=("left-right", "validity"),
        default="left-right",
    )
    parser.add_argument(
        "--depth-lr-interval",
        type=int,
        default=3,
        help="Run full left-right confidence validation every N depth frames.",
    )
    parser.add_argument("--depth-torch-compile", action="store_true")
    parser.add_argument("--depth-startup-timeout-s", type=float, default=120.0)
    parser.add_argument("--depth-request-timeout-s", type=float, default=2.0)
    parser.add_argument("--depth-maximum-retries", type=int, default=1)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--no-throttle", action="store_true")
    parser.add_argument(
        "--allow-source-bursts",
        action="store_true",
        help=(
            "Preserve scaled sub-period source bursts instead of treating --rate-hz "
            "as a maximum dispatch rate. Absolute frame timestamps are unchanged."
        ),
    )
    parser.add_argument(
        "--stage-rate-multiplier",
        type=float,
        default=1.0,
        help=(
            "Development-only multiplier for worker-stage service-rate caps. "
            "The default preserves the documented per-stage rates."
        ),
    )
    parser.add_argument("--queue-capacity", type=int, default=8)
    parser.add_argument("--stage-deadline-ms", type=float)
    parser.add_argument("--drain-timeout-s", type=float, default=30.0)
    parser.add_argument("--pose-position-std-m", type=float, default=0.05)
    parser.add_argument("--pose-rotation-std-deg", type=float, default=2.0)
    parser.add_argument("--minimum-dynamic-pixels", type=int, default=40)
    parser.add_argument("--motion-analysis-width", type=int, default=160)
    parser.add_argument("--submap-frames", type=int, default=30)
    parser.add_argument(
        "--static-map-backend",
        choices=("submaps", "hydra"),
        default="submaps",
    )
    parser.add_argument("--hydra-config-path", type=Path)
    parser.add_argument("--hydra-labelspace-path", type=Path)
    parser.add_argument("--hydra-labelspace-colors", type=Path)
    parser.add_argument(
        "--semantic-mode",
        choices=("disabled", "frontend", "dam"),
        default="disabled",
        help=(
            "Run the real FastSAM/BotSort side branch, optionally with asynchronous "
            "DAM grounding. Geometry never waits for this branch."
        ),
    )
    parser.add_argument(
        "--semantic-config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "pipeline_config_realtime.yaml",
    )
    parser.add_argument("--segmentation-rate-hz", type=float, default=5.0)
    parser.add_argument("--semantic-frontend-rate-hz", type=float, default=10.0)
    parser.add_argument("--semantic-queue-capacity", type=int, default=2)
    parser.add_argument("--semantic-minimum-observations", type=int, default=5)
    parser.add_argument("--semantic-drain-timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--gpu-sharing-mode",
        choices=("staggered", "unmanaged"),
        default="staggered",
        help=(
            "Serialize CUDA models and defer DAM until the realtime frontend is "
            "idle. Use unmanaged only for development on independently assigned GPUs."
        ),
    )
    parser.add_argument("--dam-minimum-gpu-idle-s", type=float, default=1.0)
    parser.add_argument("--no-write-fusion-products", action="store_true")
    parser.add_argument(
        "--fault-stage",
        choices=STAGES + (SEMANTIC_STAGE,) + tuple(LEGACY_STAGE_ALIASES),
    )
    parser.add_argument("--fault-delay-ms", type=float, default=0.0)
    parser.add_argument("--fault-every", type=int, default=1)
    parser.add_argument("--fault-error-every", type=int, default=0)
    parser.add_argument(
        "--quality-config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "realtime_quality_gates.yaml",
    )
    parser.add_argument("--quality-report-only", action="store_true")
    parser.add_argument("--map-metrics-json", type=Path)
    return parser.parse_args()


def read_poses(dataset: Path) -> list[np.ndarray]:
    poses = []
    for line in (dataset / "pose" / "poses.txt").read_text().splitlines():
        if not line.strip():
            continue
        values = np.asarray([float(value) for value in line.split()], dtype=np.float64)
        if values.size != 16:
            raise ValueError("Every pose row must contain a 4x4 matrix")
        poses.append(values.reshape(4, 4))
    return poses


def resolve_environment_python(
    environment_name: str,
    explicit_python: Optional[Path] = None,
) -> Path:
    if explicit_python is not None:
        executable = explicit_python.expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(executable)
        return executable
    result = subprocess.run(
        ["conda", "env", "list", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to list Conda environments: {result.stderr.strip()}")
    try:
        environments = json.loads(result.stdout)["envs"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("Conda environment list is invalid") from error
    for environment in environments:
        prefix = Path(environment)
        if prefix.name == environment_name or str(prefix) == environment_name:
            executable = prefix / "bin" / "python"
            if executable.is_file():
                return executable.resolve()
    raise FileNotFoundError(f"Conda environment not found: {environment_name}")


def load_intrinsics(dataset: Path, metadata: dict) -> np.ndarray:
    camera_path = dataset / "camera_info.json"
    camera = json.loads(camera_path.read_text()) if camera_path.is_file() else metadata
    if "intrinsics" in camera:
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    else:
        intrinsics = np.array(
            [
                [float(camera["fx"]), 0.0, float(camera["cx"])],
                [0.0, float(camera.get("fy", camera["fx"])), float(camera["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if intrinsics.shape != (3, 3) or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("Dataset camera intrinsics are invalid")
    return intrinsics


def frame_value(reason: str, *, is_last: bool) -> FrameValue:
    mapping = {
        "strict_duplicate": FrameValue.STRICT_DUPLICATE,
        "watchdog": FrameValue.WATCHDOG,
        "pose_motion": FrameValue.POSE_MOTION,
        "image_event_at_static_pose": FrameValue.IMAGE_EVENT_AT_STATIC_POSE,
        "loop_candidate": FrameValue.LOOP_CANDIDATE,
        "initial_frame": FrameValue.WATCHDOG,
        "final_frame": FrameValue.WATCHDOG,
    }
    return mapping.get(reason, FrameValue.WATCHDOG if is_last else FrameValue.ROUTINE)


def scheduled_confidence_mode(
    frame_index: int,
    *,
    configured_mode: str,
    left_right_interval: int,
) -> str:
    if configured_mode not in {"left-right", "validity"}:
        raise ValueError("Unsupported depth confidence mode")
    if left_right_interval <= 0:
        raise ValueError("left_right_interval must be positive")
    if configured_mode == "left-right" and frame_index % left_right_interval == 0:
        return "left-right"
    return "validity"


def build_frames(dataset: Path, metadata: dict, poses: list[np.ndarray]) -> list[ReplayFrame]:
    intrinsics = load_intrinsics(dataset, metadata)
    output = []
    for index, frame in enumerate(metadata["frames"]):
        pose_row = int(frame["pose_row"])
        frame_index = int(frame["idx"])
        output.append(
            ReplayFrame(
                frame_index=frame_index,
                sensor_time_ns=int(frame["sensor_time_ns"]),
                rgb_path=Path(frame["cam0"]),
                right_path=Path(frame["cam1"]),
                depth_path=dataset / "depth" / f"{frame_index:08d}.png",
                confidence_path=dataset
                / "depth_confidence"
                / f"{frame_index:08d}.png",
                consistency_path=dataset
                / "depth_consistency"
                / f"{frame_index:08d}.png",
                depth_metadata_path=dataset
                / "depth_metadata"
                / f"{frame_index:08d}.json",
                world_T_camera=poses[pose_row],
                intrinsics=intrinsics,
                value=frame_value(
                    str(frame.get("selection_reason", "routine")),
                    is_last=index == len(metadata["frames"]) - 1,
                ),
            )
        )
    return output


def load_precomputed_depth_provenance(dataset: Path) -> Optional[dict]:
    """Resolve the report that produced a precomputed metric-depth dataset."""

    for name in ("foundation_stereo_run.json", "foundation_stereo_nominal_run.json"):
        path = dataset / name
        if not path.is_file():
            continue
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid precomputed depth report: {path}") from error
        profile_value = report.get("profile")
        if isinstance(profile_value, dict):
            profile_name = profile_value.get("name")
        else:
            profile_name = profile_value
        checkpoint_value = report.get("checkpoint")
        checkpoint_path = (
            Path(checkpoint_value).expanduser().resolve()
            if checkpoint_value
            else None
        )
        return {
            "report": str(path.resolve()),
            "report_sha256": sha256_file(path),
            "profile": profile_name,
            "valid_iters": report.get("valid_iters"),
            "scale": report.get("scale"),
            "precision": report.get("precision"),
            "confidence_mode": report.get("confidence_mode"),
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "checkpoint_sha256": (
                sha256_file(checkpoint_path)
                if checkpoint_path is not None and checkpoint_path.is_file()
                else None
            ),
            "processed": report.get("processed"),
            "failed": report.get("failed"),
        }
    return None


def load_semantic_model_provenance(
    config_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict:
    """Resolve semantic model artifacts without initializing CUDA runtimes."""

    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid semantic pipeline config: {config_path}") from error
    if not isinstance(config, dict):
        raise ValueError("Semantic pipeline config must contain a mapping")

    def model_artifact(value: Optional[str], *, checkpoint_relative: bool) -> dict:
        if not value:
            return {"configured": value, "path": None, "sha256": None}
        configured = Path(value)
        if configured.is_absolute():
            path = configured
        elif checkpoint_relative and configured.parts[:1] != ("checkpoints",):
            path = repository_root / "checkpoints" / configured
        else:
            path = repository_root / configured
        path = path.resolve()
        return {
            "configured": value,
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }

    segmentation = config.get("segmentation", {})
    tracking = config.get("tracking", {})
    workers = config.get("workers", {})
    dam_config = workers.get("dam_grounding_config", {})
    dam_model = dam_config.get("dam_model_path")
    cache_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path.home() / ".cache" / "huggingface" / "hub",
        )
    )
    dam_revision_path = (
        cache_root
        / f"models--{str(dam_model).replace('/', '--')}"
        / "refs"
        / "main"
        if dam_model
        else None
    )
    dam_revision = (
        dam_revision_path.read_text().strip()
        if dam_revision_path is not None and dam_revision_path.is_file()
        else None
    )
    semantic_labelspace = model_artifact(
        config.get("semantic_config_path"), checkpoint_relative=False
    )
    labelspace_colors = model_artifact(
        config.get("labelspace_colors_path"), checkpoint_relative=False
    )
    return {
        "pipeline_config": str(config_path.resolve()),
        "pipeline_config_sha256": sha256_file(config_path),
        "fastsam": model_artifact(
            segmentation.get("model_name"), checkpoint_relative=True
        ),
        "botsort_reid": model_artifact(
            tracking.get("reid_weights"), checkpoint_relative=True
        ),
        "dam": {
            "model_id": dam_model,
            "cached_revision": dam_revision,
        },
        "semantic_labelspace": semantic_labelspace,
        "labelspace_colors": labelspace_colors,
    }


class ReplayEngine:
    def __init__(
        self,
        run_dir: Path,
        checkpoint: RealtimeCheckpoint,
        *,
        pose_position_std_m: float,
        pose_rotation_std_deg: float,
        minimum_dynamic_pixels: int,
        motion_analysis_width: int,
        submap_frames: int,
        write_fusion_products: bool,
        depth_worker: Optional[SubprocessDepthBackend] = None,
        stereo_fx: Optional[float] = None,
        stereo_baseline_m: Optional[float] = None,
        maximum_depth_m: float = 20.0,
        depth_confidence_mode: str = "left-right",
        depth_lr_interval: int = 3,
        static_map_backend: Optional[HydraStaticMapBackend] = None,
        checkpoint_state: Optional[dict] = None,
    ) -> None:
        self.run_dir = run_dir
        self.checkpoint = checkpoint
        self.write_fusion_products = write_fusion_products
        self.depth_worker = depth_worker
        self.stereo_fx = stereo_fx
        self.stereo_baseline_m = stereo_baseline_m
        self.maximum_depth_m = maximum_depth_m
        if depth_confidence_mode not in {"left-right", "validity"}:
            raise ValueError("Unsupported depth confidence mode")
        if depth_lr_interval <= 0:
            raise ValueError("depth_lr_interval must be positive")
        self.depth_confidence_mode = depth_confidence_mode
        self.depth_lr_interval = depth_lr_interval
        self.static_map_backend = static_map_backend
        self.semantic_label_provider: Optional[Callable[[int], Optional[np.ndarray]]] = None
        position_variance = pose_position_std_m**2
        rotation_variance = math.radians(pose_rotation_std_deg) ** 2
        self.pose_covariance = np.diag(
            [position_variance] * 3 + [rotation_variance] * 3
        )
        self.pose_validator = PoseInputValidator(
            PoseBackendConfig(
                maximum_gap_s=2.0,
                maximum_clock_jump_s=30.0,
                maximum_position_std_m=max(0.1, pose_position_std_m * 5),
                maximum_rotation_std_deg=max(5.0, pose_rotation_std_deg * 5),
            )
        )
        self.motion_config = MotionConfig(
            minimum_dynamic_pixels=minimum_dynamic_pixels
        )
        if motion_analysis_width < 64:
            raise ValueError("motion_analysis_width must be at least 64 pixels")
        self.motion_analysis_width = motion_analysis_width
        self.dynamic_config = DynamicLayerConfig()
        self.state_lock = threading.RLock()
        if checkpoint_state:
            self.dynamic_layer = DynamicLayer.from_snapshot(
                checkpoint_state.get("dynamic_layer", {}), self.dynamic_config
            )
            self.submaps = SubmapManager.from_snapshot(
                checkpoint_state.get("submaps", {}), maximum_frames=submap_frames
            )
            self.path_repository = PathRepository.from_snapshot(
                checkpoint_state.get("paths", {})
            )
            path_buffer = checkpoint_state.get("path_buffer", {})
            self.path_times_ns = [
                int(value) for value in path_buffer.get("sensor_times_ns", [])
            ]
            self.path_points_m = [
                np.asarray(value, dtype=np.float64)
                for value in path_buffer.get("points_m", [])
            ]
        else:
            self.dynamic_layer = DynamicLayer(self.dynamic_config)
            self.submaps = SubmapManager(maximum_frames=submap_frames)
            self.path_repository = PathRepository()
            self.path_times_ns = []
            self.path_points_m = []
        self.path_segment_frames = submap_frames
        self.previous: Optional[DepthFrame] = None
        self.previous_depth_quality: Optional[DepthFrame] = None
        self.pose_translation_steps = []
        self.pose_rotation_steps = []
        self.previous_pose: Optional[np.ndarray] = None
        self.depth_valid_ratios = []
        self.stereo_consistency = []
        self.depth_frames_evaluated = 0
        self.left_right_verified_frames = 0
        self.depth_peak_cuda_memory_bytes = 0
        self.depth_peak_worker_rss_bytes = 0
        self.temporal_agreements = []
        self.motion_unknown_ratios = []
        self.dynamic_ratios = []
        self.contamination_rates = []
        self.frames_by_stage = {stage: 0 for stage in STAGES}
        self.memory = MapMemory(run_dir / "map_memory.sqlite3")
        try:
            self.memory.create_session("replay", time.time_ns(), canonical=True)
        except sqlite3.IntegrityError:
            pass

    def initialize_previous(self, frame: ReplayFrame) -> None:
        try:
            if self.depth_worker is not None:
                frame = self._generated_depth_frame(frame)
            payload = self._load_depth(frame, record_metrics=False)
        except Exception:
            return
        self.previous = payload
        self.previous_depth_quality = payload
        self.previous_pose = frame.world_T_camera.copy()

    def pose(self, envelope: RealtimeEnvelope) -> RealtimeEnvelope:
        frame: ReplayFrame = envelope.payload
        estimate = PoseEstimate(
            frame.sensor_time_ns,
            frame.world_T_camera,
            self.pose_covariance,
            "dataset_pose_prior",
        )
        validation = self.pose_validator.validate(estimate, calibration_revision=0)
        if not validation.accepted:
            raise ValueError(f"pose rejected: {validation.reason}")
        with self.state_lock:
            if self.previous_pose is not None:
                relative = np.linalg.inv(self.previous_pose) @ frame.world_T_camera
                self.pose_translation_steps.append(
                    float(np.linalg.norm(relative[:3, 3]))
                )
                rotation_trace = np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
                self.pose_rotation_steps.append(float(np.degrees(np.arccos(rotation_trace))))
            self.previous_pose = frame.world_T_camera.copy()
            self.frames_by_stage["pose"] += 1
        return envelope

    def _load_depth(
        self,
        frame: ReplayFrame,
        *,
        record_metrics: bool = True,
    ) -> DepthFrame:
        rgb_bgr = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None:
            raise FileNotFoundError(f"RGB frame is missing: {frame.rgb_path}")
        if depth_raw is None:
            raise FileNotFoundError(f"Depth frame is missing: {frame.depth_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth = depth_raw.astype(np.float32) / 1000.0
        if depth.shape != rgb.shape[:2]:
            raise ValueError("RGB and depth dimensions do not match")
        depth_metadata = None
        if frame.depth_metadata_path.is_file():
            try:
                depth_metadata = json.loads(frame.depth_metadata_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid depth metadata: {frame.depth_metadata_path}"
                ) from error
            if int(depth_metadata.get("sensor_time_ns", -1)) != frame.sensor_time_ns:
                raise ValueError("Depth metadata timestamp does not match the frame")
        confidence_raw = (
            cv2.imread(str(frame.confidence_path), cv2.IMREAD_UNCHANGED)
            if frame.confidence_path.is_file()
            else None
        )
        if depth_metadata is not None:
            left_right_verified = bool(
                depth_metadata.get("left_right_verified", False)
            ) and depth_metadata.get("confidence_mode") == "left-right"
            consistency_raw = (
                cv2.imread(str(frame.consistency_path), cv2.IMREAD_UNCHANGED)
                if left_right_verified and frame.consistency_path.is_file()
                else None
            )
        else:
            consistency_raw = (
                cv2.imread(str(frame.consistency_path), cv2.IMREAD_UNCHANGED)
                if frame.consistency_path.is_file()
                else None
            )
            left_right_verified = consistency_raw is not None
        has_confidence = (
            confidence_raw is not None
            and consistency_raw is not None
            and left_right_verified
        )
        if confidence_raw is None:
            confidence = (depth > 0.0).astype(np.float32)
        else:
            confidence = confidence_raw.astype(np.float32) / 255.0
        if confidence.shape != depth.shape:
            raise ValueError("Depth confidence dimensions do not match")
        if record_metrics:
            with self.state_lock:
                self.depth_frames_evaluated += 1
                self.depth_valid_ratios.append(float(np.mean(depth > 0.0)))
                if left_right_verified and consistency_raw is not None:
                    self.left_right_verified_frames += 1
                    if (
                        depth_metadata is not None
                        and depth_metadata.get("left_right_consistency") is not None
                    ):
                        consistency = float(
                            depth_metadata["left_right_consistency"]
                        )
                    else:
                        consistent_pixels = int(
                            np.count_nonzero(
                                consistency_raw.astype(np.float32) > 0.0
                            )
                        )
                        consistency = min(
                            1.0,
                            float(
                                consistent_pixels
                                / max(1, int(np.count_nonzero(depth > 0.0)))
                            ),
                        )
                    self.stereo_consistency.append(consistency)
                if depth_metadata is not None:
                    self.depth_peak_cuda_memory_bytes = max(
                        self.depth_peak_cuda_memory_bytes,
                        int(depth_metadata.get("peak_cuda_memory_bytes", 0)),
                    )
                    self.depth_peak_worker_rss_bytes = max(
                        self.depth_peak_worker_rss_bytes,
                        int(depth_metadata.get("peak_worker_rss_bytes", 0)),
                    )
                self.frames_by_stage["depth"] += 1
        return DepthFrame(frame, rgb, depth, confidence, has_confidence)

    def _generated_depth_frame(self, frame: ReplayFrame) -> ReplayFrame:
        root = self.run_dir / "generated_depth"
        name = f"{frame.frame_index:08d}.png"
        return replace(
            frame,
            depth_path=root / "depth" / name,
            confidence_path=root / "depth_confidence" / name,
            consistency_path=root / "depth_consistency" / name,
            depth_metadata_path=root
            / "depth_metadata"
            / f"{frame.frame_index:08d}.json",
        )

    def materialize_depth(self, frame: ReplayFrame) -> ReplayFrame:
        if self.depth_worker is None:
            return frame
        generated = self._generated_depth_frame(frame)
        if all(
            path.is_file()
            for path in (
                generated.depth_path,
                generated.confidence_path,
                generated.consistency_path,
                generated.depth_metadata_path,
            )
        ):
            return generated
        if self.stereo_fx is None or self.stereo_baseline_m is None:
            raise RuntimeError("FoundationStereo worker geometry is not configured")
        confidence_mode = scheduled_confidence_mode(
            frame.frame_index,
            configured_mode=self.depth_confidence_mode,
            left_right_interval=self.depth_lr_interval,
        )
        self.depth_worker.infer(
            {
                "sensor_time_ns": frame.sensor_time_ns,
                "frame_idx": frame.frame_index,
                "left_path": str(frame.rgb_path),
                "right_path": str(frame.right_path),
                "fx": self.stereo_fx,
                "baseline_m": self.stereo_baseline_m,
                "maximum_depth_m": self.maximum_depth_m,
                "output_dir": str(self.run_dir / "generated_depth"),
                "confidence_mode": confidence_mode,
            }
        )
        return generated

    def depth(self, envelope: RealtimeEnvelope) -> RealtimeEnvelope:
        frame: ReplayFrame = envelope.payload
        frame = self.materialize_depth(frame)
        payload = self._load_depth(frame)
        with self.state_lock:
            previous_quality = self.previous_depth_quality
        if previous_quality is not None:
            self._temporal_agreement(previous_quality, payload)
        with self.state_lock:
            self.previous_depth_quality = payload
        return RealtimeEnvelope(
            envelope.key,
            payload,
            envelope.value,
            source="precomputed_depth",
            created_monotonic_ns=envelope.created_monotonic_ns,
            deadline_monotonic_ns=envelope.deadline_monotonic_ns,
            trace_id=envelope.trace_id,
        )

    def _temporal_agreement(self, previous: DepthFrame, current: DepthFrame) -> None:
        height, width = current.depth_m.shape
        analysis_width = min(width, 160)
        if analysis_width < width:
            scale = analysis_width / width
            analysis_size = (analysis_width, max(48, int(round(height * scale))))
            previous_depth = cv2.resize(
                previous.depth_m, analysis_size, interpolation=cv2.INTER_NEAREST
            )
            previous_confidence = cv2.resize(
                previous.confidence, analysis_size, interpolation=cv2.INTER_NEAREST
            )
            current_depth = cv2.resize(
                current.depth_m, analysis_size, interpolation=cv2.INTER_NEAREST
            )
            intrinsics = current.source.intrinsics.copy()
            intrinsics[0, :] *= analysis_size[0] / width
            intrinsics[1, :] *= analysis_size[1] / height
        else:
            previous_depth = previous.depth_m
            previous_confidence = previous.confidence
            current_depth = current.depth_m
            intrinsics = current.source.intrinsics
        try:
            propagated = propagate_depth(
                previous_depth,
                previous_confidence,
                intrinsics,
                previous.source.world_T_camera,
                current.source.world_T_camera,
                source_time_ns=previous.source.sensor_time_ns,
                target_time_ns=current.source.sensor_time_ns,
                config=DepthPropagationConfig(
                    maximum_age_s=2.0,
                    maximum_translation_m=1.0,
                    maximum_rotation_deg=30.0,
                    minimum_output_valid_ratio=0.01,
                ),
            )
        except ValueError:
            return
        overlap = (propagated.depth_m > 0.0) & (current_depth > 0.0)
        if np.any(overlap):
            error = np.abs(propagated.depth_m[overlap] - current_depth[overlap])
            tolerance = 0.05 + 0.05 * current_depth[overlap]
            self.temporal_agreements.append(float(np.mean(error <= tolerance)))

    def tracking(self, envelope: RealtimeEnvelope) -> RealtimeEnvelope:
        current: DepthFrame = envelope.payload
        with self.state_lock:
            previous = self.previous
        if previous is None:
            dynamic = np.zeros(current.depth_m.shape, dtype=bool)
            unknown = np.ones(current.depth_m.shape, dtype=bool)
            motion_metrics = {
                "reason": "first_frame_no_motion_baseline",
                "dynamic_ratio": 0.0,
                "unknown_ratio": 1.0,
            }
        else:
            try:
                height, width = current.depth_m.shape
                if width > self.motion_analysis_width:
                    scale = self.motion_analysis_width / width
                    analysis_size = (
                        self.motion_analysis_width,
                        max(48, int(round(height * scale))),
                    )
                    previous_rgb = cv2.resize(
                        previous.rgb_image, analysis_size, interpolation=cv2.INTER_AREA
                    )
                    current_rgb = cv2.resize(
                        current.rgb_image, analysis_size, interpolation=cv2.INTER_AREA
                    )
                    previous_depth = cv2.resize(
                        previous.depth_m, analysis_size, interpolation=cv2.INTER_NEAREST
                    )
                    analysis_intrinsics = current.source.intrinsics.copy()
                    analysis_intrinsics[0, :] *= analysis_size[0] / width
                    analysis_intrinsics[1, :] *= analysis_size[1] / height
                else:
                    scale = 1.0
                    analysis_size = (width, height)
                    previous_rgb = previous.rgb_image
                    current_rgb = current.rgb_image
                    previous_depth = previous.depth_m
                    analysis_intrinsics = current.source.intrinsics
                motion = estimate_motion_masks(
                    previous_rgb,
                    current_rgb,
                    previous_depth,
                    previous.source.world_T_camera,
                    current.source.world_T_camera,
                    analysis_intrinsics,
                    config=self.motion_config,
                )
                if analysis_size != (width, height):
                    dynamic = cv2.resize(
                        motion.dynamic_mask.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    unknown = cv2.resize(
                        motion.unknown_mask.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    residual_px = cv2.resize(
                        motion.residual_px,
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    ) / scale
                else:
                    dynamic = motion.dynamic_mask
                    unknown = motion.unknown_mask
                    residual_px = motion.residual_px
                motion_metrics = dict(motion.metrics)
                motion_metrics.update(
                    {
                        "analysis_width": analysis_size[0],
                        "analysis_height": analysis_size[1],
                        "dynamic_ratio": float(dynamic.mean()),
                        "unknown_ratio": float(unknown.mean()),
                    }
                )
            except Exception as error:
                dynamic = np.zeros(current.depth_m.shape, dtype=bool)
                unknown = np.ones(current.depth_m.shape, dtype=bool)
                motion_metrics = {
                    "reason": "motion_uncertain",
                    "error": repr(error),
                    "dynamic_ratio": 0.0,
                    "unknown_ratio": 1.0,
                }

            components, labels, stats, centroids = cv2.connectedComponentsWithStats(
                dynamic.astype(np.uint8), connectivity=8
            )
            for component in range(1, components):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if area < self.motion_config.minimum_dynamic_pixels:
                    continue
                mask = labels == component
                valid_depth = current.depth_m[mask]
                valid_depth = valid_depth[valid_depth > 0.0]
                if not len(valid_depth):
                    continue
                z = float(np.median(valid_depth))
                u, v = centroids[component]
                fx, fy = current.source.intrinsics[0, 0], current.source.intrinsics[1, 1]
                cx, cy = current.source.intrinsics[0, 2], current.source.intrinsics[1, 2]
                camera_point = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
                world_point = (
                    current.source.world_T_camera @ np.r_[camera_point, 1.0]
                )[:3]
                width_m = max(0.05, stats[component, cv2.CC_STAT_WIDTH] * z / fx)
                height_m = max(0.05, stats[component, cv2.CC_STAT_HEIGHT] * z / fy)
                try:
                    self.dynamic_layer.update(
                        ObjectObservation(
                            track_id=current.source.frame_index * 10000 + component,
                            entity_id=None,
                            sensor_time_ns=current.source.sensor_time_ns,
                            position_m=world_point,
                            dimensions_m=np.array([width_m, 0.2, height_m]),
                            position_covariance=np.eye(3) * 0.01,
                            semantic_probabilities={"dynamic object": 1.0},
                            motion_score=min(
                                1.0,
                                float(
                                    np.median(
                                        residual_px[mask]
                                        / self.motion_config.residual_threshold_px
                                    )
                                ),
                            ),
                        )
                    )
                except ValueError:
                    continue
        with self.state_lock:
            self.dynamic_layer.advance_time(current.source.sensor_time_ns)
            if previous is not None:
                self.motion_unknown_ratios.append(float(unknown.mean()))
                self.dynamic_ratios.append(float(dynamic.mean()))
            self.previous = current
            self.frames_by_stage["dynamic"] += 1
        payload = TrackedFrame(
            current.source,
            current.rgb_image,
            current.depth_m,
            current.confidence,
            dynamic,
            unknown,
            motion_metrics,
        )
        return RealtimeEnvelope(
            envelope.key,
            payload,
            envelope.value,
            source="motion_tracking",
            created_monotonic_ns=envelope.created_monotonic_ns,
            deadline_monotonic_ns=envelope.deadline_monotonic_ns,
            trace_id=envelope.trace_id,
        )

    def fusion(self, envelope: RealtimeEnvelope) -> RealtimeEnvelope:
        tracked: TrackedFrame = envelope.payload
        static = isolate_static_depth(
            tracked.depth_m,
            tracked.dynamic_mask,
            tracked.unknown_mask,
            confidence=tracked.confidence,
            minimum_confidence=0.05,
        )
        if self.write_fusion_products:
            name = f"{tracked.source.frame_index:08d}.png"
            depth_dir = self.run_dir / "static_depth"
            mask_dir = self.run_dir / "dynamic_masks"
            unknown_dir = self.run_dir / "unknown_masks"
            for directory in (depth_dir, mask_dir, unknown_dir):
                directory.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(depth_dir / name),
                np.rint(static.depth_m * 1000.0).astype(np.uint16),
            )
            cv2.imwrite(
                str(mask_dir / name), tracked.dynamic_mask.astype(np.uint8) * 255
            )
            cv2.imwrite(
                str(unknown_dir / name), tracked.unknown_mask.astype(np.uint8) * 255
            )
        with self.state_lock:
            self.contamination_rates.append(
                float(static.metrics["dynamic_contamination_rate"])
            )
            self.frames_by_stage["fusion"] += 1
        payload = FusionFrame(
            tracked, static.depth_m, static.confidence, static.metrics
        )
        return RealtimeEnvelope(
            envelope.key,
            payload,
            envelope.value,
            source="static_fusion_filter",
            created_monotonic_ns=envelope.created_monotonic_ns,
            deadline_monotonic_ns=envelope.deadline_monotonic_ns,
            trace_id=envelope.trace_id,
        )

    def global_map(self, envelope: RealtimeEnvelope) -> RealtimeEnvelope:
        payload: FusionFrame = envelope.payload
        frame = payload.tracked.source
        with self.state_lock:
            self.submaps.add_frame(frame.sensor_time_ns, frame.world_T_camera)
            self.path_times_ns.append(frame.sensor_time_ns)
            self.path_points_m.append(frame.world_T_camera[:3, 3].copy())
            if len(self.path_times_ns) >= self.path_segment_frames:
                self._commit_path_segment(final=False)
            self.frames_by_stage["global"] += 1
        if self.static_map_backend is not None:
            semantic_labels = (
                self.semantic_label_provider(frame.sensor_time_ns)
                if self.semantic_label_provider is not None
                else None
            )
            self.static_map_backend.integrate(
                sensor_time_ns=frame.sensor_time_ns,
                rgb_image=payload.tracked.rgb_image,
                static_depth_m=payload.static_depth_m,
                world_T_camera=frame.world_T_camera,
                semantic_labels=semantic_labels,
            )
        return envelope

    def _path_buffer_snapshot(self) -> dict:
        return {
            "sensor_times_ns": list(self.path_times_ns),
            "points_m": [point.tolist() for point in self.path_points_m],
        }

    def _commit_path_segment(self, *, final: bool) -> None:
        if len(self.path_times_ns) < 2:
            return
        observation = PathObservation(
            session_id="replay",
            sensor_times_ns=np.asarray(self.path_times_ns, dtype=np.int64),
            points_m=np.asarray(self.path_points_m, dtype=np.float64),
            map_revision=self.submaps.map_revision,
        )
        self.path_repository.add(observation)
        if final:
            self.path_times_ns = []
            self.path_points_m = []
        else:
            self.path_times_ns = [self.path_times_ns[-1]]
            self.path_points_m = [self.path_points_m[-1].copy()]

    def finalize_paths(self) -> None:
        with self.state_lock:
            self._commit_path_segment(final=True)
            snapshot = self.path_repository.snapshot()
            output = self.run_dir / "canonical_paths.json"
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
            temporary.replace(output)
            self.checkpoint.update_mapping_state(
                map_revision=self.submaps.map_revision,
                dynamic_layer=self.dynamic_layer.snapshot(),
                submaps=self.submaps.snapshot(),
                paths=snapshot,
                path_buffer=self._path_buffer_snapshot(),
            )

    def finalize_static_map(self) -> None:
        if self.static_map_backend is not None:
            self.static_map_backend.finalize()

    def static_map_stats(self) -> dict:
        if self.static_map_backend is None:
            return {"backend": "submaps", "submaps": len(self.submaps.submaps)}
        return self.static_map_backend.stats()

    def terminal(self, stage: str, handler):
        def wrapped(envelope):
            output = handler(envelope)
            payload = output.payload if output is not None else envelope.payload
            if isinstance(payload, FusionFrame):
                frame = payload.tracked.source
            elif isinstance(payload, TrackedFrame):
                frame = payload.source
            elif isinstance(payload, DepthFrame):
                frame = payload.source
            else:
                frame = payload
            with self.state_lock:
                self.checkpoint.mark_completed(
                    frame.frame_index,
                    frame.sensor_time_ns,
                    map_revision=self.submaps.map_revision,
                    dynamic_layer=self.dynamic_layer.snapshot(),
                    submaps=self.submaps.snapshot(),
                    paths=self.path_repository.snapshot(),
                    path_buffer=self._path_buffer_snapshot(),
                )
            return output

        return wrapped

    def quality_context(self, scheduler_report: dict, *, map_metrics: Optional[dict]) -> dict:
        maximum_position_std = float(
            np.sqrt(np.max(np.diag(self.pose_covariance)[:3]))
        )
        runtime = json.loads(json.dumps(scheduler_report))
        runtime["resources"] = {
            "depth_peak_cuda_memory_bytes": self.depth_peak_cuda_memory_bytes,
            "depth_peak_worker_rss_bytes": self.depth_peak_worker_rss_bytes,
            "depth_worker_restarts": (
                self.depth_worker.stats().get("restarts", 0)
                if self.depth_worker is not None
                else 0
            ),
        }
        context = {
            "depth": {
                "valid_ratio": float(np.mean(self.depth_valid_ratios))
                if self.depth_valid_ratios
                else 0.0,
                "temporal_agreement": float(np.mean(self.temporal_agreements))
                if self.temporal_agreements
                else 0.0,
                "left_right_consistency": float(np.mean(self.stereo_consistency))
                if self.stereo_consistency
                else 0.0,
                "left_right_coverage": (
                    self.left_right_verified_frames / self.depth_frames_evaluated
                    if self.depth_frames_evaluated
                    else 0.0
                ),
                "left_right_evidence_available": bool(
                    self.left_right_verified_frames and self.stereo_consistency
                ),
            },
            "pose": {
                "maximum_translation_step_m": max(self.pose_translation_steps, default=0.0),
                "maximum_rotation_step_deg": max(self.pose_rotation_steps, default=0.0),
                "maximum_position_std_m": maximum_position_std,
                "timestamps_monotonic": True,
            },
            "dynamic": {
                "dynamic_contamination_rate": max(self.contamination_rates, default=0.0),
                "unknown_ratio": float(np.mean(self.motion_unknown_ratios))
                if self.motion_unknown_ratios
                else 1.0,
                "dynamic_ratio": float(np.mean(self.dynamic_ratios))
                if self.dynamic_ratios
                else 0.0,
            },
            "runtime": runtime,
            "semantic": self.memory.correction_stats(),
        }
        if map_metrics is not None:
            context["map"] = map_metrics
        return context

    def close(self) -> None:
        if self.depth_worker is not None:
            self.depth_worker.close()
        if self.static_map_backend is not None:
            self.static_map_backend.close()
        self.memory.close()


def terminal_prefix(checkpoint: RealtimeCheckpoint, total_frames: int) -> int:
    state = checkpoint.state
    terminal = checkpoint.completed_indices | {
        int(value) for value in state.get("dropped_frames", {})
    }
    index = 0
    while index < total_frames and index in terminal:
        index += 1
    return index


def rebuild_static_map_prefix(
    backend: HydraStaticMapBackend,
    run_dir: Path,
    frames: list[ReplayFrame],
    completed_indices: set[int],
) -> int:
    """Rebuild Hydra state from committed static products before resuming."""
    rebuilt = 0
    for frame in frames:
        if frame.frame_index not in completed_indices:
            continue
        rgb_bgr = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
        static_raw = cv2.imread(
            str(run_dir / "static_depth" / f"{frame.frame_index:08d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        if rgb_bgr is None:
            raise FileNotFoundError(
                f"Hydra resume RGB frame is missing: {frame.rgb_path}"
            )
        if static_raw is None:
            raise FileNotFoundError(
                "Hydra resume requires committed static depth for frame "
                f"{frame.frame_index}"
            )
        if static_raw.shape != rgb_bgr.shape[:2]:
            raise ValueError(
                f"Hydra resume RGB-D dimensions differ for frame {frame.frame_index}"
            )
        backend.integrate(
            sensor_time_ns=frame.sensor_time_ns,
            rgb_image=cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
            static_depth_m=static_raw.astype(np.float32) / 1000.0,
            world_T_camera=frame.world_T_camera,
        )
        rebuilt += 1
    return rebuilt


def add_semantic_runtime_metrics(
    scheduler_report: dict,
    semantic_stats: dict,
) -> None:
    """Expose model service time separately from sidecar orchestration time."""

    elapsed_s = max(float(scheduler_report.get("elapsed_seconds", 0.0)), 1.0e-9)
    empty_latency = {
        "samples": 0,
        "p50": None,
        "p95": None,
        "p99": None,
        "max": None,
    }
    stage_contracts = {
        "segmentation": (
            "segmentation_calls",
            "segmentation_failures",
            "segmentation_ms",
        ),
        "tracking": ("tracking_calls", "tracking_failures", "tracking_ms"),
    }
    stages = scheduler_report.setdefault("stages", {})
    for stage, (calls_key, failures_key, latency_key) in stage_contracts.items():
        processed = int(semantic_stats.get(calls_key, 0))
        failures = int(semantic_stats.get(failures_key, 0))
        service_latency = dict(
            semantic_stats.get("latency", {}).get(latency_key, empty_latency)
        )
        stages[stage] = {
            "processed": processed,
            "errors": failures,
            "throughput_hz": processed / elapsed_s,
            "queue_high_water": 0,
            "drops": {},
            "latency": {
                "queue_wait_ms": dict(empty_latency),
                "service_ms": service_latency,
                "end_to_end_ms": service_latency,
            },
        }


def main() -> None:
    args = parse_args()
    requested_stop_after = args.stop_after
    requested_fault_stage = args.fault_stage
    args.stop_after = LEGACY_STAGE_ALIASES.get(args.stop_after, args.stop_after)
    args.fault_stage = LEGACY_STAGE_ALIASES.get(args.fault_stage, args.fault_stage)
    if args.rate_hz <= 0 or args.stage_rate_multiplier <= 0:
        raise ValueError("Replay and stage rates must be positive")
    if args.queue_capacity <= 0 or args.drain_timeout_s <= 0:
        raise ValueError("Queue and drain settings must be positive")
    if (
        args.segmentation_rate_hz <= 0
        or args.semantic_frontend_rate_hz <= 0
        or args.semantic_queue_capacity <= 0
        or args.semantic_minimum_observations <= 0
        or args.semantic_drain_timeout_s <= 0
        or args.dam_minimum_gpu_idle_s < 0
    ):
        raise ValueError("Semantic rates, queues, observations, and drain timeout must be positive")
    if args.semantic_mode != "disabled" and not args.semantic_config.is_file():
        raise ValueError("Real semantic mode requires a valid --semantic-config")
    if args.depth_lr_interval <= 0:
        raise ValueError("--depth-lr-interval must be positive")
    if args.static_map_backend == "hydra":
        if args.stop_after != "global":
            raise ValueError("Hydra static map backend requires --stop-after global")
        if args.hydra_config_path is None or not args.hydra_config_path.is_file():
            raise ValueError("Hydra static map backend requires a valid config path")
    if args.semantic_mode == "dam" and args.static_map_backend != "hydra":
        raise ValueError("DAM mode requires the Hydra backend for durable DSG ACKs")
    dataset = args.dataset.resolve()
    precomputed_depth_provenance = (
        load_precomputed_depth_provenance(dataset)
        if args.depth_backend == "precomputed"
        else None
    )
    effective_depth_confidence_mode = (
        precomputed_depth_provenance.get("confidence_mode")
        if precomputed_depth_provenance
        and precomputed_depth_provenance.get("confidence_mode")
        else args.depth_confidence_mode
    )
    semantic_model_provenance = (
        load_semantic_model_provenance(args.semantic_config)
        if args.semantic_mode != "disabled"
        else None
    )
    foundation_stereo_python = None
    effective_depth_profile = None
    if args.depth_backend == "foundation-worker":
        if args.checkpoint is None or not args.checkpoint.expanduser().is_file():
            raise ValueError(
                "--checkpoint is required and must exist for foundation-worker"
            )
        if not args.foundation_stereo_root.is_dir():
            raise ValueError("FoundationStereo root is missing")
        foundation_stereo_python = resolve_environment_python(
            args.foundation_stereo_env,
            args.foundation_stereo_python,
        )
        effective_depth_profile = resolve_inference_profile(
            argparse.Namespace(
                profile=args.depth_profile,
                valid_iters=args.depth_valid_iters,
                scale=args.depth_scale,
                precision=args.depth_precision,
                torch_compile=args.depth_torch_compile,
                confidence_mode=args.depth_confidence_mode,
                lr_absolute_tolerance_px=0.75,
                lr_relative_tolerance=0.03,
            )
        )
    time_contract = validate_time_contract(dataset)
    metadata = json.loads((dataset / "tick_index.json").read_text())
    time_contract.update(
        {
            "monotonic": True,
            "pose_exact_match": True,
            "relative_time_consistent": True,
            "maximum_stereo_delta_ms": max(
                (float(frame.get("stereo_delta_ms", 0.0)) for frame in metadata["frames"]),
                default=0.0,
            ),
            "projection_model": metadata.get("projection_model"),
        }
    )
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir
        else REPOSITORY_ROOT
        / "output"
        / "realtime_replay"
        / f"{dataset.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if run_dir.exists() and any(run_dir.iterdir()) and not (args.resume or args.overwrite):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    if args.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu_lock_path = (
        run_dir / "gpu_coordination" / "single_gpu.lock"
        if args.gpu_sharing_mode == "staggered"
        else None
    )
    gpu_activity_path = (
        run_dir / "gpu_coordination" / "realtime.activity"
        if args.gpu_sharing_mode == "staggered"
        else None
    )
    gpu_coordinator = SharedGpuCoordinator(
        lock_path=gpu_lock_path,
        activity_path=gpu_activity_path,
    )

    configuration = {
        "stop_after": args.stop_after,
        "requested_stop_after": requested_stop_after,
        "depth_backend": args.depth_backend,
        "depth_confidence_mode": effective_depth_confidence_mode,
        "requested_depth_confidence_mode": args.depth_confidence_mode,
        "depth_lr_interval": args.depth_lr_interval,
        "rate_hz": args.rate_hz,
        "no_throttle": args.no_throttle,
        "allow_source_bursts": args.allow_source_bursts,
        "stage_rate_multiplier": args.stage_rate_multiplier,
        "queue_capacity": args.queue_capacity,
        "stage_deadline_ms": args.stage_deadline_ms,
        "pose_position_std_m": args.pose_position_std_m,
        "pose_rotation_std_deg": args.pose_rotation_std_deg,
        "minimum_dynamic_pixels": args.minimum_dynamic_pixels,
        "motion_analysis_width": args.motion_analysis_width,
        "submap_frames": args.submap_frames,
        "static_map_backend": args.static_map_backend,
        "hydra_config_path": (
            str(args.hydra_config_path.resolve())
            if args.hydra_config_path is not None
            else None
        ),
        "hydra_config_sha256": (
            sha256_file(args.hydra_config_path)
            if args.hydra_config_path is not None
            else None
        ),
        "semantic": {
            "mode": args.semantic_mode,
            "config": str(args.semantic_config.resolve()),
            "config_sha256": sha256_file(args.semantic_config),
            "segmentation_rate_hz": args.segmentation_rate_hz,
            "frontend_rate_hz": args.semantic_frontend_rate_hz,
            "queue_capacity": args.semantic_queue_capacity,
            "minimum_observations": args.semantic_minimum_observations,
            "drain_timeout_s": args.semantic_drain_timeout_s,
        },
        "gpu_coordination": {
            "mode": args.gpu_sharing_mode,
            "lock_path": str(gpu_lock_path) if gpu_lock_path is not None else None,
            "activity_path": (
                str(gpu_activity_path) if gpu_activity_path is not None else None
            ),
            "dam_minimum_idle_s": args.dam_minimum_gpu_idle_s,
        },
        "quality_config": str(args.quality_config.resolve()),
        "quality_config_sha256": sha256_file(args.quality_config),
        "fault": {
            "stage": args.fault_stage,
            "requested_stage": requested_fault_stage,
            "delay_ms": args.fault_delay_ms,
            "every": args.fault_every,
            "error_every": args.fault_error_every,
        },
    }
    depth_profile_manifest = effective_depth_profile or {
        "name": (
            precomputed_depth_provenance.get("profile")
            if precomputed_depth_provenance
            else None
        ),
        "valid_iters": (
            precomputed_depth_provenance.get("valid_iters")
            if precomputed_depth_provenance
            else None
        ),
        "scale": (
            precomputed_depth_provenance.get("scale")
            if precomputed_depth_provenance
            else None
        ),
        "precision": (
            precomputed_depth_provenance.get("precision")
            if precomputed_depth_provenance
            else None
        ),
    }
    manifest = build_run_manifest(
        REPOSITORY_ROOT,
        dataset,
        configuration=configuration,
        time_contract=time_contract,
        model_configuration={
            "depth_backend": args.depth_backend,
            "foundation_stereo": {
                "environment": args.foundation_stereo_env,
                "python": (
                    str(foundation_stereo_python)
                    if foundation_stereo_python is not None
                    else None
                ),
                "root": str(args.foundation_stereo_root.resolve()),
                "checkpoint": str(args.checkpoint.resolve())
                if args.checkpoint is not None
                else (
                    precomputed_depth_provenance.get("checkpoint")
                    if precomputed_depth_provenance
                    else None
                ),
                "checkpoint_sha256": (
                    sha256_file(args.checkpoint.expanduser().resolve())
                    if args.checkpoint is not None
                    else (
                        precomputed_depth_provenance.get("checkpoint_sha256")
                        if precomputed_depth_provenance
                        else None
                    )
                ),
                "license": "NVIDIA research/non-commercial",
                "profile": depth_profile_manifest["name"],
                "valid_iters": depth_profile_manifest["valid_iters"],
                "scale": depth_profile_manifest["scale"],
                "precision": depth_profile_manifest["precision"],
                "confidence_mode": effective_depth_confidence_mode,
                "left_right_interval": args.depth_lr_interval,
                "torch_compile": args.depth_torch_compile,
                "precomputed_provenance": precomputed_depth_provenance,
            },
            "semantic_frontend": semantic_model_provenance,
        },
    )
    write_run_manifest(run_dir / "run_manifest.json", manifest)
    if args.dry_run:
        plan = {
            "status": "planned",
            "run_dir": str(run_dir),
            "dataset": str(dataset),
            "frame_count": len(metadata["frames"]),
            "stages": list(STAGES[: STAGES.index(args.stop_after) + 1]),
            "semantic_branch": (
                []
                if args.semantic_mode == "disabled"
                else ["depth", SEMANTIC_STAGE, args.semantic_mode]
            ),
            "time_contract": time_contract,
        }
        (run_dir / "dry_run_plan.json").write_text(
            json.dumps(plan, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(plan, indent=2, allow_nan=False))
        return

    poses = read_poses(dataset)
    frames = build_frames(dataset, metadata, poses)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        frames = frames[: args.max_frames]
    fingerprint = manifest["dataset"]["tick_index_sha256"]
    checkpoint = RealtimeCheckpoint(
        run_dir / "realtime_checkpoint.json", dataset_fingerprint=fingerprint
    )
    checkpoint_state = None
    if args.resume:
        if not checkpoint.load():
            raise FileNotFoundError("--resume requested but checkpoint is missing")
        checkpoint_state = checkpoint.state
    start_index = terminal_prefix(checkpoint, len(frames)) if args.resume else 0
    depth_worker = None
    static_map_backend = None
    depth_backend_startup_seconds = 0.0
    depth_backend_warmup_seconds = 0.0
    static_map_backend_startup_seconds = 0.0
    static_map_frames_rebuilt = 0
    if args.depth_backend == "foundation-worker":
        assert foundation_stereo_python is not None
        command = [
            str(foundation_stereo_python),
            "-u",
            str(REPOSITORY_ROOT / "scripts" / "foundation_stereo_worker.py"),
            "--fs-root",
            str(args.foundation_stereo_root.resolve()),
            "--checkpoint",
            str(args.checkpoint.expanduser().resolve()),
            "--profile",
            args.depth_profile,
            "--confidence-mode",
            args.depth_confidence_mode,
            "--valid-iters",
            str(effective_depth_profile["valid_iters"]),
            "--scale",
            str(effective_depth_profile["scale"]),
            "--precision",
            effective_depth_profile["precision"],
        ]
        if args.depth_torch_compile:
            command.append("--torch-compile")
        if gpu_lock_path is not None:
            command.extend(["--gpu-lock-path", str(gpu_lock_path)])
        if gpu_activity_path is not None:
            command.extend(["--gpu-activity-path", str(gpu_activity_path)])
        depth_worker = SubprocessDepthBackend(
            command,
            startup_timeout_s=args.depth_startup_timeout_s,
            request_timeout_s=args.depth_request_timeout_s,
            maximum_retries=args.depth_maximum_retries,
        )
    if args.static_map_backend == "hydra":
        first_rgb = cv2.imread(str(frames[0].rgb_path), cv2.IMREAD_COLOR)
        if first_rgb is None:
            raise FileNotFoundError(frames[0].rgb_path)
        map_startup_started = time.monotonic()
        hydra_output_dir = run_dir / "hydra_realtime"
        if args.resume and hydra_output_dir.exists():
            shutil.rmtree(hydra_output_dir)
        static_map_backend = HydraStaticMapBackend(
            args.hydra_config_path,
            hydra_output_dir,
            labelspace_path=args.hydra_labelspace_path,
            labelspace_colors=args.hydra_labelspace_colors,
            maximum_depth_m=float(metadata.get("recommended_max_depth_m", 20.0)),
        )
        static_map_backend.initialize(
            first_rgb.shape[1],
            first_rgb.shape[0],
            frames[0].intrinsics,
        )
        static_map_backend_startup_seconds = time.monotonic() - map_startup_started
    engine = ReplayEngine(
        run_dir,
        checkpoint,
        pose_position_std_m=args.pose_position_std_m,
        pose_rotation_std_deg=args.pose_rotation_std_deg,
        minimum_dynamic_pixels=args.minimum_dynamic_pixels,
        motion_analysis_width=args.motion_analysis_width,
        submap_frames=args.submap_frames,
        write_fusion_products=not args.no_write_fusion_products,
        depth_worker=depth_worker,
        stereo_fx=float(metadata.get("fx", frames[0].intrinsics[0, 0])),
        stereo_baseline_m=float(metadata["baseline"]),
        maximum_depth_m=float(metadata.get("recommended_max_depth_m", 20.0)),
        depth_confidence_mode=args.depth_confidence_mode,
        depth_lr_interval=args.depth_lr_interval,
        static_map_backend=static_map_backend,
        checkpoint_state=checkpoint_state,
    )
    if static_map_backend is not None and start_index > 0:
        static_map_frames_rebuilt = rebuild_static_map_prefix(
            static_map_backend,
            run_dir,
            frames[:start_index],
            checkpoint.completed_indices,
        )
    if depth_worker is not None:
        startup_started = time.monotonic()
        depth_worker.start()
        depth_backend_startup_seconds = time.monotonic() - startup_started
        if start_index < len(frames) and "depth" in STAGES[: STAGES.index(args.stop_after) + 1]:
            warmup_started = time.monotonic()
            engine.materialize_depth(frames[start_index])
            depth_backend_warmup_seconds = time.monotonic() - warmup_started
    if start_index > 0:
        engine.initialize_previous(frames[start_index - 1])

    semantic_adapter = None
    semantic_startup_seconds = 0.0
    if args.semantic_mode != "disabled":
        from daaam.config import PipelineConfig
        from daaam.realtime.semantic import (
            RealtimeSemanticAdapter,
            RealtimeSemanticConfig,
        )

        pipeline_config = PipelineConfig.from_yaml(str(args.semantic_config.resolve()))
        pipeline_config.log_dir = str(run_dir / "semantic_sidecar" / "logs")
        pipeline_config.workers.dam_grounding_config.gpu_lock_path = (
            str(gpu_lock_path) if gpu_lock_path is not None else None
        )
        pipeline_config.workers.dam_grounding_config.gpu_activity_path = (
            str(gpu_activity_path) if gpu_activity_path is not None else None
        )
        pipeline_config.workers.dam_grounding_config.gpu_minimum_idle_s = (
            args.dam_minimum_gpu_idle_s
        )
        gpu_coordinator.touch_activity()
        semantic_started = time.monotonic()
        semantic_adapter = RealtimeSemanticAdapter(
            pipeline_config,
            engine.memory,
            session_id="replay",
            output_dir=run_dir / "semantic_sidecar",
            config=RealtimeSemanticConfig(
                segmentation_rate_hz=args.segmentation_rate_hz,
                minimum_observations=args.semantic_minimum_observations,
                prompt_queue_capacity=max(2, args.semantic_queue_capacity * 10),
                correction_queue_capacity=max(10, args.semantic_queue_capacity * 20),
                grounding_enabled=args.semantic_mode == "dam",
                gpu_lock_path=gpu_lock_path,
                gpu_activity_path=gpu_activity_path,
            ),
        )
        semantic_adapter.start()
        engine.semantic_label_provider = semantic_adapter.label_image_for
        semantic_startup_seconds = time.monotonic() - semantic_started

    injector = FaultInjector(
        args.fault_stage,
        delay_ms=args.fault_delay_ms,
        every=args.fault_every,
        error_every=args.fault_error_every,
    )
    scheduler = MultiRateScheduler(active_revision=engine.submaps.map_revision)
    base_rates = {
        "pose": 50.0,
        "depth": 30.0,
        "dynamic": 10.0,
        "fusion": 10.0,
        "global": 10.0,
    }
    handlers = {
        "pose": engine.pose,
        "depth": engine.depth,
        "dynamic": engine.tracking,
        "fusion": engine.fusion,
        "global": engine.global_map,
    }
    enabled_stages = STAGES[: STAGES.index(args.stop_after) + 1]
    for stage in enabled_stages:
        handler = handlers[stage]
        if stage == args.stop_after:
            handler = engine.terminal(stage, handler)
        scheduler.add_stage(
            StageSpec(
                stage,
                injector.wrap(stage, handler),
                base_rates[stage] * args.stage_rate_multiplier,
                args.queue_capacity,
                args.stage_deadline_ms,
            )
        )
    if semantic_adapter is not None and "depth" in enabled_stages:
        scheduler.add_stage(
            StageSpec(
                SEMANTIC_STAGE,
                injector.wrap(SEMANTIC_STAGE, semantic_adapter.handle),
                args.semantic_frontend_rate_hz * args.stage_rate_multiplier,
                args.semantic_queue_capacity,
                args.stage_deadline_ms,
            )
        )
        # Offer each depth frame to the independently bounded sidecar before
        # continuing the geometry chain. Neither destination waits for the other.
        scheduler.connect("depth", SEMANTIC_STAGE)
    for source, destination in zip(enabled_stages, enabled_stages[1:]):
        scheduler.connect(source, destination)
    scheduler.start()

    source_frames = frames[start_index:]
    source_deltas = np.diff([frame.sensor_time_ns for frame in source_frames]) / 1e9
    positive_deltas = source_deltas[source_deltas > 0]
    nominal_hz = (
        1.0 / float(np.median(positive_deltas)) if len(positive_deltas) else args.rate_hz
    )
    previous_source_time = None
    dispatched = 0
    replay_sleep_seconds = 0.0
    for frame in source_frames:
        gpu_coordinator.touch_activity()
        if not args.no_throttle and previous_source_time is not None:
            capture_delta_s = (frame.sensor_time_ns - previous_source_time) / 1e9
            scaled_delay = capture_delta_s * nominal_hz / args.rate_hz
            if not args.allow_source_bursts:
                scaled_delay = max(scaled_delay, 1.0 / args.rate_hz)
            if scaled_delay > 0:
                time.sleep(scaled_delay)
                replay_sleep_seconds += scaled_delay
        previous_source_time = frame.sensor_time_ns
        envelope = RealtimeEnvelope(
            MessageKey(
                frame.sensor_time_ns,
                map_revision=engine.submaps.map_revision,
                calibration_revision=0,
            ),
            frame,
            frame.value,
            source="absolute_time_replay",
            trace_id=f"frame-{frame.frame_index}",
        )
        if scheduler.submit("pose", envelope):
            dispatched += 1
        else:
            checkpoint.mark_dropped(frame.frame_index, "source_queue_rejected")

    idle = scheduler.wait_until_idle(args.drain_timeout_s, stable_for=0.1)
    scheduler_report = scheduler.stop(
        timeout=max(5.0, args.drain_timeout_s), drain=not idle
    )
    if "global" in enabled_stages:
        engine.finalize_paths()
        engine.finalize_static_map()
    semantic_stats = None
    if semantic_adapter is not None:
        if static_map_backend is not None:
            semantic_adapter.attach_hydra_dsg(
                run_dir / "hydra_realtime" / "backend" / "dsg.json"
            )
        semantic_stats = semantic_adapter.stop(
            timeout_s=args.semantic_drain_timeout_s,
            drain=True,
        )
        add_semantic_runtime_metrics(scheduler_report, semantic_stats)
        scheduler_report["semantic_sidecar"] = semantic_stats
    terminal_now = checkpoint.completed_indices | {
        int(value) for value in checkpoint.state.get("dropped_frames", {})
    }
    for frame in source_frames:
        if frame.frame_index not in terminal_now:
            checkpoint.mark_dropped(frame.frame_index, "pipeline_not_completed")
    (run_dir / "realtime_metrics.json").write_text(
        json.dumps(scheduler_report, indent=2, allow_nan=False) + "\n"
    )

    map_metrics = (
        json.loads(args.map_metrics_json.read_text())
        if args.map_metrics_json is not None
        else (
            static_map_backend.map_metrics()
            if static_map_backend is not None
            else None
        )
    )
    context = engine.quality_context(scheduler_report, map_metrics=map_metrics)
    context["time"] = time_contract
    if args.semantic_mode == "dam":
        assert semantic_stats is not None
        corrections = semantic_stats.get("memory", {}).get("corrections", {})
        context["semantic"] = {
            **corrections,
            "required": True,
            "submitted": int(semantic_stats.get("corrections_submitted", 0)),
            "prompts_submitted": int(semantic_stats.get("prompts_submitted", 0)),
            "dsg": semantic_stats.get("dsg", {}),
            "grounding_workers": semantic_stats.get("grounding_workers", {}),
        }
    (run_dir / "quality_context.json").write_text(
        json.dumps(context, indent=2, allow_nan=False) + "\n"
    )
    required = ["time", "pose", "runtime"]
    if "depth" in enabled_stages:
        required.append("depth")
    if "dynamic" in enabled_stages:
        required.append("dynamic")
    if args.semantic_mode == "dam":
        required.append("semantic")
    if args.stop_after == "global":
        required.append("map")
    quality = QualityGateRunner(
        QualityGateConfig.from_yaml(args.quality_config)
    ).evaluate(context, required_stages=required)
    (run_dir / "quality_report.json").write_text(
        json.dumps(quality, indent=2, allow_nan=False) + "\n"
    )
    report = {
        "status": (
            "drain_timeout"
            if not idle
            else (
                "stage_error"
                if scheduler_report.get("handler_errors")
                else "complete"
            )
        ),
        "dataset": str(dataset),
        "run_dir": str(run_dir),
        "frames_requested": len(frames),
        "frames_resumed_from": start_index,
        "frames_dispatched": dispatched,
        "frames_completed": len(checkpoint.completed_indices),
        "dropped_frames": checkpoint.state["dropped_frames"],
        "frames_by_stage": engine.frames_by_stage,
        "dynamic_objects_active": len(engine.dynamic_layer.active_objects),
        "dynamic_objects_expired": len(engine.dynamic_layer.history),
        "submaps": len(engine.submaps.submaps),
        "canonical_paths": len(engine.path_repository.paths),
        "static_map_backend": engine.static_map_stats(),
        "static_map_backend_startup_seconds": static_map_backend_startup_seconds,
        "static_map_frames_rebuilt": static_map_frames_rebuilt,
        "map_metrics": map_metrics,
        "depth_backend": args.depth_backend,
        "depth_backend_startup_seconds": depth_backend_startup_seconds,
        "depth_backend_warmup_seconds": depth_backend_warmup_seconds,
        "depth_backend_stats": depth_worker.stats() if depth_worker is not None else None,
        "depth_peak_cuda_memory_bytes": engine.depth_peak_cuda_memory_bytes,
        "depth_peak_worker_rss_bytes": engine.depth_peak_worker_rss_bytes,
        "semantic_mode": args.semantic_mode,
        "semantic_startup_seconds": semantic_startup_seconds,
        "semantic_stats": semantic_stats,
        "left_right_verified_frames": engine.left_right_verified_frames,
        "depth_frames_evaluated": engine.depth_frames_evaluated,
        "left_right_coverage": (
            engine.left_right_verified_frames / engine.depth_frames_evaluated
            if engine.depth_frames_evaluated
            else 0.0
        ),
        "replay_pacing": {
            "configured_max_rate_hz": args.rate_hz,
            "source_nominal_rate_hz": nominal_hz,
            "source_bursts_allowed": args.allow_source_bursts,
            "sleep_seconds": replay_sleep_seconds,
            "absolute_timestamps_preserved": True,
        },
        "quality_passed": quality["passed"],
        "hard_quality_failures": quality["hard_failures"],
    }
    (run_dir / "realtime_run_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    engine.close()
    if not idle:
        raise SystemExit(3)
    if not quality["passed"] and not args.quality_report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
