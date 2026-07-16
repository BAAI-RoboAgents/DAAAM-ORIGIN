"""End-to-end tests for dry-run, replay, fault injection, and resume."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "scripts" / "run_realtime_mapping.py"
ORIGIN_NS = 1_783_933_507_759_540_877
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import run_realtime_mapping as realtime_runner  # noqa: E402
from run_realtime_mapping import (  # noqa: E402
    RealtimeResourceState,
    add_semantic_runtime_metrics,
    cleanup_realtime_resources,
    load_precomputed_depth_provenance,
    load_semantic_model_provenance,
    resolve_environment_python,
    scheduled_confidence_mode,
)


def test_resource_cleanup_is_reverse_order_best_effort_and_idempotent():
    calls = []

    class Scheduler:
        def stop(self, *, timeout, drain):
            calls.append(("scheduler", timeout, drain))

    class SemanticAdapter:
        def stop(self, *, timeout_s, drain):
            calls.append(("semantic_adapter", timeout_s, drain))
            raise RuntimeError("semantic cleanup failed")

    class DepthWorker:
        def close(self):
            calls.append(("depth_worker",))

    class ActivityHeartbeat:
        def stop(self, *, timeout_s):
            calls.append(("gpu_activity_heartbeat", timeout_s))

    class Memory:
        def close(self):
            calls.append(("map_memory",))
            raise RuntimeError("memory cleanup failed")

    class StaticMapBackend:
        def close(self, *, finalize):
            calls.append(("static_map_backend", finalize))

    resources = RealtimeResourceState(
        scheduler=Scheduler(),
        semantic_adapter=SemanticAdapter(),
        gpu_activity_heartbeat=ActivityHeartbeat(),
        depth_worker=DepthWorker(),
        memory=Memory(),
        static_map_backend=StaticMapBackend(),
    )

    errors = cleanup_realtime_resources(resources, timeout_s=1.25)

    assert calls == [
        ("scheduler", 1.25, False),
        ("semantic_adapter", 1.25, False),
        ("gpu_activity_heartbeat", 1.25),
        ("depth_worker",),
        ("map_memory",),
        ("static_map_backend", False),
    ]
    assert [name for name, _error in errors] == ["semantic_adapter", "map_memory"]
    assert cleanup_realtime_resources(resources) == []


def test_main_preserves_run_error_when_cleanup_also_fails(monkeypatch, capsys):
    calls = []
    run_error = ValueError("semantic startup failed")

    class Scheduler:
        def stop(self, *, timeout, drain):
            calls.append((timeout, drain))
            raise RuntimeError("scheduler cleanup failed")

    def fail_during_run(resources):
        resources.scheduler = Scheduler()
        raise run_error

    monkeypatch.setattr(realtime_runner, "_run_realtime_mapping", fail_during_run)

    with pytest.raises(ValueError) as captured:
        realtime_runner.main()

    assert captured.value is run_error
    assert calls == [(5.0, False)]
    assert "cleanup failed for scheduler" in capsys.readouterr().err


def create_dataset(root: Path, frame_count: int = 4) -> Path:
    dataset = root / "dataset"
    for directory in (
        "rgb",
        "stereo_right",
        "depth",
        "depth_confidence",
        "depth_consistency",
        "pose",
    ):
        (dataset / directory).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    texture = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    frames = []
    poses = []
    timestamps = []
    offsets = [0, 100_000_000, 260_000_000, 480_000_000]
    for index in range(frame_count):
        timestamp = ORIGIN_NS + offsets[index]
        timestamps.append(timestamp)
        rgb_path = dataset / "rgb" / f"{index:08d}.png"
        right_path = dataset / "stereo_right" / f"{index:08d}.png"
        cv2.imwrite(str(rgb_path), texture)
        cv2.imwrite(str(right_path), texture)
        cv2.imwrite(
            str(dataset / "depth" / f"{index:08d}.png"),
            np.full((48, 64), 1500, dtype=np.uint16),
        )
        cv2.imwrite(
            str(dataset / "depth_confidence" / f"{index:08d}.png"),
            np.full((48, 64), 255, dtype=np.uint8),
        )
        cv2.imwrite(
            str(dataset / "depth_consistency" / f"{index:08d}.png"),
            np.full((48, 64), 255, dtype=np.uint8),
        )
        pose = np.eye(4)
        pose[0, 3] = index * 0.01
        poses.append(pose)
        frames.append(
            {
                "idx": index,
                "source_idx": index,
                "pose_row": index,
                "cam0": str(rgb_path),
                "cam1": str(right_path),
                "timestamp": offsets[index] / 1e9,
                "sensor_time_ns": timestamp,
                "cam0_sensor_time_ns": timestamp,
                "cam1_sensor_time_ns": timestamp,
                "pose_sensor_time_ns": timestamp,
                "stereo_delta_ms": 0.0,
                "selection_reason": "initial_frame" if index == 0 else "routine",
            }
        )
    (dataset / "pose" / "poses.txt").write_text(
        "".join(
            " ".join(str(value) for value in pose.reshape(-1)) + "\n"
            for pose in poses
        )
    )
    (dataset / "pose" / "pose_timestamps_ns.txt").write_text(
        "".join(f"{value}\n" for value in timestamps)
    )
    (dataset / "camera_info.json").write_text(
        json.dumps(
            {
                "width": 64,
                "height": 48,
                "intrinsics": [
                    [60.0, 0.0, 31.5],
                    [0.0, 60.0, 23.5],
                    [0.0, 0.0, 1.0],
                ],
            }
        )
    )
    (dataset / "tick_index.json").write_text(
        json.dumps(
            {
                "time_origin_ns": ORIGIN_NS,
                "projection_model": "pinhole",
                "fx": 60.0,
                "fy": 60.0,
                "cx": 31.5,
                "cy": 23.5,
                "baseline": 0.07,
                "frames": frames,
            }
        )
    )
    return dataset


def run_replay(dataset: Path, run_dir: Path, *extra, check=True):
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dataset",
            str(dataset),
            "--run-dir",
            str(run_dir),
            "--no-throttle",
            "--stage-rate-multiplier",
            "100",
            "--quality-report-only",
            *extra,
        ],
        check=check,
        capture_output=True,
        text=True,
    )


def test_dry_run_writes_manifest_and_absolute_time_plan(tmp_path):
    dataset = create_dataset(tmp_path)
    run_dir = tmp_path / "dry"
    run_replay(dataset, run_dir, "--dry-run")
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    plan = json.loads((run_dir / "dry_run_plan.json").read_text())
    assert manifest["time_contract"]["valid"]
    assert manifest["dataset"]["tick_index_sha256"]
    assert plan["frame_count"] == 4
    assert plan["stages"] == ["pose", "depth", "dynamic", "fusion"]
    assert not (run_dir / "realtime_checkpoint.json").exists()


def test_foundation_dry_run_records_effective_profile_values(tmp_path):
    dataset = create_dataset(tmp_path)
    foundation_root = tmp_path / "FoundationStereo"
    foundation_root.mkdir()
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"test checkpoint")
    run_dir = tmp_path / "foundation-dry"
    run_replay(
        dataset,
        run_dir,
        "--dry-run",
        "--depth-backend",
        "foundation-worker",
        "--foundation-stereo-python",
        sys.executable,
        "--foundation-stereo-root",
        str(foundation_root),
        "--checkpoint",
        str(checkpoint),
        "--depth-profile",
        "online",
    )
    profile = json.loads((run_dir / "run_manifest.json").read_text())["models"][
        "foundation_stereo"
    ]
    assert profile["profile"] == "online"
    assert profile["valid_iters"] == 8
    assert profile["scale"] == 0.15
    assert profile["precision"] == "fp16"


def test_precomputed_depth_manifest_keeps_generation_profile_and_report_hash(tmp_path):
    dataset = create_dataset(tmp_path)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"depth model")
    (dataset / "foundation_stereo_run.json").write_text(
        json.dumps(
            {
                "profile": {"name": "online"},
                "valid_iters": 8,
                "scale": 0.15,
                "precision": "fp16",
                "confidence_mode": "left-right",
                "checkpoint": str(checkpoint),
                "processed": 4,
                "failed": 0,
            }
        )
    )
    provenance = load_precomputed_depth_provenance(dataset)
    assert provenance is not None
    assert provenance["report_sha256"]
    run_dir = tmp_path / "precomputed-dry"
    run_replay(dataset, run_dir, "--dry-run")
    model = json.loads((run_dir / "run_manifest.json").read_text())["models"][
        "foundation_stereo"
    ]
    assert model["profile"] == "online"
    assert model["valid_iters"] == 8
    assert model["scale"] == 0.15
    assert model["precision"] == "fp16"
    assert model["checkpoint_sha256"]
    assert model["precomputed_provenance"]["report_sha256"]


def test_semantic_dry_run_records_independent_real_frontend_branch(tmp_path):
    dataset = create_dataset(tmp_path)
    run_dir = tmp_path / "semantic-dry"
    run_replay(
        dataset,
        run_dir,
        "--dry-run",
        "--semantic-mode",
        "frontend",
    )
    plan = json.loads((run_dir / "dry_run_plan.json").read_text())
    assert plan["stages"] == ["pose", "depth", "dynamic", "fusion"]
    assert plan["semantic_branch"] == [
        "depth",
        "semantic_frontend",
        "frontend",
    ]


def test_dam_dry_run_enables_replay_activity_heartbeat(tmp_path):
    dataset = create_dataset(tmp_path)
    hydra_config = tmp_path / "hydra.yaml"
    hydra_config.write_text("frontend: {}\n")
    run_dir = tmp_path / "dam-dry"
    run_replay(
        dataset,
        run_dir,
        "--dry-run",
        "--semantic-mode",
        "dam",
        "--stop-after",
        "global",
        "--static-map-backend",
        "hydra",
        "--hydra-config-path",
        str(hydra_config),
    )

    gpu_configuration = json.loads(
        (run_dir / "run_manifest.json").read_text()
    )["configuration"]["gpu_coordination"]
    assert gpu_configuration["dam_minimum_idle_s"] == 1.0
    assert gpu_configuration["activity_heartbeat_interval_s"] == 0.25


def test_semantic_model_latency_is_reported_under_real_stage_names():
    report = {"elapsed_seconds": 2.0, "stages": {}}
    semantic_stats = {
        "segmentation_calls": 5,
        "segmentation_failures": 0,
        "tracking_calls": 10,
        "tracking_failures": 1,
        "latency": {
            "segmentation_ms": {
                "samples": 5,
                "p50": 90.0,
                "p95": 120.0,
                "p99": 125.0,
                "max": 130.0,
            },
            "tracking_ms": {
                "samples": 11,
                "p50": 12.0,
                "p95": 20.0,
                "p99": 23.0,
                "max": 25.0,
            },
        },
    }
    add_semantic_runtime_metrics(report, semantic_stats)
    assert report["stages"]["segmentation"]["throughput_hz"] == 2.5
    assert report["stages"]["tracking"]["errors"] == 1
    assert (
        report["stages"]["tracking"]["latency"]["service_ms"]["p95"]
        == 20.0
    )


def test_semantic_model_provenance_hashes_local_artifacts(tmp_path):
    repository = tmp_path / "repository"
    (repository / "checkpoints" / "fastsam").mkdir(parents=True)
    (repository / "checkpoints" / "reid").mkdir()
    (repository / "config").mkdir()
    (repository / "checkpoints" / "fastsam" / "model.engine").write_bytes(b"sam")
    (repository / "checkpoints" / "reid" / "model.engine").write_bytes(b"reid")
    (repository / "config" / "labels.yaml").write_text("labels: []")
    (repository / "config" / "colors.csv").write_text("0,0,0")
    config = repository / "pipeline.yaml"
    config.write_text(
        """
segmentation:
  model_name: fastsam/model.engine
tracking:
  reid_weights: checkpoints/reid/model.engine
workers:
  dam_grounding_config:
    dam_model_path: nvidia/DAM-3B
semantic_config_path: config/labels.yaml
labelspace_colors_path: config/colors.csv
""".strip()
    )
    provenance = load_semantic_model_provenance(
        config,
        repository_root=repository,
    )
    assert provenance["fastsam"]["sha256"]
    assert provenance["botsort_reid"]["sha256"]
    assert provenance["semantic_labelspace"]["sha256"]
    assert provenance["labelspace_colors"]["sha256"]


def test_periodic_left_right_validation_never_skips_left_depth_inference():
    modes = [
        scheduled_confidence_mode(
            index,
            configured_mode="left-right",
            left_right_interval=3,
        )
        for index in range(7)
    ]
    assert modes == [
        "left-right",
        "validity",
        "validity",
        "left-right",
        "validity",
        "validity",
        "left-right",
    ]
    assert all(mode in {"left-right", "validity"} for mode in modes)


def test_explicit_depth_python_is_resolved_without_parent_virtualenv_leakage():
    assert resolve_environment_python("unused", Path(sys.executable)) == Path(
        sys.executable
    ).resolve()


def test_replay_completes_all_frames_and_writes_static_fusion_products(tmp_path):
    dataset = create_dataset(tmp_path)
    run_dir = tmp_path / "run"
    run_replay(dataset, run_dir)
    report = json.loads((run_dir / "realtime_run_report.json").read_text())
    checkpoint = json.loads((run_dir / "realtime_checkpoint.json").read_text())
    quality = json.loads((run_dir / "quality_report.json").read_text())
    context = json.loads((run_dir / "quality_context.json").read_text())
    assert report["frames_completed"] == 4
    assert report["dropped_frames"] == {}
    assert report["frames_by_stage"]["fusion"] == 4
    assert checkpoint["completed_frame_indices"] == [0, 1, 2, 3]
    assert len(list((run_dir / "static_depth").glob("*.png"))) == 4
    assert quality["required_stages"] == ["time", "pose", "runtime", "depth", "dynamic"]
    assert context["depth"]["left_right_coverage"] == 1.0


def test_depth_metadata_distinguishes_verified_consistency_from_validity_fallback(
    tmp_path,
):
    dataset = create_dataset(tmp_path)
    metadata = json.loads((dataset / "tick_index.json").read_text())
    metadata_dir = dataset / "depth_metadata"
    metadata_dir.mkdir()
    for frame in metadata["frames"]:
        verified = frame["idx"] == 0
        (metadata_dir / f"{frame['idx']:08d}.json").write_text(
            json.dumps(
                {
                    "frame_idx": frame["idx"],
                    "sensor_time_ns": frame["sensor_time_ns"],
                    "confidence_mode": "left-right" if verified else "validity",
                    "left_right_verified": verified,
                    "left_right_consistency": 0.7 if verified else None,
                }
            )
        )
    run_dir = tmp_path / "depth-evidence"
    run_replay(dataset, run_dir)
    context = json.loads((run_dir / "quality_context.json").read_text())
    assert context["depth"]["left_right_coverage"] == 0.25
    assert context["depth"]["left_right_consistency"] == 0.7


def test_resume_continues_from_checkpoint_without_reprocessing_prefix(tmp_path):
    dataset = create_dataset(tmp_path)
    run_dir = tmp_path / "resume"
    run_replay(dataset, run_dir, "--max-frames", "2")
    first = json.loads((run_dir / "realtime_run_report.json").read_text())
    assert first["frames_completed"] == 2
    run_replay(dataset, run_dir, "--resume")
    resumed = json.loads((run_dir / "realtime_run_report.json").read_text())
    assert resumed["frames_resumed_from"] == 2
    assert resumed["frames_completed"] == 4
    assert resumed["frames_by_stage"]["fusion"] == 2


def test_failed_resume_does_not_leave_stale_complete_report(tmp_path):
    dataset = create_dataset(tmp_path)
    run_dir = tmp_path / "failed-resume"
    run_replay(dataset, run_dir)
    assert json.loads((run_dir / "realtime_run_report.json").read_text())[
        "status"
    ] == "complete"

    foundation_root = tmp_path / "FoundationStereo"
    foundation_root.mkdir()
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"test checkpoint")
    failed = run_replay(
        dataset,
        run_dir,
        "--resume",
        "--depth-backend",
        "foundation-worker",
        "--foundation-stereo-python",
        "/bin/false",
        "--foundation-stereo-root",
        str(foundation_root),
        "--checkpoint",
        str(checkpoint),
        check=False,
    )

    assert failed.returncode != 0
    assert not (run_dir / "realtime_run_report.json").exists()


def test_global_path_history_survives_checkpoint_resume(tmp_path):
    dataset = create_dataset(tmp_path)
    run_dir = tmp_path / "global-resume"
    run_replay(
        dataset,
        run_dir,
        "--max-frames",
        "2",
        "--stop-after",
        "global",
    )
    run_replay(dataset, run_dir, "--resume", "--stop-after", "global")
    paths = json.loads((run_dir / "canonical_paths.json").read_text())
    checkpoint = json.loads((run_dir / "realtime_checkpoint.json").read_text())
    assert len(paths["paths"]) == 1
    assert len(paths["paths"][0]["observations"]) == 2
    assert checkpoint["paths"] == paths


def test_stage_failure_is_observable_and_frame_is_not_silently_committed(tmp_path):
    dataset = create_dataset(tmp_path)
    run_dir = tmp_path / "fault"
    run_replay(
        dataset,
        run_dir,
        "--fault-stage",
        "tracking",
        "--fault-error-every",
        "2",
    )
    report = json.loads((run_dir / "realtime_run_report.json").read_text())
    metrics = json.loads((run_dir / "realtime_metrics.json").read_text())
    quality = json.loads((run_dir / "quality_report.json").read_text())
    runtime_gate = next(
        result for result in quality["results"] if result["stage"] == "runtime"
    )
    assert report["status"] == "stage_error"
    assert report["frames_completed"] == 2
    assert set(report["dropped_frames"].values()) == {"pipeline_not_completed"}
    assert metrics["stages"]["dynamic"]["errors"] == 2
    assert len(metrics["handler_errors"]) == 2
    assert runtime_gate["code"] == "runtime.stage_error"
    assert runtime_gate["blocks_pipeline"]
