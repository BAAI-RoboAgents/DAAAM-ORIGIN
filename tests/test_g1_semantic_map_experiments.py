from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import daaam.experiments.g1_semantic_map as experiments  # noqa: E402
from daaam.experiments import (  # noqa: E402
    EXPERIMENT_CATALOG,
    ExperimentConfig,
    ExperimentManager,
)
from daaam.realtime.audit import JsonlAuditWriter  # noqa: E402
from daaam.realtime.contracts import (  # noqa: E402
    FrameValue,
    MessageKey,
    RealtimeEnvelope,
)
from daaam.realtime.scheduler import MultiRateScheduler, StageSpec  # noqa: E402


def make_raw_dataset(root: Path) -> Path:
    raw = root / "raw"
    required = (
        "timestamps/000000.txt",
        "poses/dense_global/000000/poses.txt",
        "poses/dense_global/000000/poses_7d.txt",
        "poses/dense_global/000000/aux_poses.jsonl",
        "state/000000/odom.jsonl",
        "state/000000/map_pose.jsonl",
        "state/000000/joint_states.jsonl",
        "calibrations/000000/camera_info.json",
    )
    for relative in required:
        path = raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    (raw / "manifest.json").write_text('{"layout":"capture4daaam_like"}\n')
    (raw / "quality_report.json").write_text('{"alignment":{"ok":true}}\n')
    records = []
    for index in range(4):
        left = raw / f"2d_rect/cam0/000000/{index:06d}.png"
        right = raw / f"2d_rect/cam1/000000/{index:06d}.png"
        lidar = raw / f"lidar/000000/lidar0/{index:06d}.npy"
        for path in (left, right, lidar):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"frame-{index}".encode())
        records.append(
            {
                "tick": index,
                "images": [
                    {"camera": "cam0", "path": str(left.relative_to(raw))},
                    {"camera": "cam1", "path": str(right.relative_to(raw))},
                ],
                "lidar": [
                    {
                        "path": str(lidar.relative_to(raw)),
                        "sensor_time_ns": 1_000_000_000 + index,
                    }
                ],
            }
        )
    (raw / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    return raw


def make_config(tmp_path: Path) -> ExperimentConfig:
    raw = make_raw_dataset(tmp_path)
    config = {
        "schema": experiments.SCHEMA,
        "repository_root": str(REPOSITORY_ROOT),
        "workspace": str(tmp_path / "workspace"),
        "raw_dataset": str(raw),
        "source_frames": {"start": 0, "end": 3},
        "repeats": 3,
        "seeds": [0, 1, 2],
        "splits": {
            "calibration": [0, 0],
            "development": [1, 1],
            "stress": [2, 2],
            "held_out": [3, 3],
        },
        "inputs": {
            "fast_foundation_repo": "/models/fast",
            "fast_foundation_checkpoint": "/models/checkpoint.pth",
        },
        "artifacts": {
            name: str(tmp_path / "artifacts" / name)
            for name in (
                "prepared_dataset",
                "selected_dataset",
                "geometry_dataset",
                "temporal_report",
                "odometry_dataset",
                "loop_report",
                "optimized_dataset",
                "filtered_dataset",
            )
        },
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    return ExperimentConfig.from_yaml(path)


def test_workspace_provenance_splits_and_matrix(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    manager = ExperimentManager(config)
    monkeypatch.setattr(experiments, "_command_output", lambda *args, **kwargs: "mock")
    manifest = manager.initialize()
    assert manifest["dataset"]["frame_count"] == 4
    assert all(record["sha256"] for record in manifest["dataset"]["files"])

    splits = manager.build_splits_and_annotation_tasks()
    assert splits["splits"]["held_out"]["frames"] == [3]
    tasks = (config.workspace / "ground_truth/annotation_tasks.jsonl").read_text().splitlines()
    assert len(tasks) == 4
    assert json.loads(tasks[-1])["status"] == "unlabeled"

    runs = manager.generate_registry()
    expected = sum(
        len(experiment["variants"]) for experiment in EXPERIMENT_CATALOG.values()
    ) * 3
    assert len(runs) == expected
    assert len({run.run_id for run in runs}) == expected

    realtime_run = next(run for run in runs if run.experiment_id == "E10")
    command = manager.commands_for(realtime_run)[0]
    assert "--experiment-telemetry" in command
    assert command.count("--run-dir") == 1


def test_dry_run_materializes_immutable_run_spec(tmp_path):
    manager = ExperimentManager(make_config(tmp_path))
    run = next(item for item in manager.runs() if item.experiment_id == "E1")
    status = manager.execute(run, dry_run=True)
    assert status["status"] == "planned"
    specification = json.loads(
        (run.run_dir / "manifest/run_spec.json").read_text()
    )
    assert specification["run_id"] == run.run_id
    assert specification["commands"]


def test_formal_execution_is_blocked_when_geometry_gate_is_missing(tmp_path):
    manager = ExperimentManager(make_config(tmp_path))
    run = next(item for item in manager.runs() if item.experiment_id == "E1")
    status = manager.execute(run, dry_run=False)
    assert status["status"] == "blocked_failed_geometry_gate"
    assert status["formal_eligibility"]["eligible"] is False
    assert not (run.run_dir / "logs/command_01.log").exists()


def test_realtime_variants_materialize_semantic_and_hydra_overrides(tmp_path):
    manager = ExperimentManager(make_config(tmp_path))
    fastsam_run = next(
        item
        for item in manager.runs()
        if item.experiment_id == "E11" and item.variant == "conf_0.2"
    )
    manager.materialize_run(fastsam_run)
    fastsam = yaml.safe_load(
        (fastsam_run.run_dir / "configs/fastsam_variant.yaml").read_text()
    )
    assert fastsam["fastsam_conf"] == 0.2
    semantic = yaml.safe_load(
        (
            fastsam_run.run_dir / "configs/pipeline_config_variant.yaml"
        ).read_text()
    )
    assert semantic["segmentation"]["model_config_path"].endswith(
        "fastsam_variant.yaml"
    )

    hydra_run = next(
        item
        for item in manager.runs()
        if item.experiment_id == "E16" and item.variant == "object_max_range_2m"
    )
    manager.materialize_run(hydra_run)
    hydra = yaml.safe_load(
        (hydra_run.run_dir / "configs/hydra_variant.yaml").read_text()
    )
    assert hydra["active_window"]["object_detector"]["max_range"] == 2.0


def test_scheduler_writes_strict_queue_and_service_events(tmp_path):
    writer = JsonlAuditWriter(
        tmp_path / "queue_events.jsonl",
        schema="daaam.test.queue.v1",
    )
    scheduler = MultiRateScheduler(audit_writer=writer)
    scheduler.add_stage(StageSpec("pose", lambda item: None, None, 2))
    scheduler.start()
    envelope = RealtimeEnvelope(
        MessageKey(1_000_000_000),
        {"frame": 0},
        FrameValue.ROUTINE,
        trace_id="frame-0",
    )
    assert scheduler.submit("pose", envelope)
    assert scheduler.wait_until_idle(1.0)
    scheduler.stop()
    writer.close()
    events = [
        json.loads(line)
        for line in (tmp_path / "queue_events.jsonl").read_text().splitlines()
    ]
    assert {event["event"] for event in events} >= {
        "stage_registered",
        "queue_submit",
        "service_start",
        "service_complete",
    }
    assert all(event["schema"] == "daaam.test.queue.v1" for event in events)
