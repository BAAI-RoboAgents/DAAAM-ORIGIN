from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from daaam.dashboard.api import create_dashboard_app
from daaam.dashboard.commands import CommandValidationError, build_command
from daaam.dashboard.status import discover_runs, run_snapshot
from daaam.dashboard.workflows import OFFLINE_STAGES, get_workflow


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_declared_workflows_preserve_real_branching() -> None:
    offline = get_workflow("offline_hq")
    assert tuple(node.id for node in offline.nodes) == OFFLINE_STAGES
    assert offline.parameter("recommended_max_depth_m").default == 5.0
    assert offline.parameter("max_depth_m").default == 5.0
    assert offline.parameter("geometry_max_depth_m").default == 5.0
    assert offline.parameter("depth_ub").default == 5.0
    edges = {(edge.source, edge.target, edge.kind) for edge in offline.edges}
    assert ("calibrate", "temporal", "branch") in edges
    assert ("calibrate", "odometry", "branch") in edges
    assert ("calibrate", "loops", "branch") in edges
    assert ("temporal", "optimize", "join") in edges
    assert ("odometry", "optimize", "join") in edges
    assert ("loops", "optimize", "join") in edges

    realtime = get_workflow("realtime_semantic")
    realtime_edges = {
        (edge.source, edge.target, edge.kind) for edge in realtime.edges
    }
    assert ("depth", "semantic_frontend", "async") in realtime_edges
    assert ("global", "postpass", "join") in realtime_edges
    assert ("postpass", "commit", "gate") in realtime_edges


def test_tabletop_preset_emits_explicit_cli_distances(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    preview = build_command(
        get_workflow("realtime_semantic"),
        repository_root=repository,
        output_root=repository / "output",
        preset_id="tabletop_dam",
        supplied={
            "dataset": str(tmp_path / "dataset"),
            "run_dir": str(repository / "output" / "dashboard-test"),
        },
    )
    argv = list(preview.argv)
    assert argv[argv.index("--entity-merge-distance-m") + 1] == "0.075"
    assert argv[argv.index("--object-binding-maximum-center-distance-m") + 1] == "0.1"
    assert argv[argv.index("--object-binding-maximum-aabb-gap-m") + 1] == "0.025"
    assert argv[argv.index("--semantic-config") + 1].endswith(
        "config/pipeline_config_tabletop.yaml"
    )


def test_command_builder_is_argv_only_and_restricts_output_root(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    suspicious = str(tmp_path / "data;touch-not-executed")
    preview = build_command(
        get_workflow("realtime_semantic"),
        repository_root=repository,
        output_root=repository / "output",
        preset_id="realtime_geometry",
        supplied={
            "dataset": suspicious,
            "run_dir": str(repository / "output" / "safe-run"),
        },
    )
    dataset_value = preview.argv[preview.argv.index("--dataset") + 1]
    assert dataset_value == str(Path(suspicious).resolve())
    assert len([value for value in preview.argv if "touch-not-executed" in value]) == 1

    with pytest.raises(CommandValidationError, match="运行目录必须位于"):
        build_command(
            get_workflow("realtime_semantic"),
            repository_root=repository,
            output_root=repository / "output",
            supplied={"dataset": suspicious, "run_dir": str(tmp_path / "outside")},
        )


def test_real_offline_run_requires_explicit_hard_gates(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    workflow = get_workflow("offline_hq")
    (tmp_path / "g1").mkdir()
    supplied = {
        "src": str(tmp_path / "g1"),
        "run_dir": str(repository / "output" / "hard-gate-test"),
        "dry_run": False,
    }
    with pytest.raises(CommandValidationError, match="checkpoint"):
        build_command(
            workflow,
            repository_root=repository,
            output_root=repository / "output",
            preset_id="offline_g1",
            supplied=supplied,
            strict=True,
        )
    supplied["dry_run"] = True
    preview = build_command(
        workflow,
        repository_root=repository,
        output_root=repository / "output",
        preset_id="offline_g1",
        supplied=supplied,
        strict=True,
    )
    assert any("许可证" in warning for warning in preview.warnings)


def test_offline_snapshot_distinguishes_weak_map_completion(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    run = output / "offline-run"
    _write_json(
        run / "mapping_run.json",
        {
            "status": "complete",
            "stages_completed": list(OFFLINE_STAGES),
            "stage_results": {stage: "executed" for stage in OFFLINE_STAGES},
            "loop_closures": {"verified_count": 2},
            "temporal_validation": {
                "gate": {
                    "passed": True,
                    "checks": [
                        {
                            "metric": "agreement",
                            "actual": 0.91,
                            "operator": ">=",
                            "threshold": 0.85,
                            "passed": True,
                        }
                    ],
                }
            },
            "direct_rgbd_fusion": {"manually_accepted": True},
        },
    )
    snapshot = run_snapshot(output, "offline-run")
    assert snapshot["status"] == "warning"
    assert snapshot["node_states"]["map"]["status"] == "warning"
    assert snapshot["progress"] == 1.0
    assert any("不可靠" in warning for warning in snapshot["warnings"])


def test_offline_depth_commits_keep_external_foundation_run_live(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    run = output / "offline-depth"
    _write_json(
        run / "mapping_run.json",
        {
            "status": "running",
            "stages_completed": ["prepare", "select"],
            "stage_results": {"prepare": "executed", "select": "executed"},
        },
    )
    _write_json(
        run / "02_selected" / "depth_metadata" / "00000000.json",
        {"frame_idx": 0},
    )
    _write_json(
        run / "02_selected" / "keyframe_selection_report.json",
        {"selected_frame_count": 2},
    )

    snapshot = run_snapshot(output, "offline-depth")
    assert snapshot["status"] == "running"
    assert snapshot["node_states"]["depth"]["status"] == "running"
    assert snapshot["node_states"]["depth"]["message"] == "检测到阶段产物持续更新"
    assert snapshot["node_states"]["depth"]["progress"] == pytest.approx(0.5)
    assert snapshot["node_states"]["depth"]["metrics"] == {
        "completed_frames": 1,
        "total_frames": 2,
        "progress": 0.5,
    }
    assert snapshot["progress"] == pytest.approx(2.5 / len(OFFLINE_STAGES))
    assert snapshot["started_at"]
    [summary] = discover_runs(output)
    assert summary["status"] == "running"
    assert summary["progress"] == pytest.approx(2.5 / len(OFFLINE_STAGES))
    assert summary["started_at"]


def test_realtime_snapshot_uses_checkpoint_report_commit_and_quality(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    run = output / "realtime-run"
    _write_json(run / "run_manifest.json", {"configuration": {"semantic": {"mode": "dam"}}})
    _write_json(
        run / "realtime_checkpoint.json",
        {
            "completed_frame_indices": [0, 1],
            "dropped_frames": {},
            "map_revision": 2,
        },
    )
    _write_json(
        run / "realtime_run_report.json",
        {
            "status": "complete",
            "frames_requested": 2,
            "frames_by_stage": {stage: 2 for stage in ("pose", "depth", "dynamic", "fusion", "global")},
            "semantic_mode": "dam",
            "semantic_stats": {"segmentation_failures": 0, "tracking_failures": 0},
            "dam_runtime_gate": {"passed": True},
            "semantic_postpass": {
                "status": "complete",
                "frames_expected": 2,
                "frames_replayed": 2,
                "label_coverage": 1.0,
                "missing_frame_indices": [],
            },
        },
    )
    _write_json(
        run / "quality_report.json",
        {
            "passed": True,
            "hard_failures": 0,
            "warnings": 0,
            "results": [
                {
                    "code": "time.contract",
                    "stage": "time",
                    "status": "PASS",
                    "hard": True,
                    "message": "ok",
                    "metrics": {},
                    "thresholds": {},
                    "blocks_pipeline": False,
                }
            ],
        },
    )
    _write_json(run / "hydra_semantic_postpass.json", {"status": "complete", "label_coverage": 1.0})
    _write_json(
        run / "hydra_realtime" / "backend" / "semantic_dsg_commit.json",
        {"status": "complete", "hash_verified": True},
    )
    snapshot = run_snapshot(output, "realtime-run")
    assert snapshot["status"] == "succeeded"
    assert snapshot["progress"] == 1.0
    assert all(
        snapshot["node_states"][node]["status"] == "succeeded"
        for node in ("pose", "depth", "dynamic", "fusion", "global", "dam", "postpass", "commit", "quality")
    )
    assert snapshot["quality_gates"][0]["code"] == "time.contract"


def test_discovery_scans_only_first_level_markers(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _write_json(
        output / "offline" / "mapping_run.json",
        {"status": "planned", "stages_completed": [], "stage_results": {}},
    )
    _write_json(
        output / "realtime" / "realtime_checkpoint.json",
        {"completed_frame_indices": [], "dropped_frames": {}},
    )
    _write_json(
        output / "nested-parent" / "nested" / "mapping_run.json",
        {"status": "complete", "stages_completed": []},
    )
    runs = discover_runs(output)
    assert {run["id"] for run in runs} == {"offline", "realtime"}


def test_realtime_dry_run_is_listed_as_planned(tmp_path: Path) -> None:
    output = tmp_path / "output"
    run = output / "dry-run"
    _write_json(run / "run_manifest.json", {"configuration": {}})
    _write_json(run / "dry_run_plan.json", {"status": "planned"})
    [summary] = discover_runs(output)
    assert summary["status"] == "planned"
    assert run_snapshot(output, "dry-run")["status"] == "planned"


def test_dashboard_api_and_fake_process_lifecycle(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    output = repository / "output"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    (tmp_path / "dataset").mkdir()
    fake_runner = scripts / "run_realtime_mapping.py"
    fake_runner.write_text(
        "import sys, time\n"
        "print('fake mapping started', flush=True)\n"
        "time.sleep(0.05)\n"
        "print('fake mapping complete', flush=True)\n"
    )
    app = create_dashboard_app(repository, output_root=output)
    client = TestClient(app)

    assert client.get("/").status_code == 200
    workflows = client.get("/api/workflows")
    assert workflows.status_code == 200
    assert workflows.json()["default_workflow"] == "realtime_semantic"

    body = {
        "workflow_id": "realtime_semantic",
        "preset_id": "realtime_geometry",
        "parameters": {
            "dataset": str(tmp_path / "dataset"),
            "run_dir": str(output / "fake-run"),
        },
    }
    preview = client.post("/api/commands/preview", json=body)
    assert preview.status_code == 200
    assert isinstance(preview.json()["argv"], list)

    started = client.post("/api/runs", json=body)
    assert started.status_code == 202
    assert started.json()["started_at"].endswith("+00:00")
    process_id = started.json()["process_id"]
    status = "running"
    deadline = time.monotonic() + 3.0
    while status in {"starting", "running"} and time.monotonic() < deadline:
        response = client.get(f"/api/processes/{process_id}")
        assert response.status_code == 200
        status = response.json()["status"]
        time.sleep(0.02)
    assert status == "succeeded"
    events = client.get(f"/api/processes/{process_id}/events?after=0")
    assert events.status_code == 200
    messages = [event["message"] for event in events.json()["events"]]
    assert any("fake mapping started" in message for message in messages)
    assert any("fake mapping complete" in message for message in messages)


def test_artifact_endpoint_rejects_path_traversal(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "scripts").mkdir(parents=True)
    output = repository / "output"
    _write_json(
        output / "run" / "mapping_run.json",
        {"status": "planned", "stages_completed": [], "stage_results": {}},
    )
    outside = output / "secret.json"
    outside.write_text("{}")
    client = TestClient(create_dashboard_app(repository, output_root=output))
    response = client.get("/api/runs/run/artifact", params={"path": "../secret.json"})
    assert response.status_code == 403


def test_depth_preview_lists_committed_frames_and_renders_fixed_rainbow(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    (repository / "scripts").mkdir(parents=True)
    output = repository / "output"
    run = output / "offline"
    _write_json(
        run / "mapping_run.json",
        {
            "status": "running",
            "stages_completed": ["prepare", "select", "depth"],
            "stage_results": {"depth": "executed"},
        },
    )
    depth_dir = run / "02_selected" / "depth"
    depth_dir.mkdir(parents=True)
    depth = np.array([[0, 250, 1000, 3000], [500, 1500, 2250, 3500]], dtype=np.uint16)
    assert cv2.imwrite(str(depth_dir / "00000000.png"), depth)
    assert cv2.imwrite(str(depth_dir / "00000002.png"), depth)
    _write_json(
        run / "02_selected" / "depth_metadata" / "00000000.json",
        {"frame_idx": 0, "valid_ratio": 0.875},
    )
    _write_json(
        run / "02_selected" / "foundation_stereo_run.json",
        {"status": "complete", "processed": 1, "maximum_depth_m": 3.0},
    )

    client = TestClient(create_dashboard_app(repository, output_root=output))
    index = client.get("/api/runs/offline/depth-frames")
    assert index.status_code == 200
    depth_index = index.json()
    assert depth_index["available"] is True
    assert depth_index["source"] == "foundation-stereo"
    assert depth_index["frames"] == [0]
    assert depth_index["count"] == 1
    assert depth_index["latest_frame"] == 0
    assert depth_index["complete"] is True
    assert depth_index["live"] is False
    assert depth_index["updated_at"]
    assert depth_index["has_more"] is False
    assert depth_index["minimum_depth_m"] == 0.25
    assert depth_index["maximum_depth_m"] == 3.0
    assert depth_index["range_source"] == "foundation_stereo_report"
    assert depth_index["colormap"] == "turbo"

    preview = client.get(
        "/api/runs/offline/depth-frames/0.png",
        params={"minimum_m": 0.25, "maximum_m": 3.0},
    )
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
    assert float(preview.headers["x-depth-valid-ratio"]) == pytest.approx(0.875)
    rendered = cv2.imdecode(np.frombuffer(preview.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert rendered.shape == (2, 4, 3)
    assert tuple(rendered[0, 0]) == (245, 248, 252)
    assert not np.array_equal(rendered[0, 2], rendered[0, 3])

    uncommitted = client.get("/api/runs/offline/depth-frames/2.png")
    assert uncommitted.status_code == 404
