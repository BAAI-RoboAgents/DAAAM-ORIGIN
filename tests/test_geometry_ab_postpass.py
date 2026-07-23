"""Tests for isolated, read-only semantic geometry A/B reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import run_geometry_ab_postpass as geometry_ab  # noqa: E402


def _source_run(root: Path, *, coverage: float = 1.0) -> Path:
    source = (root / "source-a").resolve()
    label_dir = source / "semantic_sidecar" / "label_frames"
    static_dir = source / "static_depth"
    label_dir.mkdir(parents=True)
    static_dir.mkdir(parents=True)
    rgb = source / "rgb.png"
    rgb.write_bytes(b"rgb")
    (label_dir / "00000000.png").write_bytes(b"labels")
    (static_dir / "00000000.png").write_bytes(b"depth")
    plan = {
        "schema": "daaam.hydra_semantic_postpass_plan.v1",
        "run_dir": str(source),
        "output_dir": str(source / "hydra_realtime"),
        "semantic_label_dir": str(label_dir),
        "hydra_config_path": str(source / "a.yaml"),
        "labelspace_path": None,
        "labelspace_colors": None,
        "maximum_depth_m": 20.0,
        "frames": [
            {
                "frame_index": 0,
                "sensor_time_ns": 1_784_000_000_000_000_000,
                "rgb_path": str(rgb),
                "world_T_camera": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "intrinsics": [
                    [60.0, 0.0, 31.5],
                    [0.0, 60.0, 23.5],
                    [0.0, 0.0, 1.0],
                ],
            }
        ],
    }
    report = {
        "schema": "daaam.hydra_semantic_postpass.v1",
        "status": "complete",
        "frames_expected": 1,
        "frames_replayed": 1,
        "frames_with_labels": 1,
        "label_coverage": coverage,
        "missing_frame_indices": [] if coverage == 1.0 else [0],
        "label_manifest_sha256": "a" * 64,
    }
    plan_path = source / geometry_ab.POSTPASS_PLAN
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan))
    (source / geometry_ab.POSTPASS_REPORT).write_text(json.dumps(report))
    return source


def test_geometry_ab_rewrites_only_b_plan_and_runs_isolated_child(
    tmp_path, monkeypatch
):
    source = _source_run(tmp_path)
    output = (tmp_path / "output-b").resolve()
    config = tmp_path / "hydra-b.yaml"
    config.write_text("frontend: {}\n")
    source_plan_before = (source / geometry_ab.POSTPASS_PLAN).read_bytes()
    source_report_before = (source / geometry_ab.POSTPASS_REPORT).read_bytes()
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        plan_path = Path(command[command.index("--plan") + 1])
        report_path = Path(command[command.index("--report") + 1])
        plan = json.loads(plan_path.read_text())
        backend = Path(plan["output_dir"]) / "backend"
        backend.mkdir(parents=True)
        (backend / "mesh.ply").write_bytes(b"mesh-b")
        (backend / "dsg.json").write_text("{}\n")
        report_path.write_text(
            json.dumps(
                {
                    "schema": "daaam.hydra_semantic_postpass.v1",
                    "status": "complete",
                    "frames_expected": 1,
                    "frames_replayed": 1,
                    "frames_with_labels": 1,
                    "label_coverage": 1.0,
                    "missing_frame_indices": [],
                    "label_manifest_sha256": "b" * 64,
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(geometry_ab.subprocess, "run", fake_run)

    report = geometry_ab.run_geometry_ab_postpass(
        source_run=source,
        hydra_config=config,
        output_run=output,
        timeout_s=123.0,
    )

    output_plan = json.loads((output / geometry_ab.POSTPASS_PLAN).read_text())
    assert output_plan["run_dir"] == str(source)
    assert output_plan["semantic_label_dir"] == str(
        source / "semantic_sidecar" / "label_frames"
    )
    assert output_plan["output_dir"] == str(output / "hydra_realtime")
    assert output_plan["hydra_config_path"] == str(config.resolve())
    assert observed["kwargs"]["timeout"] == 123.0
    assert observed["kwargs"]["check"] is False
    assert report["status"] == "complete"
    assert report["child_exit_code"] == 0
    assert report["frames_replayed"] == 1
    assert (output / "geometry_ab_run.json").is_file()
    assert (output / "hydra_realtime" / "backend" / "mesh.ply").is_file()
    assert (source / geometry_ab.POSTPASS_PLAN).read_bytes() == source_plan_before
    assert (source / geometry_ab.POSTPASS_REPORT).read_bytes() == source_report_before


def test_geometry_ab_rejects_incomplete_source_before_subprocess(
    tmp_path, monkeypatch
):
    source = _source_run(tmp_path, coverage=0.5)
    output = tmp_path / "output-b"
    config = tmp_path / "hydra-b.yaml"
    config.write_text("frontend: {}\n")

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("subprocess must not run for incomplete source A")

    monkeypatch.setattr(geometry_ab.subprocess, "run", unexpected_run)

    with pytest.raises(ValueError, match="complete semantic postpass contract"):
        geometry_ab.run_geometry_ab_postpass(
            source_run=source,
            hydra_config=config,
            output_run=output,
        )

    assert not output.exists()
