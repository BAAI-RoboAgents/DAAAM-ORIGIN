"""Contracts for durable semantic labels and deterministic Hydra replay."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from daaam.realtime.semantic_labels import (  # noqa: E402
    load_semantic_label,
    persist_semantic_label,
    semantic_label_path,
    validate_semantic_label_binding,
)
import run_realtime_mapping as realtime_module  # noqa: E402
from run_hydra_semantic_postpass import (  # noqa: E402
    validate_plan_frame_records,
)
from run_realtime_mapping import (  # noqa: E402
    ReplayFrame,
    rebuild_and_promote_semantic_hydra,
    rebuild_static_map_with_semantics,
)


ORIGIN_NS = 1_783_933_507_759_540_877
RUN_CONFIGURATION_SHA256 = "a" * 64


class RecordingBackend:
    def __init__(self):
        self.frames = []

    def integrate(self, **frame):
        self.frames.append(frame)


class FakeLiveBackend:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.hydra_config_path = output_dir.parent / "hydra.yaml"
        self.labelspace_path = None
        self.labelspace_colors = None
        self.maximum_depth_m = 20.0
        self.closed = False

    def stats(self):
        return {"frames_processed": 2}

    def close(self, *, finalize: bool):
        assert finalize is False
        self.closed = True


class FakeAdoptedBackend:
    fail_adoption = False

    def __init__(
        self,
        hydra_config_path,
        output_dir,
        *,
        labelspace_path=None,
        labelspace_colors=None,
        maximum_depth_m=20.0,
    ):
        self.output_dir = Path(output_dir)

    def adopt_finalized_output(self, stats):
        if self.fail_adoption:
            raise RuntimeError("synthetic adoption failure")
        assert (self.output_dir / "backend" / "mesh.ply").is_file()
        assert (self.output_dir / "backend" / "dsg.json").is_file()


def _frame(root: Path, index: int) -> ReplayFrame:
    rgb_path = root / f"rgb-{index}.png"
    right_path = root / f"right-{index}.png"
    rgb = np.full((4, 6, 3), [10 + index, 20, 30], dtype=np.uint8)
    cv2.imwrite(str(rgb_path), rgb)
    cv2.imwrite(str(right_path), rgb)
    return ReplayFrame(
        frame_index=index,
        sensor_time_ns=ORIGIN_NS + index * 100_000_000,
        rgb_path=rgb_path,
        right_path=right_path,
        depth_path=root / "unused-depth.png",
        confidence_path=root / "unused-confidence.png",
        consistency_path=root / "unused-consistency.png",
        depth_metadata_path=root / "unused.json",
        world_T_camera=np.eye(4),
        intrinsics=np.eye(3),
        value=1.0,
    )


def _promotion_inputs(tmp_path: Path):
    run_dir = tmp_path / "run"
    final_output = run_dir / "hydra_realtime"
    final_output.mkdir(parents=True)
    (final_output / "live-marker.txt").write_text("original-live-output\n")
    frame = _frame(tmp_path, 0)
    label_dir = run_dir / "semantic_sidecar" / "label_frames"
    persist_semantic_label(
        label_dir,
        0,
        np.ones((4, 6), dtype=np.uint16),
        sensor_time_ns=frame.sensor_time_ns,
        run_configuration_sha256=RUN_CONFIGURATION_SHA256,
    )
    return run_dir, final_output, FakeLiveBackend(final_output), frame, label_dir


def _successful_child(command, *, stdout, stderr, check, timeout):
    del stdout, stderr
    assert check is False
    assert timeout >= 300.0
    plan_path = Path(command[command.index("--plan") + 1])
    report_path = Path(command[command.index("--report") + 1])
    plan = json.loads(plan_path.read_text())
    output_dir = Path(plan["output_dir"])
    backend_dir = output_dir / "backend"
    backend_dir.mkdir(parents=True)
    (backend_dir / "mesh.ply").write_text("ply\n")
    (backend_dir / "dsg.json").write_text("{}\n")
    report_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "frames_expected": 1,
                "frames_replayed": 1,
                "frames_with_labels": 1,
                "label_coverage": 1.0,
                "missing_frame_indices": [],
                "label_run_configuration_sha256": (
                    plan["label_run_configuration_sha256"]
                ),
                "backend_stats": {"finalized": True},
            }
        )
    )
    return SimpleNamespace(returncode=0)


def test_semantic_label_png_is_atomic_lossless_uint16(tmp_path):
    labels = np.asarray([[0, 1], [255, 65_535]], dtype=np.int32)

    record = persist_semantic_label(
        tmp_path,
        7,
        labels,
        sensor_time_ns=ORIGIN_NS,
        run_configuration_sha256=RUN_CONFIGURATION_SHA256,
    )

    assert semantic_label_path(tmp_path, 7).is_file()
    assert np.array_equal(load_semantic_label(tmp_path, 7), labels)
    assert record["maximum_label"] == 65_535
    assert record["nonzero_pixels"] == 3
    assert validate_semantic_label_binding(
        tmp_path,
        7,
        sensor_time_ns=ORIGIN_NS,
        run_configuration_sha256=RUN_CONFIGURATION_SHA256,
    )["image_sha256"] == record["sha256"]
    with pytest.raises(ValueError, match="sensor-time binding mismatch"):
        validate_semantic_label_binding(
            tmp_path,
            7,
            sensor_time_ns=ORIGIN_NS + 1,
            run_configuration_sha256=RUN_CONFIGURATION_SHA256,
        )
    assert not list(tmp_path.glob(".*.tmp.png"))


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([[-1]], dtype=np.int32),
        np.asarray([[65_536]], dtype=np.int32),
        np.asarray([[1.5]], dtype=np.float32),
    ],
)
def test_semantic_label_png_rejects_lossy_values(tmp_path, labels):
    with pytest.raises(ValueError):
        persist_semantic_label(
            tmp_path,
            0,
            labels,
            sensor_time_ns=ORIGIN_NS,
            run_configuration_sha256=RUN_CONFIGURATION_SHA256,
        )

    assert not semantic_label_path(tmp_path, 0).exists()


def test_hydra_postpass_replays_exact_labels_with_full_coverage(tmp_path):
    run_dir = tmp_path / "run"
    static_dir = run_dir / "static_depth"
    label_dir = run_dir / "semantic_sidecar" / "label_frames"
    static_dir.mkdir(parents=True)
    frames = [_frame(tmp_path, index) for index in range(2)]
    expected_labels = []
    for index in range(2):
        cv2.imwrite(
            str(static_dir / f"{index:08d}.png"),
            np.full((4, 6), 1_250 + index, dtype=np.uint16),
        )
        labels = np.zeros((4, 6), dtype=np.uint16)
        labels[index : index + 2, 1:4] = index + 1
        persist_semantic_label(
            label_dir,
            index,
            labels,
            sensor_time_ns=frames[index].sensor_time_ns,
            run_configuration_sha256=RUN_CONFIGURATION_SHA256,
        )
        expected_labels.append(labels)
    backend = RecordingBackend()

    report = rebuild_static_map_with_semantics(
        backend,
        run_dir,
        frames,
        {0, 1},
        label_dir,
        RUN_CONFIGURATION_SHA256,
    )

    assert report["frames_replayed"] == 2
    assert report["label_coverage"] == 1.0
    assert report["nonzero_label_frames"] == 2
    assert report["unique_semantic_labels"] == [0, 1, 2]
    assert len(report["label_manifest_sha256"]) == 64
    assert [frame["sensor_time_ns"] for frame in backend.frames] == [
        ORIGIN_NS,
        ORIGIN_NS + 100_000_000,
    ]
    assert np.array_equal(
        backend.frames[0]["semantic_labels"], expected_labels[0]
    )
    assert np.array_equal(
        backend.frames[1]["semantic_labels"], expected_labels[1]
    )


def test_hydra_postpass_plan_rejects_duplicate_and_out_of_order_frames():
    records = [
        {"frame_index": 0, "sensor_time_ns": ORIGIN_NS},
        {"frame_index": 0, "sensor_time_ns": ORIGIN_NS + 1},
    ]
    with pytest.raises(ValueError, match="indices must be unique"):
        validate_plan_frame_records(records)

    records[1]["frame_index"] = 1
    assert validate_plan_frame_records(records) == records
    with pytest.raises(ValueError, match="ordered by frame index"):
        validate_plan_frame_records(list(reversed(records)))


def test_hydra_postpass_plan_rejects_nonmonotonic_sensor_time():
    records = [
        {"frame_index": 0, "sensor_time_ns": ORIGIN_NS},
        {"frame_index": 1, "sensor_time_ns": ORIGIN_NS},
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_plan_frame_records(records)


def test_hydra_postpass_fails_before_fusion_when_one_label_is_missing(tmp_path):
    run_dir = tmp_path / "run"
    static_dir = run_dir / "static_depth"
    label_dir = run_dir / "semantic_sidecar" / "label_frames"
    static_dir.mkdir(parents=True)
    frames = [_frame(tmp_path, index) for index in range(2)]
    for index in range(2):
        cv2.imwrite(
            str(static_dir / f"{index:08d}.png"),
            np.full((4, 6), 1_250, dtype=np.uint16),
        )
    persist_semantic_label(
        label_dir,
        0,
        np.zeros((4, 6), dtype=np.uint16),
        sensor_time_ns=frames[0].sensor_time_ns,
        run_configuration_sha256=RUN_CONFIGURATION_SHA256,
    )
    backend = RecordingBackend()

    with pytest.raises(RuntimeError, match="100% exact-frame label coverage"):
        rebuild_static_map_with_semantics(
            backend,
            run_dir,
            frames,
            {0, 1},
            label_dir,
            RUN_CONFIGURATION_SHA256,
        )

    assert backend.frames == []


def test_hydra_postpass_rejects_stale_run_binding_before_fusion(tmp_path):
    run_dir = tmp_path / "run"
    static_dir = run_dir / "static_depth"
    label_dir = run_dir / "semantic_sidecar" / "label_frames"
    static_dir.mkdir(parents=True)
    frame = _frame(tmp_path, 0)
    cv2.imwrite(
        str(static_dir / "00000000.png"),
        np.full((4, 6), 1_250, dtype=np.uint16),
    )
    persist_semantic_label(
        label_dir,
        0,
        np.zeros((4, 6), dtype=np.uint16),
        sensor_time_ns=frame.sensor_time_ns,
        run_configuration_sha256=RUN_CONFIGURATION_SHA256,
    )
    backend = RecordingBackend()

    with pytest.raises(RuntimeError, match="run-configuration binding mismatch"):
        rebuild_static_map_with_semantics(
            backend,
            run_dir,
            [frame],
            {0},
            label_dir,
            "b" * 64,
        )

    assert backend.frames == []


def test_postpass_promotion_replaces_output_only_after_validation(
    tmp_path, monkeypatch
):
    run_dir, final_output, live_backend, frame, label_dir = _promotion_inputs(
        tmp_path
    )
    monkeypatch.setattr(realtime_module.subprocess, "run", _successful_child)
    FakeAdoptedBackend.fail_adoption = False
    monkeypatch.setattr(
        realtime_module, "HydraStaticMapBackend", FakeAdoptedBackend
    )

    adopted, report = rebuild_and_promote_semantic_hydra(
        live_backend,
        run_dir,
        [frame],
        {0},
        label_dir,
        RUN_CONFIGURATION_SHA256,
    )

    assert live_backend.closed is True
    assert adopted.output_dir == final_output
    assert not (final_output / "live-marker.txt").exists()
    assert (final_output / "backend" / "mesh.ply").is_file()
    assert report["postpass_promoted"] is True
    assert report["postpass_timeout_seconds"] >= 300.0
    assert not list(run_dir.glob(".hydra_realtime*.bak"))
    assert not (run_dir / ".hydra_realtime.semantic-postpass.tmp").exists()


def test_postpass_adoption_failure_rolls_back_original_output(
    tmp_path, monkeypatch
):
    run_dir, final_output, live_backend, frame, label_dir = _promotion_inputs(
        tmp_path
    )
    monkeypatch.setattr(realtime_module.subprocess, "run", _successful_child)
    FakeAdoptedBackend.fail_adoption = True
    monkeypatch.setattr(
        realtime_module, "HydraStaticMapBackend", FakeAdoptedBackend
    )

    with pytest.raises(RuntimeError, match="synthetic adoption failure"):
        rebuild_and_promote_semantic_hydra(
            live_backend,
            run_dir,
            [frame],
            {0},
            label_dir,
            RUN_CONFIGURATION_SHA256,
        )

    FakeAdoptedBackend.fail_adoption = False
    assert (final_output / "live-marker.txt").read_text() == (
        "original-live-output\n"
    )
    assert not (final_output / "backend").exists()
    assert not list(run_dir.glob(".hydra_realtime*.bak"))
    assert not (run_dir / ".hydra_realtime.semantic-postpass.tmp").exists()


def test_postpass_incomplete_temp_never_replaces_original_output(
    tmp_path, monkeypatch
):
    run_dir, final_output, live_backend, frame, label_dir = _promotion_inputs(
        tmp_path
    )

    def incomplete_child(command, *, stdout, stderr, check, timeout):
        result = _successful_child(
            command,
            stdout=stdout,
            stderr=stderr,
            check=check,
            timeout=timeout,
        )
        plan_path = Path(command[command.index("--plan") + 1])
        output_dir = Path(json.loads(plan_path.read_text())["output_dir"])
        (output_dir / "backend" / "mesh.ply").unlink()
        return result

    monkeypatch.setattr(realtime_module.subprocess, "run", incomplete_child)

    with pytest.raises(RuntimeError, match="output is incomplete"):
        rebuild_and_promote_semantic_hydra(
            live_backend,
            run_dir,
            [frame],
            {0},
            label_dir,
            RUN_CONFIGURATION_SHA256,
        )

    assert (final_output / "live-marker.txt").is_file()
    assert not (run_dir / ".hydra_realtime.semantic-postpass.tmp").exists()


def test_postpass_timeout_preserves_original_and_cleans_temporary_output(
    tmp_path, monkeypatch
):
    run_dir, final_output, live_backend, frame, label_dir = _promotion_inputs(
        tmp_path
    )

    def timeout_child(command, *, stdout, stderr, check, timeout):
        del stdout, stderr, check
        plan_path = Path(command[command.index("--plan") + 1])
        output_dir = Path(json.loads(plan_path.read_text())["output_dir"])
        output_dir.mkdir(parents=True)
        (output_dir / "partial.txt").write_text("partial\n")
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(realtime_module.subprocess, "run", timeout_child)

    with pytest.raises(RuntimeError, match="exceeded its timeout"):
        rebuild_and_promote_semantic_hydra(
            live_backend,
            run_dir,
            [frame],
            {0},
            label_dir,
            RUN_CONFIGURATION_SHA256,
        )

    assert (final_output / "live-marker.txt").is_file()
    assert not (run_dir / ".hydra_realtime.semantic-postpass.tmp").exists()
