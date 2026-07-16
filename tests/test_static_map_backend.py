"""Acceptance tests for static-only Hydra map integration."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import cv2
import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.mapping.backends import HydraStaticMapBackend, matrix_to_xyzw  # noqa: E402

sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
from run_realtime_mapping import ReplayFrame, rebuild_static_map_prefix  # noqa: E402


ORIGIN_NS = 1_783_933_507_759_540_877


def test_g1_hydra_mesh_rejects_clamped_extra_distance_padding():
    configuration = yaml.safe_load(
        (REPOSITORY_ROOT / "config" / "hydra_g1_high_quality.yaml").read_text()
    )
    active_window = configuration["active_window"]
    projective = active_window["projective_integrator"]
    assert projective["extra_integration_distance"] > 0.0
    assert active_window["mesh_integrator"]["min_weight"] > projective.get(
        "min_measurement_weight", 1.0e-4
    )


class FakeHydraIntegration:
    def __init__(self, **configuration):
        self.configuration = configuration
        self.camera = None
        self.frames = []
        self.saved = False
        self.closed = False
        self.save_on_shutdown = None

    def initialize_camera(self, **camera):
        self.camera = camera

    def initialize_pipeline(self):
        return True

    def process_frame(self, **frame):
        self.frames.append(frame)
        return True

    def save_results(self, _output):
        self.saved = True
        return True

    def get_stats(self):
        return {"frames_processed": len(self.frames)}

    def shutdown(self, *, save_results=True):
        self.save_on_shutdown = save_results
        self.closed = True


def test_matrix_to_xyzw_preserves_rotation_and_translation_convention():
    angle = np.pi / 2.0
    transform = np.eye(4)
    transform[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    quaternion = matrix_to_xyzw(transform)
    assert np.allclose(
        np.abs(quaternion),
        [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
    )


def test_hydra_backend_receives_static_depth_and_original_absolute_time(tmp_path):
    config = tmp_path / "hydra.yaml"
    config.write_text("frontend: {}\n")
    created = []

    def factory(**kwargs):
        integration = FakeHydraIntegration(**kwargs)
        created.append(integration)
        return integration

    backend = HydraStaticMapBackend(
        config,
        tmp_path / "map",
        integration_factory=factory,
    )
    intrinsics = np.array([[80.0, 0.0, 31.5], [0.0, 81.0, 23.5], [0.0, 0.0, 1.0]])
    backend.initialize(64, 48, intrinsics)
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    depth = np.ones((48, 64), dtype=np.float32)
    depth[10:20, 10:20] = 0.0
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    backend.integrate(
        sensor_time_ns=ORIGIN_NS,
        rgb_image=rgb,
        static_depth_m=depth,
        world_T_camera=pose,
    )
    backend.integrate(
        sensor_time_ns=ORIGIN_NS + 100_000_000,
        rgb_image=rgb,
        static_depth_m=depth,
        world_T_camera=pose,
    )
    backend.finalize()

    integration = created[0]
    assert integration.camera["fx"] == 80.0
    assert integration.frames[0]["timestamp_ns"] == ORIGIN_NS
    assert integration.frames[0]["timestamp"] == 0.0
    assert integration.frames[1]["timestamp"] == 0.1
    assert np.array_equal(integration.frames[0]["depth_image"], depth)
    assert np.allclose(integration.frames[0]["transform"][:3], [1.0, 2.0, 3.0])
    assert integration.saved
    assert backend.stats()["frames_processed"] == 2
    backend.close()
    assert integration.closed
    assert integration.save_on_shutdown is False


def test_hydra_close_without_finalize_only_releases_pipeline(tmp_path):
    config = tmp_path / "hydra.yaml"
    config.write_text("frontend: {}\n")
    integration = FakeHydraIntegration()
    backend = HydraStaticMapBackend(
        config,
        tmp_path / "map",
        integration_factory=lambda **_kwargs: integration,
    )
    backend.initialize(64, 48, np.eye(3))

    backend.close(finalize=False)

    assert not integration.saved
    assert integration.closed
    assert integration.save_on_shutdown is False
    assert not backend.stats()["finalized"]


def test_hydra_close_shuts_down_when_finalize_fails(tmp_path):
    class FailingSaveIntegration(FakeHydraIntegration):
        def save_results(self, _output):
            self.saved = True
            return False

    config = tmp_path / "hydra.yaml"
    config.write_text("frontend: {}\n")
    integration = FailingSaveIntegration()
    backend = HydraStaticMapBackend(
        config,
        tmp_path / "map",
        integration_factory=lambda **_kwargs: integration,
    )
    backend.initialize(64, 48, np.eye(3))

    with pytest.raises(RuntimeError, match="failed to save"):
        backend.close()

    assert integration.saved
    assert integration.closed
    backend.close()


def test_hydra_resume_rebuilds_only_committed_frames_with_absolute_time(tmp_path):
    class RecordingBackend:
        def __init__(self):
            self.frames = []

        def integrate(self, **frame):
            self.frames.append(frame)

    run_dir = tmp_path / "run"
    (run_dir / "static_depth").mkdir(parents=True)
    rgb_path = tmp_path / "rgb.png"
    right_path = tmp_path / "right.png"
    rgb = np.full((4, 6, 3), [10, 20, 30], dtype=np.uint8)
    cv2.imwrite(str(rgb_path), rgb)
    cv2.imwrite(str(right_path), rgb)
    for index in (0, 2):
        cv2.imwrite(
            str(run_dir / "static_depth" / f"{index:08d}.png"),
            np.full((4, 6), 1250 + index, dtype=np.uint16),
        )
    frames = []
    for index in range(3):
        pose = np.eye(4)
        pose[0, 3] = index
        frames.append(
            ReplayFrame(
                frame_index=index,
                sensor_time_ns=ORIGIN_NS + index * 100_000_000,
                rgb_path=rgb_path,
                right_path=right_path,
                depth_path=tmp_path / "unused.png",
                confidence_path=tmp_path / "unused-confidence.png",
                consistency_path=tmp_path / "unused-consistency.png",
                depth_metadata_path=tmp_path / "unused.json",
                world_T_camera=pose,
                intrinsics=np.eye(3),
                value=1.0,
            )
        )
    backend = RecordingBackend()
    rebuilt = rebuild_static_map_prefix(backend, run_dir, frames, {0, 2})

    assert rebuilt == 2
    assert [frame["sensor_time_ns"] for frame in backend.frames] == [
        ORIGIN_NS,
        ORIGIN_NS + 200_000_000,
    ]
    assert np.allclose(backend.frames[0]["static_depth_m"], 1.25)
    assert np.allclose(backend.frames[1]["world_T_camera"][:3, 3], [2, 0, 0])
