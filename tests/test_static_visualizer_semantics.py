"""Small contract tests for object-focused Rerun rendering helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from daaam.utils.static_visualizer import (
    StaticDSGVisualizer,
    discover_dense_rgbd_map,
    filter_mesh_faces_by_component_area,
    masked_image_card_geometry,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_semantic_fallback_colors_are_vivid_and_deterministic() -> None:
    first = StaticDSGVisualizer._semantic_color(15)
    repeated = StaticDSGVisualizer._semantic_color(15)
    other = StaticDSGVisualizer._semantic_color(16)

    assert first.dtype == np.uint8
    assert first.shape == (3,)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, other)
    assert int(first.max()) >= 240
    assert int(first.max()) - int(first.min()) >= 100


def test_masked_image_card_geometry_omits_transparent_cells() -> None:
    alpha = np.zeros((17, 17), dtype=np.uint8)
    alpha[4:13, 5:12] = 255
    vertices, faces, texcoords = masked_image_card_geometry(
        alpha,
        width_m=0.8,
        height_m=0.6,
        center=[1.0, 2.0, 3.0],
        face_direction=[0.0, -1.0],
        grid_step_px=4,
    )

    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert texcoords.shape == (len(vertices), 2)
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert len(faces) > 0
    assert len(faces) < 4 * 4 * 4
    assert np.all((texcoords >= 0.0) & (texcoords <= 1.0))
    assert np.isclose(vertices[:, 1], 2.0).all()


def test_mesh_component_filter_removes_only_tiny_fragment() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.01, 0.0, 0.0],
            [3.0, 0.01, 0.0],
        ]
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)

    filtered, stats = filter_mesh_faces_by_component_area(
        vertices, faces, minimum_area_m2=0.005
    )

    assert filtered.tolist() == [[0, 1, 2]]
    assert stats["removed_faces"] == 1
    assert np.isclose(stats["removed_area_m2"], 0.00005)


def test_dense_rgbd_map_discovery_requires_checksum_bound_acceptance(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "source_run"
    pose_path = run_root / "08_temporal_depth_filtered" / "pose" / "poses.txt"
    pose_path.parent.mkdir(parents=True)
    pose_path.write_text("pose-data", encoding="utf-8")
    dense_dir = run_root / "11_dense_rgbd_visualization_candidate"
    dense_dir.mkdir()
    cloud_path = dense_dir / "direct_rgbd_fusion.ply"
    report_path = dense_dir / "direct_rgbd_fusion_report.json"
    cloud_path.write_bytes(b"ply-data")
    report_path.write_text("{}", encoding="utf-8")
    (dense_dir / "visualization_acceptance.json").write_text(
        json.dumps(
            {
                "schema": "daaam.dense_rgbd_visualization_acceptance.v1",
                "accepted": True,
                "source_pose_sha256": _sha256(pose_path),
                "artifacts": {
                    "point_cloud": cloud_path.name,
                    "point_cloud_sha256": _sha256(cloud_path),
                    "report": report_path.name,
                    "report_sha256": _sha256(report_path),
                },
            }
        ),
        encoding="utf-8",
    )
    dsg_path = tmp_path / "query" / "dsg_updated.json"
    dsg_path.parent.mkdir()
    dsg_path.write_text("{}", encoding="utf-8")
    dsg_path.with_suffix(".evidence.json").write_text(
        json.dumps(
            {
                "source": {
                    "camera_pose_file": str(pose_path),
                    "camera_pose_file_sha256": _sha256(pose_path),
                }
            }
        ),
        encoding="utf-8",
    )

    assert discover_dense_rgbd_map(dsg_path) == cloud_path.resolve()

    cloud_path.write_bytes(b"tampered")
    assert discover_dense_rgbd_map(dsg_path) is None
