"""Tests for dependency-free mesh fragmentation evidence."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.quality.mesh import analyze_ascii_ply_mesh  # noqa: E402


def test_mesh_metrics_find_disconnected_components_and_largest_ratio(tmp_path):
    mesh = tmp_path / "mesh.ply"
    mesh.write_text(
        """ply
format ascii 1.0
element vertex 7
property float x
property float y
property float z
element face 2
property list uchar uint vertex_indices
end_header
0 0 0
1 0 0
0 1 0
10 0 0
11 0 0
10 1 0
99 99 99
3 0 1 2
3 3 4 5
"""
    )
    metrics = analyze_ascii_ply_mesh(mesh)
    assert metrics["vertices"] == 7
    assert metrics["triangles"] == 2
    assert metrics["connected_components"] == 2
    assert metrics["largest_component_vertices"] == 3
    assert metrics["largest_component_ratio"] == 0.5
    assert metrics["isolated_vertices"] == 1
    assert metrics["significant_connected_components"] == 2
    assert metrics["tiny_component_area_ratio"] == 0.0


def test_mesh_metrics_weld_duplicate_voxel_block_boundary_vertices(tmp_path):
    mesh = tmp_path / "block-mesh.ply"
    mesh.write_text(
        """ply
format ascii 1.0
element vertex 6
property float x
property float y
property float z
element face 2
property list uchar uint vertex_indices
end_header
0 0 0
1 0 0
0 1 0
1.00000001 0 0
0 1.00000001 0
1 1 0
3 0 1 2
3 3 4 5
"""
    )
    metrics = analyze_ascii_ply_mesh(mesh, weld_tolerance_m=1.0e-4)
    assert metrics["raw_connected_components"] == 2
    assert metrics["connected_components"] == 1
    assert metrics["welded_vertices"] == 4
    assert metrics["largest_component_ratio"] == 1.0
    assert metrics["significant_connected_components"] == 1
    assert metrics["largest_component_area_ratio"] == 1.0


def test_mesh_metrics_separate_tiny_surface_islands_from_significant_components(
    tmp_path,
):
    mesh = tmp_path / "tiny-island.ply"
    mesh.write_text(
        """ply
format ascii 1.0
element vertex 6
property float x
property float y
property float z
element face 2
property list uchar uint vertex_indices
end_header
0 0 0
1 0 0
0 1 0
10 0 0
10.01 0 0
10 0.01 0
3 0 1 2
3 3 4 5
"""
    )
    metrics = analyze_ascii_ply_mesh(mesh)
    assert metrics["connected_components"] == 2
    assert metrics["significant_connected_components"] == 1
    assert metrics["tiny_component_area_m2"] > 0.0
    assert metrics["tiny_component_area_ratio"] < 0.001
