"""Dependency-free quality metrics for Hydra ASCII PLY triangle meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.component_size = [1] * size

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.component_size[first_root] < self.component_size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.component_size[first_root] += self.component_size[second_root]


def _component_counts(
    vertex_count: int,
    faces: list[list[int]],
) -> tuple[list[int], int]:
    components = _DisjointSet(vertex_count)
    used_vertices: set[int] = set()
    for indices in faces:
        used_vertices.update(indices)
        anchor = indices[0]
        for index in indices[1:]:
            components.union(anchor, index)
    counts: dict[int, int] = {}
    for vertex in used_vertices:
        root = components.find(vertex)
        counts[root] = counts.get(root, 0) + 1
    return list(counts.values()), len(used_vertices)


def _polygon_area(vertices: np.ndarray, indices: list[int]) -> float:
    """Return the area of a planar polygon using a triangle fan."""
    anchor = vertices[indices[0]]
    area = 0.0
    for offset in range(1, len(indices) - 1):
        first = vertices[indices[offset]] - anchor
        second = vertices[indices[offset + 1]] - anchor
        area += 0.5 * float(np.linalg.norm(np.cross(first, second)))
    return area


def _component_areas(
    vertex_count: int,
    topology_faces: list[list[int]],
    source_vertices: np.ndarray,
    source_faces: list[list[int]],
) -> list[float]:
    """Accumulate source-face area using welded topology connectivity."""
    if len(topology_faces) != len(source_faces):
        raise ValueError("topology and source face counts differ")
    components = _DisjointSet(vertex_count)
    for indices in topology_faces:
        anchor = indices[0]
        for index in indices[1:]:
            components.union(anchor, index)
    areas: dict[int, float] = {}
    for topology, source in zip(topology_faces, source_faces):
        root = components.find(topology[0])
        areas[root] = areas.get(root, 0.0) + _polygon_area(source_vertices, source)
    return list(areas.values())


def analyze_ascii_ply_mesh(
    path: Path | str,
    *,
    weld_tolerance_m: float = 1.0e-4,
    minimum_significant_component_area_m2: float = 5.0e-3,
) -> dict:
    mesh_path = Path(path)
    if weld_tolerance_m <= 0.0:
        raise ValueError("mesh weld tolerance must be positive")
    if minimum_significant_component_area_m2 <= 0.0:
        raise ValueError("minimum significant component area must be positive")
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    with mesh_path.open("r", encoding="ascii") as stream:
        first = stream.readline().strip()
        if first != "ply":
            raise ValueError("mesh is not a PLY file")
        vertex_count = face_count = None
        ascii_format = False
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header is incomplete")
            fields = line.strip().split()
            if fields[:3] == ["format", "ascii", "1.0"]:
                ascii_format = True
            elif fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
            elif fields[:2] == ["element", "face"]:
                face_count = int(fields[2])
            elif fields == ["end_header"]:
                break
        if not ascii_format:
            raise ValueError("only ASCII PLY meshes are supported")
        if vertex_count is None or face_count is None or vertex_count < 0 or face_count < 0:
            raise ValueError("PLY vertex/face counts are invalid")

        vertices = np.empty((vertex_count, 3), dtype=np.float64)
        for index in range(vertex_count):
            fields = stream.readline().split()
            if len(fields) < 3:
                raise ValueError(f"PLY vertex {index} is incomplete")
            vertices[index] = [float(fields[0]), float(fields[1]), float(fields[2])]
        if not np.all(np.isfinite(vertices)):
            raise ValueError("PLY contains non-finite vertices")

        valid_faces: list[list[int]] = []
        triangle_count = 0
        invalid_faces = 0
        for _ in range(face_count):
            fields = stream.readline().split()
            if not fields:
                invalid_faces += 1
                continue
            count = int(fields[0])
            if count < 3 or len(fields) < count + 1:
                invalid_faces += 1
                continue
            indices = [int(value) for value in fields[1 : count + 1]]
            if min(indices) < 0 or max(indices) >= vertex_count:
                invalid_faces += 1
                continue
            valid_faces.append(indices)
            triangle_count += count - 2

    raw_counts, raw_used_count = _component_counts(vertex_count, valid_faces)
    quantized = np.rint(vertices / weld_tolerance_m).astype(np.int64)
    _, welded_indices = np.unique(quantized, axis=0, return_inverse=True)
    welded_faces = [
        [int(welded_indices[index]) for index in face]
        for face in valid_faces
    ]
    welded_vertex_count = int(welded_indices.max()) + 1 if vertex_count else 0
    welded_counts, welded_used_count = _component_counts(
        welded_vertex_count,
        welded_faces,
    )
    welded_areas = _component_areas(
        welded_vertex_count,
        welded_faces,
        vertices,
        valid_faces,
    )
    connected_components = len(welded_counts)
    largest = max(welded_counts, default=0)
    total_area_m2 = float(sum(welded_areas))
    largest_area_m2 = max(welded_areas, default=0.0)
    significant_areas = [
        area
        for area in welded_areas
        if area >= minimum_significant_component_area_m2
    ]
    tiny_area_m2 = total_area_m2 - float(sum(significant_areas))
    bounds = {
        "minimum_m": vertices.min(axis=0).tolist() if vertex_count else None,
        "maximum_m": vertices.max(axis=0).tolist() if vertex_count else None,
    }
    return {
        "mesh_path": str(mesh_path.resolve()),
        "vertices": vertex_count,
        "faces": face_count,
        "triangles": triangle_count,
        "vertices_used_by_faces": welded_used_count,
        "isolated_vertices": welded_vertex_count - welded_used_count,
        "weld_tolerance_m": weld_tolerance_m,
        "welded_vertices": welded_vertex_count,
        "connected_components": connected_components,
        "largest_component_vertices": largest,
        "largest_component_ratio": largest / max(1, welded_used_count),
        "surface_area_m2": total_area_m2,
        "largest_component_area_m2": largest_area_m2,
        "largest_component_area_ratio": largest_area_m2
        / max(total_area_m2, np.finfo(np.float64).eps),
        "minimum_significant_component_area_m2": (
            minimum_significant_component_area_m2
        ),
        "significant_connected_components": len(significant_areas),
        "significant_component_area_m2": float(sum(significant_areas)),
        "tiny_component_area_m2": tiny_area_m2,
        "tiny_component_area_ratio": tiny_area_m2
        / max(total_area_m2, np.finfo(np.float64).eps),
        "raw_vertices_used_by_faces": raw_used_count,
        "raw_isolated_vertices": vertex_count - raw_used_count,
        "raw_connected_components": len(raw_counts),
        "raw_largest_component_vertices": max(raw_counts, default=0),
        "raw_largest_component_ratio": max(raw_counts, default=0)
        / max(1, raw_used_count),
        "invalid_faces": invalid_faces,
        "bounds": bounds,
    }
