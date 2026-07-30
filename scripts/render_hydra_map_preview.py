#!/usr/bin/env python3
"""Render a headless audit preview from a completed Hydra run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Realtime run containing hydra_realtime/backend.",
    )
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    return parser.parse_args()


def load_ascii_ply_vertices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertex_count: int | None = None
    with path.open("r", encoding="utf-8") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            fields = line.split()
            if fields[:2] == ["format", "ascii"]:
                continue
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
            if fields[0] == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY vertex count is missing: {path}")
        rows = []
        for _ in range(vertex_count):
            fields = stream.readline().split()
            if len(fields) < 6:
                raise ValueError(f"Malformed PLY vertex row in {path}")
            rows.append([float(value) for value in fields[:6]])
    vertices = np.asarray(rows, dtype=np.float32)
    return vertices[:, :3], np.clip(vertices[:, 3:6] / 255.0, 0.0, 1.0)


def load_trajectory(path: Path) -> np.ndarray:
    points: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            points.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return np.asarray(points, dtype=np.float32)


def load_objects(path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    positions: list[list[float]] = []
    colors: list[list[float]] = []
    labels: list[int] = []
    for node in graph.get("nodes", []):
        if node.get("layer") != 2 or node.get("partition", 0) != 0:
            continue
        attributes = node.get("attributes", {})
        position = attributes.get("position")
        if not isinstance(position, list) or len(position) != 3:
            continue
        color = attributes.get("color", [255, 60, 60, 255])
        if not isinstance(color, list) or len(color) < 3:
            color = [255, 60, 60, 255]
        positions.append([float(value) for value in position])
        colors.append([float(value) / 255.0 for value in color[:3]])
        labels.append(int(attributes.get("semantic_label", -1)))
    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(colors, dtype=np.float32),
        labels,
    )


def draw_projection(
    axis: plt.Axes,
    vertices: np.ndarray,
    vertex_colors: np.ndarray,
    trajectory: np.ndarray,
    objects: np.ndarray,
    object_colors: np.ndarray,
    dimensions: tuple[int, int],
    title: str,
) -> None:
    horizontal, vertical = dimensions
    stride = max(1, len(vertices) // 120_000)
    axis.scatter(
        vertices[::stride, horizontal],
        vertices[::stride, vertical],
        c=vertex_colors[::stride],
        s=0.35,
        alpha=0.60,
        linewidths=0,
        rasterized=True,
    )
    if len(trajectory):
        axis.plot(
            trajectory[:, horizontal],
            trajectory[:, vertical],
            color="#00a6ff",
            linewidth=1.4,
            alpha=0.95,
            label="camera trajectory",
        )
        axis.scatter(
            trajectory[0, horizontal],
            trajectory[0, vertical],
            color="#00c853",
            s=28,
            marker="o",
            zorder=4,
            label="start",
        )
        axis.scatter(
            trajectory[-1, horizontal],
            trajectory[-1, vertical],
            color="#ff1744",
            s=32,
            marker="x",
            zorder=4,
            label="end",
        )
    if len(objects):
        axis.scatter(
            objects[:, horizontal],
            objects[:, vertical],
            c=object_colors,
            s=18,
            marker="o",
            edgecolors="black",
            linewidths=0.25,
            alpha=0.95,
            zorder=3,
            label="Hydra objects",
        )
    labels = ("X", "Y", "Z")
    axis.set_title(title)
    axis.set_xlabel(f"{labels[horizontal]} (m)")
    axis.set_ylabel(f"{labels[vertical]} (m)")
    axis.grid(alpha=0.18)
    axis.set_aspect("equal", adjustable="box")


def main() -> None:
    args = parse_args()
    backend = args.run_dir.resolve() / "hydra_realtime" / "backend"
    output = (
        args.output.resolve()
        if args.output
        else args.run_dir.resolve() / "hydra_map_preview.png"
    )
    vertices, vertex_colors = load_ascii_ply_vertices(backend / "mesh.ply")
    trajectory = load_trajectory(backend / "trajectory.csv")
    objects, object_colors, object_labels = load_objects(backend / "dsg.json")

    figure, axes = plt.subplots(1, 3, figsize=(20, 6.5), constrained_layout=True)
    draw_projection(
        axes[0],
        vertices,
        vertex_colors,
        trajectory,
        objects,
        object_colors,
        (0, 1),
        "Top: X / Y",
    )
    draw_projection(
        axes[1],
        vertices,
        vertex_colors,
        trajectory,
        objects,
        object_colors,
        (0, 2),
        "Side: X / Z",
    )
    draw_projection(
        axes[2],
        vertices,
        vertex_colors,
        trajectory,
        objects,
        object_colors,
        (1, 2),
        "Side: Y / Z",
    )
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle(
        "Hydra semantic map audit"
        f" — {len(vertices):,} mesh vertices, {len(objects):,} objects,"
        f" {len(set(object_labels)):,} object labels, {len(trajectory):,} poses",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    report = {
        "schema": "daaam.hydra_map_preview.v1",
        "run_dir": str(args.run_dir.resolve()),
        "mesh_vertices": int(len(vertices)),
        "object_count": int(len(objects)),
        "unique_object_labels": int(len(set(object_labels))),
        "trajectory_pose_count": int(len(trajectory)),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
