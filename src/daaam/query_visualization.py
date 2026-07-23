"""Export image evidence and a mesh top view for semantic-query results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from daaam.query_evidence import sha256_file
from daaam.semantic_query import ObjectRecord, SemanticQueryEngine


QUERY_VISUAL_REPORT_SCHEMA = "daaam.query_visual_result.v1"
_MAX_RENDER_POINTS = 200_000

# Noto CJK is present in the supported mapping environment; DejaVu remains a
# portable fallback for Latin labels and installations without Noto.
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class QueryVisualizationError(RuntimeError):
    """Raised when a query-result visualization cannot be generated."""


@dataclass(frozen=True)
class QueryVisualArtifacts:
    """Paths written for one semantic query."""

    output_directory: Path
    topdown_image: Path
    report: Path
    evidence_images: tuple[Path, ...]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_node_name(node_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", node_id).strip("_")
    return normalized or "object"


def _unique_query_directory(root: Path, query: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    stem = f"{timestamp}_{_sha256_text(query)[:10]}"
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _mesh_projection(dsg_path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        payload = json.loads(dsg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueryVisualizationError(f"无法读取 DSG mesh：{dsg_path}") from exc

    mesh = payload.get("mesh") if isinstance(payload, dict) else None
    if not isinstance(mesh, dict):
        raise QueryVisualizationError("DSG 不包含可绘制的 mesh")
    try:
        points = np.asarray(mesh.get("points"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise QueryVisualizationError("DSG mesh points 格式无效") from exc
    if points.ndim != 2 or points.shape[1] < 3 or not len(points):
        raise QueryVisualizationError("DSG mesh 没有有效三维顶点")

    finite = np.isfinite(points[:, :3]).all(axis=1)
    points = points[finite, :3]
    if not len(points):
        raise QueryVisualizationError("DSG mesh 顶点均为非有限值")

    raw_colors = mesh.get("colors")
    colors = np.full((len(finite), 3), 0.62, dtype=float)
    if isinstance(raw_colors, list) and len(raw_colors) == len(finite):
        parsed: list[list[float]] = []
        for color in raw_colors:
            if isinstance(color, dict):
                parsed.append(
                    [float(color.get(channel, 158.0)) for channel in ("r", "g", "b")]
                )
            elif isinstance(color, (list, tuple)) and len(color) >= 3:
                parsed.append([float(value) for value in color[:3]])
            else:
                parsed = []
                break
        if parsed:
            colors = np.asarray(parsed, dtype=float) / 255.0
    colors = np.clip(colors[finite], 0.0, 1.0)

    if len(points) > _MAX_RENDER_POINTS:
        indices = np.linspace(0, len(points) - 1, _MAX_RENDER_POINTS, dtype=int)
        points = points[indices]
        colors = colors[indices]
    return points, colors


def _record_position(record: ObjectRecord) -> np.ndarray | None:
    position = np.asarray(record.position, dtype=float).reshape(-1)
    if position.size < 3 or not np.isfinite(position[:3]).all():
        return None
    return position[:3]


def render_mesh_topdown_preview(dsg_path: Path | str) -> bytes:
    """Render an unannotated RGB mesh XY projection as PNG bytes."""

    resolved = Path(dsg_path).expanduser().resolve()
    points, colors = _mesh_projection(resolved)
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    axis.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        s=0.4,
        linewidths=0,
        alpha=0.84,
        rasterized=True,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("World X (m)")
    axis.set_ylabel("World Y (m)")
    axis.set_title("Semantic mesh · top view")
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
    buffer = BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=180)
    finally:
        plt.close(figure)
    return buffer.getvalue()


def write_query_visuals(
    *,
    engine: SemanticQueryEngine,
    query: str,
    matches: Iterable[tuple[float, ObjectRecord]],
    output_root: Path | str,
    found: bool,
    rejection_reason: str | None,
    top_score: float,
    top1_margin: float | None,
) -> QueryVisualArtifacts:
    """Write exact evidence images, a marked mesh top view, and a JSON report."""

    normalized_query = query.strip()
    if not normalized_query:
        raise QueryVisualizationError("查询文本不能为空")
    ranked_matches = list(matches)
    output_directory = _unique_query_directory(
        Path(output_root).expanduser().resolve(), normalized_query
    )

    evidence_images: list[Path] = []
    report_matches: list[dict[str, object]] = []
    for rank, (score, record) in enumerate(ranked_matches, start=1):
        evidence = engine.evidence_for_node(record.node_id)
        copied_evidence: Path | None = None
        if evidence is not None:
            suffix = evidence.image_path.suffix.lower() or ".png"
            copied_evidence = output_directory / (
                f"rank_{rank:02d}_{_safe_node_name(record.node_id)}_evidence{suffix}"
            )
            shutil.copyfile(evidence.image_path, copied_evidence)
            copied_digest = sha256_file(copied_evidence)
            if copied_digest != evidence.image_sha256:
                raise QueryVisualizationError(
                    f"复制后的证据图片校验失败：{copied_evidence}"
                )
            evidence_images.append(copied_evidence)

        position = _record_position(record)
        report_matches.append(
            {
                "rank": rank,
                "score": float(score),
                "node_id": record.node_id,
                "entity_id": record.entity_id,
                "semantic_label": int(record.semantic_label),
                "description": record.description,
                "position_m": None if position is None else position.tolist(),
                "geometry_status": record.geometry_status,
                "evidence_available": evidence is not None,
                "evidence_image": (
                    None if copied_evidence is None else copied_evidence.name
                ),
                "evidence_image_sha256": (
                    None if copied_evidence is None else sha256_file(copied_evidence)
                ),
                "evidence_frame_index": (
                    None if evidence is None else int(evidence.frame_index)
                ),
                "camera_position_m": (
                    None
                    if evidence is None or evidence.camera_position_m is None
                    else list(evidence.camera_position_m)
                ),
            }
        )

    points, colors = _mesh_projection(engine.dsg_path)
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    axis.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        s=0.35,
        linewidths=0,
        alpha=0.8,
        rasterized=True,
    )

    marker_colors = ["#ef233c", "#ff9f1c", "#00b4d8", "#8338ec", "#2a9d8f"]
    plotted_positions: list[np.ndarray] = []
    for rank, (score, record) in enumerate(ranked_matches, start=1):
        position = _record_position(record)
        if position is None:
            continue
        plotted_positions.append(position)
        color = marker_colors[(rank - 1) % len(marker_colors)]
        if rank == 1:
            axis.scatter(
                position[0],
                position[1],
                marker="*",
                s=420,
                c=color,
                edgecolors="white",
                linewidths=1.6,
                zorder=6,
                label=f"Top 1: {record.node_id}",
            )
        else:
            axis.scatter(
                position[0],
                position[1],
                marker="o",
                s=150,
                c=color,
                edgecolors="white",
                linewidths=1.3,
                zorder=5,
                label=f"Top {rank}: {record.node_id}",
            )
        axis.annotate(
            f"#{rank} {record.node_id}\n{score:.3f}",
            xy=(position[0], position[1]),
            xytext=(9, 9),
            textcoords="offset points",
            fontsize=9,
            color="black",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.9},
            zorder=7,
        )

        evidence = engine.evidence_for_node(record.node_id)
        if evidence is not None and evidence.camera_position_m is not None:
            camera = np.asarray(evidence.camera_position_m, dtype=float)
            if camera.size >= 2 and np.isfinite(camera[:2]).all():
                axis.scatter(
                    camera[0],
                    camera[1],
                    marker="^",
                    s=65,
                    c=color,
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=4,
                )
                axis.plot(
                    [camera[0], position[0]],
                    [camera[1], position[1]],
                    linestyle="--",
                    linewidth=1.0,
                    color=color,
                    alpha=0.8,
                    zorder=3,
                )

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("World X (m)")
    axis.set_ylabel("World Y (m)")
    axis.set_title(f"Semantic query location (top view)\n{normalized_query}")
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
    if plotted_positions:
        axis.legend(loc="best", fontsize=9, framealpha=0.92)
    else:
        axis.text(
            0.5,
            0.98,
            "No accepted query position",
            transform=axis.transAxes,
            ha="center",
            va="top",
            color="#b00020",
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.9},
        )

    topdown_image = output_directory / "mesh_topdown_query.png"
    try:
        figure.savefig(topdown_image, dpi=180)
    finally:
        plt.close(figure)

    report = output_directory / "query_result.json"
    report_payload = {
        "schema": QUERY_VISUAL_REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query": normalized_query,
        "found": bool(found),
        "rejection_reason": rejection_reason,
        "top_score": float(top_score),
        "top1_margin": None if top1_margin is None else float(top1_margin),
        "dsg_file": str(engine.dsg_path),
        "dsg_sha256": sha256_file(engine.dsg_path),
        "topdown_image": topdown_image.name,
        "topdown_image_sha256": sha256_file(topdown_image),
        "matches": report_matches,
    }
    report.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return QueryVisualArtifacts(
        output_directory=output_directory,
        topdown_image=topdown_image,
        report=report,
        evidence_images=tuple(evidence_images),
    )
