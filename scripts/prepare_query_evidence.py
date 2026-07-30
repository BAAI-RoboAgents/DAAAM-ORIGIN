#!/usr/bin/env python3
"""Build top-1 FastSAM evidence images for a query-ready DSG."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Optional
import uuid

import click
import cv2
import numpy as np

from daaam.query_evidence import (
    EVIDENCE_SCHEMA,
    evidence_manifest_path,
    infer_segmentation_frame_indices,
    sha256_file,
)
from daaam.realtime.semantic_labels import (
    load_semantic_label,
    semantic_label_path,
    validate_semantic_label_binding,
)
from daaam.realtime.masked_geometry import backproject_masked_depth
from daaam.semantic_query import load_object_records, load_query_manifest, load_sidecar_records


@dataclass(frozen=True)
class EvidenceTarget:
    node_id: str
    semantic_label: int
    entity_id: str
    description: str
    first_observed_ns: Optional[int]
    last_observed_ns: Optional[int]


@dataclass(frozen=True)
class EvidenceCandidate:
    frame_index: int
    sensor_time_ns: int
    bbox_xyxy: tuple[int, int, int, int]
    mask_pixels: int
    border_touch: bool
    selection_score: float
    valid_depth_pixels: int = 0
    valid_depth_ratio: float = 0.0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Failed to read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise click.ClickException(f"Expected a JSON object: {path}")
    return value


def _targets_from_dsg(dsg_path: Path) -> dict[int, EvidenceTarget]:
    manifest = load_query_manifest(dsg_path)
    records = load_object_records(dsg_path) + load_sidecar_records(dsg_path, manifest)
    targets: dict[int, EvidenceTarget] = {}
    for record in records:
        semantic_label = int(record.semantic_label)
        if semantic_label in targets:
            raise click.ClickException(
                f"Queryable semantic label {semantic_label} is not unique"
            )
        targets[semantic_label] = EvidenceTarget(
            node_id=record.node_id,
            semantic_label=semantic_label,
            entity_id=record.entity_id,
            description=record.description,
            first_observed_ns=record.first_observed_ns,
            last_observed_ns=record.last_observed_ns,
        )
    if not targets:
        raise click.ClickException("The DSG has no queryable described objects")
    return targets


def largest_component(
    labels: np.ndarray, semantic_label: int
) -> tuple[np.ndarray, tuple[int, int, int, int], int, bool]:
    """Return the largest connected mask for one semantic label."""

    binary = np.asarray(labels == semantic_label, dtype=np.uint8)
    component_count, components, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if component_count <= 1:
        raise ValueError(f"semantic label {semantic_label} has no pixels")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[component, cv2.CC_STAT_LEFT])
    y = int(stats[component, cv2.CC_STAT_TOP])
    width = int(stats[component, cv2.CC_STAT_WIDTH])
    height = int(stats[component, cv2.CC_STAT_HEIGHT])
    area = int(stats[component, cv2.CC_STAT_AREA])
    mask = components == component
    image_height, image_width = binary.shape
    border_touch = bool(
        x <= 1
        or y <= 1
        or x + width >= image_width - 1
        or y + height >= image_height - 1
    )
    return mask, (x, y, x + width, y + height), area, border_touch


def _mask_sha256(mask: np.ndarray) -> str:
    shape = np.asarray(mask.shape, dtype="<u4").tobytes()
    packed = np.packbits(np.asarray(mask, dtype=np.uint8), axis=None).tobytes()
    return hashlib.sha256(shape + packed).hexdigest()


def _evidence_id(node_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", node_id).strip("_")
    if not normalized:
        raise ValueError(f"Cannot derive evidence ID from {node_id!r}")
    return normalized


def _candidate_score(area: int, border_touch: bool) -> float:
    # Prefer a large, readable mask while avoiding views cut off by an image edge.
    return float(area) * (0.40 if border_touch else 1.0)


def accumulated_segmentation_frame_indices(
    sensor_times_ns: list[int],
    segmentation_rate_hz: float,
    *,
    frames_resumed_from: int,
) -> tuple[list[int], list[int]]:
    """Replay persisted FastSAM schedules across one interrupted/resumed run.

    The timestamp scheduler restarts when a run resumes.  Labels before the
    resume boundary remain valid, checksum-bound artifacts from the preceding
    segment, while the final runtime report only counts calls made after that
    boundary.
    """

    if frames_resumed_from < 0 or frames_resumed_from > len(sensor_times_ns):
        raise ValueError("frames_resumed_from is outside the tick index")
    if frames_resumed_from == 0:
        indices = infer_segmentation_frame_indices(
            sensor_times_ns, segmentation_rate_hz
        )
        return indices, indices
    prior = infer_segmentation_frame_indices(
        sensor_times_ns[:frames_resumed_from], segmentation_rate_hz
    )
    current = [
        frames_resumed_from + index
        for index in infer_segmentation_frame_indices(
            sensor_times_ns[frames_resumed_from:], segmentation_rate_hz
        )
    ]
    return prior + current, current


def select_candidates(
    *,
    targets: dict[int, EvidenceTarget],
    frames: list[dict[str, Any]],
    segmentation_indices: list[int],
    label_directory: Path,
    run_configuration_sha256: str,
    interval_tolerance_ns: int = 0,
    depth_directory: Optional[Path] = None,
) -> dict[int, EvidenceCandidate]:
    """Select the clearest exact-FastSAM frame for every queryable object."""

    selected: dict[int, EvidenceCandidate] = {}
    if interval_tolerance_ns < 0:
        raise ValueError("interval_tolerance_ns cannot be negative")
    target_labels = set(targets)
    for frame_index in segmentation_indices:
        frame = frames[frame_index]
        sensor_time_ns = int(frame["sensor_time_ns"])
        validate_semantic_label_binding(
            label_directory,
            frame_index,
            sensor_time_ns=sensor_time_ns,
            run_configuration_sha256=run_configuration_sha256,
        )
        labels = load_semantic_label(label_directory, frame_index)
        depth_m = None
        if depth_directory is not None:
            depth_path = depth_directory / f"{int(frame['idx']):08d}.png"
            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_raw is None or depth_raw.ndim != 2:
                raise click.ClickException(f"Failed to load metric depth: {depth_path}")
            if depth_raw.shape != labels.shape:
                raise click.ClickException(
                    f"Depth/semantic shape mismatch at frame {frame_index}"
                )
            depth_m = depth_raw.astype(np.float64) / 1000.0
        present = target_labels.intersection(int(value) for value in np.unique(labels))
        for semantic_label in present:
            target = targets[semantic_label]
            if (
                target.first_observed_ns is not None
                and sensor_time_ns
                < target.first_observed_ns - interval_tolerance_ns
            ):
                continue
            if (
                target.last_observed_ns is not None
                and sensor_time_ns
                > target.last_observed_ns + interval_tolerance_ns
            ):
                continue
            mask, bbox, area, border_touch = largest_component(labels, semantic_label)
            valid_depth_pixels = (
                0
                if depth_m is None
                else int(
                    np.count_nonzero(mask & np.isfinite(depth_m) & (depth_m > 0.0))
                )
            )
            candidate = EvidenceCandidate(
                frame_index=frame_index,
                sensor_time_ns=sensor_time_ns,
                bbox_xyxy=bbox,
                mask_pixels=area,
                border_touch=border_touch,
                selection_score=_candidate_score(area, border_touch),
                valid_depth_pixels=valid_depth_pixels,
                valid_depth_ratio=float(valid_depth_pixels) / float(area),
            )
            old = selected.get(semantic_label)
            if old is None or (
                candidate.valid_depth_pixels > 0,
                candidate.valid_depth_ratio,
                candidate.selection_score,
                candidate.mask_pixels,
                -candidate.frame_index,
            ) > (
                old.valid_depth_pixels > 0,
                old.valid_depth_ratio,
                old.selection_score,
                old.mask_pixels,
                -old.frame_index,
            ):
                selected[semantic_label] = candidate
    return selected


def render_evidence_image(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    bbox_xyxy: tuple[int, int, int, int],
    node_id: str,
    semantic_label: int,
    observed_s: float,
) -> np.ndarray:
    """Overlay an exact FastSAM mask, contour, and bounding box on RGB evidence."""

    annotated = np.asarray(image).copy()
    if annotated.ndim != 3 or annotated.shape[2] != 3:
        raise ValueError("evidence source image must be BGR uint8")
    if mask.shape != annotated.shape[:2] or not np.any(mask):
        raise ValueError("evidence mask does not match the source image")
    color = np.asarray([0, 190, 255], dtype=np.float32)
    pixels = annotated[mask].astype(np.float32)
    annotated[mask] = np.clip(0.55 * pixels + 0.45 * color, 0, 255).astype(
        np.uint8
    )
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(annotated, contours, -1, (0, 120, 255), 3, cv2.LINE_AA)
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 3)
    caption = (
        f"FastSAM | {node_id} | semantic_id={semantic_label} | t={observed_s:.3f}s"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        caption, font, scale, thickness
    )
    text_x = max(0, min(x1, annotated.shape[1] - text_width - 8))
    text_y = max(text_height + baseline + 8, y1)
    cv2.rectangle(
        annotated,
        (text_x, text_y - text_height - baseline - 8),
        (text_x + text_width + 8, text_y + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        annotated,
        caption,
        (text_x + 4, text_y - baseline),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def render_masked_cutout(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    bbox_xyxy: tuple[int, int, int, int],
    max_size_px: int = 256,
) -> np.ndarray:
    """Crop one exact FastSAM foreground into a compact BGRA texture."""

    source = np.asarray(image)
    foreground = np.asarray(mask, dtype=bool)
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError("cutout source image must be BGR uint8")
    if foreground.shape != source.shape[:2] or not np.any(foreground):
        raise ValueError("cutout mask does not match the source image")
    if max_size_px < 32:
        raise ValueError("cutout max_size_px must be at least 32")

    x1, y1, x2, y2 = bbox_xyxy
    if not (0 <= x1 < x2 <= source.shape[1] and 0 <= y1 < y2 <= source.shape[0]):
        raise ValueError("cutout bounding box is outside the source image")
    padding = max(2, int(round(0.05 * max(x2 - x1, y2 - y1))))
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(source.shape[1], x2 + padding)
    y2 = min(source.shape[0], y2 + padding)
    crop_mask = foreground[y1:y2, x1:x2]
    crop_bgr = source[y1:y2, x1:x2].copy()
    crop_bgr[~crop_mask] = 0
    cutout = np.dstack(
        [crop_bgr, np.asarray(crop_mask, dtype=np.uint8) * np.uint8(255)]
    )

    longest = max(cutout.shape[:2])
    if longest > max_size_px:
        scale = float(max_size_px) / float(longest)
        target_size = (
            max(1, int(round(cutout.shape[1] * scale))),
            max(1, int(round(cutout.shape[0] * scale))),
        )
        cutout = cv2.resize(cutout, target_size, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cutout)


def _atomic_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.png"
    try:
        if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(f"Failed to encode evidence image: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        json.loads(temporary.read_text(encoding="utf-8"))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.npz"
    try:
        np.savez_compressed(temporary, **arrays)
        with np.load(temporary, allow_pickle=False) as payload:
            if set(payload.files) != set(arrays):
                raise OSError(f"Failed to verify point-cloud archive: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@click.command()
@click.option(
    "--dsg-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Query-ready DSG containing descriptions, entity IDs, and embeddings.",
)
@click.option(
    "--semantic-run",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Completed DAM run containing RGB frames and exact semantic labels.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Annotated image directory; defaults to <DSG parent>/query_evidence.",
)
@click.option(
    "--allow-missing",
    is_flag=True,
    help="Write partial evidence instead of failing when an object has no FastSAM frame.",
)
@click.option(
    "--cutout-max-size-px",
    type=click.IntRange(min=32, max=1024),
    default=256,
    show_default=True,
    help="Maximum width or height of each masked Rerun texture.",
)
def main(
    dsg_file: Path,
    semantic_run: Path,
    output_dir: Optional[Path],
    allow_missing: bool,
    cutout_max_size_px: int,
) -> None:
    """Select and render one checksum-bound FastSAM evidence frame per object."""

    dsg_path = dsg_file.expanduser().resolve()
    run_dir = semantic_run.expanduser().resolve()
    report_path = run_dir / "realtime_run_report.json"
    run_manifest_path = run_dir / "run_manifest.json"
    report = _read_json(report_path)
    run_manifest = _read_json(run_manifest_path)
    if report.get("status") != "complete" or report.get("semantic_mode") != "dam":
        raise click.ClickException("--semantic-run must be a completed DAM run")
    semantic_stats = dict(report.get("semantic_stats") or {})
    if int(semantic_stats.get("segmentation_failures", -1)) != 0:
        raise click.ClickException("FastSAM segmentation failures prevent evidence use")

    semantic_config = dict(
        dict(run_manifest.get("configuration") or {}).get("semantic") or {}
    )
    segmentation_rate_hz = float(semantic_config["segmentation_rate_hz"])
    run_configuration_sha256 = str(
        semantic_config["label_run_configuration_sha256"]
    )
    postpass = dict(report.get("semantic_postpass") or {})
    label_directory = Path(postpass["semantic_label_directory"]).resolve()
    dataset_path = Path(report["dataset"]).resolve()
    tick_index_path = dataset_path / "tick_index.json"
    tick_index = _read_json(tick_index_path)
    frames = list(tick_index.get("frames") or [])
    intrinsics = np.asarray(
        [
            [float(tick_index["fx"]), 0.0, float(tick_index["cx"])],
            [0.0, float(tick_index["fy"]), float(tick_index["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if len(frames) != int(report.get("frames_requested", -1)):
        raise click.ClickException("Tick index does not cover the completed semantic run")
    for expected_index, frame in enumerate(frames):
        if int(frame.get("idx", -1)) != expected_index:
            raise click.ClickException("Tick-index frame indices are not contiguous")
    pose_path = dataset_path / "pose" / "poses.txt"
    try:
        pose_rows = np.loadtxt(pose_path, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Failed to load camera poses: {pose_path}") from exc
    if pose_rows.ndim == 1:
        pose_rows = pose_rows.reshape(1, -1)
    if pose_rows.shape[1] != 16:
        raise click.ClickException("Camera pose rows must contain 16 matrix values")
    poses = pose_rows.reshape(-1, 4, 4)
    if not np.isfinite(poses).all():
        raise click.ClickException("Camera poses contain non-finite values")
    if any(
        int(frame.get("pose_row", -1)) < 0
        or int(frame.get("pose_row", -1)) >= len(poses)
        for frame in frames
    ):
        raise click.ClickException("Tick index references an invalid camera pose row")
    sensor_times_ns = [int(frame["sensor_time_ns"]) for frame in frames]
    frames_resumed_from = int(report.get("frames_resumed_from") or 0)
    try:
        segmentation_indices, reported_segment_indices = (
            accumulated_segmentation_frame_indices(
                sensor_times_ns,
                segmentation_rate_hz,
                frames_resumed_from=frames_resumed_from,
            )
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    expected_calls = int(semantic_stats.get("segmentation_calls", -1))
    if len(reported_segment_indices) != expected_calls:
        raise click.ClickException(
            "Inferred FastSAM schedule does not match runtime evidence for "
            f"the reported segment: {len(reported_segment_indices)} != "
            f"{expected_calls}"
        )
    if frames_resumed_from > 0 and not all(
        semantic_label_path(label_directory, index).is_file()
        for index in segmentation_indices[: -len(reported_segment_indices)]
    ):
        raise click.ClickException(
            "Persisted FastSAM labels before the resume boundary are incomplete"
        )
    maximum_frame_interval_ns = max(
        (
            current - previous
            for previous, current in zip(sensor_times_ns, sensor_times_ns[1:])
        ),
        default=0,
    )
    evidence_interval_tolerance_ns = int(
        round(1.0e9 / segmentation_rate_hz)
    ) + maximum_frame_interval_ns

    targets = _targets_from_dsg(dsg_path)
    selected = select_candidates(
        targets=targets,
        frames=frames,
        segmentation_indices=segmentation_indices,
        label_directory=label_directory,
        run_configuration_sha256=run_configuration_sha256,
        interval_tolerance_ns=evidence_interval_tolerance_ns,
        depth_directory=dataset_path / "depth",
    )
    missing = sorted(target.node_id for label, target in targets.items() if label not in selected)
    if missing and not allow_missing:
        raise click.ClickException(
            f"No exact FastSAM evidence for {len(missing)} queryable objects: "
            + ", ".join(missing)
        )

    evidence_root = (
        dsg_path.parent / "query_evidence"
        if output_dir is None
        else output_dir.expanduser().resolve()
    )
    cutout_root = evidence_root / "cutouts"
    point_cloud_root = evidence_root / "point_clouds"
    manifest_path = evidence_manifest_path(dsg_path)
    origin_ns = int(tick_index["time_origin_ns"])
    object_records: list[dict[str, Any]] = []
    for semantic_label, candidate in sorted(
        selected.items(), key=lambda item: item[1].frame_index
    ):
        target = targets[semantic_label]
        frame = frames[candidate.frame_index]
        source_image_path = Path(frame["cam0"]).resolve()
        image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise click.ClickException(f"Failed to load source RGB: {source_image_path}")
        labels = load_semantic_label(label_directory, candidate.frame_index)
        mask, bbox, area, border_touch = largest_component(labels, semantic_label)
        if (
            bbox != candidate.bbox_xyxy
            or area != candidate.mask_pixels
            or border_touch != candidate.border_touch
        ):
            raise click.ClickException(
                f"Evidence selection changed while rendering {target.node_id}"
            )
        observed_s = (candidate.sensor_time_ns - origin_ns) / 1.0e9
        annotated = render_evidence_image(
            image,
            mask,
            bbox_xyxy=bbox,
            node_id=target.node_id,
            semantic_label=semantic_label,
            observed_s=observed_s,
        )
        evidence_id = _evidence_id(target.node_id)
        image_path = evidence_root / f"{evidence_id}.png"
        _atomic_png(image_path, annotated)
        cutout = render_masked_cutout(
            image,
            mask,
            bbox_xyxy=bbox,
            max_size_px=cutout_max_size_px,
        )
        cutout_path = cutout_root / f"{evidence_id}.png"
        _atomic_png(cutout_path, cutout)
        depth_path = dataset_path / "depth" / f"{int(frame['idx']):08d}.png"
        geometry_fields: dict[str, Any] = {}
        relative_point_cloud = None
        if candidate.valid_depth_pixels > 0:
            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_raw is None or depth_raw.ndim != 2:
                raise click.ClickException(f"Failed to load metric depth: {depth_path}")
            depth_m = depth_raw.astype(np.float64) / 1000.0
            try:
                geometry = backproject_masked_depth(
                    mask,
                    depth_m,
                    intrinsics,
                    poses[int(frame["pose_row"])],
                    maximum_points=5_000,
                )
            except ValueError as exc:
                raise click.ClickException(
                    f"Failed to backproject FastSAM geometry for {target.node_id}: {exc}"
                ) from exc
            pixel_y = geometry.pixel_yx[:, 0]
            pixel_x = geometry.pixel_yx[:, 1]
            colors_rgb = np.ascontiguousarray(image[pixel_y, pixel_x, ::-1])
            point_cloud_path = point_cloud_root / f"{evidence_id}.npz"
            _atomic_npz(
                point_cloud_path,
                points_world_m=geometry.points_world_m,
                colors_rgb=colors_rgb,
            )
            geometry_fields = {
                "point_cloud_sha256": sha256_file(point_cloud_path),
                "point_count": int(len(geometry.points_world_m)),
                "valid_depth_pixels": geometry.valid_pixel_count,
                "valid_depth_ratio": (
                    float(geometry.valid_pixel_count) / float(area)
                ),
                "geometry_position_m": [
                    float(value) for value in geometry.position_m
                ],
                "geometry_dimensions_m": [
                    float(value) for value in geometry.dimensions_m
                ],
                "geometry_source": "fastsam_masked_rgbd_joint_backprojection",
                "source_depth_sha256": sha256_file(depth_path),
            }
        try:
            relative_image = image_path.resolve().relative_to(manifest_path.parent)
            relative_cutout = cutout_path.resolve().relative_to(manifest_path.parent)
            if candidate.valid_depth_pixels > 0:
                relative_point_cloud = point_cloud_path.resolve().relative_to(
                    manifest_path.parent
                )
        except ValueError as exc:
            raise click.ClickException(
                "--output-dir must be inside the DSG directory"
            ) from exc
        object_record = {
                "evidence_id": evidence_id,
                "node_id": target.node_id,
                "entity_id": target.entity_id,
                "semantic_label": semantic_label,
                "frame_index": candidate.frame_index,
                "sensor_time_ns": candidate.sensor_time_ns,
                "observed_s": observed_s,
                "bbox_xyxy": list(bbox),
                "mask_pixels": area,
                "mask_source": "fastsam_segmentation",
                "mask_sha256": _mask_sha256(mask),
                "border_touch": border_touch,
                "selection_score": candidate.selection_score,
                "image": str(relative_image),
                "image_sha256": sha256_file(image_path),
                "cutout": str(relative_cutout),
                "cutout_sha256": sha256_file(cutout_path),
                "cutout_size_px": [int(cutout.shape[1]), int(cutout.shape[0])],
                "source_image_sha256": sha256_file(source_image_path),
                "camera_position_m": [
                    float(value)
                    for value in poses[int(frame["pose_row"]), :3, 3]
                ],
                "semantic_label_sha256": sha256_file(
                    semantic_label_path(label_directory, candidate.frame_index)
                ),
            }
        if relative_point_cloud is not None:
            object_record["point_cloud"] = str(relative_point_cloud)
            object_record.update(geometry_fields)
        object_records.append(object_record)

    manifest = {
        "schema": EVIDENCE_SCHEMA,
        "dsg_file": dsg_path.name,
        "dsg_sha256": sha256_file(dsg_path),
        "object_count": len(object_records),
        "queryable_object_count": len(targets),
        "missing_node_ids": missing,
        "selection": {
            "policy": "depth_available_then_largest_connected_mask_with_border_penalty",
            "mask_source": "fastsam_segmentation",
            "segmentation_rate_hz": segmentation_rate_hz,
            "segmentation_frames": len(segmentation_indices),
            "runtime_report_segmentation_calls": expected_calls,
            "frames_resumed_from": frames_resumed_from,
            "segmentation_failures": 0,
            "observation_interval_tolerance_ns": evidence_interval_tolerance_ns,
            "rgbd_geometry_objects": sum(
                "point_cloud" in record for record in object_records
            ),
            "image_only_objects": sum(
                "point_cloud" not in record for record in object_records
            ),
        },
        "source": {
            "semantic_run": str(run_dir),
            "realtime_run_report": str(report_path),
            "realtime_run_report_sha256": sha256_file(report_path),
            "run_manifest": str(run_manifest_path),
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "tick_index": str(tick_index_path),
            "tick_index_sha256": sha256_file(tick_index_path),
            "semantic_label_directory": str(label_directory),
            "label_run_configuration_sha256": run_configuration_sha256,
            "camera_pose_file": str(pose_path),
            "camera_pose_file_sha256": sha256_file(pose_path),
        },
        "objects": object_records,
    }
    _atomic_json(manifest_path, manifest)
    click.echo(
        f"Saved {len(object_records)}/{len(targets)} FastSAM evidence images to "
        f"{evidence_root}"
    )
    click.echo(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
