"""Checksum-bound FastSAM image evidence for semantic query results."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from hmac import compare_digest
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Optional


EVIDENCE_SCHEMA = "daaam.query_evidence.v1"
_EVIDENCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class QueryEvidenceError(RuntimeError):
    """Raised when query evidence is missing, unsafe, or inconsistent."""


@dataclass(frozen=True)
class QueryEvidence:
    """One immutable annotated image tied to a DSG object and FastSAM mask."""

    evidence_id: str
    node_id: str
    semantic_label: int
    frame_index: int
    sensor_time_ns: int
    observed_s: Optional[float]
    bbox_xyxy: tuple[int, int, int, int]
    mask_pixels: int
    mask_source: str
    image_path: Path
    image_sha256: str
    source_image_sha256: str
    mask_sha256: str
    cutout_path: Optional[Path] = None
    cutout_sha256: Optional[str] = None
    camera_position_m: Optional[tuple[float, float, float]] = None
    point_cloud_path: Optional[Path] = None
    point_cloud_sha256: Optional[str] = None
    point_count: Optional[int] = None
    geometry_position_m: Optional[tuple[float, float, float]] = None
    geometry_dimensions_m: Optional[tuple[float, float, float]] = None
    geometry_source: Optional[str] = None


def sha256_file(path: Path | str) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_manifest_path(dsg_path: Path | str) -> Path:
    """Return ``dsg_updated.evidence.json`` for a query-ready DSG."""

    return Path(dsg_path).expanduser().resolve().with_suffix(".evidence.json")


def _require_sha256(value: Any, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise QueryEvidenceError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def _safe_evidence_image(root: Path, value: Any) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise QueryEvidenceError("evidence image paths must stay beside the manifest")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise QueryEvidenceError(
            "evidence image paths must stay beside the manifest"
        ) from exc
    if not resolved.is_file():
        raise QueryEvidenceError(f"evidence image does not exist: {resolved}")
    return resolved


def _optional_vector3(
    value: Any, *, field: str, positive: bool = False
) -> Optional[tuple[float, float, float]]:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or (positive and float(item) <= 0.0)
            for item in value
        )
    ):
        raise QueryEvidenceError(f"invalid evidence {field}")
    return tuple(float(item) for item in value)


def load_query_evidence(
    dsg_path: Path | str,
) -> tuple[dict[str, QueryEvidence], dict[str, QueryEvidence]]:
    """Load and validate the optional evidence sidecar for one exact DSG.

    Returns mappings by node ID and by public evidence ID.  A missing sidecar is
    valid and produces two empty mappings; a present but inconsistent sidecar is
    rejected so an image from another map can never be served as evidence.
    """

    dsg = Path(dsg_path).expanduser().resolve()
    manifest_path = evidence_manifest_path(dsg)
    if not manifest_path.exists():
        return {}, {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueryEvidenceError(
            f"failed to read query evidence manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != EVIDENCE_SCHEMA:
        raise QueryEvidenceError(
            f"unsupported query evidence manifest: {manifest_path}"
        )

    expected_dsg_digest = _require_sha256(
        manifest.get("dsg_sha256"), field="evidence dsg_sha256"
    )
    try:
        actual_dsg_digest = sha256_file(dsg)
    except OSError as exc:
        raise QueryEvidenceError(f"failed to hash query DSG: {dsg}") from exc
    if not compare_digest(expected_dsg_digest, actual_dsg_digest):
        raise QueryEvidenceError(
            f"query evidence belongs to a different DSG: {manifest_path}"
        )

    objects = manifest.get("objects")
    if not isinstance(objects, list):
        raise QueryEvidenceError("query evidence objects must be a list")
    by_node: dict[str, QueryEvidence] = {}
    by_id: dict[str, QueryEvidence] = {}
    for item in objects:
        if not isinstance(item, dict):
            raise QueryEvidenceError("query evidence object entries must be mappings")
        evidence_id = str(item.get("evidence_id", "")).strip()
        node_id = str(item.get("node_id", "")).strip()
        if not _EVIDENCE_ID_PATTERN.fullmatch(evidence_id):
            raise QueryEvidenceError(f"invalid evidence_id: {evidence_id!r}")
        if not node_id:
            raise QueryEvidenceError("query evidence node_id must not be empty")
        if node_id in by_node or evidence_id in by_id:
            raise QueryEvidenceError("query evidence IDs must be unique")
        mask_source = str(item.get("mask_source", ""))
        if mask_source != "fastsam_segmentation":
            raise QueryEvidenceError(
                f"unsupported query evidence mask_source: {mask_source!r}"
            )
        bbox = item.get("bbox_xyxy")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(value, int) for value in bbox)
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
        ):
            raise QueryEvidenceError(f"invalid evidence bbox for {node_id}")
        image_path = _safe_evidence_image(manifest_path.parent, item.get("image"))
        expected_image_digest = _require_sha256(
            item.get("image_sha256"), field=f"evidence image_sha256 for {node_id}"
        )
        try:
            actual_image_digest = sha256_file(image_path)
        except OSError as exc:
            raise QueryEvidenceError(
                f"failed to hash evidence image: {image_path}"
            ) from exc
        if not compare_digest(expected_image_digest, actual_image_digest):
            raise QueryEvidenceError(
                f"query evidence image checksum mismatch: {image_path}"
            )
        cutout_path = None
        cutout_digest = None
        if item.get("cutout") is not None:
            cutout_path = _safe_evidence_image(
                manifest_path.parent, item.get("cutout")
            )
            cutout_digest = _require_sha256(
                item.get("cutout_sha256"),
                field=f"evidence cutout_sha256 for {node_id}",
            )
            try:
                actual_cutout_digest = sha256_file(cutout_path)
            except OSError as exc:
                raise QueryEvidenceError(
                    f"failed to hash evidence cutout: {cutout_path}"
                ) from exc
            if not compare_digest(cutout_digest, actual_cutout_digest):
                raise QueryEvidenceError(
                    f"query evidence cutout checksum mismatch: {cutout_path}"
                )
        camera_position = _optional_vector3(
            item.get("camera_position_m"),
            field=f"camera_position_m for {node_id}",
        )
        point_cloud_path = None
        point_cloud_digest = None
        point_count = None
        geometry_position = None
        geometry_dimensions = None
        geometry_source = None
        if item.get("point_cloud") is not None:
            point_cloud_path = _safe_evidence_image(
                manifest_path.parent, item.get("point_cloud")
            )
            point_cloud_digest = _require_sha256(
                item.get("point_cloud_sha256"),
                field=f"evidence point_cloud_sha256 for {node_id}",
            )
            try:
                actual_point_cloud_digest = sha256_file(point_cloud_path)
            except OSError as exc:
                raise QueryEvidenceError(
                    f"failed to hash evidence point cloud: {point_cloud_path}"
                ) from exc
            if not compare_digest(point_cloud_digest, actual_point_cloud_digest):
                raise QueryEvidenceError(
                    f"query evidence point-cloud checksum mismatch: {point_cloud_path}"
                )
            point_count = int(item.get("point_count", 0))
            if point_count <= 0:
                raise QueryEvidenceError(
                    f"invalid evidence point_count for {node_id}"
                )
            geometry_position = _optional_vector3(
                item.get("geometry_position_m"),
                field=f"geometry_position_m for {node_id}",
            )
            geometry_dimensions = _optional_vector3(
                item.get("geometry_dimensions_m"),
                field=f"geometry_dimensions_m for {node_id}",
                positive=True,
            )
            if geometry_position is None or geometry_dimensions is None:
                raise QueryEvidenceError(
                    f"evidence point cloud has no geometry summary for {node_id}"
                )
            geometry_source = str(item.get("geometry_source") or "")
            if geometry_source != "fastsam_masked_rgbd_joint_backprojection":
                raise QueryEvidenceError(
                    f"invalid evidence geometry_source for {node_id}"
                )
            _require_sha256(
                item.get("source_depth_sha256"),
                field=f"source_depth_sha256 for {node_id}",
            )
        observed = item.get("observed_s")
        evidence = QueryEvidence(
            evidence_id=evidence_id,
            node_id=node_id,
            semantic_label=int(item["semantic_label"]),
            frame_index=int(item["frame_index"]),
            sensor_time_ns=int(item["sensor_time_ns"]),
            observed_s=None if observed is None else float(observed),
            bbox_xyxy=tuple(int(value) for value in bbox),
            mask_pixels=int(item["mask_pixels"]),
            mask_source=mask_source,
            image_path=image_path,
            image_sha256=expected_image_digest,
            source_image_sha256=_require_sha256(
                item.get("source_image_sha256"),
                field=f"source_image_sha256 for {node_id}",
            ),
            mask_sha256=_require_sha256(
                item.get("mask_sha256"), field=f"mask_sha256 for {node_id}"
            ),
            cutout_path=cutout_path,
            cutout_sha256=cutout_digest,
            camera_position_m=camera_position,
            point_cloud_path=point_cloud_path,
            point_cloud_sha256=point_cloud_digest,
            point_count=point_count,
            geometry_position_m=geometry_position,
            geometry_dimensions_m=geometry_dimensions,
            geometry_source=geometry_source,
        )
        if evidence.frame_index < 0 or evidence.sensor_time_ns <= 0:
            raise QueryEvidenceError(f"invalid evidence frame identity for {node_id}")
        if evidence.mask_pixels <= 0:
            raise QueryEvidenceError(f"empty evidence mask for {node_id}")
        by_node[node_id] = evidence
        by_id[evidence_id] = evidence
    return by_node, by_id


def infer_segmentation_frame_indices(
    sensor_times_ns: Iterable[int], segmentation_rate_hz: float
) -> list[int]:
    """Replay the exact timestamp scheduler used by the FastSAM sidecar."""

    if segmentation_rate_hz <= 0.0:
        raise ValueError("segmentation_rate_hz must be positive")
    period_ns = int(round(1.0e9 / float(segmentation_rate_hz)))
    indices: list[int] = []
    last_segmentation_ns: Optional[int] = None
    previous_ns: Optional[int] = None
    for index, raw_time in enumerate(sensor_times_ns):
        sensor_time_ns = int(raw_time)
        if sensor_time_ns <= 0:
            raise ValueError("sensor times must be absolute positive nanoseconds")
        if previous_ns is not None and sensor_time_ns <= previous_ns:
            raise ValueError("sensor times must be strictly increasing")
        previous_ns = sensor_time_ns
        if (
            last_segmentation_ns is None
            or sensor_time_ns - last_segmentation_ns >= period_ns
        ):
            indices.append(index)
            last_segmentation_ns = sensor_time_ns
    return indices
