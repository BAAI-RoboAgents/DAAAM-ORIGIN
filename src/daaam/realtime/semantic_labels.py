"""Durable exact-frame semantic labels for deterministic Hydra postpasses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

import cv2
import numpy as np


SEMANTIC_LABEL_DIRECTORY = "label_frames"
SEMANTIC_LABEL_METADATA_SCHEMA = "daaam.semantic_label_frame.v1"


def semantic_label_path(directory: Path | str, frame_index: int) -> Path:
    """Return the canonical label path for one replay frame."""

    if frame_index < 0:
        raise ValueError("semantic label frame index must be non-negative")
    return Path(directory) / f"{frame_index:08d}.png"


def semantic_label_metadata_path(
    directory: Path | str, frame_index: int
) -> Path:
    """Return the binding metadata path for one replay frame."""

    if frame_index < 0:
        raise ValueError("semantic label frame index must be non-negative")
    return Path(directory) / f"{frame_index:08d}.json"


def _validate_sha256(value: str, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def persist_semantic_label(
    directory: Path | str,
    frame_index: int,
    labels: np.ndarray,
    *,
    sensor_time_ns: int,
    run_configuration_sha256: str,
) -> dict[str, object]:
    """Atomically persist a lossless uint16 semantic label image.

    This function is called only by the independently scheduled semantic branch;
    the geometry branch never waits for this disk write.  The uint16 range check
    prevents OpenCV from silently clipping provisional Hydra label IDs.
    """

    if sensor_time_ns <= 0:
        raise ValueError("semantic label sensor time must be absolute nanoseconds")
    configuration_sha256 = _validate_sha256(
        run_configuration_sha256,
        field="semantic label run configuration",
    )
    array = np.asarray(labels)
    if array.ndim != 2:
        raise ValueError("semantic labels must be a two-dimensional image")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("semantic labels must use an integer dtype")
    minimum = int(array.min(initial=0))
    maximum = int(array.max(initial=0))
    if minimum < 0 or maximum > np.iinfo(np.uint16).max:
        raise ValueError(
            "semantic label IDs must fit losslessly in uint16 "
            f"(observed range {minimum}..{maximum})"
        )

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = semantic_label_path(root, frame_index)
    token = uuid.uuid4().hex
    temporary = root / f".{destination.stem}.{token}.tmp.png"
    encoded = array.astype(np.uint16, copy=False)
    try:
        if not cv2.imwrite(str(temporary), encoded):
            raise OSError(f"failed to encode semantic label image: {temporary}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    record = {
        "schema": SEMANTIC_LABEL_METADATA_SCHEMA,
        "frame_index": int(frame_index),
        "sensor_time_ns": int(sensor_time_ns),
        "run_configuration_sha256": configuration_sha256,
        "image": destination.name,
        "image_sha256": digest,
        "shape": [int(value) for value in encoded.shape],
        "dtype": "uint16",
        "minimum_label": minimum,
        "maximum_label": maximum,
        "nonzero_pixels": int(np.count_nonzero(encoded)),
    }
    metadata_path = semantic_label_metadata_path(root, frame_index)
    metadata_temporary = root / f".{metadata_path.stem}.{token}.tmp.json"
    try:
        metadata_temporary.write_text(
            json.dumps(record, indent=2, allow_nan=False, sort_keys=True) + "\n"
        )
        json.loads(metadata_temporary.read_text())
        metadata_temporary.replace(metadata_path)
    finally:
        metadata_temporary.unlink(missing_ok=True)
    return {
        **record,
        "path": str(destination),
        "metadata_path": str(metadata_path),
        "sha256": digest,
    }


def load_semantic_label(directory: Path | str, frame_index: int) -> np.ndarray:
    """Load one exact-frame label image without dtype coercion or fallback."""

    path = semantic_label_path(directory, frame_index)
    labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise FileNotFoundError(path)
    if labels.ndim != 2 or labels.dtype != np.uint16:
        raise ValueError(
            f"semantic label image must be single-channel uint16: {path}"
        )
    return labels


def validate_semantic_label_binding(
    directory: Path | str,
    frame_index: int,
    *,
    sensor_time_ns: int,
    run_configuration_sha256: str,
) -> dict[str, object]:
    """Validate that a label belongs to this exact frame and run configuration."""

    expected_configuration = _validate_sha256(
        run_configuration_sha256,
        field="semantic label run configuration",
    )
    image_path = semantic_label_path(directory, frame_index)
    metadata_path = semantic_label_metadata_path(directory, frame_index)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    try:
        record = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"semantic label metadata is invalid JSON: {metadata_path}"
        ) from error
    if record.get("schema") != SEMANTIC_LABEL_METADATA_SCHEMA:
        raise ValueError(f"unsupported semantic label metadata: {metadata_path}")
    if int(record.get("frame_index", -1)) != int(frame_index):
        raise ValueError(f"semantic label frame binding mismatch: {metadata_path}")
    if int(record.get("sensor_time_ns", -1)) != int(sensor_time_ns):
        raise ValueError(f"semantic label sensor-time binding mismatch: {metadata_path}")
    observed_configuration = _validate_sha256(
        record.get("run_configuration_sha256", ""),
        field="persisted semantic label run configuration",
    )
    if observed_configuration != expected_configuration:
        raise ValueError(
            f"semantic label run-configuration binding mismatch: {metadata_path}"
        )
    if record.get("image") != image_path.name:
        raise ValueError(f"semantic label image binding mismatch: {metadata_path}")
    observed_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if record.get("image_sha256") != observed_digest:
        raise ValueError(f"semantic label image hash mismatch: {image_path}")
    labels = load_semantic_label(directory, frame_index)
    if record.get("dtype") != "uint16" or record.get("shape") != [
        int(value) for value in labels.shape
    ]:
        raise ValueError(f"semantic label image metadata mismatch: {metadata_path}")
    return {
        **record,
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }
