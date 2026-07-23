"""Checksum-bound semantic sidecar records for query and visualization.

The authoritative DSG only contains entities that can be attached to real Hydra
object meshes.  DAM descriptions that have valid MapMemory geometry but no
object mesh remain useful for semantic retrieval and image evidence.  This
module keeps those records in a separate, explicitly lower-confidence sidecar
instead of materializing geometry-free nodes in the authoritative graph.
"""

from __future__ import annotations

from hmac import compare_digest
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
import uuid


QUERY_INDEX_SCHEMA = "daaam.semantic_query_index.v1"
QUERY_INDEX_STATUSES = frozenset({"spatial_only", "image_only"})


class QueryIndexError(RuntimeError):
    """Raised when a semantic query sidecar is unsafe or inconsistent."""


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 digest for one local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_index_path(dsg_path: Path | str) -> Path:
    """Return the conventional sidecar path beside one query-ready DSG."""

    return Path(dsg_path).expanduser().resolve().with_suffix(".semantic.json")


def _require_sha256(value: Any, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise QueryIndexError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def _safe_sidecar_path(dsg: Path, value: Any) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise QueryIndexError("semantic-index paths must stay beside the DSG")
    resolved = (dsg.parent / relative).resolve()
    try:
        resolved.relative_to(dsg.parent.resolve())
    except ValueError as exc:
        raise QueryIndexError("semantic-index paths must stay beside the DSG") from exc
    if not resolved.is_file():
        raise QueryIndexError(f"semantic query index does not exist: {resolved}")
    return resolved


def _normalized_record(item: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise QueryIndexError(f"semantic query record {index} must be an object")
    record_id = str(item.get("record_id", "")).strip()
    entity_id = str(item.get("entity_id", "")).strip()
    description = " ".join(str(item.get("description", "")).split()).strip()
    geometry_status = str(item.get("geometry_status", "")).strip()
    if not record_id or not entity_id or not description:
        raise QueryIndexError(
            f"semantic query record {index} has an incomplete identity or description"
        )
    if geometry_status not in QUERY_INDEX_STATUSES:
        raise QueryIndexError(
            f"semantic query record {record_id} has invalid geometry_status"
        )
    try:
        semantic_label = int(item["semantic_label"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QueryIndexError(
            f"semantic query record {record_id} has no semantic label"
        ) from exc
    if semantic_label <= 0:
        raise QueryIndexError(
            f"semantic query record {record_id} has an invalid semantic label"
        )

    normalized = dict(item)
    normalized.update(
        {
            "record_id": record_id,
            "entity_id": entity_id,
            "semantic_label": semantic_label,
            "description": description,
            "geometry_status": geometry_status,
        }
    )
    return normalized


def validate_query_records(records: Iterable[Any]) -> list[dict[str, Any]]:
    """Validate identities and return deterministic semantic-sidecar records."""

    normalized = [
        _normalized_record(item, index=index) for index, item in enumerate(records)
    ]
    identifiers: set[str] = set()
    entity_ids: set[str] = set()
    semantic_labels: set[int] = set()
    for record in normalized:
        for value, seen, field in (
            (record["record_id"], identifiers, "record_id"),
            (record["entity_id"], entity_ids, "entity_id"),
            (record["semantic_label"], semantic_labels, "semantic_label"),
        ):
            if value in seen:
                raise QueryIndexError(f"duplicate semantic query {field}: {value}")
            seen.add(value)
    return sorted(
        normalized,
        key=lambda record: (record["semantic_label"], record["entity_id"]),
    )


def write_query_index(
    dsg_path: Path | str,
    records: Iterable[Mapping[str, Any]],
    *,
    source: Optional[Mapping[str, Any]] = None,
    output_path: Path | str | None = None,
) -> tuple[Path, str]:
    """Atomically write a semantic sidecar tied to the exact DSG bytes."""

    dsg = Path(dsg_path).expanduser().resolve()
    target = (
        semantic_index_path(dsg)
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    if target.parent != dsg.parent:
        raise QueryIndexError("semantic query index must be stored beside the DSG")
    validated = validate_query_records(records)
    payload = {
        "schema": QUERY_INDEX_SCHEMA,
        "dsg_file": dsg.name,
        "dsg_sha256": sha256_file(dsg),
        "record_count": len(validated),
        "geometry_counts": {
            status: sum(record["geometry_status"] == status for record in validated)
            for status in sorted(QUERY_INDEX_STATUSES)
        },
        "source": dict(source or {}),
        "records": validated,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        json.loads(temporary.read_text(encoding="utf-8"))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target, sha256_file(target)


def load_query_index(
    dsg_path: Path | str,
    *,
    manifest: Optional[Mapping[str, Any]] = None,
    allow_conventional_path: bool = False,
) -> list[dict[str, Any]]:
    """Load and checksum-validate optional lower-confidence query records.

    Query services should pass the DSG manifest so the sidecar checksum is part
    of the deployment contract.  Visualization may opt into conventional-path
    discovery, but the sidecar still has to bind itself to the exact DSG hash.
    """

    dsg = Path(dsg_path).expanduser().resolve()
    reference = None if manifest is None else manifest.get("semantic_index")
    if reference is None:
        if not allow_conventional_path:
            return []
        path = semantic_index_path(dsg)
        if not path.exists():
            return []
        expected_index_digest = None
    else:
        if not isinstance(reference, Mapping):
            raise QueryIndexError("manifest semantic_index must be an object")
        path = _safe_sidecar_path(dsg, reference.get("file"))
        expected_index_digest = _require_sha256(
            reference.get("sha256"), field="semantic_index.sha256"
        )

    if expected_index_digest is not None:
        actual_index_digest = sha256_file(path)
        if not compare_digest(expected_index_digest, actual_index_digest):
            raise QueryIndexError(f"semantic query index checksum mismatch: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueryIndexError(f"failed to read semantic query index: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != QUERY_INDEX_SCHEMA:
        raise QueryIndexError(f"unsupported semantic query index: {path}")
    expected_dsg_digest = _require_sha256(
        payload.get("dsg_sha256"), field="semantic index dsg_sha256"
    )
    actual_dsg_digest = sha256_file(dsg)
    if not compare_digest(expected_dsg_digest, actual_dsg_digest):
        raise QueryIndexError(f"semantic query index belongs to another DSG: {path}")
    records = validate_query_records(payload.get("records", []))
    if int(payload.get("record_count", -1)) != len(records):
        raise QueryIndexError(f"semantic query index record count is inconsistent: {path}")
    if reference is not None and reference.get("records") is not None:
        if int(reference["records"]) != len(records):
            raise QueryIndexError("manifest semantic-index record count is inconsistent")
    return records
