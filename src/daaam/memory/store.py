"""SQLite-backed editable, versioned, and cross-session semantic map memory."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import threading
from typing import Iterable, Optional
import uuid

import numpy as np

from daaam.realtime.contracts import SemanticCorrection


@dataclass(frozen=True)
class MapMemoryConfig:
    minimum_registration_inliers: int = 20
    maximum_registration_rms_m: float = 0.20
    maximum_registration_std: float = 0.50
    entity_merge_distance_m: float = 0.50

    def __post_init__(self) -> None:
        if self.minimum_registration_inliers < 3:
            raise ValueError("registration requires at least three inliers")
        if min(
            self.maximum_registration_rms_m,
            self.maximum_registration_std,
            self.entity_merge_distance_m,
        ) <= 0:
            raise ValueError("map memory quality thresholds must be positive")


@dataclass(frozen=True)
class SessionRegistration:
    session_id: str
    accepted: bool
    status: str
    reason: str
    inlier_count: int
    rms_error_m: float


@dataclass(frozen=True)
class CorrectionReceipt:
    operation_id: str
    status: str
    reason: str
    duplicate: bool = False


def _json_array(value: np.ndarray) -> str:
    return json.dumps(np.asarray(value, dtype=np.float64).tolist(), separators=(",", ":"))


def _parse_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=np.float64)


def _validate_transform(value: np.ndarray) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("session transform must be finite 4x4")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("session transform must be homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("session transform rotation must be orthonormal")
    return transform


def _validate_covariance(value: np.ndarray) -> np.ndarray:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
        raise ValueError("registration covariance must be finite 6x6")
    if not np.allclose(covariance, covariance.T, atol=1e-10):
        raise ValueError("registration covariance must be symmetric")
    if np.min(np.linalg.eigvalsh(covariance)) < -1e-10:
        raise ValueError("registration covariance must be positive semidefinite")
    return covariance


class MapMemory:
    """Transaction-safe persistent source of truth for map entities and edits."""

    def __init__(
        self,
        database_path: Path | str,
        config: MapMemoryConfig = MapMemoryConfig(),
    ) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path), timeout=30.0, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "MapMemory":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _create_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS revisions (
            revision INTEGER PRIMARY KEY,
            parent_revision INTEGER,
            sensor_time_ns INTEGER NOT NULL,
            reason TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_ns INTEGER NOT NULL,
            status TEXT NOT NULL,
            canonical_T_session TEXT,
            covariance TEXT,
            inlier_count INTEGER,
            rms_error_m REAL,
            reason TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            name_locked INTEGER NOT NULL DEFAULT 0,
            position_m TEXT,
            dimensions_m TEXT,
            geometry_confidence REAL NOT NULL DEFAULT 0.0,
            geometry_revision INTEGER NOT NULL,
            created_ns INTEGER NOT NULL,
            updated_ns INTEGER NOT NULL,
            deleted_ns INTEGER
        );
        CREATE TABLE IF NOT EXISTS aliases (
            entity_id TEXT NOT NULL REFERENCES entities(entity_id),
            alias TEXT COLLATE NOCASE NOT NULL,
            source TEXT NOT NULL,
            created_ns INTEGER NOT NULL,
            PRIMARY KEY(entity_id, alias)
        );
        CREATE INDEX IF NOT EXISTS aliases_name_idx ON aliases(alias);
        CREATE TABLE IF NOT EXISTS session_entities (
            session_id TEXT NOT NULL REFERENCES sessions(session_id),
            local_entity_id TEXT NOT NULL,
            entity_id TEXT NOT NULL REFERENCES entities(entity_id),
            PRIMARY KEY(session_id, local_entity_id)
        );
        CREATE TABLE IF NOT EXISTS semantic_operations (
            operation_id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            sensor_time_ns INTEGER NOT NULL,
            map_revision INTEGER NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            aliases_json TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_ns INTEGER NOT NULL,
            applied_ns INTEGER
        );
        CREATE INDEX IF NOT EXISTS semantic_pending_idx
            ON semantic_operations(status, map_revision, sensor_time_ns);
        CREATE TABLE IF NOT EXISTS semantic_deliveries (
            operation_id TEXT PRIMARY KEY REFERENCES semantic_operations(operation_id),
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            delivered_ns INTEGER
        );
        CREATE TABLE IF NOT EXISTS entity_versions (
            entity_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            sensor_time_ns INTEGER NOT NULL,
            action TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            PRIMARY KEY(entity_id, revision, sensor_time_ns, action)
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_time_ns INTEGER NOT NULL,
            map_revision INTEGER NOT NULL,
            action TEXT NOT NULL,
            entity_id TEXT,
            details_json TEXT NOT NULL
        );
        """
        with self._lock, self._connection:
            self._connection.executescript(schema)
            self._connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES('current_revision', '0')"
            )
            self._connection.execute(
                """INSERT OR IGNORE INTO revisions(
                    revision, parent_revision, sensor_time_ns, reason
                ) VALUES(0, NULL, 1, 'initial')"""
            )
            self._connection.execute(
                """UPDATE semantic_deliveries SET status='retry'
                    WHERE status='delivering'"""
            )

    @property
    def current_revision(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key='current_revision'"
            ).fetchone()
            return int(row[0])

    def _audit(
        self,
        action: str,
        sensor_time_ns: int,
        *,
        entity_id: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        self._connection.execute(
            """INSERT INTO audit_log(
                sensor_time_ns, map_revision, action, entity_id, details_json
            ) VALUES(?, ?, ?, ?, ?)""",
            (
                int(sensor_time_ns),
                self.current_revision,
                action,
                entity_id,
                json.dumps(details or {}, sort_keys=True),
            ),
        )

    def advance_revision(self, reason: str, sensor_time_ns: int) -> int:
        if not reason.strip() or sensor_time_ns <= 0:
            raise ValueError("revision update requires reason and absolute time")
        with self._lock, self._connection:
            previous = self.current_revision
            revision = previous + 1
            self._connection.execute(
                "INSERT INTO revisions VALUES(?, ?, ?, ?)",
                (revision, previous, sensor_time_ns, reason),
            )
            self._connection.execute(
                "UPDATE metadata SET value=? WHERE key='current_revision'",
                (str(revision),),
            )
            self._audit("advance_revision", sensor_time_ns, details={"reason": reason})
            return revision

    def create_session(
        self, session_id: str, started_ns: int, *, canonical: bool = False
    ) -> None:
        if not session_id.strip() or started_ns <= 0:
            raise ValueError("session requires id and absolute start time")
        status = "registered" if canonical else "isolated"
        transform = _json_array(np.eye(4)) if canonical else None
        covariance = _json_array(np.zeros((6, 6))) if canonical else None
        reason = "canonical_origin" if canonical else "awaiting_registration"
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO sessions(
                    session_id, started_ns, status, canonical_T_session,
                    covariance, inlier_count, rms_error_m, reason
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    started_ns,
                    status,
                    transform,
                    covariance,
                    0 if canonical else None,
                    0.0 if canonical else None,
                    reason,
                ),
            )

    def register_session(
        self,
        session_id: str,
        canonical_T_session: np.ndarray,
        covariance: np.ndarray,
        *,
        inlier_count: int,
        rms_error_m: float,
    ) -> SessionRegistration:
        transform = _validate_transform(canonical_T_session)
        uncertainty = _validate_covariance(covariance)
        if inlier_count < 0 or rms_error_m < 0 or not np.isfinite(rms_error_m):
            raise ValueError("registration quality values are invalid")
        maximum_std = float(np.sqrt(np.max(np.diag(uncertainty))))
        accepted = (
            inlier_count >= self.config.minimum_registration_inliers
            and rms_error_m <= self.config.maximum_registration_rms_m
            and maximum_std <= self.config.maximum_registration_std
        )
        if inlier_count < self.config.minimum_registration_inliers:
            reason = "insufficient_inliers"
        elif rms_error_m > self.config.maximum_registration_rms_m:
            reason = "registration_residual"
        elif maximum_std > self.config.maximum_registration_std:
            reason = "registration_covariance"
        else:
            reason = "verified"
        status = "registered" if accepted else "rejected"
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT 1 FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown session: {session_id}")
            self._connection.execute(
                """UPDATE sessions SET status=?, canonical_T_session=?, covariance=?,
                    inlier_count=?, rms_error_m=?, reason=? WHERE session_id=?""",
                (
                    status,
                    _json_array(transform),
                    _json_array(uncertainty),
                    inlier_count,
                    rms_error_m,
                    reason,
                    session_id,
                ),
            )
        return SessionRegistration(
            session_id, accepted, status, reason, inlier_count, rms_error_m
        )

    def _session_transform(self, session_id: str) -> np.ndarray:
        row = self._connection.execute(
            "SELECT status, canonical_T_session FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown session: {session_id}")
        if row["status"] != "registered":
            raise RuntimeError(f"session is not registered: {session_id}")
        return _parse_array(row["canonical_T_session"])

    def transform_session_points(
        self, session_id: str, points_m: np.ndarray
    ) -> np.ndarray:
        points = np.asarray(points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("session points must be Nx3")
        with self._lock:
            transform = self._session_transform(session_id)
        homogeneous = np.c_[points, np.ones(len(points))]
        return (homogeneous @ transform.T)[:, :3]

    def _entity_snapshot(self, entity_id: str) -> dict:
        row = self._connection.execute(
            "SELECT * FROM entities WHERE entity_id=?", (entity_id,)
        ).fetchone()
        if row is None:
            raise KeyError(entity_id)
        aliases = [
            alias[0]
            for alias in self._connection.execute(
                "SELECT alias FROM aliases WHERE entity_id=? ORDER BY alias", (entity_id,)
            )
        ]
        snapshot = dict(row)
        snapshot["aliases"] = aliases
        return snapshot

    def _record_entity_version(
        self, entity_id: str, sensor_time_ns: int, action: str
    ) -> None:
        snapshot = self._entity_snapshot(entity_id)
        self._connection.execute(
            """INSERT OR REPLACE INTO entity_versions(
                entity_id, revision, sensor_time_ns, action, snapshot_json
            ) VALUES(?, ?, ?, ?, ?)""",
            (
                entity_id,
                self.current_revision,
                sensor_time_ns,
                action,
                json.dumps(snapshot, sort_keys=True),
            ),
        )
        self._audit(action, sensor_time_ns, entity_id=entity_id)

    def _add_alias(
        self, entity_id: str, alias: str, source: str, sensor_time_ns: int
    ) -> None:
        cleaned = alias.strip()
        if not cleaned:
            return
        self._connection.execute(
            """INSERT OR IGNORE INTO aliases(entity_id, alias, source, created_ns)
                VALUES(?, ?, ?, ?)""",
            (entity_id, cleaned, source, sensor_time_ns),
        )

    def observe_entity(
        self,
        session_id: str,
        local_entity_id: str,
        position_session_m: np.ndarray,
        *,
        sensor_time_ns: int,
        semantic_label: str,
        dimensions_m: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        entity_type: str = "object",
    ) -> tuple[str, bool]:
        if not local_entity_id.strip() or sensor_time_ns <= 0:
            raise ValueError("local entity observation is invalid")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("entity confidence must be in [0, 1]")
        position = np.asarray(position_session_m, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("entity position must be finite xyz")
        dimensions = None if dimensions_m is None else np.asarray(dimensions_m, dtype=np.float64)
        if dimensions is not None and (dimensions.shape != (3,) or np.any(dimensions <= 0)):
            raise ValueError("entity dimensions must be positive xyz")

        with self._lock, self._connection:
            transform = self._session_transform(session_id)
            canonical_position = (transform @ np.r_[position, 1.0])[:3]
            mapped = self._connection.execute(
                """SELECT entity_id FROM session_entities
                    WHERE session_id=? AND local_entity_id=?""",
                (session_id, local_entity_id),
            ).fetchone()
            matched_entity = str(mapped[0]) if mapped is not None else None
            matched_distance = float("inf")
            if matched_entity is None:
                candidates = self._connection.execute(
                    "SELECT * FROM entities WHERE deleted_ns IS NULL AND entity_type=?",
                    (entity_type,),
                ).fetchall()
                label_key = semantic_label.strip().casefold()
                for candidate in candidates:
                    if candidate["position_m"] is None:
                        continue
                    names = {str(candidate["canonical_name"]).casefold()}
                    names.update(
                        str(row[0]).casefold()
                        for row in self._connection.execute(
                            "SELECT alias FROM aliases WHERE entity_id=?",
                            (candidate["entity_id"],),
                        )
                    )
                    if label_key and label_key not in names:
                        continue
                    distance = float(
                        np.linalg.norm(_parse_array(candidate["position_m"]) - canonical_position)
                    )
                    if distance <= self.config.entity_merge_distance_m and distance < matched_distance:
                        matched_entity = str(candidate["entity_id"])
                        matched_distance = distance

            created = matched_entity is None
            entity_id = matched_entity or f"entity-{uuid.uuid4().hex}"
            if created:
                self._connection.execute(
                    """INSERT INTO entities(
                        entity_id, entity_type, canonical_name, name_locked,
                        position_m, dimensions_m, geometry_confidence,
                        geometry_revision, created_ns, updated_ns, deleted_ns
                    ) VALUES(?, ?, ?, 0, ?, ?, ?, ?, ?, ?, NULL)""",
                    (
                        entity_id,
                        entity_type,
                        semantic_label.strip() or "unknown",
                        _json_array(canonical_position),
                        _json_array(dimensions) if dimensions is not None else None,
                        confidence,
                        self.current_revision,
                        sensor_time_ns,
                        sensor_time_ns,
                    ),
                )
            else:
                existing = self._connection.execute(
                    "SELECT position_m, geometry_confidence FROM entities WHERE entity_id=?",
                    (entity_id,),
                ).fetchone()
                old_confidence = float(existing["geometry_confidence"])
                weight = confidence / max(confidence + old_confidence, 1e-9)
                updated_position = (
                    (1.0 - weight) * _parse_array(existing["position_m"])
                    + weight * canonical_position
                )
                self._connection.execute(
                    """UPDATE entities SET position_m=?, geometry_confidence=?,
                        geometry_revision=?, updated_ns=? WHERE entity_id=?""",
                    (
                        _json_array(updated_position),
                        max(confidence, old_confidence),
                        self.current_revision,
                        sensor_time_ns,
                        entity_id,
                    ),
                )
                self._add_alias(entity_id, semantic_label, "observation", sensor_time_ns)
            self._connection.execute(
                "INSERT OR IGNORE INTO session_entities VALUES(?, ?, ?)",
                (session_id, local_entity_id, entity_id),
            )
            self._record_entity_version(
                entity_id, sensor_time_ns, "entity_created" if created else "entity_observed"
            )
            return entity_id, created

    def set_user_name(
        self,
        entity_id: str,
        canonical_name: str,
        *,
        sensor_time_ns: int,
        aliases: Iterable[str] = (),
        lock: bool = True,
    ) -> None:
        cleaned = canonical_name.strip()
        if not cleaned or sensor_time_ns <= 0:
            raise ValueError("user name and absolute time are required")
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT canonical_name FROM entities WHERE entity_id=? AND deleted_ns IS NULL",
                (entity_id,),
            ).fetchone()
            if row is None:
                raise KeyError(entity_id)
            previous_name = str(row["canonical_name"])
            if previous_name.casefold() != cleaned.casefold():
                self._add_alias(entity_id, previous_name, "previous_name", sensor_time_ns)
            self._connection.execute(
                """UPDATE entities SET canonical_name=?, name_locked=?, updated_ns=?
                    WHERE entity_id=?""",
                (cleaned, int(lock), sensor_time_ns, entity_id),
            )
            for alias in aliases:
                self._add_alias(entity_id, alias, "user", sensor_time_ns)
            self._record_entity_version(entity_id, sensor_time_ns, "user_name")

    def get_entity(self, entity_id: str) -> dict:
        with self._lock:
            snapshot = self._entity_snapshot(entity_id)
        if snapshot["position_m"] is not None:
            snapshot["position_m"] = json.loads(snapshot["position_m"])
        if snapshot["dimensions_m"] is not None:
            snapshot["dimensions_m"] = json.loads(snapshot["dimensions_m"])
        snapshot["name_locked"] = bool(snapshot["name_locked"])
        return snapshot

    def list_entities(self, *, include_deleted: bool = False) -> list[dict]:
        predicate = "" if include_deleted else " WHERE deleted_ns IS NULL"
        with self._lock:
            identifiers = [
                str(row[0])
                for row in self._connection.execute(
                    f"SELECT entity_id FROM entities{predicate} ORDER BY updated_ns DESC, entity_id"
                )
            ]
        return [self.get_entity(identifier) for identifier in identifiers]

    def list_sessions(self) -> list[dict]:
        with self._lock:
            rows = [
                dict(row)
                for row in self._connection.execute(
                    "SELECT * FROM sessions ORDER BY started_ns, session_id"
                )
            ]
        for row in rows:
            for field in ("canonical_T_session", "covariance"):
                if row[field] is not None:
                    row[field] = json.loads(row[field])
        return rows

    def add_user_alias(
        self,
        entity_id: str,
        alias: str,
        *,
        sensor_time_ns: int,
    ) -> None:
        if not alias.strip() or sensor_time_ns <= 0:
            raise ValueError("user alias and absolute time are required")
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT 1 FROM entities WHERE entity_id=? AND deleted_ns IS NULL",
                (entity_id,),
            ).fetchone()
            if row is None:
                raise KeyError(entity_id)
            self._add_alias(entity_id, alias, "user", sensor_time_ns)
            self._record_entity_version(entity_id, sensor_time_ns, "user_alias")

    def find_by_name(self, name: str) -> list[dict]:
        query = name.strip().casefold()
        if not query:
            return []
        with self._lock:
            identifiers = {
                str(row[0])
                for row in self._connection.execute(
                    "SELECT entity_id FROM entities WHERE lower(canonical_name)=?",
                    (query,),
                )
            }
            identifiers.update(
                str(row[0])
                for row in self._connection.execute(
                    "SELECT entity_id FROM aliases WHERE lower(alias)=?", (query,)
                )
            )
        return [self.get_entity(identifier) for identifier in sorted(identifiers)]

    def stats(self) -> dict:
        with self._lock:
            entity_rows = self._connection.execute(
                "SELECT deleted_ns IS NOT NULL AS deleted, COUNT(*) FROM entities GROUP BY deleted"
            ).fetchall()
            session_rows = self._connection.execute(
                "SELECT status, COUNT(*) FROM sessions GROUP BY status"
            ).fetchall()
        entity_counts = {
            "active": 0,
            "deleted": 0,
        }
        for deleted, count in entity_rows:
            entity_counts["deleted" if deleted else "active"] = int(count)
        return {
            "current_revision": self.current_revision,
            "entities": entity_counts,
            "sessions": {str(status): int(count) for status, count in session_rows},
            "corrections": self.correction_stats(),
            "deliveries": self.delivery_stats(),
        }

    def enqueue_correction(self, correction: SemanticCorrection) -> CorrectionReceipt:
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT status, reason FROM semantic_operations WHERE operation_id=?",
                (correction.operation_id,),
            ).fetchone()
            if existing is not None:
                return CorrectionReceipt(
                    correction.operation_id,
                    str(existing["status"]),
                    str(existing["reason"]),
                    duplicate=True,
                )

            status = "pending"
            reason = "queued"
            pending = self._connection.execute(
                """SELECT operation_id, sensor_time_ns, confidence
                    FROM semantic_operations
                    WHERE entity_id=? AND map_revision=? AND status='pending'""",
                (correction.entity_id, correction.map_revision),
            ).fetchall()
            incoming_rank = (correction.sensor_time_ns, correction.confidence)
            for row in pending:
                old_rank = (int(row["sensor_time_ns"]), float(row["confidence"]))
                if incoming_rank >= old_rank:
                    self._connection.execute(
                        """UPDATE semantic_operations SET status='superseded',
                            reason='newer_entity_operation' WHERE operation_id=?""",
                        (row["operation_id"],),
                    )
                    self._connection.execute(
                        """UPDATE semantic_deliveries SET status='superseded'
                            WHERE operation_id=?""",
                        (row["operation_id"],),
                    )
                else:
                    status = "superseded"
                    reason = "newer_entity_operation_exists"
                    break
            self._connection.execute(
                """INSERT INTO semantic_operations(
                    operation_id, entity_id, sensor_time_ns, map_revision, label,
                    confidence, source, aliases_json, status, reason, created_ns, applied_ns
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
                (
                    correction.operation_id,
                    correction.entity_id,
                    correction.sensor_time_ns,
                    correction.map_revision,
                    correction.label,
                    correction.confidence,
                    correction.source,
                    json.dumps(correction.aliases),
                    status,
                    reason,
                    correction.sensor_time_ns,
                ),
            )
            self._connection.execute(
                """INSERT INTO semantic_deliveries(
                    operation_id, status, attempts, last_error, delivered_ns
                ) VALUES(?, ?, 0, NULL, NULL)""",
                (
                    correction.operation_id,
                    "waiting" if status == "pending" else status,
                ),
            )
            return CorrectionReceipt(correction.operation_id, status, reason)

    def apply_pending_corrections(self, *, limit: int = 100) -> list[CorrectionReceipt]:
        if limit <= 0:
            raise ValueError("correction limit must be positive")
        receipts = []
        with self._lock, self._connection:
            revision = self.current_revision
            rows = self._connection.execute(
                """SELECT * FROM semantic_operations WHERE status='pending'
                    AND map_revision <= ? ORDER BY sensor_time_ns, operation_id LIMIT ?""",
                (revision, limit),
            ).fetchall()
            for row in rows:
                operation_id = str(row["operation_id"])
                entity_id = str(row["entity_id"])
                if int(row["map_revision"]) < revision:
                    status, reason = "rejected", "stale_revision"
                else:
                    entity = self._connection.execute(
                        """SELECT canonical_name, name_locked FROM entities
                            WHERE entity_id=? AND deleted_ns IS NULL""",
                        (entity_id,),
                    ).fetchone()
                    if entity is None:
                        status, reason = "rejected", "missing_entity"
                    elif bool(entity["name_locked"]) and row["source"] != "user":
                        self._add_alias(
                            entity_id,
                            str(row["label"]),
                            "automatic_correction",
                            int(row["sensor_time_ns"]),
                        )
                        for alias in json.loads(row["aliases_json"]):
                            self._add_alias(
                                entity_id,
                                alias,
                                "automatic_correction",
                                int(row["sensor_time_ns"]),
                            )
                        self._record_entity_version(
                            entity_id, int(row["sensor_time_ns"]), "semantic_alias"
                        )
                        status, reason = "applied_alias", "user_name_locked"
                    else:
                        previous = str(entity["canonical_name"])
                        label = str(row["label"]).strip()
                        if previous.casefold() != label.casefold():
                            self._add_alias(
                                entity_id, previous, "previous_name", int(row["sensor_time_ns"])
                            )
                        self._connection.execute(
                            """UPDATE entities SET canonical_name=?, updated_ns=?
                                WHERE entity_id=?""",
                            (label, int(row["sensor_time_ns"]), entity_id),
                        )
                        for alias in json.loads(row["aliases_json"]):
                            self._add_alias(
                                entity_id, alias, str(row["source"]), int(row["sensor_time_ns"])
                            )
                        self._record_entity_version(
                            entity_id, int(row["sensor_time_ns"]), "semantic_correction"
                        )
                        status, reason = "applied", "updated"
                self._connection.execute(
                    """UPDATE semantic_operations SET status=?, reason=?, applied_ns=?
                        WHERE operation_id=?""",
                    (status, reason, int(row["sensor_time_ns"]), operation_id),
                )
                delivery_status = (
                    "ready" if status in {"applied", "applied_alias"} else status
                )
                self._connection.execute(
                    """UPDATE semantic_deliveries SET status=? WHERE operation_id=?""",
                    (delivery_status, operation_id),
                )
                receipts.append(CorrectionReceipt(operation_id, status, reason))
        return receipts

    def get_correction(self, operation_id: str) -> SemanticCorrection:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM semantic_operations WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(operation_id)
        return SemanticCorrection(
            operation_id=str(row["operation_id"]),
            entity_id=str(row["entity_id"]),
            sensor_time_ns=int(row["sensor_time_ns"]),
            map_revision=int(row["map_revision"]),
            label=str(row["label"]),
            confidence=float(row["confidence"]),
            source=str(row["source"]),
            aliases=tuple(json.loads(row["aliases_json"])),
        )

    def claim_semantic_deliveries(
        self, *, limit: int = 20
    ) -> list[tuple[SemanticCorrection, str]]:
        if limit <= 0:
            raise ValueError("delivery limit must be positive")
        claimed: list[tuple[SemanticCorrection, str]] = []
        with self._lock, self._connection:
            rows = self._connection.execute(
                """SELECT d.operation_id, o.entity_id
                    FROM semantic_deliveries d
                    JOIN semantic_operations o ON o.operation_id=d.operation_id
                    WHERE d.status IN ('ready', 'retry')
                    ORDER BY o.sensor_time_ns, o.operation_id LIMIT ?""",
                (limit,),
            ).fetchall()
            for row in rows:
                operation_id = str(row["operation_id"])
                self._connection.execute(
                    """UPDATE semantic_deliveries
                        SET status='delivering', attempts=attempts+1
                        WHERE operation_id=?""",
                    (operation_id,),
                )
                entity = self._connection.execute(
                    "SELECT canonical_name FROM entities WHERE entity_id=?",
                    (row["entity_id"],),
                ).fetchone()
                effective_label = (
                    str(entity["canonical_name"]) if entity is not None else "unknown"
                )
                claimed.append((self.get_correction(operation_id), effective_label))
        return claimed

    def complete_semantic_delivery(
        self,
        operation_id: str,
        *,
        success: bool,
        sensor_time_ns: int,
        error: Optional[str] = None,
        maximum_attempts: int = 3,
    ) -> str:
        if sensor_time_ns <= 0 or maximum_attempts <= 0:
            raise ValueError("delivery completion settings are invalid")
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT status, attempts FROM semantic_deliveries WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(operation_id)
            if row["status"] == "delivered":
                return "delivered"
            if success:
                status = "delivered"
            elif int(row["attempts"]) >= maximum_attempts:
                status = "failed"
            else:
                status = "retry"
            self._connection.execute(
                """UPDATE semantic_deliveries SET status=?, last_error=?, delivered_ns=?
                    WHERE operation_id=?""",
                (
                    status,
                    None if success else (error or "consumer_error"),
                    sensor_time_ns if success else None,
                    operation_id,
                ),
            )
            return status

    def delivery_stats(self) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT status, COUNT(*) AS count FROM semantic_deliveries GROUP BY status"
            ).fetchall()
        stats = {str(row["status"]): int(row["count"]) for row in rows}
        for status in (
            "waiting",
            "ready",
            "retry",
            "delivering",
            "delivered",
            "failed",
            "rejected",
            "superseded",
        ):
            stats.setdefault(status, 0)
        return stats

    def correction_stats(self) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT status, COUNT(*) AS count FROM semantic_operations GROUP BY status"
            ).fetchall()
        stats = {str(row["status"]): int(row["count"]) for row in rows}
        for status in ("pending", "applied", "applied_alias", "rejected", "superseded"):
            stats.setdefault(status, 0)
        return stats

    def delete_entity(self, entity_id: str, *, sensor_time_ns: int) -> None:
        with self._lock, self._connection:
            updated = self._connection.execute(
                """UPDATE entities SET deleted_ns=?, updated_ns=?
                    WHERE entity_id=? AND deleted_ns IS NULL""",
                (sensor_time_ns, sensor_time_ns, entity_id),
            )
            if updated.rowcount != 1:
                raise KeyError(entity_id)
            self._record_entity_version(entity_id, sensor_time_ns, "entity_deleted")

    def rollback_to_revision(self, target_revision: int, *, sensor_time_ns: int) -> int:
        if target_revision < 0 or target_revision > self.current_revision:
            raise ValueError("rollback target revision is invalid")
        with self._lock, self._connection:
            entities = [
                str(row[0]) for row in self._connection.execute("SELECT entity_id FROM entities")
            ]
            snapshots: dict[str, Optional[dict]] = {}
            for entity_id in entities:
                row = self._connection.execute(
                    """SELECT snapshot_json FROM entity_versions
                        WHERE entity_id=? AND revision<=?
                        ORDER BY revision DESC, sensor_time_ns DESC LIMIT 1""",
                    (entity_id, target_revision),
                ).fetchone()
                snapshots[entity_id] = json.loads(row[0]) if row is not None else None
            revision = self.advance_revision(
                f"rollback_to_{target_revision}", sensor_time_ns
            )
            for entity_id, snapshot in snapshots.items():
                if snapshot is None:
                    self._connection.execute(
                        "UPDATE entities SET deleted_ns=?, updated_ns=? WHERE entity_id=?",
                        (sensor_time_ns, sensor_time_ns, entity_id),
                    )
                    continue
                columns = (
                    "entity_type",
                    "canonical_name",
                    "name_locked",
                    "position_m",
                    "dimensions_m",
                    "geometry_confidence",
                    "created_ns",
                    "deleted_ns",
                )
                self._connection.execute(
                    f"""UPDATE entities SET {', '.join(column + '=?' for column in columns)},
                        geometry_revision=?, updated_ns=? WHERE entity_id=?""",
                    tuple(snapshot[column] for column in columns)
                    + (revision, sensor_time_ns, entity_id),
                )
                self._connection.execute("DELETE FROM aliases WHERE entity_id=?", (entity_id,))
                for alias in snapshot.get("aliases", []):
                    self._add_alias(entity_id, alias, "rollback", sensor_time_ns)
                self._record_entity_version(entity_id, sensor_time_ns, "rollback_restore")
            return revision

    def audit_log(self) -> list[dict]:
        with self._lock:
            return [
                dict(row)
                for row in self._connection.execute(
                    "SELECT * FROM audit_log ORDER BY audit_id"
                )
            ]
