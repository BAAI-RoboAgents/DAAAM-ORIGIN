#!/usr/bin/env python3
"""Deterministically verify the frozen E13/E14 safety ablation evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from run_g1_e13_e14_safety_ablation import (  # noqa: E402
    e14_trigger_simulation,
    relation_audit,
)
from run_g1_no_gt_e13_entity_merge import variant_id  # noqa: E402


DEFAULT_RUN = (
    REPOSITORY_ROOT
    / "experiments/g1_20260724_473_573_v1_1/runs"
    / "diagnostic_gt_free_e13_e14_safety_ablation_20260730"
)
INVENTORY_EXCLUDES = {
    "artifact_inventory.csv",
    "artifact_inventory.jsonl",
    "inventory_summary.json",
    "COMPLETION.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_root(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                f"{row['relative_path']}\t{row['size_bytes']}\t"
                f"{row['sha256']}\n"
            ).encode()
        )
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_inventory(root: Path) -> dict[str, Any]:
    recorded = read_jsonl(root / "artifact_inventory.jsonl")
    recorded_by_path = {str(row["relative_path"]): row for row in recorded}
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in INVENTORY_EXCLUDES
    )
    check(actual == sorted(recorded_by_path), "inventory path set")
    for relative in actual:
        row = recorded_by_path[relative]
        path = root / relative
        check(path.stat().st_size == int(row["size_bytes"]), f"size: {relative}")
        check(sha256_file(path) == row["sha256"], f"sha256: {relative}")
    summary = read_json(root / "inventory_summary.json")
    root_hash = inventory_root(recorded)
    check(root_hash == summary["inventory_root_sha256"], "inventory root")
    check(len(recorded) == int(summary["file_count"]), "inventory count")
    return {
        "file_count": len(recorded),
        "inventory_root_sha256": root_hash,
    }


def main() -> int:
    args = parse_args()
    root = args.run.resolve()
    audit_path = root / "INDEPENDENT_AUDIT.json"
    check(root.is_dir(), "run directory")
    check(not audit_path.exists(), "audit already exists")
    inventory = verify_inventory(root)
    frozen = read_json(root / "FROZEN_INPUTS.json")
    for key in ("source_completion", "frames", "geometry_observations"):
        reference = frozen[key]
        path = Path(str(reference["path"]))
        check(path.stat().st_size == int(reference["size_bytes"]), f"{key} size")
        check(sha256_file(path) == reference["sha256"], f"{key} hash")
    observation_count = int(frozen["observation_count"])
    frame_count = int(frozen["frame_count"])

    prereg = read_json(root / "PRE_REGISTRATION.json")
    policies = tuple(str(value) for value in prereg["policies"])
    thresholds = tuple(float(value) for value in prereg["thresholds_m"])
    minimum_observations = int(prereg["e14_minimum_observations"])
    summaries = read_json(root / "tables/variant_summary.json")
    summary_by_id = {str(row["variant_id"]): row for row in summaries}
    relation_rows = read_jsonl(root / "tables/known_relation_audit.jsonl")
    relation_by_key = {
        (str(row["variant_id"]), str(row["relation_id"])): row
        for row in relation_rows
    }
    e14_rows = read_json(root / "tables/e14_trigger_ablation.json")
    e14_by_id = {str(row["variant_id"]): row for row in e14_rows}
    expected_ids = {
        variant_id(threshold, policy)
        for policy in policies
        for threshold in thresholds
    }
    check(set(summary_by_id) == expected_ids, "variant set")

    variant_checks = []
    total_event_checks = 0
    total_overlay_checks = 0
    for policy in policies:
        for threshold in thresholds:
            identifier = variant_id(threshold, policy)
            variant_root = root / "variants" / identifier
            events = read_jsonl(variant_root / "merge_events.jsonl")
            check(len(events) == observation_count, f"{identifier}: event count")
            check(
                all(str(row["association_policy"]) == policy for row in events),
                f"{identifier}: policy",
            )
            check(
                all(float(row["threshold_m"]) == threshold for row in events),
                f"{identifier}: threshold",
            )
            with sqlite3.connect(
                variant_root / "map_memory.sqlite3"
            ) as connection:
                integrity = connection.execute(
                    "PRAGMA integrity_check"
                ).fetchone()[0]
                entity_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM entities"
                    ).fetchone()[0]
                )
                database_observations = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM entity_observations"
                    ).fetchone()[0]
                )
            summary = summary_by_id[identifier]
            check(integrity == "ok", f"{identifier}: sqlite")
            check(
                entity_count == int(summary["entity_count"]),
                f"{identifier}: entity count",
            )
            check(
                database_observations == observation_count,
                f"{identifier}: database observations",
            )
            recomputed_relations = relation_audit(identifier, events)
            for row in recomputed_relations:
                recorded = relation_by_key[(identifier, row["relation_id"])]
                for key in ("relation_is_merged", "passed", "expected"):
                    check(
                        recorded[key] == row[key],
                        f"{identifier}: relation {row['relation_id']} {key}",
                    )
            recomputed_e14 = e14_trigger_simulation(
                identifier,
                events,
                minimum_observations,
            )
            check(
                recomputed_e14 == e14_by_id[identifier],
                f"{identifier}: E14 trigger simulation",
            )
            frame_rows = read_jsonl(variant_root / "frame_summary.jsonl")
            check(len(frame_rows) == frame_count, f"{identifier}: frame count")
            for frame in frame_rows:
                for path_key, hash_key in (
                    ("entity_id_map_path", "entity_id_map_sha256"),
                    ("entity_overlay_path", "entity_overlay_sha256"),
                ):
                    path = Path(str(frame[path_key]))
                    check(sha256_file(path) == frame[hash_key], hash_key)
                    total_overlay_checks += 1
            total_event_checks += len(events)
            variant_checks.append(
                {
                    "variant_id": identifier,
                    "event_count": len(events),
                    "entity_count": entity_count,
                    "sqlite_integrity_check": integrity,
                    "relation_checks": len(recomputed_relations),
                    "frame_checks": len(frame_rows),
                }
            )

    audit = {
        "schema": "daaam.g1_e13_e14_safety_ablation_independent_audit.v1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "auditor": "independent deterministic verifier",
        "verifier": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "passed": True,
        "result": "PASS",
        "checks": {
            "inventory": inventory,
            "frozen_reference_count": 3,
            "variant_count": len(variant_checks),
            "variant_checks": variant_checks,
            "merge_event_checks": total_event_checks,
            "image_hash_checks": total_overlay_checks,
        },
        "correctness_audited": False,
        "correctness_reason": (
            "This verifier proves deterministic accounting, hashes, SQLite "
            "integrity, relation logic, and E14 trigger simulation; the relation "
            "labels remain Codex approximate GT."
        ),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
