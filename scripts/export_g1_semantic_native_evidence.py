#!/usr/bin/env python3
"""Losslessly export native G1 pipeline records for fine-grained analysis.

The source run is never opened for SQLite reads.  The database and sidecars are
copied first; all table exports are produced from a working copy.  Nested JSON
records are retained verbatim in JSONL and additionally written to CSV with
nested values encoded as JSON strings.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "daaam.g1_semantic_native_evidence_export.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry",
        type=Path,
        default=Path("output/g1_20260724_v1_v2_geometry"),
    )
    parser.add_argument(
        "--semantic",
        type=Path,
        default=Path("output/g1_20260724_v1_v2_semantic_map"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/g1_20260724_473_573_v1_1/"
            "diagnostic_no_gt_native_records_20260728"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            count += 1
    return count


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        if not fields:
            return
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: csv_value(row.get(field)) for field in fields}
            )


def export_collection(
    output: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    source: Path,
) -> dict[str, Any]:
    jsonl_path = output / "records" / f"{name}.jsonl"
    csv_path = output / "records" / f"{name}.csv"
    count = write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)
    return {
        "name": name,
        "rows": count,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "jsonl": str(jsonl_path.relative_to(output)),
        "jsonl_bytes": jsonl_path.stat().st_size,
        "jsonl_sha256": sha256_file(jsonl_path),
        "csv": str(csv_path.relative_to(output)),
        "csv_bytes": csv_path.stat().st_size,
        "csv_sha256": sha256_file(csv_path),
    }


def json_collections(
    geometry: Path, semantic: Path, output: Path
) -> list[dict[str, Any]]:
    exports = []

    def one(
        name: str,
        source: Path,
        key: str,
        *,
        decorate: str | None = None,
    ) -> None:
        rows = load_json(source).get(key, [])
        if decorate is not None:
            rows = [{decorate: key, **row} for row in rows]
        exports.append(
            export_collection(output, name, rows, source=source)
        )

    one(
        "keyframe_decisions",
        geometry / "02_selected/keyframe_selection_report.json",
        "decisions",
    )
    one(
        "depth_frame_stats",
        geometry / "02_selected/fast_foundation_stereo_run.json",
        "frame_stats",
    )
    one(
        "temporal_pairs_before_filter",
        geometry
        / "04_temporal_input/temporal_depth_consistency_report.json",
        "pairs",
    )
    one(
        "temporal_pairs_after_filter",
        geometry
        / "09_temporal_validation/temporal_depth_consistency_report.json",
        "pairs",
    )
    one(
        "temporal_filter_per_frame",
        geometry
        / "08_temporal_depth_filtered/temporal_depth_filter_report.json",
        "per_frame",
    )
    one(
        "rgbd_constraints",
        geometry / "05_rgbd_window_graph/trajectory_refinement.json",
        "constraints",
    )

    loop_source = geometry / "06_loop_closures/loop_closure_report.json"
    loop_report = load_json(loop_source)
    candidates = []
    for collection in (
        "retrieved_candidates",
        "geometric_candidates",
        "dense_candidates",
        "verified_links",
    ):
        candidates.extend(
            {
                "collection": collection,
                "rank": rank,
                **record,
            }
            for rank, record in enumerate(loop_report.get(collection, []))
        )
    exports.append(
        export_collection(
            output, "loop_candidates", candidates, source=loop_source
        )
    )

    quality_source = semantic / "quality_report.json"
    exports.append(
        export_collection(
            output,
            "quality_gate_results",
            load_json(quality_source).get("results", []),
            source=quality_source,
        )
    )
    realtime_source = semantic / "realtime_run_report.json"
    rejection_rows = (
        (load_json(realtime_source).get("semantic_stats") or {}).get("dsg")
        or {}
    ).get("rejection_audit", [])
    exports.append(
        export_collection(
            output,
            "dsg_binding_rejection_audit",
            rejection_rows,
            source=realtime_source,
        )
    )

    label_root = semantic / "semantic_sidecar/label_frames"
    label_sources = sorted(label_root.glob("*.json"))
    label_rows = [load_json(path) for path in label_sources]
    label_jsonl = output / "records/semantic_label_metadata.jsonl"
    label_csv = output / "records/semantic_label_metadata.csv"
    write_jsonl(label_jsonl, label_rows)
    write_csv(label_csv, label_rows)
    label_manifest = semantic / "hydra_semantic_postpass.json"
    exports.append(
        {
            "name": "semantic_label_metadata",
            "rows": len(label_rows),
            "source": str(label_root),
            "source_manifest": str(label_manifest),
            "source_manifest_sha256": sha256_file(label_manifest),
            "jsonl": str(label_jsonl.relative_to(output)),
            "jsonl_bytes": label_jsonl.stat().st_size,
            "jsonl_sha256": sha256_file(label_jsonl),
            "csv": str(label_csv.relative_to(output)),
            "csv_bytes": label_csv.stat().st_size,
            "csv_sha256": sha256_file(label_csv),
        }
    )
    return exports


def snapshot_database(
    database: Path, output: Path
) -> tuple[Path, list[dict[str, Any]]]:
    snapshot_root = output / "database/source_snapshot"
    working_root = output / "database/working_copy"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    working_root.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_root / database.name
    shutil.copy2(database, snapshot)
    members = [snapshot]
    for suffix in ("-wal", "-shm"):
        source = Path(str(database) + suffix)
        if source.is_file():
            target = Path(str(snapshot) + suffix)
            shutil.copy2(source, target)
            members.append(target)
    working = working_root / database.name
    for source in members:
        suffix = str(source).removeprefix(str(snapshot))
        shutil.copy2(source, Path(str(working) + suffix))
    evidence = [
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in members
    ]
    return working, evidence


def export_database(database: Path, output: Path) -> dict[str, Any]:
    before = {
        "bytes": database.stat().st_size,
        "mtime_ns": database.stat().st_mtime_ns,
        "sha256": sha256_file(database),
    }
    working, snapshot_evidence = snapshot_database(database, output)
    connection = sqlite3.connect(f"file:{working}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        integrity = [
            str(row[0]) for row in connection.execute("PRAGMA integrity_check")
        ]
        objects = [
            dict(row)
            for row in connection.execute(
                "SELECT name, type, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            )
        ]
        tables = [
            str(row["name"]) for row in objects if row["type"] == "table"
        ]
        exports = []
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            rows = [
                dict(row)
                for row in connection.execute(f"SELECT * FROM {quoted}")
            ]
            exports.append(
                export_collection(
                    output, f"map_memory_{table}", rows, source=database
                )
            )
    finally:
        connection.close()
    after = {
        "bytes": database.stat().st_size,
        "mtime_ns": database.stat().st_mtime_ns,
        "sha256": sha256_file(database),
    }
    report = {
        "source": str(database),
        "source_before": before,
        "source_after": after,
        "source_unchanged": before == after,
        "snapshot_evidence": snapshot_evidence,
        "working_copy": str(working.relative_to(output)),
        "integrity_check": integrity,
        "objects": objects,
        "table_exports": exports,
    }
    write_json(output / "database/database_export_manifest.json", report)
    return report


def generated_inventory(output: Path) -> dict[str, Any]:
    records = []
    excluded = {
        output / "artifact_inventory.json",
        output / "artifact_inventory.jsonl",
    }
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path in excluded:
            continue
        records.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_jsonl(output / "artifact_inventory.jsonl", records)
    report = {
        "files": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "records": records,
    }
    write_json(output / "artifact_inventory.json", report)
    return report


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    geometry = (
        args.geometry if args.geometry.is_absolute() else root / args.geometry
    ).resolve()
    semantic = (
        args.semantic if args.semantic.is_absolute() else root / args.semantic
    ).resolve()
    output = (
        args.output if args.output.is_absolute() else root / args.output
    ).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace evidence output: {output}")
    output.mkdir(parents=True)

    collections = json_collections(geometry, semantic, output)
    database = export_database(semantic / "map_memory.sqlite3", output)
    summary = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "native_records_no_human_gt",
        "accuracy_claims_allowed": False,
        "geometry": str(geometry),
        "semantic": str(semantic),
        "collections": collections,
        "database": database,
        "notes": [
            "JSONL exports retain every nested candidate/rejection field.",
            "CSV files are convenience views; nested values are JSON strings.",
            "The MapMemory source was copied before any SQLite connection.",
            "These records are system outputs under test, not ground truth.",
        ],
    }
    write_json(output / "summary.json", summary)
    inventory = generated_inventory(output)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "collections": len(collections),
                "collection_rows": sum(
                    int(record["rows"]) for record in collections
                ),
                "database_tables": len(database["table_exports"]),
                "database_rows": sum(
                    int(record["rows"])
                    for record in database["table_exports"]
                ),
                "derived_files": inventory["files"],
                "derived_bytes": inventory["bytes"],
                "source_database_unchanged": database["source_unchanged"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
