#!/usr/bin/env python3
"""Verify and seal the Codex E16 adaptive-object comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


EXCLUDES = {
    "artifact_inventory.csv",
    "artifact_inventory.jsonl",
    "inventory_summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
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


def main() -> int:
    root = parse_args().comparison.resolve()
    complete = root / "adaptive_obs6_near5_far8_complete"
    selection = read_jsonl(complete / "selection_manifest.jsonl")
    adaptive = read_json(complete / "SUMMARY.json")
    graph = read_json(complete / "hydra_realtime/backend/dsg.json")
    targets = read_jsonl(root / "TARGET_REVIEW.jsonl")
    summary = read_json(root / "SUMMARY.json")

    object_nodes = [
        node
        for node in graph["nodes"]
        if node.get("layer") == 2 and node.get("partition", 0) == 0
    ]
    object_labels = [
        int(node["attributes"]["semantic_label"]) for node in object_nodes
    ]
    check(len(object_nodes) == 57, "adaptive object node count")
    check(len(object_labels) == len(set(object_labels)), "unique object labels")
    check(len(selection) == len(object_nodes), "selection/node count")
    check(
        {int(row["semantic_label"]) for row in selection}
        == set(object_labels),
        "selection/node label identity",
    )
    check(adaptive["near_object_count"] == 45, "near object count")
    check(adaptive["far_object_count"] == 55, "far object count")
    check(adaptive["adaptive_object_count"] == 57, "adaptive summary count")
    check(adaptive["near_selected"] == 44, "near selected count")
    check(adaptive["far_selected"] == 13, "far selected count")

    check(len(targets) == 20, "20 reviewed targets")
    check(
        len({row["instance_id"] for row in targets}) == 20,
        "unique target IDs",
    )
    core = [row for row in targets if row["denominator"] == "core"]
    inclusive = [
        row for row in targets if row["denominator"] == "inclusive_only"
    ]
    check(len(core) == 19 and len(inclusive) == 1, "19+1 denominator")
    check(
        Counter(row["adaptive_verdict"] for row in core)
        == Counter({"strict": 12, "partial": 5, "failure": 2}),
        "adaptive core verdict counts",
    )
    adaptive_row = next(
        row
        for row in summary["variant_codex_estimates"]
        if row["variant"] == "adaptive_obs6_near5_far8"
    )
    check(
        (
            adaptive_row["strict"],
            adaptive_row["partial"],
            adaptive_row["failure"],
        )
        == (12, 5, 2),
        "machine-readable adaptive summary",
    )

    checks = {
        "adaptive_object_nodes": len(object_nodes),
        "adaptive_unique_labels": len(set(object_labels)),
        "selection_rows": len(selection),
        "target_rows": len(targets),
        "core_verdicts": dict(
            Counter(row["adaptive_verdict"] for row in core)
        ),
        "inclusive_verdict": inclusive[0]["adaptive_verdict"],
    }
    integrity = {
        "schema": "daaam.g1_e16_adaptive_validation_integrity.v1",
        "status": "passed",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "formal_claims_permitted": False,
        "formal_status": (
            "single_codex_retrospective_engineering_estimate_not_human_gt"
        ),
    }
    (root / "INTEGRITY.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in EXCLUDES:
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    with (root / "artifact_inventory.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in rows:
            stream.write(
                json.dumps(row, separators=(",", ":")) + "\n"
            )
    with (root / "artifact_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    inventory = {
        "schema": "daaam.g1_e16_adaptive_validation_inventory.v1",
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "inventory_root_sha256": inventory_root(rows),
    }
    (root / "inventory_summary.json").write_text(
        json.dumps(inventory, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(inventory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
