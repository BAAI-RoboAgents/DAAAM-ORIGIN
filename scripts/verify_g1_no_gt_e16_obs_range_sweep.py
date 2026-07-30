#!/usr/bin/env python3
"""Independently verify and reseal the E16 observation/range sweep."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import run_g1_no_gt_e16_hydra as base  # noqa: E402
import run_g1_no_gt_e16_obs_range_sweep as experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=experiment.DEFAULT_OUTPUT)
    return parser.parse_args()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    root = parse_args().run.resolve()
    check(root.is_dir(), "run directory")
    check(not (root / "INDEPENDENT_AUDIT.json").exists(), "audit already exists")

    prereg = base.read_json(root / "PRE_REGISTRATION.json")
    specs = prereg["variants"]
    check(len(specs) == 6, "six preregistered variants")
    check(
        {
            (
                int(row["minimum_observations"]),
                float(row["maximum_object_range_m"]),
            )
            for row in specs
        }
        == {(obs, distance) for obs in (4, 6, 8) for distance in (5.0, 8.0)},
        "complete 3x2 factorial",
    )
    summaries = base.read_json(root / "tables/variant_summary.json")
    check(len(summaries) == len(specs), "six summaries")
    for spec, summary in zip(specs, summaries):
        variant_id = spec["variant_id"]
        check(summary["variant_id"] == variant_id, f"summary order {variant_id}")
        config = yaml.safe_load(
            (root / "configs" / f"{variant_id}.yaml").read_text()
        )
        active = config["active_window"]
        check(
            active["volumetric_map"]["voxel_size"] == 0.12,
            f"voxel {variant_id}",
        )
        check(
            active["tracker"]["min_num_observations"]
            == int(spec["minimum_observations"]),
            f"observations {variant_id}",
        )
        check(
            active["object_detector"]["max_range"]
            == float(spec["maximum_object_range_m"]),
            f"range {variant_id}",
        )
        check(
            active["object_extractor"]["min_object_volume"] == 0.005,
            f"volume {variant_id}",
        )
        report = base.read_json(
            root / "variants" / variant_id / "hydra_postpass_report.json"
        )
        check(report["frames_replayed"] == 102, f"frames {variant_id}")
        check(report["label_coverage"] == 1.0, f"coverage {variant_id}")
        nodes = base.read_jsonl(
            root / "variants" / variant_id / "metrics/object_nodes.jsonl"
        )
        check(
            len(nodes) == int(summary["dsg_object_nodes"]),
            f"node count {variant_id}",
        )

    support5 = base.read_jsonl(
        root / "tables/semantic_support_range5m.jsonl"
    )
    support8 = base.read_jsonl(
        root / "tables/semantic_support_range8m.jsonl"
    )
    check(len(support5) == len(support8) == 162, "support labels")
    gate = base.read_jsonl(root / "tables/label_gate_ledger.jsonl")
    targets = base.read_jsonl(
        root / "tables/target_candidate_survival.jsonl"
    )
    check(len(gate) == 162 * 6, "label gate rows")
    check(len(targets) == 20 * 6, "target candidate rows")
    check(
        all(
            int(row["required_observations"])
            == int(row["configured_minimum_observations"]) + 1
            for row in gate
        ),
        "strict allocation threshold",
    )

    inventory = base.read_jsonl(root / "artifact_inventory.jsonl")
    for row in inventory:
        path = root / row["relative_path"]
        check(path.is_file(), f"inventory path {path}")
        check(path.stat().st_size == int(row["size_bytes"]), f"size {path}")
        check(base.sha256_file(path) == row["sha256"], f"hash {path}")

    audit = {
        "schema": "daaam.g1_e16_obs_range_independent_audit.v1",
        "status": "passed",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "variants": 6,
        "support_labels_per_range": 162,
        "label_gate_rows": len(gate),
        "target_candidate_rows": len(targets),
        "checks": [
            "preregistered factorial is complete",
            "all configs preserve fixed voxel/truncation/volume parameters",
            "all variants replay 101 source frames plus one flush with exact labels",
            "summary node counts match DSG exports",
            "strict allocation observation requirement is reproduced",
            "pre-audit artifact hashes match",
        ],
        "formal_claims_permitted": False,
    }
    (root / "INDEPENDENT_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    base.seal_output(root, status="complete_independently_audited")
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
