#!/usr/bin/env python3
"""Integrate temporal and sparse-LiDAR D0 observers without overstating scope."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, output: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    excluded = {
        output.resolve(),
        (output.parent / "final_artifact_inventory.jsonl").resolve(),
        (output.parent / "final_inventory_summary.json").resolve(),
    }
    for path in sorted(root.rglob("*")):
        if path.resolve() in excluded:
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            records.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": str(path.readlink()),
                    "size_bytes": None,
                    "sha256": None,
                }
            )
        elif path.is_file():
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "target": None,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = output.parent / "final_artifact_inventory.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
    summary = {
        "schema": "daaam.no_gt_d0_final_inventory.v1",
        "root": str(root),
        "records": len(records),
        "regular_files": sum(record["kind"] == "file" for record in records),
        "symlinks": sum(record["kind"] == "symlink" for record in records),
        "regular_file_bytes": sum(
            int(record["size_bytes"] or 0) for record in records
        ),
        "regular_files_hashed": sum(
            record["kind"] == "file" and bool(record["sha256"])
            for record in records
        ),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
    }
    write_json(output.parent / "final_inventory_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    d0_root = args.d0_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    output.mkdir(parents=True)

    temporal_path = d0_root / "d0_partial_qualification.json"
    scale_path = d0_root / "lidar_scale_d0/qualification.json"
    temporal = read_json(temporal_path)
    scale = read_json(scale_path)
    medium_heavy = {
        record["variant_id"]: record for record in temporal["medium_heavy_variants"]
    }
    pose_translation_pass = all(
        medium_heavy[name]["passed"]
        for name in ("pose_translation_05cm", "pose_translation_10cm")
    )
    pose_yaw_pass = all(
        medium_heavy[name]["passed"] for name in ("pose_yaw_1deg", "pose_yaw_3deg")
    )
    pose_direction_pass = all(
        temporal["ordered_direction"][family][
            "at_least_one_primary_metric_monotone"
        ]
        for family in ("pose_translation", "pose_yaw")
    )
    tested = {
        "depth_scale": {
            "primary_observer": "E4 sparse camera-LiDAR signed residual",
            "status": "pass" if scale["e4_scale_observability_passed"] else "fail",
            "medium_heavy_detection": {
                record["variant_id"]: record["frame_detection_rate"]
                for record in scale["medium_heavy"]
            },
            "dose_direction_monotone": scale["both_directions_monotone"],
            "cross_observer_finding": (
                "E5 temporal consistency is blind to absolute scale and improves "
                "for negative scale injection; it must not be used as the scale gate."
            ),
        },
        "pose_translation": {
            "primary_observer": "E5 adjacent temporal reprojection",
            "status": (
                "pass"
                if pose_translation_pass and pose_direction_pass
                else "fail"
            ),
            "medium_heavy_detection": {
                name: medium_heavy[name]["detection_rate"]
                for name in ("pose_translation_05cm", "pose_translation_10cm")
            },
            "dose_direction_monotone": temporal["ordered_direction"][
                "pose_translation"
            ]["at_least_one_primary_metric_monotone"],
        },
        "pose_yaw": {
            "primary_observer": "E5 adjacent temporal reprojection",
            "status": "pass" if pose_yaw_pass and pose_direction_pass else "fail",
            "medium_heavy_detection": {
                name: medium_heavy[name]["detection_rate"]
                for name in ("pose_yaw_1deg", "pose_yaw_3deg")
            },
            "dose_direction_monotone": temporal["ordered_direction"]["pose_yaw"][
                "at_least_one_primary_metric_monotone"
            ],
        },
    }
    not_tested = list(temporal["not_tested"])
    integrated = {
        "schema": "daaam.no_gt_d0_integrated_partial_qualification.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "diagnostic_gt_free_observability_only",
        "tested_fault_family_count": len(tested),
        "protocol_fault_family_count": len(tested) + len(not_tested),
        "tested": tested,
        "all_tested_families_passed": all(
            record["status"] == "pass" for record in tested.values()
        ),
        "not_tested": not_tested,
        "full_d0_status": "unavailable_incomplete",
        "formal_d0_passed": False,
        "reason_formal_d0_not_passed": (
            "Only 3 fault families were tested and no human-reviewed GT was used. "
            "Passing tested families cannot be promoted to full D0 qualification."
        ),
        "important_negative_result": (
            "E5 temporal agreement alone cannot observe absolute metric scale; "
            "negative scale injection increased agreement. E4 sparse LiDAR signed "
            "residual detected all six scale doses on all 83 eligible frames."
        ),
        "inputs": {
            "temporal_qualification": str(temporal_path),
            "temporal_qualification_sha256": sha256_file(temporal_path),
            "lidar_scale_qualification": str(scale_path),
            "lidar_scale_qualification_sha256": sha256_file(scale_path),
        },
    }
    write_json(output / "integrated_partial_qualification.json", integrated)

    report = [
        "# D0 综合资格结论（无人工 GT，部分覆盖）",
        "",
        (
            f"- 已测故障族：`{integrated['tested_fault_family_count']}/"
            f"{integrated['protocol_fault_family_count']}`"
        ),
        (
            f"- 已测三族：`"
            f"{'PASS' if integrated['all_tested_families_passed'] else 'FAIL'}`"
        ),
        "- 完整 D0：`unavailable_incomplete`（不得写为 PASS）",
        "- 权限：`diagnostic_gt_free_observability_only`",
        "",
        "| 故障族 | 主观测器 | 中/重剂量结果 | 剂量方向 |",
        "|---|---|---|---|",
    ]
    for name, record in tested.items():
        rates = ", ".join(
            f"{variant}={rate:.1%}"
            for variant, rate in record["medium_heavy_detection"].items()
        )
        report.append(
            f"| {name} | {record['primary_observer']} | "
            f"{record['status'].upper()} ({rates}) | "
            f"{'monotone' if record['dose_direction_monotone'] else 'non-monotone'} |"
        )
    report.extend(
        [
            "",
            "## 关键负结果",
            "",
            (
                "E5 内部时序一致性不能认证绝对公制尺度：深度统一缩小 5/10/20% 时，"
                "agreement 反而提高。E4 稀疏 LiDAR signed residual 对六个尺度剂量均在 "
                "83/83 帧检出，并且正负方向和剂量均单调。"
            ),
            "",
            "这不是互相矛盾，而是明确了观测职责：E5 检查相对位姿/帧间形状一致性，E4 "
            "负责绝对尺度。任何只用 E5 得出的“尺度更好”结论都应判为无效。",
            "",
            "## 尚未覆盖",
            "",
            *[f"- {item}" for item in not_tested],
            "",
            "## 证据入口",
            "",
            f"- E5 逐对证据：`{d0_root / 'd0_pair_detection.jsonl'}`",
            f"- E4 逐帧证据：`{d0_root / 'lidar_scale_d0/per_frame_lidar_scale_response.jsonl'}`",
            f"- 原生注入与面板：`{d0_root / 'variants'}`",
            "- `final_artifact_inventory.jsonl` 对成功 D0 根中的最终普通文件逐个 SHA-256。",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    provenance = {
        "schema": "daaam.no_gt_d0_integrated_summary_provenance.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(output / "provenance.json", provenance)
    inventory_summary = inventory(d0_root, output)
    write_json(output / "inventory_completion.json", inventory_summary)
    print(
        json.dumps(
            {
                "output": str(output),
                "all_tested_families_passed": integrated[
                    "all_tested_families_passed"
                ],
                "full_d0_status": integrated["full_d0_status"],
                "inventory": inventory_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
