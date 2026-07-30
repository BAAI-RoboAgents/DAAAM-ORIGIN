#!/usr/bin/env python3
"""Restore one missing exact-frame label from a deterministic prefix replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.realtime.semantic_labels import persist_semantic_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-run", required=True, type=Path)
    parser.add_argument("--repair-run", required=True, type=Path)
    parser.add_argument("--frame-index", required=True, type=int)
    parser.add_argument(
        "--label-map",
        action="append",
        default=[],
        help="Translate one repair-run semantic ID as SOURCE:TARGET.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    target_run = args.target_run.expanduser().resolve()
    repair_run = args.repair_run.expanduser().resolve()
    frame_index = args.frame_index
    filename = f"{frame_index:08d}"
    target_dir = target_run / "semantic_sidecar" / "label_frames"
    repair_dir = repair_run / "semantic_sidecar" / "label_frames"
    target_image = target_dir / f"{filename}.png"
    target_metadata = target_dir / f"{filename}.json"
    if target_image.exists() or target_metadata.exists():
        raise FileExistsError(
            f"Refusing to replace an existing target binding: {target_image}"
        )

    target_manifest_path = target_run / "run_manifest.json"
    repair_manifest_path = repair_run / "run_manifest.json"
    target_manifest = json.loads(target_manifest_path.read_text())
    repair_manifest = json.loads(repair_manifest_path.read_text())
    target_fingerprint = target_manifest["dataset"]["tick_index_sha256"]
    repair_fingerprint = repair_manifest["dataset"]["tick_index_sha256"]
    if target_fingerprint != repair_fingerprint:
        raise ValueError("Target and prefix replay use different datasets")

    repair_image = repair_dir / f"{filename}.png"
    repair_metadata_path = repair_dir / f"{filename}.json"
    repair_metadata = json.loads(repair_metadata_path.read_text())
    labels = cv2.imread(str(repair_image), cv2.IMREAD_UNCHANGED)
    if labels is None or labels.dtype != np.uint16:
        raise ValueError(f"Invalid repair label image: {repair_image}")
    if int(repair_metadata["frame_index"]) != frame_index:
        raise ValueError("Repair label metadata has the wrong frame index")

    mappings: dict[int, int] = {}
    for item in args.label_map:
        source_text, separator, target_text = item.partition(":")
        if not separator:
            raise ValueError(f"Invalid --label-map value: {item}")
        source = int(source_text)
        target = int(target_text)
        if not 0 < source <= np.iinfo(np.uint16).max:
            raise ValueError(f"Invalid source semantic ID: {source}")
        if not 0 < target <= np.iinfo(np.uint16).max:
            raise ValueError(f"Invalid target semantic ID: {target}")
        mappings[source] = target
    translated = labels.copy()
    for source, target in mappings.items():
        translated[labels == source] = target

    run_configuration_sha256 = target_manifest["configuration"]["semantic"][
        "label_run_configuration_sha256"
    ]
    output_record = persist_semantic_label(
        target_dir,
        frame_index,
        translated,
        sensor_time_ns=int(repair_metadata["sensor_time_ns"]),
        run_configuration_sha256=run_configuration_sha256,
    )
    provenance = {
        "schema": "daaam.semantic_label_repair.v1",
        "method": "deterministic_prefix_replay_with_audited_id_translation",
        "target_run": str(target_run),
        "repair_run": str(repair_run),
        "frame_index": frame_index,
        "sensor_time_ns": int(repair_metadata["sensor_time_ns"]),
        "dataset_tick_index_sha256": target_fingerprint,
        "repair_image": str(repair_image),
        "repair_image_sha256": sha256_file(repair_image),
        "repair_metadata": str(repair_metadata_path),
        "repair_metadata_sha256": sha256_file(repair_metadata_path),
        "label_map": {str(key): value for key, value in sorted(mappings.items())},
        "output_record": output_record,
        "target_run_configuration_sha256": run_configuration_sha256,
        "target_manifest_sha256": sha256_file(target_manifest_path),
        "repair_manifest_sha256": sha256_file(repair_manifest_path),
    }
    provenance_path = (
        target_run / "semantic_sidecar" / f"label_repair_{filename}.json"
    )
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
