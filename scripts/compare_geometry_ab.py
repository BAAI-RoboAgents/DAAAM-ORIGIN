#!/usr/bin/env python3
"""Compare two mapping runs/DSGs using existing, read-only audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare geometry, semantic binding, runtime, and disk metrics."
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open() as stream:
        value = json.load(stream)
    return value if isinstance(value, dict) else {"value": value}


def _run_root(path: Path) -> Path:
    if path.is_dir():
        return path.resolve()
    resolved = path.resolve()
    for directory in resolved.parents:
        if (directory / "realtime_run_report.json").is_file():
            return directory
    return resolved.parent


def _resolve_dsg(requested: Path, root: Path) -> Path:
    if requested.is_file():
        return requested.resolve()
    candidates = (
        root / "semantic_bound_dsg.json",
        root / "dsg_rebound.json",
        root / "dsg_updated.json",
        root / "hydra_realtime" / "backend" / "dsg_with_mesh.json",
        root / "hydra_realtime" / "backend" / "dsg.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"no DSG found under {root}")


def _ply_counts(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    vertices = faces = 0
    with path.open("rb") as stream:
        for raw_line in stream:
            line = raw_line.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                vertices = int(line.rsplit(" ", 1)[1])
            elif line.startswith("element face "):
                faces = int(line.rsplit(" ", 1)[1])
            elif line == "end_header":
                break
    return {"vertices": vertices, "faces": faces}


def _object_metrics(dsg: dict[str, Any]) -> dict[str, int]:
    objects = [node for node in dsg.get("nodes", []) if node.get("layer") == 2]
    object_meshes = described = embedded = described_meshes = matched = 0
    unique_entities = set()
    for node in objects:
        attributes = node.get("attributes") or {}
        metadata = attributes.get("metadata") or {}
        mesh = attributes.get("mesh") or {}
        has_mesh = bool(mesh.get("points"))
        has_description = bool(str(metadata.get("description") or "").strip())
        has_embedding = bool(metadata.get("sentence_embedding_feature"))
        entity_id = str(metadata.get("entity_id") or "").strip()
        if entity_id:
            unique_entities.add(entity_id)
        object_meshes += int(has_mesh)
        described += int(has_description)
        embedded += int(has_embedding)
        described_meshes += int(has_description and has_mesh)
        matched += int(metadata.get("mesh_binding_status") == "matched_real_mesh")
    return {
        "objects": len(objects),
        "unique_entities": len(unique_entities),
        "object_meshes": object_meshes,
        "described_objects": described,
        "embedded_objects": embedded,
        "described_object_meshes": described_meshes,
        "matched_real_mesh": matched,
    }


def _binding_audit(root: Path, dsg: Path) -> dict[str, Any] | None:
    candidates = [
        dsg.with_suffix(dsg.suffix + ".binding.json"),
        root / "semantic_binding.json",
        root / "dsg_binding.json",
    ]
    candidates.extend(sorted(root.glob("*.binding.json")))
    for candidate in candidates:
        audit = _json(candidate)
        if audit is None:
            continue
        events = audit.get("events") or []
        rejected = sum(
            "rejected_no_mesh" in json.dumps(event, sort_keys=True)
            for event in events
        )
        return {
            "path": str(candidate.resolve()),
            "sha256": _sha256(candidate),
            "verification": audit.get("verification"),
            "event_count": len(events),
            "rejected_no_mesh_events": rejected,
        }
    return None


def _disk_bytes(root: Path) -> tuple[int, int]:
    files = bytes_total = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        files += 1
        bytes_total += path.stat().st_size
    return files, bytes_total


def summarize(requested: Path) -> dict[str, Any]:
    requested = requested.expanduser().resolve()
    if not requested.exists():
        raise FileNotFoundError(requested)
    root = _run_root(requested)
    dsg_path = _resolve_dsg(requested, root)
    dsg = _json(dsg_path)
    if dsg is None:
        raise ValueError(f"DSG is not a JSON object: {dsg_path}")
    report = _json(root / "realtime_run_report.json") or {}
    mesh_quality = (
        _json(root / "mesh_quality.json")
        or report.get("map_metrics")
        or {}
    )
    metrics = _json(root / "realtime_metrics.json") or {}
    benchmark = _json(root / "benchmark_validation.json") or {}
    quality = _json(root / "quality_report.json") or {}
    postpass = _json(root / "hydra_semantic_postpass.json") or {}
    geometry_ab = _json(root / "geometry_ab_run.json") or {}
    ply_path = dsg_path.parent / "mesh.ply"
    if not ply_path.is_file():
        ply_path = root / "hydra_realtime" / "backend" / "mesh.ply"
    files, disk_bytes = _disk_bytes(root)
    main_mesh = dsg.get("mesh") or {}
    binding_audit = _binding_audit(root, dsg_path)
    if binding_audit is None:
        live_binding = (
            (report.get("semantic_stats") or {}).get("dsg") or {}
        )
        if live_binding:
            rejection_audit = live_binding.get("rejection_audit") or []
            binding_audit = {
                "path": str((root / "realtime_run_report.json").resolve()),
                "sha256": _sha256(root / "realtime_run_report.json"),
                "verification": {
                    "matched_real_mesh": live_binding.get("applied"),
                    "rejected_no_mesh": live_binding.get("rejected_no_mesh"),
                    "commit_valid": live_binding.get("commit_valid"),
                },
                "event_count": len(rejection_audit),
                "rejected_no_mesh_events": sum(
                    event.get("status") == "rejected_no_mesh"
                    for event in rejection_audit
                    if isinstance(event, dict)
                ),
            }
    return {
        "requested": str(requested),
        "run_root": str(root),
        "dsg": {
            "path": str(dsg_path),
            "sha256": _sha256(dsg_path),
            "bytes": dsg_path.stat().st_size,
            "main_mesh_vertices": len(main_mesh.get("points") or []),
            "main_mesh_faces": len(main_mesh.get("faces") or []),
        },
        "ply": {
            "path": str(ply_path.resolve()) if ply_path.is_file() else None,
            "counts": _ply_counts(ply_path),
        },
        "mesh_quality": {
            key: mesh_quality.get(key)
            for key in (
                "vertices",
                "triangles",
                "surface_area_m2",
                "connected_components",
                "significant_connected_components",
                "largest_component_area_ratio",
                "tiny_component_area_ratio",
                "isolated_vertices",
                "invalid_faces",
            )
        },
        "objects": _object_metrics(dsg),
        "binding_audit": binding_audit,
        "runtime": {
            "status": report.get("status"),
            "frames_requested": report.get("frames_requested"),
            "frames_completed": report.get("frames_completed"),
            "semantic_startup_seconds": report.get("semantic_startup_seconds"),
            "elapsed_seconds": metrics.get("elapsed_seconds"),
            "semantic_postpass_elapsed_seconds": postpass.get(
                "elapsed_seconds"
            ),
            "geometry_ab_elapsed_seconds": geometry_ab.get(
                "elapsed_seconds"
            ),
            "quality_passed": report.get("quality_passed", quality.get("passed")),
            "benchmark_passed": benchmark.get("passed"),
            "benchmark_authoritative": benchmark.get("authoritative"),
        },
        "geometry_configuration": {
            "path": (
                geometry_ab.get("hydra_config")
                or postpass.get("hydra_config_path")
                or (report.get("configuration") or {}).get("hydra_config_path")
            ),
            "sha256": (
                geometry_ab.get("hydra_config_sha256")
                or postpass.get("hydra_config_sha256")
            ),
        },
        "disk": {"files": files, "bytes": disk_bytes},
    }


def _numeric_delta(baseline: Any, candidate: Any) -> Any:
    if (
        isinstance(baseline, (int, float))
        and not isinstance(baseline, bool)
        and isinstance(candidate, (int, float))
        and not isinstance(candidate, bool)
    ):
        return candidate - baseline
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        values = {
            key: _numeric_delta(baseline[key], candidate[key])
            for key in baseline.keys() & candidate.keys()
        }
        return {key: value for key, value in values.items() if value is not None}
    return None


def main() -> int:
    args = parse_args()
    try:
        baseline = summarize(args.baseline)
        candidate = summarize(args.candidate)
        result = {
            "schema": "daaam.geometry_ab.v1",
            "baseline": baseline,
            "candidate": candidate,
            "candidate_minus_baseline": _numeric_delta(baseline, candidate),
        }
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered)
        print(rendered, end="")
        return 0
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
