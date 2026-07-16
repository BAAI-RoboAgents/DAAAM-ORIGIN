"""Authoritative realtime benchmark validation with optional stress pairing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid benchmark JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark JSON must contain an object: {path}")
    return value


def _semantic_object_bindings(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = document.get("nodes")
    edges = document.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("DSG nodes and edges must be arrays")
    places = {
        int(node["id"])
        for node in nodes
        if int(node.get("layer", -1)) == 3
        and int(node.get("partition", 0)) == 0
    }
    parented_objects = {
        int(edge["target"])
        for edge in edges
        if int(edge.get("source", -1)) in places
    }
    bindings: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if int(node.get("layer", -1)) != 2 or int(node.get("partition", 0)) != 0:
            continue
        attributes = node.get("attributes")
        if not isinstance(attributes, dict):
            raise ValueError("DSG Object attributes must be an object")
        metadata = attributes.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("DSG Object metadata must be an object")
        entity_id = str(metadata.get("entity_id", "")).strip()
        description = " ".join(str(metadata.get("description", "")).split())
        if not entity_id and not description:
            continue
        if not entity_id or not description:
            raise ValueError("semantic DSG Object binding is incomplete")
        if entity_id in bindings:
            raise ValueError(f"semantic DSG entity is duplicated: {entity_id}")
        node_id = int(node["id"])
        bindings[entity_id] = {
            "node_id": node_id,
            "semantic_label": int(attributes["semantic_label"]),
            "description": description,
            "is_active": bool(attributes.get("is_active", False)),
            "has_place_parent": not places or node_id in parented_objects,
        }
    return bindings


def _validate_semantic_dsg_commit(
    root: Path,
    dsg_stats: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    backend = (root / "hydra_realtime" / "backend").resolve()
    expected_manifest = backend / "semantic_dsg_commit.json"
    detail: dict[str, Any] = {
        "manifest": str(expected_manifest),
        "reported_manifest": dsg_stats.get("commit_manifest_path"),
    }
    try:
        reported_manifest = Path(str(dsg_stats["commit_manifest_path"])).resolve()
        if reported_manifest != expected_manifest:
            raise ValueError("semantic DSG manifest is outside the canonical backend")
        manifest = _read_json(expected_manifest)
        manifest_sha256 = _sha256(expected_manifest)
        detail["manifest_sha256"] = manifest_sha256
        if manifest.get("schema") != "daaam.semantic_dsg_commit.v1":
            raise ValueError("semantic DSG manifest schema is unsupported")
        if manifest_sha256 != dsg_stats.get("commit_manifest_sha256"):
            raise ValueError("semantic DSG manifest hash disagrees with run report")

        records = manifest.get("artifacts")
        expected_names = {"dsg.json", "dsg_with_mesh.json"}
        if not isinstance(records, dict) or set(records) != expected_names:
            raise ValueError("semantic DSG manifest must commit both DSG artifacts")
        reported_artifacts = {
            str(Path(str(path)).resolve())
            for path in dsg_stats.get("verified_artifacts", [])
        }
        expected_artifacts = {str((backend / name).resolve()) for name in expected_names}
        if reported_artifacts != expected_artifacts:
            raise ValueError("verified DSG artifact paths are incomplete")

        documents: dict[str, dict[str, Any]] = {}
        bindings: dict[str, dict[str, dict[str, Any]]] = {}
        artifact_detail: dict[str, Any] = {}
        for name in sorted(expected_names):
            path = backend / name
            record = records[name]
            if not isinstance(record, dict):
                raise ValueError(f"semantic DSG record is invalid: {name}")
            actual_sha256 = _sha256(path)
            if actual_sha256 != record.get("sha256"):
                raise ValueError(f"semantic DSG artifact hash changed: {name}")
            document = _read_json(path)
            documents[name] = document
            bindings[name] = _semantic_object_bindings(document)
            object_count = sum(
                int(node.get("layer", -1)) == 2
                and int(node.get("partition", 0)) == 0
                for node in document.get("nodes", [])
            )
            if object_count != int(record.get("object_count", -1)):
                raise ValueError(f"semantic DSG Object count changed: {name}")
            mesh = document.get("mesh")
            has_mesh = isinstance(mesh, dict)
            if has_mesh != bool(record.get("has_mesh")):
                raise ValueError(f"semantic DSG mesh presence changed: {name}")
            if name == "dsg.json" and has_mesh:
                raise ValueError("plain DSG unexpectedly contains a mesh")
            if name == "dsg_with_mesh.json" and not has_mesh:
                raise ValueError("mesh DSG no longer contains a mesh")
            mesh_vertices = len(mesh.get("points", [])) if has_mesh else 0
            mesh_faces = len(mesh.get("faces", [])) if has_mesh else 0
            if mesh_vertices != int(record.get("mesh_vertices", -1)):
                raise ValueError(f"semantic DSG mesh vertex count changed: {name}")
            if mesh_faces != int(record.get("mesh_faces", -1)):
                raise ValueError(f"semantic DSG mesh face count changed: {name}")
            artifact_detail[name] = {
                "sha256": actual_sha256,
                "object_count": object_count,
                "semantic_entities": len(bindings[name]),
                "mesh_vertices": mesh_vertices,
                "mesh_faces": mesh_faces,
            }

        if bindings["dsg.json"] != bindings["dsg_with_mesh.json"]:
            raise ValueError("plain and mesh DSG semantic bindings disagree")
        committed_bindings = bindings["dsg.json"]
        if not committed_bindings:
            raise ValueError("semantic DSG contains no durable entity bindings")
        if not all(
            binding["is_active"] and binding["has_place_parent"]
            for binding in committed_bindings.values()
        ):
            raise ValueError("semantic DSG contains inactive or orphaned bindings")
        verified_entities = int(manifest.get("verified_entity_count", -1))
        verified_operations = int(manifest.get("verified_operation_count", -1))
        if verified_entities != int(dsg_stats.get("verified_entities", -2)):
            raise ValueError("verified semantic entity count disagrees with report")
        if verified_entities > len(committed_bindings) or verified_entities <= 0:
            raise ValueError("verified semantic entity count is invalid")
        if verified_operations != int(dsg_stats.get("verified_operations", -2)):
            raise ValueError("verified semantic operation count disagrees with report")
        if verified_operations != int(dsg_stats.get("applied", -3)):
            raise ValueError("durable semantic operation count disagrees with ACKs")
        if verified_operations <= 0:
            raise ValueError("semantic DSG committed no operations")

        detail.update(
            {
                "artifacts": artifact_detail,
                "verified_entities": verified_entities,
                "verified_operations": verified_operations,
            }
        )
        return True, detail
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        detail["error"] = str(error)
        return False, detail


def validate_realtime_run(
    run_dir: Path | str,
    *,
    expected_rate_hz: float,
    require_dam: bool = True,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    if expected_rate_hz <= 0.0:
        raise ValueError("expected_rate_hz must be positive")
    root = Path(run_dir).resolve()
    paths = {
        "manifest": root / "run_manifest.json",
        "metrics": root / "realtime_metrics.json",
        "quality": root / "quality_report.json",
        "report": root / "realtime_run_report.json",
    }
    values = {name: _read_json(path) for name, path in paths.items()}
    manifest = values["manifest"]
    metrics = values["metrics"]
    quality = values["quality"]
    report = values["report"]
    configuration = manifest.get("configuration", {})
    repository = manifest.get("repository", {})
    time_contract = manifest.get("time_contract", {})
    semantic = report.get("semantic_stats") or {}
    stages = metrics.get("stages", {})
    foundation = manifest.get("models", {}).get("foundation_stereo", {})
    semantic_models = manifest.get("models", {}).get("semantic_frontend") or {}

    checks: list[dict[str, Any]] = []

    def check(code: str, passed: bool, detail: Any = None, *, authority=True) -> None:
        checks.append(
            {
                "code": code,
                "passed": bool(passed),
                "blocks_authority": bool(authority),
                "detail": detail,
            }
        )

    clean = repository.get("git_dirty") is False
    check(
        "provenance.clean_head",
        clean or allow_dirty,
        {"git_sha": repository.get("git_sha"), "git_dirty": not clean},
    )
    check("provenance.git_sha", bool(repository.get("git_sha")))
    check(
        "configuration.rate",
        abs(float(configuration.get("rate_hz", -1.0)) - expected_rate_hz) <= 1e-9,
        configuration.get("rate_hz"),
    )
    replay_pacing = report.get("replay_pacing") or {}
    check(
        "configuration.replay_pacing_rate",
        abs(float(replay_pacing.get("configured_max_rate_hz", -1.0)) - expected_rate_hz)
        <= 1e-9,
        replay_pacing.get("configured_max_rate_hz"),
    )
    check(
        "configuration.stage_rate_multiplier",
        abs(float(configuration.get("stage_rate_multiplier", -1.0)) - 1.0) <= 1e-9,
        configuration.get("stage_rate_multiplier"),
    )
    check(
        "configuration.throttled",
        configuration.get("no_throttle") is False,
        configuration.get("no_throttle"),
    )
    check(
        "configuration.no_source_bursts",
        configuration.get("allow_source_bursts") is False,
        configuration.get("allow_source_bursts"),
    )
    check(
        "configuration.full_dataset",
        configuration.get("max_frames") is None,
        configuration.get("max_frames"),
    )
    fault = configuration.get("fault") or {}
    check(
        "configuration.no_fault_injection",
        fault.get("stage") is None
        and float(fault.get("delay_ms", 0.0)) == 0.0
        and int(fault.get("error_every", 0)) == 0,
        fault,
    )
    expected_quality_config = (
        Path(__file__).resolve().parents[3] / "config" / "realtime_quality_gates.yaml"
    )
    check(
        "configuration.standard_quality_gates",
        expected_quality_config.is_file()
        and configuration.get("quality_config_sha256")
        == _sha256(expected_quality_config),
        {
            "reported": configuration.get("quality_config_sha256"),
            "expected": (
                _sha256(expected_quality_config)
                if expected_quality_config.is_file()
                else None
            ),
        },
    )
    check("run.complete", report.get("status") == "complete", report.get("status"))
    check(
        "run.all_frames_completed",
        int(report.get("frames_completed", -1))
        == int(report.get("frames_requested", -2)),
        {
            "requested": report.get("frames_requested"),
            "completed": report.get("frames_completed"),
        },
    )
    requested_frames = int(report.get("frames_requested", -1))
    expected_frames = int(time_contract.get("frame_count", -2))
    check(
        "run.full_dataset",
        requested_frames == expected_frames,
        {"requested": requested_frames, "dataset_frames": expected_frames},
    )
    check(
        "run.all_frames_dispatched",
        int(report.get("frames_dispatched", -1)) == requested_frames,
        report.get("frames_dispatched"),
    )
    check(
        "run.no_resume",
        int(report.get("frames_resumed_from", -1)) == 0,
        report.get("frames_resumed_from"),
    )
    check(
        "run.zero_drops", not report.get("dropped_frames"), report.get("dropped_frames")
    )
    check(
        "quality.all_hard_gates",
        quality.get("passed") is True
        and int(quality.get("hard_failures", -1)) == 0
        and report.get("quality_passed") is True,
        quality.get("hard_failures"),
    )
    check(
        "runtime.zero_errors",
        int(metrics.get("totals", {}).get("errors", -1)) == 0,
        metrics.get("totals", {}).get("errors"),
    )
    check(
        "runtime.zero_scheduler_drops",
        int(metrics.get("totals", {}).get("dropped", -1)) == 0,
        metrics.get("totals", {}).get("dropped"),
    )
    check(
        "runtime.real_segmentation",
        int(stages.get("segmentation", {}).get("processed", 0)) > 0
        and stages.get("segmentation", {})
        .get("latency", {})
        .get("service_ms", {})
        .get("p95")
        is not None,
    )
    check(
        "runtime.real_tracking",
        int(stages.get("tracking", {}).get("processed", 0)) > 0
        and stages.get("tracking", {})
        .get("latency", {})
        .get("service_ms", {})
        .get("p95")
        is not None,
    )
    segmentation_calls = int(semantic.get("segmentation_calls", -1))
    tracking_calls = int(semantic.get("tracking_calls", -1))
    check(
        "runtime.semantic_metric_consistency",
        int(stages.get("segmentation", {}).get("processed", -2)) == segmentation_calls
        and int(stages.get("tracking", {}).get("processed", -2)) == tracking_calls,
        {
            "metrics_segmentation": stages.get("segmentation", {}).get("processed"),
            "reported_segmentation": segmentation_calls,
            "metrics_tracking": stages.get("tracking", {}).get("processed"),
            "reported_tracking": tracking_calls,
        },
    )
    check(
        "runtime.full_rate_tracking",
        tracking_calls == requested_frames,
        {"tracking_calls": tracking_calls, "frames_requested": requested_frames},
    )
    mapping_cycle_p95_ms = (
        stages.get("global", {}).get("latency", {}).get("end_to_end_ms", {}).get("p95")
    )
    mapping_cycle_limit_ms = 1000.0 / expected_rate_hz
    check(
        "runtime.mapping_cycle_p95",
        mapping_cycle_p95_ms is not None
        and float(mapping_cycle_p95_ms) <= mapping_cycle_limit_ms,
        {
            "p95_ms": mapping_cycle_p95_ms,
            "limit_ms": mapping_cycle_limit_ms,
            "definition": "pose dispatch through committed global map update",
        },
    )
    replay_sleep_s = float(replay_pacing.get("sleep_seconds", -1.0))
    minimum_replay_sleep_s = max(0, requested_frames - 1) / expected_rate_hz
    metrics_elapsed_s = float(metrics.get("elapsed_seconds", -1.0))
    check(
        "runtime.replay_pacing_evidence",
        replay_pacing.get("source_bursts_allowed") is False
        and replay_pacing.get("absolute_timestamps_preserved") is True
        and replay_sleep_s + 1.0e-6 >= minimum_replay_sleep_s
        and metrics_elapsed_s + 1.0e-6 >= replay_sleep_s,
        {
            "sleep_seconds": replay_sleep_s,
            "minimum_sleep_seconds": minimum_replay_sleep_s,
            "metrics_elapsed_seconds": metrics_elapsed_s,
            "source_bursts_allowed": replay_pacing.get("source_bursts_allowed"),
            "absolute_timestamps_preserved": replay_pacing.get(
                "absolute_timestamps_preserved"
            ),
        },
    )
    if require_dam:
        check("semantic.mode_dam", report.get("semantic_mode") == "dam")
        check(
            "semantic.real_corrections",
            int(semantic.get("prompts_submitted", 0)) > 0
            and int(semantic.get("corrections_submitted", 0)) > 0,
            {
                "prompts": semantic.get("prompts_submitted"),
                "corrections": semantic.get("corrections_submitted"),
            },
        )
        dsg = semantic.get("dsg", {})
        check(
            "semantic.hydra_dsg_ack",
            bool(dsg.get("graph_attached"))
            and bool(dsg.get("commit_valid"))
            and int(dsg.get("applied", 0)) > 0
            and int(dsg.get("pending", 0)) == 0
            and int(dsg.get("unmapped", 0)) == 0
            and not dsg.get("errors"),
            dsg,
        )
        dsg_commit_valid, dsg_commit_detail = _validate_semantic_dsg_commit(root, dsg)
        check(
            "semantic.final_dsg_artifacts",
            dsg_commit_valid,
            dsg_commit_detail,
        )
        check(
            "semantic.resolved_models",
            bool(semantic_models.get("fastsam", {}).get("sha256"))
            and bool(semantic_models.get("botsort_reid", {}).get("sha256"))
            and bool(semantic_models.get("dam", {}).get("cached_revision"))
            and bool(semantic_models.get("semantic_labelspace", {}).get("sha256"))
            and bool(semantic_models.get("labelspace_colors", {}).get("sha256")),
            semantic_models,
        )
    check(
        "depth.resolved_profile",
        foundation.get("valid_iters") is not None
        and foundation.get("scale") is not None
        and foundation.get("precision") is not None
        and bool(foundation.get("checkpoint_sha256")),
        {
            key: foundation.get(key)
            for key in ("valid_iters", "scale", "precision", "checkpoint_sha256")
        },
    )

    blocking_failures = [
        item["code"]
        for item in checks
        if item["blocks_authority"] and not item["passed"]
    ]
    authoritative = not blocking_failures and clean and require_dam and not allow_dirty
    return {
        "run_dir": str(root),
        "expected_rate_hz": expected_rate_hz,
        "passed": not blocking_failures,
        "authoritative": authoritative,
        "require_dam": require_dam,
        "allow_dirty": allow_dirty,
        "blocking_failures": blocking_failures,
        "git_sha": repository.get("git_sha"),
        "dataset_tick_index_sha256": manifest.get("dataset", {}).get(
            "tick_index_sha256"
        ),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "checks": checks,
    }


def validate_benchmark_pair(
    run_5hz: Path | str,
    run_10hz: Path | str,
    *,
    require_dam: bool = True,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Validate the legacy 5/10 Hz pair as an optional stress benchmark."""
    runs = [
        validate_realtime_run(
            run_5hz,
            expected_rate_hz=5.0,
            require_dam=require_dam,
            allow_dirty=allow_dirty,
        ),
        validate_realtime_run(
            run_10hz,
            expected_rate_hz=10.0,
            require_dam=require_dam,
            allow_dirty=allow_dirty,
        ),
    ]
    same_commit = bool(runs[0]["git_sha"]) and runs[0]["git_sha"] == runs[1]["git_sha"]
    same_dataset = bool(runs[0]["dataset_tick_index_sha256"]) and (
        runs[0]["dataset_tick_index_sha256"] == runs[1]["dataset_tick_index_sha256"]
    )
    pair_checks = {
        "same_clean_commit": same_commit,
        "same_dataset": same_dataset,
    }
    passed = all(run["passed"] for run in runs) and all(pair_checks.values())
    # The pair remains useful for overload characterization, but the product
    # acceptance target is the clean 1 Hz single-run validator above.
    authoritative = False
    return {
        "schema_version": 1,
        "passed": passed,
        "authoritative": authoritative,
        "require_dam": require_dam,
        "allow_dirty": allow_dirty,
        "pair_checks": pair_checks,
        "runs": runs,
    }
