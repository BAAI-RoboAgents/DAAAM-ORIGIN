"""Authoritative paired 5/10 Hz realtime benchmark validation."""

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


def validate_realtime_run(
    run_dir: Path | str,
    *,
    expected_rate_hz: float,
    require_dam: bool = True,
    allow_dirty: bool = False,
) -> dict[str, Any]:
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
    check("run.zero_drops", not report.get("dropped_frames"), report.get("dropped_frames"))
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
            and int(dsg.get("applied", 0)) > 0
            and int(dsg.get("pending", 0)) == 0
            and int(dsg.get("unmapped", 0)) == 0
            and not dsg.get("errors"),
            dsg,
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
    return {
        "run_dir": str(root),
        "expected_rate_hz": expected_rate_hz,
        "passed": not blocking_failures,
        "authoritative": not blocking_failures and clean,
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
        runs[0]["dataset_tick_index_sha256"]
        == runs[1]["dataset_tick_index_sha256"]
    )
    pair_checks = {
        "same_clean_commit": same_commit,
        "same_dataset": same_dataset,
    }
    passed = all(run["passed"] for run in runs) and all(pair_checks.values())
    authoritative = (
        passed
        and all(run["authoritative"] for run in runs)
        and not allow_dirty
        and require_dam
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "authoritative": authoritative,
        "require_dam": require_dam,
        "allow_dirty": allow_dirty,
        "pair_checks": pair_checks,
        "runs": runs,
    }
