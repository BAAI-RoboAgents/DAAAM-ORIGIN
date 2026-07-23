#!/usr/bin/env python3
"""Rebuild Hydra geometry with a second config from a completed semantic run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POSTPASS_PLAN = Path("semantic_sidecar/hydra_semantic_postpass_plan.json")
POSTPASS_REPORT = Path("hydra_semantic_postpass.json")
CHILD_REPORT = Path("semantic_sidecar/hydra_semantic_postpass_child.json")
STDOUT_LOG = Path("semantic_sidecar/hydra_semantic_postpass.stdout.log")
STDERR_LOG = Path("semantic_sidecar/hydra_semantic_postpass.stderr.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an existing complete semantic-label journal through a second "
            "Hydra geometry configuration without modifying the source run."
        )
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--hydra-config", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=7_200.0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _validate_complete_report(
    report: dict[str, Any],
    *,
    expected_frames: int,
    label: str,
) -> None:
    checks = {
        "status": report.get("status") == "complete",
        "frames_expected": int(report.get("frames_expected", -1))
        == expected_frames,
        "frames_replayed": int(report.get("frames_replayed", -1))
        == expected_frames,
        "frames_with_labels": int(report.get("frames_with_labels", -1))
        == expected_frames,
        "label_coverage": float(report.get("label_coverage", 0.0)) == 1.0,
        "missing_frame_indices": not report.get("missing_frame_indices"),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(
            f"{label} does not satisfy the complete semantic postpass contract: "
            + ", ".join(failures)
        )


def _validate_source(
    source_run: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    plan_path = source_run / POSTPASS_PLAN
    report_path = source_run / POSTPASS_REPORT
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    plan = _read_json(plan_path)
    report = _read_json(report_path)
    if plan.get("schema") != "daaam.hydra_semantic_postpass_plan.v1":
        raise ValueError("source run has an unsupported semantic postpass plan")
    frames = plan.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("source semantic postpass plan contains no frames")
    _validate_complete_report(
        report,
        expected_frames=len(frames),
        label="source run",
    )

    planned_run_dir = Path(str(plan.get("run_dir", ""))).resolve()
    expected_label_dir = (
        source_run / "semantic_sidecar" / "label_frames"
    ).resolve()
    planned_label_dir = Path(str(plan.get("semantic_label_dir", ""))).resolve()
    if planned_run_dir != source_run or planned_label_dir != expected_label_dir:
        raise ValueError(
            "source postpass plan does not point at the selected source run"
        )
    if not expected_label_dir.is_dir():
        raise FileNotFoundError(expected_label_dir)
    static_depth_dir = source_run / "static_depth"
    if not static_depth_dir.is_dir():
        raise FileNotFoundError(static_depth_dir)

    frame_indices: set[int] = set()
    for record in frames:
        if not isinstance(record, dict):
            raise ValueError("source postpass frame records must be objects")
        frame_index = int(record.get("frame_index", -1))
        if frame_index < 0 or frame_index in frame_indices:
            raise ValueError("source postpass frame indices must be unique")
        frame_indices.add(frame_index)
        rgb_path = Path(str(record.get("rgb_path", "")))
        label_path = expected_label_dir / f"{frame_index:08d}.png"
        static_path = static_depth_dir / f"{frame_index:08d}.png"
        for required in (rgb_path, label_path, static_path):
            if not required.is_file():
                raise FileNotFoundError(required)
    return plan, report, plan_path, report_path


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def run_geometry_ab_postpass(
    *,
    source_run: Path,
    hydra_config: Path,
    output_run: Path,
    overwrite: bool = False,
    timeout_s: float = 7_200.0,
) -> dict[str, Any]:
    """Execute the isolated B reconstruction and return its completion record."""

    source_run = source_run.resolve()
    hydra_config = hydra_config.resolve()
    output_run = output_run.resolve()
    if timeout_s <= 0.0:
        raise ValueError("geometry A/B timeout must be positive")
    if not source_run.is_dir():
        raise FileNotFoundError(source_run)
    if not hydra_config.is_file():
        raise FileNotFoundError(hydra_config)
    if _paths_overlap(source_run, output_run):
        raise ValueError("source and output runs must be disjoint directories")

    source_plan, source_report, source_plan_path, source_report_path = (
        _validate_source(source_run)
    )
    if output_run.exists():
        if not overwrite:
            raise FileExistsError(output_run)
        shutil.rmtree(output_run)

    started_ns = time.time_ns()
    started = time.monotonic()
    try:
        output_run.mkdir(parents=True)
        output_plan = copy.deepcopy(source_plan)
        output_plan.update(
            {
                "run_dir": str(source_run),
                "semantic_label_dir": str(
                    source_run / "semantic_sidecar" / "label_frames"
                ),
                "output_dir": str(output_run / "hydra_realtime"),
                "hydra_config_path": str(hydra_config),
                "geometry_ab_source_run": str(source_run),
            }
        )
        output_plan_path = output_run / POSTPASS_PLAN
        child_report_path = output_run / CHILD_REPORT
        stdout_path = output_run / STDOUT_LOG
        stderr_path = output_run / STDERR_LOG
        _write_json_atomic(output_plan_path, output_plan)

        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_hydra_semantic_postpass.py"),
            "--plan",
            str(output_plan_path),
            "--report",
            str(child_report_path),
        ]
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            result = subprocess.run(
                command,
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=timeout_s,
            )
        if result.returncode != 0:
            error_tail = stderr_path.read_text(errors="replace")[-4_000:]
            raise RuntimeError(
                "isolated Hydra geometry B postpass failed with exit code "
                f"{result.returncode}: {error_tail}"
            )
        if not child_report_path.is_file():
            raise RuntimeError("geometry B postpass produced no child report")
        child_report = _read_json(child_report_path)
        expected_frames = len(output_plan["frames"])
        _validate_complete_report(
            child_report,
            expected_frames=expected_frames,
            label="geometry B child",
        )
        backend_dir = output_run / "hydra_realtime" / "backend"
        required_outputs = {
            "mesh": backend_dir / "mesh.ply",
            "dsg": backend_dir / "dsg.json",
        }
        for required in required_outputs.values():
            if not required.is_file():
                raise FileNotFoundError(required)

        canonical_report = {
            **child_report,
            "output_dir": str(output_run / "hydra_realtime"),
            "geometry_ab_source_run": str(source_run),
            "geometry_ab_source_label_manifest_sha256": source_report.get(
                "label_manifest_sha256"
            ),
            "hydra_config_path": str(hydra_config),
            "hydra_config_sha256": _sha256(hydra_config),
        }
        canonical_report_path = output_run / POSTPASS_REPORT
        _write_json_atomic(canonical_report_path, canonical_report)
        completed_ns = time.time_ns()
        geometry_report = {
            "schema": "daaam.geometry_ab_postpass_run.v1",
            "status": "complete",
            "source_run": str(source_run),
            "output_run": str(output_run),
            "source_plan": str(source_plan_path),
            "source_plan_sha256": _sha256(source_plan_path),
            "source_postpass_report": str(source_report_path),
            "source_postpass_report_sha256": _sha256(source_report_path),
            "output_plan": str(output_plan_path),
            "output_plan_sha256": _sha256(output_plan_path),
            "output_postpass_report": str(canonical_report_path),
            "output_postpass_report_sha256": _sha256(canonical_report_path),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "hydra_config": str(hydra_config),
            "hydra_config_sha256": _sha256(hydra_config),
            "timeout_s": timeout_s,
            "child_exit_code": result.returncode,
            "frames_replayed": expected_frames,
            "label_coverage": float(child_report["label_coverage"]),
            "label_manifest_sha256": child_report.get(
                "label_manifest_sha256"
            ),
            "artifacts": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in required_outputs.items()
            },
            "started_ns": started_ns,
            "completed_ns": completed_ns,
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_json_atomic(output_run / "geometry_ab_run.json", geometry_report)
        return geometry_report
    except BaseException:
        shutil.rmtree(output_run, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    report = run_geometry_ab_postpass(
        source_run=args.source_run,
        hydra_config=args.hydra_config,
        output_run=args.output_run,
        overwrite=args.overwrite,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
