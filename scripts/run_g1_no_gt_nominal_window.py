#!/usr/bin/env python3
"""Run one zero-resume G1 semantic-map window with maximum diagnostic retention."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REALTIME = REPOSITORY_ROOT / "scripts/run_realtime_mapping.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sample-interval-s", type=float, default=5.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_proc_status(pid: int) -> dict[str, Any]:
    path = Path(f"/proc/{pid}/status")
    if not path.is_file():
        return {}
    values: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {"VmRSS", "VmHWM", "VmSize", "Threads"}:
            values[key] = value.strip()
    return values


def nvidia_sample() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,temperature.gpu,utilization.gpu,"
        "utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def key_artifacts(run_dir: Path) -> dict[str, Any]:
    names = (
        "run_manifest.json",
        "realtime_run_report.json",
        "realtime_metrics.json",
        "quality_report.json",
        "quality_context.json",
        "realtime_checkpoint.json",
        "hydra_semantic_postpass.json",
        "map_memory.sqlite3",
        "progress.sqlite3",
    )
    output: dict[str, Any] = {}
    for name in names:
        path = run_dir / name
        if path.is_file():
            output[name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    for name, path in (
        ("hydra_mesh", run_dir / "hydra_realtime/backend/mesh.ply"),
        ("hydra_dsg", run_dir / "hydra_realtime/backend/dsg.json"),
        (
            "hydra_dsg_with_mesh",
            run_dir / "hydra_realtime/backend/dsg_with_mesh.json",
        ),
    ):
        if path.is_file():
            output[name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return output


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / "semantic_run"
    if run_dir.exists():
        raise FileExistsError(
            f"Zero-resume run directory already exists; refusing reuse: {run_dir}"
        )
    if args.sample_interval_s <= 0:
        raise ValueError("--sample-interval-s must be positive")

    command = [
        sys.executable,
        str(REALTIME),
        "--dataset",
        str(dataset),
        "--run-dir",
        str(run_dir),
        "--stop-after",
        "global",
        "--depth-backend",
        "precomputed",
        "--maximum-depth-m",
        "65.535",
        "--hydra-maximum-range-m",
        "8.0",
        "--rate-hz",
        "1.0",
        "--queue-capacity",
        "8",
        "--static-map-backend",
        "hydra",
        "--hydra-config-path",
        str(REPOSITORY_ROOT / "config/hydra_g1_8m_12cm.yaml"),
        "--hydra-labelspace-path",
        str(REPOSITORY_ROOT / "config/labels_pseudo.yaml"),
        "--hydra-labelspace-colors",
        str(REPOSITORY_ROOT / "config/labels_pseudo.csv"),
        "--semantic-mode",
        "dam",
        "--semantic-config",
        str(REPOSITORY_ROOT / "config/pipeline_config_realtime.yaml"),
        "--segmentation-rate-hz",
        "5.0",
        "--semantic-frontend-rate-hz",
        "10.0",
        "--semantic-queue-capacity",
        "256",
        "--semantic-minimum-observations",
        "5",
        "--semantic-startup-timeout-s",
        "180",
        "--semantic-drain-timeout-s",
        "1200",
        "--gpu-sharing-mode",
        "staggered",
        "--quality-config",
        str(REPOSITORY_ROOT / "config/realtime_quality_gates.yaml"),
        "--experiment-telemetry",
        "--telemetry-sample-interval-s",
        "0.5",
        "--experiment-visualizations",
        "--visualization-stride",
        "1",
    ]
    invocation = {
        "schema": "daaam.no_gt_nominal_window_invocation.v1",
        "authority": "diagnostic_gt_free",
        "zero_resume": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_tick_index_sha256": sha256_file(dataset / "tick_index.json"),
        "dataset_window_manifest_sha256": (
            sha256_file(dataset / "window_manifest.json")
            if (dataset / "window_manifest.json").is_file()
            else None
        ),
        "run_dir": str(run_dir),
        "argv": command,
        "wrapper_argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "script_sha256": sha256_file(Path(__file__)),
        "realtime_script_sha256": sha256_file(REALTIME),
    }
    write_json(output_root / "invocation.json", invocation)

    stdout_path = output_root / "semantic_run.stdout.log"
    stderr_path = output_root / "semantic_run.stderr.log"
    samples_path = output_root / "wrapper_resource_samples.jsonl"
    started_monotonic = time.monotonic()
    started_utc = datetime.now(timezone.utc)
    with stdout_path.open("w", encoding="utf-8") as stdout_stream, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_stream, samples_path.open("w", encoding="utf-8") as sample_stream:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
            env=os.environ.copy(),
        )
        sequence = 0
        while True:
            returncode = process.poll()
            sample = {
                "sequence": sequence,
                "utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": time.monotonic() - started_monotonic,
                "pid": process.pid,
                "process_returncode": returncode,
                "process_status": read_proc_status(process.pid),
                "load_average": os.getloadavg(),
                "nvidia_smi": nvidia_sample(),
            }
            sample_stream.write(
                json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n"
            )
            sample_stream.flush()
            print(
                json.dumps(
                    {
                        "elapsed_s": round(sample["elapsed_s"], 1),
                        "pid": process.pid,
                        "returncode": returncode,
                        "rss": sample["process_status"].get("VmRSS"),
                        "gpu": sample["nvidia_smi"]["stdout"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if returncode is not None:
                break
            sequence += 1
            time.sleep(args.sample_interval_s)

    completion = {
        "schema": "daaam.no_gt_nominal_window_completion.v1",
        "status": "complete" if returncode == 0 else "failed",
        "returncode": returncode,
        "started_utc": started_utc.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.monotonic() - started_monotonic,
        "stdout": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
        "resource_samples": str(samples_path),
        "resource_samples_sha256": sha256_file(samples_path),
        "key_artifacts": key_artifacts(run_dir),
    }
    if (run_dir / "realtime_run_report.json").is_file():
        report = json.loads(
            (run_dir / "realtime_run_report.json").read_text(encoding="utf-8")
        )
        completion["run_report_extract"] = {
            key: report.get(key)
            for key in (
                "status",
                "frames_requested",
                "frames_resumed_from",
                "frames_dispatched",
                "frames_completed",
                "dropped_frames",
                "quality_passed",
                "hard_quality_failures",
            )
        }
    write_json(output_root / "completion.json", completion)
    print(json.dumps(completion, indent=2, ensure_ascii=False), flush=True)
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
