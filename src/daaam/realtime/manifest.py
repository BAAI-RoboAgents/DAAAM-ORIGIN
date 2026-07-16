"""Reproducible run manifests for online and replay mapping."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Optional


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command_output(command: list[str], cwd: Path, timeout_s: float = 2.0) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _package_versions() -> dict[str, Optional[str]]:
    packages = (
        "numpy",
        "scipy",
        "opencv-python",
        "pydantic",
        "PyYAML",
        "torch",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _memory_total_bytes() -> Optional[int]:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def build_run_manifest(
    repository_root: Path,
    dataset: Path,
    *,
    configuration: Mapping[str, Any],
    time_contract: Mapping[str, Any],
    model_configuration: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    dataset = dataset.resolve()
    tick_index = dataset / "tick_index.json"
    camera_info = dataset / "camera_info.json"
    submodule = repository_root / "third_party" / "FoundationStereo"
    manifest = {
        "manifest_version": 1,
        "repository": {
            "root": str(repository_root),
            "git_sha": _command_output(["git", "rev-parse", "HEAD"], repository_root),
            "git_dirty": bool(
                _command_output(["git", "status", "--porcelain"], repository_root)
            ),
            "foundation_stereo_sha": _command_output(
                ["git", "rev-parse", "HEAD"], submodule
            )
            if submodule.is_dir()
            else None,
        },
        "dataset": {
            "path": str(dataset),
            "tick_index_sha256": sha256_file(tick_index) if tick_index.is_file() else None,
            "camera_info_sha256": sha256_file(camera_info) if camera_info.is_file() else None,
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "memory_total_bytes": _memory_total_bytes(),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "packages": _package_versions(),
            "nvidia_smi": _command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                repository_root,
            ),
        },
        "configuration": dict(configuration),
        "models": dict(model_configuration or {}),
        "time_contract": dict(time_contract),
    }
    validate_run_manifest(manifest)
    return manifest


def validate_run_manifest(manifest: Mapping[str, Any]) -> None:
    required = ("manifest_version", "repository", "dataset", "runtime", "configuration", "time_contract")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"run manifest is missing fields: {missing}")
    if int(manifest["manifest_version"]) != 1:
        raise ValueError("unsupported run manifest version")
    if not manifest["dataset"].get("path"):
        raise ValueError("run manifest dataset path is required")
    if not bool(manifest["time_contract"].get("valid", False)):
        raise ValueError("run manifest requires a valid absolute-time contract")


def write_run_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    validate_run_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
