"""G1 semantic-map experiment registry, provenance, execution, and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import yaml


SCHEMA = "daaam.g1_semantic_map_experiment.v1"
REGISTRY_SCHEMA = "daaam.g1_semantic_map_registry.v1"
SPLIT_SCHEMA = "daaam.g1_semantic_map_splits.v1"
ANNOTATION_SCHEMA = "daaam.g1_semantic_annotation_task.v1"


def _variant(name: str, **parameters: Any) -> dict[str, Any]:
    return {"name": name, "parameters": parameters}


EXPERIMENT_CATALOG: dict[str, dict[str, Any]] = {
    "E0": {
        "module": "end_to_end_baseline",
        "runner": "baseline",
        "variants": [_variant("baseline")],
    },
    "E1": {
        "module": "input_contract",
        "runner": "input_audit",
        "variants": [
            _variant("sync_2ms", maximum_stereo_delta_ms=2.0),
            _variant("sync_5ms", maximum_stereo_delta_ms=5.0),
            _variant("sync_10ms", maximum_stereo_delta_ms=10.0),
            _variant("camera_order_diagnostic", camera_order="both"),
            _variant(
                "projection_pinhole_visual_right_only",
                preparation_model="pinhole_rectified",
                use_lidar_right_rectification=False,
            ),
            _variant(
                "projection_pinhole_full_stereo",
                preparation_model="pinhole_unrectified",
                use_lidar_right_rectification=False,
            ),
            _variant(
                "projection_kannala_brandt_virtual",
                preparation_model="kannala_brandt",
                use_lidar_right_rectification=False,
            ),
            _variant(
                "projection_lidar_right_projective",
                preparation_model="pinhole_rectified",
                use_lidar_right_rectification=True,
            ),
        ],
    },
    "E2": {
        "module": "keyframe_selection",
        "runner": "keyframes",
        "variants": [
            _variant("all_frames", bypass=True),
            _variant(
                "default",
                soft_translation_m=0.06,
                soft_rotation_deg=5.0,
                hard_translation_m=0.15,
                hard_rotation_deg=12.0,
                max_gap_s=1.5,
            ),
            _variant(
                "dense",
                soft_translation_m=0.03,
                soft_rotation_deg=2.5,
                hard_translation_m=0.08,
                hard_rotation_deg=6.0,
                max_gap_s=0.75,
            ),
            _variant(
                "sparse",
                soft_translation_m=0.12,
                soft_rotation_deg=10.0,
                hard_translation_m=0.30,
                hard_rotation_deg=20.0,
                max_gap_s=3.0,
            ),
        ],
    },
    "E3": {
        "module": "stereo_depth",
        "runner": "fast_depth",
        "variants": [
            _variant("iters_4", iters=4, max_disp=416, confidence_mode="left-right"),
            _variant("baseline", iters=8, max_disp=416, confidence_mode="left-right"),
            _variant("iters_16", iters=16, max_disp=416, confidence_mode="left-right"),
            _variant("validity", iters=8, max_disp=416, confidence_mode="validity"),
            _variant("max_disp_320", iters=8, max_disp=320, confidence_mode="left-right"),
        ],
    },
    "E4": {
        "module": "floor_calibration",
        "runner": "floor_calibration",
        "variants": [
            _variant("none", bypass=True),
            _variant("scale_only", floor_rotation_policy="identity"),
            _variant("full_report", floor_rotation_policy="report"),
        ],
    },
    "E5": {
        "module": "temporal_diagnosis",
        "runner": "temporal_diagnosis",
        "variants": [
            _variant("normal"),
            _variant(
                "depth_noise_diagnostic",
                perturbation="depth_noise",
                depth_noise_standard_deviation_m=0.05,
            ),
            _variant(
                "pose_time_offset_diagnostic",
                perturbation="pose_time_offset",
                pose_offset_frames=1,
            ),
            _variant(
                "depth_scale_diagnostic",
                perturbation="depth_scale",
                depth_scale=1.10,
            ),
        ],
    },
    "E6": {
        "module": "local_rgbd_odometry",
        "runner": "rgbd_odometry",
        "variants": [
            _variant("source_pose", bypass=True),
            _variant("default", keyframe_distance_m=0.10, neighbor_span=6, min_inliers=80),
            _variant("conservative", keyframe_distance_m=0.08, neighbor_span=4, min_inliers=120),
            _variant("aggressive", keyframe_distance_m=0.15, neighbor_span=10, min_inliers=50),
        ],
    },
    "E7": {
        "module": "loop_closure",
        "runner": "loop_closure",
        "variants": [
            _variant("none", bypass=True),
            _variant("retrieval_only", dense_candidate_count=0, min_loop_inliers=1000000),
            _variant("sparse_verified", dense_candidate_count=0, min_loop_inliers=140),
            _variant("sparse_dense_verified", dense_candidate_count=80, min_loop_inliers=140),
        ],
    },
    "E8": {
        "module": "global_pose_graph",
        "runner": "global_optimization",
        "variants": [
            _variant("source_pose", bypass=True),
            _variant("local_only", max_loop_edges=0),
            _variant("global_default", max_loop_edges=8, iterations=250),
            _variant("strong_gravity", max_loop_edges=8, iterations=250, gravity_sigma_deg=1.0),
        ],
    },
    "E9": {
        "module": "temporal_depth_filter",
        "runner": "temporal_filter",
        "variants": [
            _variant("off", bypass=True),
            _variant("lenient", offsets="1,2", scale=0.5, min_judged=4, min_support=0.35),
            _variant("default", offsets="1,2,3", scale=0.5, min_judged=3, min_support=0.5),
            _variant("strict", offsets="1,2,3", scale=1.0, min_judged=2, min_support=0.75),
        ],
    },
    "E10": {
        "module": "dynamic_isolation",
        "runner": "realtime",
        "variants": [
            _variant("off_diagnostic", cli__minimum_dynamic_pixels=100000000),
            _variant("default", cli__motion_analysis_width=160, cli__minimum_dynamic_pixels=40),
            _variant("intermediate", cli__motion_analysis_width=240, cli__minimum_dynamic_pixels=30),
            _variant("sensitive", cli__motion_analysis_width=320, cli__minimum_dynamic_pixels=20),
        ],
    },
    "E11": {
        "module": "fastsam",
        "runner": "realtime",
        "variants": [
            _variant("baseline"),
            *[_variant(f"conf_{value:g}", fastsam__fastsam_conf=value) for value in (0.2, 0.4)],
            *[_variant(f"area_{value}", semantic__segmentation__min_mask_region_area=value) for value in (150, 600)],
            *[_variant(f"iou_{value:g}", fastsam__fastsam_iou=value) for value in (0.4, 0.6)],
        ],
    },
    "E12": {
        "module": "botsort_reid",
        "runner": "realtime",
        "variants": [
            _variant("baseline"),
            _variant("without_reid", semantic__tracking__with_reid=False),
            _variant("ecc_100", semantic__tracking__cmc_ecc_max_iterations=100),
            _variant("buffer_10", semantic__tracking__track_buffer=10),
            _variant("buffer_60", semantic__tracking__track_buffer=60),
        ],
    },
    "E13": {
        "module": "map_memory_merge",
        "runner": "realtime",
        "variants": [
            _variant(f"merge_{value:g}m", cli__entity_merge_distance_m=value)
            for value in (0.2, 0.5, 0.8)
        ],
    },
    "E14": {
        "module": "dam_grounding",
        "runner": "realtime",
        "variants": [
            _variant(f"observations_{value}", cli__semantic_minimum_observations=value)
            for value in (3, 5, 8)
        ],
    },
    "E15": {
        "module": "semantic_increment",
        "runner": "realtime",
        "variants": [
            _variant("geometry_only", cli__semantic_mode="disabled"),
            _variant("frontend", cli__semantic_mode="frontend"),
            _variant("dam", cli__semantic_mode="dam"),
        ],
    },
    "E16": {
        "module": "hydra_geometry",
        "runner": "realtime",
        "variants": [
            _variant("voxel_5cm", hydra_profile="config/hydra_g1_high_quality.yaml"),
            _variant("voxel_3cm", hydra_profile="config/hydra_g1_high_quality_3cm.yaml"),
            _variant("object_observations_4", hydra__active_window__tracker__min_num_observations=4),
            _variant("object_min_volume_0_0001", hydra__active_window__object_extractor__min_object_volume=0.0001),
            _variant("object_max_range_2m", hydra__active_window__object_detector__max_range=2.0),
        ],
    },
    "E17": {
        "module": "dsg_binding",
        "runner": "realtime",
        "variants": [
            _variant("strict", cli__object_binding_maximum_center_distance_m=0.10, cli__object_binding_maximum_aabb_gap_m=0.025),
            _variant("medium", cli__object_binding_maximum_center_distance_m=0.35, cli__object_binding_maximum_aabb_gap_m=0.075),
            _variant("wide", cli__object_binding_maximum_center_distance_m=0.75, cli__object_binding_maximum_aabb_gap_m=0.15),
        ],
    },
    "E18": {
        "module": "exact_label_postpass",
        "runner": "realtime",
        "variants": [
            _variant("live_provisional_diagnostic", diagnostic_only=True),
            _variant("exact_postpass_commit"),
        ],
    },
    "Q1": {
        "module": "query_utility",
        "runner": "query",
        "variants": [_variant("multilingual_evidence")],
    },
}


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand(item) for key, item in value.items()}
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    temporary.replace(path)


def _set_nested(mapping: dict[str, Any], path: list[str], value: Any) -> None:
    if not path:
        raise ValueError("configuration override path is empty")
    current = mapping
    for key in path[:-1]:
        child = current.get(key)
        if child is None:
            child = {}
            current[key] = child
        if not isinstance(child, dict):
            raise ValueError(f"configuration override crosses scalar key: {key}")
        current = child
    current[path[-1]] = value


def _command_output(
    command: list[str],
    cwd: Path,
    timeout_s: float = 15.0,
    env: Mapping[str, str] | None = None,
) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    repository_root: Path
    python_executable: Path
    environment_setup: Path | None
    workspace: Path
    raw_dataset: Path
    source_start: int
    source_end: int
    repeats: int
    seeds: tuple[int, ...]
    inputs: dict[str, Any]
    artifacts: dict[str, Path]
    split_ranges: dict[str, tuple[int, int]]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        config_path = Path(path).expanduser().resolve()
        raw = _expand(yaml.safe_load(config_path.read_text()) or {})
        if raw.get("schema") != SCHEMA:
            raise ValueError(f"unsupported experiment schema: {raw.get('schema')}")
        repository_root = Path(raw.get("repository_root", config_path.parents[1])).resolve()

        def resolve(value: str | Path) -> Path:
            candidate = Path(value)
            return (repository_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

        source = raw.get("source_frames") or {}
        source_start = int(source["start"])
        source_end = int(source["end"])
        if source_start < 0 or source_end < source_start:
            raise ValueError("source frame range is invalid")
        repeats = int(raw.get("repeats", 3))
        if repeats < 3:
            raise ValueError("formal experiment repeats must be at least 3")
        seeds = tuple(int(value) for value in raw.get("seeds", range(repeats)))
        if len(seeds) < repeats:
            raise ValueError("at least one seed is required per repeat")
        workspace = resolve(raw["workspace"])
        raw_dataset = resolve(raw["raw_dataset"])
        python_executable = Path(
            raw.get("python_executable", sys.executable)
        ).expanduser().absolute()
        if not python_executable.is_file():
            raise FileNotFoundError(
                f"experiment Python executable is missing: {python_executable}"
            )
        environment_setup_value = raw.get("environment_setup")
        environment_setup = (
            resolve(environment_setup_value)
            if environment_setup_value
            else None
        )
        if environment_setup is not None and not environment_setup.is_file():
            raise FileNotFoundError(
                f"experiment environment setup is missing: {environment_setup}"
            )
        inputs = dict(raw.get("inputs") or {})
        artifacts = {
            key: resolve(value)
            for key, value in (raw.get("artifacts") or {}).items()
            if value
        }
        split_ranges = {
            str(name): (int(bounds[0]), int(bounds[1]))
            for name, bounds in (raw.get("splits") or {}).items()
        }
        expected = {"calibration", "development", "stress", "held_out"}
        if set(split_ranges) != expected:
            raise ValueError(f"splits must be exactly {sorted(expected)}")
        return cls(
            path=config_path,
            repository_root=repository_root,
            python_executable=python_executable,
            environment_setup=environment_setup,
            workspace=workspace,
            raw_dataset=raw_dataset,
            source_start=source_start,
            source_end=source_end,
            repeats=repeats,
            seeds=seeds,
            inputs=inputs,
            artifacts=artifacts,
            split_ranges=split_ranges,
        )


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    experiment_id: str
    module: str
    variant: str
    repeat: int
    seed: int
    runner: str
    parameters: dict[str, Any]
    run_dir: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REGISTRY_SCHEMA,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "module": self.module,
            "variant": self.variant,
            "repeat": self.repeat,
            "seed": self.seed,
            "runner": self.runner,
            "parameters": self.parameters,
            "run_dir": str(self.run_dir),
            "status": "planned",
        }


class ExperimentManager:
    """Materialize immutable inputs, a full run matrix, and auditable results."""

    WORKSPACE_DIRECTORIES = (
        "registry",
        "ground_truth",
        "shared_inputs",
        "runs",
        "reports",
        "failed_runs",
    )

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def runtime_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        setup = self.config.environment_setup
        if setup is None:
            return environment
        command = [
            "bash",
            "-c",
            'source "$1" >/dev/null 2>&1 && env -0',
            "bash",
            str(setup),
        ]
        result = subprocess.run(
            command,
            cwd=self.config.repository_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Could not source experiment environment setup: "
                f"{setup}: {result.stderr.decode(errors='replace')}"
            )
        for record in result.stdout.split(b"\0"):
            if not record or b"=" not in record:
                continue
            key, value = record.split(b"=", 1)
            environment[key.decode()] = value.decode()
        return environment

    def _selected_manifest_records(self) -> list[dict[str, Any]]:
        path = self.config.raw_dataset / "manifest.jsonl"
        records = []
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                tick = int(record["tick"])
                if self.config.source_start <= tick <= self.config.source_end:
                    records.append(record)
        expected = self.config.source_end - self.config.source_start + 1
        if len(records) != expected:
            raise ValueError(f"expected {expected} source records, found {len(records)}")
        return records

    def _selected_input_files(self, records: Iterable[Mapping[str, Any]]) -> list[Path]:
        root = self.config.raw_dataset
        relative = {
            "manifest.json",
            "manifest.jsonl",
            "quality_report.json",
            "timestamps/000000.txt",
            "poses/dense_global/000000/poses.txt",
            "poses/dense_global/000000/poses_7d.txt",
            "poses/dense_global/000000/aux_poses.jsonl",
            "state/000000/odom.jsonl",
            "state/000000/map_pose.jsonl",
            "state/000000/joint_states.jsonl",
        }
        for calibration in (root / "calibrations" / "000000").glob("*"):
            if calibration.is_file():
                relative.add(str(calibration.relative_to(root)))
        for record in records:
            for image in record.get("images", []):
                relative.add(str(image["path"]))
            for lidar in record.get("lidar", []):
                relative.add(str(lidar["path"]))
        return [root / value for value in sorted(relative)]

    def _model_manifest(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {}
        for name, configured_value in self.config.inputs.items():
            configured = str(configured_value or "").strip()
            if name == "dam_model":
                cache = (
                    Path.home()
                    / ".cache/huggingface/hub"
                    / f"models--{configured.replace('/', '--')}"
                )
                blobs = []
                if cache.is_dir():
                    for path in sorted((cache / "blobs").glob("*")):
                        if path.is_file():
                            blobs.append(
                                {
                                    "name": path.name,
                                    "bytes": path.stat().st_size,
                                }
                            )
                manifest[name] = {
                    "model_id": configured or None,
                    "cache": str(cache),
                    "cache_exists": cache.is_dir(),
                    "content_addressed_blobs": blobs,
                    "refs": {
                        str(path.relative_to(cache)): path.read_text().strip()
                        for path in sorted((cache / "refs").rglob("*"))
                        if path.is_file()
                    }
                    if cache.is_dir()
                    else {},
                }
                continue
            path = Path(configured) if configured else None
            if path is not None and not path.is_absolute():
                path = self.config.repository_root / path
            record: dict[str, Any] = {
                "configured": configured or None,
                "resolved": str(path.resolve()) if path is not None else None,
                "exists": bool(path is not None and path.exists()),
            }
            if path is not None and path.is_file():
                record.update(
                    {
                        "kind": "file",
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
            elif path is not None and path.is_dir():
                record.update(
                    {
                        "kind": "directory",
                        "git_sha": _command_output(
                            ["git", "rev-parse", "HEAD"], path
                        ),
                        "git_status": _command_output(
                            ["git", "status", "--porcelain"], path
                        ),
                    }
                )
            manifest[name] = record
        return manifest

    def initialize(self) -> dict[str, Any]:
        for name in self.WORKSPACE_DIRECTORIES:
            (self.config.workspace / name).mkdir(parents=True, exist_ok=True)
        records = self._selected_manifest_records()
        files = self._selected_input_files(records)
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError("selected experiment inputs are missing: " + ", ".join(missing[:5]))
        inventory = [
            {
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(self.config.raw_dataset)),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": _sha256(path),
            }
            for path in files
        ]
        diff = _command_output(["git", "diff", "--binary", "HEAD"], self.config.repository_root, 60.0) or ""
        patch_path = self.config.workspace / "registry" / "worktree.patch"
        patch_path.write_text(diff + ("\n" if diff and not diff.endswith("\n") else ""))
        runtime_environment = self.runtime_environment()
        manifest = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository": {
                "root": str(self.config.repository_root),
                "git_sha": _command_output(["git", "rev-parse", "HEAD"], self.config.repository_root),
                "git_status": _command_output(["git", "status", "--porcelain"], self.config.repository_root),
                "worktree_patch": str(patch_path),
                "worktree_patch_sha256": _sha256(patch_path),
                "submodules": _command_output(
                    ["git", "submodule", "status", "--recursive"],
                    self.config.repository_root,
                ),
            },
            "dataset": {
                "root": str(self.config.raw_dataset),
                "source_frames": [self.config.source_start, self.config.source_end],
                "frame_count": len(records),
                "selected_record_sha256": hashlib.sha256(
                    "\n".join(
                        json.dumps(record, sort_keys=True, separators=(",", ":"))
                        for record in records
                    ).encode()
                ).hexdigest(),
                "files": inventory,
            },
            "runtime": {
                "python": sys.version,
                "executable": sys.executable,
                "experiment_executable": str(self.config.python_executable),
                "environment_setup": (
                    str(self.config.environment_setup)
                    if self.config.environment_setup is not None
                    else None
                ),
                "environment_setup_sha256": (
                    _sha256(self.config.environment_setup)
                    if self.config.environment_setup is not None
                    else None
                ),
                "platform": sys.platform,
                "nvidia_smi": _command_output(["nvidia-smi", "-q"], self.config.repository_root),
                "pip_freeze": _command_output(
                    [str(self.config.python_executable), "-m", "pip", "freeze"],
                    self.config.repository_root,
                    60.0,
                    runtime_environment,
                ),
            },
            "models": self._model_manifest(),
            "config_path": str(self.config.path),
            "config_sha256": _sha256(self.config.path),
        }
        _write_json(self.config.workspace / "registry" / "experiment_manifest.json", manifest)
        _write_json(
            self.config.workspace / "registry" / "raw_manifest.json",
            {
                "schema": "daaam.g1_selected_raw_manifest.v1",
                "root": str(self.config.raw_dataset),
                "source_frames": [self.config.source_start, self.config.source_end],
                "files": inventory,
            },
        )
        return manifest

    def build_splits_and_annotation_tasks(self) -> dict[str, Any]:
        records = {int(record["tick"]): record for record in self._selected_manifest_records()}
        owners: dict[int, str] = {}
        splits: dict[str, list[int]] = {}
        for name, (start, end) in self.config.split_ranges.items():
            if start < self.config.source_start or end > self.config.source_end or end < start:
                raise ValueError(f"split range is outside selected frames: {name}")
            indices = list(range(start, end + 1))
            overlap = set(indices) & set(owners)
            if overlap:
                raise ValueError(f"split ranges overlap at {sorted(overlap)}")
            owners.update({index: name for index in indices})
            splits[name] = indices
        expected = set(range(self.config.source_start, self.config.source_end + 1))
        if set(owners) != expected:
            raise ValueError(f"split ranges do not cover frames: {sorted(expected - set(owners))}")
        split_manifest = {
            "schema": SPLIT_SCHEMA,
            "selection_basis": "contiguous scene blocks; review content labels before formal held-out use",
            "source_dataset": str(self.config.raw_dataset),
            "source_frames": [self.config.source_start, self.config.source_end],
            "splits": {
                name: {
                    "frames": indices,
                    "count": len(indices),
                    "first_sensor_time_ns": int(records[indices[0]]["lidar"][0]["sensor_time_ns"]),
                    "last_sensor_time_ns": int(records[indices[-1]]["lidar"][0]["sensor_time_ns"]),
                }
                for name, indices in splits.items()
            },
        }
        output = self.config.workspace / "ground_truth" / "split_manifest.json"
        _write_json(output, split_manifest)
        tasks_path = self.config.workspace / "ground_truth" / "annotation_tasks.jsonl"
        with tasks_path.open("w") as stream:
            for source_index in sorted(records):
                record = records[source_index]
                task = {
                    "schema": ANNOTATION_SCHEMA,
                    "annotation_version": 1,
                    "source_index": source_index,
                    "split": owners[source_index],
                    "sensor_time_ns": int(record["lidar"][0]["sensor_time_ns"]),
                    "images": {
                        str(image["camera"]): str(self.config.raw_dataset / image["path"])
                        for image in record["images"]
                    },
                    "lidar": str(self.config.raw_dataset / record["lidar"][0]["path"]),
                    "required_fields": [
                        "instances[].mask",
                        "instances[].object_id",
                        "instances[].canonical_name",
                        "instances[].acceptable_synonyms",
                        "instances[].attributes",
                        "instances[].dynamic_state",
                        "instances[].should_have_mesh",
                        "instances[].dsg_object_id",
                    ],
                    "status": "unlabeled",
                }
                stream.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
        return split_manifest

    def runs(self) -> list[ExperimentRun]:
        runs = []
        for experiment_id, experiment in EXPERIMENT_CATALOG.items():
            for variant in experiment["variants"]:
                for repeat in range(1, self.config.repeats + 1):
                    seed = self.config.seeds[repeat - 1]
                    run_id = (
                        f"G1_{self.config.source_start:06d}-{self.config.source_end:06d}_"
                        f"{experiment_id}_{variant['name']}_seed{seed}_rep{repeat}"
                    )
                    runs.append(
                        ExperimentRun(
                            run_id=run_id,
                            experiment_id=experiment_id,
                            module=experiment["module"],
                            variant=variant["name"],
                            repeat=repeat,
                            seed=seed,
                            runner=experiment["runner"],
                            parameters=dict(variant["parameters"]),
                            run_dir=self.config.workspace / "runs" / run_id,
                        )
                    )
        return runs

    def generate_registry(self) -> list[ExperimentRun]:
        registry_path = self.config.workspace / "registry" / "experiment_registry.jsonl"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        runs = self.runs()
        with registry_path.open("w") as stream:
            for run in runs:
                stream.write(json.dumps(run.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        _write_json(
            self.config.workspace / "registry" / "experiment_catalog.json",
            {"schema": REGISTRY_SCHEMA, "experiments": EXPERIMENT_CATALOG},
        )
        return runs

    def preflight(self) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []

        def add(name: str, passed: bool, detail: Any, *, required: bool = True) -> None:
            checks.append(
                {
                    "name": name,
                    "passed": bool(passed),
                    "required": required,
                    "detail": detail,
                }
            )

        for key in (
            "fast_foundation_repo",
            "fast_foundation_checkpoint",
            "foundation_stereo_checkpoint",
            "fastsam_engine",
            "reid_engine",
            "floor_calibration_report",
            "right_rectification_report",
        ):
            configured = str(self.config.inputs.get(key) or "").strip()
            path = Path(configured) if configured else None
            if path is not None and not path.is_absolute():
                path = self.config.repository_root / path
            add(
                f"asset:{key}",
                bool(path is not None and path.exists()),
                {
                    "configured": configured or None,
                    "resolved": str(path.resolve()) if path is not None else None,
                },
            )
        dam_model = str(self.config.inputs.get("dam_model") or "").strip()
        dam_cache = (
            Path.home()
            / ".cache/huggingface/hub"
            / f"models--{dam_model.replace('/', '--')}"
        )
        add(
            "asset:dam_model_cache",
            bool(dam_model and dam_cache.is_dir()),
            {
                "model": dam_model or None,
                "cache": str(dam_cache),
            },
        )

        module_probe = (
            "import importlib,json;"
            "names=['cv2','torch','boxmot','hydra_python','spark_dsg'];"
            "results={};"
            "\nfor n in names:\n"
            " try:\n"
            "  importlib.import_module(n); results[n]={'importable':True,'error':None}\n"
            " except BaseException as e:\n"
            "  results[n]={'importable':False,'error':repr(e)}\n"
            "print(json.dumps(results))"
        )
        modules_text = _command_output(
            [str(self.config.python_executable), "-c", module_probe],
            self.config.repository_root,
            env=self.runtime_environment(),
        )
        modules = json.loads(modules_text) if modules_text else {}
        for name in ("cv2", "torch", "boxmot", "hydra_python", "spark_dsg"):
            detail = modules.get(name) or {}
            add(
                f"python_module:{name}",
                bool(detail.get("importable")),
                detail,
            )

        prepared = self.config.artifacts.get("prepared_dataset")
        selected = self.config.artifacts.get("selected_dataset")
        add(
            "prepared_dataset",
            bool(prepared and (prepared / "tick_index.json").is_file()),
            str(prepared) if prepared else None,
        )
        add(
            "selected_dataset",
            bool(selected and (selected / "tick_index.json").is_file()),
            str(selected) if selected else None,
        )
        lidar_manifest = (
            self.config.workspace
            / (
                "ground_truth/"
                f"lidar_camera_{self.config.source_start}_{self.config.source_end}/"
                "manifest.json"
            )
        )
        add("lidar_camera_ground_truth", lidar_manifest.is_file(), str(lidar_manifest))

        geometry_report = (
            self.config.workspace
            / "shared_inputs/prepared_stereo_geometry_audit_visual"
            / "prepared_stereo_geometry_audit.json"
        )
        geometry = _json(geometry_report) if geometry_report.is_file() else {}
        aggregate = geometry.get("aggregate") or {}
        evaluated = int(aggregate.get("evaluated_frame_count", 0))
        passed = int(aggregate.get("passed_frame_count", 0))
        add(
            "prepared_stereo_geometry_gate",
            evaluated > 0 and passed == evaluated,
            {
                "report": str(geometry_report),
                "evaluated_frames": evaluated,
                "passed_frames": passed,
            },
        )

        annotations_path = (
            self.config.workspace / "ground_truth/annotation_tasks.jsonl"
        )
        annotation_package = (
            self.config.workspace
            / "ground_truth/manual_annotation_package/annotations.json"
        )
        annotation_windows = (
            self.config.workspace / "ground_truth/annotation_windows.json"
        )
        lidar_summary = (
            self.config.workspace
            / (
                "ground_truth/"
                f"lidar_camera_{self.config.source_start}_{self.config.source_end}/"
                "range_edge_texture_summary.json"
            )
        )
        annotation_statuses: dict[str, int] = {}
        if annotations_path.is_file():
            for line in annotations_path.read_text().splitlines():
                if not line.strip():
                    continue
                status = str(json.loads(line).get("status", "missing"))
                annotation_statuses[status] = annotation_statuses.get(status, 0) + 1
        windows = _json(annotation_windows) if annotation_windows.is_file() else {}
        add(
            "annotation_package_and_windows",
            annotation_package.is_file()
            and annotation_windows.is_file()
            and bool(windows.get("windows"))
            and int(windows.get("dual_review_count") or 0) > 0,
            {
                "annotation_package": str(annotation_package),
                "annotation_windows": str(annotation_windows),
                "window_count": len(windows.get("windows") or []),
                "dual_review_count": int(windows.get("dual_review_count") or 0),
            },
        )
        add(
            "lidar_range_edge_texture_summary",
            lidar_summary.is_file(),
            str(lidar_summary),
        )
        human_labels_complete = bool(annotation_statuses) and set(
            annotation_statuses
        ).issubset({"complete", "reviewed"})
        add(
            "manual_semantic_ground_truth",
            human_labels_complete,
            annotation_statuses,
            required=False,
        )

        required_failures = [
            check["name"]
            for check in checks
            if check["required"] and not check["passed"]
        ]
        soft_failures = [
            check["name"]
            for check in checks
            if (not check["required"]) and (not check["passed"])
        ]
        report = {
            "schema": "daaam.g1_semantic_map_preflight.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ready": not required_failures,
            "ready_for_semantic_eval": not required_failures and human_labels_complete,
            "required_failures": required_failures,
            "soft_failures": soft_failures,
            "checks": checks,
        }
        _write_json(
            self.config.workspace / "reports/experiment_preflight.json",
            report,
        )
        return report

    def inventory_workspace(self) -> dict[str, Any]:
        inventory_path = (
            self.config.workspace / "registry/shared_artifact_inventory.json"
        )
        records = []
        for root_name in ("registry", "ground_truth", "shared_inputs", "reports"):
            root = self.config.workspace / root_name
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path == inventory_path or (not path.is_file() and not path.is_symlink()):
                    continue
                if path.is_symlink():
                    records.append(
                        {
                            "path": str(path.relative_to(self.config.workspace)),
                            "kind": "symlink",
                            "target": str(path.resolve()),
                        }
                    )
                else:
                    records.append(
                        {
                            "path": str(path.relative_to(self.config.workspace)),
                            "kind": "file",
                            "bytes": path.stat().st_size,
                            "sha256": _sha256(path),
                        }
                    )
        report = {
            "schema": "daaam.g1_shared_artifact_inventory.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "workspace": str(self.config.workspace),
            "files": sum(record["kind"] == "file" for record in records),
            "symlinks": sum(record["kind"] == "symlink" for record in records),
            "bytes": sum(record.get("bytes", 0) for record in records),
            "records": records,
        }
        _write_json(inventory_path, report)
        return report

    @staticmethod
    def _flag(name: str) -> str:
        return "--" + name.replace("_", "-")

    def _realtime_command(
        self,
        run: ExperimentRun,
        *,
        dataset: Path | None = None,
    ) -> list[str]:
        dataset = dataset or self.config.artifacts["filtered_dataset"]
        realtime_run_dir = run.run_dir / "map_artifacts/realtime"
        semantic_config = run.run_dir / "configs/pipeline_config_variant.yaml"
        if not semantic_config.is_file():
            semantic_config = (
                self.config.repository_root / "config/pipeline_config_experiment.yaml"
            )
        hydra_variant_config = run.run_dir / "configs/hydra_variant.yaml"
        if hydra_variant_config.is_file():
            hydra_config = hydra_variant_config
        elif run.parameters.get("hydra_profile"):
            hydra_config = (
                self.config.repository_root / run.parameters["hydra_profile"]
            )
        else:
            hydra_config = (
                self.config.repository_root / "config/hydra_g1_high_quality.yaml"
            )
        parameters = dict(run.parameters)
        parameters.pop("hydra_profile", None)
        command = [
            str(self.config.python_executable),
            str(self.config.repository_root / "scripts/run_realtime_mapping.py"),
            "--dataset",
            str(dataset),
            "--run-dir",
            str(realtime_run_dir),
            "--rate-hz",
            "1",
            "--queue-capacity",
            "8",
            "--depth-backend",
            "precomputed",
            "--static-map-backend",
            "hydra",
            "--hydra-config-path",
            str(hydra_config),
            "--hydra-labelspace-path",
            str(self.config.repository_root / "config/labels_pseudo.yaml"),
            "--hydra-labelspace-colors",
            str(self.config.repository_root / "config/labels_pseudo.csv"),
            "--semantic-mode",
            "dam",
            "--semantic-config",
            str(semantic_config),
            "--semantic-startup-timeout-s",
            "180",
            "--semantic-drain-timeout-s",
            "600",
            "--stop-after",
            "global",
            "--experiment-telemetry",
            "--experiment-visualizations",
            "--visualization-stride",
            "1",
        ]
        for key, value in parameters.items():
            if not key.startswith("cli__"):
                continue
            flag = self._flag(key.removeprefix("cli__"))
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command.extend([flag, str(value)])
        return command

    def commands_for(self, run: ExperimentRun) -> list[list[str]]:
        root = self.config.repository_root
        python = str(self.config.python_executable)
        run_dir = run.run_dir
        artifacts = self.config.artifacts
        parameters = run.parameters
        if run.runner == "baseline":
            offline_run = run_dir / "map_artifacts/offline"
            prepared_dataset = offline_run / "01_pinhole"
            selected_dataset = offline_run / "02_selected"
            filtered_dataset = offline_run / "08_temporal_depth_filtered"
            # Formal baseline uses visual-only right epipolar remap. LiDAR-guided
            # right rectification remains an E1 ablation and must not silently
            # re-enter the E0 prepare path after the geometry gate was unlocked
            # on the visual-only subset.
            prepare_command = [
                python,
                str(root / "scripts/prepare_g1_pinhole_stereo_dataset.py"),
                "--src",
                str(self.config.raw_dataset),
                "--output",
                str(prepared_dataset),
                "--base-pose-source",
                "map",
                "--input-projection-model",
                "pinhole_rectified",
                "--max-delta-ms",
                "10",
                "--source-indices",
                *(
                    str(value)
                    for value in sorted(
                        json.loads(
                            (
                                self.config.workspace
                                / "shared_inputs/geometry_passing_source_indices.json"
                            ).read_text()
                        ).get("passed_source_indices", [])
                    )
                    or range(self.config.source_start, self.config.source_end + 1)
                ),
                "--recommended-max-depth-m",
                "65.535",
            ]
            return [
                prepare_command,
                [
                    python,
                    str(root / "scripts/run_stereo_mapping.py"),
                    "--adapter",
                    "prepared-stereo",
                    "--src",
                    str(prepared_dataset),
                    "--run-dir",
                    str(offline_run),
                    "--recommended-max-depth-m",
                    "65.535",
                    "--stop-after",
                    "select",
                ],
                [
                    python,
                    str(root / "scripts/visualize_keyframe_selection.py"),
                    "--prepared-dataset",
                    str(prepared_dataset),
                    "--selected-dataset",
                    str(selected_dataset),
                    "--output",
                    str(run_dir / "visualizations/keyframe_selection"),
                ],
                [
                    python,
                    str(root / "scripts/run_fast_foundation_stereo_depth.py"),
                    "--dataset",
                    str(selected_dataset),
                    "--output",
                    str(selected_dataset),
                    "--repo",
                    str(self.config.inputs.get("fast_foundation_repo", "")),
                    "--checkpoint",
                    str(self.config.inputs.get("fast_foundation_checkpoint", "")),
                    "--iters",
                    "8",
                    "--max-disp",
                    "416",
                    "--volume-builder",
                    "triton",
                    "--confidence-mode",
                    "left-right",
                    "--max-depth-m",
                    "65.535",
                    "--save-raw-products",
                ],
                [
                    python,
                    str(root / "scripts/evaluate_stereo_depth_batch_with_lidar.py"),
                    "--raw-dataset",
                    str(self.config.raw_dataset),
                    "--depth-dataset",
                    str(selected_dataset),
                    "--output",
                    str(run_dir / "analysis/depth_lidar"),
                    "--maximum-depth-m",
                    "30",
                ],
                [
                    python,
                    str(root / "scripts/run_stereo_mapping.py"),
                    "--adapter",
                    "prepared-stereo",
                    "--src",
                    str(prepared_dataset),
                    "--run-dir",
                    str(offline_run),
                    "--recommended-max-depth-m",
                    "65.535",
                    "--max-depth-m",
                    "65.535",
                    "--geometry-max-depth-m",
                    "65.535",
                    "--depth-ub",
                    "65.535",
                    "--floor-calibration-report",
                    str(self.config.inputs.get("floor_calibration_report", "")),
                    "--allow-missing-verified-loops",
                    "--accept-direct-fusion-preview",
                    "--stop-after",
                    "fuse",
                    "--resume",
                ],
                self._realtime_command(run, dataset=filtered_dataset),
            ]
        if run.runner == "input_audit":
            if parameters.get("preparation_model"):
                prepared_output = run_dir / "map_artifacts/prepared"
                prepare = [
                    python,
                    str(root / "scripts/prepare_g1_pinhole_stereo_dataset.py"),
                    "--src",
                    str(self.config.raw_dataset),
                    "--output",
                    str(prepared_output),
                    "--base-pose-source",
                    "map",
                    "--input-projection-model",
                    str(parameters["preparation_model"]),
                    "--max-delta-ms",
                    "10",
                    "--source-indices",
                    *(
                        str(value)
                        for value in range(
                            self.config.source_start,
                            self.config.source_end + 1,
                        )
                    ),
                    "--recommended-max-depth-m",
                    "65.535",
                ]
                if parameters.get("use_lidar_right_rectification"):
                    prepare.extend(
                        [
                            "--right-rectification-report",
                            str(
                                self.config.inputs.get(
                                    "right_rectification_report",
                                    "",
                                )
                            ),
                        ]
                    )
                return [
                    prepare,
                    [
                        python,
                        str(root / "scripts/audit_prepared_stereo_geometry.py"),
                        "--dataset",
                        str(prepared_output),
                        "--output",
                        str(run_dir / "stage_reports/projection_geometry"),
                        "--maximum-stereo-delta-ms",
                        "10",
                        "--save-match-visualizations",
                    ],
                ]
            if parameters.get("camera_order") == "both":
                return [[
                    python,
                    str(root / "scripts/audit_stereo_camera_order_features.py"),
                    "--dataset",
                    str(artifacts["prepared_dataset"]),
                    "--output",
                    str(run_dir / "stage_reports/camera_order_audit"),
                ]]
            return [[
                python,
                str(root / "scripts/audit_prepared_stereo_geometry.py"),
                "--dataset",
                str(artifacts["prepared_dataset"]),
                "--output",
                str(run_dir / "stage_reports/prepared_stereo_geometry_audit"),
                "--maximum-stereo-delta-ms",
                str(parameters.get("maximum_stereo_delta_ms", 10.0)),
                "--save-match-visualizations",
            ]]
        if run.runner == "keyframes":
            if parameters.get("bypass"):
                return []
            selected_output = run_dir / "map_artifacts/selected"
            return [[
                python,
                str(root / "scripts/select_mapping_keyframes.py"),
                "--dataset",
                str(artifacts["prepared_dataset"]),
                "--output",
                str(selected_output),
                "--soft-translation-m",
                str(parameters["soft_translation_m"]),
                "--soft-rotation-deg",
                str(parameters["soft_rotation_deg"]),
                "--hard-translation-m",
                str(parameters["hard_translation_m"]),
                "--hard-rotation-deg",
                str(parameters["hard_rotation_deg"]),
                "--max-gap-s",
                str(parameters["max_gap_s"]),
            ], [
                python,
                str(root / "scripts/visualize_keyframe_selection.py"),
                "--prepared-dataset",
                str(artifacts["prepared_dataset"]),
                "--selected-dataset",
                str(selected_output),
                "--output",
                str(run_dir / "visualizations/keyframe_selection"),
            ]]
        if run.runner == "fast_depth":
            depth_output = run_dir / "map_artifacts/depth"
            return [[
                python,
                str(root / "scripts/run_fast_foundation_stereo_depth.py"),
                "--dataset",
                str(artifacts["selected_dataset"]),
                "--output",
                str(depth_output),
                "--repo",
                str(self.config.inputs.get("fast_foundation_repo", "")),
                "--checkpoint",
                str(self.config.inputs.get("fast_foundation_checkpoint", "")),
                "--iters",
                str(parameters["iters"]),
                "--max-disp",
                str(parameters["max_disp"]),
                "--volume-builder",
                "triton",
                "--confidence-mode",
                str(parameters["confidence_mode"]),
                "--max-depth-m",
                "65.535",
                "--save-raw-products",
            ], [
                python,
                str(root / "scripts/evaluate_stereo_depth_batch_with_lidar.py"),
                "--raw-dataset",
                str(self.config.raw_dataset),
                "--depth-dataset",
                str(depth_output),
                "--output",
                str(run_dir / "analysis/depth_lidar"),
                "--maximum-depth-m",
                "30",
            ]]
        if run.runner == "floor_calibration":
            if parameters.get("bypass"):
                return []
            geometry_output = run_dir / "map_artifacts/geometry"
            return [[
                python,
                str(root / "scripts/apply_g1_floor_calibration.py"),
                "--dataset",
                str(artifacts["selected_dataset"]),
                "--calibration-report",
                str(self.config.inputs.get("floor_calibration_report", "")),
                "--output",
                str(geometry_output),
                "--max-depth-m",
                "65.535",
                "--rotation-policy",
                str(parameters["floor_rotation_policy"]),
            ], [
                python,
                str(root / "scripts/evaluate_stereo_depth_batch_with_lidar.py"),
                "--raw-dataset",
                str(self.config.raw_dataset),
                "--depth-dataset",
                str(geometry_output),
                "--output",
                str(run_dir / "analysis/depth_lidar"),
                "--maximum-depth-m",
                "30",
            ]]
        if run.runner == "temporal_diagnosis":
            diagnosis_dataset = artifacts["geometry_dataset"]
            commands: list[list[str]] = []
            if parameters.get("perturbation"):
                diagnosis_dataset = run_dir / "map_artifacts/perturbed_geometry"
                perturbation_command = [
                    python,
                    str(root / "scripts/build_geometry_perturbation_dataset.py"),
                    "--dataset",
                    str(artifacts["geometry_dataset"]),
                    "--output",
                    str(diagnosis_dataset),
                    "--mode",
                    str(parameters["perturbation"]),
                    "--seed",
                    str(run.seed),
                ]
                if "depth_noise_standard_deviation_m" in parameters:
                    perturbation_command.extend(
                        [
                            "--depth-noise-standard-deviation-m",
                            str(parameters["depth_noise_standard_deviation_m"]),
                        ]
                    )
                if "depth_scale" in parameters:
                    perturbation_command.extend(
                        ["--depth-scale", str(parameters["depth_scale"])]
                    )
                if "pose_offset_frames" in parameters:
                    perturbation_command.extend(
                        [
                            "--pose-offset-frames",
                            str(parameters["pose_offset_frames"]),
                        ]
                    )
                commands.append(perturbation_command)
            commands.append([
                python,
                str(root / "scripts/diagnose_temporal_depth_consistency.py"),
                "--dataset",
                str(diagnosis_dataset),
                "--output-dir",
                str(run_dir / "stage_reports/temporal"),
                "--max-depth-m",
                "65.535",
                "--frame-step",
                "1",
                "--neighbor-offsets",
                "1,2,3",
                "--max-panels",
                "300",
                "--worst-pair-count",
                "100",
                "--require-time-contract",
            ])
            return commands
        if run.runner == "rgbd_odometry":
            if parameters.get("bypass"):
                return []
            return [[
                python,
                str(root / "scripts/refine_rgbd_trajectory.py"),
                "--dataset",
                str(artifacts["geometry_dataset"]),
                "--output",
                str(run_dir / "map_artifacts/odometry"),
                "--keyframe-distance-m",
                str(parameters["keyframe_distance_m"]),
                "--local-neighbor-span",
                str(parameters["neighbor_span"]),
                "--min-inliers",
                str(parameters["min_inliers"]),
                "--max-depth-m",
                "65.535",
            ]]
        if run.runner == "loop_closure":
            if parameters.get("bypass"):
                return []
            return [[
                python,
                str(root / "scripts/discover_rgbd_loop_closures.py"),
                "--dataset",
                str(artifacts["geometry_dataset"]),
                "--output-dir",
                str(run_dir / "stage_reports/loops"),
                "--dense-candidate-count",
                str(parameters["dense_candidate_count"]),
                "--min-loop-inliers",
                str(parameters["min_loop_inliers"]),
                "--max-depth-m",
                "65.535",
                "--seed",
                str(run.seed),
            ]]
        if run.runner == "global_optimization":
            if parameters.get("bypass"):
                return []
            command = [
                python,
                str(root / "scripts/optimize_rgbd_pose_graph.py"),
                "--dataset",
                str(artifacts["geometry_dataset"]),
                "--rgbd-odometry-dataset",
                str(artifacts["odometry_dataset"]),
                "--temporal-report",
                str(artifacts["temporal_report"]),
                "--loop-report",
                str(artifacts["loop_report"]),
                "--output",
                str(run_dir / "map_artifacts/global"),
                "--max-loop-edges",
                str(parameters["max_loop_edges"]),
                "--iterations",
                str(parameters.get("iterations", 250)),
            ]
            if "gravity_sigma_deg" in parameters:
                command.extend(
                    [
                        "--gravity-roll-pitch-sigma-deg",
                        str(parameters["gravity_sigma_deg"]),
                    ]
                )
            return [command]
        if run.runner == "temporal_filter":
            if parameters.get("bypass"):
                return []
            filtered_output = run_dir / "map_artifacts/filtered"
            return [[
                python,
                str(root / "scripts/filter_temporal_depth_consistency.py"),
                "--dataset",
                str(artifacts["optimized_dataset"]),
                "--depth-evidence-dataset",
                str(artifacts["selected_dataset"]),
                "--output",
                str(filtered_output),
                "--neighbor-offsets",
                str(parameters["offsets"]),
                "--filter-scale",
                str(parameters["scale"]),
                "--min-judged-neighbors",
                str(parameters["min_judged"]),
                "--min-support-ratio",
                str(parameters["min_support"]),
                "--max-depth-m",
                "65.535",
            ], [
                python,
                str(root / "scripts/evaluate_stereo_depth_batch_with_lidar.py"),
                "--raw-dataset",
                str(self.config.raw_dataset),
                "--depth-dataset",
                str(filtered_output),
                "--output",
                str(run_dir / "analysis/depth_lidar"),
                "--maximum-depth-m",
                "30",
            ]]
        if run.runner == "realtime":
            return [self._realtime_command(run)]
        if run.runner == "query":
            query_dsg = artifacts.get("query_dsg")
            if query_dsg is None:
                return []
            return [[
                python,
                str(root / "scripts/evaluate_semantic_query_set.py"),
                "--dsg",
                str(query_dsg),
                "--query-set",
                str(root / "config/g1_semantic_query_set.yaml"),
                "--output",
                str(run_dir / "analysis/query_evaluation"),
            ]]
        return []

    def _materialize_variant_configs(self, run: ExperimentRun) -> None:
        if run.runner not in {"realtime", "baseline"}:
            return
        config_directory = run.run_dir / "configs"
        semantic_source = (
            self.config.repository_root / "config/pipeline_config_experiment.yaml"
        )
        semantic = yaml.safe_load(semantic_source.read_text()) or {}
        fastsam_source = (
            self.config.repository_root / "config/fastsam/fastsam_config.yaml"
        )
        fastsam = yaml.safe_load(fastsam_source.read_text()) or {}
        hydra_source = (
            self.config.repository_root
            / str(
                run.parameters.get(
                    "hydra_profile",
                    "config/hydra_g1_high_quality.yaml",
                )
            )
        )
        hydra = yaml.safe_load(hydra_source.read_text()) or {}
        for key, value in run.parameters.items():
            if key.startswith("semantic__"):
                _set_nested(
                    semantic,
                    key.removeprefix("semantic__").split("__"),
                    value,
                )
            elif key.startswith("fastsam__"):
                _set_nested(
                    fastsam,
                    key.removeprefix("fastsam__").split("__"),
                    value,
                )
            elif key.startswith("hydra__"):
                _set_nested(
                    hydra,
                    key.removeprefix("hydra__").split("__"),
                    value,
                )
        fastsam_output = config_directory / "fastsam_variant.yaml"
        fastsam_output.write_text(
            yaml.safe_dump(fastsam, sort_keys=False, allow_unicode=True)
        )
        semantic.setdefault("segmentation", {})["model_config_path"] = str(
            fastsam_output
        )
        (config_directory / "pipeline_config_variant.yaml").write_text(
            yaml.safe_dump(semantic, sort_keys=False, allow_unicode=True)
        )
        (config_directory / "hydra_variant.yaml").write_text(
            yaml.safe_dump(hydra, sort_keys=False, allow_unicode=True)
        )

    def materialize_run(self, run: ExperimentRun) -> dict[str, Any]:
        for name in (
            "manifest",
            "configs",
            "logs",
            "telemetry",
            "frame_artifacts",
            "stage_reports",
            "map_artifacts",
            "analysis",
        ):
            (run.run_dir / name).mkdir(parents=True, exist_ok=True)
        self._materialize_variant_configs(run)
        commands = self.commands_for(run)
        geometry_gate_path = (
            self.config.workspace
            / "shared_inputs/prepared_stereo_geometry_audit_visual"
            / "prepared_stereo_geometry_audit.json"
        )
        geometry_gate = (
            (_json(geometry_gate_path).get("aggregate") or {})
            if geometry_gate_path.is_file()
            else {}
        )
        geometry_evaluated = int(geometry_gate.get("evaluated_frame_count", 0))
        geometry_passed = int(geometry_gate.get("passed_frame_count", 0))
        record = {
            **run.to_dict(),
            "commands": commands,
            "command_strings": [
                subprocess.list2cmdline(command) for command in commands
            ],
            "config_sha256": _sha256(self.config.path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "requires_manual_or_external_step": not commands,
            "formal_eligibility": {
                "eligible": (
                    geometry_evaluated > 0
                    and geometry_passed == geometry_evaluated
                ),
                "geometry_gate_report": str(geometry_gate_path),
                "geometry_frames_evaluated": geometry_evaluated,
                "geometry_frames_passed": geometry_passed,
                "failed_runs_must_be_labeled_diagnostic": (
                    geometry_evaluated == 0
                    or geometry_passed != geometry_evaluated
                ),
            },
            "retention_policy": {
                "raw_numeric_products": "all_frames",
                "visualizations": "all_frames",
                "visualization_stride": 1,
                "rejected_and_failed_candidates": "retain",
                "command_stdout_stderr": "retain_per_command",
                "resource_usage": "retain_per_command_and_realtime_samples",
                "artifact_inventory": "sha256_every_file",
            },
        }
        _write_json(run.run_dir / "manifest/run_spec.json", record)
        shutil.copy2(self.config.path, run.run_dir / "configs/experiment.yaml")
        return record

    def inventory_artifacts(self, run: ExperimentRun) -> dict[str, Any]:
        inventory_path = run.run_dir / "manifest/artifact_inventory.json"
        records = []
        for path in sorted(run.run_dir.rglob("*")):
            if not path.is_file() or path == inventory_path:
                continue
            suffix = path.suffix.lower()
            artifact_class = "log_or_other"
            if suffix in {".png", ".jpg", ".jpeg", ".svg", ".mp4"}:
                artifact_class = "visualization"
            elif suffix in {".npy", ".npz", ".bin", ".ply"}:
                artifact_class = "raw_numeric"
            elif suffix in {".json", ".jsonl", ".yaml", ".yml", ".csv"}:
                artifact_class = "structured_record"
            records.append(
                {
                    "path": str(path.relative_to(run.run_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "artifact_class": artifact_class,
                }
            )
        inventory = {
            "schema": "daaam.g1_artifact_inventory.v1",
            "run_id": run.run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "files": records,
            "counts": {
                kind: sum(record["artifact_class"] == kind for record in records)
                for kind in (
                    "visualization",
                    "raw_numeric",
                    "structured_record",
                    "log_or_other",
                )
            },
            "total_bytes": sum(record["bytes"] for record in records),
        }
        _write_json(inventory_path, inventory)
        return inventory

    def _command_already_succeeded(
        self,
        run: ExperimentRun,
        command_index: int,
        command: list[str],
        log_path: Path,
    ) -> bool:
        """Skip re-running a command when an interrupted formal run left valid artifacts."""
        script = Path(command[1]).name if len(command) > 1 else ""
        offline = run.run_dir / "map_artifacts/offline"
        if script == "run_stereo_mapping.py":
            joined = " ".join(command)
            if "--stop-after select" in joined:
                marker = offline / "02_selected" / "tick_index.json"
            else:
                marker = (
                    offline
                    / "10_direct_rgbd_fusion"
                    / "direct_rgbd_fusion_preview.png"
                )
            return marker.is_file()
        markers = {
            "prepare_g1_pinhole_stereo_dataset.py": offline
            / "01_pinhole"
            / "tick_index.json",
            "visualize_keyframe_selection.py": run.run_dir
            / "visualizations/keyframe_selection",
            "run_fast_foundation_stereo_depth.py": offline
            / "02_selected"
            / "fast_foundation_stereo_run.json",
            "evaluate_stereo_depth_batch_with_lidar.py": run.run_dir
            / "analysis/depth_lidar/lidar_batch_evaluation.json",
            "run_realtime_mapping.py": run.run_dir
            / "map_artifacts/realtime/realtime_run_report.json",
        }
        marker = markers.get(script)
        if marker is None:
            return False
        if marker.is_dir():
            return any(marker.iterdir())
        return marker.is_file()

    def execute(
        self,
        run: ExperimentRun,
        *,
        dry_run: bool = True,
        diagnostic: bool = False,
    ) -> dict[str, Any]:
        specification = self.materialize_run(run)
        status_path = run.run_dir / "manifest/status.json"
        if dry_run:
            status = {
                "schema": REGISTRY_SCHEMA,
                "run_id": run.run_id,
                "status": "planned",
                "diagnostic": diagnostic,
                "commands": specification["commands"],
            }
            _write_json(status_path, status)
            self.inventory_artifacts(run)
            return status
        if (
            not diagnostic
            and not specification["formal_eligibility"]["eligible"]
        ):
            status = {
                "schema": REGISTRY_SCHEMA,
                "run_id": run.run_id,
                "status": "blocked_failed_geometry_gate",
                "diagnostic": False,
                "formal_eligibility": specification["formal_eligibility"],
                "hint": (
                    "Fix the upstream gate, or explicitly request a "
                    "non-formal diagnostic run."
                ),
            }
            _write_json(status_path, status)
            self.inventory_artifacts(run)
            return status
        if not specification["commands"]:
            status = {
                "schema": REGISTRY_SCHEMA,
                "run_id": run.run_id,
                "status": "blocked_manual_or_external_step",
                "diagnostic": diagnostic,
            }
            _write_json(status_path, status)
            self.inventory_artifacts(run)
            return status
        started = time.time_ns()
        completed = []
        execution_environment = self.runtime_environment()
        execution_environment["PYTHONUNBUFFERED"] = "1"
        resume_existing = bool(diagnostic is False)  # formal runs may resume
        # Allow explicit resume of interrupted runs via env without rematerializing.
        resume_existing = os.environ.get("G1_EXPERIMENT_RESUME", "1") not in {
            "0",
            "false",
            "False",
        }
        try:
            for command_index, command in enumerate(specification["commands"], start=1):
                log_path = (
                    run.run_dir / "logs" / f"command_{command_index:02d}.log"
                )
                resource_path = (
                    run.run_dir
                    / "telemetry"
                    / f"command_{command_index:02d}_resources.txt"
                )
                if resume_existing and self._command_already_succeeded(
                    run, command_index, command, log_path
                ):
                    completed.append(
                        {
                            "argv": command,
                            "returncode": 0,
                            "started_time_ns": started,
                            "finished_time_ns": time.time_ns(),
                            "log": str(log_path),
                            "log_sha256": _sha256(log_path) if log_path.is_file() else None,
                            "resource_log": (
                                str(resource_path) if resource_path.is_file() else None
                            ),
                            "resource_log_sha256": (
                                _sha256(resource_path)
                                if resource_path.is_file()
                                else None
                            ),
                            "resumed": True,
                        }
                    )
                    continue
                timed_command = list(command)
                time_executable = shutil.which("time")
                if time_executable:
                    timed_command = [
                        time_executable,
                        "-v",
                        "-o",
                        str(resource_path),
                        *command,
                    ]
                command_started = time.time_ns()
                result = subprocess.run(
                    timed_command,
                    cwd=self.config.repository_root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                    env=execution_environment,
                )
                log_path.write_text(result.stdout)
                completed.append(
                    {
                        "argv": command,
                        "returncode": result.returncode,
                        "started_time_ns": command_started,
                        "finished_time_ns": time.time_ns(),
                        "log": str(log_path),
                        "log_sha256": _sha256(log_path),
                        "resource_log": (
                            str(resource_path) if resource_path.is_file() else None
                        ),
                        "resource_log_sha256": (
                            _sha256(resource_path)
                            if resource_path.is_file()
                            else None
                        ),
                    }
                )
                if result.returncode != 0:
                    raise RuntimeError(f"experiment command failed: {command[1]}")
            status_name = "complete"
        except BaseException as error:
            status_name = "failed"
            status = {
                "schema": REGISTRY_SCHEMA,
                "run_id": run.run_id,
                "status": status_name,
                "diagnostic": diagnostic,
                "started_time_ns": started,
                "finished_time_ns": time.time_ns(),
                "commands": completed,
                "error": repr(error),
            }
            _write_json(status_path, status)
            failed = self.config.workspace / "failed_runs" / run.run_id
            if not failed.exists():
                failed.symlink_to(run.run_dir, target_is_directory=True)
            self.inventory_artifacts(run)
            return status
        status = {
            "schema": REGISTRY_SCHEMA,
            "run_id": run.run_id,
            "status": status_name,
            "diagnostic": diagnostic,
            "started_time_ns": started,
            "finished_time_ns": time.time_ns(),
            "commands": completed,
        }
        _write_json(status_path, status)
        self.inventory_artifacts(run)
        return status

    @staticmethod
    def _extract_metrics(run: ExperimentRun) -> dict[str, Any]:
        root = run.run_dir
        realtime_root = root / "map_artifacts/realtime"
        if not realtime_root.is_dir():
            realtime_root = root
        report = _json(realtime_root / "realtime_run_report.json") if (realtime_root / "realtime_run_report.json").is_file() else {}
        metrics = _json(realtime_root / "realtime_metrics.json") if (realtime_root / "realtime_metrics.json").is_file() else {}
        execution_status_path = root / "manifest/status.json"
        execution_status = (
            _json(execution_status_path) if execution_status_path.is_file() else {}
        )
        run_specification_path = root / "manifest/run_spec.json"
        run_specification = (
            _json(run_specification_path)
            if run_specification_path.is_file()
            else {}
        )
        semantic = report.get("semantic_stats") or {}
        dsg = semantic.get("dsg") or {}
        depth_run_path = (
            root / "map_artifacts/depth/fast_foundation_stereo_run.json"
        )
        depth_run = _json(depth_run_path) if depth_run_path.is_file() else {}
        depth_aggregate = depth_run.get("aggregate") or {}
        lidar_evaluation_path = (
            root / "analysis/depth_lidar/lidar_batch_evaluation.json"
        )
        lidar_evaluation = (
            _json(lidar_evaluation_path)
            if lidar_evaluation_path.is_file()
            else {}
        )
        lidar_aggregate = lidar_evaluation.get("aggregate") or {}
        lidar_policies = lidar_aggregate.get("policies") or {}
        lidar_policy = lidar_policies.get("default_adaptive_left_right", {})
        if not lidar_policy:
            filtered_policy_name = lidar_evaluation.get("filtered_policy_name")
            lidar_policy = lidar_policies.get(str(filtered_policy_name), {})
        scale_stability = lidar_aggregate.get("candidate_scale_stability") or {}
        artifact_inventory = [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]
        return {
            "run_id": run.run_id,
            "experiment_id": run.experiment_id,
            "module": run.module,
            "variant": run.variant,
            "repeat": run.repeat,
            "seed": run.seed,
            "status": report.get("status") or execution_status.get("status"),
            "diagnostic": bool(
                execution_status.get("diagnostic")
                or (
                    run_specification.get("formal_eligibility", {}).get(
                        "eligible"
                    )
                    is False
                    and execution_status.get("status") in {"complete", "failed"}
                )
            ),
            "quality_passed": report.get("quality_passed"),
            "hard_quality_failures": report.get("hard_quality_failures"),
            "frames_requested": report.get("frames_requested"),
            "frames_completed": report.get("frames_completed"),
            "elapsed_seconds": metrics.get("elapsed_seconds"),
            "semantic_applied": dsg.get("applied"),
            "semantic_pending": dsg.get("pending"),
            "rejected_no_mesh": dsg.get("rejected_no_mesh"),
            "depth_mean_valid_ratio": depth_aggregate.get("mean_valid_ratio"),
            "depth_mean_left_right_consistency": depth_aggregate.get(
                "mean_left_right_consistency"
            ),
            "depth_lidar_median_absolute_error_m": lidar_policy.get(
                "median_absolute_error_m"
            ),
            "depth_lidar_p90_absolute_error_m": lidar_policy.get(
                "p90_absolute_error_m"
            ),
            "depth_lidar_within_0_5m_ratio": lidar_policy.get(
                "within_0_50_m_ratio"
            ),
            "depth_lidar_scale_relative_span": scale_stability.get(
                "relative_span"
            ),
            "telemetry_files": {
                path.name: _sha256(path)
                for path in sorted((realtime_root / "telemetry").glob("*.jsonl"))
                if path.is_file()
            },
            "artifact_count": len(artifact_inventory),
            "artifact_bytes": sum(
                artifact["bytes"] for artifact in artifact_inventory
            ),
            "artifact_inventory": artifact_inventory,
        }

    @staticmethod
    def _summary(values: list[float], seed: int = 0) -> dict[str, Any]:
        if not values:
            return {"samples": 0, "mean": None, "median": None, "ci95": [None, None]}
        rng = random.Random(seed)
        bootstrap = [
            statistics.fmean(rng.choice(values) for _ in values)
            for _ in range(2000)
        ]
        bootstrap.sort()
        return {
            "samples": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
            "ci95": [
                bootstrap[int(0.025 * (len(bootstrap) - 1))],
                bootstrap[int(0.975 * (len(bootstrap) - 1))],
            ],
        }

    @staticmethod
    def _paired_effect(
        baseline: list[dict[str, Any]],
        candidate: list[dict[str, Any]],
        field: str,
    ) -> dict[str, Any]:
        baseline_by_key = {
            (record["seed"], record["repeat"]): record[field]
            for record in baseline
            if isinstance(record.get(field), (int, float))
            and not isinstance(record.get(field), bool)
        }
        candidate_by_key = {
            (record["seed"], record["repeat"]): record[field]
            for record in candidate
            if isinstance(record.get(field), (int, float))
            and not isinstance(record.get(field), bool)
        }
        keys = sorted(set(baseline_by_key) & set(candidate_by_key))
        differences = [
            float(candidate_by_key[key]) - float(baseline_by_key[key])
            for key in keys
        ]
        if not differences:
            return {
                "pairs": 0,
                "candidate_minus_baseline": ExperimentManager._summary([]),
                "exact_sign_flip_p": None,
            }
        observed = abs(statistics.fmean(differences))
        count = len(differences)
        if count <= 20:
            extreme = 0
            assignments = 1 << count
            for mask in range(assignments):
                permuted = statistics.fmean(
                    value if mask & (1 << index) else -value
                    for index, value in enumerate(differences)
                )
                extreme += abs(permuted) >= observed - 1.0e-15
            p_value = extreme / assignments
        else:
            rng = random.Random(0)
            assignments = 100_000
            extreme = sum(
                abs(
                    statistics.fmean(
                        value if rng.random() < 0.5 else -value
                        for value in differences
                    )
                )
                >= observed - 1.0e-15
                for _ in range(assignments)
            )
            p_value = extreme / assignments
        return {
            "pairs": count,
            "paired_keys": [
                {"seed": seed, "repeat": repeat} for seed, repeat in keys
            ],
            "candidate_minus_baseline": ExperimentManager._summary(differences),
            "exact_sign_flip_p": p_value,
        }

    def collect(self) -> dict[str, Any]:
        records = [self._extract_metrics(run) for run in self.runs()]
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault((record["experiment_id"], record["variant"]), []).append(record)
        summaries = []
        numeric_fields = (
            "hard_quality_failures",
            "frames_completed",
            "elapsed_seconds",
            "semantic_applied",
            "semantic_pending",
            "rejected_no_mesh",
            "depth_mean_valid_ratio",
            "depth_mean_left_right_consistency",
            "depth_lidar_median_absolute_error_m",
            "depth_lidar_p90_absolute_error_m",
            "depth_lidar_within_0_5m_ratio",
            "depth_lidar_scale_relative_span",
        )
        for (experiment_id, variant), group in sorted(grouped.items()):
            quality_records = [
                record
                for record in group
                if isinstance(record["quality_passed"], bool)
            ]
            summaries.append(
                {
                    "experiment_id": experiment_id,
                    "variant": variant,
                    "runs": len(group),
                    "complete_runs": sum(record["status"] == "complete" for record in group),
                    "diagnostic_runs": sum(record["diagnostic"] for record in group),
                    "quality_evaluated_runs": len(quality_records),
                    "quality_pass_rate": (
                        sum(record["quality_passed"] is True for record in quality_records)
                        / len(quality_records)
                        if quality_records
                        else None
                    ),
                    "metrics": {
                        field: self._summary(
                            [
                                float(record[field])
                                for record in group
                                if isinstance(record[field], (int, float))
                                and not isinstance(record[field], bool)
                            ]
                        )
                        for field in numeric_fields
                    },
                }
            )
        comparisons = []
        for experiment_id, experiment in EXPERIMENT_CATALOG.items():
            variants = [variant["name"] for variant in experiment["variants"]]
            if len(variants) < 2:
                continue
            baseline_variant = (
                "baseline" if "baseline" in variants else variants[0]
            )
            baseline = grouped.get((experiment_id, baseline_variant), [])
            for variant in variants:
                if variant == baseline_variant:
                    continue
                candidate = grouped.get((experiment_id, variant), [])
                comparisons.append(
                    {
                        "experiment_id": experiment_id,
                        "baseline_variant": baseline_variant,
                        "candidate_variant": variant,
                        "diagnostic_only": (
                            any(
                                record["diagnostic"]
                                for record in baseline + candidate
                            )
                            or sum(
                                record["status"] == "complete"
                                and not record["diagnostic"]
                                for record in baseline
                            )
                            < self.config.repeats
                            or sum(
                                record["status"] == "complete"
                                and not record["diagnostic"]
                                for record in candidate
                            )
                            < self.config.repeats
                        ),
                        "metrics": {
                            field: self._paired_effect(
                                baseline,
                                candidate,
                                field,
                            )
                            for field in numeric_fields
                        },
                    }
                )
        result = {
            "schema": "daaam.g1_semantic_map_analysis.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "records": records,
            "summaries": summaries,
            "paired_comparisons": comparisons,
            "formal_comparisons_ready": (
                bool(comparisons)
                and all(
                    not comparison["diagnostic_only"]
                    for comparison in comparisons
                )
            ),
            "statistical_unit": "run/continuous scene block/object track; pixels are not independent samples",
            "minimum_formal_repeats": self.config.repeats,
        }
        _write_json(self.config.workspace / "reports" / "aggregate_results.json", result)
        row_height = 28
        width = 1400
        height = 90 + row_height * len(summaries)
        maximum_runs = max(
            (int(summary["runs"]) for summary in summaries),
            default=1,
        )
        rows = []
        for index, summary in enumerate(summaries):
            y = 70 + index * row_height
            planned_width = 600 * int(summary["runs"]) / maximum_runs
            complete_width = 600 * int(summary["complete_runs"]) / maximum_runs
            label = escape(
                f"{summary['experiment_id']} / {summary['variant']}"
            )
            rows.append(
                f'<text x="20" y="{y + 15}" font-size="13">{label}</text>'
                f'<rect x="700" y="{y}" width="{planned_width:.2f}" '
                f'height="18" fill="#d9d9d9"/>'
                f'<rect x="700" y="{y}" width="{complete_width:.2f}" '
                f'height="18" fill="#31a354"/>'
                f'<text x="1310" y="{y + 15}" font-size="12">'
                f'{summary["complete_runs"]}/{summary["runs"]}</text>'
            )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
            '<rect width="100%" height="100%" fill="white"/>'
            '<text x="20" y="30" font-size="22" font-weight="bold">'
            'G1 semantic-map experiment completion</text>'
            '<text x="700" y="52" font-size="13">'
            'gray=planned, green=complete</text>'
            + "".join(rows)
            + "</svg>"
        )
        (
            self.config.workspace / "reports/module_completion.svg"
        ).write_text(svg + "\n")
        depth_summaries = [
            summary
            for summary in summaries
            if summary["experiment_id"] == "E3"
            and summary["metrics"][
                "depth_lidar_median_absolute_error_m"
            ]["mean"]
            is not None
        ]
        if depth_summaries:
            maximum_error = max(
                summary["metrics"][
                    "depth_lidar_median_absolute_error_m"
                ]["mean"]
                for summary in depth_summaries
            )
            depth_rows = []
            for index, summary in enumerate(depth_summaries):
                y = 75 + index * 54
                error = summary["metrics"][
                    "depth_lidar_median_absolute_error_m"
                ]["mean"]
                validity = summary["metrics"]["depth_mean_valid_ratio"]["mean"]
                depth_rows.append(
                    f'<text x="20" y="{y + 17}" font-size="15">'
                    f'{escape(summary["variant"])}</text>'
                    f'<rect x="220" y="{y}" width="{520 * error / maximum_error:.2f}" '
                    f'height="18" fill="#de2d26"/>'
                    f'<text x="755" y="{y + 15}" font-size="13">'
                    f'median |error|={error:.4f}m</text>'
                    f'<rect x="220" y="{y + 23}" '
                    f'width="{520 * float(validity or 0.0):.2f}" '
                    f'height="14" fill="#3182bd"/>'
                    f'<text x="755" y="{y + 36}" font-size="13">'
                    f'valid={float(validity or 0.0):.3f}</text>'
                )
            depth_svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="1100" '
                f'height="{120 + len(depth_summaries) * 54}">'
                '<rect width="100%" height="100%" fill="white"/>'
                '<text x="20" y="30" font-size="22" font-weight="bold">'
                'Diagnostic stereo-depth ablation vs LiDAR</text>'
                '<text x="220" y="52" font-size="12">'
                'red=median absolute depth error, blue=valid depth ratio</text>'
                + "".join(depth_rows)
                + "</svg>"
            )
            (
                self.config.workspace / "reports/depth_ablation.svg"
            ).write_text(depth_svg + "\n")
        report_lines = [
            "# G1 语义地图模块作用报告",
            "",
            f"- 计划运行数：{len(records)}",
            f"- 已完成运行数：{sum(record['status'] == 'complete' for record in records)}",
            f"- 正式重复数：{self.config.repeats}",
            "- 统计单位：运行、连续场景块、对象轨迹；像素不视为独立样本。",
            "",
            "## 模块汇总",
            "",
        ]
        for summary in summaries:
            quality_text = (
                f"{summary['quality_pass_rate']:.3f}"
                if summary["quality_pass_rate"] is not None
                else "未评估"
            )
            report_lines.append(
                f"- {summary['experiment_id']} / {summary['variant']}: "
                f"{summary['complete_runs']}/{summary['runs']} 完成，"
                f"诊断运行 {summary['diagnostic_runs']}，"
                f"质量通过率 {quality_text}"
            )
        report_lines.extend(
            [
                "",
                "## 配对比较",
                "",
                "完整效应量、bootstrap 95% CI 与配对符号翻转检验见 "
                "`aggregate_results.json`。三重复时最小双侧 p 值有限，"
                "必须同时报告效应量和区间，不能只按 p 值筛选模块。",
            ]
        )
        (
            self.config.workspace / "reports/module_effect_report.md"
        ).write_text("\n".join(report_lines) + "\n")
        return result

    def validate(self) -> dict[str, Any]:
        checks = []
        for run in self.runs():
            status_path = run.run_dir / "manifest/status.json"
            status = _json(status_path).get("status") if status_path.is_file() else "missing"
            checks.append(
                {
                    "run_id": run.run_id,
                    "status": status,
                    "complete": status == "complete",
                    "has_spec": (run.run_dir / "manifest/run_spec.json").is_file(),
                }
            )
        report = {
            "schema": "daaam.g1_semantic_map_validation.v1",
            "passed": all(check["complete"] for check in checks),
            "planned_runs": len(checks),
            "complete_runs": sum(check["complete"] for check in checks),
            "checks": checks,
        }
        _write_json(self.config.workspace / "reports" / "experiment_validation.json", report)
        return report
