"""Uniform PASS/WARN/FAIL checks between mapping pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml


class GateStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class GateResult:
    code: str
    stage: str
    status: GateStatus
    hard: bool
    message: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()

    @property
    def blocks_pipeline(self) -> bool:
        return self.hard and self.status is GateStatus.FAIL

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["blocks_pipeline"] = self.blocks_pipeline
        return value


@dataclass(frozen=True)
class QualityGateConfig:
    maximum_stereo_delta_ms: float = 10.0
    require_pinhole_projection: bool = True
    minimum_depth_valid_ratio: float = 0.15
    minimum_depth_temporal_agreement: float = 0.70
    minimum_left_right_consistency: float = 0.60
    minimum_left_right_coverage: float = 0.25
    maximum_pose_translation_step_m: float = 0.50
    maximum_pose_rotation_step_deg: float = 20.0
    maximum_pose_position_std_m: float = 0.50
    maximum_dynamic_contamination_rate: float = 0.01
    maximum_dynamic_unknown_ratio: float = 0.60
    minimum_largest_mesh_component_ratio: float = 0.10
    maximum_mesh_components: int = 1000
    maximum_semantic_pending_ratio: float = 0.10
    stage_p95_limits_ms: Mapping[str, float] = field(
        default_factory=lambda: {
            "pose": 30.0,
            "tracking": 50.0,
            "segmentation": 250.0,
            "depth": 250.0,
            "fusion": 250.0,
            "dynamic": 100.0,
            "semantic": 5000.0,
            "global": 250.0,
        }
    )
    stage_queue_p95_limits_ms: Mapping[str, float] = field(
        default_factory=lambda: {
            "pose": 30.0,
            "tracking": 50.0,
            "segmentation": 250.0,
            "depth": 250.0,
            "fusion": 250.0,
            "dynamic": 100.0,
            "semantic": 5000.0,
            "global": 1000.0,
        }
    )
    maximum_runtime_drop_ratio: float = 0.10
    maximum_depth_peak_cuda_memory_bytes: int = 20_000_000_000
    maximum_depth_worker_rss_bytes: int = 16_000_000_000
    maximum_depth_worker_restarts: int = 2
    maximum_runtime_errors: int = 0

    def __post_init__(self) -> None:
        ratios = (
            self.minimum_depth_valid_ratio,
            self.minimum_depth_temporal_agreement,
            self.minimum_left_right_consistency,
            self.minimum_left_right_coverage,
            self.maximum_dynamic_contamination_rate,
            self.maximum_dynamic_unknown_ratio,
            self.minimum_largest_mesh_component_ratio,
            self.maximum_semantic_pending_ratio,
        )
        if any(value < 0.0 or value > 1.0 for value in ratios):
            raise ValueError("quality ratio thresholds must be in [0, 1]")
        if self.maximum_mesh_components < 1:
            raise ValueError("maximum_mesh_components must be positive")
        if any(value <= 0 for value in self.stage_p95_limits_ms.values()):
            raise ValueError("runtime latency limits must be positive")
        if any(value <= 0 for value in self.stage_queue_p95_limits_ms.values()):
            raise ValueError("runtime queue limits must be positive")
        if not 0.0 <= self.maximum_runtime_drop_ratio <= 1.0:
            raise ValueError("maximum_runtime_drop_ratio must be in [0, 1]")
        if min(
            self.maximum_depth_peak_cuda_memory_bytes,
            self.maximum_depth_worker_rss_bytes,
        ) <= 0 or min(
            self.maximum_depth_worker_restarts,
            self.maximum_runtime_errors,
        ) < 0:
            raise ValueError("runtime resource limits are invalid")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "QualityGateConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError("quality gate config must be a mapping")
        return cls(**data)


class QualityGateRunner:
    """Evaluate stage evidence and stop progression on explainable hard failures."""

    STAGES = ("time", "depth", "pose", "dynamic", "runtime", "map", "semantic")

    def __init__(self, config: QualityGateConfig = QualityGateConfig()) -> None:
        self.config = config

    def _missing(self, stage: str) -> GateResult:
        return GateResult(
            f"{stage}.missing_evidence",
            stage,
            GateStatus.FAIL,
            True,
            f"Required {stage} quality evidence is missing",
        )

    def evaluate_time(self, evidence: Mapping[str, Any]) -> GateResult:
        required_true = (
            bool(evidence.get("valid", False))
            and bool(evidence.get("monotonic", True))
            and bool(evidence.get("pose_exact_match", True))
            and bool(evidence.get("relative_time_consistent", True))
        )
        stereo_delta = float(evidence.get("maximum_stereo_delta_ms", 0.0))
        projection = str(evidence.get("projection_model", "pinhole"))
        passed = (
            required_true
            and stereo_delta <= self.config.maximum_stereo_delta_ms
            and (
                not self.config.require_pinhole_projection
                or projection == "pinhole"
            )
        )
        return GateResult(
            "time.contract" if passed else "time.contract_violation",
            "time",
            GateStatus.PASS if passed else GateStatus.FAIL,
            True,
            "Absolute time and calibration contract passed"
            if passed
            else "Absolute time, stereo synchronization, or projection contract failed",
            metrics={
                "valid": required_true,
                "maximum_stereo_delta_ms": stereo_delta,
                "projection_model": projection,
            },
            thresholds={
                "maximum_stereo_delta_ms": self.config.maximum_stereo_delta_ms,
                "required_projection_model": (
                    "pinhole" if self.config.require_pinhole_projection else "any"
                ),
            },
        )

    def evaluate_depth(self, evidence: Mapping[str, Any]) -> GateResult:
        valid_ratio = float(evidence.get("valid_ratio", 0.0))
        temporal = float(evidence.get("temporal_agreement", 0.0))
        lr_consistency = float(evidence.get("left_right_consistency", 0.0))
        lr_coverage = float(evidence.get("left_right_coverage", 1.0))
        passed = (
            valid_ratio >= self.config.minimum_depth_valid_ratio
            and temporal >= self.config.minimum_depth_temporal_agreement
            and lr_consistency >= self.config.minimum_left_right_consistency
            and lr_coverage >= self.config.minimum_left_right_coverage
        )
        return GateResult(
            "depth.quality" if passed else "depth.inconsistent",
            "depth",
            GateStatus.PASS if passed else GateStatus.FAIL,
            True,
            "Depth validity and consistency passed"
            if passed
            else "Depth validity, temporal agreement, or left/right consistency is too low",
            metrics={
                "valid_ratio": valid_ratio,
                "temporal_agreement": temporal,
                "left_right_consistency": lr_consistency,
                "left_right_coverage": lr_coverage,
            },
            thresholds={
                "minimum_valid_ratio": self.config.minimum_depth_valid_ratio,
                "minimum_temporal_agreement": self.config.minimum_depth_temporal_agreement,
                "minimum_left_right_consistency": self.config.minimum_left_right_consistency,
                "minimum_left_right_coverage": self.config.minimum_left_right_coverage,
            },
        )

    def evaluate_pose(self, evidence: Mapping[str, Any]) -> GateResult:
        translation = float(evidence.get("maximum_translation_step_m", float("inf")))
        rotation = float(evidence.get("maximum_rotation_step_deg", float("inf")))
        position_std = float(evidence.get("maximum_position_std_m", float("inf")))
        passed = (
            translation <= self.config.maximum_pose_translation_step_m
            and rotation <= self.config.maximum_pose_rotation_step_deg
            and position_std <= self.config.maximum_pose_position_std_m
            and bool(evidence.get("timestamps_monotonic", True))
        )
        return GateResult(
            "pose.quality" if passed else "pose.jump_or_uncertainty",
            "pose",
            GateStatus.PASS if passed else GateStatus.FAIL,
            True,
            "Pose continuity and covariance passed"
            if passed
            else "Pose jump, time order, or covariance exceeded its limit",
            metrics={
                "maximum_translation_step_m": translation,
                "maximum_rotation_step_deg": rotation,
                "maximum_position_std_m": position_std,
            },
            thresholds={
                "maximum_translation_step_m": self.config.maximum_pose_translation_step_m,
                "maximum_rotation_step_deg": self.config.maximum_pose_rotation_step_deg,
                "maximum_position_std_m": self.config.maximum_pose_position_std_m,
            },
        )

    def evaluate_dynamic(self, evidence: Mapping[str, Any]) -> GateResult:
        contamination = float(evidence.get("dynamic_contamination_rate", 1.0))
        unknown = float(evidence.get("unknown_ratio", 1.0))
        passed = (
            contamination <= self.config.maximum_dynamic_contamination_rate
            and unknown <= self.config.maximum_dynamic_unknown_ratio
        )
        return GateResult(
            "dynamic.isolation" if passed else "dynamic.static_contamination",
            "dynamic",
            GateStatus.PASS if passed else GateStatus.FAIL,
            True,
            "Dynamic and uncertain geometry is isolated from permanent fusion"
            if passed
            else "Dynamic leakage or unknown geometry exceeds the fusion limit",
            metrics={
                "dynamic_contamination_rate": contamination,
                "unknown_ratio": unknown,
            },
            thresholds={
                "maximum_dynamic_contamination_rate": self.config.maximum_dynamic_contamination_rate,
                "maximum_unknown_ratio": self.config.maximum_dynamic_unknown_ratio,
            },
        )

    def evaluate_runtime(self, evidence: Mapping[str, Any]) -> GateResult:
        stages = evidence.get("stages", {})
        if not isinstance(stages, Mapping):
            return self._missing("runtime")
        exceeded = {}
        queue_exceeded = {}
        measured = {}
        measured_queue = {}
        configured_stages = set(self.config.stage_p95_limits_ms) | set(
            self.config.stage_queue_p95_limits_ms
        )
        for stage in sorted(configured_stages):
            if stage not in stages:
                continue
            stage_data = stages[stage]
            latency = stage_data.get("latency", {}).get("service_ms", {})
            p95 = latency.get("p95")
            service_limit = self.config.stage_p95_limits_ms.get(stage)
            if p95 is not None and service_limit is not None:
                measured[stage] = float(p95)
                if float(p95) > service_limit:
                    exceeded[stage] = {
                        "p95_ms": float(p95),
                        "limit_ms": service_limit,
                    }
            queue_p95 = (
                stage_data.get("latency", {})
                .get("queue_wait_ms", {})
                .get("p95")
            )
            if queue_p95 is not None:
                measured_queue[stage] = float(queue_p95)
                queue_limit = self.config.stage_queue_p95_limits_ms.get(stage)
                if queue_limit is not None and float(queue_p95) > queue_limit:
                    queue_exceeded[stage] = {
                        "p95_ms": float(queue_p95),
                        "limit_ms": queue_limit,
                    }
        if not measured:
            return self._missing("runtime")
        totals = evidence.get("totals", {})
        processed = int(totals.get("processed", 0))
        dropped = int(totals.get("dropped", 0))
        errors = int(totals.get("errors", 0))
        drop_ratio = dropped / max(1, processed + dropped)
        drop_exceeded = drop_ratio > self.config.maximum_runtime_drop_ratio
        errors_exceeded = errors > self.config.maximum_runtime_errors
        resources = evidence.get("resources", {})
        peak_cuda = int(resources.get("depth_peak_cuda_memory_bytes", 0))
        peak_worker_rss = int(resources.get("depth_peak_worker_rss_bytes", 0))
        worker_restarts = int(resources.get("depth_worker_restarts", 0))
        resource_exceeded = {}
        if peak_cuda > self.config.maximum_depth_peak_cuda_memory_bytes:
            resource_exceeded["depth_peak_cuda_memory_bytes"] = peak_cuda
        if peak_worker_rss > self.config.maximum_depth_worker_rss_bytes:
            resource_exceeded["depth_peak_worker_rss_bytes"] = peak_worker_rss
        if worker_restarts > self.config.maximum_depth_worker_restarts:
            resource_exceeded["depth_worker_restarts"] = worker_restarts
        status = (
            GateStatus.PASS
            if not exceeded
            and not queue_exceeded
            and not drop_exceeded
            and not errors_exceeded
            and not resource_exceeded
            else GateStatus.FAIL
        )
        failed_parts = []
        if exceeded:
            failed_parts.append(f"service: {', '.join(sorted(exceeded))}")
        if queue_exceeded:
            failed_parts.append(f"queue: {', '.join(sorted(queue_exceeded))}")
        if drop_exceeded:
            failed_parts.append("drop ratio")
        if errors_exceeded:
            failed_parts.append("stage errors")
        if resource_exceeded:
            failed_parts.append("resource limit")
        code = "runtime.latency"
        if status is GateStatus.FAIL:
            if errors_exceeded:
                code = "runtime.stage_error"
            elif exceeded or queue_exceeded or drop_exceeded:
                code = "runtime.p95_exceeded"
            else:
                code = "runtime.resource_exceeded"
        return GateResult(
            code,
            "runtime",
            status,
            True,
            "Runtime stage latency passed"
            if status is GateStatus.PASS
            else f"Runtime limits exceeded ({'; '.join(failed_parts)})",
            metrics={
                "stage_p95_ms": measured,
                "stage_queue_p95_ms": measured_queue,
                "service_exceeded": exceeded,
                "queue_exceeded": queue_exceeded,
                "drop_ratio": drop_ratio,
                "errors": errors,
                "resources": {
                    "depth_peak_cuda_memory_bytes": peak_cuda,
                    "depth_peak_worker_rss_bytes": peak_worker_rss,
                    "depth_worker_restarts": worker_restarts,
                },
                "resource_exceeded": resource_exceeded,
            },
            thresholds={
                "stage_p95_limits_ms": dict(self.config.stage_p95_limits_ms),
                "stage_queue_p95_limits_ms": dict(
                    self.config.stage_queue_p95_limits_ms
                ),
                "maximum_drop_ratio": self.config.maximum_runtime_drop_ratio,
                "maximum_runtime_errors": self.config.maximum_runtime_errors,
                "maximum_depth_peak_cuda_memory_bytes": (
                    self.config.maximum_depth_peak_cuda_memory_bytes
                ),
                "maximum_depth_worker_rss_bytes": (
                    self.config.maximum_depth_worker_rss_bytes
                ),
                "maximum_depth_worker_restarts": (
                    self.config.maximum_depth_worker_restarts
                ),
            },
        )

    def evaluate_map(self, evidence: Mapping[str, Any]) -> GateResult:
        largest = float(evidence.get("largest_component_ratio", 0.0))
        components = int(evidence.get("connected_components", 2**31 - 1))
        passed = (
            largest >= self.config.minimum_largest_mesh_component_ratio
            and components <= self.config.maximum_mesh_components
        )
        return GateResult(
            "map.connectivity" if passed else "map.fragmented_mesh",
            "map",
            GateStatus.PASS if passed else GateStatus.FAIL,
            True,
            "Map connectivity passed"
            if passed
            else "Mesh is too fragmented for a coherent scene map",
            metrics={
                "largest_component_ratio": largest,
                "connected_components": components,
            },
            thresholds={
                "minimum_largest_component_ratio": self.config.minimum_largest_mesh_component_ratio,
                "maximum_connected_components": self.config.maximum_mesh_components,
            },
        )

    def evaluate_semantic(self, evidence: Mapping[str, Any]) -> GateResult:
        pending = int(evidence.get("pending", 0))
        applied = int(evidence.get("applied", 0)) + int(
            evidence.get("applied_alias", 0)
        )
        rejected = int(evidence.get("rejected", 0))
        total = max(1, pending + applied + rejected)
        pending_ratio = pending / total
        passed = pending_ratio <= self.config.maximum_semantic_pending_ratio
        return GateResult(
            "semantic.delivery" if passed else "semantic.pending_backlog",
            "semantic",
            GateStatus.PASS if passed else GateStatus.FAIL,
            True,
            "Semantic corrections are acknowledged"
            if passed
            else "Semantic correction backlog exceeds its limit",
            metrics={
                "pending": pending,
                "applied": applied,
                "rejected": rejected,
                "pending_ratio": pending_ratio,
            },
            thresholds={
                "maximum_pending_ratio": self.config.maximum_semantic_pending_ratio
            },
        )

    def evaluate(
        self,
        context: Mapping[str, Mapping[str, Any]],
        *,
        required_stages: Optional[Iterable[str]] = None,
    ) -> dict[str, Any]:
        selected = tuple(required_stages or self.STAGES)
        unknown = set(selected) - set(self.STAGES)
        if unknown:
            raise ValueError(f"unknown quality stages: {sorted(unknown)}")
        results = []
        for stage in selected:
            evidence = context.get(stage)
            if evidence is None:
                results.append(self._missing(stage))
            else:
                results.append(getattr(self, f"evaluate_{stage}")(evidence))
        blocked = [result for result in results if result.blocks_pipeline]
        return {
            "passed": not blocked,
            "hard_failures": len(blocked),
            "warnings": sum(result.status is GateStatus.WARN for result in results),
            "required_stages": list(selected),
            "results": [result.to_dict() for result in results],
        }
