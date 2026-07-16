"""No-ground-truth acceptance metrics for a stamped VIO trajectory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from daaam.slam.openvins import (
    OPENVINS_PINNED_COMMIT,
    OpenVinsOdometryStream,
    StampedOdometry,
)


@dataclass(frozen=True)
class VioAcceptanceConfig:
    """Hard thresholds for the G1 no-ground-truth acceptance run."""

    maximum_loop_translation_m: float = 0.50
    maximum_loop_rotation_deg: float = 10.0
    maximum_aligned_p95_m: float = 0.30
    minimum_duration_s: float = 1800.0
    minimum_odometry_rate_hz: float = 20.0
    alignment_sample_count: int = 128
    require_sensor_evidence: bool = True
    required_sensor_streams: tuple[str, ...] = (
        "imu",
        "stereo_left",
        "stereo_right",
    )

    def __post_init__(self) -> None:
        if (
            min(
                self.maximum_loop_translation_m,
                self.maximum_loop_rotation_deg,
                self.maximum_aligned_p95_m,
            )
            <= 0.0
        ):
            raise ValueError("VIO error thresholds must be positive")
        if self.minimum_duration_s < 0.0 or self.minimum_odometry_rate_hz < 0.0:
            raise ValueError("VIO minimum duration/rate cannot be negative")
        if self.alignment_sample_count < 4:
            raise ValueError("alignment_sample_count must be at least four")
        if self.require_sensor_evidence and not self.required_sensor_streams:
            raise ValueError("required_sensor_streams cannot be empty")
        if any(not stream.strip() for stream in self.required_sensor_streams):
            raise ValueError("required sensor stream names cannot be empty")


@dataclass(frozen=True)
class SensorEvent:
    """Minimal absolute-time evidence emitted by a robot sensor capture."""

    sensor_time_ns: int
    stream: str
    frame_id: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.sensor_time_ns, bool)
            or not isinstance(self.sensor_time_ns, Integral)
            or self.sensor_time_ns <= 0
        ):
            raise ValueError(
                "sensor event time must be a positive absolute nanosecond value"
            )
        if not self.stream.strip():
            raise ValueError("sensor event stream is required")
        object.__setattr__(self, "sensor_time_ns", int(self.sensor_time_ns))

    @classmethod
    def from_json_record(cls, record: Mapping[str, Any]) -> "SensorEvent":
        missing = [key for key in ("sensor_time_ns", "stream") if key not in record]
        if missing:
            raise ValueError(f"sensor record is missing required fields: {missing}")
        return cls(
            sensor_time_ns=record["sensor_time_ns"],
            stream=str(record["stream"]),
            frame_id=str(record.get("frame_id", "")),
        )


def load_sensor_jsonl(path: Path | str) -> tuple[SensorEvent, ...]:
    """Load a JSONL sensor inventory without depending on ROS bag libraries."""

    source = Path(path)
    events: list[SensorEvent] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                events.append(SensorEvent.from_json_record(record))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid sensor JSONL line {line_number}: {error}"
                ) from error
    if not events:
        raise ValueError(f"sensor JSONL contains no events: {source}")
    return tuple(events)


def _stream_statistics(events: Sequence[SensorEvent]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[SensorEvent]] = {}
    for event in events:
        grouped.setdefault(event.stream, []).append(event)
    report: dict[str, dict[str, Any]] = {}
    for name, values in sorted(grouped.items()):
        times = np.asarray([value.sensor_time_ns for value in values], dtype=np.int64)
        deltas_s = np.diff(times).astype(np.float64) / 1.0e9
        monotonic = bool(np.all(deltas_s > 0.0))
        duration_s = float((times[-1] - times[0]) / 1.0e9) if len(times) > 1 else 0.0
        report[name] = {
            "sample_count": int(len(values)),
            "first_sensor_time_ns": int(times[0]),
            "last_sensor_time_ns": int(times[-1]),
            "duration_s": duration_s,
            "rate_hz": float((len(times) - 1) / duration_s)
            if duration_s > 0.0
            else 0.0,
            "maximum_gap_s": float(np.max(deltas_s)) if len(deltas_s) else None,
            "monotonic": monotonic,
            "frame_ids": sorted({value.frame_id for value in values if value.frame_id}),
        }
    return report


def _trajectory_duration_and_rate(
    samples: Sequence[StampedOdometry],
) -> tuple[float, float]:
    if len(samples) < 2:
        return 0.0, 0.0
    duration_s = (samples[-1].sensor_time_ns - samples[0].sensor_time_ns) / 1.0e9
    rate_hz = (len(samples) - 1) / duration_s if duration_s > 0.0 else 0.0
    return float(duration_s), float(rate_hz)


def _rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def _resample_path(points: np.ndarray, count: int) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("a traversal needs at least two finite 3D points")
    if not np.all(np.isfinite(points)):
        raise ValueError("traversal contains non-finite points")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    if cumulative[-1] <= 1.0e-9:
        raise ValueError("a traversal must have non-zero path length")
    keep = np.r_[True, np.diff(cumulative) > 1.0e-12]
    cumulative = cumulative[keep]
    points = points[keep]
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.stack(
        [np.interp(targets, cumulative, points[:, axis]) for axis in range(3)],
        axis=1,
    )


def _rigid_align(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Align corresponding points with a rigid transform and no scale."""

    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u_matrix, _singular_values, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T
    translation = target_mean - rotation @ source_mean
    aligned = (rotation @ source.T).T + translation
    return aligned, np.c_[rotation, translation]


def align_repeated_traversal(
    reference_points: np.ndarray,
    candidate_points: np.ndarray,
    *,
    sample_count: int,
) -> dict[str, Any]:
    """Choose forward/reverse correspondence and report rigid-aligned errors."""

    reference = _resample_path(
        np.asarray(reference_points, dtype=np.float64), sample_count
    )
    candidate = _resample_path(
        np.asarray(candidate_points, dtype=np.float64), sample_count
    )
    alternatives = []
    for reversed_direction in (False, True):
        ordered = candidate[::-1] if reversed_direction else candidate
        aligned, transform = _rigid_align(ordered, reference)
        errors = np.linalg.norm(aligned - reference, axis=1)
        alternatives.append(
            {
                "reversed": reversed_direction,
                "aligned_p50_m": float(np.percentile(errors, 50.0)),
                "aligned_p95_m": float(np.percentile(errors, 95.0)),
                "aligned_maximum_m": float(np.max(errors)),
                "alignment_parent_T_candidate_3x4": transform.tolist(),
                "sample_count": sample_count,
            }
        )
    return min(
        alternatives, key=lambda value: (value["aligned_p95_m"], value["reversed"])
    )


def _check(
    code: str,
    passed: bool,
    message: str,
    *,
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "status": "PASS" if passed else "FAIL",
        "hard": True,
        "message": message,
        "metrics": dict(metrics),
        "thresholds": dict(thresholds or {}),
    }


def evaluate_vio_acceptance(
    stream: OpenVinsOdometryStream,
    *,
    sensor_events: Iterable[SensorEvent] | None = None,
    reference_traversal_id: str | None = None,
    config: VioAcceptanceConfig = VioAcceptanceConfig(),
) -> dict[str, Any]:
    """Evaluate duration, loop closure, and repeated-path consistency.

    This is intentionally a no-GT gate: loop closure uses the first and last
    stamped pose, while repeated traversals are resampled by arc length and
    aligned with SE(3) (never Sim(3), so scale drift is not hidden).
    """

    samples = stream.samples
    if len(samples) < 2:
        raise ValueError("VIO acceptance requires at least two odometry samples")
    duration_s, rate_hz = _trajectory_duration_and_rate(samples)
    start = samples[0]
    end = samples[-1]
    loop_translation_m = float(np.linalg.norm(end.position_m - start.position_m))
    loop_rotation_deg = _rotation_angle_deg(
        start.parent_T_child[:3, :3], end.parent_T_child[:3, :3]
    )

    checks = [
        _check(
            "vio.duration",
            duration_s >= config.minimum_duration_s,
            "odometry duration meets the acceptance window"
            if duration_s >= config.minimum_duration_s
            else "odometry duration is shorter than the acceptance window",
            metrics={"duration_s": duration_s, "sample_count": len(samples)},
            thresholds={"minimum_duration_s": config.minimum_duration_s},
        ),
        _check(
            "vio.rate",
            rate_hz >= config.minimum_odometry_rate_hz,
            "odometry output rate passed"
            if rate_hz >= config.minimum_odometry_rate_hz
            else "odometry output rate is too low",
            metrics={"rate_hz": rate_hz},
            thresholds={"minimum_odometry_rate_hz": config.minimum_odometry_rate_hz},
        ),
        _check(
            "vio.loop_translation",
            loop_translation_m <= config.maximum_loop_translation_m,
            "loop translation closure passed"
            if loop_translation_m <= config.maximum_loop_translation_m
            else "loop translation closure exceeded the limit",
            metrics={"loop_translation_m": loop_translation_m},
            thresholds={
                "maximum_loop_translation_m": config.maximum_loop_translation_m
            },
        ),
        _check(
            "vio.loop_rotation",
            loop_rotation_deg <= config.maximum_loop_rotation_deg,
            "loop rotation closure passed"
            if loop_rotation_deg <= config.maximum_loop_rotation_deg
            else "loop rotation closure exceeded the limit",
            metrics={"loop_rotation_deg": loop_rotation_deg},
            thresholds={"maximum_loop_rotation_deg": config.maximum_loop_rotation_deg},
        ),
    ]

    traversals = stream.traversals()
    traversal_names = list(traversals)
    reference_name = reference_traversal_id or traversal_names[0]
    if reference_name not in traversals:
        raise ValueError(f"unknown reference traversal: {reference_name}")
    comparisons: list[dict[str, Any]] = []
    reference_points = np.asarray(
        [sample.position_m for sample in traversals[reference_name]], dtype=np.float64
    )
    for name, traversal in traversals.items():
        if name == reference_name:
            continue
        candidate_points = np.asarray(
            [sample.position_m for sample in traversal], dtype=np.float64
        )
        try:
            result = align_repeated_traversal(
                reference_points,
                candidate_points,
                sample_count=config.alignment_sample_count,
            )
            result.update(
                {
                    "reference_traversal_id": reference_name,
                    "candidate_traversal_id": name,
                }
            )
        except ValueError as error:
            result = {
                "reference_traversal_id": reference_name,
                "candidate_traversal_id": name,
                "error": str(error),
                "aligned_p95_m": None,
                "reversed": None,
            }
        comparisons.append(result)

    valid_p95 = [
        float(result["aligned_p95_m"])
        for result in comparisons
        if result.get("aligned_p95_m") is not None
    ]
    worst_p95_m = max(valid_p95) if valid_p95 else None
    repeated_passed = (
        len(comparisons) > 0
        and len(valid_p95) == len(comparisons)
        and worst_p95_m is not None
        and worst_p95_m <= config.maximum_aligned_p95_m
    )
    checks.append(
        _check(
            "vio.repeated_path",
            repeated_passed,
            "reverse/repeated trajectory aligned P95 passed"
            if repeated_passed
            else "reverse/repeated trajectory evidence is missing or inconsistent",
            metrics={
                "reference_traversal_id": reference_name,
                "comparison_count": len(comparisons),
                "worst_aligned_p95_m": worst_p95_m,
            },
            thresholds={"maximum_aligned_p95_m": config.maximum_aligned_p95_m},
        )
    )

    sensor_values = tuple(sensor_events or ())
    sensor_stats = _stream_statistics(sensor_values) if sensor_values else {}
    required_sensor_results = {
        name: sensor_stats.get(name) for name in config.required_sensor_streams
    }
    sensor_passed = all(
        result is not None
        and bool(result["monotonic"])
        and float(result["duration_s"]) >= config.minimum_duration_s
        for result in required_sensor_results.values()
    )
    if config.require_sensor_evidence:
        checks.append(
            _check(
                "vio.sensor_evidence",
                sensor_passed,
                "required stereo and IMU evidence passed"
                if sensor_passed
                else "required stereo/IMU evidence is missing, short, or non-monotonic",
                metrics={"required_streams": required_sensor_results},
                thresholds={"minimum_duration_s": config.minimum_duration_s},
            )
        )

    passed = all(check["status"] == "PASS" for check in checks)
    return {
        "schema_version": 1,
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "estimator": {
            "implementation": "OpenVINS",
            "pinned_commit": OPENVINS_PINNED_COMMIT,
            "source_topic": start.source_topic,
            "frame_id": start.frame_id,
            "child_frame_id": start.child_frame_id,
            "timestamp_semantics": start.lookup_semantics,
        },
        "thresholds": asdict(config),
        "metrics": {
            "sample_count": len(samples),
            "first_sensor_time_ns": start.sensor_time_ns,
            "last_sensor_time_ns": end.sensor_time_ns,
            "duration_s": duration_s,
            "odometry_rate_hz": rate_hz,
            "loop_translation_m": loop_translation_m,
            "loop_rotation_deg": loop_rotation_deg,
            "reverse_repeated_aligned_p95_m": worst_p95_m,
        },
        "traversal_comparisons": comparisons,
        "sensor_streams": sensor_stats,
        "checks": checks,
        "failures": [check["code"] for check in checks if check["status"] == "FAIL"],
    }


def write_vio_acceptance_report(report: Mapping[str, Any], path: Path | str) -> None:
    """Atomically write a strict, machine-readable acceptance report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
