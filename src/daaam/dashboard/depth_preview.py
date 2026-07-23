"""Discover and render completed FoundationStereo depth frames for the dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import re
import time
from typing import Any


_FRAME_PATTERN = re.compile(r"^(\d{8})\.png$")
_TERMINAL_STATUSES = {
    "complete",
    "completed",
    "succeeded",
    "success",
    "failed",
    "cancelled",
    "canceled",
    "stopped",
    "aborted",
    "error",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _source(run_dir: Path) -> dict[str, Any] | None:
    candidates = (
        (
            "foundation-worker",
            run_dir / "generated_depth" / "depth",
            run_dir / "generated_depth" / "depth_metadata",
            run_dir / "realtime_run_report.json",
        ),
        (
            "foundation-stereo",
            run_dir / "02_selected" / "depth",
            run_dir / "02_selected" / "depth_metadata",
            run_dir / "02_selected" / "foundation_stereo_run.json",
        ),
    )
    for source_id, depth_dir, metadata_dir, report_path in candidates:
        if depth_dir.is_dir():
            return {
                "id": source_id,
                "depth_dir": depth_dir,
                "metadata_dir": metadata_dir,
                "report_path": report_path,
            }
    return None


def _completed_indices(depth_dir: Path, metadata_dir: Path) -> list[int]:
    depth_indices = {
        int(match.group(1))
        for path in depth_dir.iterdir()
        if path.is_file() and (match := _FRAME_PATTERN.fullmatch(path.name))
    }
    if not metadata_dir.is_dir():
        return sorted(depth_indices)
    metadata_indices = {
        int(path.stem)
        for path in metadata_dir.iterdir()
        if path.is_file() and path.suffix == ".json" and path.stem.isdigit()
    }
    return sorted(depth_indices & metadata_indices)


def _depth_range(source: dict[str, Any], run_dir: Path) -> tuple[float, float, str]:
    report = _read_json(Path(source["report_path"]))
    maximum = report.get("maximum_depth_m")
    range_source = "foundation_stereo_report"
    if not isinstance(maximum, (int, float)) or not 0.0 < float(maximum) <= 65.535:
        manifest = _read_json(run_dir / "mapping_run.json")
        maximum = (manifest.get("foundation_stereo") or {}).get("maximum_depth_m")
        range_source = "mapping_manifest"
    if not isinstance(maximum, (int, float)) or not 0.0 < float(maximum) <= 65.535:
        maximum = 5.0
        range_source = "indoor_default"
    minimum = min(0.25, max(0.0, float(maximum) - 0.001))
    return minimum, float(maximum), range_source


def _source_complete(source: dict[str, Any], run_dir: Path) -> bool:
    if source["id"] == "foundation-stereo":
        manifest = _read_json(run_dir / "mapping_run.json")
        completed = manifest.get("stages_completed") or []
        return "depth" in completed or Path(source["report_path"]).is_file()
    report = _read_json(run_dir / "realtime_run_report.json")
    return str(report.get("status", "")).lower() in _TERMINAL_STATUSES


def list_depth_frames(
    run_dir: Path,
    *,
    after: int = -1,
    limit: int = 5000,
) -> dict[str, Any]:
    """Return a compact, incremental index of fully written depth frames."""

    source = _source(run_dir)
    if source is None:
        return {
            "available": False,
            "source": None,
            "frames": [],
            "count": 0,
            "latest_frame": None,
            "complete": False,
            "live": False,
            "updated_at": None,
            "has_more": False,
            "minimum_depth_m": None,
            "maximum_depth_m": None,
            "range_source": None,
            "colormap": "turbo",
        }
    indices = _completed_indices(source["depth_dir"], source["metadata_dir"])
    pending = [index for index in indices if index > after]
    returned = pending[:limit]
    minimum, maximum, range_source = _depth_range(source, run_dir)
    complete = _source_complete(source, run_dir)
    activity_path = Path(source["metadata_dir"])
    if not activity_path.is_dir():
        activity_path = Path(source["depth_dir"])
    activity_mtime = activity_path.stat().st_mtime
    return {
        "available": True,
        "source": source["id"],
        "frames": returned,
        "count": len(indices),
        "latest_frame": indices[-1] if indices else None,
        "complete": complete,
        "live": not complete and time.time() - activity_mtime < 30.0,
        "updated_at": datetime.fromtimestamp(activity_mtime, timezone.utc).isoformat(),
        "has_more": len(pending) > len(returned),
        "minimum_depth_m": minimum,
        "maximum_depth_m": maximum,
        "range_source": range_source,
        "colormap": "turbo",
    }


def depth_frame_path(run_dir: Path, frame_index: int) -> Path:
    """Resolve an indexed frame inside one of the two supported depth sources."""

    if frame_index < 0 or frame_index > 99_999_999:
        raise FileNotFoundError(frame_index)
    source = _source(run_dir)
    if source is None:
        raise FileNotFoundError(frame_index)
    path = Path(source["depth_dir"]) / f"{frame_index:08d}.png"
    metadata_dir = Path(source["metadata_dir"])
    if metadata_dir.is_dir() and not (
        metadata_dir / f"{frame_index:08d}.json"
    ).is_file():
        raise FileNotFoundError(frame_index)
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise FileNotFoundError(frame_index)
    return path


@lru_cache(maxsize=24)
def _render_cached(
    path_text: str,
    modified_ns: int,
    size: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> tuple[bytes, dict[str, float]]:
    del modified_ns, size
    try:
        import cv2
        import numpy as np
    except ImportError as error:  # pragma: no cover - exercised only in minimal envs
        raise RuntimeError("深度预览需要项目环境中的 numpy 和 OpenCV") from error

    depth_mm = cv2.imread(path_text, cv2.IMREAD_UNCHANGED)
    if depth_mm is None or depth_mm.ndim != 2:
        raise ValueError("深度 PNG 尚未写完或不是单通道图像")
    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    span = maximum_depth_m - minimum_depth_m
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    if span > 0.0:
        normalized[valid] = np.rint(
            np.clip(
                (depth_m[valid] - minimum_depth_m) / span,
                0.0,
                1.0,
            )
            * 255.0
        ).astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(normalized, colormap)
    colored[~valid] = (245, 248, 252)
    ok, encoded = cv2.imencode(".png", colored, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    if not ok:
        raise ValueError("无法编码彩虹深度预览")
    valid_values = depth_m[valid]
    stats = {
        "valid_ratio": float(valid.mean()),
        "observed_minimum_m": float(valid_values.min()) if valid_values.size else 0.0,
        "observed_maximum_m": float(valid_values.max()) if valid_values.size else 0.0,
    }
    return encoded.tobytes(), stats


def render_depth_frame(
    path: Path,
    *,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> tuple[bytes, dict[str, float]]:
    """Render a uint16 millimetre PNG with a fixed Turbo rainbow scale."""

    if not 0.0 <= minimum_depth_m < maximum_depth_m <= 65.535:
        raise ValueError("深度色条范围必须满足 0 <= minimum < maximum <= 65.535 m")
    stat = path.stat()
    return _render_cached(
        str(path),
        stat.st_mtime_ns,
        stat.st_size,
        round(float(minimum_depth_m), 4),
        round(float(maximum_depth_m), 4),
    )
