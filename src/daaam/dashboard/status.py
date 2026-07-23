"""Read existing runner artifacts into one compact dashboard state contract."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping

from .workflows import OFFLINE_STAGES, REALTIME_STAGES, get_workflow


KNOWN_MARKERS = (
    "mapping_run.json",
    "run_manifest.json",
    "realtime_checkpoint.json",
    "realtime_run_report.json",
    "quality_report.json",
)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Best-effort reader tolerant of the offline runner's non-atomic writes."""

    if not path.is_file():
        return None, None
    last_error: Exception | None = None
    for _ in range(2):
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                return None, f"{path.name} 顶层不是 JSON object"
            return value, None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            last_error = error
    return None, f"读取 {path.name} 失败：{last_error}"


def _read_checkpoint_summary(
    path: Path, *, maximum_prefix_bytes: int = 8 * 1024 * 1024
) -> tuple[dict[str, Any] | None, str | None]:
    """Read only the small prefix fields, not multi-megabyte trajectory state."""

    if not path.is_file():
        return None, None
    wanted = (
        "completed_frame_indices",
        "dropped_frames",
        "last_sensor_time_ns",
        "map_revision",
    )
    data = ""
    decoder = json.JSONDecoder()
    values: dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            while len(data) < maximum_prefix_bytes:
                chunk = handle.read(min(128 * 1024, maximum_prefix_bytes - len(data)))
                if not chunk:
                    break
                data += chunk
                for key in wanted:
                    if key in values:
                        continue
                    match = re.search(rf'"{re.escape(key)}"\s*:\s*', data)
                    if match is None:
                        continue
                    try:
                        values[key], _ = decoder.raw_decode(data, match.end())
                    except json.JSONDecodeError:
                        continue
                if all(key in values for key in wanted):
                    return values, None
    except (OSError, UnicodeDecodeError) as error:
        return None, f"读取 {path.name} 失败：{error}"
    if "completed_frame_indices" in values:
        return values, None
    return None, f"读取 {path.name} 的进度前缀失败"


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _dir_mtime(path: Path) -> float:
    mtimes = [path.stat().st_mtime]
    for name in KNOWN_MARKERS:
        marker = path / name
        if marker.is_file():
            mtimes.append(marker.stat().st_mtime)
    return max(mtimes)


def _offline_depth_progress(path: Path) -> tuple[int, int | None, float | None]:
    """Read the live FoundationStereo frame count without parsing MB-scale reports."""

    metadata_dir = path / "02_selected" / "depth_metadata"
    completed = 0
    if metadata_dir.is_dir():
        try:
            completed = sum(
                entry.is_file()
                and entry.suffix == ".json"
                and len(entry.stem) == 8
                and entry.stem.isdigit()
                for entry in metadata_dir.iterdir()
            )
        except OSError:
            completed = 0
    selected = None
    report_path = path / "02_selected" / "keyframe_selection_report.json"
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            prefix = handle.read(16 * 1024)
        match = re.search(r'"selected_frame_count"\s*:\s*(\d+)', prefix)
        if match:
            selected = int(match.group(1))
    except OSError:
        pass
    progress = completed / selected if selected and selected > 0 else None
    return completed, selected, min(1.0, progress) if progress is not None else None


def _workflow_for_run(path: Path) -> str | None:
    if (path / "mapping_run.json").is_file():
        return "offline_hq"
    if any((path / name).is_file() for name in KNOWN_MARKERS[1:]):
        return "realtime_semantic"
    if (path / "dsg_updated.manifest.json").is_file():
        return "query_assets"
    return None


def resolve_run_directory(output_root: Path, run_id: str) -> Path:
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("无效的 run id")
    output_root = output_root.resolve()
    path = (output_root / run_id).resolve()
    try:
        path.relative_to(output_root)
    except ValueError as error:
        raise ValueError("run 不在允许的输出目录中") from error
    if not path.is_dir():
        raise FileNotFoundError(run_id)
    return path


def _offline_list_summary(path: Path) -> dict[str, Any]:
    manifest, error = _read_json(path / "mapping_run.json")
    completed_stages = (manifest or {}).get("stages_completed", [])
    completed = len(completed_stages)
    raw_status = (manifest or {}).get("status", "unknown")
    activity_mtime = (path / "mapping_run.json").stat().st_mtime
    next_stage = next(
        (stage for stage in OFFLINE_STAGES if stage not in completed_stages), None
    )
    if next_stage == "depth":
        depth_commits = path / "02_selected" / "depth_metadata"
        if depth_commits.is_dir():
            activity_mtime = max(activity_mtime, depth_commits.stat().st_mtime)
    depth_progress = None
    if next_stage == "depth":
        _, _, depth_progress = _offline_depth_progress(path)
    partial_progress = depth_progress if depth_progress is not None else 0.0
    if raw_status == "complete":
        status = "warning" if "map" in completed_stages else "succeeded"
    elif raw_status == "planned":
        status = "planned"
    elif raw_status == "running":
        age_s = time.time() - activity_mtime
        status = "stale" if age_s > 300 else "running"
    else:
        status = "unknown"
    return {
        "id": path.name,
        "workflow_id": "offline_hq",
        "name": path.name,
        "status": status,
        "progress": (completed + partial_progress) / len(OFFLINE_STAGES),
        "summary": f"{completed}/{len(OFFLINE_STAGES)} 个离线阶段有完成记录",
        "started_at": _iso_mtime(path / "mapping_run.json"),
        "modified_at": datetime.fromtimestamp(
            activity_mtime, timezone.utc
        ).isoformat(),
        "warning": error,
    }


def _realtime_list_summary(path: Path) -> dict[str, Any]:
    report, report_error = _read_json(path / "realtime_run_report.json")
    checkpoint, checkpoint_error = _read_checkpoint_summary(
        path / "realtime_checkpoint.json"
    )
    quality, quality_error = _read_json(path / "quality_report.json")
    completed = (report or {}).get("frames_completed")
    if completed is None:
        completed = len((checkpoint or {}).get("completed_frame_indices", []))
    requested = (report or {}).get("frames_requested")
    raw_status = (report or {}).get("status")
    if raw_status == "complete":
        status = "succeeded" if (quality or {}).get("passed", (report or {}).get("quality_passed", False)) else "failed"
    elif raw_status:
        status = "failed"
    elif (path / "dry_run_plan.json").is_file():
        status = "planned"
    else:
        marker = path / "realtime_checkpoint.json"
        age_s = time.time() - marker.stat().st_mtime if marker.is_file() else float("inf")
        status = "stale" if age_s > 300 else "running"
    progress = completed / requested if isinstance(requested, int) and requested > 0 else None
    return {
        "id": path.name,
        "workflow_id": "realtime_semantic",
        "name": path.name,
        "status": status,
        "progress": progress,
        "summary": f"{completed}/{requested if requested is not None else '?'} 帧完成",
        "modified_at": datetime.fromtimestamp(_dir_mtime(path), timezone.utc).isoformat(),
        "warning": report_error or checkpoint_error or quality_error,
    }


def discover_runs(
    output_root: Path,
    *,
    workflow_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Scan only first-level output directories and known small marker files."""

    output_root = output_root.resolve()
    if not output_root.is_dir():
        return []
    candidates: list[tuple[float, Path, str]] = []
    for path in output_root.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue
        detected = _workflow_for_run(path)
        if detected is None or (workflow_id and detected != workflow_id):
            continue
        candidates.append((_dir_mtime(path), path, detected))
    candidates.sort(key=lambda item: item[0], reverse=True)
    output: list[dict[str, Any]] = []
    for _, path, detected in candidates[: max(1, min(limit, 500))]:
        if detected == "offline_hq":
            output.append(_offline_list_summary(path))
        elif detected == "realtime_semantic":
            output.append(_realtime_list_summary(path))
        else:
            output.append(
                {
                    "id": path.name,
                    "workflow_id": detected,
                    "name": path.name,
                    "status": "succeeded",
                    "progress": 1.0,
                    "summary": "查询资产已发现",
                    "modified_at": datetime.fromtimestamp(_dir_mtime(path), timezone.utc).isoformat(),
                }
            )
    return output


def _empty_node_states(workflow_id: str) -> dict[str, dict[str, Any]]:
    workflow = get_workflow(workflow_id)
    return {
        node.id: {
            "id": node.id,
            "status": "pending",
            "progress": 0.0,
            "message": node.status_hint,
            "metrics": {},
            "inputs": list(node.inputs),
            "outputs": list(node.outputs),
        }
        for node in workflow.nodes
    }


def _artifact(path: Path, label: str, *, preview: bool = False) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "label": label,
        "path": str(path),
        "relative_path": path.name,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "preview": preview and exists,
    }


def _offline_snapshot(
    path: Path, process: Mapping[str, Any] | None
) -> dict[str, Any]:
    manifest, read_error = _read_json(path / "mapping_run.json")
    manifest = manifest or {}
    raw_status = manifest.get("status")
    states = _empty_node_states("offline_hq")
    completed = list(dict.fromkeys(manifest.get("stages_completed", [])))
    results = manifest.get("stage_results", {})
    for stage in completed:
        if stage not in states:
            continue
        result = str(results.get(stage, "executed"))
        state = "resumed" if result == "resumed" else "skipped" if result in {"planned", "not_required_for_prepared_stereo"} else "succeeded"
        states[stage].update(status=state, progress=1.0, message=result)

    process_status = (process or {}).get("status")
    next_stage = next((stage for stage in OFFLINE_STAGES if stage not in completed), None)
    activity_mtime = _dir_mtime(path)
    if next_stage == "depth":
        depth_commits = path / "02_selected" / "depth_metadata"
        if depth_commits.is_dir():
            activity_mtime = max(activity_mtime, depth_commits.stat().st_mtime)
    external_running = (
        raw_status == "running" and time.time() - activity_mtime <= 300
    )
    if next_stage == "depth":
        depth_completed, depth_total, depth_progress = _offline_depth_progress(path)
    elif "depth" in completed:
        selected_count = manifest.get("keyframe_selection", {}).get(
            "selected_frame_count"
        )
        depth_total = selected_count if isinstance(selected_count, int) else None
        depth_completed = depth_total or 0
        depth_progress = 1.0
    else:
        depth_completed, depth_total, depth_progress = 0, None, None
    states["depth"]["metrics"] = {
        "completed_frames": depth_completed,
        "total_frames": depth_total,
        "progress": depth_progress,
    }
    if process_status in {"starting", "running", "stopping"} and next_stage:
        states[next_stage].update(
            status="running",
            progress=depth_progress if next_stage == "depth" else None,
            message="子进程正在执行",
        )
    elif external_running and next_stage:
        states[next_stage].update(
            status="running",
            progress=depth_progress if next_stage == "depth" else None,
            message="检测到阶段产物持续更新",
        )
    elif process_status in {"failed", "cancelled"} and next_stage:
        states[next_stage].update(status=process_status, progress=0.0, message=(process or {}).get("message", "子进程未完成"))

    keyframes = manifest.get("keyframe_selection", {})
    states["select"]["metrics"] = {
        "source_frames": keyframes.get("source_frame_count"),
        "selected_frames": keyframes.get("selected_frame_count"),
        "reduction_ratio": keyframes.get("reduction_ratio"),
    }
    time_contract = manifest.get("prepared_time_contract", {})
    states["prepare"]["metrics"] = {
        "valid": time_contract.get("valid"),
        "frame_count": time_contract.get("frame_count"),
    }
    loops = manifest.get("loop_closures", {})
    states["loops"]["metrics"] = {
        "verified_count": loops.get("verified_count"),
        "dense_tested_count": loops.get("dense_tested_count"),
    }
    optimization = manifest.get("global_pose_graph", {}).get("optimization", {})
    states["optimize"]["metrics"] = {
        "success": optimization.get("success"),
        "initial_residual_rms": optimization.get("initial_residual_rms"),
        "final_residual_rms": optimization.get("final_residual_rms"),
    }
    validation = manifest.get("temporal_validation", {})
    states["validate"]["metrics"] = {
        "agreement": validation.get("overall_agreement_rate_weighted"),
        "passed": validation.get("gate", {}).get("passed"),
    }
    fusion = manifest.get("direct_rgbd_fusion", {})
    states["fuse"]["metrics"] = {
        "manually_accepted": fusion.get("manually_accepted"),
    }

    gates: list[dict[str, Any]] = []
    if loops:
        verified = int(loops.get("verified_count", 0))
        gates.append({"code": "geometry.verified_loop", "stage": "loops", "status": "PASS" if verified >= 1 else "FAIL", "hard": True, "message": f"verified loops: {verified}", "metrics": {"verified_count": verified}, "thresholds": {"minimum": 1}})
    validation_gate = validation.get("gate", {})
    for index, check in enumerate(validation_gate.get("checks", [])):
        gates.append({"code": f"depth.temporal.{index}", "stage": "validate", "status": "PASS" if check.get("passed") else "FAIL", "hard": True, "message": check.get("metric", "temporal depth gate"), "metrics": {"actual": check.get("actual")}, "thresholds": {"operator": check.get("operator"), "threshold": check.get("threshold")}})
    if fusion:
        accepted = bool(fusion.get("manually_accepted"))
        gates.append({"code": "fusion.manual_acceptance", "stage": "fuse", "status": "PASS" if accepted else "BLOCKED", "hard": True, "message": "直接 RGB-D 融合预览已人工验收" if accepted else "等待人工验收直接 RGB-D 融合预览", "metrics": {}, "thresholds": {}})

    if process_status in {"starting", "running", "stopping"}:
        overall = "running"
    elif process_status in {"failed", "cancelled"}:
        overall = process_status
    elif raw_status == "complete":
        overall = "warning" if "map" in completed else "succeeded"
        if "map" in completed:
            states["map"].update(status="warning", message="旧离线完成门只证明 mesh/DSG 文件存在，不证明严格语义提交。")
    elif raw_status == "planned":
        overall = "planned"
    elif raw_status == "running":
        overall = "running" if external_running else "stale"
    else:
        overall = "unknown"

    partial_progress = (
        depth_progress
        if next_stage == "depth" and depth_progress is not None
        else 0.0
    )
    progress = (len(completed) + partial_progress) / len(OFFLINE_STAGES)
    preview_path = Path(fusion.get("preview")) if fusion.get("preview") else path / "10_direct_rgbd_fusion" / "direct_rgbd_fusion_preview.png"
    artifacts = [
        _artifact(path / "mapping_run.json", "离线运行清单"),
        _artifact(preview_path, "直接 RGB-D 融合预览", preview=True),
        _artifact(path / "10_direct_rgbd_fusion" / "direct_rgbd_fusion.ply", "直接 RGB-D 点云"),
        _artifact(path / "11_daaam" / "backend" / "mesh.ply", "目标 Hydra mesh"),
        _artifact(path / "11_daaam" / "backend" / "dsg_with_mesh.json", "目标带 mesh DSG"),
    ]
    warnings = [warning for warning in (read_error,) if warning]
    if "map" in completed:
        warnings.append("离线 map 的目标目录路由和语义完成检查不可靠；请优先以准实时 authoritative 链验收。")
    return {
        "run_id": path.name,
        "workflow_id": "offline_hq",
        "path": str(path),
        "status": overall,
        "started_at": _iso_mtime(path / "mapping_run.json"),
        "progress": progress,
        "summary_metrics": {"stages_completed": len(completed), "stages_total": len(OFFLINE_STAGES)},
        "node_states": states,
        "quality_gates": gates,
        "artifacts": artifacts,
        "warnings": warnings,
        "process": dict(process) if process else None,
        "updated_at": datetime.fromtimestamp(activity_mtime, timezone.utc).isoformat(),
    }


def _stage_metrics(metrics: Mapping[str, Any], stage: str) -> dict[str, Any]:
    value = metrics.get("stages", {}).get(stage, {})
    latency = value.get("latency", {})
    return {
        "processed": value.get("processed"),
        "errors": value.get("errors"),
        "throughput_hz": value.get("throughput_hz"),
        "queue_high_water": value.get("queue_high_water"),
        "drops": value.get("drops"),
        "service_p95_ms": latency.get("service_ms", {}).get("p95"),
        "end_to_end_p95_ms": latency.get("end_to_end_ms", {}).get("p95"),
    }


def _realtime_snapshot(
    path: Path, process: Mapping[str, Any] | None
) -> dict[str, Any]:
    report, report_error = _read_json(path / "realtime_run_report.json")
    checkpoint, checkpoint_error = _read_checkpoint_summary(
        path / "realtime_checkpoint.json"
    )
    metrics, metrics_error = _read_json(path / "realtime_metrics.json")
    quality, quality_error = _read_json(path / "quality_report.json")
    manifest, manifest_error = _read_json(path / "run_manifest.json")
    postpass_file, postpass_error = _read_json(path / "hydra_semantic_postpass.json")
    commit_path = path / "hydra_realtime" / "backend" / "semantic_dsg_commit.json"
    commit, commit_error = _read_json(commit_path)
    report, checkpoint, metrics, quality, manifest = report or {}, checkpoint or {}, metrics or {}, quality or {}, manifest or {}
    states = _empty_node_states("realtime_semantic")

    completed_indices = checkpoint.get("completed_frame_indices", [])
    completed = len(completed_indices)
    requested = report.get("frames_requested")
    if requested is None:
        requested = (process or {}).get("parameters", {}).get("max_frames")
    progress = completed / requested if isinstance(requested, int) and requested > 0 else None
    process_status = (process or {}).get("status")
    final_status = report.get("status")
    running = process_status in {"starting", "running", "stopping"} or (not final_status and completed > 0)

    states["scheduler"]["metrics"] = {
        "completed_frames": completed,
        "requested_frames": requested,
        "dropped_frames": len(checkpoint.get("dropped_frames", {})),
        "map_revision": checkpoint.get("map_revision"),
    }
    for stage in REALTIME_STAGES:
        states[stage]["metrics"] = _stage_metrics(metrics, stage)
    semantic_stats = report.get("semantic_stats") or report.get("semantic_sidecar") or {}
    states["semantic_frontend"]["metrics"] = {
        key: semantic_stats.get(key)
        for key in (
            "frames", "segmentation_calls", "segmentation_failures", "detections",
            "tracking_calls", "tracking_failures", "tracked_instances",
            "label_frames_persisted", "propagation_frames",
        )
    }
    dam_gate = report.get("dam_runtime_gate") or semantic_stats.get("dam_runtime_gate") or {}
    memory = semantic_stats.get("map_memory") or {}
    dsg = semantic_stats.get("dsg") or {}
    states["dam"]["metrics"] = {
        "passed": dam_gate.get("passed"),
        "prompts_submitted": semantic_stats.get("prompts_submitted"),
        "corrections_received": semantic_stats.get("corrections_received"),
        "corrections_submitted": semantic_stats.get("corrections_submitted"),
        "pending": dsg.get("pending") if isinstance(dsg, dict) else None,
        "rejected_no_mesh": dsg.get("rejected_no_mesh") if isinstance(dsg, dict) else memory.get("rejected_no_mesh"),
    }
    postpass = report.get("semantic_postpass") or postpass_file or {}
    states["postpass"]["metrics"] = {
        "status": postpass.get("status"),
        "frames_expected": postpass.get("frames_expected"),
        "frames_replayed": postpass.get("frames_replayed"),
        "label_coverage": postpass.get("label_coverage"),
        "missing_frames": len(postpass.get("missing_frame_indices", [])),
    }
    states["commit"]["metrics"] = {
        "status": (commit or {}).get("status"),
        "hash_verified": (commit or {}).get("hash_verified", (commit or {}).get("verified")),
    }
    states["quality"]["metrics"] = {
        "passed": quality.get("passed"),
        "hard_failures": quality.get("hard_failures"),
        "warnings": quality.get("warnings"),
    }

    semantic_mode = report.get("semantic_mode") or manifest.get("configuration", {}).get("semantic", {}).get("mode")
    if final_status:
        frames_by_stage = report.get("frames_by_stage", {})
        failed = final_status != "complete"
        for node_id in ("scheduler",) + REALTIME_STAGES:
            count = completed if node_id == "scheduler" else int(frames_by_stage.get(node_id, 0))
            if failed and count < (requested or completed):
                state = "failed"
            elif count > 0:
                state = "succeeded"
            else:
                state = "skipped"
            states[node_id].update(status=state, progress=1.0 if state == "succeeded" else 0.0)
        if semantic_mode in {None, "disabled"}:
            for node_id in ("semantic_frontend", "dam", "postpass", "commit"):
                states[node_id].update(status="skipped", progress=1.0, message="本次运行未启用语义旁路")
        else:
            frontend_ok = not semantic_stats.get("segmentation_failures") and not semantic_stats.get("tracking_failures")
            states["semantic_frontend"].update(status="succeeded" if frontend_ok else "failed", progress=1.0)
            if semantic_mode == "dam":
                states["dam"].update(status="succeeded" if dam_gate.get("passed") else "failed", progress=1.0)
            else:
                states["dam"].update(status="skipped", progress=1.0, message="frontend 模式不运行 DAM correction")
            postpass_ok = postpass.get("status") == "complete" and postpass.get("label_coverage") == 1.0
            states["postpass"].update(status="succeeded" if postpass_ok else "failed", progress=1.0)
            commit_ok = bool(commit) and (commit.get("status") in {None, "complete", "committed"})
            states["commit"].update(status="succeeded" if commit_ok else "failed", progress=1.0)
        states["quality"].update(status="succeeded" if quality.get("passed") else "failed", progress=1.0)
    elif running:
        for node_id in ("scheduler",) + REALTIME_STAGES:
            states[node_id].update(status="running", progress=progress)
        if semantic_mode not in {None, "disabled"}:
            states["semantic_frontend"].update(status="running", progress=progress)
            if semantic_mode == "dam":
                states["dam"].update(status="running", progress=None)

    if process_status in {"failed", "cancelled"} and not final_status:
        for node_id in ("scheduler",) + REALTIME_STAGES:
            if states[node_id]["status"] == "running":
                states[node_id].update(status=process_status, message=(process or {}).get("message", "子进程未完成"))

    gates = []
    for result in quality.get("results", []):
        gates.append({
            "code": result.get("code"),
            "stage": result.get("stage"),
            "status": result.get("status"),
            "hard": bool(result.get("hard")),
            "message": result.get("message"),
            "metrics": result.get("metrics", {}),
            "thresholds": result.get("thresholds", {}),
            "blocks_pipeline": bool(result.get("blocks_pipeline")),
        })

    if process_status in {"starting", "running", "stopping"}:
        overall = "running"
    elif process_status in {"failed", "cancelled"} and not final_status:
        overall = process_status
    elif final_status == "complete":
        overall = "succeeded" if quality.get("passed", report.get("quality_passed", False)) else "failed"
    elif final_status:
        overall = "failed"
    elif completed:
        marker = path / "realtime_checkpoint.json"
        overall = "stale" if marker.is_file() and time.time() - marker.stat().st_mtime > 300 else "running"
    else:
        overall = "planned" if (path / "dry_run_plan.json").is_file() else "unknown"

    artifacts = [
        _artifact(path / "run_manifest.json", "运行可追溯清单"),
        _artifact(path / "realtime_checkpoint.json", "实时 checkpoint"),
        _artifact(path / "realtime_metrics.json", "阶段性能指标"),
        _artifact(path / "quality_report.json", "质量门报告"),
        _artifact(path / "realtime_run_report.json", "最终运行报告"),
        _artifact(path / "hydra_semantic_postpass.json", "Exact-label postpass 报告"),
        _artifact(path / "hydra_realtime" / "backend" / "mesh.ply", "最终静态 mesh"),
        _artifact(path / "hydra_realtime" / "backend" / "dsg.json", "最终语义 DSG"),
        _artifact(path / "hydra_realtime" / "backend" / "dsg_with_mesh.json", "带 mesh 语义 DSG"),
        _artifact(commit_path, "语义 DSG durable commit"),
    ]
    warnings = [warning for warning in (report_error, checkpoint_error, metrics_error, quality_error, manifest_error, postpass_error, commit_error) if warning]
    rejected = states["dam"]["metrics"].get("rejected_no_mesh")
    if isinstance(rejected, int) and rejected > 0:
        warnings.append(f"有 {rejected} 个语义实体 rejected_no_mesh；质量 PASS 不会隐藏这一数量。")
    return {
        "run_id": path.name,
        "workflow_id": "realtime_semantic",
        "path": str(path),
        "status": overall,
        "progress": progress if progress is not None else (1.0 if final_status else None),
        "summary_metrics": {
            "frames_completed": completed,
            "frames_requested": requested,
            "dropped_frames": len(checkpoint.get("dropped_frames", {})),
            "dynamic_active": report.get("dynamic_objects_active"),
            "submaps": report.get("submaps"),
            "quality_passed": quality.get("passed"),
        },
        "node_states": states,
        "quality_gates": gates,
        "artifacts": artifacts,
        "warnings": warnings,
        "process": dict(process) if process else None,
        "updated_at": datetime.fromtimestamp(_dir_mtime(path), timezone.utc).isoformat(),
    }


def run_snapshot(
    output_root: Path,
    run_id: str,
    *,
    workflow_id: str | None = None,
    process: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = resolve_run_directory(output_root, run_id)
    detected = _workflow_for_run(path)
    selected = workflow_id or detected
    if selected == "offline_hq":
        return _offline_snapshot(path, process)
    if selected == "realtime_semantic":
        return _realtime_snapshot(path, process)
    raise ValueError(f"无法识别 run 工作流：{run_id}")
