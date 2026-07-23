"""FastAPI application for the local DAAAM mapping dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .commands import CommandValidationError, build_command
from .depth_preview import depth_frame_path, list_depth_frames, render_depth_frame
from .processes import ProcessManager
from .status import discover_runs, resolve_run_directory, run_snapshot
from .workflows import DEFAULT_WORKFLOW_ID, get_workflow, list_workflows


def _default_repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _body_fields(body: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
    workflow_id = str(body.get("workflow_id") or DEFAULT_WORKFLOW_ID)
    preset_id = body.get("preset_id")
    if preset_id is not None:
        preset_id = str(preset_id)
    parameters = body.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise CommandValidationError("parameters 必须是 JSON object")
    return workflow_id, preset_id, parameters


def _add_artifact_urls(snapshot: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(snapshot["path"]).resolve()
    run_id = snapshot["run_id"]
    for artifact in snapshot.get("artifacts", []):
        if not artifact.get("exists"):
            continue
        try:
            relative = Path(artifact["path"]).resolve().relative_to(run_dir)
        except (KeyError, ValueError):
            continue
        artifact["relative_path"] = str(relative)
        artifact["url"] = (
            f"/api/runs/{quote(run_id, safe='')}/artifact"
            f"?path={quote(str(relative), safe='/')}"
        )
    return snapshot


def create_dashboard_app(
    repository_root: Path | str | None = None,
    *,
    output_root: Path | str | None = None,
    process_manager: ProcessManager | None = None,
) -> FastAPI:
    repository = Path(repository_root or _default_repository_root()).resolve()
    output = Path(output_root or repository / "output").resolve()
    output.mkdir(parents=True, exist_ok=True)
    static = Path(__file__).resolve().parent / "static"
    manager = process_manager or ProcessManager(repository)

    app = FastAPI(
        title="DAAAM Mapping Dashboard",
        version="0.1.0",
        description="Local workflow control and artifact-backed observability for DAAAM.",
    )
    app.state.repository_root = repository
    app.state.output_root = output
    app.state.process_manager = manager
    app.mount("/static", StaticFiles(directory=static), name="dashboard-assets")

    @app.middleware("http")
    async def add_local_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static / "index.html", media_type="text/html")

    @app.get("/api/health", tags=["dashboard"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "repository_root": str(repository),
            "output_root": str(output),
            "active_processes": sum(
                info["status"] in {"starting", "running", "stopping"}
                for info in manager.list()
            ),
        }

    @app.get("/api/workflows", tags=["workflows"])
    def workflows() -> dict[str, Any]:
        return {
            "workflows": [workflow.to_dict() for workflow in list_workflows()],
            "default_workflow": DEFAULT_WORKFLOW_ID,
        }

    @app.get("/api/runs", tags=["runs"])
    def runs(
        workflow_id: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        if workflow_id is not None:
            try:
                get_workflow(workflow_id)
            except KeyError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
        return {
            "runs": discover_runs(output, workflow_id=workflow_id, limit=limit),
        }

    @app.get("/api/runs/{run_id}", tags=["runs"])
    def get_run(
        run_id: str,
        workflow_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        managed = manager.for_run(run_id)
        try:
            snapshot = run_snapshot(
                output,
                run_id,
                workflow_id=workflow_id,
                process=managed.info() if managed else None,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="run 不存在") from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _add_artifact_urls(snapshot)

    @app.get("/api/runs/{run_id}/artifact", tags=["runs"])
    def get_artifact(run_id: str, path: str = Query(...)) -> FileResponse:
        try:
            run_dir = resolve_run_directory(output, run_id)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail="run 不存在") from error
        requested = (run_dir / path).resolve()
        try:
            requested.relative_to(run_dir)
        except ValueError as error:
            raise HTTPException(status_code=403, detail="产物路径越界") from error
        if not requested.is_file():
            raise HTTPException(status_code=404, detail="产物不存在")
        allowed = {
            ".json", ".yaml", ".yml", ".txt", ".log", ".png", ".jpg",
            ".jpeg", ".ply", ".rrd", ".sqlite3",
        }
        if requested.suffix.lower() not in allowed:
            raise HTTPException(status_code=403, detail="该产物类型不允许通过面板读取")
        return FileResponse(requested)

    @app.get("/api/runs/{run_id}/depth-frames", tags=["runs"])
    def depth_frames(
        run_id: str,
        after: int = Query(default=-1, ge=-1),
        limit: int = Query(default=5000, ge=1, le=5000),
    ) -> dict[str, Any]:
        try:
            run_dir = resolve_run_directory(output, run_id)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail="run 不存在") from error
        return list_depth_frames(run_dir, after=after, limit=limit)

    @app.get("/api/runs/{run_id}/depth-frames/{frame_index}.png", tags=["runs"])
    def depth_frame_preview(
        run_id: str,
        frame_index: int,
        minimum_m: float = Query(default=0.25, ge=0.0, le=65.535),
        maximum_m: float = Query(default=5.0, gt=0.0, le=65.535),
    ) -> Response:
        try:
            run_dir = resolve_run_directory(output, run_id)
            path = depth_frame_path(run_dir, frame_index)
            content, stats = render_depth_frame(
                path,
                minimum_depth_m=minimum_m,
                maximum_depth_m=maximum_m,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="深度帧不存在或尚未写完") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return Response(
            content=content,
            media_type="image/png",
            headers={
                "X-Depth-Frame": str(frame_index),
                "X-Depth-Min-M": f"{minimum_m:.4f}",
                "X-Depth-Max-M": f"{maximum_m:.4f}",
                "X-Depth-Valid-Ratio": f"{stats['valid_ratio']:.6f}",
                "X-Depth-Observed-Min-M": f"{stats['observed_minimum_m']:.4f}",
                "X-Depth-Observed-Max-M": f"{stats['observed_maximum_m']:.4f}",
            },
        )

    @app.post("/api/commands/preview", tags=["control"])
    def preview_command(body: dict[str, Any]) -> dict[str, Any]:
        try:
            workflow_id, preset_id, parameters = _body_fields(body)
            workflow = get_workflow(workflow_id)
            preview = build_command(
                workflow,
                repository_root=repository,
                output_root=output,
                preset_id=preset_id,
                supplied=parameters,
                strict=False,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except CommandValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return preview.to_dict()

    @app.post("/api/runs", tags=["control"], status_code=202)
    def start_run(body: dict[str, Any]) -> dict[str, Any]:
        try:
            workflow_id, preset_id, parameters = _body_fields(body)
            workflow = get_workflow(workflow_id)
            preview = build_command(
                workflow,
                repository_root=repository,
                output_root=output,
                preset_id=preset_id,
                supplied=parameters,
                strict=True,
            )
            managed = manager.start(preview)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except CommandValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        info = managed.info()
        return {
            "process_id": info["process_id"],
            "run_id": info["run_id"],
            "status": info["status"],
            "started_at": info["started_at"],
            "warnings": info["warnings"],
        }

    @app.get("/api/processes", tags=["control"])
    def processes() -> dict[str, Any]:
        return {"processes": manager.list()}

    @app.get("/api/processes/{process_id}", tags=["control"])
    def get_process(process_id: str) -> dict[str, Any]:
        try:
            return manager.get(process_id).info()
        except KeyError as error:
            raise HTTPException(status_code=404, detail="process 不存在") from error

    @app.get("/api/processes/{process_id}/events", tags=["control"])
    def process_events(
        process_id: str,
        after: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        try:
            managed = manager.get(process_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="process 不存在") from error
        events, next_cursor = managed.events_after(after)
        info = managed.info()
        return {
            "events": events,
            "next_cursor": next_cursor,
            "status": info["status"],
            "process": info,
        }

    @app.post("/api/processes/{process_id}/stop", tags=["control"])
    def stop_process(process_id: str) -> dict[str, Any]:
        try:
            managed = manager.get(process_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="process 不存在") from error
        if not managed.stop():
            raise HTTPException(status_code=409, detail="process 已结束或尚未启动")
        return managed.info()

    return app
