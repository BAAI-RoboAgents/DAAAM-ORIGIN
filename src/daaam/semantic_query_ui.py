"""Independent semantic-map query service and browser interface."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator


class SemanticQueryUIError(RuntimeError):
    """Raised when a semantic map or one of its query assets is invalid."""


class SemanticMapOpenRequest(BaseModel):
    """Request to load one query-ready semantic-map output directory."""

    run_path: str = Field(..., min_length=1)

    @field_validator("run_path")
    @classmethod
    def normalize_run_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("语义地图路径不能为空")
        return value


class SemanticQueryRequest(SemanticMapOpenRequest):
    """Local multilingual semantic retrieval requested by the query UI."""

    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)
    require_mesh: bool = False
    min_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    min_margin: float | None = Field(default=None, ge=0.0, le=2.0)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("查询内容不能为空")
        return value


@dataclass(frozen=True)
class ResolvedSemanticMap:
    run_directory: Path
    dsg_path: Path
    relative_run: str


def _default_engine_factory(dsg_path: Path):
    from daaam.semantic_query import SemanticQueryEngine

    return SemanticQueryEngine(dsg_path)


def _artifact_url(run_path: Path, relative_path: Path | str) -> str:
    return "/api/file?" + urlencode(
        {"run_path": str(run_path), "path": str(relative_path)}
    )


class SemanticMapService:
    """Validate, cache, query, and export maps below one output root."""

    def __init__(
        self,
        output_root: Path | str,
        *,
        engine_factory: Callable[[Path], Any] | None = None,
        maximum_cached_maps: int = 2,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.engine_factory = engine_factory or _default_engine_factory
        self.maximum_cached_maps = max(1, int(maximum_cached_maps))
        self._engines: OrderedDict[
            Path, tuple[tuple[tuple[str, int, int], ...], Any]
        ] = OrderedDict()
        self._preview_cache: dict[
            Path, tuple[tuple[tuple[str, int, int], ...], bytes]
        ] = {}
        self._lock = RLock()

    @property
    def cached_map_count(self) -> int:
        with self._lock:
            return len(self._engines)

    def resolve(self, value: str | Path) -> ResolvedSemanticMap:
        supplied = Path(value).expanduser()
        candidate = supplied if supplied.is_absolute() else self.output_root / supplied
        candidate = candidate.resolve()
        dsg_path = candidate if candidate.is_file() else candidate / "dsg_updated.json"
        run_directory = dsg_path.parent
        try:
            relative_run = run_directory.relative_to(self.output_root)
        except ValueError as exc:
            raise SemanticQueryUIError(
                f"语义地图必须位于输出根目录 {self.output_root} 内"
            ) from exc
        if not relative_run.parts:
            raise SemanticQueryUIError("请选择一个具体的语义地图运行目录")
        if dsg_path.name != "dsg_updated.json" or not dsg_path.is_file():
            raise SemanticQueryUIError(
                f"目录中没有查询就绪的 dsg_updated.json：{run_directory}"
            )
        return ResolvedSemanticMap(
            run_directory=run_directory,
            dsg_path=dsg_path,
            relative_run=relative_run.as_posix(),
        )

    @staticmethod
    def _signature(dsg_path: Path) -> tuple[tuple[str, int, int], ...]:
        paths = [
            dsg_path,
            dsg_path.with_suffix(".manifest.json"),
            dsg_path.with_suffix(".semantic.json"),
            dsg_path.with_suffix(".evidence.json"),
        ]
        signature: list[tuple[str, int, int]] = []
        for path in paths:
            if path.is_file():
                stat = path.stat()
                signature.append((path.name, stat.st_mtime_ns, stat.st_size))
        return tuple(signature)

    def engine_for(self, semantic_map: ResolvedSemanticMap):
        signature = self._signature(semantic_map.dsg_path)
        with self._lock:
            cached = self._engines.get(semantic_map.dsg_path)
            if cached is not None and cached[0] == signature:
                self._engines.move_to_end(semantic_map.dsg_path)
                return cached[1]
            try:
                engine = self.engine_factory(semantic_map.dsg_path)
            except Exception as exc:
                raise SemanticQueryUIError(f"语义查询地图加载失败：{exc}") from exc
            self._engines[semantic_map.dsg_path] = (signature, engine)
            self._engines.move_to_end(semantic_map.dsg_path)
            while len(self._engines) > self.maximum_cached_maps:
                self._engines.popitem(last=False)
            return engine

    def open_map(self, value: str | Path) -> dict[str, Any]:
        semantic_map = self.resolve(value)
        engine = self.engine_for(semantic_map)
        queryable = len(engine.records)
        evidence_count = len(engine.evidence_by_node)
        return {
            "run_path": str(semantic_map.run_directory),
            "run_name": semantic_map.run_directory.name,
            "relative_run": semantic_map.relative_run,
            "dsg_path": str(semantic_map.dsg_path),
            "queryable_objects": queryable,
            "geometry_counts": dict(engine.geometry_counts),
            "evidence_available_objects": evidence_count,
            "evidence_missing_objects": max(0, queryable - evidence_count),
            "evidence_coverage": 0.0 if not queryable else evidence_count / queryable,
            "embedding_dimension": int(engine.embedding_dim),
            "sentence_model": engine.sentence_model_name,
            "encoder_device": engine.encoder_device,
            "min_similarity": float(engine.min_similarity),
            "min_margin": float(engine.min_margin),
            "mesh_preview_url": "/api/map/mesh-preview.png?"
            + urlencode({"run_path": str(semantic_map.run_directory)}),
        }

    def discover(self) -> list[dict[str, Any]]:
        """List first-level query-ready maps without loading embedding models."""

        maps: list[dict[str, Any]] = []
        if not self.output_root.is_dir():
            return maps
        for run_directory in self.output_root.iterdir():
            dsg_path = run_directory / "dsg_updated.json"
            if not run_directory.is_dir() or not dsg_path.is_file():
                continue
            stat = dsg_path.stat()
            summary: dict[str, Any] = {
                "run_name": run_directory.name,
                "run_path": str(run_directory.resolve()),
                "updated_at_ns": stat.st_mtime_ns,
                "size_bytes": stat.st_size,
            }
            manifest_path = dsg_path.with_suffix(".manifest.json")
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    manifest = {}
                if isinstance(manifest, dict):
                    summary["queryable_objects"] = manifest.get("queryable_objects")
                    summary["geometry_counts"] = manifest.get("geometry_counts")
            maps.append(summary)
        return sorted(maps, key=lambda item: item["updated_at_ns"], reverse=True)

    def mesh_preview(self, value: str | Path) -> bytes:
        from daaam.query_visualization import render_mesh_topdown_preview

        semantic_map = self.resolve(value)
        signature = self._signature(semantic_map.dsg_path)
        with self._lock:
            cached = self._preview_cache.get(semantic_map.dsg_path)
            if cached is not None and cached[0] == signature:
                return cached[1]
        try:
            preview = render_mesh_topdown_preview(semantic_map.dsg_path)
        except Exception as exc:
            raise SemanticQueryUIError(f"mesh 俯视图生成失败：{exc}") from exc
        with self._lock:
            self._preview_cache[semantic_map.dsg_path] = (signature, preview)
        return preview

    def query(self, request: SemanticQueryRequest) -> dict[str, Any]:
        from daaam.query_visualization import write_query_visuals
        from daaam.semantic_query import results_to_dicts

        semantic_map = self.resolve(request.run_path)
        engine = self.engine_for(semantic_map)
        try:
            decision = engine.retrieve_with_decision(
                request.query,
                request.top_k,
                min_similarity=request.min_similarity,
                min_margin=request.min_margin,
                require_mesh=request.require_mesh,
            )
            artifacts = write_query_visuals(
                engine=engine,
                query=request.query,
                matches=decision.matches,
                output_root=semantic_map.run_directory / "query_results",
                found=decision.found,
                rejection_reason=decision.rejection_reason,
                top_score=decision.top_score,
                top1_margin=decision.top1_margin,
            )
        except Exception as exc:
            raise SemanticQueryUIError(f"语义查询失败：{exc}") from exc

        report_payload = json.loads(artifacts.report.read_text(encoding="utf-8"))
        report_by_node = {
            item["node_id"]: item for item in report_payload.get("matches", [])
        }
        matches = results_to_dicts(decision.matches)
        for item in matches:
            node_id = item["node_id"]
            exported = report_by_node.get(node_id, {})
            evidence = engine.evidence_for_node(node_id)
            evidence_image = exported.get("evidence_image")
            if evidence is None or not evidence_image:
                item["evidence"] = None
                continue
            relative_image = artifacts.output_directory.relative_to(
                semantic_map.run_directory
            ) / str(evidence_image)
            item["evidence"] = {
                "image_url": _artifact_url(
                    semantic_map.run_directory, relative_image
                ),
                "frame_index": int(evidence.frame_index),
                "observed_s": evidence.observed_s,
                "bbox_xyxy": list(evidence.bbox_xyxy),
                "mask_pixels": int(evidence.mask_pixels),
                "mask_source": evidence.mask_source,
                "camera_position_m": (
                    None
                    if evidence.camera_position_m is None
                    else list(evidence.camera_position_m)
                ),
                "image_sha256": evidence.image_sha256,
            }

        relative_output = artifacts.output_directory.relative_to(
            semantic_map.run_directory
        )
        return {
            "query": request.query,
            "found": bool(decision.found),
            "rejection_reason": decision.rejection_reason,
            "top_score": float(decision.top_score),
            "top1_margin": decision.top1_margin,
            "min_similarity": float(decision.min_similarity),
            "min_margin": float(decision.min_margin),
            "require_mesh": request.require_mesh,
            "matches": matches,
            "topdown_image_url": _artifact_url(
                semantic_map.run_directory,
                relative_output / artifacts.topdown_image.name,
            ),
            "report_url": _artifact_url(
                semantic_map.run_directory,
                relative_output / artifacts.report.name,
            ),
            "output_directory": str(artifacts.output_directory),
        }

    def artifact(self, value: str | Path, relative_value: str | Path) -> Path:
        semantic_map = self.resolve(value)
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise SemanticQueryUIError("查询产物路径越界")
        requested = (semantic_map.run_directory / relative).resolve()
        try:
            requested.relative_to(semantic_map.run_directory)
        except ValueError as exc:
            raise SemanticQueryUIError("查询产物路径越界") from exc
        if not requested.is_file():
            raise FileNotFoundError(requested)
        if requested.suffix.lower() not in {".png", ".jpg", ".jpeg", ".json"}:
            raise SemanticQueryUIError("该查询产物类型不允许读取")
        return requested


def create_semantic_query_app(
    output_root: Path | str,
    *,
    semantic_map_service: SemanticMapService | None = None,
) -> FastAPI:
    """Create the standalone, read-mostly semantic-query web application."""

    output = Path(output_root).expanduser().resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"output root does not exist: {output}")
    static = Path(__file__).resolve().parent / "query_static"
    service = semantic_map_service or SemanticMapService(output)
    app = FastAPI(
        title="DAAAM Semantic Map Query",
        version="1.0.0",
        description=(
            "Independent local semantic-map retrieval UI with checksum-bound image "
            "evidence and mesh top-view visualization."
        ),
    )
    app.state.output_root = output
    app.state.semantic_map_service = service
    app.mount("/static", StaticFiles(directory=static), name="query-assets")

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

    @app.get("/api/health", tags=["service"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "semantic-query-ui",
            "output_root": str(output),
            "cached_maps": service.cached_map_count,
        }

    @app.get("/api/maps", tags=["maps"])
    def discover_maps() -> dict[str, Any]:
        return {"maps": service.discover()}

    @app.post("/api/map/open", tags=["maps"])
    def open_semantic_map(body: SemanticMapOpenRequest) -> dict[str, Any]:
        try:
            return service.open_map(body.run_path)
        except SemanticQueryUIError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get(
        "/api/map/mesh-preview.png",
        response_class=Response,
        tags=["maps"],
    )
    def semantic_mesh_preview(run_path: str = Query(...)) -> Response:
        try:
            content = service.mesh_preview(run_path)
        except SemanticQueryUIError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return Response(content=content, media_type="image/png")

    @app.post("/api/query", tags=["query"])
    def query_semantic_map(body: SemanticQueryRequest) -> dict[str, Any]:
        try:
            return service.query(body)
        except SemanticQueryUIError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/file", response_class=FileResponse, tags=["artifacts"])
    def semantic_query_artifact(
        run_path: str = Query(...), path: str = Query(...)
    ) -> FileResponse:
        try:
            requested = service.artifact(run_path, path)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="查询产物不存在") from error
        except SemanticQueryUIError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        return FileResponse(requested)

    return app
