"""Validated command construction for dashboard-managed mapping runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping

from .models import ParameterDefinition, WorkflowDefinition


class CommandValidationError(ValueError):
    """Raised when dashboard parameters cannot safely form a runner command."""


@dataclass(frozen=True)
class CommandPreview:
    workflow_id: str
    argv: tuple[str, ...]
    command: str
    parameters: dict[str, Any]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "argv": list(self.argv),
            "command": self.command,
            "parameters": self.parameters,
            "warnings": list(self.warnings),
        }


def _as_bool(value: Any, parameter: ParameterDefinition) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise CommandValidationError(f"{parameter.label} 必须是布尔值")


def _coerce(parameter: ParameterDefinition, value: Any) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        if parameter.required:
            raise CommandValidationError(f"缺少必填参数：{parameter.label}")
        return None
    try:
        if parameter.kind == "boolean":
            output = _as_bool(value, parameter)
        elif parameter.kind == "integer":
            if isinstance(value, bool):
                raise ValueError
            output = int(value)
        elif parameter.kind == "number":
            if isinstance(value, bool):
                raise ValueError
            output = float(value)
        else:
            output = str(value).strip()
    except (TypeError, ValueError) as error:
        raise CommandValidationError(f"{parameter.label} 的值无效：{value!r}") from error

    if parameter.choices and output not in parameter.choices:
        choices = "、".join(parameter.choices)
        raise CommandValidationError(f"{parameter.label} 必须是：{choices}")
    if parameter.minimum is not None and output < parameter.minimum:
        raise CommandValidationError(
            f"{parameter.label} 不能小于 {parameter.minimum}"
        )
    if parameter.maximum is not None and output > parameter.maximum:
        raise CommandValidationError(
            f"{parameter.label} 不能大于 {parameter.maximum}"
        )
    return output


def resolve_parameters(
    workflow: WorkflowDefinition,
    preset_id: str | None,
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge defaults, a named preset and explicit UI values, then coerce types."""

    supplied = supplied or {}
    known = {parameter.id for parameter in workflow.parameters}
    unknown = sorted(set(supplied) - known)
    if unknown:
        raise CommandValidationError("未知参数：" + "、".join(unknown))

    values = {parameter.id: parameter.default for parameter in workflow.parameters}
    preset = workflow.preset(preset_id)
    if preset is not None:
        values.update(preset.values)
    values.update(supplied)

    return {
        parameter.id: _coerce(parameter, values.get(parameter.id))
        for parameter in workflow.parameters
    }


def _resolve_path(value: str, repository_root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return str(path.resolve())


def _contains_path(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _workflow_warnings(
    workflow: WorkflowDefinition, values: Mapping[str, Any]
) -> list[str]:
    warnings = list(workflow.warnings)
    if workflow.id == "offline_hq":
        stop_after = str(values.get("stop_after") or "map")
        adapter = values.get("adapter")
        depth_needed = stop_after not in {"prepare", "select"}
        if adapter == "g1-fisheye" and not values.get("stereo_calibration_report"):
            warnings.append("G1 正式运行缺少固定双目标定报告。")
        if depth_needed and not values.get("checkpoint"):
            warnings.append("FoundationStereo checkpoint 尚未设置。")
        if depth_needed and not values.get("accept_license"):
            warnings.append("尚未显式确认 FoundationStereo 研究/非商用许可证。")
        if adapter == "g1-fisheye" and stop_after not in {"prepare", "select", "depth"} and not values.get("floor_calibration_report"):
            warnings.append("G1 深度之后的阶段缺少固定地面标定报告。")
        if stop_after == "map" and not values.get("accept_preview"):
            warnings.append("运行到 map 前必须由用户显式接受直接 RGB-D 融合预览。")
    elif workflow.id == "realtime_semantic":
        if values.get("semantic_mode") != "disabled" and values.get("stop_after") != "global":
            warnings.append("完整语义 Hydra 需要 --stop-after global。")
        if values.get("semantic_mode") == "dam" and values.get("static_map_backend") != "hydra":
            warnings.append("DAM 严格 mesh 绑定与 commit 需要 Hydra 静态地图后端。")
        if values.get("depth_backend") == "foundation-worker" and not values.get("checkpoint"):
            warnings.append("Foundation worker 模式必须提供 checkpoint。")
        config = str(values.get("semantic_config") or "")
        tabletop = "tabletop" in config
        generic_distances = (
            values.get("entity_merge_distance_m") == 0.50
            or values.get("binding_center_distance_m") == 0.75
            or values.get("binding_aabb_gap_m") == 0.15
        )
        if tabletop and generic_distances:
            warnings.append("桌面配置仍含通用实体合并/mesh 绑定距离；请使用桌面预设的 0.075/0.10/0.025 m。")
    return warnings


def _strict_gate_errors(
    workflow: WorkflowDefinition, values: Mapping[str, Any]
) -> list[str]:
    """Return only pre-run gates that can be decided without executing models."""

    if values.get("dry_run"):
        return []
    errors: list[str] = []
    if workflow.id == "offline_hq":
        stop_after = str(values.get("stop_after") or "map")
        adapter = values.get("adapter")
        depth_needed = stop_after not in {"prepare", "select"}
        if depth_needed and not values.get("checkpoint"):
            errors.append("真实深度运行必须提供 FoundationStereo checkpoint")
        if depth_needed and not values.get("accept_license"):
            errors.append("真实深度运行必须显式确认 FoundationStereo 研究/非商用许可证")
        if adapter == "g1-fisheye" and not values.get("stereo_calibration_report"):
            errors.append("G1 运行必须提供固定双目标定报告")
        if adapter == "g1-fisheye" and stop_after not in {"prepare", "select", "depth"} and not values.get("floor_calibration_report"):
            errors.append("G1 深度之后的阶段必须提供固定地面标定报告")
        if stop_after == "map" and not values.get("accept_preview"):
            errors.append("运行 map 前必须显式接受直接 RGB-D 融合预览")
    elif workflow.id == "realtime_semantic":
        if values.get("depth_backend") == "foundation-worker" and not values.get("checkpoint"):
            errors.append("Foundation worker 模式必须提供 checkpoint")
        if values.get("semantic_mode") != "disabled" and values.get("stop_after") != "global":
            errors.append("完整语义路径必须运行到 global")
        if values.get("semantic_mode") == "dam" and values.get("static_map_backend") != "hydra":
            errors.append("DAM 严格提交必须选择 Hydra 静态地图后端")
    return errors


def _strict_path_errors(
    workflow: WorkflowDefinition, values: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    primary_id = "src" if workflow.id == "offline_hq" else "dataset"
    primary = values.get(primary_id)
    if primary and not Path(str(primary)).exists():
        errors.append(f"输入路径不存在：{primary}")
    for parameter_id in (
        "checkpoint",
        "stereo_calibration_report",
        "floor_calibration_report",
        "foundation_stereo_python",
    ):
        value = values.get(parameter_id)
        if value and not Path(str(value)).exists():
            errors.append(f"{parameter_id} 路径不存在：{value}")
    return errors


def build_command(
    workflow: WorkflowDefinition,
    *,
    repository_root: Path,
    output_root: Path,
    preset_id: str | None = None,
    supplied: Mapping[str, Any] | None = None,
    strict: bool = False,
    python_executable: str | Path | None = None,
) -> CommandPreview:
    """Build an argv-only command.  No value is ever evaluated by a shell."""

    if not workflow.runnable or not workflow.runner:
        raise CommandValidationError(f"工作流 {workflow.name} 当前为只读流程")
    repository_root = repository_root.resolve()
    output_root = output_root.resolve()
    runner = (repository_root / workflow.runner).resolve()
    scripts_root = (repository_root / "scripts").resolve()
    if not _contains_path(runner, scripts_root) or not runner.is_file():
        raise CommandValidationError("工作流 runner 不在允许的 scripts 目录中")

    values = resolve_parameters(workflow, preset_id, supplied)
    for parameter in workflow.parameters:
        value = values[parameter.id]
        if parameter.kind == "path" and value is not None:
            values[parameter.id] = _resolve_path(value, repository_root)

    if workflow.run_dir_parameter:
        run_dir_value = values.get(workflow.run_dir_parameter)
        if not run_dir_value:
            raise CommandValidationError("缺少运行目录")
        run_dir = Path(run_dir_value).resolve()
        if not _contains_path(run_dir, output_root):
            raise CommandValidationError(
                f"运行目录必须位于允许的输出根目录：{output_root}"
            )

    validation_errors = []
    if strict:
        validation_errors.extend(_strict_gate_errors(workflow, values))
        validation_errors.extend(_strict_path_errors(workflow, values))
    if validation_errors:
        raise CommandValidationError("；".join(validation_errors))

    executable = str(python_executable or sys.executable)
    argv: list[str] = [executable, str(runner)]
    for parameter in workflow.parameters:
        value = values[parameter.id]
        if parameter.kind == "boolean":
            if value:
                argv.append(parameter.flag)
        elif value is not None:
            argv.extend((parameter.flag, str(value)))

    warnings = tuple(dict.fromkeys(_workflow_warnings(workflow, values)))
    return CommandPreview(
        workflow.id,
        tuple(argv),
        shlex.join(argv),
        values,
        warnings,
    )
