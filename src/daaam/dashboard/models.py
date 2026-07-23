"""Serializable declarations used by the mapping dashboard.

The dashboard deliberately keeps this layer dependency-free.  Workflow
declarations are consumed both by FastAPI and by unit tests without importing
the GPU mapping stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParameterDefinition:
    id: str
    label: str
    flag: str
    kind: str = "text"
    default: Any = None
    required: bool = False
    choices: tuple[str, ...] = ()
    help: str = ""
    placeholder: str = ""
    advanced: bool = False
    hard_gate: bool = False
    source: str = "CLI"
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParameterGroup:
    id: str
    label: str
    description: str
    parameters: tuple[ParameterDefinition, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }


@dataclass(frozen=True)
class WorkflowNode:
    id: str
    label: str
    category: str
    description: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    status_hint: str = ""
    gate: bool = False
    x: int = 0
    y: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowEdge:
    source: str
    target: str
    kind: str = "main"
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowPreset:
    id: str
    name: str
    description: str
    values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowDefinition:
    id: str
    name: str
    description: str
    runner: str | None
    run_dir_parameter: str | None
    nodes: tuple[WorkflowNode, ...]
    edges: tuple[WorkflowEdge, ...]
    parameter_groups: tuple[ParameterGroup, ...] = ()
    presets: tuple[WorkflowPreset, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def runnable(self) -> bool:
        return self.runner is not None

    @property
    def parameters(self) -> tuple[ParameterDefinition, ...]:
        return tuple(
            parameter
            for group in self.parameter_groups
            for parameter in group.parameters
        )

    def parameter(self, parameter_id: str) -> ParameterDefinition:
        for parameter in self.parameters:
            if parameter.id == parameter_id:
                return parameter
        raise KeyError(parameter_id)

    def preset(self, preset_id: str | None) -> WorkflowPreset | None:
        if not preset_id:
            return None
        for preset in self.presets:
            if preset.id == preset_id:
                return preset
        raise KeyError(preset_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "runner": self.runner,
            "runnable": self.runnable,
            "run_dir_parameter": self.run_dir_parameter,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "parameter_groups": [group.to_dict() for group in self.parameter_groups],
            "presets": [preset.to_dict() for preset in self.presets],
            "warnings": list(self.warnings),
        }
