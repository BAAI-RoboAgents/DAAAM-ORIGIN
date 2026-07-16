"""Pipeline public API with heavy runtime components loaded on demand."""

from daaam.config import PipelineConfig

__all__ = ["PipelineOrchestrator", "PipelineConfig"]


def __getattr__(name: str):
    if name == "PipelineOrchestrator":
        from .orchestrator import PipelineOrchestrator

        return PipelineOrchestrator
    raise AttributeError(name)
