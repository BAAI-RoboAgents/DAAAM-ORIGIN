"""Stereo confidence and conservative non-keyframe depth propagation."""

from .confidence import (
    StereoConfidence,
    compute_left_right_confidence,
    disparity_to_metric_depth,
)
from .propagation import (
    DepthPropagationConfig,
    PropagatedDepth,
    propagate_depth,
)
from .worker import DepthBackendError, SubprocessDepthBackend

__all__ = [
    "DepthPropagationConfig",
    "DepthBackendError",
    "PropagatedDepth",
    "StereoConfidence",
    "SubprocessDepthBackend",
    "compute_left_right_confidence",
    "disparity_to_metric_depth",
    "propagate_depth",
]
