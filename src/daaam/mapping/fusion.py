"""Auditable static-fusion input preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class StaticFusionInput:
    depth_m: np.ndarray
    confidence: np.ndarray
    excluded_mask: np.ndarray
    metrics: dict[str, float | int]


def isolate_static_depth(
    depth_m: np.ndarray,
    dynamic_mask: np.ndarray,
    unknown_mask: np.ndarray,
    *,
    confidence: Optional[np.ndarray] = None,
    minimum_confidence: float = 0.0,
) -> StaticFusionInput:
    """Remove dynamic, unknown, and low-confidence pixels before static TSDF."""
    depth = np.asarray(depth_m, dtype=np.float32)
    dynamic = np.asarray(dynamic_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    if depth.ndim != 2 or dynamic.shape != depth.shape or unknown.shape != depth.shape:
        raise ValueError("depth and masks must have matching HxW shapes")
    if minimum_confidence < 0.0 or minimum_confidence > 1.0:
        raise ValueError("minimum_confidence must be in [0, 1]")
    if confidence is None:
        weights = np.ones(depth.shape, dtype=np.float32)
    else:
        weights = np.asarray(confidence, dtype=np.float32).copy()
        if weights.shape != depth.shape:
            raise ValueError("confidence shape does not match depth")
        weights[~np.isfinite(weights)] = 0.0
        np.clip(weights, 0.0, 1.0, out=weights)

    valid_input = np.isfinite(depth) & (depth > 0.0)
    excluded = dynamic | unknown | ~valid_input | (weights < minimum_confidence)
    output_depth = depth.copy()
    output_depth[excluded] = 0.0
    weights[excluded] = 0.0
    dynamic_valid = dynamic & valid_input
    leakage = int(np.count_nonzero(dynamic_valid & (output_depth > 0.0)))
    metrics: dict[str, float | int] = {
        "pixels": int(depth.size),
        "input_valid_pixels": int(valid_input.sum()),
        "output_valid_pixels": int(np.count_nonzero(output_depth > 0.0)),
        "dynamic_pixels": int(dynamic.sum()),
        "unknown_pixels": int(unknown.sum()),
        "excluded_pixels": int(excluded.sum()),
        "excluded_valid_ratio": float(
            np.count_nonzero(excluded & valid_input) / max(1, int(valid_input.sum()))
        ),
        "dynamic_leakage_pixels": leakage,
        "dynamic_contamination_rate": float(
            leakage / max(1, int(dynamic_valid.sum()))
        ),
    }
    return StaticFusionInput(output_depth, weights, excluded, metrics)
