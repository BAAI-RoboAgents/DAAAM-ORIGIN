"""Data models for the pipeline orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Dict, Any, Optional

try:
	from pydantic import BaseModel
except ImportError:
	class BaseModel:
		"""Deferred error for semantic models when pydantic is not installed."""

		def __init__(self, *_args, **_kwargs):
			raise RuntimeError(
				"pydantic is required for semantic pipeline messages; "
				"install project runtime dependencies"
			)

import numpy as np
import time

if TYPE_CHECKING:
	from daaam.tracking.models import Track

@dataclass
class Frame:
	"""Data model for robotics frame observation."""

	# core data
	frame_id: int
	timestamp: float
	rgb_image: np.ndarray  # RGB image
	depth_image: Optional[np.ndarray] = None  # Depth image (optional)
	transform: Optional[np.ndarray] = None  # [x, y, z, qx, qy, qz, qw] (optional, tf2 can fail)

	# derived data (computed from transform history)
	lin_vel: Optional[np.ndarray] = field(default_factory=lambda: np.zeros(3))  # linear velocity [vx, vy, vz]
	ang_vel: Optional[np.ndarray] = field(default_factory=lambda: np.zeros(3))  # angular velocity [wx, wy, wz]

	# camera calibration
	camera_intrinsics: Optional[Dict[str, float]] = None  # {'fx', 'fy', 'cx', 'cy'}

	# updated data (filled by orchestrator during processing)
	tracks: List[Track] = field(default_factory=list)  # List of Track objects
	timestamp_ns: Optional[int] = None  # Original capture clock, if available


@dataclass
class PromptRecord:
	"""Model for communication between orchestrator and grounding workers."""
	frame: np.ndarray  # RGB image frame
	tracks: List[Track]  # List of Track objects for grounding
	object_labels: Dict[int, int]  # Mapping from track_id to semantic_id
	frame_id: int = -1  # Frame identifier, default -1 if unknown
	timestamp: float = 0.0  # Observation timestamp in seconds
	sensor_time_ns: int = 0  # Absolute capture timestamp for versioned delivery
	map_revision: int = 0
	request_id: str = ""
	entity_ids: Dict[int, str] = field(default_factory=dict)  # track_id -> entity_id


class MinimalCorrection(BaseModel):
	"""Minimal correction data for assignment workers (without embeddings)."""
	semantic_id: int
	semantic_label: str
	confidence: float
	task_relevance: Optional[List[str]] = None


class TemporalObservation(BaseModel):
	"""Temporal observation data for a semantic entity."""
	frame_ids: List[int]
	timestamps: List[float]
	observation_count: int
	first_observed: Optional[float] = None
	last_observed: Optional[float] = None


class SemanticFeatures(BaseModel):
	"""Feature vectors for a semantic entity."""
	clip_feature: Optional[List[float]] = None
	semantic_embedding_feature: Optional[List[float]] = None


class SemanticUpdate(BaseModel):
	"""Incremental semantic update message for publishing."""
	timestamp: float
	semantic_labels: Dict[int, str]  # semantic_id -> label
	temporal_observations: Dict[int, TemporalObservation]  # semantic_id -> temporal data
	features: Dict[int, SemanticFeatures]  # semantic_id -> features
