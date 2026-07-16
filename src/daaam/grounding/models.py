from typing import List, Dict
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass
import numpy as np
from typing import Annotated, Optional
import time


class Annotation(BaseModel):
	"""Base class for annotations."""
	timestamp: float = Field(
		default_factory=time.time,
		description="Timestamp of the image in seconds.",
	)
	semantic_label: str = Field(..., description="Semantic label of the annotation.")
	confidence: Optional[float] = Field(default=0.0, description="Confidence of the label.", ge=0.0, le=10.0)
	embedding: Optional[List[float]] = Field(default=None, description="Embedding vector of the annotation.")

class ObjectAnnotation(Annotation):
	semantic_id: int = Field(
		..., description="Semantic ID of the object displayed in the image."
	)
	entity_id: Optional[str] = Field(
		default=None, description="Stable MapMemory entity addressed by this result."
	)
	request_id: Optional[str] = Field(
		default=None, description="Idempotent grounding request identifier."
	)
	sensor_time_ns: Optional[int] = Field(
		default=None, description="Absolute capture timestamp of the grounding evidence."
	)
	map_revision: Optional[int] = Field(
		default=None, description="Map revision used to create the grounding request."
	)
	source_model: Optional[str] = Field(
		default=None, description="Grounding model/version that produced the label."
	)
	# Temporal observation history fields
	frame_ids: Optional[List[int]] = Field(
		default=None, description="List of frame IDs where this object was observed."
	)
	timestamps: Optional[List[float]] = Field(
		default=None, description="List of timestamps when this object was observed."
	)
	observation_count: Optional[int] = Field(
		default=None, description="Total number of observations of this object."
	)
	first_observed: Optional[float] = Field(
		default=None, description="Timestamp of first observation."
	)
	last_observed: Optional[float] = Field(
		default=None, description="Timestamp of last observation."
	)
	selectframe_clip_feature: Optional[List[float]] = Field(
		default=None, description="CLIP feature vector for the object patch."
	)
	
class ImageAnnotation(Annotation):
	"""Classification for an entire image."""
	transform: Optional[List[float]] = Field(
		default=None,
		description="Transformation vector for the image (x, y, z, qx, qy, qz, qw).",
	)
