"""Base dataset interfaces for daaam."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class DatasetFrame:
	"""Container for a single frame from a dataset.
	
	Matches the Frame structure used by PipelineOrchestrator but without
	ROS dependencies.
	"""
	frame_id: int
	timestamp: float
	rgb_image: np.ndarray  # HxWx3 uint8
	depth_image: Optional[np.ndarray] = None  # HxW float32 in meters
	transform: Optional[np.ndarray] = None  # 7D pose [x, y, z, qx, qy, qz, qw]
	camera_info: Optional[Dict[str, Any]] = None  # intrinsics, distortion, etc.
	lin_vel: Optional[np.ndarray] = None  # [vx, vy, vz] 
	ang_vel: Optional[np.ndarray] = None  # [wx, wy, wz]
	timestamp_ns: Optional[int] = None  # Original capture time when available
	
	def to_pipeline_frame(self):
		"""Convert to pipeline Frame format."""
		from daaam.pipeline.models import Frame
		camera_intrinsics = None
		if self.camera_info and 'intrinsics' in self.camera_info:
			K = self.camera_info['intrinsics']
			if hasattr(K, '__getitem__'):
				camera_intrinsics = {
					'fx': float(K[0][0]), 'fy': float(K[1][1]),
					'cx': float(K[0][2]), 'cy': float(K[1][2]),
				}
		return Frame(
			frame_id=self.frame_id,
			timestamp=self.timestamp,
			rgb_image=self.rgb_image,
			timestamp_ns=self.timestamp_ns,
			depth_image=self.depth_image,
			transform=self.transform,
			lin_vel=self.lin_vel if self.lin_vel is not None else np.zeros(3),
			ang_vel=self.ang_vel if self.ang_vel is not None else np.zeros(3),
			camera_intrinsics=camera_intrinsics,
		)


class BaseDataset(ABC):
	"""Abstract base class for datasets."""
	
	def __init__(self, data_path: Path, config: Optional[Dict[str, Any]] = None):
		"""Initialize dataset.
		
		Args:
			data_path: Path to dataset root directory or file
			config: Optional configuration dictionary
		"""
		self.data_path = Path(data_path)
		self.config = config or {}
		self._current_idx = 0
		
	@abstractmethod
	def __len__(self) -> int:
		"""Return number of frames in dataset."""
		pass
		
	@abstractmethod
	def __getitem__(self, idx: int) -> DatasetFrame:
		"""Get frame by index."""
		pass
		
	def __iter__(self):
		"""Iterate over dataset frames."""
		self._current_idx = 0
		return self
		
	def __next__(self) -> DatasetFrame:
		"""Get next frame."""
		if self._current_idx >= len(self):
			raise StopIteration
		frame = self[self._current_idx]
		self._current_idx += 1
		return frame
		
	@abstractmethod
	def get_camera_info(self) -> Dict[str, Any]:
		"""Get camera calibration information.
		
		Returns:
			Dictionary containing:
			- intrinsics: 3x3 camera matrix
			- distortion: distortion coefficients
			- width: image width
			- height: image height
		"""
		pass
		
	@property
	@abstractmethod
	def fps(self) -> float:
		"""Return dataset framerate."""
		pass
		
	def get_frame_range(self, start: int, end: int) -> list[DatasetFrame]:
		"""Get range of frames.
		
		Args:
			start: Start frame index
			end: End frame index (exclusive)
			
		Returns:
			List of DatasetFrame objects
		"""
		return [self[i] for i in range(start, min(end, len(self)))]
		
	def validate(self) -> bool:
		"""Validate dataset integrity.
		
		Returns:
			True if dataset is valid, False otherwise
		"""
		try:
			# Check if dataset path exists
			if not self.data_path.exists():
				return False
				
			# Try to load first frame
			if len(self) > 0:
				frame = self[0]
				assert frame.rgb_image is not None
				assert frame.rgb_image.shape[2] == 3
				
			return True
		except Exception:
			return False
