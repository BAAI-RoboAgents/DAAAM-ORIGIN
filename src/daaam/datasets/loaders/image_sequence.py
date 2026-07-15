"""Image sequence dataset loader for folder-based RGB-D data."""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import cv2
from collections import deque

from daaam.datasets.interfaces import BaseDataset, DatasetFrame
from daaam.utils.transform import compute_ema_velocity

class ImageSequenceDataset(BaseDataset):
	"""Dataset loader for folder-based image sequences.
	
	Expected folder structure:
		data_path/
			rgb/          # RGB images (*.jpg, *.png)
			depth/        # Depth images (*.png, *.npy, *.exr)
			pose/         # Pose files (*.txt, *.json) or single poses.txt
			camera_info.json  # Camera calibration info
			
	Pose format can be:
		- Individual files per frame: 0000.txt, 0001.txt, etc.
		- Single poses.txt with one pose per line
		- JSON files with pose data
		
	Each pose is 7D: [x, y, z, qx, qy, qz, qw]
	"""
	
	def __init__(
		self, 
		data_path: Path,
		config: Optional[Dict[str, Any]] = None,
		depth_scale: float = 1.0,
		compute_velocities: bool = True,
		velocity_window: int = 10,
		velocity_alpha: float = 0.4
	):
		"""Initialize image sequence dataset.
		
		Args:
			data_path: Path to dataset root directory
			config: Optional configuration dictionary
			depth_scale: Scale factor to convert depth to meters
			compute_velocities: Whether to compute velocities from pose history
			velocity_window: Window size for velocity computation
			velocity_alpha: EMA alpha for velocity smoothing
		"""
		super().__init__(data_path, config)
		
		self.depth_scale = depth_scale
		self.compute_velocities = compute_velocities
		self.velocity_window = velocity_window
		self.velocity_alpha = velocity_alpha
		
		# Validate directory structure
		self.rgb_dir = self.data_path / "rgb"
		self.depth_dir = self.data_path / "depth"
		self.pose_dir = self.data_path / "pose"
		
		if not self.rgb_dir.exists():
			raise ValueError(f"RGB directory not found: {self.rgb_dir}")
			
		# Load file lists
		self._load_file_lists()
		self._load_timestamps()
		
		# Load camera info if available
		self._load_camera_info()
		
		# For velocity computation
		self.transforms_history = deque(maxlen=velocity_window)
		self.timestamps_history = deque(maxlen=velocity_window)
		
	def _load_file_lists(self):
		"""Load and sort file lists for each modality."""
		# RGB files
		rgb_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
		self.rgb_files = sorted([
			f for f in self.rgb_dir.iterdir() 
			if f.suffix.lower() in rgb_extensions
		])
		
		if not self.rgb_files:
			raise ValueError(f"No RGB images found in {self.rgb_dir}")
			
		# Depth files (optional)
		self.depth_files = []
		if self.depth_dir.exists():
			depth_extensions = [".png", ".npy", ".exr", ".tiff"]
			self.depth_files = sorted([
				f for f in self.depth_dir.iterdir() 
				if f.suffix.lower() in depth_extensions
			])
			
		# Pose files (optional)
		self.poses = None
		self.pose_timestamps_ns = None
		if self.pose_dir.exists():
			self._load_poses()
			
	def _load_poses(self):
		"""Load pose data from files."""
		# Check for single poses.txt file
		poses_file = self.pose_dir / "poses.txt"
		if poses_file.exists():
			self.poses = self._load_poses_from_txt(poses_file)
			self._load_pose_timestamps()
		else:
			# Load individual pose files
			pose_files = sorted([
				f for f in self.pose_dir.iterdir() 
				if f.suffix in [".txt", ".json"]
			])
			if pose_files:
				self.poses = self._load_individual_poses(pose_files)
				self._load_pose_timestamps()

	def _load_pose_timestamps(self):
		"""Load the optional absolute timestamp corresponding to each pose row."""
		self.pose_timestamps_ns = None
		path = self.pose_dir / "pose_timestamps_ns.txt"
		if not path.exists():
			return
		values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
		try:
			timestamps = [int(value) for value in values]
		except ValueError as error:
			raise ValueError(f"Invalid pose timestamp in {path}") from error
		if self.poses is not None and len(timestamps) != len(self.poses):
			raise ValueError(
				f"Pose timestamp count mismatch: {len(timestamps)} != {len(self.poses)}"
			)
		if len(timestamps) > 1 and any(
			second <= first for first, second in zip(timestamps, timestamps[1:])
		):
			raise ValueError(f"Pose timestamps must be strictly increasing: {path}")
		self.pose_timestamps_ns = timestamps

	def _load_timestamps(self):
		"""Load real frame timestamps from tick_index.json when available."""
		self.timestamps = None
		self.timestamps_ns = None
		self.time_origin_ns = None
		tick_index_path = self.data_path / "tick_index.json"
		if not tick_index_path.exists():
			return
		with open(tick_index_path, "r") as f:
			tick_index = json.load(f)
		frames = tick_index.get("frames", [])
		if not frames:
			return
		has_absolute_time = all("sensor_time_ns" in frame for frame in frames)
		if not has_absolute_time and not all("timestamp" in frame for frame in frames):
			return
		if len(frames) != len(self.rgb_files):
			raise ValueError(
				f"Timestamp count mismatch: {len(frames)} != {len(self.rgb_files)}"
			)
		if has_absolute_time:
			try:
				absolute_timestamps = [int(frame["sensor_time_ns"]) for frame in frames]
				time_origin_ns = int(
					tick_index.get("time_origin_ns", absolute_timestamps[0])
				)
			except (TypeError, ValueError) as error:
				raise ValueError("Invalid sensor_time_ns metadata") from error
			if len(absolute_timestamps) > 1 and any(
				second <= first
				for first, second in zip(absolute_timestamps, absolute_timestamps[1:])
			):
				raise ValueError("sensor_time_ns values must be strictly increasing")
			self.timestamps_ns = absolute_timestamps
			self.time_origin_ns = time_origin_ns
			self.timestamps = [
				(timestamp - time_origin_ns) / 1.0e9
				for timestamp in absolute_timestamps
			]
			for index, frame in enumerate(frames):
				if "timestamp" in frame and not np.isclose(
					float(frame["timestamp"]), self.timestamps[index], atol=1.0e-6
				):
					raise ValueError(
						f"Frame timestamp mismatch at {index}: absolute and relative clocks disagree"
					)
				if self.pose_timestamps_ns is not None and "pose_sensor_time_ns" in frame:
					pose_row = int(frame.get("pose_row", index))
					if pose_row < 0 or pose_row >= len(self.pose_timestamps_ns):
						raise ValueError(f"Invalid pose_row at frame {index}: {pose_row}")
					if int(frame["pose_sensor_time_ns"]) != self.pose_timestamps_ns[pose_row]:
						raise ValueError(
							f"Pose/image absolute time mismatch at frame {index}"
						)
			return
		self.timestamps = [float(frame["timestamp"]) for frame in frames]
				
	def _load_poses_from_txt(self, file_path: Path) -> List[np.ndarray]:
		"""Load poses from a single text file."""
		poses = []
		with open(file_path, 'r') as f:
			for line in f:
				if line.strip():
					values = [float(x) for x in line.strip().split()]
					assert len(values) == 16, f"Invalid pose format: {line.strip()}"
					# 4x4 transformation matrix - convert to pose
					T = np.array(values).reshape(4, 4)
					pose = self._matrix_to_pose(T)
					poses.append(pose)
		return poses
		
	def _load_individual_poses(self, pose_files: List[Path]) -> List[np.ndarray]:
		"""Load poses from individual files."""
		poses = []
		for file_path in pose_files:
			if file_path.suffix == ".txt":
				with open(file_path, 'r') as f:
					values = [float(x) for x in f.read().strip().split()]
					assert len(values) == 16, f"Invalid pose format: {values}"
					T = np.array(values).reshape(4, 4)
					pose = self._matrix_to_pose(T)
					poses.append(pose)
			elif file_path.suffix == ".json":
				with open(file_path, 'r') as f:
					data = json.load(f)
					if "pose" in data:
						poses.append(np.array(data["pose"]))
					elif "transform" in data:
						poses.append(np.array(data["transform"]))
		return poses
		
	def _matrix_to_pose(self, T: np.ndarray) -> np.ndarray:
		"""Convert 4x4 transformation matrix to 7D pose."""
		from scipy.spatial.transform import Rotation
		
		# Extract translation
		translation = T[:3, 3]
		
		# Extract rotation and convert to quaternion
		rotation_matrix = T[:3, :3]
		rotation = Rotation.from_matrix(rotation_matrix)
		quaternion = rotation.as_quat()  # Returns [qx, qy, qz, qw]
		
		return np.concatenate([translation, quaternion])
		
	def _load_camera_info(self):
		"""Load camera calibration information."""
		self.camera_info = None
		info_file = self.data_path / "camera_info.json"
		
		if info_file.exists():
			with open(info_file, 'r') as f:
				self.camera_info = json.load(f)
		else:
			# Try to infer from first image
			if self.rgb_files:
				img = cv2.imread(str(self.rgb_files[0]))
				if img is not None:
					h, w = img.shape[:2]
					# Default camera parameters
					self.camera_info = {
						"width": w,
						"height": h,
						"intrinsics": self._default_intrinsics(w, h),
						"distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
					}
					
	def _default_intrinsics(self, width: int, height: int) -> List[List[float]]:
		"""Generate default camera intrinsics."""
		# Assume a reasonable field of view
		focal_length = width  # Approximate
		cx = width / 2.0
		cy = height / 2.0
		
		return [
			[focal_length, 0.0, cx],
			[0.0, focal_length, cy],
			[0.0, 0.0, 1.0]
		]
		
	def __len__(self) -> int:
		"""Return number of frames in dataset."""
		return len(self.rgb_files)
		
	def __getitem__(self, idx: int) -> DatasetFrame:
		"""Get frame by index."""
		if idx < 0 or idx >= len(self):
			raise IndexError(f"Index {idx} out of range [0, {len(self)})")
			
		# Load RGB image
		rgb_path = self.rgb_files[idx]
		rgb_image = cv2.imread(str(rgb_path))
		if rgb_image is None:
			raise RuntimeError(f"Failed to load RGB image: {rgb_path}")
		rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
		
		# Load depth image if available
		depth_image = None
		if idx < len(self.depth_files):
			depth_path = self.depth_files[idx]
			depth_image = self._load_depth(depth_path)
			
		# Get pose if available
		transform = None
		if self.poses and idx < len(self.poses):
			transform = self.poses[idx]
			
		# Prefer capture timestamps so filtered datasets retain real temporal gaps.
		timestamp = self.timestamps[idx] if self.timestamps else idx / self.fps
		
		# Compute velocities if requested
		lin_vel, ang_vel = np.zeros(3), np.zeros(3)
		if self.compute_velocities and transform is not None:
			lin_vel, ang_vel = self._compute_frame_velocity(transform, timestamp)
			
		return DatasetFrame(
			frame_id=idx,
			timestamp=timestamp,
			rgb_image=rgb_image,
			timestamp_ns=(
				self.timestamps_ns[idx]
				if self.timestamps_ns is not None
				else None
			),
			depth_image=depth_image,
			transform=transform,
			camera_info=self.camera_info,
			lin_vel=lin_vel,
			ang_vel=ang_vel
		)
		
	def _load_depth(self, depth_path: Path) -> np.ndarray:
		"""Load depth image from file."""
		if depth_path.suffix == ".npy":
			depth = np.load(depth_path)
		elif depth_path.suffix == ".png":
			# Assume 16-bit PNG
			depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
			depth = depth.astype(np.float32)
		elif depth_path.suffix == ".exr":
			depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
		else:
			raise ValueError(f"Unsupported depth format: {depth_path.suffix}")
			
		# Scale to meters
		depth = depth / self.depth_scale
		
		return depth
		
	def _compute_frame_velocity(
		self, 
		transform: np.ndarray, 
		timestamp: float
	) -> tuple[np.ndarray, np.ndarray]:
		"""Compute linear and angular velocities from transform history."""
		# Add current transform and timestamp to history
		self.transforms_history.append(transform)
		self.timestamps_history.append(timestamp)
		
		# Need at least 2 frames to compute velocity
		if len(self.transforms_history) < 2:
			return np.zeros(3), np.zeros(3)
			
		# Convert deques to numpy arrays
		transforms = np.array(list(self.transforms_history))
		timestamps = np.array(list(self.timestamps_history))
		
		# Compute velocity using EMA
		velocity = compute_ema_velocity(transforms, timestamps, alpha=self.velocity_alpha)
		
		if velocity is not None:
			lin_vel = velocity[1:4]  # [vx, vy, vz]
			ang_vel = velocity[4:7]  # [wx, wy, wz]
			return lin_vel, ang_vel
			
		return np.zeros(3), np.zeros(3)
		
	def get_camera_info(self) -> Dict[str, Any]:
		"""Get camera calibration information."""
		return self.camera_info or {}
		
	@property
	def fps(self) -> float:
		"""Return dataset framerate."""
		# Default to 30 FPS if not specified
		if self.config and "fps" in self.config:
			return self.config["fps"]
		return 30.0
