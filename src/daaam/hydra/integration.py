"""Hydra integration module for proper pipeline execution with scene graph updates."""

import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import yaml
import logging
from datetime import datetime

try:
	import hydra_python
	from hydra_python import HydraPipeline, make_camera, set_glog_dir, set_glog_level
	from hydra_python.pipeline import load_pipeline
	from spark_dsg import DynamicSceneGraph
	from .config_loader import load_hydra_config
	HYDRA_AVAILABLE = True
except ImportError:
	HYDRA_AVAILABLE = False
	print("Warning: hydra_python not available. Hydra integration disabled.")

from daaam.utils.logging import PipelineLogger


class HydraIntegration:
	"""Manages Hydra pipeline integration with daaam."""
	
	def __init__(
		self,
		hydra_config_path: str,
		labelspace_path: Optional[str] = None,
		labelspace_colors: Optional[str] = None,
		output_dir: Optional[Path] = None,
		logger: Optional[PipelineLogger] = None,
		zmq_url: Optional[str] = None,
		glog_level: int = 0,
		verbosity: int = 0,
	):
		"""Initialize Hydra integration.
		
		Args:
			hydra_config_path: Path to Hydra config YAML file
			labelspace_path: Path to labelspace YAML file (optional)
			labelspace_colors: Path to labelspace colors CSV file (optional)
			output_dir: Output directory for Hydra results
			logger: Pipeline logger instance
			zmq_url: ZMQ URL for DSG visualization (None to disable)
			glog_level: Glog level for Hydra
			verbosity: Verbosity level for Hydra
		"""
		if not HYDRA_AVAILABLE:
			raise RuntimeError("hydra_python is not installed. Cannot use HydraIntegration.")
			
		self.hydra_config_path = hydra_config_path
		self.labelspace_path = labelspace_path
		self.labelspace_colors = labelspace_colors
		self.output_dir = output_dir or Path("output/hydra")
		self.logger = logger or logging.getLogger("hydra_integration")
		self.zmq_url = zmq_url
		
		# Set glog configuration
		set_glog_level(glog_level, verbosity)
		if output_dir:
			glog_dir = output_dir / "logs"
			glog_dir.mkdir(parents=True, exist_ok=True)
			set_glog_dir(str(glog_dir))
		
		self.pipeline = None
		self.camera = None
		self.dsg = DynamicSceneGraph()
		
		# Statistics
		self.stats = {
			"frames_processed": 0,
			"processing_times": [],
			"dsg_nodes": 0,
			"dsg_edges": 0
		}
		
		self.logger.info(f"HydraIntegration initialized with config: {hydra_config_path}")
	
	def initialize_camera(
		self,
		width: int,
		height: int,
		fx: float,
		fy: float,
		cx: Optional[float] = None,
		cy: Optional[float] = None,
		min_range: float = 0.1,
		max_range: float = 10.0
	) -> None:
		"""Initialize camera configuration.
		
		Args:
			width: Image width
			height: Image height
			fx: Focal length x
			fy: Focal length y
			cx: Principal point x (defaults to width/2)
			cy: Principal point y (defaults to height/2)
			min_range: Minimum depth range in meters
			max_range: Maximum depth range in meters
		"""
		if cx is None:
			cx = width / 2.0
		if cy is None:
			cy = height / 2.0
			
		self.camera = make_camera(
			fx=fx,
			fy=fy,
			cx=cx,
			cy=cy,
			width=width,
			height=height,
			min_range=min_range,
			max_range=max_range,
			name="daaam_camera"
		)
		
		self.logger.info(f"Camera initialized: {width}x{height}, fx={fx:.2f}, fy={fy:.2f}")
	
	def initialize_pipeline(self, camera_config: Optional[Dict[str, Any]] = None) -> bool:
		"""Initialize Hydra pipeline.
		
		Args:
			camera_config: Optional camera configuration dict with keys:
				width, height, fx, fy, cx, cy, min_range, max_range
		
		Returns:
			True if pipeline initialized successfully
		"""
		try:
			# Initialize camera if config provided
			if camera_config and not self.camera:
				self.initialize_camera(**camera_config)
			
			if not self.camera:
				self.logger.error("Camera must be initialized before pipeline")
				return False
			
			# Load Hydra configs from file paths
			configs = load_hydra_config(
				self.hydra_config_path,
				labelspace_path=self.labelspace_path,
				labelspace_colors=self.labelspace_colors
			)
				
			if not configs:
				self.logger.error(f"Failed to load Hydra config from: {self.hydra_config_path}")
				return False
			
			# Add output path to config
			if self.output_dir:
				configs["log_path"] = str(self.output_dir)
			
			with hydra_python.external_plugins("khronos"):
				try:
					self.pipeline = HydraPipeline.from_config(
						yaml.safe_dump(configs),
						self.camera,
						robot_id=0,
						config_verbosity=2,
						use_step_mode=True,
						zmq_url="" if self.zmq_url is None else self.zmq_url
					)
				except Exception as e:
					self.logger.error(f"Error creating Hydra pipeline: {e}")

			if not self.pipeline:
				self.logger.error("Failed to create Hydra pipeline")
				return False
			
			self.logger.info("Hydra pipeline initialized successfully")
			if self.zmq_url:
				self.logger.info(f"Publishing DSG to ZMQ: {self.zmq_url}")
			
			return True
			
		except Exception as e:
			self.logger.error(f"Error initializing Hydra pipeline: {e}")
			return False
	
	def process_frame(
		self,
		timestamp: float,
		rgb_image: np.ndarray,
		depth_image: np.ndarray,
		semantic_labels: np.ndarray,
		transform: Optional[np.ndarray] = None,
		timestamp_ns: Optional[int] = None,
	) -> bool:
		"""Process a single frame through Hydra.
		
		Args:
			timestamp: Timestamp in seconds
			rgb_image: RGB image (H, W, 3) uint8
			depth_image: Depth image in meters (H, W) float32
			semantic_labels: Semantic segmentation labels (H, W) int32
			transform: Optional 7D pose [x, y, z, qx, qy, qz, qw]
			timestamp_ns: Optional original capture timestamp in nanoseconds
		
		Returns:
			True if frame processed successfully
		"""
		if not self.pipeline:
			self.logger.error("Pipeline not initialized")
			return False
		
		try:
			start_time = time.time()
			
			# Preserve the capture clock when the dataset provides it. The float
			# timestamp remains useful for local semantic history and readability.
			step_timestamp_ns = (
				int(timestamp_ns) if timestamp_ns is not None else int(round(timestamp * 1e9))
			)
			
			# Parse transform
			if transform is not None and len(transform) >= 7:
				translation = transform[:3].flatten()
				# IMPORTANT: Hydra expects quaternion as [qw, qx, qy, qz] (scalar first!)
				quaternion = np.array([transform[6], transform[3], transform[4], transform[5]])
			else:
				# Default to identity
				translation = np.zeros(3)
				quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (w, x, y, z)
			
			# Convert depth to uint16 millimeters for Hydra
			depth_mm = (depth_image * 1000).astype(np.uint16)
			
			# Process through Hydra
			success = self.pipeline.step(
				step_timestamp_ns,
				translation,
				quaternion,
				depth_mm,
				semantic_labels.astype(np.int32),
				rgb_image
			)
			
			if success:
				self.stats["frames_processed"] += 1
				elapsed = time.time() - start_time
				self.stats["processing_times"].append(elapsed)
				
				if self.stats["frames_processed"] % 10 == 0:
					self.logger.debug(f"Hydra processed {self.stats['frames_processed']} frames")
			else:
				self.logger.warning(f"Hydra failed to process frame at timestamp {timestamp}")
			
			return success
			
		except Exception as e:
			self.logger.error(f"Error processing frame in Hydra: {e}")
			return False
	
	def get_dsg_update(self) -> Optional[bytes]:
		"""Get the latest DSG update as binary data.
		
		Returns:
			Binary DSG data or None if not available
		"""
		try:
			if self.pipeline:
				# Get DSG from pipeline
				# Note: Actual API may differ - this is a placeholder
				# You may need to save and load the DSG to get binary data
				return None
		except Exception as e:
			self.logger.error(f"Error getting DSG update: {e}")
			return None
	
	def save_results(self, output_path: Optional[Path] = None) -> bool:
		"""Save Hydra pipeline results.
		
		Args:
			output_path: Optional output path (uses default if not provided)
		
		Returns:
			True if saved successfully
		"""
		try:
			if not self.pipeline:
				self.logger.warning("No pipeline to save")
				return False
			
			save_path = output_path or self.output_dir
			save_path.mkdir(parents=True, exist_ok=True)
			
			self.pipeline.save(save_path)
			self.logger.info(f"Saved Hydra results to {save_path}")
			
			# Update DSG statistics
			try:
				dsg_path = save_path / "backend" / "dsg.json"
				if dsg_path.exists():
					import json
					with open(dsg_path, 'r') as f:
						dsg_data = json.load(f)
						self.stats["dsg_nodes"] = len(dsg_data.get("nodes", []))
						self.stats["dsg_edges"] = len(dsg_data.get("edges", []))
			except Exception as e:
				self.logger.debug(f"Could not load DSG statistics: {e}")
			
			return True
			
		except Exception as e:
			self.logger.error(f"Error saving Hydra results: {e}")
			return False
	
	def get_stats(self) -> Dict[str, Any]:
		"""Get processing statistics.
		
		Returns:
			Dictionary of statistics
		"""
		stats = self.stats.copy()
		if stats["processing_times"]:
			stats["avg_processing_time"] = np.mean(stats["processing_times"])
			stats["total_processing_time"] = np.sum(stats["processing_times"])
		return stats
	
	def shutdown(self) -> None:
		"""Shutdown Hydra pipeline."""
		try:
			if self.pipeline:
				self.save_results()
				# Pipeline cleanup if needed
				self.pipeline = None
			self.logger.info("Hydra integration shutdown complete")
		except Exception as e:
			self.logger.error(f"Error during Hydra shutdown: {e}")
