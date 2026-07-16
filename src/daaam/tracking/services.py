from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from boxmot import BotSort
from boxmot.appearance.backends.tensorrt_backend import TensorRTBackend

from daaam.tracking.interfaces import TrackerInterface
from daaam.utils.logging import PipelineLogger, get_default_logger
from daaam.config import TrackingConfig
from daaam import ROOT_DIR


class _BatchedTensorRTCrops:
	"""Preserve BoxMot preprocessing while using one host-to-device transfer."""

	def __init__(self, backend: TensorRTBackend):
		self.backend = backend
		binding = backend.bindings.get("images")
		if binding is None or tuple(binding.shape[1:]) != (3, 256, 128):
			raise ValueError("unsupported TensorRT ReID input shape")
		requested_dtype = np.dtype(np.float16 if backend.half else np.float32)
		if np.dtype(binding.dtype) != requested_dtype:
			raise ValueError("reid_half does not match TensorRT engine input dtype")
		self.torch_dtype = (
			torch.float16 if requested_dtype == np.dtype(np.float16) else torch.float32
		)
		self.mean = torch.tensor(
			[0.485, 0.456, 0.406],
			device=backend.device,
			dtype=self.torch_dtype,
		).view(1, 3, 1, 1)
		self.std = torch.tensor(
			[0.229, 0.224, 0.225],
			device=backend.device,
			dtype=self.torch_dtype,
		).view(1, 3, 1, 1)

	@torch.no_grad()
	def __call__(self, xyxys: np.ndarray, img: np.ndarray) -> torch.Tensor:
		boxes = np.asarray(xyxys)
		if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
			raise ValueError("ReID input image must be HxWx3 uint8")
		if boxes.size and not np.isfinite(boxes).all():
			raise ValueError("ReID bounding boxes must be finite")
		num_crops = len(boxes)
		if num_crops == 0:
			return torch.empty(
				(0, 3, 256, 128),
				device=self.backend.device,
				dtype=self.torch_dtype,
			)

		h, w = img.shape[:2]
		host_crops = np.empty((num_crops, 3, 256, 128), dtype=np.uint8)
		for index, box in enumerate(boxes):
			x1, y1, x2, y2 = box.astype("int")
			x1, y1 = max(0, x1), max(0, y1)
			x2, y2 = min(w - 1, x2), min(h - 1, y2)
			if x2 <= x1 or y2 <= y1:
				raise ValueError(f"degenerate ReID bounding box at index {index}")
			crop = cv2.resize(
				img[y1:y2, x1:x2],
				(128, 256),
				interpolation=cv2.INTER_LINEAR,
			)
			crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
			host_crops[index] = crop.transpose(2, 0, 1)

		crops = torch.from_numpy(host_crops).to(
			device=self.backend.device,
			dtype=self.torch_dtype,
		)
		crops = crops / 255.0
		return (crops - self.mean) / self.std


def _enable_batched_tensorrt_crops(tracker: BotSort) -> bool:
	backend = getattr(tracker, "model", None)
	if (
		not isinstance(backend, TensorRTBackend)
		or backend.device.type != "cuda"
		or getattr(backend, "nhwc", True)
	):
		return False
	backend.get_crops = _BatchedTensorRTCrops(backend)
	return True


class TrackingService:
	"""Service for handling object tracking operations."""
	
	def __init__(self, config: TrackingConfig, logger: Optional[PipelineLogger] = None):
		self.config = config
		self.logger = logger or get_default_logger()
		self.tracker: Optional[TrackerInterface] = None
		self.track_buffer = config.track_buffer if hasattr(config, 'track_buffer') else 30
		self._initialize_tracker()

	def warmup(self) -> None:
		"""Warmup full tracking pipeline (ReID, Kalman, association) then reset state."""
		try:
			from boxmot.trackers.botsort.basetrack import BaseTrack

			h, w = 480, 640
			dummy_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
			n = 20
			dets = np.column_stack([
				np.random.randint(0, w // 2, n),
				np.random.randint(0, h // 2, n),
				np.random.randint(w // 2, w, n),
				np.random.randint(h // 2, h, n),
				np.random.uniform(0.5, 1.0, n),
				np.zeros(n),
			]).astype(np.float32)

			self.tracker.update(dets, dummy_img)

			# Reset track state (not config: keep _first_frame_processed, h, w, asso_func)
			bot = self.tracker.tracker
			bot.active_tracks = []
			bot.lost_stracks = []
			bot.removed_stracks = []
			bot.frame_count = 0
			BaseTrack._count = 0
			if hasattr(bot, 'cmc') and hasattr(bot.cmc, 'prev_img'):
				bot.cmc.prev_img = None

			self.logger.info(f"Tracking warmup complete (full pipeline, n_dets={n})")
		except Exception as e:
			self.logger.warning(f"Tracking warmup failed — skipping ({e})")

	def _initialize_tracker(self) -> None:
		"""Initialize the tracking model."""
		reid_weights = getattr(self.config, 'reid_weights', 'checkpoints/reid_weights/clip_general.engine')
		with_reid = getattr(self.config, 'with_reid', True)
		reid_half = getattr(self.config, 'reid_half', False)
		batch_reid_crops = getattr(self.config, 'batch_reid_crops', False)
		cmc_method = getattr(self.config, 'cmc_method', 'ecc')
		cmc_ecc_max_iterations = getattr(self.config, 'cmc_ecc_max_iterations', 100)
		try:
			self.tracker = BotSortAdapter(
				device=self.config.device,
				track_buffer=self.track_buffer,
				reid_weights=reid_weights,
				with_reid=with_reid,
				reid_half=reid_half,
				batch_reid_crops=batch_reid_crops,
				cmc_method=cmc_method,
				cmc_ecc_max_iterations=cmc_ecc_max_iterations,
			)
			self.logger.info(
				"Initialized BotSort tracker with "
				f"track_buffer={self.track_buffer}, reid_weights={reid_weights}, "
				f"with_reid={with_reid}, reid_half={reid_half}, "
				f"batch_reid_crops={self.tracker.batch_reid_crops_enabled}, "
				f"cmc_method={cmc_method}, "
				f"cmc_ecc_max_iterations={cmc_ecc_max_iterations}"
			)
		except Exception as e:
			self.logger.error(f"Failed to initialize tracker: {e}")
			raise
	
	def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
		"""
		Update tracker with new detections.
		
		Args:
			detections: N x 6 array of [x1, y1, x2, y2, conf, cls]
			frame: Current RGB frame
			
		Returns:
			M x 8 array of [x1, y1, x2, y2, track_id, conf, cls, mask_idx]
		"""
		if self.tracker is None:
			raise RuntimeError("Tracker not initialized")
		
		detections = np.asarray(detections, dtype=np.float32)
		if detections.size == 0:
			detections = np.empty((0, 6), dtype=np.float32)
		tracks = self.tracker.update(detections, frame)
		if tracks is None or len(tracks) == 0:
			return np.empty((0, 8), dtype=np.float32)
		tracks = np.asarray(tracks)
		if tracks.ndim != 2 or tracks.shape[1] < 8:
			raise ValueError("BotSort output must be an Mx8 array")
		# BotSort IDs are zero-based; public DAAAM semantic IDs reserve zero.
		tracks = tracks.copy()
		tracks[:, 4] += 1
		return tracks
	
	def get_track_buffer(self) -> int:
		"""Get the track buffer value."""
		return self.track_buffer


class BotSortAdapter(TrackerInterface):
	"""Adapter to make BotSort comply with TrackerInterface."""

	def __init__(
		self,
		device: str = None,
		track_buffer: int = 30,
		reid_weights: str = "checkpoints/reid_weights/clip_general.engine",
		with_reid: bool = True,
		reid_half: bool = False,
		batch_reid_crops: bool = False,
		cmc_method: str = "ecc",
		cmc_ecc_max_iterations: int = 100,
	):
		if cmc_method == "ecc" and cmc_ecc_max_iterations <= 0:
			raise ValueError("cmc_ecc_max_iterations must be positive")
		self.device = device or "cpu"
		self.tracker = BotSort(
			reid_weights=ROOT_DIR / Path(reid_weights),
			device=0 if self.device == "cuda" else "cpu",
			half=reid_half,
			track_buffer=track_buffer,
			with_reid=with_reid,
			cmc_method=cmc_method,
		)
		self.batch_reid_crops_enabled = bool(
			batch_reid_crops
			and with_reid
			and _enable_batched_tensorrt_crops(self.tracker)
		)
		if cmc_method == "ecc":
			termination = getattr(self.tracker.cmc, "termination_criteria", None)
			if not isinstance(termination, tuple) or len(termination) != 3:
				raise RuntimeError("BoxMot ECC termination criteria are unavailable")
			self.tracker.cmc.termination_criteria = (
				termination[0],
				cmc_ecc_max_iterations,
				termination[2],
			)
	
	def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
		"""Update tracker with new detections."""
		return self.tracker.update(detections, frame)
	
	def initialize(self, config: dict = None) -> None:
		"""Initialize method for interface compliance."""
		# BotSort doesn't require additional initialization
		pass
