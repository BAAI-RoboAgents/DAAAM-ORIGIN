from typing import Tuple, List, Optional
import numpy as np
import torch
from pathlib import Path

from daaam.segmentation.interfaces import SegmenterInterface
from daaam.utils.segmentation import UniversalSegmenter
from daaam.utils.logging import PipelineLogger, get_default_logger
from daaam.config import SegmentationConfig

from daaam import ROOT_DIR


class SegmentationService:
	"""Service for handling segmentation operations."""
	
	def __init__(self, config: SegmentationConfig, logger: Optional[PipelineLogger] = None):
		self.config = config
		self.logger = logger or get_default_logger()
		self.segmenter: Optional[SegmenterInterface] = None
		self.last_status = "initializing"
		self.last_error: Optional[str] = None

		self._initialize_segmenter()
	
	def _initialize_segmenter(self) -> None:
		"""Initialize the segmentation model."""
		try:
			# Use the existing UniversalSegmenter as an adapter
			self.segmenter = UniversalSegmenterAdapter(
				model_checkpoint_path=self.config.model_name,
				model_config_path=self.config.model_config_path,
				device=self.config.device,
				min_mask_region_area=self.config.min_mask_region_area,
				imgsz=self.config.imgsz,
				logger=self.logger
			)
			self.logger.info(f"Initialized segmenter with model: {self.config.model_name}")
		except Exception as e:
			self.logger.error(f"Failed to initialize segmenter: {e}")
			raise
	
	def warmup(self) -> None:
		"""Run dummy inference to trigger TensorRT/CUDA JIT compilation."""
		h, w = self.config.imgsz if self.config.imgsz else (480, 640)
		# Colored rectangles on black — reliable detection trigger for SAM models
		dummy = np.zeros((h, w, 3), dtype=np.uint8)
		dummy[h//6:h//3, w//6:w//3] = [255, 0, 0]
		dummy[h//6:h//3, w//2:5*w//6] = [0, 255, 0]
		dummy[h//2:5*h//6, w//4:3*w//4] = [0, 0, 255]
		dets, masks = self.segment(dummy)
		n_dets = len(dets)
		if n_dets == 0:
			self._warmup_postprocess_kernels(h, w)
		self.logger.info(f"Segmentation warmup complete (input: {h}x{w}, detections: {n_dets})")

	def _warmup_postprocess_kernels(self, h: int, w: int) -> None:
		"""Force-warm mask post-processing CUDA kernels when model produced 0 detections."""
		import torch.nn.functional as F
		from torchvision.ops import nms

		device = self.config.device
		# process_mask_native equivalent: masks_in @ protos
		protos = torch.randn(32, h // 4, w // 4, device=device)
		masks_in = torch.randn(10, 32, device=device)
		masks = (masks_in @ protos.float().view(32, -1)).sigmoid()
		# F.interpolate (bilinear upsampling)
		masks = F.interpolate(masks.view(-1, 1, h // 4, w // 4), (h, w), mode='bilinear', align_corners=False)
		# NMS
		boxes = torch.rand(10, 4, device=device) * torch.tensor([w, h, w, h], device=device, dtype=torch.float32)
		boxes[:, 2:] += boxes[:, :2]
		nms(boxes, torch.rand(10, device=device), 0.5)
		torch.cuda.synchronize()
		self.logger.info("Segmentation warmup: forced CUDA kernel warmup (0 model detections)")

	def segment(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
		"""
		Perform segmentation on input frame.
		
		Args:
			frame: RGB image as numpy array
			
		Returns:
			Tuple of (detection_boxes, segmentation_masks)
		"""
		if self.segmenter is None:
			raise RuntimeError("Segmenter not initialized")
		
		try:
			result = self.segmenter(frame)
			self.last_status = "ok" if len(result[0]) else "ok_empty"
			self.last_error = None
			return result
		except NotImplementedError as e:
			self.logger.info(f"Segmentation failed: {e}. This likely means SAM2 was selected but is not implemented.")
			self.last_status = "failed"
			self.last_error = repr(e)
			return np.empty((0, 6)), []
		except Exception as e:
			self.logger.error(f"Error during segmentation: {e}")
			self.last_status = "failed"
			self.last_error = repr(e)
			return np.empty((0, 6)), []

	def segment_checked(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
		"""Run inference while preserving the difference between empty and failed."""
		if self.segmenter is None:
			raise RuntimeError("Segmenter not initialized")
		try:
			result = self.segmenter(frame)
		except Exception as error:
			self.last_status = "failed"
			self.last_error = repr(error)
			raise
		self.last_status = "ok" if len(result[0]) else "ok_empty"
		self.last_error = None
		return result


class UniversalSegmenterAdapter(SegmenterInterface):
	"""Adapter to make UniversalSegmenter comply with SegmenterInterface."""

	def __init__(self, model_checkpoint_path: str, model_config_path: str = None, device: str = None, min_mask_region_area: int = 100, imgsz: Optional[Tuple[int, int]] = None, logger: Optional[PipelineLogger] = None):
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.logger = logger or get_default_logger()
		self.min_mask_region_area = min_mask_region_area
		self.imgsz = imgsz

		full_checkpoint_path = Path(ROOT_DIR) / "checkpoints" / model_checkpoint_path
		full_config_path = Path(ROOT_DIR) / "config" / model_config_path if model_config_path else None

		self.segmenter = UniversalSegmenter(
			model_checkpoint_path=full_checkpoint_path,
			model_config_path=full_config_path or "",
			device=self.device,
			min_mask_region_area=self.min_mask_region_area,
			imgsz=self.imgsz,
			logger=self.logger
		)
	
	def __call__(self, source: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
		"""Perform segmentation using UniversalSegmenter."""
		return self.segmenter(source=source)
	
	def initialize(self, model_path: str, config_path: str = None, device: str = None) -> None:
		"""Initialize method for interface compliance."""
		# Already initialized in __init__
		pass
