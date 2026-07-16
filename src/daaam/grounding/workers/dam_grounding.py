import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple
from PIL import Image as PILImage
import torch
import traceback
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
import re
from daaam.pipeline.models import PromptRecord
from daaam.tracking.models import Track
from pathlib import Path
import cv2
import yaml
import queue
import time

from daaam.grounding.interfaces import GroundingWorkerInterface
from daaam.utils.embedding import CLIPHandler, SentenceEmbeddingHandler
from daaam.utils.logging import PipelineLogger
from daaam.utils.performance import performance_measure
from daaam.query_manager.dam import DAMAgentPanoptic
from daaam.grounding.models import ObjectAnnotation, ImageAnnotation
from daaam.grounding.utils import (
	prepare_masks_from_tracks,
	create_corrections_from_descriptions
)


class DAMGroundingWorkerBase(GroundingWorkerInterface):
	"""Base class for DAM grounding workers."""
	
	def _initialize_models(self):
		"""Initialize DAM agent and embedding model."""
		self.worker_logger.info("Initializing DAM agent...")
		dam_model_path = self.config.get("dam_model_path", "nvidia/DAM-3B")
		dam_conv_mode = self.config.get("dam_conv_mode", "v1")
		dam_prompt_mode = self.config.get("dam_prompt_mode", "focal_prompt")
		self.compute_full_image_description = self.config.get("compute_full_image_description", False)

		try:
			self.dam_agent = DAMAgentPanoptic(
				model_path=dam_model_path,
				conv_mode=dam_conv_mode,
				prompt_mode=dam_prompt_mode,
			)
			self.worker_logger.info(f"[Worker {self.worker_id}] DAM agent initialized successfully.")
		except Exception as e:
			self.worker_logger.error(f"[Worker {self.worker_id}] Failed to initialize DAM agent: {e}")
			self.dam_agent = None
			raise RuntimeError("DAM model initialization failed") from e
		
		# Initialize sentence embedding handler
		sentence_embedding_model_name = self.config.get("sentence_embedding_model_name")
		if sentence_embedding_model_name:
			device = "cuda" if torch.cuda.is_available() else "cpu"
			self.sentence_embedding_handler = SentenceEmbeddingHandler(
				model_name=sentence_embedding_model_name,
				device=device,
				logger=self.worker_logger
			)
			self.worker_logger.info(f"[Worker {self.worker_id}] Initialized sentence embedding: {sentence_embedding_model_name}")
		else:
			self.sentence_embedding_handler = None
			self.worker_logger.info(f"[Worker {self.worker_id}] No sentence embedding model configured")
		
		# Initialize CLIP handler if enabled
		if self.config.get("enable_selectframe_clip_features", False):
			model_name = self.config.get("selectframe_clip_model_name", "ViT-B-16")
			pretrained = self.config.get("selectframe_clip_model_dataset", "openai")
			backend = self.config.get("selectframe_clip_backend", "openclip")

			device = "cuda" if torch.cuda.is_available() else "cpu"
			self.clip_handler = CLIPHandler(
				model_name=model_name,
				device=device,
				pretrained=pretrained,
				backend=backend,
				logger=self.worker_logger
			)

			self.worker_logger.info(f"CLIP model loaded successfully on {device}")
		else:
			self.clip_handler = None

		# Initialize image saving settings
		self.save_grounding_images = self.config.get("save_grounding_images", False)
		self.save_plain_grounding_images = self.config.get("save_plain_grounding_images", False)
		self.save_object_images = self.config.get("save_object_images", False)
		self.output_dir = self.config.get("output_dir", None)
		self.color_map: Dict[int, Tuple[int, int, int]] = self.config.get("color_map", None)
		self.grounding_images_dir = None
		self.grounding_annotations_dir = None
		self.object_images_dir = None
		self.image_counter = 0
		
		if self.save_grounding_images and self.output_dir:
			# Create organized folder structure
			worker_dir = Path(self.output_dir) / f"worker_{self.worker_id}"
			self.grounding_images_dir = worker_dir / "grounding_images"

			if self.save_plain_grounding_images:
				self.grounding_images_dir_plain = worker_dir / "grounding_images_plain"
				self.grounding_images_dir_masks_only = worker_dir / "grounding_images_masks_only"
			self.grounding_annotations_dir = worker_dir / "grounding_annotations"

			self.grounding_images_dir.mkdir(parents=True, exist_ok=True)
			if self.save_plain_grounding_images:
				self.grounding_images_dir_plain.mkdir(parents=True, exist_ok=True)
				self.grounding_images_dir_masks_only.mkdir(parents=True, exist_ok=True)
			self.grounding_annotations_dir.mkdir(parents=True, exist_ok=True)

			if self.save_object_images:
				self.object_images_dir = worker_dir / "object_images"
				self.object_images_dir.mkdir(parents=True, exist_ok=True)

			self.worker_logger.info(f"[Worker {self.worker_id}] Will save grounding data to {worker_dir}")



		self.worker_logger.info(f"[Worker {self.worker_id}] Finished initializing successfully.")

	def _save_grounding_image(
		self,
		image: np.ndarray,
		masks: List[PILImage.Image],
		semantic_ids: List[int],
		timestamp: Optional[float] = None
	) -> Optional[str]:
		"""Save image with overlaid masks for debugging. Returns base filename."""
		if not self.save_grounding_images or not self.grounding_images_dir:
			return None
			
		try:
			if isinstance(image, PILImage.Image):
				image_np = np.array(image)
			else:
				image_np = image.copy()
			
			overlay = image_np.copy()
			if self.save_plain_grounding_images:
				masks_only = np.zeros(image_np.shape[:2], dtype=np.uint16)
			weight = 0.3

			if self.color_map and np.all([id_ in self.color_map.keys() for id_ in semantic_ids if id_ != -1]):
				self.worker_logger.debug(f"[Worker {self.worker_id}] Using provided color map for grounding visualization.")
			else:
				self.worker_logger.warning(f"[Worker {self.worker_id}] No valid color map provided, using default colors.")
			
			for idx, (mask, sem_id) in enumerate(zip(masks, semantic_ids)):
				if sem_id == -1:  # full image mask
					continue
					
				mask_np = np.array(mask) > 128
				
				if self.color_map and sem_id in self.color_map:
					color = self.color_map[sem_id] 
				else:
					default_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
					color = default_colors[idx % len(default_colors)]
				
				overlay[mask_np] = overlay[mask_np] * (1 - weight) + np.array(color) * weight

				if self.save_plain_grounding_images:
					masks_only[mask_np] = np.uint16(sem_id)
				
				# semantic ID label
				y, x = np.where(mask_np)
				if len(y) > 0:
					cy, cx = int(np.mean(y)), int(np.mean(x))
					cv2.putText(overlay, f"ID:{sem_id}", (cx-20, cy), 
							   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
			
			# Generate filename with timestamp (if available) and counter for uniqueness
			if timestamp is not None and timestamp > 0.0:
				base_filename = f"grounding_{timestamp:.6f}_{self.image_counter:06d}"
			else:
				base_filename = f"grounding_{self.image_counter:06d}"
			image_filename = f"{base_filename}.jpg"
			filepath = self.grounding_images_dir / image_filename
			cv2.imwrite(str(filepath), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

			if self.save_plain_grounding_images:
				cv2.imwrite(str(self.grounding_images_dir_plain / image_filename),cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
				cv2.imwrite(str(self.grounding_images_dir_masks_only / image_filename.replace(".jpg", ".png")), masks_only.astype(np.uint16))

			self.image_counter += 1

			del overlay
			
			self.worker_logger.debug(f"[Worker {self.worker_id}] Saved grounding image: {filepath}")
			return base_filename
			
		except Exception as e:
			self.worker_logger.error(f"[Worker {self.worker_id}] Failed to save grounding image: {e}")
			return None
	
	def _save_grounding_annotations(
		self,
		base_filename: str,
		semantic_ids: List[int],
		descriptions: List[str],
		timestamp: Optional[float] = None
	) -> None:
		"""Save annotations to YAML file."""
		if not self.save_grounding_images or not self.grounding_annotations_dir or not base_filename:
			return

		try:
			annotations = []
			for sem_id, desc in zip(semantic_ids, descriptions):
				annotations.append({
					"semantic_id": int(sem_id),
					"description": desc,
					"is_full_image": sem_id == -1
				})

			# Create annotation data
			annotation_data = {
				"image_file": f"{base_filename}.jpg",
				"timestamp": timestamp if timestamp is not None else float(self.image_counter - 1),
				"frame_counter": self.image_counter - 1,
				"num_masks": len(semantic_ids),
				"annotations": annotations
			}
			
			# Save to YAML
			yaml_filename = f"{base_filename}.yaml"
			yaml_filepath = self.grounding_annotations_dir / yaml_filename
			
			with open(yaml_filepath, 'w') as f:
				yaml.dump(annotation_data, f, default_flow_style=False, sort_keys=False)
			
			self.worker_logger.debug(f"[Worker {self.worker_id}] Saved annotations: {yaml_filepath}")
			
		except Exception as e:
			self.worker_logger.error(f"[Worker {self.worker_id}] Failed to save annotations: {e}")

	def _save_object_image(self, crop: np.ndarray, semantic_id: int) -> bool:
		"""Save individual object crop for a semantic ID.

		Args:
			crop: RGB image crop as numpy array
			semantic_id: Semantic ID for this object

		Returns:
			True if saved successfully, False otherwise
		"""
		if not self.save_object_images or not self.object_images_dir:
			return False

		# Skip full image masks (semantic_id == -1)
		if semantic_id == -1:
			return False

		try:
			filename = f"{semantic_id:05d}.jpg"
			filepath = self.object_images_dir / filename

			# Convert RGB to BGR for cv2
			crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
			cv2.imwrite(str(filepath), crop_bgr)

			self.worker_logger.debug(
				f"[Worker {self.worker_id}] Saved object image: {filepath}"
			)
			return True

		except Exception as e:
			self.worker_logger.error(
				f"[Worker {self.worker_id}] Failed to save object image for "
				f"semantic_id {semantic_id}: {e}"
			)
			return False

	def _extract_clip_features_for_tracks(
		self, 
		frame: np.ndarray, 
		tracks: List[Track], 
		semantic_ids: List[int]
	) -> Dict[int, List[float]]:
		"""Extract CLIP features for each track's patch.
		
		Args:
			frame: RGB image frame
			tracks: List of tracks with bboxes
			semantic_ids: List of semantic IDs corresponding to tracks
			
		Returns:
			Dictionary mapping semantic_id to CLIP feature vector
		"""
		if self.clip_handler is None:
			return {}
			
		try:
			# Collect valid crops and their semantic IDs
			valid_crops = []
			valid_semantic_ids = []
			
			for track, sem_id in zip(tracks, semantic_ids):
				# Skip full image masks
				if sem_id == -1:
					continue
					
				# Get bbox from track
				if hasattr(track, 'bbox') and track.bbox is not None:
					bbox = track.bbox
					if isinstance(bbox, np.ndarray):
						bbox = bbox.tolist()
					
					# Extract crop using bbox (x1, y1, x2, y2)
					x1, y1, x2, y2 = [int(coord) for coord in bbox]
					
					# Ensure bbox is within image bounds
					h, w = frame.shape[:2]
					x1 = max(0, min(x1, w-1))
					x2 = max(x1+1, min(x2, w))
					y1 = max(0, min(y1, h-1))
					y2 = max(y1+1, min(y2, h))
					
					crop = frame[y1:y2, x1:x2]
					
					# Skip if crop is too small
					if crop.shape[0] < 10 or crop.shape[1] < 10:
						self.worker_logger.debug(
							f"[Worker {self.worker_id}] Skipping CLIP feature for semantic_id {sem_id}: "
							f"crop too small ({crop.shape[0]}x{crop.shape[1]})"
						)
						continue
					
					valid_crops.append(crop)
					valid_semantic_ids.append(sem_id)

					# Save object image if enabled
					if self.save_object_images:
						self._save_object_image(crop, sem_id)
				else:
					self.worker_logger.debug(
						f"[Worker {self.worker_id}] Track for semantic_id {sem_id} has no bbox"
					)
			
			if not valid_crops:
				return {}
			
			# Compute CLIP features in batch
			with performance_measure(f"DAMGroundingWorker {self.worker_id}: extract_clip_features", self.worker_logger.debug):
				features_batch = self.clip_handler.extract_image_features_from_arrays(
					valid_crops,
					show_progress=False
				)
			
			# Map features to semantic IDs
			feature_map = {}
			for sem_id, features in zip(valid_semantic_ids, features_batch):
				feature_map[sem_id] = features.tolist()  # Convert to list for JSON serialization
			
			self.worker_logger.info(
				f"[Worker {self.worker_id}] Extracted CLIP features for {len(feature_map)} objects"
			)
			
			return feature_map
			
		except Exception as e:
			self.worker_logger.error(
				f"[Worker {self.worker_id}] Error extracting CLIP features: {e}"
			)
			return {}
	
	@abstractmethod
	def process_prompt_record(self, prompt_record: PromptRecord) -> Optional[List[ObjectAnnotation]]:
		return None


class DAMGroundingWorkerMultiImage(DAMGroundingWorkerBase):
	"""DAM-specific grounding worker that aggregates multiple images/masks for batch processing."""
	
	def __init__(self, query_group_queue, correction_queue, stop_event, config, ready_queue=None, name=None):
		super().__init__(
			query_group_queue,
			correction_queue,
			stop_event,
			config,
			ready_queue=ready_queue,
			name=name,
		)
		self.multi_image_min_n_masks = config.get("multi_image_min_n_masks", None)
		self.max_batch_age_s = float(config.get("max_batch_age_s", 1.0))
		if self.max_batch_age_s <= 0.0:
			raise ValueError("max_batch_age_s must be positive")
		
		if self.multi_image_min_n_masks is None:
			self.worker_logger.error("multi_image_min_n_masks not set in config.")
			self.multi_image_min_n_masks = 1  # default to single image behavior
		
		self.aggregated_records = []
		self.total_masks = 0
		self._batch_started_monotonic = None
	
	def process_prompt_record(self, prompt_record: PromptRecord) -> Optional[List[ObjectAnnotation]]:
		"""Process prompt record - aggregate until we have enough masks."""
		if self.dam_agent is None:
			return None
		
		# validate and count masks in record
		prompt_tracks = prompt_record.tracks
		object_labels = prompt_record.object_labels
		
		valid_tracks = [
			t for t in prompt_tracks 
			if object_labels.get(t.id) is not None and 
			   t.segmentation is not None and
			   isinstance(t.segmentation, np.ndarray)
		]
		
		if not valid_tracks:
			self.worker_logger.warning("No valid tracks in record, skipping")
			return None
		
		# add to aggregation
		if not self.aggregated_records:
			self._batch_started_monotonic = time.monotonic()
		self.aggregated_records.append(prompt_record)
		self.total_masks += len(valid_tracks)
		
		self.worker_logger.debug(
			f"Aggregated record with {len(valid_tracks)} masks. "
			f"Total: {self.total_masks}/{self.multi_image_min_n_masks}"
		)
		
		# process if enough masks
		if self.total_masks >= self.multi_image_min_n_masks:
			return self._process_aggregated_batch()
		
		# continue aggregating
		return None
	
	def _process_aggregated_batch(self) -> Optional[List[ObjectAnnotation]]:
		"""Process one DAM batch only while holding the shared GPU lease."""
		with self.gpu.lease():
			return self._process_aggregated_batch_under_lease()

	def _process_aggregated_batch_under_lease(self) -> Optional[List[ObjectAnnotation]]:
		"""Process the aggregated batch of records."""
		if not self.aggregated_records:
			return None
		
		self.worker_logger.info(
			f"Processing batch of {len(self.aggregated_records)} records "
			f"with {self.total_masks} total masks"
		)
		
		try:
			# prep image-mask pairs for multi-image processing
			image_mask_pairs = []
			all_semantic_ids_by_record = []
			all_clip_features_by_record = []  # Store CLIP features for each record
			all_timestamps_by_record = []  # Store timestamps for each record
			all_context_by_record = []
			base_filenames = []

			for record in self.aggregated_records:
				prompt_frame = record.frame
				prompt_tracks = record.tracks
				object_labels = record.object_labels
				record_timestamp = record.timestamp
				
				pil_frame = PILImage.fromarray(prompt_frame)
				
				masks, semantic_ids = prepare_masks_from_tracks(
					prompt_tracks, object_labels, self.worker_id
				)

				if not masks:
					self.worker_logger.error(f"[Worker {self.worker_id}] No valid masks to process.")
					return None

				# Extract CLIP features if enabled
				clip_features = {}
				if self.config.get("enable_selectframe_clip_features", False) and self.clip_handler is not None:
					# Filter tracks to match semantic_ids (excluding -1 for full image)
					valid_tracks = [t for t, sid in zip(prompt_tracks, semantic_ids[:len(prompt_tracks)]) if sid != -1]
					valid_semantic_ids = [sid for sid in semantic_ids[:len(prompt_tracks)] if sid != -1]
					
					if valid_tracks:
						clip_features = self._extract_clip_features_for_tracks(
							prompt_frame, valid_tracks, valid_semantic_ids
						)

				# Add full image mask if enabled
				if self.compute_full_image_description:
					full_image_mask = PILImage.fromarray((np.ones_like(masks[0]) * 255).astype(np.uint8), mode='L')
					masks.append(full_image_mask)
					semantic_ids.append(-1)  # Special semantic_id for full image
					self.worker_logger.debug(f"[Worker {self.worker_id}] Added full image mask for image {len(image_mask_pairs)}")
				
				if masks:
					image_mask_pairs.append((pil_frame, masks))
					all_semantic_ids_by_record.append(semantic_ids)
					all_clip_features_by_record.append(clip_features)
					all_timestamps_by_record.append(record_timestamp)
					all_context_by_record.append(record)

					# Save grounding image if enabled
					base_filename = self._save_grounding_image(prompt_frame, masks, semantic_ids, timestamp=record_timestamp)
					base_filenames.append(base_filename)
			
			if not image_mask_pairs:
				self.worker_logger.error(f"[Worker {self.worker_id}] No valid image-mask pairs to process.")
				return None
			
			total_masks = sum(len(masks) for _, masks in image_mask_pairs)
			self.worker_logger.info(
				f"[Worker {self.worker_id}] Querying DAM agent with {len(image_mask_pairs)} images "
				f"and {total_masks} total masks..."
			)
			
			# DAM Agent
			# Use smaller max_new_tokens if full image description is included to prevent repetition
			max_tokens = 196 if self.compute_full_image_description else 512
			with performance_measure(f"DAMGroundingWorker {self.worker_id}: dam_agent.query_multi_image_multi_mask", self.worker_logger.debug):
				descriptions_by_image = self.dam_agent.query_multi_image_multi_mask(
					image_mask_pairs=image_mask_pairs,
					query="Describe what you see in this region.",
					temperature=0.2, 
					top_p=0.9,
					max_new_tokens=max_tokens
				)
			
			self.worker_logger.info(
				f"[Worker {self.worker_id}] DAM agent returned descriptions for {len(descriptions_by_image)} images"
			)
			
			# ObjectAnnotation objects
			corrections = []
			for img_idx, descriptions in enumerate(descriptions_by_image):
				if img_idx < len(all_semantic_ids_by_record):
					semantic_ids = all_semantic_ids_by_record[img_idx]
					clip_features = all_clip_features_by_record[img_idx] if img_idx < len(all_clip_features_by_record) else {}
					record_timestamp = all_timestamps_by_record[img_idx] if img_idx < len(all_timestamps_by_record) else None
					record_context = all_context_by_record[img_idx]

					# Save annotations if enabled
					if img_idx < len(base_filenames) and base_filenames[img_idx] and descriptions:
						self._save_grounding_annotations(base_filenames[img_idx], semantic_ids, descriptions, timestamp=record_timestamp)

					# Create corrections and attach CLIP features
					record_corrections = create_corrections_from_descriptions(
						descriptions, semantic_ids, self.worker_id
					)
					
					# Attach CLIP features to corrections
					for correction in record_corrections:
						if hasattr(correction, 'semantic_id'):
							track_id = next(
								(
									track_id
									for track_id, semantic_id in record_context.object_labels.items()
									if semantic_id == correction.semantic_id
								),
								None,
							)
							correction.entity_id = (
								record_context.entity_ids.get(track_id)
								if track_id is not None
								else None
							)
							correction.request_id = record_context.request_id or None
							correction.sensor_time_ns = record_context.sensor_time_ns or None
							correction.map_revision = record_context.map_revision
							correction.source_model = self.config.get(
								"dam_model_path", "nvidia/DAM-3B"
							)
						if hasattr(correction, 'semantic_id') and correction.semantic_id in clip_features:
							correction.selectframe_clip_feature = clip_features[correction.semantic_id]
							self.worker_logger.debug(
								f"[Worker {self.worker_id}] Attached CLIP feature to semantic_id {correction.semantic_id}"
							)
					
					corrections.extend(record_corrections)

			# Clear GPU cache after processing all images
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			
			if corrections:
				# Add sentence embeddings if handler is available
				if self.sentence_embedding_handler is not None:
					texts = [c.semantic_label for c in corrections]
					embeddings = self.sentence_embedding_handler.extract_text_embeddings_with_format(
						texts,
						prefix="an object that looks like '",
						postfix="'",
						show_progress=False
					)
					for correction, embedding in zip(corrections, embeddings):
						correction.embedding = embedding.tolist()
				return corrections
			else:
				self.worker_logger.warning(f"[Worker {self.worker_id}] Received no valid corrections from DAM agent.")
				return None
			
		except Exception as e:
			self.worker_logger.error(f"[Worker {self.worker_id}] ERROR processing multi-image batch: {e}")

			self.worker_logger.error(traceback.format_exc())
			return None
		finally:
			# clear batch after processing
			self.aggregated_records = []
			self.total_masks = 0
			self._batch_started_monotonic = None
			
			# clear GPU cache after batch processing
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				self.worker_logger.debug(f"[Worker {self.worker_id}] Cleared GPU cache after multi-image batch processing")
	
	def _emit_corrections(self, corrections) -> None:
		if not corrections:
			return
		self.worker_logger.info(
			f"[Worker {self.worker_id}] Putting {len(corrections)} corrections on output queue"
		)
		for correction in corrections:
			self.output_queue.put(correction)

	def run(self):
		"""Flush underfilled batches by age and again during clean shutdown."""
		if self.dam_agent is None:
			self.worker_logger.error("DAM agent not initialized. Exiting worker.")
			return
		try:
			while not self.stop_event.is_set():
				try:
					item = self.input_queue.get(timeout=self.timeout)
				except queue.Empty:
					if (
						self.aggregated_records
						and self._batch_started_monotonic is not None
						and time.monotonic() - self._batch_started_monotonic
						>= self.max_batch_age_s
					):
						self.worker_logger.info("Flushing underfilled DAM batch by age")
						self._emit_corrections(self._process_aggregated_batch())
					continue
				result = self.process_item(item)
				self._emit_corrections(result)
		except KeyboardInterrupt:
			self.worker_logger.info("Received keyboard interrupt")
		finally:
			if self.aggregated_records and self.total_masks > 0:
				self.worker_logger.info(
					f"Processing final batch of {len(self.aggregated_records)} records "
					f"with {self.total_masks} total masks"
				)
				self._emit_corrections(self._process_aggregated_batch())
