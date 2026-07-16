from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

import multiprocessing as mp
from multiprocessing.synchronize import Event
import queue
import time
from typing import Any, Optional, Callable
import logging

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Union
import json
import traceback

from daaam.utils.logging import setup_worker_logging, get_default_logger
from daaam.interfaces import QueueWorkerProcessInterface
from daaam.pipeline.models import PromptRecord
from daaam.realtime.gpu import SharedGpuCoordinator


class GroundingWorkerInterface(QueueWorkerProcessInterface):
	"""
	Base class for grounding workers that process PromptRecord objects
	and output ObjectAnnotation corrections.
	"""
	
	def __init__(
		self,
		query_group_queue: mp.Queue,
		correction_queue: mp.Queue,
		stop_event: Event,
		config: dict,
		ready_queue: Optional[mp.Queue] = None,
		name: Optional[str] = None
	):
		"""
		Initialize the grounding worker.
		
		Args:
			query_group_queue: Queue containing PromptRecord objects
			correction_queue: Queue to write ObjectAnnotation lists to
			stop_event: Shared event to signal process termination
			config: Worker-specific configuration dictionary
			ready_queue: Optional queue to signal when worker is ready
			name: Optional name for the process
		"""
		# Set CUDA device BEFORE any model initialization to ensure correct GPU routing
		cuda_device = config.get('cuda_device')
		if cuda_device is not None:
			import os
			os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)

		super().__init__(
			input_queue=query_group_queue,
			output_queue=correction_queue,
			stop_event=stop_event,
			name=name,
			timeout=0.1
		)
		self.config = config
		self.ready_queue = ready_queue
		self.worker_id = mp.current_process().name
		self._setup_worker_logging()
		self.gpu = SharedGpuCoordinator(
			lock_path=config.get("gpu_lock_path"),
			activity_path=config.get("gpu_activity_path"),
			minimum_idle_s=float(config.get("gpu_minimum_idle_s", 0.0)),
		)

		import torch
		import os
		effective_cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')
		device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
		self.worker_logger.info(f"[GPU] CUDA_VISIBLE_DEVICES={effective_cvd}, device_count={device_count}")

		try:
			with self.gpu.lease(self.stop_event):
				self._initialize_models()
		except Exception as error:
			if self.ready_queue:
				self.ready_queue.put(
					{"worker": self.worker_id, "ready": False, "error": repr(error)}
				)
			raise
		
		# Signal readiness only after every model initialized successfully.
		if self.ready_queue:
			self.ready_queue.put(
				{"worker": self.worker_id, "ready": True, "error": None}
			)
			if hasattr(self, 'worker_logger'):
				self.worker_logger.info(f"Worker {self.worker_id} signaled ready")
		
	def _setup_worker_logging(self):
		"""Setup logging for this worker."""
		
		log_dir = self.config.get("log_dir")
		if log_dir:
			self.worker_logger = setup_worker_logging(log_dir)
			self.worker_logger.debug(f"Worker {mp.current_process().name} logging to directory: {log_dir}")
		else:
			self.worker_logger = get_default_logger()
			print(f"[WARNING] Worker {mp.current_process().name} using default console logger (no log_dir in config)")


	@abstractmethod
	def _initialize_models(self):
		"""Initialize any models needed by the worker (e.g., embedding models, agents)."""
		pass
		
	@abstractmethod
	def process_prompt_record(self, prompt_record, **kwargs) -> Optional[List]:
		"""
		Process a prompt record and return corrections.
		
		Args:
			prompt_record: PromptRecord object containing frame and tracks
			**kwargs: Additional parameters specific to the worker implementation
			
		Returns:
			List of ObjectAnnotation objects or None
		"""
		pass
		
	def process_item(self, item) -> Optional[List]:
		"""
		Process a PromptRecord from the queue.
		
		Args:
			item: PromptRecord object
			
		Returns:
			List of ObjectAnnotation objects or None
		"""
		try:
			
			# validate record type
			if not isinstance(item, PromptRecord):
				self.worker_logger.error(f"Invalid prompt record type: {type(item)}")
				return None
				
			# call implementation-specific processing method
			corrections = self.process_prompt_record(item)
			
			# put corrections on queue if any
			if corrections:
				return corrections
			else:
				return None
				
		except Exception as e:
			self.worker_logger.error(f"ERROR processing prompt record: {e}")
			self.worker_logger.error(traceback.format_exc())
			return None
