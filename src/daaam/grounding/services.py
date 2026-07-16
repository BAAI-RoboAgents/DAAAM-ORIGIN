from typing import Optional, Dict, List, Any
import multiprocessing as mp
from multiprocessing.synchronize import Event
import queue

from daaam.utils.logging import PipelineLogger, get_default_logger
from daaam.config import WorkerConfig


class GroundingService:
	"""Service for handling semantic grounding operations."""
	
	def __init__(self, config: WorkerConfig, logger: Optional[PipelineLogger] = None):
		self.config = config
		self.logger = logger or get_default_logger()
		# DAM initializes CUDA in a child process, so every synchronization
		# primitive crossing that boundary must come from the spawn context.
		# Do not mutate multiprocessing's process-wide default context here.
		self._mp_context = mp.get_context("spawn")
		self.workers: List[mp.Process] = []
		self.stop_event: Optional[Event] = None
		self.query_group_queue: Optional[mp.Queue] = None
		self.correction_queue: Optional[mp.Queue] = None
		self.worker_ready_queue: Optional[mp.Queue] = None
		self._worker_status: Dict[str, Dict[str, Any]] = {}
		self._last_worker_health: Optional[Dict[str, Any]] = None
		self._running = False
	
	def start(self, query_group_queue: mp.Queue, correction_queue: mp.Queue, pipeline_config, output_dir=None, color_map=None, log_dir=None) -> None:
		"""Start grounding worker processes."""
		if self._running:
			self.logger.warning("Grounding service already running")
			return
		
		self.query_group_queue = query_group_queue
		self.correction_queue = correction_queue
		self.stop_event = self._mp_context.Event()
		self.worker_ready_queue = self._mp_context.Queue()
		self._worker_status = {}
		self._last_worker_health = None
		
		# Get worker-specific configuration - ensure proper parameter passing for multi_image_min_n_masks
		if hasattr(pipeline_config, 'get_worker_config'):
			worker_config = pipeline_config.get_worker_config(self.config.grounding_worker)
		else:
			# Fallback: construct worker config manually
			if self.config.grounding_worker == "dam_multi_image":
				worker_config = {
					"grounding_worker": self.config.grounding_worker,
					**self.config.dam_grounding_config.__dict__
				}
			else:
				worker_config = {"grounding_worker": self.config.grounding_worker}
		
		# Add output_dir to worker config if provided
		if output_dir:
			worker_config['output_dir'] = output_dir
		
		# Add log_dir to worker config if provided
		if log_dir:
			worker_config['log_dir'] = log_dir

		# Propagate CUDA device selection to worker subprocess
		if hasattr(pipeline_config, 'grounding') and pipeline_config.grounding.cuda_device is not None:
			worker_config['cuda_device'] = str(pipeline_config.grounding.cuda_device)
			
		# Add color_map to worker config if provided
		if color_map:
			worker_config['color_map'] = color_map
		
		# Start grounding worker processes
		self._running = True
		try:
			for i in range(self.config.num_grounding_workers):
				worker = self._mp_context.Process(
					target=self._get_grounding_worker_process(),
					args=(
						self.query_group_queue,
						self.correction_queue,
						self.stop_event,
						worker_config,
						self.worker_ready_queue,
					),
					name=f"GroundingWorker-{i}"
				)
				worker.start()
				self.workers.append(worker)
		except BaseException:
			# A later Process.start() may fail after earlier workers are live.
			# Roll those workers back, but never replace the spawn exception.
			try:
				self.stop()
			except BaseException as cleanup_error:
				self.logger.error(
					f"Grounding startup rollback failed: {cleanup_error!r}"
				)
			raise

		self.logger.info(f"Started {self.config.num_grounding_workers} {self.config.grounding_worker} grounding workers")
	
	def stop(self) -> None:
		"""Stop grounding worker processes.

		Shutdown sequence:
		  1. Set stop_event (workers check between batches)
		  2. Join with 20s timeout (DAM batch can take 5-15s)
		  3. Send SIGINT to survivors — allows clean queue flush via KeyboardInterrupt handler
		  4. Join again with 5s
		  5. Force terminate as last resort
		"""
		if not self._running and not self.workers and self.stop_event is None:
			return

		cleanup_errors = []
		if self.stop_event:
			try:
				self.stop_event.set()
			except BaseException as error:
				cleanup_errors.append(error)

		for worker in self.workers:
			try:
				worker.join(timeout=20.0)
				if worker.is_alive():
					# Try SIGINT first — allows clean queue flush in worker's finally block
					self.logger.warning(f"Sending SIGINT to grounding worker {worker.name}")
					try:
						import os
						import signal
						os.kill(worker.pid, signal.SIGINT)
					except (ProcessLookupError, OSError):
						pass
					worker.join(timeout=5.0)
					if worker.is_alive():
						self.logger.warning(f"Force terminating grounding worker {worker.name}")
						worker.terminate()
						worker.join(timeout=2.0)
			except BaseException as error:
				cleanup_errors.append(error)

		self._running = False
		try:
			self._drain_worker_status()
			self._last_worker_health = self._build_worker_health()
		except BaseException as error:
			cleanup_errors.append(error)
		self.workers.clear()
		self.stop_event = None
		self.logger.info("Stopped grounding workers")
		if cleanup_errors:
			raise cleanup_errors[0]
	
	def _get_grounding_worker_process(self):
		"""Get grounding worker process function dynamically."""
		grounding_worker_name = self.config.grounding_worker
		
		if grounding_worker_name == "dam_multi_image":
			from .workers.dam_grounding import DAMGroundingWorkerMultiImage
			return DAMGroundingWorkerMultiImage.create_worker
		else:
			self.logger.error(f"Unknown grounding worker: {grounding_worker_name}")
			raise ValueError(f"Unknown grounding worker: {grounding_worker_name}")
	
	def is_running(self) -> bool:
		"""Check if grounding service is running."""
		return self._running
	
	def get_worker_health(self) -> Dict[str, Any]:
		"""Get health status of grounding workers."""
		self._drain_worker_status()
		if not self.workers and self._last_worker_health is not None:
			return dict(self._last_worker_health)
		return self._build_worker_health()

	def _build_worker_health(self) -> Dict[str, Any]:
		"""Build a stable readiness snapshot, including after worker shutdown."""
		health_status = {
			"running": self._running,
			"num_workers": len(self.workers),
			"workers": []
		}
		
		for worker in self.workers:
			status = self._worker_status.get(worker.name, {})
			health_status["workers"].append({
				"name": worker.name,
				"pid": worker.pid,
				"is_alive": worker.is_alive(),
				"is_ready": bool(status.get("ready", False)),
				"error": status.get("error"),
				"exitcode": worker.exitcode
			})
		health_status["ready_count"] = sum(
			bool(worker["is_ready"]) for worker in health_status["workers"]
		)
		health_status["all_ready"] = bool(health_status["workers"]) and all(
			bool(worker["is_ready"]) and not worker["error"]
			for worker in health_status["workers"]
		)
		return health_status

	def _drain_worker_status(self) -> None:
		"""Cache structured readiness without requeueing and duplicating messages."""
		if not self.worker_ready_queue:
			return
		while True:
			try:
				message = self.worker_ready_queue.get_nowait()
			except queue.Empty:
				break
			if isinstance(message, dict):
				name = str(message.get("worker", ""))
				if name:
					self._worker_status[name] = {
						"ready": bool(message.get("ready", False)),
						"error": message.get("error"),
					}
			else:
				# Backward-compatible status from older/custom workers.
				self._worker_status[str(message)] = {"ready": True, "error": None}
	
	def check_worker_ready(self, worker_name: str) -> bool:
		"""Check if a specific worker has reported ready."""
		self._drain_worker_status()
		return bool(self._worker_status.get(worker_name, {}).get("ready", False))
