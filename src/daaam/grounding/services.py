from typing import Optional, Dict, List, Any
import multiprocessing as mp
from multiprocessing.synchronize import Event
import queue
import time
import uuid

from daaam.utils.logging import PipelineLogger, get_default_logger
from daaam.config import WorkerConfig
from daaam.grounding.control import (
	GroundingCorrectionsDrained,
	GroundingDrainRequest,
)


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
		self._shutdown_status: Dict[str, Any] = {}
		self._running = False
	
	def start(self, query_group_queue: mp.Queue, correction_queue: mp.Queue, pipeline_config, output_dir=None, color_map=None, log_dir=None) -> None:
		"""Start grounding worker processes."""
		if self._running:
			self.logger.warning("Grounding service already running")
			return
		if self.workers:
			raise RuntimeError("grounding workers from a previous stop are still alive")
		
		self.query_group_queue = query_group_queue
		self.correction_queue = correction_queue
		self.stop_event = self._mp_context.Event()
		self.worker_ready_queue = self._mp_context.Queue()
		self._worker_status = {}
		self._last_worker_health = None
		self._shutdown_status = {}
		
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
				self.stop(timeout_s=5.0, drain=False)
			except BaseException as cleanup_error:
				self.logger.error(
					f"Grounding startup rollback failed: {cleanup_error!r}"
				)
			raise

		self.logger.info(f"Started {self.config.num_grounding_workers} {self.config.grounding_worker} grounding workers")
	
	def stop(
		self,
		timeout_s: float = 30.0,
		*,
		drain: bool = False,
	) -> Dict[str, Any]:
		"""Stop workers within one budget, optionally draining every FIFO prompt."""
		if timeout_s <= 0.0:
			raise ValueError("grounding stop timeout must be positive")
		if not self._running and not self.workers and self.stop_event is None:
			return dict(self._shutdown_status)

		started_at = time.monotonic()
		deadline = started_at + timeout_s
		# Reserve a small part of the caller's total budget for forced cleanup.
		cleanup_reserve_s = min(1.0, max(0.05, timeout_s * 0.1))
		graceful_deadline = max(started_at, deadline - cleanup_reserve_s)
		drain_token = uuid.uuid4().hex if drain else None
		expected_workers = [worker.name for worker in self.workers]
		self._shutdown_status = {
			"drain_requested": bool(drain),
			"drain_complete": False,
			"drain_token": drain_token,
			"correction_tail_enqueued": False,
			"timeout_s": float(timeout_s),
			"timed_out": False,
			"forced_termination": False,
			"error": None,
		}
		primary_error: Optional[BaseException] = None
		cleanup_errors: List[BaseException] = []

		try:
			if drain and expected_workers:
				self._enqueue_drain_requests(
					drain_token,
					expected_workers,
					graceful_deadline,
				)
				self._wait_for_drain_acknowledgements(
					drain_token,
					expected_workers,
					graceful_deadline,
				)
				self._join_drained_workers(graceful_deadline)
				self._enqueue_correction_tail(drain_token, graceful_deadline)
				self._shutdown_status["drain_complete"] = True
			elif drain:
				self._enqueue_correction_tail(drain_token, graceful_deadline)
				self._shutdown_status["drain_complete"] = True
			else:
				self._set_stop_event()
				cleanup_errors.extend(self._cancel_and_reap(deadline))
		except BaseException as error:
			primary_error = error
			self._shutdown_status["error"] = repr(error)
			if isinstance(error, TimeoutError):
				self._shutdown_status["timed_out"] = True
				self._mark_unacknowledged_workers_timed_out(
					drain_token,
					expected_workers,
				)
			try:
				self._set_stop_event()
			except BaseException as cleanup_error:
				cleanup_errors.append(cleanup_error)
			cleanup_errors.extend(self._cancel_and_reap(deadline))
		finally:
			self._running = False
			try:
				self._drain_worker_status()
			except BaseException as error:
				cleanup_errors.append(error)
			try:
				self._record_worker_exit_state()
			except BaseException as error:
				cleanup_errors.append(error)
			self._shutdown_status["forced_termination"] = any(
				bool(status.get("forced_termination", False))
				for status in self._worker_status.values()
			)
			if primary_error is None and cleanup_errors:
				primary_error = cleanup_errors[0]
				self._shutdown_status["error"] = repr(primary_error)
			try:
				self._last_worker_health = self._build_worker_health()
			except BaseException as error:
				cleanup_errors.append(error)
				if primary_error is None:
					primary_error = error
					self._shutdown_status["error"] = repr(error)
			try:
				self._release_stopped_resources()
			except BaseException as error:
				cleanup_errors.append(error)
				if primary_error is None:
					primary_error = error
					self._shutdown_status["error"] = repr(error)

		if primary_error is not None:
			self.logger.error(
				f"Grounding worker shutdown failed: {primary_error!r}"
			)
			raise primary_error
		self.logger.info("Stopped grounding workers")
		return dict(self._shutdown_status)

	def _remaining(self, deadline: float, operation: str) -> float:
		remaining = deadline - time.monotonic()
		if remaining <= 0.0:
			raise TimeoutError(f"timed out while {operation}")
		return remaining

	def _worker_status_for(self, worker_name: str) -> Dict[str, Any]:
		return self._worker_status.setdefault(
			worker_name,
			{"ready": False, "error": None},
		)

	def _enqueue_drain_requests(
		self,
		drain_token: str,
		expected_workers: List[str],
		deadline: float,
	) -> None:
		if self.query_group_queue is None:
			raise RuntimeError("grounding prompt queue is unavailable")
		workers_by_name = {worker.name: worker for worker in self.workers}
		for worker_name in expected_workers:
			worker = workers_by_name[worker_name]
			if not worker.is_alive():
				raise RuntimeError(
					f"grounding worker {worker_name} exited before drain request"
				)
			try:
				self.query_group_queue.put(
					GroundingDrainRequest(drain_token),
					timeout=self._remaining(
						deadline,
						f"enqueueing drain request for {worker_name}",
					),
				)
			except queue.Full as error:
				raise TimeoutError(
					f"prompt queue stayed full while draining {worker_name}"
				) from error
			self._worker_status_for(worker_name)["drain_request_enqueued"] = True

	def _wait_for_drain_acknowledgements(
		self,
		drain_token: str,
		expected_workers: List[str],
		deadline: float,
	) -> None:
		if self.worker_ready_queue is None:
			raise RuntimeError("grounding worker status queue is unavailable")
		workers_by_name = {worker.name: worker for worker in self.workers}
		while True:
			self._drain_worker_status()
			missing = []
			for worker_name in expected_workers:
				status = self._worker_status_for(worker_name)
				if status.get("drain_token") == drain_token:
					if status.get("drained") is False:
						raise RuntimeError(
							f"grounding worker {worker_name} failed to drain: "
							f"{status.get('error')}"
						)
					if status.get("drained") is True:
						continue
				worker = workers_by_name[worker_name]
				if not worker.is_alive() and worker.exitcode not in (None, 0):
					raise RuntimeError(
						f"grounding worker {worker_name} exited with code "
						f"{worker.exitcode} before drain acknowledgement"
					)
				missing.append(worker_name)
			if not missing:
				return
			remaining = self._remaining(
				deadline,
				"waiting for grounding drain acknowledgements "
				+ ", ".join(missing),
			)
			try:
				message = self.worker_ready_queue.get(timeout=min(0.05, remaining))
			except queue.Empty:
				continue
			self._record_worker_status(message)

	def _join_drained_workers(self, deadline: float) -> None:
		for worker in self.workers:
			worker.join(
				timeout=self._remaining(
					deadline,
					f"joining drained worker {worker.name}",
				)
			)
			if worker.is_alive():
				raise TimeoutError(
					f"grounding worker {worker.name} acknowledged drain but did not exit"
				)
			if worker.exitcode not in (None, 0):
				raise RuntimeError(
					f"grounding worker {worker.name} exited with code {worker.exitcode}"
				)

	def _enqueue_correction_tail(self, drain_token: str, deadline: float) -> None:
		if self.correction_queue is None:
			raise RuntimeError("grounding correction queue is unavailable")
		try:
			self.correction_queue.put(
				GroundingCorrectionsDrained(drain_token),
				timeout=self._remaining(
					deadline,
					"enqueueing the correction drain marker",
				),
			)
		except queue.Full as error:
			raise TimeoutError(
				"correction queue stayed full while publishing drain completion"
			) from error
		self._shutdown_status["correction_tail_enqueued"] = True

	def _set_stop_event(self) -> None:
		if self.stop_event is not None:
			self.stop_event.set()

	def _cancel_and_reap(self, deadline: float) -> List[BaseException]:
		"""Cancel promptly, then terminate survivors within the total deadline."""
		errors: List[BaseException] = []
		remaining = max(0.0, deadline - time.monotonic())
		grace_deadline = min(
			deadline,
			time.monotonic() + min(0.25, remaining * 0.25),
		)
		for worker in self.workers:
			try:
				worker.join(timeout=max(0.0, grace_deadline - time.monotonic()))
			except BaseException as error:
				errors.append(error)
		for worker in self.workers:
			try:
				alive = worker.is_alive()
			except BaseException as error:
				errors.append(error)
				continue
			if not alive:
				continue
			status = self._worker_status_for(worker.name)
			status["forced_termination"] = True
			self.logger.warning(f"Force terminating grounding worker {worker.name}")
			try:
				worker.terminate()
			except BaseException as error:
				errors.append(error)
		for worker in self.workers:
			try:
				worker.join(timeout=max(0.0, deadline - time.monotonic()))
				if worker.is_alive():
					status = self._worker_status_for(worker.name)
					status["timed_out"] = True
					errors.append(
						TimeoutError(
							f"grounding worker {worker.name} survived forced termination"
						)
					)
			except BaseException as error:
				errors.append(error)
		return errors

	def _mark_unacknowledged_workers_timed_out(
		self,
		drain_token: Optional[str],
		expected_workers: List[str],
	) -> None:
		for worker_name in expected_workers:
			status = self._worker_status_for(worker_name)
			if not (
				status.get("drain_token") == drain_token
				and status.get("drained") is True
			):
				status["timed_out"] = True

	def _record_worker_exit_state(self) -> None:
		for worker in self.workers:
			status = self._worker_status_for(worker.name)
			status["exitcode"] = worker.exitcode
			status["is_alive"] = worker.is_alive()

	def _release_stopped_resources(self) -> None:
		survivors = [worker for worker in self.workers if worker.is_alive()]
		if survivors:
			self.workers = survivors
			return
		self.workers.clear()
		self.stop_event = None
		if self.worker_ready_queue is not None:
			try:
				close = getattr(self.worker_ready_queue, "close", None)
				if close is not None:
					close()
				join_thread = getattr(self.worker_ready_queue, "join_thread", None)
				if join_thread is not None:
					join_thread()
			except (OSError, ValueError):
				pass
		self.worker_ready_queue = None
	
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
		if not self._running and self._last_worker_health is not None:
			return dict(self._last_worker_health)
		return self._build_worker_health()

	def _build_worker_health(self) -> Dict[str, Any]:
		"""Build a stable readiness snapshot, including after worker shutdown."""
		health_status = {
			"running": self._running,
			"num_workers": len(self.workers),
			"workers": [],
			"shutdown": dict(self._shutdown_status),
		}
		
		for worker in self.workers:
			status = self._worker_status.get(worker.name, {})
			health_status["workers"].append({
				"name": worker.name,
				"pid": worker.pid,
				"is_alive": worker.is_alive(),
				"is_ready": bool(status.get("ready", False)),
				"error": status.get("error"),
				"exitcode": worker.exitcode,
				"drain_request_enqueued": bool(
					status.get("drain_request_enqueued", False)
				),
				"drain_token": status.get("drain_token"),
				"drained": bool(status.get("drained", False)),
				"timed_out": bool(status.get("timed_out", False)),
				"forced_termination": bool(
					status.get("forced_termination", False)
				),
			})
		health_status["ready_count"] = sum(
			bool(worker["is_ready"]) for worker in health_status["workers"]
		)
		health_status["all_ready"] = bool(health_status["workers"]) and all(
			bool(worker["is_ready"]) and not worker["error"]
			for worker in health_status["workers"]
		)
		return health_status

	def _record_worker_status(self, message: Any) -> None:
		if isinstance(message, dict):
			name = str(message.get("worker", ""))
			if not name:
				return
			status = self._worker_status_for(name)
			for key in (
				"ready",
				"error",
				"drain_token",
				"drained",
			):
				if key in message:
					status[key] = message[key]
			return
		# Backward-compatible status from older/custom workers.
		self._worker_status_for(str(message)).update(
			{"ready": True, "error": None}
		)

	def _drain_worker_status(self) -> None:
		"""Cache structured readiness without requeueing and duplicating messages."""
		if not self.worker_ready_queue:
			return
		while True:
			try:
				message = self.worker_ready_queue.get_nowait()
			except queue.Empty:
				break
			self._record_worker_status(message)
	
	def check_worker_ready(self, worker_name: str) -> bool:
		"""Check if a specific worker has reported ready."""
		self._drain_worker_status()
		return bool(self._worker_status.get(worker_name, {}).get("ready", False))
