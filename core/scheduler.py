"""Main runtime scheduler that coordinates managers without owning inference logic."""

from __future__ import annotations

import logging
import time
from threading import Event

from camera.camera_manager import CameraManager, CapturedFrame
from config.constants import InferenceBackend, ModelAssetTier, ModuleName, OcrPolicy, ValidationSeverity
from config.settings import AppSettings, validate_settings
from core.event_manager import EngineEvent, EventManager, EventType
from core.exceptions import SchedulerError, build_error_report, is_recoverable
from core.frame_store import FrameStore
from core.model_runtime import LocalModelLoader
from decision.decision_engine import DecisionEngine
from diagnostics.performance_monitor import PerformanceMonitor
from state.state_manager import StateManager


CAMERA_RESTART_FAILURES = 5
CAMERA_FATAL_FAILURES = 20
EVENT_DISPATCH_SAFETY_MULTIPLIER = 4


class EngineScheduler:
    """Frame-paced scheduler for camera, state, decisions, and diagnostics."""

    def __init__(
        self,
        settings: AppSettings,
        event_manager: EventManager,
        frame_store: FrameStore,
        camera_manager: CameraManager,
        state_manager: StateManager,
        decision_engine: DecisionEngine,
        performance_monitor: PerformanceMonitor,
        service_handlers: tuple[object, ...] = (),
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._events = event_manager
        self._frame_store = frame_store
        self._camera = camera_manager
        self._state = state_manager
        self._decision = decision_engine
        self._performance = performance_monitor
        self._service_handlers = service_handlers
        self._logger = logger or logging.getLogger(__name__)
        self._stop_requested = Event()
        self._frame_counter = 0
        self._consecutive_camera_failures = 0

    @property
    def running(self) -> bool:
        return not self._stop_requested.is_set()

    def request_stop(self, reason: str = "stop requested") -> None:
        self._events.publish_type(EventType.SHUTDOWN_REQUESTED, source="scheduler", payload={"reason": reason})
        self._stop_requested.set()

    def run(self, max_frames: int | None = None, require_valid_settings: bool = True) -> int:
        """Run until stopped. Returns process-style exit code."""

        exit_code = 0
        try:
            self._validate_for_startup(require_valid_settings)
            self._start_services()
            self._logger.info("Starting scheduler")
            self._camera.start()
            self._events.publish_type(
                EventType.CAMERA_STARTED,
                source="scheduler",
                payload={"backend": self._camera.backend, "mode": self._state.mode.value},
            )
            self._events.publish_type(
                EventType.SYSTEM_READY,
                source="scheduler",
                payload={"mode": self._state.mode.value},
            )

            while not self._stop_requested.is_set():
                loop_started = time.monotonic()
                with self._performance.measure_loop():
                    self._run_once()
                    if max_frames is not None and self._frame_counter >= max_frames:
                        self.request_stop("max frames reached")
                self._sleep_for_profile(loop_started)
        except KeyboardInterrupt:
            self.request_stop("keyboard interrupt")
        except Exception as exc:
            exit_code = self._handle_runtime_exception(exc)
        finally:
            if self._camera.is_started:
                self._camera.stop()
                self._events.publish_type(EventType.CAMERA_STOPPED, source="scheduler")
            self._stop_services()
            self._logger.info("Scheduler stopped")
        return exit_code

    def _validate_for_startup(self, require_valid_settings: bool) -> None:
        issues = validate_settings(self._settings)
        errors = [issue for issue in issues if issue.severity == ValidationSeverity.ERROR]
        for issue in issues:
            log_method = self._logger.error if issue.severity == ValidationSeverity.ERROR else self._logger.warning
            log_method("Configuration validation: %s", issue.message)
        if require_valid_settings and errors:
            raise SchedulerError(f"Startup blocked by {len(errors)} configuration error(s).")
        if require_valid_settings:
            self._validate_safety_model_runtimes()

    def _validate_safety_model_runtimes(self) -> None:
        loader = LocalModelLoader(self._logger)
        for asset in self._settings.models.installed_model_files():
            if asset.tier != ModelAssetTier.SAFETY_CRITICAL:
                continue
            if asset.backend in {InferenceBackend.STATIC_FILE, InferenceBackend.TESSERACT}:
                continue
            self._logger.info("Probing safety-critical model backend: %s", asset.asset_id)
            loader.probe(
                asset.path,
                preferred_backend=asset.backend,
                num_threads=self._settings.optimization.tflite_threads,
                input_shape=asset.input_shape,
                run_inference=True,
            )

    def _run_once(self) -> None:
        self._state.update()
        profile = self._settings.profile_for(self._state.mode)
        frame = self._capture_frame()
        if frame is not None:
            self._frame_counter = frame.frame_id
            self._frame_store.put(frame)
            self._publish_frame(frame)
            self._publish_module_triggers(frame, profile.active_modules)
        self._process_events()
        self._performance.maybe_publish_sample(self._state.mode)

    def _start_services(self) -> None:
        for handler in self._service_handlers:
            start = getattr(handler, "start", None)
            if start is not None:
                start()

    def _stop_services(self) -> None:
        for handler in reversed(self._service_handlers):
            stop = getattr(handler, "stop", None)
            if stop is not None:
                try:
                    stop()
                except Exception as exc:
                    self._logger.warning("Service stop failed for %s: %s", type(handler).__name__, exc)

    def _capture_frame(self) -> CapturedFrame | None:
        try:
            frame = self._camera.read()
        except Exception as exc:
            self._consecutive_camera_failures += 1
            report = build_error_report(exc, "camera")
            self._events.publish_type(EventType.ERROR, source="scheduler", payload=report.__dict__)
            self._performance.frame_dropped()
            if self._consecutive_camera_failures >= CAMERA_FATAL_FAILURES:
                raise SchedulerError(
                    f"Camera failed {self._consecutive_camera_failures} consecutive times; stopping runtime."
                ) from exc
            if self._consecutive_camera_failures % CAMERA_RESTART_FAILURES == 0:
                try:
                    self._restart_camera()
                except Exception as restart_exc:
                    restart_report = build_error_report(restart_exc, "camera_restart")
                    self._events.publish_type(EventType.ERROR, source="scheduler", payload=restart_report.__dict__)
                    self._logger.warning("Camera restart failed: %s", restart_report.message)
            if not report.recoverable:
                raise
            self._logger.warning("Recoverable camera error: %s", report.message)
            return None

        self._consecutive_camera_failures = 0
        if self._settings.camera.drop_stale_frames and frame.age_ms > self._settings.camera.max_frame_age_ms:
            self._events.publish_type(
                EventType.FRAME_DROPPED,
                source="scheduler",
                payload={"frame_id": frame.frame_id, "age_ms": frame.age_ms},
            )
            self._performance.frame_dropped()
            return None
        self._performance.frame_captured(frame.age_ms)
        return frame

    def _restart_camera(self) -> None:
        self._logger.warning("Restarting camera after repeated capture failures")
        if self._camera.is_started:
            self._camera.stop()
            self._events.publish_type(EventType.CAMERA_STOPPED, source="scheduler", payload={"reason": "restart"})
        time.sleep(0.25)
        self._camera.start()
        self._events.publish_type(
            EventType.CAMERA_STARTED,
            source="scheduler",
            payload={"backend": self._camera.backend, "reason": "restart"},
        )

    def _publish_frame(self, frame: CapturedFrame) -> None:
        self._events.publish_type(
            EventType.FRAME_CAPTURED,
            source="camera_manager",
            payload={
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "width": frame.width,
                "height": frame.height,
                "backend": frame.backend,
                "age_ms": frame.age_ms,
            },
        )

    def _publish_module_triggers(self, frame: CapturedFrame, active_modules: tuple[ModuleName, ...]) -> None:
        profile = self._settings.profile_for(self._state.mode)
        schedules = {
            ModuleName.OBJECT_DETECTION: profile.object_detection_every_n_frames,
            ModuleName.FACE_DETECTION: profile.face_detection_every_n_frames,
        }
        if ModuleName.OCR in active_modules and profile.ocr_policy == OcrPolicy.ACTIVE_REQUEST:
            schedules[ModuleName.OCR] = 1
        for module, interval in schedules.items():
            if module not in active_modules or interval is None:
                continue
            if frame.frame_id % interval == 0:
                self._events.publish_type(
                    EventType.MODULE_TRIGGER,
                    source="scheduler",
                    payload={
                        "module": module.value,
                        "mode": self._state.mode.value,
                        "frame_id": frame.frame_id,
                        "frame_timestamp": frame.timestamp,
                    },
                )

    def _process_events(self) -> None:
        processed = 0
        normal_budget = self._settings.decision_queue_size
        safety_limit = max(self._settings.decision_queue_size * EVENT_DISPATCH_SAFETY_MULTIPLIER, 64)
        while processed < safety_limit:
            if self._events.critical_count() > 0:
                batch = self._events.drain(max_events=safety_limit - processed, critical_only=True)
            elif normal_budget > 0:
                batch = self._events.drain(max_events=1)
                normal_budget -= len(batch)
            else:
                break
            if not batch:
                break
            for event in batch:
                self._dispatch_event(event)
                processed += 1
                if processed >= safety_limit:
                    break
        if self._events.critical_count() > 0:
            self._logger.error("Critical event dispatch safety limit reached with events still queued")
        elif self._events.normal_count() > 0:
            self._logger.debug("Normal events deferred to next cycle: %s", self._events.normal_count())

    def _dispatch_event(self, event: EngineEvent) -> None:
        try:
            self._state.handle_event(event)
            for handler in self._service_handlers:
                handle_event = getattr(handler, "handle_event", None)
                if handle_event is not None:
                    handle_event(event)
            self._decision.handle_event(event)
            if event.event_type == EventType.AUDIO_INTENT:
                self._logger.info(
                    "Audio intent dispatched: priority=%s text=%s",
                    event.payload.get("priority"),
                    event.payload.get("text"),
                )
            elif event.event_type == EventType.MODULE_TRIGGER:
                self._logger.debug(
                    "Module trigger: module=%s frame=%s mode=%s",
                    event.payload.get("module"),
                    event.payload.get("frame_id"),
                    event.payload.get("mode"),
                )
        except Exception as exc:
            report = build_error_report(exc, event.source)
            self._events.publish_type(EventType.ERROR, source="scheduler", payload=report.__dict__)
            if not is_recoverable(exc):
                raise
            self._logger.warning("Recoverable event dispatch error: %s", report.message)

    def _sleep_for_profile(self, loop_started: float) -> None:
        profile = self._settings.profile_for(self._state.mode)
        target_frame_seconds = 1.0 / max(1, profile.camera_fps)
        elapsed = time.monotonic() - loop_started
        remaining = target_frame_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _handle_runtime_exception(self, exc: Exception) -> int:
        report = build_error_report(exc, "scheduler")
        self._events.publish_type(EventType.ERROR, source="scheduler", payload=report.__dict__)
        self._logger.error("Scheduler failed: %s", report.message)
        self._logger.debug(report.traceback)
        return 1
