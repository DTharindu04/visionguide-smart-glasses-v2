"""Runtime performance and thermal monitoring."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from config.constants import RuntimeMode
from config.settings import AppSettings
from core.event_manager import EventManager, EventType


@dataclass(frozen=True)
class PerformanceSnapshot:
    """Point-in-time performance metrics."""

    timestamp: float
    mode: RuntimeMode
    cpu_percent: float | None
    memory_percent: float | None
    temperature_c: float | None
    queued_events: int
    dropped_events: int
    frames_captured: int
    frames_dropped: int
    average_frame_ms: float | None
    average_loop_ms: float | None


class PerformanceMonitor:
    """Collects lightweight runtime metrics and emits warnings."""

    def __init__(
        self,
        settings: AppSettings,
        event_manager: EventManager,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._events = event_manager
        self._logger = logger or logging.getLogger(__name__)
        self._frames_captured = 0
        self._frames_dropped = 0
        self._frame_ms_total = 0.0
        self._frame_ms_count = 0
        self._loop_ms_total = 0.0
        self._loop_ms_count = 0
        self._last_sample_at = 0.0
        self._thermal_warning: str | None = None
        self._cpu_pressure_active = False
        self._psutil = self._load_psutil()

    def _load_psutil(self) -> object | None:
        try:
            import psutil  # type: ignore[import-not-found]
        except Exception:
            self._logger.warning("psutil unavailable; CPU and memory metrics disabled")
            return None
        return psutil

    def frame_captured(self, frame_age_ms: float) -> None:
        self._frames_captured += 1
        self._frame_ms_total += frame_age_ms
        self._frame_ms_count += 1

    def frame_dropped(self) -> None:
        self._frames_dropped += 1

    @contextmanager
    def measure_loop(self) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            self._loop_ms_total += elapsed_ms
            self._loop_ms_count += 1

    def maybe_publish_sample(self, mode: RuntimeMode) -> PerformanceSnapshot | None:
        now = time.monotonic()
        if now - self._last_sample_at < self._settings.diagnostics.metrics_interval_seconds:
            return None
        self._last_sample_at = now
        snapshot = self.sample(mode)
        self._events.publish_type(
            EventType.PERFORMANCE_SAMPLE,
            source="performance_monitor",
            payload={
                "mode": snapshot.mode.value,
                "cpu_percent": snapshot.cpu_percent,
                "memory_percent": snapshot.memory_percent,
                "temperature_c": snapshot.temperature_c,
                "queued_events": snapshot.queued_events,
                "dropped_events": snapshot.dropped_events,
                "frames_captured": snapshot.frames_captured,
                "frames_dropped": snapshot.frames_dropped,
                "average_frame_ms": snapshot.average_frame_ms,
                "average_loop_ms": snapshot.average_loop_ms,
            },
        )
        self._publish_warnings_if_needed(snapshot)
        return snapshot

    def sample(self, mode: RuntimeMode) -> PerformanceSnapshot:
        cpu_percent = None
        memory_percent = None
        if self._psutil is not None:
            cpu_percent = float(self._psutil.cpu_percent(interval=None))
            memory_percent = float(self._psutil.virtual_memory().percent)
        return PerformanceSnapshot(
            timestamp=time.monotonic(),
            mode=mode,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            temperature_c=self._read_temperature_c(),
            queued_events=self._events.queued_count(),
            dropped_events=self._events.dropped_events,
            frames_captured=self._frames_captured,
            frames_dropped=self._frames_dropped,
            average_frame_ms=self._average(self._frame_ms_total, self._frame_ms_count),
            average_loop_ms=self._average(self._loop_ms_total, self._loop_ms_count),
        )

    def _publish_warnings_if_needed(self, snapshot: PerformanceSnapshot) -> None:
        for warning in self._performance_warnings(snapshot):
            self._logger.warning("Performance warning: %s", warning)
            self._events.publish_type(
                EventType.PERFORMANCE_WARNING,
                source="performance_monitor",
                payload={
                    "warning": warning,
                    "mode": snapshot.mode.value,
                    "cpu_percent": snapshot.cpu_percent,
                    "temperature_c": snapshot.temperature_c,
                },
            )

    def _performance_warnings(self, snapshot: PerformanceSnapshot) -> tuple[str, ...]:
        warnings: list[str] = []
        if snapshot.temperature_c is not None:
            if snapshot.temperature_c >= self._settings.optimization.thermal_hard_limit_c:
                if self._thermal_warning != "thermal_hard_limit":
                    warnings.append("thermal_hard_limit")
                self._thermal_warning = "thermal_hard_limit"
            elif snapshot.temperature_c >= self._settings.optimization.thermal_soft_limit_c:
                if self._thermal_warning != "thermal_soft_limit":
                    warnings.append("thermal_soft_limit")
                self._thermal_warning = "thermal_soft_limit"
            elif (
                self._thermal_warning is not None
                and snapshot.temperature_c <= self._settings.optimization.thermal_soft_limit_c - 3.0
            ):
                warnings.append("thermal_recovered")
                self._thermal_warning = None

        profile = self._settings.profile_for(snapshot.mode)
        if snapshot.cpu_percent is not None:
            if not self._cpu_pressure_active and snapshot.cpu_percent >= profile.cpu_budget_percent:
                warnings.append("cpu_over_budget")
                self._cpu_pressure_active = True
            elif (
                self._cpu_pressure_active
                and snapshot.cpu_percent <= self._settings.optimization.low_power_exit_cpu_percent
            ):
                warnings.append("cpu_recovered")
                self._cpu_pressure_active = False
        return tuple(warnings)

    def _read_temperature_c(self) -> float | None:
        thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
        try:
            return int(thermal_path.read_text(encoding="utf-8").strip()) / 1000.0
        except Exception:
            return None

    @staticmethod
    def _average(total: float, count: int) -> float | None:
        if count == 0:
            return None
        return total / count
