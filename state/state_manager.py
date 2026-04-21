"""Runtime state manager for operating modes."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from config.constants import RuntimeMode
from config.settings import AppSettings
from core.event_manager import EngineEvent, EventManager, EventType
from core.exceptions import StateTransitionError


@dataclass(frozen=True)
class RuntimeState:
    """Current state-machine snapshot."""

    mode: RuntimeMode
    previous_mode: RuntimeMode | None
    entered_at: float
    reason: str
    sequence: int

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.entered_at


class StateManager:
    """Owns mode transitions and safety overrides."""

    def __init__(
        self,
        settings: AppSettings,
        event_manager: EventManager,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._events = event_manager
        self._logger = logger or logging.getLogger(__name__)
        self._state = RuntimeState(
            mode=RuntimeMode.NORMAL,
            previous_mode=None,
            entered_at=time.monotonic(),
            reason="startup",
            sequence=0,
        )
        self._last_danger_at: float | None = None
        self._manual_quiet = False
        self._thermal_pressure = False
        self._cpu_pressure = False

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def mode(self) -> RuntimeMode:
        return self._state.mode

    def request_transition(self, target_mode: RuntimeMode, reason: str, payload: dict[str, Any] | None = None) -> None:
        self._events.publish_type(
            EventType.STATE_TRANSITION_REQUEST,
            source="state_manager",
            payload={"target_mode": target_mode.value, "reason": reason, **(payload or {})},
        )

    def handle_event(self, event: EngineEvent) -> None:
        if event.event_type == EventType.STATE_TRANSITION_REQUEST:
            raw_mode = event.payload.get("target_mode")
            if not isinstance(raw_mode, str):
                raise StateTransitionError("State transition request missing target_mode.")
            self.transition_to(RuntimeMode(raw_mode), str(event.payload.get("reason", event.source)))
        elif event.event_type == EventType.OBJECT_DETECTIONS:
            max_severity = float(event.payload.get("max_severity", 0.0))
            if max_severity >= self._settings.thresholds.objects.severity_danger_threshold:
                self._last_danger_at = time.monotonic()
                if self.mode != RuntimeMode.DANGER:
                    self.transition_to(RuntimeMode.DANGER, "object danger threshold exceeded")
        elif event.event_type == EventType.PERFORMANCE_WARNING:
            warning = event.payload.get("warning")
            if warning in {"thermal_hard_limit", "thermal_soft_limit"}:
                self._thermal_pressure = True
                if self.mode == RuntimeMode.DANGER:
                    self._logger.warning("Low-power pressure deferred while DANGER is active: %s", warning)
                    return
                if self.mode != RuntimeMode.LOW_POWER:
                    self.transition_to(RuntimeMode.LOW_POWER, str(warning))
            elif warning == "cpu_over_budget":
                self._cpu_pressure = True
                if self.mode == RuntimeMode.DANGER:
                    self._logger.warning("Low-power pressure deferred while DANGER is active: %s", warning)
                    return
                if self.mode != RuntimeMode.LOW_POWER:
                    self.transition_to(RuntimeMode.LOW_POWER, str(warning))
            elif warning == "thermal_recovered":
                self._thermal_pressure = False
                self._exit_low_power_if_clear("thermal recovered")
            elif warning == "cpu_recovered":
                self._cpu_pressure = False
                self._exit_low_power_if_clear("cpu recovered")
        elif event.event_type == EventType.SHUTDOWN_REQUESTED:
            self._logger.info("Shutdown requested: %s", event.payload.get("reason", "unspecified"))

    def transition_to(self, target_mode: RuntimeMode, reason: str, allow_danger_exit: bool = False) -> RuntimeState:
        """Move to a new mode when policy allows it."""

        current = self._state.mode
        if target_mode == current:
            return self._state

        if current == RuntimeMode.DANGER and not allow_danger_exit:
            raise StateTransitionError(f"Cannot leave DANGER for {target_mode.value} before danger clears.")
        if target_mode == RuntimeMode.QUIET:
            self._manual_quiet = True
        elif current == RuntimeMode.QUIET and target_mode == RuntimeMode.NORMAL:
            self._manual_quiet = False

        previous = self._state
        self._state = RuntimeState(
            mode=target_mode,
            previous_mode=previous.mode,
            entered_at=time.monotonic(),
            reason=reason,
            sequence=previous.sequence + 1,
        )
        self._logger.info("Mode changed %s -> %s: %s", previous.mode.value, target_mode.value, reason)
        self._events.publish_type(
            EventType.STATE_CHANGED,
            source="state_manager",
            payload={
                "mode": target_mode.value,
                "previous_mode": previous.mode.value,
                "reason": reason,
                "sequence": self._state.sequence,
            },
        )
        return self._state

    def update(self) -> RuntimeState:
        """Apply time-based transitions such as DANGER clear."""

        if self.mode == RuntimeMode.DANGER:
            danger_clear_after = self._settings.cooldowns.danger_clear_seconds
            last_danger = self._last_danger_at or self._state.entered_at
            if time.monotonic() - last_danger >= danger_clear_after:
                if self._low_power_requested:
                    next_mode = RuntimeMode.LOW_POWER
                elif self._manual_quiet:
                    next_mode = RuntimeMode.QUIET
                else:
                    next_mode = RuntimeMode.NORMAL
                self.transition_to(next_mode, "danger clear cooldown elapsed", allow_danger_exit=True)
        return self._state

    @property
    def _low_power_requested(self) -> bool:
        return self._thermal_pressure or self._cpu_pressure

    def _exit_low_power_if_clear(self, reason: str) -> None:
        if self._low_power_requested or self.mode != RuntimeMode.LOW_POWER:
            return
        target = RuntimeMode.QUIET if self._manual_quiet else RuntimeMode.NORMAL
        self.transition_to(target, reason)
