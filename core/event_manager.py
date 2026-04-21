"""Thread-safe event bus for runtime modules."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from typing import Any, Callable, Deque, Mapping

from config.constants import AudioPriority, MAX_CRITICAL_EVENT_QUEUE_SIZE, MAX_DIAGNOSTIC_EVENT_QUEUE_SIZE
from core.exceptions import EventManagerError


class EventType(StrEnum):
    """Canonical runtime event types."""

    SYSTEM_READY = "system_ready"
    SHUTDOWN_REQUESTED = "shutdown_requested"
    CAMERA_STARTED = "camera_started"
    CAMERA_STOPPED = "camera_stopped"
    FRAME_CAPTURED = "frame_captured"
    FRAME_DROPPED = "frame_dropped"
    MODULE_TRIGGER = "module_trigger"
    OBJECT_DETECTIONS = "object_detections"
    FACE_DETECTED = "face_detected"
    FACE_RECOGNIZED = "face_recognized"
    OCR_RESULT = "ocr_result"
    OCR_REQUEST = "ocr_request"
    EMOTION_DETECTED = "emotion_detected"
    AUDIO_INTENT = "audio_intent"
    AUDIO_DROPPED = "audio_dropped"
    AUDIO_STARTED = "audio_started"
    AUDIO_SPOKEN = "audio_spoken"
    AUDIO_INTERRUPTED = "audio_interrupted"
    STATE_TRANSITION_REQUEST = "state_transition_request"
    STATE_CHANGED = "state_changed"
    PERFORMANCE_SAMPLE = "performance_sample"
    PERFORMANCE_WARNING = "performance_warning"
    ERROR = "error"


@dataclass(frozen=True)
class EngineEvent:
    """A single event flowing through the runtime."""

    event_type: EventType
    source: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @classmethod
    def create(
        cls,
        event_type: EventType,
        source: str,
        payload: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> "EngineEvent":
        return cls(
            event_type=event_type,
            source=source,
            payload=payload or {},
            correlation_id=correlation_id or uuid.uuid4().hex,
        )


EventHandler = Callable[[EngineEvent], None]


CRITICAL_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.SHUTDOWN_REQUESTED,
        EventType.STATE_TRANSITION_REQUEST,
        EventType.STATE_CHANGED,
        EventType.ERROR,
        EventType.PERFORMANCE_WARNING,
    }
)


def is_critical_event(event: EngineEvent) -> bool:
    """Return whether an event belongs on the non-droppable safety lane."""

    if event.event_type in CRITICAL_EVENT_TYPES:
        return True
    if event.event_type == EventType.OBJECT_DETECTIONS:
        try:
            return float(event.payload.get("max_severity", 0.0)) >= 0.70
        except (TypeError, ValueError):
            return False
    if event.event_type == EventType.AUDIO_INTENT:
        priority_value = event.payload.get("priority_value")
        priority_name = event.payload.get("priority")
        return priority_value == int(AudioPriority.P1_DANGER) or priority_name == AudioPriority.P1_DANGER.name
    return False


class EventManager:
    """Thread-safe event queue with a protected critical lane.

    Subscribers are observers for telemetry or tests. The scheduler owns core
    event dispatch by draining the queue, which avoids accidental double
    ownership of business logic.
    """

    def __init__(
        self,
        max_queue_size: int = MAX_DIAGNOSTIC_EVENT_QUEUE_SIZE,
        max_critical_queue_size: int = MAX_CRITICAL_EVENT_QUEUE_SIZE,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_queue_size <= 0:
            raise EventManagerError("Event queue size must be greater than zero.")
        if max_critical_queue_size <= 0:
            raise EventManagerError("Critical event queue size must be greater than zero.")
        self._normal_events: Deque[EngineEvent] = deque()
        self._critical_events: Deque[EngineEvent] = deque()
        self._max_queue_size = max_queue_size
        self._max_critical_queue_size = max_critical_queue_size
        self._observers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._wildcard_observers: list[EventHandler] = []
        self._lock = RLock()
        self._logger = logger or logging.getLogger(__name__)
        self._dropped_normal_events = 0
        self._dropped_critical_events = 0

    @property
    def dropped_events(self) -> int:
        return self._dropped_normal_events

    @property
    def dropped_critical_events(self) -> int:
        return self._dropped_critical_events

    def subscribe_observer(self, event_type: EventType | None, handler: EventHandler) -> Callable[[], None]:
        """Observe one event type or all events without taking dispatch ownership."""

        with self._lock:
            if event_type is None:
                self._wildcard_observers.append(handler)
            else:
                self._observers[event_type].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                target = self._wildcard_observers if event_type is None else self._observers[event_type]
                if handler in target:
                    target.remove(handler)

        return unsubscribe

    def subscribe(self, event_type: EventType | None, handler: EventHandler) -> Callable[[], None]:
        """Backward-compatible alias for observer subscription."""

        return self.subscribe_observer(event_type, handler)

    def publish(self, event: EngineEvent) -> None:
        """Publish an event to the safety-aware queue and observer callbacks."""

        with self._lock:
            if is_critical_event(event):
                if len(self._critical_events) >= self._max_critical_queue_size:
                    dropped = self._drop_oldest_critical_locked()
                    self._dropped_critical_events += 1
                    self._logger.error("Dropped critical event under sustained load: %s", dropped.event_type.value)
                self._critical_events.append(event)
            else:
                if len(self._normal_events) >= self._max_queue_size:
                    dropped = self._normal_events.popleft()
                    self._dropped_normal_events += 1
                    self._logger.warning("Dropped normal event under load: %s", dropped.event_type.value)
                self._normal_events.append(event)
            handlers = tuple(self._observers.get(event.event_type, ())) + tuple(self._wildcard_observers)

        for handler in handlers:
            try:
                handler(event)
            except Exception:
                self._logger.exception("Event handler failed for %s", event.event_type)

    def _drop_oldest_critical_locked(self) -> EngineEvent:
        for index, queued_event in enumerate(self._critical_events):
            if queued_event.event_type != EventType.SHUTDOWN_REQUESTED:
                del self._critical_events[index]
                return queued_event
        return self._critical_events.popleft()

    def publish_type(
        self,
        event_type: EventType,
        source: str,
        payload: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> EngineEvent:
        event = EngineEvent.create(event_type, source, payload, correlation_id)
        self.publish(event)
        return event

    def drain(self, max_events: int | None = None, critical_only: bool = False) -> tuple[EngineEvent, ...]:
        """Remove and return queued events, always prioritizing critical events."""

        with self._lock:
            available = len(self._critical_events) if critical_only else self.queued_count()
            if max_events is None:
                max_events = available
            drained: list[EngineEvent] = []
            while self._critical_events and len(drained) < max_events:
                drained.append(self._critical_events.popleft())
            if not critical_only:
                while self._normal_events and len(drained) < max_events:
                    drained.append(self._normal_events.popleft())
            return tuple(drained)

    def queued_count(self) -> int:
        with self._lock:
            return len(self._critical_events) + len(self._normal_events)

    def critical_count(self) -> int:
        with self._lock:
            return len(self._critical_events)

    def normal_count(self) -> int:
        with self._lock:
            return len(self._normal_events)

    def clear(self) -> None:
        with self._lock:
            self._critical_events.clear()
            self._normal_events.clear()
