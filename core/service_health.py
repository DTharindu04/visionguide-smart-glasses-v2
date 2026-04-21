"""Small retry/backoff helper for optional and safety-adjacent services."""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from typing import Any, Mapping

from core.event_manager import EventManager, EventType


@dataclass(frozen=True)
class ServiceFailure:
    """A recorded service failure and the retry delay it caused."""

    failure_count: int
    retry_after_seconds: float
    message: str


class ServiceHealth:
    """Tracks transient failures without permanently disabling a service."""

    def __init__(
        self,
        service_name: str,
        event_manager: EventManager,
        logger: logging.Logger,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        self._service_name = service_name
        self._events = event_manager
        self._logger = logger
        self._base_backoff_seconds = max(0.1, base_backoff_seconds)
        self._max_backoff_seconds = max(self._base_backoff_seconds, max_backoff_seconds)
        self._failure_count = 0
        self._next_retry_at = 0.0
        self._last_message = ""

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def can_attempt(self) -> bool:
        return time.monotonic() >= self._next_retry_at

    def retry_after_seconds(self) -> float:
        return max(0.0, self._next_retry_at - time.monotonic())

    def record_success(self) -> None:
        if self._failure_count > 0:
            self._logger.info("%s recovered after %s failure(s)", self._service_name, self._failure_count)
        self._failure_count = 0
        self._next_retry_at = 0.0
        self._last_message = ""

    def record_failure(
        self,
        error: BaseException,
        correlation_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> ServiceFailure:
        self._failure_count += 1
        retry_after = min(
            self._max_backoff_seconds,
            self._base_backoff_seconds * (2 ** min(self._failure_count - 1, 5)),
        )
        self._next_retry_at = time.monotonic() + retry_after
        message = str(error) or type(error).__name__
        log_method = self._logger.warning if message != self._last_message else self._logger.debug
        log_method(
            "%s failure #%s; retrying in %.1fs: %s",
            self._service_name,
            self._failure_count,
            retry_after,
            message,
        )
        self._last_message = message
        payload: dict[str, Any] = {
            "error_type": type(error).__name__,
            "message": message,
            "recoverable": True,
            "source": self._service_name,
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            "failure_count": self._failure_count,
            "retry_after_seconds": retry_after,
        }
        if context:
            payload["context"] = dict(context)
        self._events.publish_type(
            EventType.ERROR,
            source=self._service_name,
            payload=payload,
            correlation_id=correlation_id,
        )
        return ServiceFailure(self._failure_count, retry_after, message)
