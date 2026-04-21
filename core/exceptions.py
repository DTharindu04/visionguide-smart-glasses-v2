"""Exception hierarchy and error conversion for the smart glasses runtime."""

from __future__ import annotations

import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


class SmartGlassesError(Exception):
    """Base exception for expected project-specific runtime errors."""

    recoverable = False


class ConfigurationError(SmartGlassesError):
    """Raised when configuration prevents safe startup."""


class CameraError(SmartGlassesError):
    """Raised for camera startup or frame acquisition failures."""

    recoverable = True


class EventManagerError(SmartGlassesError):
    """Raised for invalid event publishing or dispatch behavior."""

    recoverable = True


class StateTransitionError(SmartGlassesError):
    """Raised when a state transition request is invalid."""

    recoverable = True


class DecisionEngineError(SmartGlassesError):
    """Raised when decision processing receives invalid input."""

    recoverable = True


class SchedulerError(SmartGlassesError):
    """Raised when the runtime scheduler cannot continue safely."""


class PerformanceMonitorError(SmartGlassesError):
    """Raised when performance monitoring fails unexpectedly."""

    recoverable = True


class RecoverableRuntimeError(SmartGlassesError):
    """Generic recoverable runtime error."""

    recoverable = True


class FatalRuntimeError(SmartGlassesError):
    """Generic fatal runtime error."""


@dataclass(frozen=True)
class ErrorReport:
    """Structured error information suitable for logs and events."""

    error_type: str
    message: str
    recoverable: bool
    source: str
    traceback: str


def is_recoverable(error: BaseException) -> bool:
    """Return whether runtime code can continue after this exception."""

    if isinstance(error, SmartGlassesError):
        return bool(error.recoverable)
    return isinstance(error, (TimeoutError, InterruptedError))


def build_error_report(error: BaseException, source: str) -> ErrorReport:
    """Convert an exception into a structured report."""

    return ErrorReport(
        error_type=type(error).__name__,
        message=str(error),
        recoverable=is_recoverable(error),
        source=source,
        traceback="".join(traceback.format_exception(type(error), error, error.__traceback__)),
    )


@contextmanager
def exception_boundary(source: str) -> Iterator[None]:
    """Wrap runtime sections and normalize unexpected exceptions."""

    try:
        yield
    except SmartGlassesError:
        raise
    except Exception as exc:
        raise FatalRuntimeError(f"{source} failed unexpectedly: {exc}") from exc

