"""Shared audio message types."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from config.constants import AudioPriority, SpeechCategory


@dataclass(frozen=True)
class SpeechMessage:
    """A normalized speech item ready for queueing and TTS."""

    text: str
    priority: AudioPriority
    category: SpeechCategory
    dedup_key: str
    language: str
    created_at: float = field(default_factory=time.monotonic)
    ttl_seconds: float = 5.0
    interrupt: bool = False
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def expires_at(self) -> float:
        return self.created_at + max(0.1, self.ttl_seconds)

    @property
    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at

    @property
    def normalized_text(self) -> str:
        return " ".join(self.text.split()).casefold()


@dataclass(frozen=True)
class QueueDecision:
    """Result of applying priority rules to an incoming message."""

    accepted: bool
    interrupt_current: bool = False
    drop_priorities: tuple[AudioPriority, ...] = ()
    reason: str = ""
