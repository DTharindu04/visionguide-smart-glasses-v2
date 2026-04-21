"""Duplicate and near-duplicate speech suppression."""

from __future__ import annotations

import difflib
import time
from collections import deque
from dataclasses import dataclass

from audio.types import SpeechMessage


@dataclass(frozen=True)
class SpokenFingerprint:
    """Recent spoken text/key fingerprint."""

    dedup_key: str
    text: str
    timestamp: float


class DuplicateSuppressor:
    """Suppresses repeated text spam before it reaches TTS."""

    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int = 48,
        similarity_threshold: float = 0.92,
    ) -> None:
        self._ttl_seconds = max(1.0, ttl_seconds)
        self._similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self._entries: deque[SpokenFingerprint] = deque(maxlen=max(1, max_entries))

    def is_duplicate(self, message: SpeechMessage) -> bool:
        self._purge_expired()
        text = message.normalized_text
        for entry in self._entries:
            if entry.dedup_key == message.dedup_key:
                return True
            if entry.text and text:
                ratio = difflib.SequenceMatcher(None, entry.text, text).ratio()
                if ratio >= self._similarity_threshold:
                    return True
        return False

    def remember(self, message: SpeechMessage) -> None:
        self._purge_expired()
        self._entries.append(
            SpokenFingerprint(
                dedup_key=message.dedup_key,
                text=message.normalized_text,
                timestamp=time.monotonic(),
            )
        )

    def reset(self) -> None:
        self._entries.clear()

    def _purge_expired(self) -> None:
        now = time.monotonic()
        while self._entries and now - self._entries[0].timestamp > self._ttl_seconds:
            self._entries.popleft()
