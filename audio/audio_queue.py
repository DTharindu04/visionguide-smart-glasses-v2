"""Priority audio queue with stale-message expiry."""

from __future__ import annotations

from collections import deque
from threading import RLock
from typing import Deque

from config.constants import AudioPriority
from audio.types import QueueDecision, SpeechMessage


class AudioQueue:
    """Bounded FIFO-with-priority queue for speech messages."""

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("Audio queue size must be greater than zero.")
        self._max_size = max_size
        self._messages: Deque[SpeechMessage] = deque()
        self._lock = RLock()
        self._dropped_messages = 0

    @property
    def dropped_messages(self) -> int:
        return self._dropped_messages

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)

    def enqueue(self, message: SpeechMessage, decision: QueueDecision) -> bool:
        if not decision.accepted or message.is_expired:
            return False
        with self._lock:
            self._drop_expired_locked()
            if decision.drop_priorities:
                self._drop_priorities_locked(decision.drop_priorities)
            if len(self._messages) >= self._max_size and not self._make_room_locked(message.priority):
                self._dropped_messages += 1
                return False
            self._insert_by_priority_locked(message)
            return True

    def pop_next(self) -> SpeechMessage | None:
        with self._lock:
            self._drop_expired_locked()
            if not self._messages:
                return None
            return self._messages.popleft()

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()

    def contains_duplicate(self, message: SpeechMessage) -> bool:
        with self._lock:
            text = message.normalized_text
            return any(
                existing.priority <= message.priority
                and (existing.dedup_key == message.dedup_key or existing.normalized_text == text)
                for existing in self._messages
            )

    def remove_priorities(self, priorities: tuple[AudioPriority, ...]) -> int:
        with self._lock:
            before = len(self._messages)
            self._drop_priorities_locked(priorities)
            return before - len(self._messages)

    def _insert_by_priority_locked(self, message: SpeechMessage) -> None:
        for index, existing in enumerate(self._messages):
            if message.priority < existing.priority:
                self._messages.insert(index, message)
                return
        self._messages.append(message)

    def _make_room_locked(self, incoming_priority: AudioPriority) -> bool:
        worst_index: int | None = None
        worst_priority = incoming_priority
        for index, existing in enumerate(self._messages):
            if existing.priority >= worst_priority:
                worst_priority = existing.priority
                worst_index = index
        if worst_index is None:
            return False
        del self._messages[worst_index]
        self._dropped_messages += 1
        return True

    def _drop_priorities_locked(self, priorities: tuple[AudioPriority, ...]) -> None:
        blocked = set(priorities)
        kept = [message for message in self._messages if message.priority not in blocked]
        dropped = len(self._messages) - len(kept)
        self._messages.clear()
        self._messages.extend(kept)
        self._dropped_messages += dropped

    def _drop_expired_locked(self) -> None:
        kept = [message for message in self._messages if not message.is_expired]
        dropped = len(self._messages) - len(kept)
        self._messages.clear()
        self._messages.extend(kept)
        self._dropped_messages += dropped
