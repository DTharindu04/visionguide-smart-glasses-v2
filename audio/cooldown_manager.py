"""Speech cooldown enforcement to prevent repeated announcements."""

from __future__ import annotations

import time

from config.constants import AudioPriority, SpeechCategory
from config.settings import CooldownSettings
from audio.types import SpeechMessage


class CooldownManager:
    """Tracks category and message-level cooldown windows."""

    def __init__(self, cooldowns: CooldownSettings) -> None:
        self._cooldowns = cooldowns
        self._last_by_category: dict[SpeechCategory, float] = {}
        self._last_by_key: dict[str, float] = {}

    def can_speak(self, message: SpeechMessage, cooldown_multiplier: float = 1.0) -> bool:
        """Return whether a message is outside its cooldown windows."""

        now = time.monotonic()
        multiplier = max(0.1, cooldown_multiplier)
        category_interval = self._cooldowns.for_category(message.category) * multiplier
        if message.priority == AudioPriority.P1_DANGER:
            key_interval = category_interval
        else:
            key_interval = self._cooldowns.same_message_dedup_seconds * multiplier

        last_category = self._last_by_category.get(message.category)
        if last_category is not None and now - last_category < category_interval:
            return False

        last_key = self._last_by_key.get(message.dedup_key)
        if last_key is not None and now - last_key < key_interval:
            return False

        return True

    def mark_spoken(self, message: SpeechMessage) -> None:
        now = time.monotonic()
        self._last_by_category[message.category] = now
        self._last_by_key[message.dedup_key] = now

    def reset(self) -> None:
        self._last_by_category.clear()
        self._last_by_key.clear()
