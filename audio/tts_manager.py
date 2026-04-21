"""Offline TTS manager and audio event handler."""

from __future__ import annotations

import logging
import json
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from importlib.util import find_spec
from typing import Any, Mapping

from config.constants import AudioPriority, RuntimeMode, SpeechCategory
from config.settings import AppSettings
from core.event_manager import EngineEvent, EventManager, EventType
from audio.audio_queue import AudioQueue
from audio.cooldown_manager import CooldownManager
from audio.deduplication import DuplicateSuppressor
from audio.priority_engine import PriorityInterruptEngine
from audio.speech_formatter import SpeechFormatter
from audio.types import SpeechMessage


class OfflineTtsBackend(ABC):
    """Interruptible offline TTS backend contract."""

    @abstractmethod
    def speak(self, message: SpeechMessage, stop_event: threading.Event) -> bool:
        """Speak a message. Returns True when fully spoken."""

    @abstractmethod
    def interrupt(self) -> None:
        """Stop the current utterance as quickly as the backend allows."""

    def close(self) -> None:
        """Release backend resources."""


class EspeakNgBackend(OfflineTtsBackend):
    """Subprocess-backed espeak-ng TTS with hard interruption support."""

    def __init__(self, voice: str, rate_wpm: int, volume: float, logger: logging.Logger) -> None:
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if executable is None:
            raise RuntimeError("espeak-ng/espeak executable is not installed.")
        self._executable = executable
        self._voice = voice
        self._rate_wpm = rate_wpm
        self._volume = max(0, min(200, int(round(volume * 100))))
        self._logger = logger
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.RLock()

    def speak(self, message: SpeechMessage, stop_event: threading.Event) -> bool:
        command = [
            self._executable,
            "-v",
            self._voice_for(message.language),
            "-s",
            str(self._rate_wpm),
            "-a",
            str(self._volume),
            "--stdin",
        ]
        with self._lock:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            process = self._process
        try:
            if process.stdin is not None:
                process.stdin.write(message.text.encode("utf-8"))
                process.stdin.close()
            while process.poll() is None:
                if stop_event.is_set():
                    self.interrupt()
                    return False
                time.sleep(0.02)
            return process.returncode == 0
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None

    def interrupt(self) -> None:
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=0.5)

    def _voice_for(self, language: str) -> str:
        if language.startswith("si"):
            return "si"
        return self._voice


class Pyttsx3Backend(OfflineTtsBackend):
    """Subprocess pyttsx3 fallback with hard interruption support."""

    def __init__(self, voice: str, rate_wpm: int, volume: float, logger: logging.Logger) -> None:
        if find_spec("pyttsx3") is None:
            raise RuntimeError("pyttsx3 is not installed.")
        self._voice = voice
        self._rate_wpm = rate_wpm
        self._volume = max(0.0, min(1.0, volume))
        self._logger = logger
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.RLock()

    def speak(self, message: SpeechMessage, stop_event: threading.Event) -> bool:
        payload = {
            "text": message.text,
            "voice": self._voice_for(message.language),
            "rate_wpm": self._rate_wpm,
            "volume": self._volume,
        }
        command = [sys.executable, "-m", "audio.pyttsx3_worker"]
        with self._lock:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            process = self._process
        try:
            if process.stdin is not None:
                process.stdin.write(json.dumps(payload).encode("utf-8"))
                process.stdin.close()
            while process.poll() is None:
                if stop_event.is_set():
                    self.interrupt()
                    return False
                time.sleep(0.02)
            return process.returncode == 0
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None

    def interrupt(self) -> None:
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=0.5)

    def close(self) -> None:
        self.interrupt()

    def _voice_for(self, language: str) -> str:
        if language.startswith("si") and self._voice.startswith("si"):
            return self._voice
        return self._voice


class LoggingTtsBackend(OfflineTtsBackend):
    """Dry-run backend that preserves queue semantics without audio output."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def speak(self, message: SpeechMessage, stop_event: threading.Event) -> bool:
        self._logger.info("Dry-run TTS: %s", message.text)
        deadline = time.monotonic() + min(0.15, max(0.02, len(message.text) * 0.002))
        while time.monotonic() < deadline:
            if stop_event.is_set():
                return False
            time.sleep(0.01)
        return True

    def interrupt(self) -> None:
        return


class OfflineTtsManager:
    """Consumes AUDIO_INTENT events and speaks them with priority rules."""

    def __init__(
        self,
        settings: AppSettings,
        event_manager: EventManager,
        logger: logging.Logger | None = None,
        backend: OfflineTtsBackend | None = None,
    ) -> None:
        self._settings = settings
        self._events = event_manager
        self._logger = logger or logging.getLogger(__name__)
        self._formatter = SpeechFormatter(settings)
        self._queue = AudioQueue(settings.audio.queue_size)
        self._priority = PriorityInterruptEngine(settings.audio.p1_interrupts_all, settings.audio.p2_replaces_p3_p4)
        self._cooldowns = CooldownManager(settings.cooldowns)
        self._dedup = DuplicateSuppressor(settings.cooldowns.same_message_dedup_seconds)
        self._backend = backend
        self._condition = threading.Condition()
        self._stop_requested = threading.Event()
        self._interrupt_current = threading.Event()
        self._worker: threading.Thread | None = None
        self._current_message: SpeechMessage | None = None
        self._current_lock = threading.RLock()
        self._mode = RuntimeMode.NORMAL
        self._started = False

    @property
    def current_message(self) -> SpeechMessage | None:
        with self._current_lock:
            return self._current_message

    @property
    def queued_count(self) -> int:
        return len(self._queue)

    def start(self) -> None:
        if self._started:
            return
        if self._backend is None:
            self._backend = self._build_backend()
        self._started = True
        self._stop_requested.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="offline-tts", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop_requested.set()
        self._interrupt_current.set()
        if self._backend is not None:
            self._backend.interrupt()
        with self._condition:
            self._condition.notify_all()
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=2.0)
        self._started = False
        if self._backend is not None:
            self._backend.close()

    def handle_event(self, event: EngineEvent) -> None:
        if event.event_type == EventType.STATE_CHANGED:
            self._handle_state_changed(event.payload)
            return
        if event.event_type == EventType.SHUTDOWN_REQUESTED:
            self.stop()
            return
        if event.event_type != EventType.AUDIO_INTENT:
            return
        self.enqueue_intent(event.payload, event.correlation_id)

    def enqueue_intent(self, payload: Mapping[str, Any], correlation_id: str) -> bool:
        message = self._formatter.from_intent(payload, correlation_id)
        if message is None:
            self._logger.debug("Dropped malformed audio intent: %r", payload)
            self._publish_drop(None, "malformed_intent", correlation_id)
            return False
        if not self._mode_allows(message):
            self._logger.debug("Dropped audio intent blocked by mode %s: %s", self._mode.value, message.dedup_key)
            self._publish_drop(message, "mode_blocked", correlation_id)
            return False
        profile = self._settings.profile_for(self._mode)
        if self._is_current_duplicate(message) or self._queue.contains_duplicate(message):
            self._publish_drop(message, "already_current_or_queued", correlation_id)
            return False
        if message.priority != AudioPriority.P1_DANGER and self._dedup.is_duplicate(message):
            self._publish_drop(message, "duplicate_recently_spoken", correlation_id)
            return False
        if not self._cooldowns.can_speak(message, profile.cooldown_multiplier):
            self._publish_drop(message, "cooldown_active", correlation_id)
            return False

        decision = self._priority.decide(
            message,
            self.current_message,
            allow_interruptions=profile.allow_audio_interruptions,
        )
        if decision.interrupt_current:
            self._interrupt_current.set()
            if self._backend is not None:
                self._backend.interrupt()
        accepted = self._queue.enqueue(message, decision)
        if accepted:
            with self._condition:
                self._condition.notify()
        else:
            self._publish_drop(message, "queue_rejected", correlation_id)
        return accepted

    def _worker_loop(self) -> None:
        while not self._stop_requested.is_set():
            backend = self._backend
            if backend is None:
                self._logger.error("Audio worker started without a TTS backend")
                return
            with self._condition:
                while len(self._queue) == 0 and not self._stop_requested.is_set():
                    self._condition.wait(timeout=0.25)
            if self._stop_requested.is_set():
                break
            message = self._queue.pop_next()
            if message is None or message.is_expired:
                if message is not None:
                    self._publish_drop(message, "expired_before_speech", message.correlation_id)
                continue
            profile = self._settings.profile_for(self._mode)
            if (
                message.priority != AudioPriority.P1_DANGER
                and self._dedup.is_duplicate(message)
            ):
                self._publish_drop(message, "duplicate_recently_spoken", message.correlation_id)
                continue
            if not self._cooldowns.can_speak(message, profile.cooldown_multiplier):
                self._publish_drop(message, "cooldown_active", message.correlation_id)
                continue
            with self._current_lock:
                self._current_message = message
            self._interrupt_current.clear()
            completed = False
            backend_failed = False
            self._publish_audio_event(EventType.AUDIO_STARTED, message)
            try:
                completed = backend.speak(message, self._interrupt_current)
            except Exception as exc:
                backend_failed = True
                self._logger.warning("TTS backend failed: %s", exc)
            finally:
                with self._current_lock:
                    self._current_message = None
            if completed:
                self._cooldowns.mark_spoken(message)
                self._dedup.remember(message)
                self._publish_audio_event(EventType.AUDIO_SPOKEN, message)
            elif backend_failed:
                self._publish_drop(message, "tts_backend_failed", message.correlation_id)
            elif self._interrupt_current.is_set():
                self._publish_audio_event(EventType.AUDIO_INTERRUPTED, message)

    def _handle_state_changed(self, payload: Mapping[str, Any]) -> None:
        raw_mode = payload.get("mode")
        if isinstance(raw_mode, str):
            try:
                self._mode = RuntimeMode(raw_mode)
            except ValueError:
                self._logger.debug("Ignoring unknown mode for audio manager: %s", raw_mode)

    def _mode_allows(self, message: SpeechMessage) -> bool:
        if message.priority == AudioPriority.P1_DANGER:
            return True
        if self._mode == RuntimeMode.QUIET and not self._settings.audio.allows_in_quiet(message.priority):
            return False
        return self._settings.profile_for(self._mode).allows_audio_priority(message.priority)

    def _is_current_duplicate(self, message: SpeechMessage) -> bool:
        current = self.current_message
        if current is None:
            return False
        if message.priority < current.priority:
            return False
        return current.dedup_key == message.dedup_key or current.normalized_text == message.normalized_text

    def _publish_drop(self, message: SpeechMessage | None, reason: str, correlation_id: str) -> None:
        payload: dict[str, Any] = {"reason": reason, "mode": self._mode.value}
        if message is not None:
            payload.update(self._message_payload(message))
        self._events.publish_type(
            EventType.AUDIO_DROPPED,
            source="audio_manager",
            payload=payload,
            correlation_id=correlation_id,
        )

    def _publish_audio_event(self, event_type: EventType, message: SpeechMessage) -> None:
        self._events.publish_type(
            event_type,
            source="audio_manager",
            payload=self._message_payload(message),
            correlation_id=message.correlation_id,
        )

    @staticmethod
    def _message_payload(message: SpeechMessage) -> dict[str, Any]:
        return {
            "priority": message.priority.name,
            "priority_value": int(message.priority),
            "category": message.category.value,
            "dedup_key": message.dedup_key,
            "language": message.language,
            "text": message.text,
        }

    def _build_backend(self) -> OfflineTtsBackend:
        backend_name = self._settings.audio.backend.strip().casefold()
        if backend_name in {"espeak-ng", "espeak"}:
            try:
                return EspeakNgBackend(
                    self._settings.audio.voice,
                    self._settings.audio.speech_rate_wpm,
                    self._settings.audio.volume,
                    self._logger,
                )
            except RuntimeError as exc:
                self._logger.warning("espeak-ng unavailable, falling back to pyttsx3: %s", exc)
        return Pyttsx3Backend(
            self._settings.audio.voice,
            self._settings.audio.speech_rate_wpm,
            self._settings.audio.volume,
            self._logger,
        )
