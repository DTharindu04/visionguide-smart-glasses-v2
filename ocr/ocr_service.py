"""Offline OCR service using local Tesseract data."""

from __future__ import annotations

import difflib
import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from config.constants import ModuleName
from config.settings import AppSettings
from core.event_manager import EngineEvent, EventManager, EventType
from core.frame_store import FrameStore
from core.service_health import ServiceHealth
from vision.image_utils import ensure_rgb


@dataclass(frozen=True)
class CachedOcrText:
    """Recent OCR output for repeat suppression."""

    text: str
    timestamp: float


class OcrService:
    """Preprocesses frames and runs offline OCR on demand."""

    def __init__(
        self,
        settings: AppSettings,
        frame_store: FrameStore,
        event_manager: EventManager,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._frame_store = frame_store
        self._events = event_manager
        self._logger = logger or logging.getLogger(__name__)
        self._last_trigger_at = 0.0
        self._cache: deque[CachedOcrText] = deque(maxlen=settings.thresholds.ocr.max_cache_entries)
        self._health = ServiceHealth("ocr_service", event_manager, self._logger, max_backoff_seconds=60.0)

    def handle_event(self, event: EngineEvent) -> None:
        if not self._health.can_attempt():
            return
        frame_id = self._frame_id_from_event(event)
        if frame_id == "ignore":
            return
        if frame_id == "invalid":
            self._health.record_failure(ValueError("OCR event has invalid frame_id"), event.correlation_id)
            return
        if not self._cooldown_ready():
            return
        frame = self._frame_store.get(frame_id) if isinstance(frame_id, int) else self._frame_store.latest()
        if frame is None:
            return
        self._mark_triggered()
        try:
            result = self.recognize(frame.data)
        except Exception as exc:
            self._health.record_failure(exc, event.correlation_id, {"frame_id": frame.frame_id})
            return
        self._health.record_success()
        if result is None:
            return
        self._events.publish_type(
            EventType.OCR_RESULT,
            source="ocr_service",
            payload={"frame_id": frame.frame_id, **result},
            correlation_id=event.correlation_id,
        )

    def recognize(self, frame: object) -> dict[str, object] | None:
        image = ensure_rgb(frame)
        processed = self._preprocess(image)
        text, confidence = self._run_tesseract(processed)
        text = " ".join(text.split())
        thresholds = self._settings.thresholds.ocr
        if len(text) < thresholds.min_text_characters or confidence < thresholds.min_text_confidence:
            return None
        if self._is_recent_repeat(text):
            self._logger.debug("Suppressing repeated OCR text")
            return None
        self._cache.append(CachedOcrText(text=text, timestamp=time.monotonic()))
        return {"text": text, "confidence": confidence}

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        import cv2  # type: ignore[import-not-found]

        thresholds = self._settings.thresholds.ocr
        height, width = image.shape[:2]
        scale = thresholds.preprocessing_target_width_px / max(1, width)
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image,
            (thresholds.preprocessing_target_width_px, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        block_size = thresholds.adaptive_threshold_block_size
        if block_size % 2 == 0:
            block_size += 1
        return cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            thresholds.adaptive_threshold_c,
        )

    def _run_tesseract(self, image: np.ndarray) -> tuple[str, float]:
        try:
            import pytesseract  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError("pytesseract is not installed.") from exc

        if not self._settings.models.ocr_eng.exists():
            raise RuntimeError(f"Missing required English Tesseract data: {self._settings.models.ocr_eng}")
        languages = ["eng"]
        if self._settings.models.ocr_sin.exists():
            languages.append("sin")
        config = f'--tessdata-dir "{self._settings.models.ocr_tessdata_dir}" --oem 1 --psm 6'
        data = pytesseract.image_to_data(
            image,
            lang="+".join(languages),
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        words: list[str] = []
        confidences: list[float] = []
        for text, confidence in zip(data.get("text", []), data.get("conf", []), strict=False):
            clean = str(text).strip()
            try:
                value = float(confidence)
            except (TypeError, ValueError):
                value = -1.0
            if clean and value >= 0:
                words.append(clean)
                confidences.append(value)
        if not words:
            return "", 0.0
        return " ".join(words), float(sum(confidences) / max(1, len(confidences)))

    def _frame_id_from_event(self, event: EngineEvent) -> int | str | None:
        if event.event_type == EventType.OCR_REQUEST:
            if "frame_id" not in event.payload:
                return None
            try:
                return int(event.payload["frame_id"])
            except (TypeError, ValueError):
                return "invalid"
        if event.event_type == EventType.MODULE_TRIGGER and event.payload.get("module") == ModuleName.OCR.value:
            try:
                return int(event.payload["frame_id"])
            except (KeyError, TypeError, ValueError):
                return "invalid"
        return "ignore"

    def _cooldown_ready(self) -> bool:
        now = time.monotonic()
        if now - self._last_trigger_at < self._settings.thresholds.ocr.min_trigger_interval_seconds:
            return False
        return True

    def _mark_triggered(self) -> None:
        self._last_trigger_at = time.monotonic()

    def _is_recent_repeat(self, text: str) -> bool:
        now = time.monotonic()
        ttl = self._settings.thresholds.ocr.cache_ttl_seconds
        while self._cache and now - self._cache[0].timestamp > ttl:
            self._cache.popleft()
        normalized = text.casefold()
        for cached in self._cache:
            ratio = difflib.SequenceMatcher(None, normalized, cached.text.casefold()).ratio()
            if ratio >= self._settings.thresholds.ocr.repeat_similarity_threshold:
                return True
        return False
