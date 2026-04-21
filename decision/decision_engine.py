"""Decision engine that converts inference events into user-facing intents."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping

from config.constants import AudioPriority, RuntimeMode, SpeechCategory
from config.settings import AppSettings
from core.event_manager import EngineEvent, EventManager, EventType
from core.exceptions import DecisionEngineError
from vision.obstacle_analyzer import DetectionBox, ObjectDetectionResult, ObstacleAnalyzer, ObstacleWarning


@dataclass(frozen=True)
class SpeechIntent:
    """A decision produced for the audio manager."""

    text: str
    priority: AudioPriority
    category: SpeechCategory
    interrupt: bool
    dedup_key: str
    ttl_seconds: float
    metadata: Mapping[str, Any]


class DecisionEngine:
    """Turns raw inference results into conservative guidance intents."""

    def __init__(
        self,
        settings: AppSettings,
        event_manager: EventManager,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._events = event_manager
        self._logger = logger or logging.getLogger(__name__)
        self._obstacles = ObstacleAnalyzer(settings.thresholds.objects)

    def handle_event(self, event: EngineEvent) -> None:
        if event.event_type == EventType.OBJECT_DETECTIONS:
            self._handle_object_detections(event)
        elif event.event_type == EventType.FACE_RECOGNIZED:
            self._handle_face_recognized(event)
        elif event.event_type == EventType.OCR_RESULT:
            self._handle_ocr_result(event)
        elif event.event_type == EventType.EMOTION_DETECTED:
            self._handle_emotion_detected(event)

    def _handle_object_detections(self, event: EngineEvent) -> None:
        detections = self._parse_object_detections(event.payload)
        if not detections:
            return

        ranked = sorted(detections, key=lambda item: item.severity, reverse=True)
        detection = ranked[0]
        severity = detection.severity

        objects = self._settings.thresholds.objects
        if severity >= objects.severity_danger_threshold:
            intent = SpeechIntent(
                text=self._danger_text(detection),
                priority=AudioPriority.P1_DANGER,
                category=SpeechCategory.DANGER,
                interrupt=True,
                dedup_key=f"danger:{detection.label}:{detection.zone}",
                ttl_seconds=self._settings.cooldowns.ttl_for_priority(AudioPriority.P1_DANGER),
                metadata={
                    "severity": severity,
                    "label": detection.label,
                    "zone": detection.zone,
                    "confidence": detection.confidence,
                },
            )
            self._publish_audio_intent(intent, event.correlation_id)
            self._events.publish_type(
                EventType.STATE_TRANSITION_REQUEST,
                source="decision_engine",
                payload={"target_mode": RuntimeMode.DANGER.value, "reason": f"{detection.label} danger"},
                correlation_id=event.correlation_id,
            )
        elif severity >= objects.severity_warning_threshold:
            intent = SpeechIntent(
                text=self._guidance_text(detection),
                priority=AudioPriority.P3_NAVIGATION,
                category=SpeechCategory.OBJECT_GUIDANCE,
                interrupt=False,
                dedup_key=f"guidance:{detection.label}:{detection.zone}",
                ttl_seconds=self._settings.cooldowns.ttl_for_priority(AudioPriority.P3_NAVIGATION),
                metadata={
                    "severity": severity,
                    "label": detection.label,
                    "zone": detection.zone,
                    "confidence": detection.confidence,
                },
            )
            self._publish_audio_intent(intent, event.correlation_id)

    def _parse_object_detections(self, payload: Mapping[str, Any]) -> tuple[ObstacleWarning, ...]:
        raw_detections = payload.get("detections", ())
        if not isinstance(raw_detections, (list, tuple)):
            raise DecisionEngineError("object_detections payload must contain a detections list.")

        parsed: list[ObstacleWarning] = []
        allowed = set(self._settings.thresholds.objects.navigation_classes)
        for item in raw_detections:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label", "")).strip()
            confidence = self._safe_float(item.get("confidence", 0.0))
            if confidence is None:
                self._logger.debug("Skipping detection with invalid confidence: %r", item)
                continue
            if label not in allowed or confidence < self._settings.thresholds.objects.min_detection_confidence:
                continue
            box_value = item.get("box")
            if not isinstance(box_value, (list, tuple)) or len(box_value) != 4:
                continue
            box = self._parse_box(box_value)
            if box is None:
                self._logger.debug("Skipping detection with invalid box: %r", item)
                continue
            parsed.append(self._warning_from_payload(item, label, confidence, box))
            if len(parsed) >= self._settings.thresholds.objects.max_objects_per_frame:
                break
        return tuple(parsed)

    def _parse_box(self, box_value: tuple[Any, ...] | list[Any]) -> DetectionBox | None:
        values: list[float] = []
        for value in box_value:
            parsed = self._safe_float(value)
            if parsed is None:
                return None
            values.append(max(0.0, min(1.0, parsed)))
        x_min, y_min, x_max, y_max = values
        left, right = sorted((x_min, x_max))
        top, bottom = sorted((y_min, y_max))
        return DetectionBox(left, top, right, bottom)

    def _warning_from_payload(
        self,
        item: Mapping[str, Any],
        label: str,
        confidence: float,
        box: DetectionBox,
    ) -> ObstacleWarning:
        severity = self._safe_float(item.get("severity"))
        zone = str(item.get("zone", "")).strip()
        if severity is None or zone not in {"left", "center", "right"}:
            return self._obstacles.score(ObjectDetectionResult(label=label, confidence=confidence, box=box))
        severity = max(0.0, min(1.0, severity))
        thresholds = self._settings.thresholds.objects
        area_ratio = self._safe_float(item.get("area_ratio"))
        return ObstacleWarning(
            label=label,
            confidence=confidence,
            box=box,
            zone=zone,
            severity=severity,
            area_ratio=box.area_ratio if area_ratio is None else max(0.0, min(1.0, area_ratio)),
            is_danger=severity >= thresholds.severity_danger_threshold,
            is_critical=severity >= thresholds.severity_critical_threshold,
        )

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    def _danger_text(self, detection: ObstacleWarning) -> str:
        zone = detection.zone
        if zone == "center":
            return f"Stop. {detection.label} ahead."
        return f"Careful. {detection.label} on your {zone}."

    def _guidance_text(self, detection: ObstacleWarning) -> str:
        zone = detection.zone
        if zone == "center":
            return f"{detection.label} ahead."
        return f"{detection.label} on your {zone}."

    def _handle_face_recognized(self, event: EngineEvent) -> None:
        name = str(event.payload.get("name", "")).strip()
        confidence = self._safe_float(event.payload.get("confidence", 0.0)) or 0.0
        faces = self._settings.thresholds.faces
        if name and confidence >= faces.recognition_min_confidence:
            text = f"{name} is nearby."
            category = SpeechCategory.KNOWN_FACE
            dedup_key = f"face:{name}"
        else:
            text = "Unknown person nearby."
            category = SpeechCategory.UNKNOWN_FACE
            dedup_key = "face:unknown"

        intent = SpeechIntent(
            text=text,
            priority=AudioPriority.P2_IDENTITY_OCR,
            category=category,
            interrupt=False,
            dedup_key=dedup_key,
            ttl_seconds=self._settings.cooldowns.ttl_for_priority(AudioPriority.P2_IDENTITY_OCR),
            metadata={"confidence": confidence, "name": name},
        )
        self._publish_audio_intent(intent, event.correlation_id)

    def _handle_ocr_result(self, event: EngineEvent) -> None:
        text = " ".join(str(event.payload.get("text", "")).split())
        confidence = self._safe_float(event.payload.get("confidence", 0.0)) or 0.0
        ocr = self._settings.thresholds.ocr
        if len(text) < ocr.min_text_characters or confidence < ocr.min_text_confidence:
            self._logger.info("OCR result suppressed due to low confidence or short text")
            return
        intent = SpeechIntent(
            text=text,
            priority=AudioPriority.P2_IDENTITY_OCR,
            category=SpeechCategory.OCR_TEXT,
            interrupt=False,
            dedup_key=f"ocr:{text[:80].lower()}",
            ttl_seconds=self._settings.cooldowns.ttl_for_priority(AudioPriority.P2_IDENTITY_OCR),
            metadata={"confidence": confidence},
        )
        self._publish_audio_intent(intent, event.correlation_id)

    def _handle_emotion_detected(self, event: EngineEvent) -> None:
        emotion = " ".join(str(event.payload.get("emotion", "")).split())
        confidence = self._safe_float(event.payload.get("confidence", 0.0)) or 0.0
        if not emotion or confidence < self._settings.thresholds.faces.emotion_min_confidence:
            return
        intent = SpeechIntent(
            text=f"Emotion appears to be {emotion}.",
            priority=AudioPriority.P4_CONTEXT,
            category=SpeechCategory.EMOTION,
            interrupt=False,
            dedup_key=f"emotion:{emotion}",
            ttl_seconds=self._settings.cooldowns.ttl_for_priority(AudioPriority.P4_CONTEXT),
            metadata={"emotion": emotion, "confidence": confidence},
        )
        self._publish_audio_intent(intent, event.correlation_id)

    def _publish_audio_intent(self, intent: SpeechIntent, correlation_id: str) -> None:
        self._events.publish_type(
            EventType.AUDIO_INTENT,
            source="decision_engine",
            payload={
                "text": intent.text,
                "priority": intent.priority.name,
                "priority_value": int(intent.priority),
                "category": intent.category.value,
                "interrupt": intent.interrupt,
                "dedup_key": intent.dedup_key,
                "ttl_seconds": intent.ttl_seconds,
                "metadata": dict(intent.metadata),
            },
            correlation_id=correlation_id,
        )
