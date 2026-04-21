"""Offline lightweight emotion detection service."""

from __future__ import annotations

import logging

import numpy as np

from config.constants import RuntimeMode
from config.settings import AppSettings
from core.event_manager import EngineEvent, EventManager, EventType
from core.frame_store import FrameStore
from core.model_runtime import LocalModelLoader, ModelRunner, nhwc_to_nchw
from core.service_health import ServiceHealth
from vision.image_utils import crop_normalized, ensure_rgb, resize_rgb


EMOTION_LABELS = ("neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt")


class EmotionDetectionService:
    """Runs emotion inference less frequently than face detection."""

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
        self._runner: ModelRunner | None = None
        self._face_update_count = 0
        self._health = ServiceHealth("emotion_detection_service", event_manager, self._logger)

    def handle_event(self, event: EngineEvent) -> None:
        if not self._health.can_attempt():
            return
        if event.event_type != EventType.FACE_DETECTED:
            return
        try:
            frame_id = int(event.payload["frame_id"])
        except (KeyError, TypeError, ValueError) as exc:
            self._health.record_failure(exc, event.correlation_id, {"event_type": event.event_type.value})
            return
        mode = self._runtime_mode(event.payload.get("mode"))
        profile = self._settings.profile_for(mode)
        interval = profile.emotion_detection_every_n_face_updates
        if interval is None:
            return
        self._face_update_count += 1
        if self._face_update_count % interval != 0:
            return
        frame = self._frame_store.get(frame_id)
        if frame is None:
            return
        faces = event.payload.get("faces", [])
        if not isinstance(faces, list) or not faces:
            return
        face = next((item for item in faces if isinstance(item, dict) and item.get("quality_passed", False)), None)
        if face is None:
            return
        try:
            result = self.detect(frame.data, face)
        except Exception as exc:
            self._health.record_failure(exc, event.correlation_id, {"frame_id": frame_id})
            return
        self._health.record_success()
        self._events.publish_type(
            EventType.EMOTION_DETECTED,
            source="emotion_detection_service",
            payload={"frame_id": frame_id, **result},
            correlation_id=event.correlation_id,
        )

    def detect(self, frame: object, face: dict[str, object]) -> dict[str, object]:
        image = ensure_rgb(frame)
        box = tuple(float(value) for value in face["box"])  # type: ignore[index]
        crop = crop_normalized(image, box, margin=0.10)
        if crop is None:
            return {"emotion": "", "confidence": 0.0}
        crop = resize_rgb(crop, 64, 64)
        gray = np.mean(crop, axis=2, keepdims=True).astype(np.float32) / 255.0
        tensor = gray[None, ...]
        runner = self._get_runner()
        if len(runner.input_info.shape) == 4 and runner.input_info.shape[1] in {1, 3}:
            tensor = nhwc_to_nchw(gray)
        outputs = runner.infer(tensor.astype(np.float32))
        if not outputs:
            return {"emotion": "", "confidence": 0.0}
        scores = np.squeeze(outputs[0]).astype(np.float32)
        if scores.size == 0:
            return {"emotion": "", "confidence": 0.0}
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / max(float(np.sum(exp_scores)), 1e-12)
        index = int(np.argmax(probs))
        confidence = float(probs[index])
        if confidence < self._settings.thresholds.faces.emotion_min_confidence:
            return {"emotion": "", "confidence": confidence}
        label = EMOTION_LABELS[index] if index < len(EMOTION_LABELS) else str(index)
        return {"emotion": label, "confidence": confidence}

    def _get_runner(self) -> ModelRunner:
        if self._runner is None:
            asset = self._settings.models.asset_by_id("emotion_detection")
            self._runner = LocalModelLoader(self._logger).load(
                asset.path,
                preferred_backend=asset.backend,
                num_threads=self._settings.optimization.tflite_threads,
                input_shape=asset.input_shape,
            )
        return self._runner

    @staticmethod
    def _runtime_mode(value: object) -> RuntimeMode:
        if isinstance(value, RuntimeMode):
            return value
        if isinstance(value, str):
            try:
                return RuntimeMode(value)
            except ValueError:
                return RuntimeMode.FACE
        return RuntimeMode.FACE
