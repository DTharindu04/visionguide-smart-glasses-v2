"""Offline face detection service."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config.constants import ModuleName
from config.settings import AppSettings
from core.event_manager import EngineEvent, EventManager, EventType
from core.frame_store import FrameStore
from core.model_runtime import LocalModelLoader, ModelRunner, nhwc_to_nchw
from core.service_health import ServiceHealth
from face.face_quality import FaceQualityChecker
from vision.image_utils import ensure_rgb, normalize_box_xyxy, resize_rgb


class FaceDetectionService:
    """Runs local face detection and quality gating."""

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
        self._quality = FaceQualityChecker(settings.thresholds.faces)
        self._yunet: Any | None = None
        self._can_use_yunet_cache: bool | None = None
        self._runner: ModelRunner | None = None
        self._health = ServiceHealth("face_detection_service", event_manager, self._logger)

    def handle_event(self, event: EngineEvent) -> None:
        if not self._health.can_attempt():
            return
        if event.event_type != EventType.MODULE_TRIGGER:
            return
        if event.payload.get("module") != ModuleName.FACE_DETECTION.value:
            return
        try:
            frame_id = int(event.payload["frame_id"])
        except (KeyError, TypeError, ValueError) as exc:
            self._health.record_failure(exc, event.correlation_id, {"event_type": event.event_type.value})
            return
        frame = self._frame_store.get(frame_id)
        if frame is None:
            return
        try:
            faces = self.detect(frame.data)
        except Exception as exc:
            self._health.record_failure(exc, event.correlation_id, {"frame_id": frame_id})
            return
        self._health.record_success()
        self._events.publish_type(
            EventType.FACE_DETECTED,
            source="face_detection_service",
            payload={"frame_id": frame_id, "mode": event.payload.get("mode"), "faces": faces},
            correlation_id=event.correlation_id,
        )

    def detect(self, frame: object) -> list[dict[str, object]]:
        image = ensure_rgb(frame)
        raw_faces = self._detect_yunet(image) if self._can_use_yunet() else self._detect_dnn_fallback(image)
        faces: list[dict[str, object]] = []
        for face in raw_faces:
            quality = self._quality.check(image, tuple(face["box"]))
            face["quality"] = quality.__dict__
            face["quality_passed"] = quality.passed
            faces.append(face)
        return faces

    def _can_use_yunet(self) -> bool:
        if self._can_use_yunet_cache is None:
            try:
                import cv2  # type: ignore[import-not-found]

                self._can_use_yunet_cache = hasattr(cv2, "FaceDetectorYN_create")
            except Exception:
                self._can_use_yunet_cache = False
        return self._can_use_yunet_cache

    def _detect_yunet(self, image: np.ndarray) -> list[dict[str, object]]:
        import cv2  # type: ignore[import-not-found]

        height, width = image.shape[:2]
        if self._yunet is None:
            self._yunet = cv2.FaceDetectorYN_create(
                str(self._settings.models.face_detection),
                "",
                (width, height),
                score_threshold=0.60,
                nms_threshold=0.3,
                top_k=20,
            )
        self._yunet.setInputSize((width, height))
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, faces = self._yunet.detect(bgr)
        if faces is None:
            return []
        parsed: list[dict[str, object]] = []
        for face in faces:
            x, y, w, h = [float(value) for value in face[:4]]
            score = float(face[-1])
            box = normalize_box_xyxy(x, y, x + w, y + h, width, height)
            landmarks = [
                [float(face[index]) / width, float(face[index + 1]) / height]
                for index in range(4, 14, 2)
            ]
            parsed.append({"box": list(box), "confidence": score, "landmarks": landmarks})
        return parsed

    def _detect_dnn_fallback(self, image: np.ndarray) -> list[dict[str, object]]:
        runner = self._get_runner()
        input_width, input_height = self._input_size(runner)
        resized = resize_rgb(image, input_width, input_height).astype(np.float32) / 255.0
        if len(runner.input_info.shape) == 4 and runner.input_info.shape[1] in {1, 3}:
            tensor = nhwc_to_nchw(resized)
        else:
            tensor = resized[None, ...]
        outputs = runner.infer(tensor)
        if not outputs:
            return []
        detections = np.squeeze(outputs[0])
        if detections.ndim == 1:
            detections = detections.reshape(1, -1)
        height, width = image.shape[:2]
        parsed: list[dict[str, object]] = []
        for detection in detections:
            if detection.shape[0] < 5:
                continue
            score = float(detection[-1])
            if score < 0.60:
                continue
            x_min, y_min, x_max, y_max = [float(value) for value in detection[:4]]
            if max(x_min, y_min, x_max, y_max) <= 1.5:
                box = (x_min, y_min, x_max, y_max)
            else:
                box = normalize_box_xyxy(x_min, y_min, x_max, y_max, width, height)
            parsed.append({"box": list(box), "confidence": score, "landmarks": []})
        return parsed

    def _get_runner(self) -> ModelRunner:
        if self._runner is None:
            asset = self._settings.models.asset_by_id("face_detection")
            self._runner = LocalModelLoader(self._logger).load(
                asset.path,
                preferred_backend=asset.backend,
                num_threads=self._settings.optimization.tflite_threads,
                input_shape=asset.input_shape,
            )
        return self._runner

    @staticmethod
    def _input_size(runner: ModelRunner) -> tuple[int, int]:
        shape = runner.input_info.shape
        if len(shape) == 4 and shape[1] in {1, 3}:
            return int(shape[3]), int(shape[2])
        if len(shape) == 4:
            return int(shape[2]), int(shape[1])
        return 320, 320
