"""Offline object detection service with YOLO/SSD-style output parsing."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config.constants import ModuleName, OBJECT_CLASS_ALIASES
from config.settings import AppSettings
from core.event_manager import EngineEvent, EventManager, EventType
from core.exceptions import ConfigurationError
from core.frame_store import FrameStore
from core.model_runtime import LocalModelLoader, ModelRunner, nhwc_to_nchw
from core.service_health import ServiceHealth
from vision.image_utils import LetterboxResult, ensure_rgb, letterbox
from vision.obstacle_analyzer import DetectionBox, ObjectDetectionResult, ObstacleAnalyzer


class ObjectDetectionService:
    """Runs lightweight local object detection on scheduled frames."""

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
        self._labels = self._load_labels(settings.models.object_labels)
        self._allowed_labels = set(settings.thresholds.objects.navigation_classes)
        self._analyzer = ObstacleAnalyzer(settings.thresholds.objects)
        self._health = ServiceHealth("object_detection_service", event_manager, self._logger)

    def handle_event(self, event: EngineEvent) -> None:
        if not self._health.can_attempt():
            return
        if event.event_type != EventType.MODULE_TRIGGER:
            return
        if event.payload.get("module") != ModuleName.OBJECT_DETECTION.value:
            return
        try:
            frame_id = int(event.payload["frame_id"])
        except (KeyError, TypeError, ValueError) as exc:
            self._health.record_failure(exc, event.correlation_id, {"event_type": event.event_type.value})
            return
        frame = self._frame_store.get(frame_id)
        if frame is None:
            self._logger.debug("Frame no longer available for object detection: %s", frame_id)
            return
        try:
            detections = self.detect(frame.data)
        except Exception as exc:
            self._health.record_failure(exc, event.correlation_id, {"frame_id": frame_id})
            return
        self._health.record_success()
        warnings = self._analyzer.analyze(detections)
        self._events.publish_type(
            EventType.OBJECT_DETECTIONS,
            source="object_detection_service",
            payload={
                "frame_id": frame_id,
                "detections": [warning.to_payload() for warning in warnings],
                "max_severity": max((warning.severity for warning in warnings), default=0.0),
            },
            correlation_id=event.correlation_id,
        )

    def detect(self, frame: object) -> tuple[ObjectDetectionResult, ...]:
        runner = self._get_runner()
        image = ensure_rgb(frame)
        input_width, input_height = self._input_size(runner)
        prepared = letterbox(image, input_width, input_height)
        input_tensor = self._prepare_tensor(prepared.image, runner)
        outputs = runner.infer(input_tensor)
        detections = self._parse_outputs(outputs, prepared)
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return tuple(detections[: self._settings.thresholds.objects.max_objects_per_frame])

    def _get_runner(self) -> ModelRunner:
        if self._runner is None:
            asset = self._settings.models.asset_by_id("object_detection")
            self._runner = LocalModelLoader(self._logger).load(
                asset.path,
                preferred_backend=asset.backend,
                num_threads=self._settings.optimization.tflite_threads,
                input_shape=asset.input_shape,
            )
        return self._runner

    def _input_size(self, runner: ModelRunner) -> tuple[int, int]:
        shape = runner.input_info.shape
        if len(shape) == 4:
            if shape[1] in {1, 3}:
                return int(shape[3]), int(shape[2])
            return int(shape[2]), int(shape[1])
        return self._settings.camera.inference_width, self._settings.camera.inference_height

    def _prepare_tensor(self, image: np.ndarray, runner: ModelRunner) -> np.ndarray:
        tensor = image.astype(np.float32) / 255.0
        shape = runner.input_info.shape
        if len(shape) == 4 and shape[1] in {1, 3}:
            tensor = nhwc_to_nchw(tensor)
        else:
            tensor = tensor[None, ...]
        return tensor.astype(np.float32)

    def _parse_outputs(
        self,
        outputs: list[np.ndarray],
        prepared: LetterboxResult,
    ) -> list[ObjectDetectionResult]:
        if not outputs:
            raise ConfigurationError("Object detector returned no outputs.")
        primary = np.asarray(outputs[0])
        if self._looks_like_yolo(primary):
            return self._parse_yolo(primary, prepared)
        if len(outputs) >= 3:
            return self._parse_ssd(outputs, prepared)
        raise ConfigurationError(f"Unsupported object detector output shapes: {[output.shape for output in outputs]}")

    @staticmethod
    def _looks_like_yolo(output: np.ndarray) -> bool:
        squeezed = np.squeeze(output)
        return (squeezed.ndim == 1 and squeezed.shape[0] >= 6) or (
            squeezed.ndim == 2 and min(squeezed.shape) >= 6
        )

    def _parse_yolo(self, output: np.ndarray, prepared: LetterboxResult) -> list[ObjectDetectionResult]:
        predictions = np.squeeze(output)
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
        if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] >= 6:
            predictions = predictions.T
        detections: list[ObjectDetectionResult] = []
        for prediction in predictions:
            if prediction.shape[0] < 6:
                continue
            x_center, y_center, width, height = [float(value) for value in prediction[:4]]
            if len(self._labels) > 0 and prediction.shape[0] == len(self._labels) + 5:
                objectness = float(prediction[4])
                class_scores = prediction[5:]
            else:
                objectness = 1.0
                class_scores = prediction[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id]) * objectness
            label = self._label_for(class_id)
            if not self._should_keep(label, confidence):
                continue
            x_min = x_center - width / 2.0
            y_min = y_center - height / 2.0
            x_max = x_center + width / 2.0
            y_max = y_center + height / 2.0
            box = self._to_original_normalized_box(x_min, y_min, x_max, y_max, prepared)
            detections.append(ObjectDetectionResult(label=label, confidence=confidence, box=box))
        return self._nms(detections)

    def _parse_ssd(self, outputs: list[np.ndarray], prepared: LetterboxResult) -> list[ObjectDetectionResult]:
        boxes = np.squeeze(outputs[0])
        classes = np.squeeze(outputs[1]).astype(np.int32)
        scores = np.squeeze(outputs[2])
        detections: list[ObjectDetectionResult] = []
        for index, score in enumerate(scores.reshape(-1)):
            confidence = float(score)
            class_id = int(classes.reshape(-1)[index])
            label = self._label_for(class_id)
            if not self._should_keep(label, confidence):
                continue
            raw_box = boxes.reshape(-1, 4)[index]
            y_min, x_min, y_max, x_max = [float(value) for value in raw_box]
            box = DetectionBox(x_min, y_min, x_max, y_max)
            detections.append(ObjectDetectionResult(label=label, confidence=confidence, box=box))
        return self._nms(detections)

    def _to_original_normalized_box(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        prepared: LetterboxResult,
    ) -> DetectionBox:
        left = (x_min - prepared.pad_x) / prepared.scale
        top = (y_min - prepared.pad_y) / prepared.scale
        right = (x_max - prepared.pad_x) / prepared.scale
        bottom = (y_max - prepared.pad_y) / prepared.scale
        return DetectionBox(
            max(0.0, min(1.0, left / prepared.original_width)),
            max(0.0, min(1.0, top / prepared.original_height)),
            max(0.0, min(1.0, right / prepared.original_width)),
            max(0.0, min(1.0, bottom / prepared.original_height)),
        )

    def _should_keep(self, label: str, confidence: float) -> bool:
        return label in self._allowed_labels and confidence >= self._settings.thresholds.objects.min_detection_confidence

    def _label_for(self, class_id: int) -> str:
        if 0 <= class_id < len(self._labels):
            label = self._labels[class_id]
            return OBJECT_CLASS_ALIASES.get(label, label)
        return str(class_id)

    @staticmethod
    def _load_labels(path: Path) -> tuple[str, ...]:
        if not path.exists():
            return ()
        return tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    def _nms(self, detections: list[ObjectDetectionResult], iou_threshold: float = 0.45) -> list[ObjectDetectionResult]:
        kept: list[ObjectDetectionResult] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if all(self._iou(detection.box, existing.box) < iou_threshold for existing in kept):
                kept.append(detection)
        return kept

    @staticmethod
    def _iou(left: DetectionBox, right: DetectionBox) -> float:
        x_min = max(left.x_min, right.x_min)
        y_min = max(left.y_min, right.y_min)
        x_max = min(left.x_max, right.x_max)
        y_max = min(left.y_max, right.y_max)
        intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
        union = left.area_ratio + right.area_ratio - intersection
        if union <= 0:
            return 0.0
        return intersection / union
