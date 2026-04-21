"""Offline face recognition service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from config.constants import ModuleName, RuntimeMode
from config.settings import AppSettings
from core.event_manager import EngineEvent, EventManager, EventType
from core.frame_store import FrameStore
from core.model_runtime import LocalModelLoader, ModelRunner, nhwc_to_nchw
from core.service_health import ServiceHealth
from face.face_quality import FaceQualityChecker
from face.face_store import FaceEmbeddingStore
from vision.image_utils import crop_normalized, ensure_rgb, resize_rgb


STABLE_FACE_IOU_THRESHOLD = 0.45
MAX_STABLE_FRAME_GAP = 4


@dataclass(frozen=True)
class StableFaceTrack:
    """Minimal track used to gate identity recognition."""

    box: tuple[float, float, float, float]
    count: int
    last_frame_id: int


class FaceRecognitionService:
    """Extracts face embeddings and matches them against local storage."""

    def __init__(
        self,
        settings: AppSettings,
        frame_store: FrameStore,
        event_manager: EventManager,
        store: FaceEmbeddingStore,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._frame_store = frame_store
        self._events = event_manager
        self._store = store
        self._logger = logger or logging.getLogger(__name__)
        self._quality = FaceQualityChecker(settings.thresholds.faces)
        self._sface: Any | None = None
        self._can_use_sface_cache: bool | None = None
        self._runner: ModelRunner | None = None
        self._health = ServiceHealth("face_recognition_service", event_manager, self._logger)
        self._stable_face: StableFaceTrack | None = None

    def handle_event(self, event: EngineEvent) -> None:
        if not self._health.can_attempt():
            return
        if event.event_type != EventType.FACE_DETECTED:
            return
        mode = self._runtime_mode(event.payload.get("mode"))
        profile = self._settings.profile_for(mode)
        if ModuleName.FACE_RECOGNITION not in profile.active_modules:
            return
        interval = profile.face_recognition_every_n_stable_faces
        if interval is None:
            return
        try:
            frame_id = int(event.payload["frame_id"])
        except (KeyError, TypeError, ValueError) as exc:
            self._health.record_failure(exc, event.correlation_id, {"event_type": event.event_type.value})
            return
        frame = self._frame_store.get(frame_id)
        if frame is None:
            return
        faces = event.payload.get("faces", [])
        if not isinstance(faces, list):
            return
        for face in faces:
            if not isinstance(face, dict) or not face.get("quality_passed", False):
                continue
            if not self._stable_face_ready(face, frame_id, interval):
                continue
            if not self._has_required_alignment(face):
                continue
            try:
                result = self.recognize(frame.data, face)
            except ValueError as exc:
                self._logger.debug("Face recognition sample rejected: %s", exc)
                continue
            except Exception as exc:
                self._health.record_failure(exc, event.correlation_id, {"frame_id": frame_id})
                return
            self._health.record_success()
            self._events.publish_type(
                EventType.FACE_RECOGNIZED,
                source="face_recognition_service",
                payload={"frame_id": frame_id, **result},
                correlation_id=event.correlation_id,
            )
            break

    def recognize(self, frame: object, face: dict[str, object]) -> dict[str, object]:
        embedding = self.extract_embedding(frame, face)
        match = self._store.find_best(embedding)
        if match is None:
            return {"name": "", "person_id": "", "confidence": 0.0, "known": False}
        threshold = self._settings.thresholds.faces.recognition_similarity_threshold
        known = match.similarity >= threshold
        return {
            "name": match.name if known else "",
            "person_id": match.person_id if known else "",
            "confidence": match.similarity,
            "known": known,
            "sample_count": match.sample_count,
        }

    def extract_embedding(self, frame: object, face: dict[str, object]) -> np.ndarray:
        image = ensure_rgb(frame)
        box = tuple(float(value) for value in face["box"])  # type: ignore[index]
        quality = self._quality.check(image, box)
        if not quality.passed:
            raise ValueError(f"Face quality check failed: {quality.reason}")
        if self._can_use_sface():
            return self._extract_sface(image, face)
        crop = crop_normalized(image, box, margin=0.15)
        if crop is None:
            raise ValueError("Cannot extract empty face crop.")
        crop = resize_rgb(crop, 112, 112).astype(np.float32) / 255.0
        outputs = self._get_runner().infer(nhwc_to_nchw(crop))
        if not outputs:
            raise ValueError("Face recognition model returned no outputs.")
        return self._normalize(outputs[0])

    def _can_use_sface(self) -> bool:
        if self._can_use_sface_cache is None:
            try:
                import cv2  # type: ignore[import-not-found]

                self._can_use_sface_cache = hasattr(cv2, "FaceRecognizerSF_create")
            except Exception:
                self._can_use_sface_cache = False
        return self._can_use_sface_cache

    def _extract_sface(self, image: np.ndarray, face: dict[str, object]) -> np.ndarray:
        import cv2  # type: ignore[import-not-found]

        height, width = image.shape[:2]
        if self._sface is None:
            self._sface = cv2.FaceRecognizerSF_create(str(self._settings.models.face_recognition), "")
        face_row = self._face_row_for_sface(face, width, height)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        aligned = self._sface.alignCrop(bgr, face_row)
        feature = self._sface.feature(aligned)
        return self._normalize(feature)

    def _face_row_for_sface(self, face: dict[str, object], width: int, height: int) -> np.ndarray:
        x_min, y_min, x_max, y_max = [float(value) for value in face["box"]]  # type: ignore[index]
        row = [
            x_min * width,
            y_min * height,
            (x_max - x_min) * width,
            (y_max - y_min) * height,
        ]
        landmarks = face.get("landmarks", [])
        if not self._valid_landmarks(landmarks):
            raise ValueError("SFace recognition requires five face landmarks.")
        for point in landmarks[:5]:  # type: ignore[index]
            row.extend([float(point[0]) * width, float(point[1]) * height])
        row.append(float(face.get("confidence", 0.0)))
        return np.asarray(row, dtype=np.float32)

    def _get_runner(self) -> ModelRunner:
        if self._runner is None:
            asset = self._settings.models.asset_by_id("face_recognition")
            self._runner = LocalModelLoader(self._logger).load(
                asset.path,
                preferred_backend=asset.backend,
                num_threads=self._settings.optimization.tflite_threads,
                input_shape=asset.input_shape,
            )
        return self._runner

    def _stable_face_ready(self, face: dict[str, object], frame_id: int, interval: int) -> bool:
        box = self._face_box(face)
        if box is None:
            return False
        previous = self._stable_face
        if (
            previous is not None
            and frame_id - previous.last_frame_id <= MAX_STABLE_FRAME_GAP
            and self._iou(box, previous.box) >= STABLE_FACE_IOU_THRESHOLD
        ):
            count = previous.count + 1
        else:
            count = 1
        self._stable_face = StableFaceTrack(box=box, count=count, last_frame_id=frame_id)
        required = self._settings.thresholds.faces.stable_frames_required
        if count < required:
            return False
        return (count - required) % max(1, interval) == 0

    def _has_required_alignment(self, face: dict[str, object]) -> bool:
        if not self._can_use_sface():
            return True
        if self._valid_landmarks(face.get("landmarks", [])):
            return True
        self._logger.debug("Skipping face recognition because detector did not provide landmarks")
        return False

    @staticmethod
    def _valid_landmarks(value: object) -> bool:
        if not isinstance(value, list) or len(value) < 5:
            return False
        for point in value[:5]:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                return False
            try:
                float(point[0])
                float(point[1])
            except (TypeError, ValueError):
                return False
        return True

    @staticmethod
    def _face_box(face: dict[str, object]) -> tuple[float, float, float, float] | None:
        value = face.get("box")
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            x_min, y_min, x_max, y_max = [max(0.0, min(1.0, float(item))) for item in value]
        except (TypeError, ValueError):
            return None
        left, right = sorted((x_min, x_max))
        top, bottom = sorted((y_min, y_max))
        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)

    @staticmethod
    def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
        left_x_min, left_y_min, left_x_max, left_y_max = left
        right_x_min, right_y_min, right_x_max, right_y_max = right
        x_min = max(left_x_min, right_x_min)
        y_min = max(left_y_min, right_y_min)
        x_max = min(left_x_max, right_x_max)
        y_max = min(left_y_max, right_y_max)
        intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
        left_area = max(0.0, left_x_max - left_x_min) * max(0.0, left_y_max - left_y_min)
        right_area = max(0.0, right_x_max - right_x_min) * max(0.0, right_y_max - right_y_min)
        union = left_area + right_area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    @staticmethod
    def _runtime_mode(value: object) -> RuntimeMode:
        if isinstance(value, RuntimeMode):
            return value
        if isinstance(value, str):
            try:
                return RuntimeMode(value)
            except ValueError:
                return RuntimeMode.NORMAL
        return RuntimeMode.NORMAL

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        vector = embedding.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return vector
        return vector / norm
