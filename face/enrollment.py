"""Face enrollment workflow for local offline identity storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from config.settings import AppSettings
from face.face_detection_service import FaceDetectionService
from face.face_recognition_service import FaceRecognitionService
from face.face_store import FaceEmbeddingStore, PersonRecord


@dataclass(frozen=True)
class EnrollmentResult:
    """Outcome of a face enrollment request."""

    person: PersonRecord
    accepted_samples: int
    rejected_samples: int
    rejection_reasons: dict[str, int] = field(default_factory=dict)


class FaceEnrollmentManager:
    """Collects high-quality local face embeddings for one person."""

    def __init__(
        self,
        settings: AppSettings,
        detector: FaceDetectionService,
        recognizer: FaceRecognitionService,
        store: FaceEmbeddingStore,
    ) -> None:
        self._settings = settings
        self._detector = detector
        self._recognizer = recognizer
        self._store = store

    def enroll_from_frames(self, name: str, frames: Iterable[object]) -> EnrollmentResult:
        embeddings: list[np.ndarray] = []
        rejection_reasons: dict[str, int] = {}
        max_samples = self._settings.thresholds.faces.enrollment_max_samples
        required = self._settings.thresholds.faces.enrollment_samples_required
        clean_name = " ".join(name.split())
        for frame in frames:
            if len(embeddings) >= max_samples:
                break
            faces = [face for face in self._detector.detect(frame) if face.get("quality_passed", False)]
            if not faces:
                self._record_rejection(rejection_reasons, "no_quality_face")
                continue
            if len(faces) > 1:
                self._record_rejection(rejection_reasons, "multiple_quality_faces")
                continue
            try:
                embeddings.append(self._recognizer.extract_embedding(frame, faces[0]))
            except Exception as exc:
                self._record_rejection(rejection_reasons, f"embedding_failed:{type(exc).__name__}")
        if len(embeddings) < required:
            raise ValueError(f"Enrollment requires {required} accepted samples; got {len(embeddings)}.")
        self._reject_duplicate_identity(clean_name, tuple(embeddings))
        person = self._store.add_or_update_person(clean_name, tuple(embeddings))
        return EnrollmentResult(
            person=person,
            accepted_samples=len(embeddings),
            rejected_samples=sum(rejection_reasons.values()),
            rejection_reasons=rejection_reasons,
        )

    def _reject_duplicate_identity(self, name: str, embeddings: tuple[np.ndarray, ...]) -> None:
        threshold = self._settings.thresholds.faces.recognition_similarity_threshold
        for embedding in embeddings:
            match = self._store.find_best(embedding)
            if match is None:
                continue
            if match.similarity >= threshold and match.name.casefold() != name.casefold():
                raise ValueError(
                    "Enrollment appears to match existing person "
                    f"{match.name!r} ({match.similarity:.3f}); remove/update that record first."
                )

    @staticmethod
    def _record_rejection(rejection_reasons: dict[str, int], reason: str) -> None:
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
