"""Local face embedding storage for offline recognition."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock

import numpy as np

from vision.image_utils import cosine_similarity


@dataclass(frozen=True)
class PersonRecord:
    """Persisted person metadata."""

    person_id: str
    name: str
    created_at: float
    updated_at: float
    sample_count: int


@dataclass(frozen=True)
class RecognitionMatch:
    """Best identity match for an embedding."""

    person_id: str
    name: str
    similarity: float
    sample_count: int


class FaceEmbeddingStore:
    """Stores people metadata and embeddings under storage/faces."""

    def __init__(self, root_dir: Path, max_embeddings_per_person: int = 32) -> None:
        self._root_dir = root_dir
        self._people_path = root_dir / "people.json"
        self._embeddings_path = root_dir / "embeddings.npz"
        self._max_embeddings_per_person = max(1, max_embeddings_per_person)
        self._lock = RLock()
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._people: dict[str, PersonRecord] = self._load_people()
        self._embeddings: dict[str, np.ndarray] = self._load_embeddings()

    def list_people(self) -> tuple[PersonRecord, ...]:
        with self._lock:
            return tuple(sorted(self._people.values(), key=lambda item: item.name.lower()))

    def add_or_update_person(self, name: str, embeddings: tuple[np.ndarray, ...]) -> PersonRecord:
        clean_name = " ".join(name.split())
        if not clean_name:
            raise ValueError("Person name must not be empty.")
        if not embeddings:
            raise ValueError("At least one embedding is required.")
        with self._lock:
            person_id = self._find_person_id_by_name(clean_name) or uuid.uuid4().hex
            now = time.time()
            existing = self._people.get(person_id)
            old_embeddings = self._embeddings.get(person_id)
            new_embeddings = np.stack([self._normalize(embedding) for embedding in embeddings]).astype(np.float32)
            if old_embeddings is not None:
                new_embeddings = np.concatenate([old_embeddings, new_embeddings], axis=0)
            if new_embeddings.shape[0] > self._max_embeddings_per_person:
                new_embeddings = new_embeddings[-self._max_embeddings_per_person :]
            record = PersonRecord(
                person_id=person_id,
                name=clean_name,
                created_at=existing.created_at if existing else now,
                updated_at=now,
                sample_count=int(new_embeddings.shape[0]),
            )
            self._people[person_id] = record
            self._embeddings[person_id] = new_embeddings
            self._persist()
            return record

    def remove_person(self, person_id: str) -> bool:
        with self._lock:
            removed = self._people.pop(person_id, None) is not None
            self._embeddings.pop(person_id, None)
            if removed:
                self._persist()
            return removed

    def find_best(self, embedding: np.ndarray) -> RecognitionMatch | None:
        with self._lock:
            query = self._normalize(embedding)
            best: RecognitionMatch | None = None
            for person_id, embeddings in self._embeddings.items():
                record = self._people.get(person_id)
                if record is None or embeddings.size == 0:
                    continue
                similarity = max(cosine_similarity(query, stored) for stored in embeddings)
                if best is None or similarity > best.similarity:
                    best = RecognitionMatch(person_id, record.name, similarity, record.sample_count)
            return best

    def _find_person_id_by_name(self, name: str) -> str | None:
        normalized = name.casefold()
        for person_id, record in self._people.items():
            if record.name.casefold() == normalized:
                return person_id
        return None

    def _load_people(self) -> dict[str, PersonRecord]:
        if not self._people_path.exists():
            return {}
        try:
            data = json.loads(self._people_path.read_text(encoding="utf-8"))
            return {
                item["person_id"]: PersonRecord(
                    person_id=item["person_id"],
                    name=item["name"],
                    created_at=float(item["created_at"]),
                    updated_at=float(item["updated_at"]),
                    sample_count=int(item["sample_count"]),
                )
                for item in data.get("people", [])
            }
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            self._quarantine_corrupt_file(self._people_path)
            return {}

    def _load_embeddings(self) -> dict[str, np.ndarray]:
        if not self._embeddings_path.exists():
            return {}
        try:
            with np.load(self._embeddings_path) as loaded:
                return {key: loaded[key].astype(np.float32) for key in loaded.files}
        except (OSError, ValueError):
            self._quarantine_corrupt_file(self._embeddings_path)
            return {}

    def _persist(self) -> None:
        people_payload = {"people": [record.__dict__ for record in self._people.values()]}
        self._atomic_write_text(self._people_path, json.dumps(people_payload, indent=2, sort_keys=True))
        with NamedTemporaryFile("wb", dir=self._root_dir, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            np.savez_compressed(tmp, **self._embeddings)
            tmp.flush()
            os.fsync(tmp.fileno())
        tmp_path.replace(self._embeddings_path)

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

    @staticmethod
    def _quarantine_corrupt_file(path: Path) -> None:
        if not path.exists():
            return
        timestamp = int(time.time())
        quarantine_path = path.with_name(f"{path.name}.corrupt-{timestamp}")
        try:
            path.replace(quarantine_path)
        except OSError:
            pass

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        vector = embedding.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return vector
        return vector / norm
