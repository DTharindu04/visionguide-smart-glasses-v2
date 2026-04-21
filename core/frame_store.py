"""Bounded in-memory frame store for event-driven inference services."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any

from camera.camera_manager import CapturedFrame


@dataclass(frozen=True)
class StoredFrame:
    """Frame data retained for downstream inference services."""

    frame_id: int
    timestamp: float
    width: int
    height: int
    data: Any
    backend: str

    @property
    def age_ms(self) -> float:
        return (time.monotonic() - self.timestamp) * 1000.0


class FrameStore:
    """Small FIFO frame store keyed by frame ID."""

    def __init__(self, max_frames: int = 3) -> None:
        if max_frames <= 0:
            raise ValueError("FrameStore max_frames must be greater than zero.")
        self._max_frames = max_frames
        self._frames: OrderedDict[int, StoredFrame] = OrderedDict()
        self._lock = RLock()

    def put(self, frame: CapturedFrame) -> StoredFrame:
        stored = StoredFrame(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            width=frame.width,
            height=frame.height,
            data=frame.data,
            backend=frame.backend,
        )
        with self._lock:
            self._frames[stored.frame_id] = stored
            self._frames.move_to_end(stored.frame_id)
            while len(self._frames) > self._max_frames:
                self._frames.popitem(last=False)
        return stored

    def get(self, frame_id: int) -> StoredFrame | None:
        with self._lock:
            frame = self._frames.get(frame_id)
            if frame is not None:
                self._frames.move_to_end(frame_id)
            return frame

    def latest(self) -> StoredFrame | None:
        with self._lock:
            if not self._frames:
                return None
            return next(reversed(self._frames.values()))

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

