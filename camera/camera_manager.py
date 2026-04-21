"""Camera acquisition manager for Raspberry Pi Camera Module 2."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from config.settings import CameraSettings
from core.exceptions import CameraError

CameraBackend = Literal["picamera2", "opencv", "synthetic"]


@dataclass(frozen=True)
class CapturedFrame:
    """A captured camera frame and its metadata."""

    frame_id: int
    timestamp: float
    width: int
    height: int
    data: Any
    backend: CameraBackend

    @property
    def age_ms(self) -> float:
        return (time.monotonic() - self.timestamp) * 1000.0


class CameraManager:
    """Owns camera startup, capture, and shutdown."""

    def __init__(
        self,
        settings: CameraSettings,
        logger: logging.Logger | None = None,
        allow_synthetic: bool = False,
        prefer_synthetic: bool = False,
    ) -> None:
        self._settings = settings
        self._logger = logger or logging.getLogger(__name__)
        self._allow_synthetic = allow_synthetic
        self._prefer_synthetic = prefer_synthetic
        self._backend: CameraBackend | None = None
        self._camera: Any | None = None
        self._frame_id = 0
        self._started = False

    @property
    def backend(self) -> CameraBackend | None:
        return self._backend

    @property
    def is_started(self) -> bool:
        return self._started

    def start(self) -> None:
        """Start the best available local camera backend."""

        if self._started:
            return
        if self._allow_synthetic and self._prefer_synthetic:
            self._start_synthetic()
            return

        errors: list[str] = []
        for starter in (self._start_picamera2, self._start_opencv):
            try:
                starter()
                self._started = True
                self._logger.info("Camera started with backend=%s", self._backend)
                return
            except CameraError as exc:
                errors.append(str(exc))
                self._logger.debug("Camera backend unavailable: %s", exc)

        if self._allow_synthetic:
            self._start_synthetic()
            return

        raise CameraError("No camera backend could be started: " + "; ".join(errors))

    def _start_picamera2(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore[import-not-found]
        except Exception as exc:
            raise CameraError(f"picamera2 unavailable: {exc}") from exc

        camera: Any | None = None
        try:
            camera = Picamera2()
            config = camera.create_video_configuration(
                main={
                    "size": (self._settings.capture_width, self._settings.capture_height),
                    "format": "RGB888",
                },
                controls={"FrameRate": float(self._settings.target_fps)},
            )
            camera.configure(config)
            camera.start()
            for _ in range(self._settings.warmup_frames):
                camera.capture_array()
        except Exception as exc:
            if camera is not None:
                try:
                    camera.close()
                except Exception:
                    pass
            raise CameraError(f"picamera2 startup failed: {exc}") from exc
        self._camera = camera
        self._backend = "picamera2"

    def _start_opencv(self) -> None:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception as exc:
            raise CameraError(f"OpenCV camera backend unavailable: {exc}") from exc

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera.release()
            raise CameraError("OpenCV could not open camera index 0.")
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.capture_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.capture_height)
        camera.set(cv2.CAP_PROP_FPS, self._settings.target_fps)
        self._camera = camera
        self._backend = "opencv"

    def _start_synthetic(self) -> None:
        self._backend = "synthetic"
        self._camera = None
        self._started = True
        self._logger.warning("Camera started in synthetic dry-run mode")

    def read(self) -> CapturedFrame:
        """Read one frame from the active backend."""

        if not self._started or self._backend is None:
            raise CameraError("Camera must be started before reading frames.")

        if self._backend == "picamera2":
            data = self._camera.capture_array()
            timestamp = time.monotonic()
            height, width = data.shape[:2]
        elif self._backend == "opencv":
            ok, data = self._camera.read()
            timestamp = time.monotonic()
            if not ok:
                raise CameraError("OpenCV failed to read a frame.")
            try:
                import cv2  # type: ignore[import-not-found]

                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            except Exception as exc:
                raise CameraError(f"OpenCV failed to convert frame to RGB: {exc}") from exc
            height, width = data.shape[:2]
        else:
            data = self._synthetic_frame()
            timestamp = time.monotonic()
            height, width = data.shape[:2]

        self._frame_id += 1
        if self._settings.hflip or self._settings.vflip:
            data = self._flip_frame(data)
        return CapturedFrame(
            frame_id=self._frame_id,
            timestamp=timestamp,
            width=int(width),
            height=int(height),
            data=data,
            backend=self._backend,
        )

    def _synthetic_frame(self) -> Any:
        try:
            import numpy as np
        except Exception as exc:
            raise CameraError(f"numpy is required for synthetic camera mode: {exc}") from exc
        frame = np.zeros((self._settings.capture_height, self._settings.capture_width, 3), dtype=np.uint8)
        marker = self._frame_id % 255
        frame[:, :, 1] = marker
        return frame

    def _flip_frame(self, data: Any) -> Any:
        try:
            import numpy as np
        except Exception as exc:
            raise CameraError(f"numpy is required for frame flipping: {exc}") from exc
        if self._settings.hflip:
            data = np.flip(data, axis=1)
        if self._settings.vflip:
            data = np.flip(data, axis=0)
        return data.copy()

    def stop(self) -> None:
        """Stop and release camera resources."""

        if not self._started:
            return
        try:
            if self._backend == "picamera2" and self._camera is not None:
                self._camera.stop()
                self._camera.close()
            elif self._backend == "opencv" and self._camera is not None:
                self._camera.release()
        finally:
            self._camera = None
            self._backend = None
            self._started = False
            self._logger.info("Camera stopped")

    def __enter__(self) -> "CameraManager":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
