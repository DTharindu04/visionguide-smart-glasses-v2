"""Face quality checks before recognition and enrollment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.settings import FaceThresholds
from vision.image_utils import crop_normalized, ensure_rgb


@dataclass(frozen=True)
class FaceQualityResult:
    """Quality gate result for one face."""

    passed: bool
    blur_variance: float
    brightness: float
    width_px: int
    height_px: int
    reason: str


class FaceQualityChecker:
    """Applies deterministic quality checks before identity decisions."""

    def __init__(self, thresholds: FaceThresholds) -> None:
        self._thresholds = thresholds

    def check(self, frame: object, box: tuple[float, float, float, float]) -> FaceQualityResult:
        image = ensure_rgb(frame)
        crop = crop_normalized(image, box)
        if crop is None:
            return FaceQualityResult(False, 0.0, 0.0, 0, 0, "empty crop")
        height, width = crop.shape[:2]
        if width < self._thresholds.min_face_width_px or height < self._thresholds.min_face_height_px:
            return FaceQualityResult(False, 0.0, 0.0, width, height, "face too small")

        import cv2  # type: ignore[import-not-found]

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        if blur < self._thresholds.min_laplacian_blur_variance:
            return FaceQualityResult(False, blur, brightness, width, height, "face too blurry")
        if not self._thresholds.min_brightness <= brightness <= self._thresholds.max_brightness:
            return FaceQualityResult(False, blur, brightness, width, height, "face brightness out of range")
        return FaceQualityResult(True, blur, brightness, width, height, "passed")

