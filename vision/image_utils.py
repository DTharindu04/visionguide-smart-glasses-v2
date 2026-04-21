"""Image preprocessing helpers shared by lightweight inference services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LetterboxResult:
    """Result of resizing an image while preserving aspect ratio."""

    image: np.ndarray
    scale: float
    pad_x: int
    pad_y: int
    original_width: int
    original_height: int


def ensure_rgb(frame: Any) -> np.ndarray:
    """Return an RGB uint8 image from a numpy-like frame."""

    image = np.asarray(frame)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError(f"Expected HxWx3/4 frame, got shape {image.shape}")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2  # type: ignore[import-not-found]

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def letterbox(image: np.ndarray, width: int, height: int, fill_value: int = 114) -> LetterboxResult:
    """Resize with unchanged aspect ratio and padding."""

    import cv2  # type: ignore[import-not-found]

    original_height, original_width = image.shape[:2]
    scale = min(width / original_width, height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = np.full((height, width, 3), fill_value, dtype=np.uint8)
    pad_x = (width - resized_width) // 2
    pad_y = (height - resized_height) // 2
    padded[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return LetterboxResult(
        image=padded,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        original_width=original_width,
        original_height=original_height,
    )


def normalize_box_xyxy(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    left, right = sorted((x_min, x_max))
    top, bottom = sorted((y_min, y_max))
    return (
        max(0.0, min(1.0, left / max(1, width))),
        max(0.0, min(1.0, top / max(1, height))),
        max(0.0, min(1.0, right / max(1, width))),
        max(0.0, min(1.0, bottom / max(1, height))),
    )


def crop_normalized(image: np.ndarray, box: tuple[float, float, float, float], margin: float = 0.0) -> np.ndarray | None:
    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = box
    box_width = x_max - x_min
    box_height = y_max - y_min
    x_min -= box_width * margin
    x_max += box_width * margin
    y_min -= box_height * margin
    y_max += box_height * margin
    left = int(max(0, min(width - 1, round(x_min * width))))
    right = int(max(0, min(width, round(x_max * width))))
    top = int(max(0, min(height - 1, round(y_min * height))))
    bottom = int(max(0, min(height, round(y_max * height))))
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right].copy()


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vec = left.astype(np.float32).reshape(-1)
    right_vec = right.astype(np.float32).reshape(-1)
    denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left_vec, right_vec) / denom)

