"""Shared helpers for production scripts."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from camera.camera_manager import CameraManager
from config.settings import AppSettings, ensure_runtime_directories, get_settings


def configure_script_logging(verbose: bool = False) -> None:
    """Keep operational logs off stdout so CLI JSON remains machine-readable."""

    level = logging.INFO if verbose else logging.ERROR
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    handler.setLevel(level)
    root.addHandler(handler)
    root.setLevel(level)


def load_settings() -> AppSettings:
    settings = get_settings()
    ensure_runtime_directories(settings)
    return settings


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def print_json(payload: Any) -> None:
    print(json.dumps(payload, default=json_default, indent=2, sort_keys=True))


def load_rgb_images(directory: Path, limit: int | None = None) -> list[np.ndarray]:
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images: list[np.ndarray] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in supported:
            continue
        with Image.open(path) as image:
            images.append(np.asarray(image.convert("RGB")))
        if limit is not None and len(images) >= limit:
            break
    if not images:
        raise ValueError(f"No supported images found in {directory}")
    return images


def capture_frames(
    settings: AppSettings,
    count: int,
    timeout_seconds: float,
    dry_run: bool = False,
    warmup_frames: int | None = None,
    pace: bool = True,
) -> list[np.ndarray]:
    if count <= 0:
        raise ValueError("Frame count must be greater than zero.")
    frames: list[np.ndarray] = []
    warmup = settings.camera.warmup_frames if warmup_frames is None else max(0, warmup_frames)
    frame_interval = 1.0 / max(1, settings.camera.target_fps)
    camera = CameraManager(
        settings.camera,
        allow_synthetic=dry_run,
        prefer_synthetic=dry_run,
    )
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    camera.start()
    try:
        for _ in range(warmup):
            if time.monotonic() >= deadline:
                break
            camera.read()
        while len(frames) < count and time.monotonic() < deadline:
            started = time.monotonic()
            frames.append(np.asarray(camera.read().data))
            if pace:
                remaining = frame_interval - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        camera.stop()
    if len(frames) < count:
        raise TimeoutError(f"Captured {len(frames)} of {count} requested frame(s).")
    return frames


def percentile(values: Iterable[float], q: float) -> float | None:
    data = [float(value) for value in values]
    if not data:
        return None
    return float(np.percentile(np.asarray(data, dtype=np.float32), q))


def exit_code(has_errors: bool) -> int:
    return 1 if has_errors else 0
