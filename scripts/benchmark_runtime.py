"""Benchmark local inference and capture paths on the target device."""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:
    from ._common import capture_frames, configure_script_logging, load_rgb_images, load_settings, percentile, print_json
except ImportError:
    from _common import capture_frames, configure_script_logging, load_rgb_images, load_settings, percentile, print_json
from core.event_manager import EventManager
from core.frame_store import FrameStore
from vision.object_detection_service import ObjectDetectionService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark runtime components")
    parser.add_argument("--component", choices=("object", "camera"), default="object")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--dry-run-camera", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def summarize(times_ms: list[float]) -> dict[str, float | int]:
    if not times_ms:
        return {"count": 0}
    return {
        "count": len(times_ms),
        "min_ms": round(min(times_ms), 3),
        "median_ms": round(statistics.median(times_ms), 3),
        "mean_ms": round(statistics.mean(times_ms), 3),
        "p95_ms": round(percentile(times_ms, 95) or 0.0, 3),
        "p99_ms": round(percentile(times_ms, 99) or 0.0, 3),
        "max_ms": round(max(times_ms), 3),
    }


def time_call(callback: Callable[[], object], iterations: int) -> tuple[list[float], str | None]:
    times: list[float] = []
    for _ in range(max(1, iterations)):
        started = time.perf_counter()
        try:
            callback()
        except Exception as exc:
            return times, f"{type(exc).__name__}: {exc}"
        times.append((time.perf_counter() - started) * 1000.0)
    return times, None


def read_temperature_c() -> float | None:
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return int(path.read_text(encoding="utf-8").strip()) / 1000.0
    except Exception:
        return None


def sample_system() -> dict[str, float | None]:
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:
        return {"cpu_percent": None, "memory_percent": None, "temperature_c": read_temperature_c()}
    return {
        "cpu_percent": float(psutil.cpu_percent(interval=None)),
        "memory_percent": float(psutil.virtual_memory().percent),
        "temperature_c": read_temperature_c(),
    }


def benchmark_object(iterations: int, warmup: int, image_path: Path | None) -> dict[str, object]:
    settings = load_settings()
    if image_path is not None:
        frame = load_rgb_images(image_path, limit=1)[0] if image_path.is_dir() else load_rgb_images_from_file(image_path)
    else:
        frame = np.zeros((settings.camera.capture_height, settings.camera.capture_width, 3), dtype=np.uint8)
    model_logger = logging.getLogger("scripts.benchmark.object_detection")
    model_logger.addHandler(logging.NullHandler())
    model_logger.propagate = False
    service = ObjectDetectionService(settings, FrameStore(), EventManager(), logger=model_logger)
    started = time.perf_counter()
    try:
        service.detect(frame)
    except Exception as exc:
        return {
            "component": "object_detection",
            "load_and_first_inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "summary": summarize([]),
            "system_before": sample_system(),
            "system_after": sample_system(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    load_and_first_inference_ms = (time.perf_counter() - started) * 1000.0
    for _ in range(max(0, warmup)):
        service.detect(frame)
    before = sample_system()
    times, error = time_call(lambda: service.detect(frame), iterations)
    after = sample_system()
    return {
        "component": "object_detection",
        "load_and_first_inference_ms": round(load_and_first_inference_ms, 3),
        "warmup_iterations": max(0, warmup),
        "summary": summarize(times),
        "system_before": before,
        "system_after": after,
        "error": error,
    }


def load_rgb_images_from_file(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def benchmark_camera(iterations: int, dry_run: bool) -> dict[str, object]:
    settings = load_settings()
    started = time.perf_counter()
    before = sample_system()
    frames = capture_frames(settings, iterations, timeout_seconds=max(5.0, iterations), dry_run=dry_run)
    after = sample_system()
    elapsed = time.perf_counter() - started
    return {
        "component": "camera",
        "frames": len(frames),
        "elapsed_seconds": round(elapsed, 3),
        "fps": round(len(frames) / elapsed, 3) if elapsed > 0 else 0.0,
        "dry_run": dry_run,
        "system_before": before,
        "system_after": after,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_script_logging(args.verbose)
    if args.component == "object":
        result = benchmark_object(args.iterations, args.warmup, args.image)
    else:
        result = benchmark_camera(args.iterations, args.dry_run_camera)
    print_json(result)
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
