"""Collect camera/image quality statistics and threshold recommendations."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from ._common import capture_frames, configure_script_logging, load_rgb_images, load_settings, percentile, print_json
except ImportError:
    from _common import capture_frames, configure_script_logging, load_rgb_images, load_settings, percentile, print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Camera and face-quality calibration helper")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--images-dir", type=Path)
    source.add_argument("--camera", action="store_true")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def analyze_frames(frames: list[np.ndarray]) -> dict[str, object]:
    import cv2  # type: ignore[import-not-found]

    brightness_values: list[float] = []
    blur_values: list[float] = []
    dimensions: list[tuple[int, int]] = []
    for frame in frames:
        rgb = np.asarray(frame)
        height, width = rgb.shape[:2]
        dimensions.append((width, height))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        brightness_values.append(float(np.mean(gray)))
        blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    brightness_p10 = percentile(brightness_values, 10) or 0.0
    brightness_p90 = percentile(brightness_values, 90) or 0.0
    blur_p25 = percentile(blur_values, 25) or 0.0
    return {
        "frames_analyzed": len(frames),
        "width_height_samples": dimensions[:5],
        "brightness": {
            "min": min(brightness_values),
            "median": percentile(brightness_values, 50),
            "max": max(brightness_values),
            "p10": brightness_p10,
            "p90": brightness_p90,
        },
        "blur_laplacian_variance": {
            "min": min(blur_values),
            "median": percentile(blur_values, 50),
            "max": max(blur_values),
            "p25": blur_p25,
        },
        "recommended_face_thresholds": {
            "min_brightness": max(20.0, round(brightness_p10 * 0.75, 1)),
            "max_brightness": min(245.0, round(brightness_p90 * 1.20, 1)),
            "min_laplacian_blur_variance": max(20.0, round(blur_p25 * 0.70, 1)),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_script_logging(args.verbose)
    settings = load_settings()
    if args.images_dir:
        frames = load_rgb_images(args.images_dir, limit=args.frames)
        source = {"type": "images", "path": str(args.images_dir)}
    else:
        frames = capture_frames(settings, args.frames, args.timeout, dry_run=args.dry_run)
        source = {"type": "camera", "dry_run": args.dry_run}
    report = {
        "generated_at": time.time(),
        "source": source,
        "analysis": analyze_frames(frames),
    }
    output = args.output or (settings.paths.settings_dir / "camera_calibration_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(__import__("json").dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["output"] = str(output)
    print_json(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
