"""Process-level runtime tuning for Raspberry Pi deployments."""

from __future__ import annotations

import logging
import os

from config.constants import RASPBERRY_PI_4_CPU_CORES
from config.settings import AppSettings


def apply_runtime_optimizations(settings: AppSettings, logger: logging.Logger | None = None) -> None:
    """Apply conservative CPU/thread settings before inference services load."""

    log = logger or logging.getLogger(__name__)
    worker_threads = max(1, min(settings.optimization.worker_threads, RASPBERRY_PI_4_CPU_CORES))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, str(worker_threads))

    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        log.debug("OpenCV unavailable while applying runtime optimizations")
        return

    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(max(1, settings.optimization.opencv_threads))
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
    except Exception as exc:
        log.warning("Failed to apply OpenCV runtime optimizations: %s", exc)
