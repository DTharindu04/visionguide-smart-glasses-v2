"""Entry point for the offline smart glasses runtime."""

from __future__ import annotations

import argparse
import logging
import signal
from typing import Sequence

from audio.tts_manager import LoggingTtsBackend, OfflineTtsManager
from camera.camera_manager import CameraManager
from config.settings import ensure_runtime_directories, get_settings
from core.event_manager import EventManager
from core.frame_store import FrameStore
from core.logging_manager import configure_logging
from core.runtime_optimization import apply_runtime_optimizations
from core.scheduler import EngineScheduler
from decision.decision_engine import DecisionEngine
from diagnostics.performance_monitor import PerformanceMonitor
from state.state_manager import StateManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Raspberry Pi smart glasses runtime")
    parser.add_argument("--dry-run", action="store_true", help="Run with synthetic frames if no camera is available.")
    parser.add_argument(
        "--unsafe-allow-missing-models",
        action="store_true",
        help="Allow startup despite validation errors outside production, or with --dry-run.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many captured frames.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    ensure_runtime_directories(settings)
    logger = configure_logging(settings, args.log_level)
    apply_runtime_optimizations(settings, logger)

    if args.unsafe_allow_missing_models and settings.environment == "production" and not args.dry_run:
        logger.error("--unsafe-allow-missing-models is blocked in production unless --dry-run is also set.")
        return 2

    from face.emotion_detection_service import EmotionDetectionService
    from face.face_detection_service import FaceDetectionService
    from face.face_recognition_service import FaceRecognitionService
    from face.face_store import FaceEmbeddingStore
    from ocr.ocr_service import OcrService
    from vision.object_detection_service import ObjectDetectionService

    event_manager = EventManager(logger=logging.getLogger("smart_glasses.events"))
    frame_store = FrameStore(max_frames=max(3, settings.camera.frame_queue_size + 2))
    camera_manager = CameraManager(
        settings.camera,
        logger=logging.getLogger("smart_glasses.camera"),
        allow_synthetic=args.dry_run,
        prefer_synthetic=args.dry_run,
    )
    state_manager = StateManager(settings, event_manager, logger=logging.getLogger("smart_glasses.state"))
    decision_engine = DecisionEngine(settings, event_manager, logger=logging.getLogger("smart_glasses.decision"))
    face_store = FaceEmbeddingStore(
        settings.paths.face_store_dir,
        max_embeddings_per_person=settings.thresholds.faces.enrollment_max_samples,
    )
    object_detection_service = ObjectDetectionService(
        settings,
        frame_store,
        event_manager,
        logger=logging.getLogger("smart_glasses.object_detection"),
    )
    face_detection_service = FaceDetectionService(
        settings,
        frame_store,
        event_manager,
        logger=logging.getLogger("smart_glasses.face_detection"),
    )
    face_recognition_service = FaceRecognitionService(
        settings,
        frame_store,
        event_manager,
        face_store,
        logger=logging.getLogger("smart_glasses.face_recognition"),
    )
    emotion_detection_service = EmotionDetectionService(
        settings,
        frame_store,
        event_manager,
        logger=logging.getLogger("smart_glasses.emotion"),
    )
    ocr_service = OcrService(settings, frame_store, event_manager, logger=logging.getLogger("smart_glasses.ocr"))
    audio_logger = logging.getLogger("smart_glasses.audio")
    audio_manager = OfflineTtsManager(
        settings,
        event_manager,
        logger=audio_logger,
        backend=LoggingTtsBackend(audio_logger) if args.dry_run else None,
    )
    performance_monitor = PerformanceMonitor(
        settings,
        event_manager,
        logger=logging.getLogger("smart_glasses.performance"),
    )
    scheduler = EngineScheduler(
        settings,
        event_manager,
        frame_store,
        camera_manager,
        state_manager,
        decision_engine,
        performance_monitor,
        service_handlers=(
            object_detection_service,
            face_detection_service,
            face_recognition_service,
            emotion_detection_service,
            ocr_service,
            audio_manager,
        ),
        logger=logging.getLogger("smart_glasses.scheduler"),
    )

    def stop(signum: int, _frame: object) -> None:
        logger.info("Received signal %s", signum)
        scheduler.request_stop(f"signal {signum}")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    require_valid_settings = not (args.unsafe_allow_missing_models or args.dry_run)
    return scheduler.run(max_frames=args.max_frames, require_valid_settings=require_valid_settings)


if __name__ == "__main__":
    raise SystemExit(main())
