"""Enroll, list, and remove local face identities."""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Sequence

try:
    from ._common import capture_frames, configure_script_logging, load_rgb_images, load_settings, print_json
except ImportError:
    from _common import capture_frames, configure_script_logging, load_rgb_images, load_settings, print_json
from core.event_manager import EventManager
from core.frame_store import FrameStore
from face.enrollment import FaceEnrollmentManager
from face.face_detection_service import FaceDetectionService
from face.face_recognition_service import FaceRecognitionService
from face.face_store import FaceEmbeddingStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline face enrollment utility")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    images = subparsers.add_parser("enroll-images", help="Enroll a person from an image directory")
    images.add_argument("--name", required=True)
    images.add_argument("--images-dir", required=True, type=Path)
    images.add_argument("--limit", type=int, default=None)
    images.add_argument("--update-existing", action="store_true")
    images.add_argument("--no-backup", action="store_true")

    camera = subparsers.add_parser("enroll-camera", help="Enroll a person from live camera frames")
    camera.add_argument("--name", required=True)
    camera.add_argument("--frames", type=int, default=30)
    camera.add_argument("--timeout", type=float, default=20.0)
    camera.add_argument("--dry-run", action="store_true")
    camera.add_argument("--update-existing", action="store_true")
    camera.add_argument("--no-backup", action="store_true")

    subparsers.add_parser("list", help="List enrolled people")

    remove = subparsers.add_parser("remove", help="Remove one enrolled person")
    remove.add_argument("--person-id", required=True)
    remove.add_argument("--yes", action="store_true", help="Confirm deletion without an interactive prompt")
    return parser


def build_enrollment_manager() -> tuple[FaceEnrollmentManager, FaceEmbeddingStore]:
    settings = load_settings()
    events = EventManager(logger=logging.getLogger("scripts.face_enrollment.events"))
    frame_store = FrameStore()
    store = FaceEmbeddingStore(
        settings.paths.face_store_dir,
        max_embeddings_per_person=settings.thresholds.faces.enrollment_max_samples,
    )
    detector = FaceDetectionService(settings, frame_store, events, logging.getLogger("scripts.face_detection"))
    recognizer = FaceRecognitionService(settings, frame_store, events, store, logging.getLogger("scripts.face_recognition"))
    return FaceEnrollmentManager(settings, detector, recognizer, store), store


def backup_face_store(store_dir: Path) -> Path | None:
    files = [store_dir / "people.json", store_dir / "embeddings.npz"]
    existing = [path for path in files if path.exists()]
    if not existing:
        return None
    backup_dir = store_dir / "backups" / time.strftime("%Y%m%d-%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def name_exists(store: FaceEmbeddingStore, name: str) -> bool:
    clean = " ".join(name.split()).casefold()
    return any(person.name.casefold() == clean for person in store.list_people())


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_script_logging(args.verbose)
    settings = load_settings()
    manager, store = build_enrollment_manager()

    if args.command == "list":
        print_json({"people": [person.__dict__ for person in store.list_people()]})
        return 0

    if args.command == "remove":
        if not args.yes:
            raise SystemExit("Refusing to remove without --yes.")
        removed = store.remove_person(args.person_id)
        print_json({"removed": removed, "person_id": args.person_id})
        return 0 if removed else 1

    if args.command == "enroll-images":
        frames = load_rgb_images(args.images_dir, limit=args.limit)
    elif args.command == "enroll-camera":
        frames = capture_frames(settings, args.frames, args.timeout, dry_run=args.dry_run)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    if name_exists(store, args.name) and not args.update_existing:
        raise SystemExit("Person name already exists. Re-run with --update-existing to append samples.")
    backup_dir = None if args.no_backup else backup_face_store(settings.paths.face_store_dir)
    result = manager.enroll_from_frames(args.name, frames)
    print_json(
        {
            "person": result.person.__dict__,
            "accepted_samples": result.accepted_samples,
            "rejected_samples": result.rejected_samples,
            "rejection_reasons": result.rejection_reasons,
            "backup_dir": backup_dir,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
