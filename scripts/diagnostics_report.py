"""Generate a local diagnostics report without sending data off-device."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Sequence

try:
    from ._common import configure_script_logging, json_default, load_settings, print_json
except ImportError:
    from _common import configure_script_logging, json_default, load_settings, print_json
from config.settings import validate_settings
from core.event_manager import EventManager
from diagnostics.performance_monitor import PerformanceMonitor
from config.constants import RuntimeMode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate local diagnostics report")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_script_logging(args.verbose)
    settings = load_settings()
    events = EventManager()
    monitor = PerformanceMonitor(settings, events)
    snapshot = monitor.sample(RuntimeMode.NORMAL)
    issues = validate_settings(settings)
    report = {
        "generated_at": time.time(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "paths": {
            "project_root": settings.paths.project_root,
            "model_dir": settings.paths.model_dir,
            "storage_dir": settings.paths.storage_dir,
            "log_dir": settings.paths.log_dir,
        },
        "camera": settings.camera,
        "optimization": settings.optimization,
        "audio": {
            "backend": settings.audio.backend,
            "voice": settings.audio.voice,
            "espeak_available": command_available("espeak-ng") or command_available("espeak"),
        },
        "models": [
            {
                "asset_id": asset.asset_id,
                "path": asset.path,
                "exists": asset.path.exists(),
                "tier": asset.tier,
                "backend": asset.backend,
            }
            for asset in settings.models.assets()
        ],
        "performance_snapshot": snapshot,
        "validation_issues": [str(issue) for issue in issues],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, default=json_default, indent=2, sort_keys=True), encoding="utf-8")
    print_json(report)
    return 1 if any(issue.severity.value == "error" for issue in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
