"""Logging setup for local production diagnostics."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

from config.settings import AppSettings


class JsonLineFormatter(logging.Formatter):
    """Compact JSON-lines formatter for machine-readable local logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


class HumanFormatter(logging.Formatter):
    """Readable formatter for console logs."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")


class LoggingManager:
    """Owns process-wide logging configuration."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    def configure(self, level: str = "INFO") -> logging.Logger:
        log_level = getattr(logging, level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(HumanFormatter())
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)

        self._settings.paths.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._settings.paths.log_dir / "smart_glasses.log"
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=self._settings.diagnostics.log_rotation_mb * 1024 * 1024,
            backupCount=self._settings.diagnostics.retained_log_files,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonLineFormatter())
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logger = logging.getLogger("smart_glasses")
        logger.info("Logging initialized", extra={"log_path": str(log_path)})
        return logger

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


def configure_logging(settings: AppSettings, level: str = "INFO") -> logging.Logger:
    return LoggingManager(settings).configure(level)
