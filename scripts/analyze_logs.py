"""Analyze local JSON-line runtime logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

try:
    from ._common import configure_script_logging, load_settings, print_json
except ImportError:
    from _common import configure_script_logging, load_settings, print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze smart glasses logs")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--all-rotated", action="store_true")
    parser.add_argument("--limit-examples", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser


def iter_log_files(base: Path, all_rotated: bool) -> list[Path]:
    if not all_rotated:
        return [base]
    files = [base]
    files.extend(sorted(base.parent.glob(base.name + ".*")))
    return [path for path in files if path.exists()]


def analyze(files: list[Path], limit_examples: int) -> dict[str, object]:
    levels: Counter[str] = Counter()
    loggers: Counter[str] = Counter()
    messages: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    malformed = 0
    first_ts: str | None = None
    last_ts: str | None = None
    total = 0

    for path in files:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            total += 1
            ts = str(record.get("ts", ""))
            first_ts = first_ts or ts
            last_ts = ts or last_ts
            level = str(record.get("level", "UNKNOWN"))
            logger = str(record.get("logger", "unknown"))
            message = str(record.get("message", ""))
            levels[level] += 1
            loggers[logger] += 1
            if level in {"ERROR", "CRITICAL", "WARNING"}:
                messages[message[:180]] += 1
                if len(examples) < limit_examples:
                    examples.append(record)
    return {
        "files": [str(path) for path in files],
        "records": total,
        "malformed_lines": malformed,
        "time_range": {"first": first_ts, "last": last_ts},
        "levels": dict(levels.most_common()),
        "top_loggers": dict(loggers.most_common(12)),
        "top_warning_error_messages": dict(messages.most_common(12)),
        "examples": examples,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_script_logging(args.verbose)
    settings = load_settings()
    log_file = args.log_file or (settings.paths.log_dir / "smart_glasses.log")
    result = analyze(iter_log_files(log_file, args.all_rotated), args.limit_examples)
    print_json(result)
    return 1 if result["malformed_lines"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
