"""Run safe offline smoke checks for deployment readiness."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Sequence

try:
    from ._common import PROJECT_ROOT, configure_script_logging, load_settings, print_json
except ImportError:
    from _common import PROJECT_ROOT, configure_script_logging, load_settings, print_json
from config.settings import validate_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline smoke tests")
    parser.add_argument("--skip-dry-run", action="store_true")
    parser.add_argument("--skip-model-probe", action="store_true")
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    return parser


def run_command(args: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": args,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "ok": completed.returncode == 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_script_logging(args.verbose)
    settings = load_settings()
    validation_issues = validate_settings(settings)
    checks: list[dict[str, object]] = [
        {
            "name": "settings_validation",
            "ok": not any(issue.severity.value == "error" for issue in validation_issues),
            "issues": [str(issue) for issue in validation_issues],
        },
        {
            "name": "compileall",
            **run_command([sys.executable, "-m", "compileall", "main.py", "audio", "camera", "config", "core", "decision", "diagnostics", "face", "ocr", "state", "vision", "scripts"]),
        },
    ]
    if not args.skip_model_probe:
        checks.append(
            {
                "name": "safety_model_probe",
                **run_command([sys.executable, "scripts/validate_models.py"]),
            }
        )
    if not args.skip_dry_run:
        checks.append(
            {
                "name": "runtime_dry_run",
                **run_command(
                    [
                        sys.executable,
                        "main.py",
                        "--dry-run",
                        "--max-frames",
                        str(args.max_frames),
                        "--log-level",
                        "ERROR",
                    ]
                ),
            }
        )
    print_json({"checks": checks})
    return 1 if not all(bool(check["ok"]) for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
