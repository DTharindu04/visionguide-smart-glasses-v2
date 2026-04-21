"""Run structured troubleshooting checks for field support."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

try:
    from ._common import PROJECT_ROOT, configure_script_logging, load_settings, print_json
except ImportError:
    from _common import PROJECT_ROOT, configure_script_logging, load_settings, print_json
from config.settings import validate_settings
from config.constants import InferenceBackend, ModelAssetTier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deployment troubleshooting checks")
    parser.add_argument("--include-model-probe", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def check_writable(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path, delete=True, encoding="utf-8") as handle:
            handle.write("ok")
        return True, "writable"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_script_logging(args.verbose)
    settings = load_settings()
    checks: list[dict[str, object]] = []

    issues = validate_settings(settings)
    checks.append(
        {
            "name": "settings_validation",
            "ok": not any(issue.severity.value == "error" for issue in issues),
            "details": [str(issue) for issue in issues],
            "severity": "error" if any(issue.severity.value == "error" for issue in issues) else "info",
        }
    )

    for name, path in {
        "storage_dir": settings.paths.storage_dir,
        "face_store_dir": settings.paths.face_store_dir,
        "cache_dir": settings.paths.cache_dir,
        "log_dir": settings.paths.log_dir,
        "settings_dir": settings.paths.settings_dir,
    }.items():
        ok, detail = check_writable(path)
        checks.append({"name": f"writable_{name}", "ok": ok, "details": detail, "severity": "error"})

    for module in ("numpy", "PIL", "cv2"):
        checks.append(
            {
                "name": f"python_module_{module}",
                "ok": importlib.util.find_spec(module) is not None,
                "details": "importable" if importlib.util.find_spec(module) is not None else "missing",
                "severity": "error",
            }
        )
    pytesseract_ok = importlib.util.find_spec("pytesseract") is not None
    checks.append(
        {
            "name": "python_module_pytesseract",
            "ok": pytesseract_ok,
            "details": "importable" if pytesseract_ok else "missing",
            "severity": "error" if settings.models.ocr_eng.exists() else "warning",
        }
    )
    onnxruntime_required = any(
        asset.backend == InferenceBackend.ONNX_RUNTIME
        and asset.path.exists()
        and asset.tier in {ModelAssetTier.SAFETY_CRITICAL, ModelAssetTier.CORE_FEATURE}
        for asset in settings.models.assets()
    )
    onnxruntime_ok = importlib.util.find_spec("onnxruntime") is not None
    checks.append(
        {
            "name": "python_module_onnxruntime",
            "ok": onnxruntime_ok,
            "details": "importable" if onnxruntime_ok else "missing",
            "severity": "error" if onnxruntime_required else "warning",
        }
    )

    espeak_ok = shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None
    pyttsx3_ok = importlib.util.find_spec("pyttsx3") is not None
    checks.append(
        {
            "name": "python_module_pyttsx3",
            "ok": pyttsx3_ok,
            "details": "importable" if pyttsx3_ok else "missing",
            "severity": "warning" if espeak_ok else "error",
        }
    )

    checks.append(
        {
            "name": "offline_tts_executable",
            "ok": espeak_ok,
            "details": "espeak available" if espeak_ok else "using pyttsx3 fallback if installed",
            "severity": "warning" if pyttsx3_ok else "error",
        }
    )
    tesseract_ok = shutil.which("tesseract") is not None
    checks.append(
        {
            "name": "tesseract_executable",
            "ok": tesseract_ok,
            "details": "tesseract available" if tesseract_ok else "missing",
            "severity": "error" if settings.models.ocr_eng.exists() else "warning",
        }
    )

    for asset in settings.models.assets():
        checks.append(
            {
                "name": f"model_asset_{asset.asset_id}",
                "ok": asset.path.exists(),
                "details": str(asset.path),
                "tier": asset.tier.value,
                "severity": "error" if asset.tier.value in {"safety_critical", "core_feature"} else "warning",
            }
        )

    if args.include_model_probe:
        completed = subprocess.run(
            [sys.executable, "scripts/validate_models.py", "--probe-inference"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        checks.append(
            {
                "name": "model_probe_command",
                "ok": completed.returncode == 0,
                "severity": "error",
                "details": {
                    "returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-1000:],
                },
            }
        )

    print_json({"checks": checks})
    hard_failure = any(
        not bool(check["ok"]) and check.get("severity") == "error"
        for check in checks
    )
    return 1 if hard_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
