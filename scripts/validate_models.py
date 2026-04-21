"""Validate local model manifests and backend compatibility."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

try:
    from ._common import configure_script_logging, load_settings, print_json
except ImportError:
    from _common import configure_script_logging, load_settings, print_json
from config.constants import InferenceBackend, ModelAssetTier
from config.settings import validate_settings
from core.model_runtime import LocalModelLoader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate offline model assets")
    parser.add_argument(
        "--probe-inference",
        action="store_true",
        help="Probe every included model; safety-critical models are probed by default.",
    )
    parser.add_argument("--skip-inference-probe", action="store_true", help="Only validate files and loadable metadata")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_script_logging(args.verbose)
    settings = load_settings()
    model_logger = logging.getLogger("scripts.validate_models.loader")
    model_logger.addHandler(logging.NullHandler())
    model_logger.propagate = False
    loader = LocalModelLoader(model_logger)
    validation_issues = [str(issue) for issue in validate_settings(settings)]
    model_results: list[dict[str, object]] = []
    has_errors = any(issue.startswith("error:") for issue in validation_issues)

    for asset in settings.models.assets():
        if asset.tier.value == "optional_feature" and not args.include_optional:
            continue
        result: dict[str, object] = {
            "asset_id": asset.asset_id,
            "path": asset.path,
            "exists": asset.path.exists(),
            "tier": asset.tier,
            "backend": asset.backend,
            "input_shape": asset.input_shape,
            "ok": asset.path.exists(),
        }
        should_probe_inference = (
            not args.skip_inference_probe
            and (asset.tier == ModelAssetTier.SAFETY_CRITICAL or args.probe_inference)
        )
        if (
            asset.path.exists()
            and args.skip_inference_probe
            and asset.backend not in {InferenceBackend.STATIC_FILE, InferenceBackend.TESSERACT}
        ):
            result["runtime_backend"] = asset.backend
            result["runtime_input_shape"] = asset.input_shape
            result["inference_probed"] = False
            result["ok"] = True
        elif asset.path.exists() and asset.backend not in {InferenceBackend.STATIC_FILE, InferenceBackend.TESSERACT}:
            try:
                runner = loader.probe(
                    asset.path,
                    preferred_backend=asset.backend,
                    num_threads=settings.optimization.tflite_threads,
                    input_shape=asset.input_shape,
                    run_inference=should_probe_inference,
                )
                result["runtime_backend"] = runner.backend
                result["runtime_input_shape"] = runner.input_info.shape
                result["inference_probed"] = should_probe_inference
                result["ok"] = True
            except Exception as exc:
                result["ok"] = False
                result["error"] = f"{type(exc).__name__}: {exc}"
        if not result["ok"]:
            has_errors = True
        model_results.append(result)

    print_json({"validation_issues": validation_issues, "models": model_results})
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
