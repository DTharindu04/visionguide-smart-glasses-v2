"""Production settings for the offline Raspberry Pi smart glasses system."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import tomllib
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from config.constants import (
    DEFAULT_ENVIRONMENT,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_FILENAMES,
    DEFAULT_MODEL_INPUT_SHAPES,
    DEFAULT_OPTIMIZATION_PROFILE,
    DEFAULT_TIMEZONE,
    ENV_PREFIX,
    InferenceBackend,
    MAX_AUDIO_QUEUE_SIZE,
    MAX_DECISION_QUEUE_SIZE,
    MAX_DIAGNOSTIC_EVENT_QUEUE_SIZE,
    MAX_FRAME_QUEUE_SIZE,
    MOBILE_HAZARD_CLASSES,
    ModelAssetTier,
    ModuleName,
    NAVIGATION_RELEVANT_CLASSES,
    OcrPolicy,
    PI_DEFAULT_OPENCV_THREADS,
    PI_DEFAULT_TFLITE_THREADS,
    PI_THERMAL_HARD_LIMIT_C,
    PI_THERMAL_SOFT_LIMIT_C,
    PROJECT_ROOT,
    RASPBERRY_PI_4_CPU_CORES,
    AudioPriority,
    RuntimeMode,
    SpeechCategory,
    ValidationSeverity,
)

ConfigData = Mapping[str, Any]
FrameInterval = int | None


def _backend_for_model_path(path: Path) -> InferenceBackend:
    suffix = path.suffix.lower()
    if suffix == ".tflite":
        return InferenceBackend.TFLITE_RUNTIME
    if suffix in {".onnx", ".caffemodel"}:
        return InferenceBackend.OPENCV_DNN
    return InferenceBackend.STATIC_FILE


def _env_key(name: str) -> str:
    return f"{ENV_PREFIX}_{name}"


def _nested_value(config: ConfigData, keys: tuple[str, ...], default: Any) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _load_toml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as file_handle:
        data = tomllib.load(file_handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a TOML table: {path}")
    return data


def _load_env_file(path: Path, override: bool = False) -> None:
    """Load simple KEY=VALUE pairs from a local .env file without logging secrets."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", maxsplit=1)
        key = key.strip()
        if not key or any(character.isspace() for character in key):
            continue
        value = _parse_env_value(raw_value.strip())
        if override or key not in os.environ:
            os.environ[key] = value


def _parse_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
        if value:
            value = value.replace("\\n", "\n").replace("\\t", "\t")
        return value
    if " #" in value:
        value = value.split(" #", maxsplit=1)[0].rstrip()
    return value


def _env_str(name: str, default: str) -> str:
    value = os.getenv(_env_key(name), default).strip()
    return value if value else default


def _config_str(config: ConfigData, keys: tuple[str, ...], default: str) -> str:
    value = _nested_value(config, keys, default)
    if not isinstance(value, str):
        dotted = ".".join(keys)
        raise ValueError(f"Configuration value {dotted} must be a string.")
    return value.strip() or default


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(_env_key(name))
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{_env_key(name)} must be an integer, got {raw_value!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{_env_key(name)} must be between {minimum} and {maximum}, got {value}")
    return value


def _config_int(config: ConfigData, keys: tuple[str, ...], default: int) -> int:
    value = _nested_value(config, keys, default)
    if isinstance(value, bool) or not isinstance(value, int):
        dotted = ".".join(keys)
        raise ValueError(f"Configuration value {dotted} must be an integer.")
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw_value = os.getenv(_env_key(name))
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{_env_key(name)} must be a float, got {raw_value!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{_env_key(name)} must be between {minimum} and {maximum}, got {value}")
    return value


def _config_float(config: ConfigData, keys: tuple[str, ...], default: float) -> float:
    value = _nested_value(config, keys, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        dotted = ".".join(keys)
        raise ValueError(f"Configuration value {dotted} must be a number.")
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(_env_key(name))
    if raw_value is None or raw_value.strip() == "":
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{_env_key(name)} must be a boolean, got {raw_value!r}")


def _config_bool(config: ConfigData, keys: tuple[str, ...], default: bool) -> bool:
    value = _nested_value(config, keys, default)
    if not isinstance(value, bool):
        dotted = ".".join(keys)
        raise ValueError(f"Configuration value {dotted} must be a boolean.")
    return value


def _env_path(name: str, default: Path, base: Path = PROJECT_ROOT) -> Path:
    raw_value = os.getenv(_env_key(name))
    value = Path(raw_value).expanduser() if raw_value and raw_value.strip() else default
    return value if value.is_absolute() else (base / value).resolve()


def _config_path(config: ConfigData, keys: tuple[str, ...], default: Path, base: Path) -> Path:
    value = _nested_value(config, keys, default)
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        path = Path(value).expanduser()
    else:
        dotted = ".".join(keys)
        raise ValueError(f"Configuration value {dotted} must be a path string.")
    return path if path.is_absolute() else (base / path).resolve()


def _config_str_tuple(config: ConfigData, keys: tuple[str, ...], default: tuple[str, ...]) -> tuple[str, ...]:
    value = _nested_value(config, keys, default)
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        dotted = ".".join(keys)
        raise ValueError(f"Configuration value {dotted} must be a list of strings.")
    return tuple(item.strip() for item in value if item.strip())


def _audio_priority_from_value(value: str | int | AudioPriority) -> AudioPriority:
    if isinstance(value, AudioPriority):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return AudioPriority(value)
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.isdigit():
            return AudioPriority(int(normalized))
        return AudioPriority[normalized]
    raise ValueError(f"Invalid audio priority value: {value!r}")


def _config_audio_priority(config: ConfigData, keys: tuple[str, ...], default: AudioPriority) -> AudioPriority:
    value = _nested_value(config, keys, default)
    try:
        return _audio_priority_from_value(value)
    except (KeyError, ValueError) as exc:
        dotted = ".".join(keys)
        raise ValueError(f"Configuration value {dotted} must be an AudioPriority name or value.") from exc


def _env_audio_priority(name: str, default: AudioPriority) -> AudioPriority:
    raw_value = os.getenv(_env_key(name))
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return _audio_priority_from_value(raw_value)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{_env_key(name)} must be an AudioPriority name or value.") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ValidationIssue:
    """A validation finding with explicit severity."""

    severity: ValidationSeverity
    message: str

    def __str__(self) -> str:
        return f"{self.severity.value}: {self.message}"


@dataclass(frozen=True)
class PathSettings:
    """Filesystem locations used by the offline runtime."""

    project_root: Path
    config_dir: Path
    runtime_config_file: Path
    model_dir: Path
    storage_dir: Path
    face_store_dir: Path
    cache_dir: Path
    log_dir: Path
    settings_dir: Path


@dataclass(frozen=True)
class CameraSettings:
    """Camera and frame sizing settings tuned for Raspberry Pi Camera Module 2."""

    capture_width: int
    capture_height: int
    inference_width: int
    inference_height: int
    target_fps: int
    frame_queue_size: int = MAX_FRAME_QUEUE_SIZE
    warmup_frames: int = 8
    drop_stale_frames: bool = True
    max_frame_age_ms: int = 160
    hflip: bool = False
    vflip: bool = False


@dataclass(frozen=True)
class ModelAsset:
    """A local model or model-adjacent asset expected by the runtime."""

    asset_id: str
    path: Path
    tier: ModelAssetTier
    backend: InferenceBackend
    required_modes: tuple[RuntimeMode, ...]
    input_schema: str
    output_schema: str
    input_shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ModelPathSettings:
    """Expected local model and OCR data paths."""

    manifest_path: Path
    object_detection: Path
    object_labels: Path
    face_detection: Path
    face_recognition: Path
    emotion_detection: Path
    ocr_tessdata_dir: Path
    ocr_eng: Path
    ocr_sin: Path

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "ModelPathSettings":
        object_dir = model_dir / "object_detection"
        face_detection_dir = model_dir / "face_detection"
        face_recognition_dir = model_dir / "face_recognition"
        emotion_dir = model_dir / "emotion"
        ocr_tessdata_dir = model_dir / "ocr" / "tessdata"
        return cls(
            manifest_path=model_dir / "manifest.local.toml",
            object_detection=object_dir / DEFAULT_MODEL_FILENAMES["object_detection"],
            object_labels=object_dir / DEFAULT_MODEL_FILENAMES["object_labels"],
            face_detection=face_detection_dir / DEFAULT_MODEL_FILENAMES["face_detection"],
            face_recognition=face_recognition_dir / DEFAULT_MODEL_FILENAMES["face_recognition"],
            emotion_detection=emotion_dir / DEFAULT_MODEL_FILENAMES["emotion_detection"],
            ocr_tessdata_dir=ocr_tessdata_dir,
            ocr_eng=ocr_tessdata_dir / DEFAULT_MODEL_FILENAMES["ocr_eng"],
            ocr_sin=ocr_tessdata_dir / DEFAULT_MODEL_FILENAMES["ocr_sin"],
        )

    def assets(self) -> tuple[ModelAsset, ...]:
        all_modes = tuple(RuntimeMode)
        return (
            ModelAsset(
                asset_id="object_detection",
                path=self.object_detection,
                tier=ModelAssetTier.SAFETY_CRITICAL,
                backend=InferenceBackend.ONNX_RUNTIME
                if self.object_detection.suffix.lower() == ".onnx"
                else _backend_for_model_path(self.object_detection),
                required_modes=all_modes,
                input_schema="RGB image tensor resized according to model metadata.",
                output_schema="Detection boxes, class IDs, and confidence scores.",
                input_shape=DEFAULT_MODEL_INPUT_SHAPES["object_detection"],
            ),
            ModelAsset(
                asset_id="object_labels",
                path=self.object_labels,
                tier=ModelAssetTier.SAFETY_CRITICAL,
                backend=InferenceBackend.STATIC_FILE,
                required_modes=all_modes,
                input_schema="UTF-8 text labels, one class per line.",
                output_schema="Navigation class names used by object_detection.",
            ),
            ModelAsset(
                asset_id="face_detection",
                path=self.face_detection,
                tier=ModelAssetTier.CORE_FEATURE,
                backend=_backend_for_model_path(self.face_detection),
                required_modes=(RuntimeMode.NORMAL, RuntimeMode.FACE),
                input_schema="RGB face detector input tensor resized according to model metadata.",
                output_schema="Face boxes, landmarks when available, and confidence scores.",
                input_shape=DEFAULT_MODEL_INPUT_SHAPES["face_detection"],
            ),
            ModelAsset(
                asset_id="face_recognition",
                path=self.face_recognition,
                tier=ModelAssetTier.CORE_FEATURE,
                backend=_backend_for_model_path(self.face_recognition),
                required_modes=(RuntimeMode.FACE,),
                input_schema="Aligned face crop tensor.",
                output_schema="L2-normalized face embedding.",
                input_shape=DEFAULT_MODEL_INPUT_SHAPES["face_recognition"],
            ),
            ModelAsset(
                asset_id="emotion_detection",
                path=self.emotion_detection,
                tier=ModelAssetTier.OPTIONAL_FEATURE,
                backend=_backend_for_model_path(self.emotion_detection),
                required_modes=(RuntimeMode.FACE,),
                input_schema="Aligned face crop tensor.",
                output_schema="Emotion class scores.",
                input_shape=DEFAULT_MODEL_INPUT_SHAPES["emotion_detection"],
            ),
            ModelAsset(
                asset_id="ocr_eng",
                path=self.ocr_eng,
                tier=ModelAssetTier.CORE_FEATURE,
                backend=InferenceBackend.TESSERACT,
                required_modes=(RuntimeMode.READING,),
                input_schema="Preprocessed high-contrast text image.",
                output_schema="Recognized English text with confidence values.",
            ),
            ModelAsset(
                asset_id="ocr_sin",
                path=self.ocr_sin,
                tier=ModelAssetTier.OPTIONAL_FEATURE,
                backend=InferenceBackend.TESSERACT,
                required_modes=(RuntimeMode.READING,),
                input_schema="Preprocessed high-contrast text image.",
                output_schema="Recognized Sinhala text with confidence values.",
            ),
        )

    def missing_model_files(self, tiers: tuple[ModelAssetTier, ...] | None = None) -> tuple[ModelAsset, ...]:
        allowed_tiers = tiers or tuple(ModelAssetTier)
        return tuple(asset for asset in self.assets() if asset.tier in allowed_tiers and not asset.path.exists())

    def installed_model_files(self) -> tuple[ModelAsset, ...]:
        return tuple(asset for asset in self.assets() if asset.path.exists())

    def asset_by_id(self, asset_id: str) -> ModelAsset:
        for asset in self.assets():
            if asset.asset_id == asset_id:
                return asset
        raise KeyError(f"Unknown model asset id: {asset_id}")


@dataclass(frozen=True)
class ModelManifestEntry:
    """Provenance metadata for an installed local model asset."""

    asset_id: str
    version: str
    license: str
    sha256: str
    input_schema: str
    output_schema: str


@dataclass(frozen=True)
class ModelManifest:
    """Loaded local model manifest."""

    path: Path
    entries: Mapping[str, ModelManifestEntry]

    @classmethod
    def load(cls, path: Path) -> "ModelManifest":
        data = _load_toml_file(path)
        raw_entries = data.get("models", {})
        if not isinstance(raw_entries, Mapping):
            raise ValueError(f"Model manifest [models] table must be a mapping: {path}")

        entries: dict[str, ModelManifestEntry] = {}
        for asset_id, raw_entry in raw_entries.items():
            if not isinstance(asset_id, str) or not isinstance(raw_entry, Mapping):
                raise ValueError(f"Invalid model manifest entry for {asset_id!r}.")
            entries[asset_id] = ModelManifestEntry(
                asset_id=asset_id,
                version=str(raw_entry.get("version", "")).strip(),
                license=str(raw_entry.get("license", "")).strip(),
                sha256=str(raw_entry.get("sha256", "")).strip().lower(),
                input_schema=str(raw_entry.get("input_schema", "")).strip(),
                output_schema=str(raw_entry.get("output_schema", "")).strip(),
            )
        return cls(path=path, entries=entries)


@dataclass(frozen=True)
class ObjectThresholds:
    """Navigation-relevant object confidence and danger-zone thresholds."""

    navigation_classes: tuple[str, ...] = NAVIGATION_RELEVANT_CLASSES
    mobile_hazard_classes: tuple[str, ...] = MOBILE_HAZARD_CLASSES
    min_detection_confidence: float = 0.45
    min_danger_confidence: float = 0.55
    center_zone_x_min: float = 0.34
    center_zone_x_max: float = 0.66
    warning_bbox_area_ratio: float = 0.10
    close_bbox_area_ratio: float = 0.18
    critical_bbox_area_ratio: float = 0.28
    mobile_hazard_close_area_ratio: float = 0.14
    person_close_area_ratio: float = 0.20
    severity_warning_threshold: float = 0.45
    severity_danger_threshold: float = 0.70
    severity_critical_threshold: float = 0.88
    max_objects_per_frame: int = 8


@dataclass(frozen=True)
class FaceThresholds:
    """Face quality and recognition thresholds used before identity decisions."""

    min_face_width_px: int = 80
    min_face_height_px: int = 80
    min_laplacian_blur_variance: float = 80.0
    min_brightness: float = 45.0
    max_brightness: float = 215.0
    stable_frames_required: int = 3
    max_face_yaw_degrees: float = 28.0
    recognition_similarity_threshold: float = 0.62
    recognition_unknown_margin: float = 0.05
    recognition_min_confidence: float = 0.60
    emotion_min_confidence: float = 0.55
    enrollment_samples_required: int = 8
    enrollment_max_samples: int = 20


@dataclass(frozen=True)
class OcrThresholds:
    """OCR preprocessing and repeat-control thresholds."""

    min_trigger_interval_seconds: float = 4.0
    min_text_confidence: float = 55.0
    min_text_characters: int = 3
    repeat_similarity_threshold: float = 0.88
    cache_ttl_seconds: float = 120.0
    max_cache_entries: int = 20
    preprocessing_target_width_px: int = 960
    adaptive_threshold_block_size: int = 31
    adaptive_threshold_c: int = 11


@dataclass(frozen=True)
class ThresholdSettings:
    """All inference and decision thresholds in one immutable object."""

    objects: ObjectThresholds = field(default_factory=ObjectThresholds)
    faces: FaceThresholds = field(default_factory=FaceThresholds)
    ocr: OcrThresholds = field(default_factory=OcrThresholds)


@dataclass(frozen=True)
class CooldownSettings:
    """Speech and event cooldowns, in seconds."""

    danger_repeat_seconds: float = 1.2
    danger_clear_seconds: float = 2.5
    known_face_repeat_seconds: float = 45.0
    unknown_face_repeat_seconds: float = 25.0
    emotion_repeat_seconds: float = 60.0
    ocr_repeat_seconds: float = 30.0
    object_guidance_repeat_seconds: float = 8.0
    low_priority_context_repeat_seconds: float = 20.0
    system_status_repeat_seconds: float = 30.0
    same_message_dedup_seconds: float = 10.0
    stale_p1_ttl_seconds: float = 2.0
    stale_p2_ttl_seconds: float = 6.0
    stale_p3_ttl_seconds: float = 5.0
    stale_p4_ttl_seconds: float = 4.0

    def for_category(self, category: SpeechCategory) -> float:
        return {
            SpeechCategory.DANGER: self.danger_repeat_seconds,
            SpeechCategory.DANGER_CLEAR: self.danger_clear_seconds,
            SpeechCategory.KNOWN_FACE: self.known_face_repeat_seconds,
            SpeechCategory.UNKNOWN_FACE: self.unknown_face_repeat_seconds,
            SpeechCategory.OCR_TEXT: self.ocr_repeat_seconds,
            SpeechCategory.OBJECT_GUIDANCE: self.object_guidance_repeat_seconds,
            SpeechCategory.EMOTION: self.emotion_repeat_seconds,
            SpeechCategory.LOW_PRIORITY_CONTEXT: self.low_priority_context_repeat_seconds,
            SpeechCategory.SYSTEM_STATUS: self.system_status_repeat_seconds,
        }[category]

    def ttl_for_priority(self, priority: AudioPriority) -> float:
        return {
            AudioPriority.P1_DANGER: self.stale_p1_ttl_seconds,
            AudioPriority.P2_IDENTITY_OCR: self.stale_p2_ttl_seconds,
            AudioPriority.P3_NAVIGATION: self.stale_p3_ttl_seconds,
            AudioPriority.P4_CONTEXT: self.stale_p4_ttl_seconds,
        }[priority]


@dataclass(frozen=True)
class AudioSettings:
    """Offline TTS and audio queue behavior."""

    backend: str = "espeak-ng"
    voice: str = "en-us"
    speech_rate_wpm: int = 165
    volume: float = 0.95
    queue_size: int = MAX_AUDIO_QUEUE_SIZE
    p1_interrupts_all: bool = True
    p2_replaces_p3_p4: bool = True
    lowest_priority_in_quiet: AudioPriority = AudioPriority.P1_DANGER

    def allows_in_quiet(self, priority: AudioPriority) -> bool:
        return priority <= self.lowest_priority_in_quiet


@dataclass(frozen=True)
class DiagnosticsSettings:
    """Runtime observability settings kept local to the device."""

    enabled: bool = True
    metrics_interval_seconds: float = 5.0
    slow_frame_ms: int = 220
    slow_inference_ms: int = 180
    max_event_queue_size: int = MAX_DIAGNOSTIC_EVENT_QUEUE_SIZE
    log_rotation_mb: int = 10
    retained_log_files: int = 7


@dataclass(frozen=True)
class OptimizationPreset:
    """Hardware and backend tuning preset for Raspberry Pi 4."""

    name: str
    description: str
    capture_width: int
    capture_height: int
    inference_width: int
    inference_height: int
    target_fps: int
    max_frame_age_ms: int
    worker_threads: int
    opencv_threads: int
    tflite_threads: int
    enable_tracking: bool
    drop_stale_frames: bool
    thermal_soft_limit_c: float
    thermal_hard_limit_c: float
    low_power_enter_cpu_percent: float
    low_power_exit_cpu_percent: float


@dataclass(frozen=True)
class PerformanceProfile:
    """Mode-specific work scheduling policy."""

    mode: RuntimeMode
    active_modules: tuple[ModuleName, ...]
    camera_fps: int
    object_detection_every_n_frames: FrameInterval
    obstacle_analysis_every_n_frames: FrameInterval
    face_detection_every_n_frames: FrameInterval
    face_recognition_every_n_stable_faces: FrameInterval
    emotion_detection_every_n_face_updates: FrameInterval
    ocr_policy: OcrPolicy
    cooldown_multiplier: float
    lowest_allowed_audio_priority: AudioPriority
    allow_audio_interruptions: bool
    cpu_budget_percent: int
    stale_frame_ttl_ms: int
    max_model_latency_ms: Mapping[str, int]

    def allows_audio_priority(self, priority: AudioPriority) -> bool:
        return priority <= self.lowest_allowed_audio_priority


@dataclass(frozen=True)
class AppSettings:
    """Complete immutable application configuration."""

    environment: str
    language: str
    timezone: str
    paths: PathSettings
    camera: CameraSettings
    models: ModelPathSettings
    thresholds: ThresholdSettings
    cooldowns: CooldownSettings
    audio: AudioSettings
    diagnostics: DiagnosticsSettings
    optimization: OptimizationPreset
    performance_profiles: Mapping[RuntimeMode, PerformanceProfile]
    decision_queue_size: int = MAX_DECISION_QUEUE_SIZE

    def profile_for(self, mode: RuntimeMode) -> PerformanceProfile:
        return self.performance_profiles[mode]

    def missing_safety_model_files(self) -> tuple[ModelAsset, ...]:
        return self.models.missing_model_files((ModelAssetTier.SAFETY_CRITICAL,))

    def missing_feature_model_files(self) -> tuple[ModelAsset, ...]:
        return self.models.missing_model_files((ModelAssetTier.CORE_FEATURE,))

    def missing_optional_model_files(self) -> tuple[ModelAsset, ...]:
        return self.models.missing_model_files((ModelAssetTier.OPTIONAL_FEATURE,))


def build_optimization_presets() -> dict[str, OptimizationPreset]:
    """Return supported Raspberry Pi optimization presets."""

    return {
        "pi4_conservative": OptimizationPreset(
            name="pi4_conservative",
            description="Lowest heat and longest runtime; favors fewer inferences per second.",
            capture_width=960,
            capture_height=540,
            inference_width=416,
            inference_height=240,
            target_fps=15,
            max_frame_age_ms=220,
            worker_threads=2,
            opencv_threads=1,
            tflite_threads=1,
            enable_tracking=True,
            drop_stale_frames=True,
            thermal_soft_limit_c=68.0,
            thermal_hard_limit_c=76.0,
            low_power_enter_cpu_percent=82.0,
            low_power_exit_cpu_percent=58.0,
        ),
        "pi4_balanced": OptimizationPreset(
            name="pi4_balanced",
            description="Default field profile for Pi 4: responsive safety with controlled heat.",
            capture_width=1280,
            capture_height=720,
            inference_width=640,
            inference_height=360,
            target_fps=24,
            max_frame_age_ms=160,
            worker_threads=RASPBERRY_PI_4_CPU_CORES,
            opencv_threads=PI_DEFAULT_OPENCV_THREADS,
            tflite_threads=PI_DEFAULT_TFLITE_THREADS,
            enable_tracking=True,
            drop_stale_frames=True,
            thermal_soft_limit_c=PI_THERMAL_SOFT_LIMIT_C,
            thermal_hard_limit_c=PI_THERMAL_HARD_LIMIT_C,
            low_power_enter_cpu_percent=88.0,
            low_power_exit_cpu_percent=62.0,
        ),
        "pi4_responsive": OptimizationPreset(
            name="pi4_responsive",
            description="More frequent safety inference; requires good cooling and power.",
            capture_width=1280,
            capture_height=720,
            inference_width=640,
            inference_height=360,
            target_fps=30,
            max_frame_age_ms=120,
            worker_threads=RASPBERRY_PI_4_CPU_CORES,
            opencv_threads=2,
            tflite_threads=3,
            enable_tracking=True,
            drop_stale_frames=True,
            thermal_soft_limit_c=72.0,
            thermal_hard_limit_c=80.0,
            low_power_enter_cpu_percent=92.0,
            low_power_exit_cpu_percent=68.0,
        ),
    }


def _base_performance_profiles() -> dict[RuntimeMode, PerformanceProfile]:
    safety_modules = (
        ModuleName.CAMERA,
        ModuleName.OBJECT_DETECTION,
        ModuleName.OBSTACLE_ANALYSIS,
        ModuleName.DECISION_ENGINE,
        ModuleName.AUDIO,
        ModuleName.DIAGNOSTICS,
    )
    return {
        RuntimeMode.NORMAL: PerformanceProfile(
            mode=RuntimeMode.NORMAL,
            active_modules=safety_modules + (ModuleName.FACE_DETECTION,),
            camera_fps=24,
            object_detection_every_n_frames=3,
            obstacle_analysis_every_n_frames=1,
            face_detection_every_n_frames=6,
            face_recognition_every_n_stable_faces=None,
            emotion_detection_every_n_face_updates=None,
            ocr_policy=OcrPolicy.ON_DEMAND,
            cooldown_multiplier=1.0,
            lowest_allowed_audio_priority=AudioPriority.P4_CONTEXT,
            allow_audio_interruptions=True,
            cpu_budget_percent=78,
            stale_frame_ttl_ms=160,
            max_model_latency_ms={"object": 140, "face": 90, "recognition": 0, "emotion": 0, "ocr": 1500},
        ),
        RuntimeMode.DANGER: PerformanceProfile(
            mode=RuntimeMode.DANGER,
            active_modules=safety_modules,
            camera_fps=30,
            object_detection_every_n_frames=1,
            obstacle_analysis_every_n_frames=1,
            face_detection_every_n_frames=None,
            face_recognition_every_n_stable_faces=None,
            emotion_detection_every_n_face_updates=None,
            ocr_policy=OcrPolicy.DISABLED,
            cooldown_multiplier=0.45,
            lowest_allowed_audio_priority=AudioPriority.P1_DANGER,
            allow_audio_interruptions=True,
            cpu_budget_percent=90,
            stale_frame_ttl_ms=100,
            max_model_latency_ms={"object": 110, "face": 0, "recognition": 0, "emotion": 0, "ocr": 0},
        ),
        RuntimeMode.FACE: PerformanceProfile(
            mode=RuntimeMode.FACE,
            active_modules=safety_modules
            + (
                ModuleName.FACE_DETECTION,
                ModuleName.FACE_RECOGNITION,
                ModuleName.EMOTION_DETECTION,
            ),
            camera_fps=24,
            object_detection_every_n_frames=3,
            obstacle_analysis_every_n_frames=1,
            face_detection_every_n_frames=2,
            face_recognition_every_n_stable_faces=1,
            emotion_detection_every_n_face_updates=10,
            ocr_policy=OcrPolicy.DISABLED,
            cooldown_multiplier=1.3,
            lowest_allowed_audio_priority=AudioPriority.P4_CONTEXT,
            allow_audio_interruptions=True,
            cpu_budget_percent=84,
            stale_frame_ttl_ms=180,
            max_model_latency_ms={"object": 160, "face": 80, "recognition": 120, "emotion": 220, "ocr": 0},
        ),
        RuntimeMode.READING: PerformanceProfile(
            mode=RuntimeMode.READING,
            active_modules=safety_modules + (ModuleName.OCR,),
            camera_fps=18,
            object_detection_every_n_frames=3,
            obstacle_analysis_every_n_frames=1,
            face_detection_every_n_frames=None,
            face_recognition_every_n_stable_faces=None,
            emotion_detection_every_n_face_updates=None,
            ocr_policy=OcrPolicy.ACTIVE_REQUEST,
            cooldown_multiplier=1.2,
            lowest_allowed_audio_priority=AudioPriority.P2_IDENTITY_OCR,
            allow_audio_interruptions=True,
            cpu_budget_percent=86,
            stale_frame_ttl_ms=220,
            max_model_latency_ms={"object": 170, "face": 0, "recognition": 0, "emotion": 0, "ocr": 1800},
        ),
        RuntimeMode.QUIET: PerformanceProfile(
            mode=RuntimeMode.QUIET,
            active_modules=safety_modules,
            camera_fps=20,
            object_detection_every_n_frames=4,
            obstacle_analysis_every_n_frames=1,
            face_detection_every_n_frames=None,
            face_recognition_every_n_stable_faces=None,
            emotion_detection_every_n_face_updates=None,
            ocr_policy=OcrPolicy.ON_DEMAND_SILENT,
            cooldown_multiplier=2.0,
            lowest_allowed_audio_priority=AudioPriority.P1_DANGER,
            allow_audio_interruptions=True,
            cpu_budget_percent=70,
            stale_frame_ttl_ms=180,
            max_model_latency_ms={"object": 160, "face": 0, "recognition": 0, "emotion": 0, "ocr": 1800},
        ),
        RuntimeMode.LOW_POWER: PerformanceProfile(
            mode=RuntimeMode.LOW_POWER,
            active_modules=safety_modules,
            camera_fps=12,
            object_detection_every_n_frames=6,
            obstacle_analysis_every_n_frames=2,
            face_detection_every_n_frames=None,
            face_recognition_every_n_stable_faces=None,
            emotion_detection_every_n_face_updates=None,
            ocr_policy=OcrPolicy.DISABLED,
            cooldown_multiplier=2.5,
            lowest_allowed_audio_priority=AudioPriority.P1_DANGER,
            allow_audio_interruptions=True,
            cpu_budget_percent=55,
            stale_frame_ttl_ms=260,
            max_model_latency_ms={"object": 220, "face": 0, "recognition": 0, "emotion": 0, "ocr": 0},
        ),
    }


def build_performance_profiles(camera: CameraSettings) -> dict[RuntimeMode, PerformanceProfile]:
    """Return mode profiles capped by the selected camera and Pi preset."""

    profiles: dict[RuntimeMode, PerformanceProfile] = {}
    for mode, profile in _base_performance_profiles().items():
        profiles[mode] = replace(
            profile,
            camera_fps=min(profile.camera_fps, camera.target_fps),
            stale_frame_ttl_ms=min(profile.stale_frame_ttl_ms, camera.max_frame_age_ms),
        )
    return profiles


def build_threshold_settings(config: ConfigData) -> ThresholdSettings:
    """Build threshold settings from the local TOML configuration."""

    return ThresholdSettings(
        objects=ObjectThresholds(
            navigation_classes=_config_str_tuple(
                config, ("thresholds", "objects", "navigation_classes"), NAVIGATION_RELEVANT_CLASSES
            ),
            mobile_hazard_classes=_config_str_tuple(
                config, ("thresholds", "objects", "mobile_hazard_classes"), MOBILE_HAZARD_CLASSES
            ),
            min_detection_confidence=_config_float(
                config, ("thresholds", "objects", "min_detection_confidence"), 0.45
            ),
            min_danger_confidence=_config_float(config, ("thresholds", "objects", "min_danger_confidence"), 0.55),
            center_zone_x_min=_config_float(config, ("thresholds", "objects", "center_zone_x_min"), 0.34),
            center_zone_x_max=_config_float(config, ("thresholds", "objects", "center_zone_x_max"), 0.66),
            warning_bbox_area_ratio=_config_float(
                config, ("thresholds", "objects", "warning_bbox_area_ratio"), 0.10
            ),
            close_bbox_area_ratio=_config_float(config, ("thresholds", "objects", "close_bbox_area_ratio"), 0.18),
            critical_bbox_area_ratio=_config_float(
                config, ("thresholds", "objects", "critical_bbox_area_ratio"), 0.28
            ),
            mobile_hazard_close_area_ratio=_config_float(
                config, ("thresholds", "objects", "mobile_hazard_close_area_ratio"), 0.14
            ),
            person_close_area_ratio=_config_float(config, ("thresholds", "objects", "person_close_area_ratio"), 0.20),
            severity_warning_threshold=_config_float(
                config, ("thresholds", "objects", "severity_warning_threshold"), 0.45
            ),
            severity_danger_threshold=_config_float(
                config, ("thresholds", "objects", "severity_danger_threshold"), 0.70
            ),
            severity_critical_threshold=_config_float(
                config, ("thresholds", "objects", "severity_critical_threshold"), 0.88
            ),
            max_objects_per_frame=_config_int(config, ("thresholds", "objects", "max_objects_per_frame"), 8),
        ),
        faces=FaceThresholds(
            min_face_width_px=_config_int(config, ("thresholds", "faces", "min_face_width_px"), 80),
            min_face_height_px=_config_int(config, ("thresholds", "faces", "min_face_height_px"), 80),
            min_laplacian_blur_variance=_config_float(
                config, ("thresholds", "faces", "min_laplacian_blur_variance"), 80.0
            ),
            min_brightness=_config_float(config, ("thresholds", "faces", "min_brightness"), 45.0),
            max_brightness=_config_float(config, ("thresholds", "faces", "max_brightness"), 215.0),
            stable_frames_required=_config_int(config, ("thresholds", "faces", "stable_frames_required"), 3),
            max_face_yaw_degrees=_config_float(config, ("thresholds", "faces", "max_face_yaw_degrees"), 28.0),
            recognition_similarity_threshold=_config_float(
                config, ("thresholds", "faces", "recognition_similarity_threshold"), 0.62
            ),
            recognition_unknown_margin=_config_float(
                config, ("thresholds", "faces", "recognition_unknown_margin"), 0.05
            ),
            recognition_min_confidence=_config_float(
                config, ("thresholds", "faces", "recognition_min_confidence"), 0.60
            ),
            emotion_min_confidence=_config_float(config, ("thresholds", "faces", "emotion_min_confidence"), 0.55),
            enrollment_samples_required=_config_int(
                config, ("thresholds", "faces", "enrollment_samples_required"), 8
            ),
            enrollment_max_samples=_config_int(config, ("thresholds", "faces", "enrollment_max_samples"), 20),
        ),
        ocr=OcrThresholds(
            min_trigger_interval_seconds=_config_float(
                config, ("thresholds", "ocr", "min_trigger_interval_seconds"), 4.0
            ),
            min_text_confidence=_config_float(config, ("thresholds", "ocr", "min_text_confidence"), 55.0),
            min_text_characters=_config_int(config, ("thresholds", "ocr", "min_text_characters"), 3),
            repeat_similarity_threshold=_config_float(
                config, ("thresholds", "ocr", "repeat_similarity_threshold"), 0.88
            ),
            cache_ttl_seconds=_config_float(config, ("thresholds", "ocr", "cache_ttl_seconds"), 120.0),
            max_cache_entries=_config_int(config, ("thresholds", "ocr", "max_cache_entries"), 20),
            preprocessing_target_width_px=_config_int(
                config, ("thresholds", "ocr", "preprocessing_target_width_px"), 960
            ),
            adaptive_threshold_block_size=_config_int(
                config, ("thresholds", "ocr", "adaptive_threshold_block_size"), 31
            ),
            adaptive_threshold_c=_config_int(config, ("thresholds", "ocr", "adaptive_threshold_c"), 11),
        ),
    )


def build_cooldown_settings(config: ConfigData) -> CooldownSettings:
    """Build cooldown settings from the local TOML configuration."""

    return CooldownSettings(
        danger_repeat_seconds=_config_float(config, ("cooldowns", "danger_repeat_seconds"), 1.2),
        danger_clear_seconds=_config_float(config, ("cooldowns", "danger_clear_seconds"), 2.5),
        known_face_repeat_seconds=_config_float(config, ("cooldowns", "known_face_repeat_seconds"), 45.0),
        unknown_face_repeat_seconds=_config_float(config, ("cooldowns", "unknown_face_repeat_seconds"), 25.0),
        emotion_repeat_seconds=_config_float(config, ("cooldowns", "emotion_repeat_seconds"), 60.0),
        ocr_repeat_seconds=_config_float(config, ("cooldowns", "ocr_repeat_seconds"), 30.0),
        object_guidance_repeat_seconds=_config_float(
            config, ("cooldowns", "object_guidance_repeat_seconds"), 8.0
        ),
        low_priority_context_repeat_seconds=_config_float(
            config, ("cooldowns", "low_priority_context_repeat_seconds"), 20.0
        ),
        system_status_repeat_seconds=_config_float(config, ("cooldowns", "system_status_repeat_seconds"), 30.0),
        same_message_dedup_seconds=_config_float(config, ("cooldowns", "same_message_dedup_seconds"), 10.0),
        stale_p1_ttl_seconds=_config_float(config, ("cooldowns", "stale_p1_ttl_seconds"), 2.0),
        stale_p2_ttl_seconds=_config_float(config, ("cooldowns", "stale_p2_ttl_seconds"), 6.0),
        stale_p3_ttl_seconds=_config_float(config, ("cooldowns", "stale_p3_ttl_seconds"), 5.0),
        stale_p4_ttl_seconds=_config_float(config, ("cooldowns", "stale_p4_ttl_seconds"), 4.0),
    )


def load_settings() -> AppSettings:
    """Build application settings from defaults, local TOML, and environment overrides."""

    _load_env_file(PROJECT_ROOT / ".env")
    project_root = _env_path("ROOT", PROJECT_ROOT)
    if project_root != PROJECT_ROOT:
        _load_env_file(project_root / ".env")
    config_dir = _env_path("CONFIG_DIR", project_root / "config", base=project_root)
    runtime_config_file = _env_path("SETTINGS_FILE", config_dir / "runtime.toml", base=project_root)
    config = _load_toml_file(runtime_config_file)

    profile_name = _env_str("PROFILE", _config_str(config, ("runtime", "profile"), DEFAULT_OPTIMIZATION_PROFILE))
    presets = build_optimization_presets()
    if profile_name not in presets:
        supported = ", ".join(sorted(presets))
        raise ValueError(f"{_env_key('PROFILE')} must be one of: {supported}")

    optimization = presets[profile_name]
    model_dir = _env_path(
        "MODEL_DIR",
        _config_path(config, ("paths", "model_dir"), project_root / "models", base=project_root),
        base=project_root,
    )
    storage_dir = _env_path(
        "STORAGE_DIR",
        _config_path(config, ("paths", "storage_dir"), project_root / "storage", base=project_root),
        base=project_root,
    )
    paths = PathSettings(
        project_root=project_root,
        config_dir=config_dir,
        runtime_config_file=runtime_config_file,
        model_dir=model_dir,
        storage_dir=storage_dir,
        face_store_dir=_env_path(
            "FACE_STORE_DIR",
            _config_path(config, ("paths", "face_store_dir"), storage_dir / "faces", base=project_root),
            base=project_root,
        ),
        cache_dir=_env_path(
            "CACHE_DIR",
            _config_path(config, ("paths", "cache_dir"), storage_dir / "cache", base=project_root),
            base=project_root,
        ),
        log_dir=_env_path(
            "LOG_DIR",
            _config_path(config, ("paths", "log_dir"), storage_dir / "logs", base=project_root),
            base=project_root,
        ),
        settings_dir=_env_path(
            "SETTINGS_DIR",
            _config_path(config, ("paths", "settings_dir"), storage_dir / "settings", base=project_root),
            base=project_root,
        ),
    )

    requested_capture_width = _env_int(
        "CAMERA_WIDTH",
        _config_int(config, ("camera", "capture_width"), optimization.capture_width),
        320,
        1920,
    )
    requested_capture_height = _env_int(
        "CAMERA_HEIGHT",
        _config_int(config, ("camera", "capture_height"), optimization.capture_height),
        240,
        1080,
    )
    requested_inference_width = _env_int(
        "INFERENCE_WIDTH",
        _config_int(config, ("camera", "inference_width"), optimization.inference_width),
        224,
        960,
    )
    requested_inference_height = _env_int(
        "INFERENCE_HEIGHT",
        _config_int(config, ("camera", "inference_height"), optimization.inference_height),
        160,
        720,
    )
    requested_target_fps = _env_int(
        "TARGET_FPS",
        _config_int(config, ("camera", "target_fps"), optimization.target_fps),
        5,
        30,
    )
    requested_max_frame_age_ms = _env_int(
        "MAX_FRAME_AGE_MS",
        _config_int(config, ("camera", "max_frame_age_ms"), optimization.max_frame_age_ms),
        80,
        500,
    )
    camera = CameraSettings(
        capture_width=min(requested_capture_width, optimization.capture_width),
        capture_height=min(requested_capture_height, optimization.capture_height),
        inference_width=min(requested_inference_width, optimization.inference_width),
        inference_height=min(requested_inference_height, optimization.inference_height),
        target_fps=min(requested_target_fps, optimization.target_fps),
        max_frame_age_ms=min(requested_max_frame_age_ms, optimization.max_frame_age_ms),
        drop_stale_frames=_config_bool(config, ("camera", "drop_stale_frames"), optimization.drop_stale_frames),
        hflip=_env_bool("CAMERA_HFLIP", _config_bool(config, ("camera", "hflip"), False)),
        vflip=_env_bool("CAMERA_VFLIP", _config_bool(config, ("camera", "vflip"), False)),
    )
    if camera.inference_width > camera.capture_width or camera.inference_height > camera.capture_height:
        camera = replace(
            camera,
            inference_width=min(camera.inference_width, camera.capture_width),
            inference_height=min(camera.inference_height, camera.capture_height),
        )

    default_quiet_priority = _config_audio_priority(
        config, ("audio", "lowest_priority_in_quiet"), AudioPriority.P1_DANGER
    )
    audio = AudioSettings(
        backend=_env_str("TTS_BACKEND", _config_str(config, ("audio", "backend"), "espeak-ng")),
        voice=_env_str("TTS_VOICE", _config_str(config, ("audio", "voice"), "en-us")),
        speech_rate_wpm=_env_int(
            "TTS_RATE_WPM", _config_int(config, ("audio", "speech_rate_wpm"), 165), 120, 230
        ),
        volume=_env_float("TTS_VOLUME", _config_float(config, ("audio", "volume"), 0.95), 0.0, 1.0),
        p1_interrupts_all=_config_bool(config, ("audio", "p1_interrupts_all"), True),
        p2_replaces_p3_p4=_config_bool(config, ("audio", "p2_replaces_p3_p4"), True),
        lowest_priority_in_quiet=_env_audio_priority("QUIET_LOWEST_AUDIO_PRIORITY", default_quiet_priority),
    )

    diagnostics = DiagnosticsSettings(
        enabled=_env_bool("DIAGNOSTICS_ENABLED", _config_bool(config, ("diagnostics", "enabled"), True)),
        metrics_interval_seconds=_env_float(
            "METRICS_INTERVAL_SECONDS",
            _config_float(config, ("diagnostics", "metrics_interval_seconds"), 5.0),
            1.0,
            60.0,
        ),
        slow_frame_ms=_config_int(config, ("diagnostics", "slow_frame_ms"), 220),
        slow_inference_ms=_config_int(config, ("diagnostics", "slow_inference_ms"), 180),
        log_rotation_mb=_config_int(config, ("diagnostics", "log_rotation_mb"), 10),
        retained_log_files=_config_int(config, ("diagnostics", "retained_log_files"), 7),
    )

    return AppSettings(
        environment=_env_str("ENVIRONMENT", _config_str(config, ("runtime", "environment"), DEFAULT_ENVIRONMENT)),
        language=_env_str("LANGUAGE", _config_str(config, ("runtime", "language"), DEFAULT_LANGUAGE)),
        timezone=_env_str("TIMEZONE", _config_str(config, ("runtime", "timezone"), DEFAULT_TIMEZONE)),
        paths=paths,
        camera=camera,
        models=ModelPathSettings.from_model_dir(model_dir),
        thresholds=build_threshold_settings(config),
        cooldowns=build_cooldown_settings(config),
        audio=audio,
        diagnostics=diagnostics,
        optimization=optimization,
        performance_profiles=build_performance_profiles(camera),
    )


def ensure_runtime_directories(settings: AppSettings) -> None:
    """Create local runtime directories that are safe to generate on-device."""

    for path in (
        settings.paths.storage_dir,
        settings.paths.face_store_dir,
        settings.paths.cache_dir,
        settings.paths.log_dir,
        settings.paths.settings_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _validate_unit_interval(name: str, value: float, issues: list[ValidationIssue]) -> None:
    if not 0.0 <= value <= 1.0:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, f"{name} must be between 0.0 and 1.0."))


def _validate_positive(name: str, value: float | int, issues: list[ValidationIssue]) -> None:
    if value <= 0:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, f"{name} must be greater than zero."))


def _validate_interval(name: str, value: FrameInterval, issues: list[ValidationIssue]) -> None:
    if value is not None and value <= 0:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, f"{name} must be positive or None."))


def validate_model_manifest(settings: AppSettings) -> tuple[ValidationIssue, ...]:
    """Validate installed model provenance and hashes."""

    issues: list[ValidationIssue] = []
    installed_assets = settings.models.installed_model_files()
    if not installed_assets:
        issues.append(
            ValidationIssue(
                ValidationSeverity.WARNING,
                "No model assets are installed yet; safety runtime cannot start until safety-critical assets exist.",
            )
        )
        return tuple(issues)

    if not settings.models.manifest_path.exists():
        issues.append(
            ValidationIssue(
                ValidationSeverity.ERROR,
                f"Installed model assets require a local provenance manifest: {settings.models.manifest_path}",
            )
        )
        return tuple(issues)

    manifest = ModelManifest.load(settings.models.manifest_path)
    for asset in installed_assets:
        entry = manifest.entries.get(asset.asset_id)
        if entry is None:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Model manifest is missing entry for installed asset {asset.asset_id}.",
                )
            )
            continue
        for field_name in ("version", "license", "sha256", "input_schema", "output_schema"):
            if not getattr(entry, field_name):
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.ERROR,
                        f"Model manifest entry {asset.asset_id}.{field_name} must be set.",
                    )
                )
        if entry.sha256 and (
            len(entry.sha256) != 64 or any(character not in "0123456789abcdef" for character in entry.sha256)
        ):
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Model manifest entry {asset.asset_id}.sha256 must be a 64-character lowercase hex digest.",
                )
            )
        elif entry.sha256 and _sha256(asset.path) != entry.sha256:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Model asset hash mismatch for {asset.asset_id}: {asset.path}",
                )
            )
    return tuple(issues)


def validate_audio_backend(settings: AppSettings) -> tuple[ValidationIssue, ...]:
    """Validate that an offline TTS path is available before field runtime."""

    issues: list[ValidationIssue] = []
    backend = settings.audio.backend.strip().casefold()
    espeak_available = shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None
    pyttsx3_available = importlib.util.find_spec("pyttsx3") is not None

    if backend in {"espeak-ng", "espeak"}:
        if not espeak_available and pyttsx3_available:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "espeak-ng/espeak is unavailable; runtime will use interruptible pyttsx3 subprocess fallback.",
                )
            )
        elif not espeak_available and not pyttsx3_available:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "No offline TTS backend is available; install espeak-ng or pyttsx3.",
                )
            )
    elif backend == "pyttsx3":
        if not pyttsx3_available:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "Configured TTS backend pyttsx3 is not installed.",
                )
            )
    else:
        issues.append(
            ValidationIssue(
                ValidationSeverity.ERROR,
                f"Unsupported TTS backend {settings.audio.backend!r}; expected espeak-ng, espeak, or pyttsx3.",
            )
        )

    return tuple(issues)


def validate_settings(settings: AppSettings) -> tuple[ValidationIssue, ...]:
    """Return configuration problems without mutating runtime state."""

    issues: list[ValidationIssue] = []
    if settings.camera.inference_width > settings.camera.capture_width:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Inference width must not exceed capture width."))
    if settings.camera.inference_height > settings.camera.capture_height:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Inference height must not exceed capture height."))
    if settings.camera.target_fps > settings.optimization.target_fps:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Camera FPS must not exceed selected Pi preset FPS."))
    if settings.optimization.thermal_soft_limit_c >= settings.optimization.thermal_hard_limit_c:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Thermal soft limit must be below hard limit."))
    if settings.optimization.low_power_exit_cpu_percent >= settings.optimization.low_power_enter_cpu_percent:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Low-power CPU exit threshold must be below enter threshold."))

    objects = settings.thresholds.objects
    for name, value in (
        ("objects.min_detection_confidence", objects.min_detection_confidence),
        ("objects.min_danger_confidence", objects.min_danger_confidence),
        ("objects.center_zone_x_min", objects.center_zone_x_min),
        ("objects.center_zone_x_max", objects.center_zone_x_max),
        ("objects.warning_bbox_area_ratio", objects.warning_bbox_area_ratio),
        ("objects.close_bbox_area_ratio", objects.close_bbox_area_ratio),
        ("objects.critical_bbox_area_ratio", objects.critical_bbox_area_ratio),
        ("objects.mobile_hazard_close_area_ratio", objects.mobile_hazard_close_area_ratio),
        ("objects.person_close_area_ratio", objects.person_close_area_ratio),
        ("objects.severity_warning_threshold", objects.severity_warning_threshold),
        ("objects.severity_danger_threshold", objects.severity_danger_threshold),
        ("objects.severity_critical_threshold", objects.severity_critical_threshold),
    ):
        _validate_unit_interval(name, value, issues)
    if objects.center_zone_x_min >= objects.center_zone_x_max:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Object center danger zone min must be lower than max."))
    if not objects.warning_bbox_area_ratio < objects.close_bbox_area_ratio < objects.critical_bbox_area_ratio:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Object bbox area thresholds must increase by severity."))
    if not objects.severity_warning_threshold < objects.severity_danger_threshold < objects.severity_critical_threshold:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Object severity thresholds must increase by severity."))
    _validate_positive("objects.max_objects_per_frame", objects.max_objects_per_frame, issues)

    faces = settings.thresholds.faces
    for name, value in (
        ("faces.recognition_similarity_threshold", faces.recognition_similarity_threshold),
        ("faces.recognition_unknown_margin", faces.recognition_unknown_margin),
        ("faces.recognition_min_confidence", faces.recognition_min_confidence),
        ("faces.emotion_min_confidence", faces.emotion_min_confidence),
    ):
        _validate_unit_interval(name, value, issues)
    for name, value in (
        ("faces.min_face_width_px", faces.min_face_width_px),
        ("faces.min_face_height_px", faces.min_face_height_px),
        ("faces.min_laplacian_blur_variance", faces.min_laplacian_blur_variance),
        ("faces.stable_frames_required", faces.stable_frames_required),
        ("faces.max_face_yaw_degrees", faces.max_face_yaw_degrees),
        ("faces.enrollment_samples_required", faces.enrollment_samples_required),
        ("faces.enrollment_max_samples", faces.enrollment_max_samples),
    ):
        _validate_positive(name, value, issues)
    if not 0.0 <= faces.min_brightness < faces.max_brightness <= 255.0:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Face brightness thresholds must satisfy 0 <= min < max <= 255."))
    if faces.enrollment_samples_required > faces.enrollment_max_samples:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Face enrollment required samples must not exceed max samples."))

    ocr = settings.thresholds.ocr
    _validate_unit_interval("ocr.repeat_similarity_threshold", ocr.repeat_similarity_threshold, issues)
    for name, value in (
        ("ocr.min_trigger_interval_seconds", ocr.min_trigger_interval_seconds),
        ("ocr.min_text_confidence", ocr.min_text_confidence),
        ("ocr.min_text_characters", ocr.min_text_characters),
        ("ocr.cache_ttl_seconds", ocr.cache_ttl_seconds),
        ("ocr.max_cache_entries", ocr.max_cache_entries),
        ("ocr.preprocessing_target_width_px", ocr.preprocessing_target_width_px),
        ("ocr.adaptive_threshold_block_size", ocr.adaptive_threshold_block_size),
    ):
        _validate_positive(name, value, issues)
    if ocr.adaptive_threshold_block_size % 2 == 0:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "OCR adaptive threshold block size must be odd."))

    for field_name, value in settings.cooldowns.__dict__.items():
        _validate_positive(f"cooldowns.{field_name}", value, issues)
    if not 0.0 <= settings.audio.volume <= 1.0:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Audio volume must be between 0.0 and 1.0."))
    _validate_positive("audio.queue_size", settings.audio.queue_size, issues)
    if not settings.audio.p1_interrupts_all:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Audio P1 danger alerts must interrupt all speech."))
    if not settings.audio.p2_replaces_p3_p4:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Audio P2 identity/OCR alerts must replace P3/P4 speech."))
    _validate_positive("diagnostics.metrics_interval_seconds", settings.diagnostics.metrics_interval_seconds, issues)
    _validate_positive("diagnostics.log_rotation_mb", settings.diagnostics.log_rotation_mb, issues)
    _validate_positive("diagnostics.retained_log_files", settings.diagnostics.retained_log_files, issues)

    if set(settings.performance_profiles) != set(RuntimeMode):
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Every RuntimeMode must have a performance profile."))
    for mode, profile in settings.performance_profiles.items():
        if profile.mode != mode:
            issues.append(ValidationIssue(ValidationSeverity.ERROR, f"Profile key mismatch for mode {mode.value}."))
        if profile.camera_fps > settings.camera.target_fps:
            issues.append(ValidationIssue(ValidationSeverity.ERROR, f"{mode.value} FPS exceeds camera target FPS."))
        if profile.stale_frame_ttl_ms > settings.camera.max_frame_age_ms:
            issues.append(ValidationIssue(ValidationSeverity.ERROR, f"{mode.value} stale frame TTL exceeds camera max age."))
        for field_name in (
            "object_detection_every_n_frames",
            "obstacle_analysis_every_n_frames",
            "face_detection_every_n_frames",
            "face_recognition_every_n_stable_faces",
            "emotion_detection_every_n_face_updates",
        ):
            _validate_interval(f"{mode.value}.{field_name}", getattr(profile, field_name), issues)
        if ModuleName.OBJECT_DETECTION in profile.active_modules and profile.object_detection_every_n_frames is None:
            issues.append(ValidationIssue(ValidationSeverity.ERROR, f"{mode.value} enables object detection without a schedule."))
        if ModuleName.OBSTACLE_ANALYSIS in profile.active_modules and profile.obstacle_analysis_every_n_frames is None:
            issues.append(ValidationIssue(ValidationSeverity.ERROR, f"{mode.value} enables obstacle analysis without a schedule."))
        if ModuleName.FACE_RECOGNITION not in profile.active_modules and profile.face_recognition_every_n_stable_faces is not None:
            issues.append(ValidationIssue(ValidationSeverity.ERROR, f"{mode.value} schedules face recognition but does not enable it."))
        if ModuleName.EMOTION_DETECTION not in profile.active_modules and profile.emotion_detection_every_n_face_updates is not None:
            issues.append(ValidationIssue(ValidationSeverity.ERROR, f"{mode.value} schedules emotion detection but does not enable it."))
        if ModuleName.OCR in profile.active_modules and profile.ocr_policy == OcrPolicy.DISABLED:
            issues.append(ValidationIssue(ValidationSeverity.ERROR, f"{mode.value} enables OCR with disabled OCR policy."))

    for asset in settings.missing_safety_model_files():
        issues.append(
            ValidationIssue(
                ValidationSeverity.ERROR,
                f"Missing safety-critical local model asset: {asset.path}",
            )
        )
    for asset in settings.missing_feature_model_files():
        issues.append(
            ValidationIssue(
                ValidationSeverity.WARNING,
                f"Missing feature model asset; related mode will degrade gracefully: {asset.path}",
            )
        )
    for asset in settings.missing_optional_model_files():
        issues.append(
            ValidationIssue(
                ValidationSeverity.WARNING,
                f"Missing optional model asset: {asset.path}",
            )
        )
    for asset in settings.models.assets():
        if asset.backend in {InferenceBackend.STATIC_FILE, InferenceBackend.TESSERACT}:
            continue
        if asset.input_shape is None or len(asset.input_shape) != 4:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Model asset {asset.asset_id} must define a 4D input shape.",
                )
            )
            continue
        if any(value <= 0 for value in asset.input_shape):
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Model asset {asset.asset_id} input shape dimensions must be positive.",
                )
            )
    issues.extend(validate_model_manifest(settings))
    issues.extend(validate_audio_backend(settings))
    return tuple(issues)


def validation_errors(settings: AppSettings) -> tuple[str, ...]:
    """Return only error-level validation messages."""

    return tuple(issue.message for issue in validate_settings(settings) if issue.severity == ValidationSeverity.ERROR)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings for process-wide runtime use."""

    return load_settings()
