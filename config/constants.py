"""Stable constants for the Raspberry Pi smart glasses runtime."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
STORAGE_DIR = PROJECT_ROOT / "storage"
DOCS_DIR = PROJECT_ROOT / "docs"

ENV_PREFIX = "SMART_GLASSES"
DEFAULT_ENVIRONMENT = "production"
DEFAULT_OPTIMIZATION_PROFILE = "pi4_balanced"
DEFAULT_LANGUAGE = "en"
DEFAULT_TIMEZONE = "Asia/Colombo"


class RuntimeMode(StrEnum):
    """Supported operating modes for the user-facing runtime."""

    NORMAL = "NORMAL"
    DANGER = "DANGER"
    FACE = "FACE"
    READING = "READING"
    QUIET = "QUIET"
    LOW_POWER = "LOW_POWER"


class AudioPriority(IntEnum):
    """Audio priority values. Lower numeric value means higher priority."""

    P1_DANGER = 1
    P2_IDENTITY_OCR = 2
    P3_NAVIGATION = 3
    P4_CONTEXT = 4


class InferenceBackend(StrEnum):
    """Supported local inference backends."""

    ONNX_RUNTIME = "onnxruntime"
    TFLITE_RUNTIME = "tflite-runtime"
    OPENCV_DNN = "opencv-dnn"
    TESSERACT = "tesseract"
    STATIC_FILE = "static-file"


class ModelAssetTier(StrEnum):
    """Model criticality tiers for graceful startup and degradation."""

    SAFETY_CRITICAL = "safety_critical"
    CORE_FEATURE = "core_feature"
    OPTIONAL_FEATURE = "optional_feature"


class OcrPolicy(StrEnum):
    """Mode-specific OCR activation policies."""

    DISABLED = "disabled"
    ON_DEMAND = "on_demand"
    ACTIVE_REQUEST = "active_request"
    ON_DEMAND_SILENT = "on_demand_silent"


class ValidationSeverity(StrEnum):
    """Configuration validation severity."""

    ERROR = "error"
    WARNING = "warning"


class SpeechCategory(StrEnum):
    """Speech categories used for cooldown and deduplication policy."""

    DANGER = "danger"
    DANGER_CLEAR = "danger_clear"
    KNOWN_FACE = "known_face"
    UNKNOWN_FACE = "unknown_face"
    OCR_TEXT = "ocr_text"
    OBJECT_GUIDANCE = "object_guidance"
    EMOTION = "emotion"
    LOW_PRIORITY_CONTEXT = "low_priority_context"
    SYSTEM_STATUS = "system_status"


class ModuleName(StrEnum):
    """Canonical module names used by performance profiles."""

    CAMERA = "camera"
    OBJECT_DETECTION = "object_detection"
    OBSTACLE_ANALYSIS = "obstacle_analysis"
    FACE_DETECTION = "face_detection"
    FACE_RECOGNITION = "face_recognition"
    EMOTION_DETECTION = "emotion_detection"
    OCR = "ocr"
    DECISION_ENGINE = "decision_engine"
    AUDIO = "audio"
    DIAGNOSTICS = "diagnostics"


NAVIGATION_RELEVANT_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "chair",
    "table",
    "dog",
    "cat",
    "traffic light",
    "stop sign",
)

MOBILE_HAZARD_CLASSES: tuple[str, ...] = (
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
)

STATIC_OBSTACLE_CLASSES: tuple[str, ...] = (
    "chair",
    "table",
    "stop sign",
    "traffic light",
)

LIVING_OBSTACLE_CLASSES: tuple[str, ...] = (
    "person",
    "dog",
    "cat",
)

OBJECT_CLASS_ALIASES: dict[str, str] = {
    "traffic_light": "traffic light",
    "stop_sign": "stop sign",
    "motorbike": "motorcycle",
    "sofa": "chair",
    "dining table": "table",
}

DEFAULT_MODEL_FILENAMES: dict[str, str] = {
    "object_detection": "yolov8n.onnx",
    "object_labels": "coco_labels.txt",
    "face_detection": "face_detection_yunet_2023mar.onnx",
    "face_recognition": "face_recognition_sface_2021dec.onnx",
    "emotion_detection": "emotion-ferplus-8.onnx",
    "ocr_eng": "eng.traineddata",
    "ocr_sin": "sin.traineddata",
}

MODEL_DIRECTORY_NAMES: dict[str, str] = {
    "object_detection": "object_detection",
    "face_detection": "face_detection",
    "face_recognition": "face_recognition",
    "emotion_detection": "emotion",
    "ocr": "ocr",
}

SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SUPPORTED_AUDIO_EXTENSIONS: tuple[str, ...] = (".wav", ".mp3")

RASPBERRY_PI_4_CPU_CORES = 4
RASPBERRY_PI_4_MIN_RAM_MB = 2048
PI_THERMAL_SOFT_LIMIT_C = 72.0
PI_THERMAL_HARD_LIMIT_C = 80.0
PI_DEFAULT_OPENCV_THREADS = 2
PI_DEFAULT_TFLITE_THREADS = 2

MAX_AUDIO_QUEUE_SIZE = 12
MAX_FRAME_QUEUE_SIZE = 2
MAX_DECISION_QUEUE_SIZE = 16
MAX_DIAGNOSTIC_EVENT_QUEUE_SIZE = 256
MAX_CRITICAL_EVENT_QUEUE_SIZE = 64

DEFAULT_MODEL_INPUT_SHAPES: dict[str, tuple[int, ...]] = {
    "object_detection": (1, 3, 640, 640),
    "face_detection": (1, 3, 320, 320),
    "face_recognition": (1, 3, 112, 112),
    "emotion_detection": (1, 1, 64, 64),
}
