"""Offline model runtime adapters for TFLite, ONNX Runtime, and OpenCV DNN."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np

from config.constants import InferenceBackend
from core.exceptions import ConfigurationError, RecoverableRuntimeError


@dataclass(frozen=True)
class ModelInputInfo:
    """Minimal model input metadata used by preprocessing code."""

    name: str
    shape: tuple[int, ...]
    dtype: str


class ModelRunner(Protocol):
    """Common inference interface for all local backends."""

    @property
    def backend(self) -> InferenceBackend:
        ...

    @property
    def input_info(self) -> ModelInputInfo:
        ...

    def infer(self, input_tensor: np.ndarray | Mapping[str, np.ndarray]) -> list[np.ndarray]:
        ...


class TFLiteModelRunner:
    """TFLite interpreter wrapper using tflite-runtime when available."""

    def __init__(self, model_path: Path, num_threads: int = 2) -> None:
        if not model_path.exists():
            raise ConfigurationError(f"TFLite model file does not exist: {model_path}")
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore[import-not-found]
        except Exception:
            try:
                from tensorflow.lite.python.interpreter import Interpreter  # type: ignore[import-not-found]
            except Exception as exc:
                raise ConfigurationError("No TFLite interpreter is installed.") from exc

        self._interpreter = Interpreter(model_path=str(model_path), num_threads=max(1, num_threads))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        first_input = self._input_details[0]
        self._input_info = ModelInputInfo(
            name=str(first_input["name"]),
            shape=tuple(int(value) for value in first_input["shape"]),
            dtype=np.dtype(first_input["dtype"]).name,
        )

    @property
    def backend(self) -> InferenceBackend:
        return InferenceBackend.TFLITE_RUNTIME

    @property
    def input_info(self) -> ModelInputInfo:
        return self._input_info

    def infer(self, input_tensor: np.ndarray | Mapping[str, np.ndarray]) -> list[np.ndarray]:
        if isinstance(input_tensor, Mapping):
            tensor = input_tensor[self._input_info.name]
        else:
            tensor = input_tensor
        input_detail = self._input_details[0]
        self._interpreter.set_tensor(input_detail["index"], tensor.astype(input_detail["dtype"], copy=False))
        self._interpreter.invoke()
        return [self._interpreter.get_tensor(output["index"]) for output in self._output_details]


class OnnxRuntimeModelRunner:
    """ONNX Runtime wrapper. Optional on Pi; useful where supported."""

    def __init__(self, model_path: Path, input_shape: tuple[int, ...] | None = None) -> None:
        if not model_path.exists():
            raise ConfigurationError(f"ONNX model file does not exist: {model_path}")
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except Exception as exc:
            raise ConfigurationError("onnxruntime is not installed.") from exc

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        first_input = self._session.get_inputs()[0]
        shape = _onnx_input_shape(first_input.shape, input_shape)
        self._input_info = ModelInputInfo(name=first_input.name, shape=shape, dtype=str(first_input.type))

    @property
    def backend(self) -> InferenceBackend:
        return InferenceBackend.ONNX_RUNTIME

    @property
    def input_info(self) -> ModelInputInfo:
        return self._input_info

    def infer(self, input_tensor: np.ndarray | Mapping[str, np.ndarray]) -> list[np.ndarray]:
        feeds = dict(input_tensor) if isinstance(input_tensor, Mapping) else {self._input_info.name: input_tensor}
        return [np.asarray(output) for output in self._session.run(None, feeds)]


class OpenCVDnnModelRunner:
    """OpenCV DNN runner for ONNX and Caffe-style models."""

    def __init__(self, model_path: Path, input_shape: tuple[int, ...] | None = None) -> None:
        if not model_path.exists():
            raise ConfigurationError(f"OpenCV DNN model file does not exist: {model_path}")
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception as exc:
            raise ConfigurationError("OpenCV is not installed.") from exc

        self._cv2 = cv2
        self._net = cv2.dnn.readNet(str(model_path))
        normalized_shape = _normalize_input_shape(input_shape)
        if hasattr(self._net, "setInputShape"):
            try:
                self._net.setInputShape("input", normalized_shape)
            except Exception:
                self._logger_debug_shape(model_path, normalized_shape)
        self._input_info = ModelInputInfo(
            name="input",
            shape=normalized_shape,
            dtype="float32",
        )

    @staticmethod
    def _logger_debug_shape(model_path: Path, shape: tuple[int, int, int, int]) -> None:
        logging.getLogger(__name__).debug(
            "OpenCV DNN did not accept explicit input shape for %s: %s",
            model_path,
            shape,
        )

    @property
    def backend(self) -> InferenceBackend:
        return InferenceBackend.OPENCV_DNN

    @property
    def input_info(self) -> ModelInputInfo:
        return self._input_info

    def infer(self, input_tensor: np.ndarray | Mapping[str, np.ndarray]) -> list[np.ndarray]:
        tensor = next(iter(input_tensor.values())) if isinstance(input_tensor, Mapping) else input_tensor
        self._net.setInput(tensor)
        output_names = self._net.getUnconnectedOutLayersNames()
        output = self._net.forward(output_names)
        if isinstance(output, tuple):
            return [np.asarray(item) for item in output]
        if isinstance(output, list):
            return [np.asarray(item) for item in output]
        return [np.asarray(output)]


class FallbackModelRunner:
    """Runner that tries a primary backend, then a fallback backend."""

    def __init__(
        self,
        primary: ModelRunner,
        fallback_factory: Callable[[], ModelRunner],
        logger: logging.Logger | None = None,
    ) -> None:
        self._primary = primary
        self._fallback_factory = fallback_factory
        self._fallback: ModelRunner | None = None
        self._logger = logger or logging.getLogger(__name__)

    @property
    def backend(self) -> InferenceBackend:
        return self._fallback.backend if self._fallback is not None else self._primary.backend

    @property
    def input_info(self) -> ModelInputInfo:
        return self._fallback.input_info if self._fallback is not None else self._primary.input_info

    def infer(self, input_tensor: np.ndarray | Mapping[str, np.ndarray]) -> list[np.ndarray]:
        if self._fallback is not None:
            return self._fallback.infer(input_tensor)
        try:
            return self._primary.infer(input_tensor)
        except Exception as exc:
            self._logger.warning("Primary inference backend failed; trying fallback: %s", exc)
            try:
                self._fallback = self._fallback_factory()
            except Exception as fallback_exc:
                raise ConfigurationError(
                    f"Primary inference backend failed and fallback backend is unavailable: {fallback_exc}"
                ) from fallback_exc
            return self._fallback.infer(input_tensor)


class LocalModelLoader:
    """Factory for offline model runners with backend fallbacks."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def load(
        self,
        model_path: Path,
        preferred_backend: InferenceBackend | None = None,
        num_threads: int = 2,
        input_shape: tuple[int, ...] | None = None,
    ) -> ModelRunner:
        backend = preferred_backend or self._infer_backend(model_path)
        if backend == InferenceBackend.TFLITE_RUNTIME:
            return TFLiteModelRunner(model_path, num_threads=num_threads)
        if backend == InferenceBackend.OPENCV_DNN:
            try:
                primary = OpenCVDnnModelRunner(model_path, input_shape=input_shape)
            except ConfigurationError as exc:
                if model_path.suffix.lower() == ".onnx":
                    self._logger.warning("OpenCV DNN unavailable for %s: %s", model_path, exc)
                    return OnnxRuntimeModelRunner(model_path, input_shape=input_shape)
                raise
            if model_path.suffix.lower() == ".onnx":
                return FallbackModelRunner(
                    primary,
                    lambda: OnnxRuntimeModelRunner(model_path, input_shape=input_shape),
                    self._logger,
                )
            return primary
        if backend == InferenceBackend.ONNX_RUNTIME:
            return OnnxRuntimeModelRunner(model_path, input_shape=input_shape)
        if backend == InferenceBackend.TESSERACT:
            raise RecoverableRuntimeError("Tesseract OCR does not use ModelRunner.")
        raise ConfigurationError(f"Unsupported inference backend for {model_path}: {backend}")

    def probe(
        self,
        model_path: Path,
        preferred_backend: InferenceBackend | None = None,
        num_threads: int = 2,
        input_shape: tuple[int, ...] | None = None,
        run_inference: bool = True,
    ) -> ModelRunner:
        """Load a model and optionally execute a zero-input forward pass."""

        runner = self.load(
            model_path,
            preferred_backend=preferred_backend,
            num_threads=num_threads,
            input_shape=input_shape,
        )
        if run_inference:
            try:
                runner.infer(_dummy_input(runner.input_info))
            except ConfigurationError:
                raise
            except Exception as exc:
                raise ConfigurationError(f"Model backend probe failed for {model_path}: {exc}") from exc
        return runner

    @staticmethod
    def _infer_backend(model_path: Path) -> InferenceBackend:
        suffix = model_path.suffix.lower()
        if suffix == ".tflite":
            return InferenceBackend.TFLITE_RUNTIME
        if suffix in {".onnx", ".caffemodel"}:
            return InferenceBackend.OPENCV_DNN
        raise ConfigurationError(f"Cannot infer model backend from file extension: {model_path}")


def nhwc_to_nchw(image: np.ndarray) -> np.ndarray:
    return np.transpose(image, (2, 0, 1))[None, ...]


def _normalize_input_shape(input_shape: tuple[int, ...] | None) -> tuple[int, int, int, int]:
    if input_shape is None:
        raise ConfigurationError("OpenCV DNN models require an explicit input shape.")
    if len(input_shape) != 4:
        raise ConfigurationError(f"Model input shape must have 4 dimensions, got {input_shape}.")
    normalized = tuple(max(1, int(value)) for value in input_shape)
    return (normalized[0], normalized[1], normalized[2], normalized[3])


def _onnx_input_shape(raw_shape: object, configured_shape: tuple[int, ...] | None) -> tuple[int, ...]:
    if not isinstance(raw_shape, (list, tuple)):
        if configured_shape is not None:
            return tuple(max(1, int(value)) for value in configured_shape)
        return (1,)
    resolved: list[int] = []
    for index, value in enumerate(raw_shape):
        if isinstance(value, int) and value > 0:
            resolved.append(int(value))
        elif configured_shape is not None and index < len(configured_shape):
            resolved.append(max(1, int(configured_shape[index])))
        else:
            resolved.append(1)
    return tuple(resolved)


def _dtype_from_input_info(input_info: ModelInputInfo) -> np.dtype:
    dtype = input_info.dtype.lower()
    if "uint8" in dtype:
        return np.dtype(np.uint8)
    if "int8" in dtype:
        return np.dtype(np.int8)
    if "int32" in dtype:
        return np.dtype(np.int32)
    if "float16" in dtype:
        return np.dtype(np.float16)
    return np.dtype(np.float32)


def _dummy_input(input_info: ModelInputInfo) -> np.ndarray:
    shape = tuple(max(1, int(value)) for value in input_info.shape)
    dtype = _dtype_from_input_info(input_info)
    return np.zeros(shape, dtype=dtype)
