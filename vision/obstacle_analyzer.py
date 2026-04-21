"""Obstacle warning and danger scoring logic."""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import ObjectThresholds


@dataclass(frozen=True)
class DetectionBox:
    """Normalized object box."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area_ratio(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)


@dataclass(frozen=True)
class ObjectDetectionResult:
    """Object detection with normalized geometry."""

    label: str
    confidence: float
    box: DetectionBox


@dataclass(frozen=True)
class ObstacleWarning:
    """Scored obstacle warning for downstream decision logic."""

    label: str
    confidence: float
    box: DetectionBox
    zone: str
    severity: float
    area_ratio: float
    is_danger: bool
    is_critical: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "box": list(self.box.as_tuple()),
            "zone": self.zone,
            "severity": self.severity,
            "area_ratio": self.area_ratio,
            "is_danger": self.is_danger,
            "is_critical": self.is_critical,
        }


class ObstacleAnalyzer:
    """Scores detections using Pi-friendly geometry heuristics."""

    def __init__(self, thresholds: ObjectThresholds) -> None:
        self._thresholds = thresholds
        self._mobile_hazards = set(thresholds.mobile_hazard_classes)

    def analyze(self, detections: tuple[ObjectDetectionResult, ...]) -> tuple[ObstacleWarning, ...]:
        warnings = [self.score(detection) for detection in detections]
        warnings.sort(key=lambda item: item.severity, reverse=True)
        return tuple(warnings)

    def score(self, detection: ObjectDetectionResult) -> ObstacleWarning:
        area = detection.box.area_ratio
        severity = detection.confidence * 0.45
        if area >= self._thresholds.warning_bbox_area_ratio:
            severity += 0.15
        if area >= self._thresholds.close_bbox_area_ratio:
            severity += 0.18
        if area >= self._thresholds.critical_bbox_area_ratio:
            severity += 0.22
        if detection.label in self._mobile_hazards and area >= self._thresholds.mobile_hazard_close_area_ratio:
            severity += 0.18
        if detection.label == "person" and area >= self._thresholds.person_close_area_ratio:
            severity += 0.10
        zone = self.zone_for(detection.box)
        if zone == "center":
            severity += 0.18
        severity = min(1.0, severity)
        return ObstacleWarning(
            label=detection.label,
            confidence=detection.confidence,
            box=detection.box,
            zone=zone,
            severity=severity,
            area_ratio=area,
            is_danger=severity >= self._thresholds.severity_danger_threshold,
            is_critical=severity >= self._thresholds.severity_critical_threshold,
        )

    def zone_for(self, box: DetectionBox) -> str:
        if box.center_x < self._thresholds.center_zone_x_min:
            return "left"
        if box.center_x > self._thresholds.center_zone_x_max:
            return "right"
        return "center"

