"""Multilingual-ready speech formatting."""

from __future__ import annotations

from typing import Any, Mapping

from config.constants import AudioPriority, SpeechCategory
from config.settings import AppSettings
from audio.types import SpeechMessage


TemplateMap = Mapping[str, Mapping[SpeechCategory, str]]


DEFAULT_TEMPLATES: TemplateMap = {
    "en": {
        SpeechCategory.DANGER: "Stop. {label} {zone_text}.",
        SpeechCategory.DANGER_CLEAR: "Path is clearer.",
        SpeechCategory.KNOWN_FACE: "{name} is nearby.",
        SpeechCategory.UNKNOWN_FACE: "Unknown person nearby.",
        SpeechCategory.OCR_TEXT: "{text}",
        SpeechCategory.OBJECT_GUIDANCE: "{label} {zone_text}.",
        SpeechCategory.EMOTION: "{name_prefix}{emotion}.",
        SpeechCategory.LOW_PRIORITY_CONTEXT: "{text}",
        SpeechCategory.SYSTEM_STATUS: "{text}",
    },
    "si": {
        SpeechCategory.DANGER: "නවතින්න. {label} {zone_text}.",
        SpeechCategory.DANGER_CLEAR: "මාර්ගය පැහැදිලියි.",
        SpeechCategory.KNOWN_FACE: "{name} ඔබ අසලයි.",
        SpeechCategory.UNKNOWN_FACE: "නොදන්නා පුද්ගලයෙක් අසලයි.",
        SpeechCategory.OCR_TEXT: "{text}",
        SpeechCategory.OBJECT_GUIDANCE: "{label} {zone_text}.",
        SpeechCategory.EMOTION: "{name_prefix}{emotion}.",
        SpeechCategory.LOW_PRIORITY_CONTEXT: "{text}",
        SpeechCategory.SYSTEM_STATUS: "{text}",
    },
}


ZONE_TEXT: Mapping[str, Mapping[str, str]] = {
    "en": {"left": "on your left", "center": "ahead", "right": "on your right"},
    "si": {"left": "වම් පැත්තේ", "center": "ඉදිරියෙන්", "right": "දකුණු පැත්තේ"},
}


class SpeechFormatter:
    """Normalizes incoming audio intents into localized speech messages."""

    def __init__(self, settings: AppSettings, templates: TemplateMap | None = None) -> None:
        self._settings = settings
        self._templates = templates or DEFAULT_TEMPLATES

    def from_intent(self, payload: Mapping[str, Any], correlation_id: str) -> SpeechMessage | None:
        priority = self._parse_priority(payload)
        category = self._parse_category(payload)
        if priority is None or category is None:
            return None

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}

        language = self._language_for(payload)
        raw_text = " ".join(str(payload.get("text", "")).split())
        text = self._format_text(raw_text, category, language, metadata)
        if not text:
            return None

        dedup_key = str(payload.get("dedup_key", "")).strip()
        if not dedup_key:
            dedup_key = f"{category.value}:{text[:96].casefold()}"

        ttl_seconds = self._safe_float(payload.get("ttl_seconds"), self._settings.cooldowns.ttl_for_priority(priority))
        return SpeechMessage(
            text=text,
            priority=priority,
            category=category,
            dedup_key=dedup_key,
            language=language,
            ttl_seconds=ttl_seconds,
            interrupt=bool(payload.get("interrupt", False)),
            correlation_id=correlation_id,
            metadata=dict(metadata),
        )

    def _format_text(
        self,
        raw_text: str,
        category: SpeechCategory,
        language: str,
        metadata: Mapping[str, Any],
    ) -> str:
        if raw_text and not self._should_use_template(category, language):
            return raw_text

        templates = self._templates.get(language) or self._templates.get("en", {})
        template = templates.get(category)
        if template is None:
            return ""

        values = dict(metadata)
        values.setdefault("text", raw_text)
        values.setdefault("label", str(metadata.get("label", "object")).replace("_", " "))
        values.setdefault("name", str(metadata.get("name", "")).strip() or "Someone")
        values.setdefault("emotion", str(metadata.get("emotion", "nearby")).strip() or "nearby")
        zone = str(metadata.get("zone", "center")).strip() or "center"
        values.setdefault("zone", zone)
        values.setdefault("zone_text", ZONE_TEXT.get(language, ZONE_TEXT["en"]).get(zone, "ahead"))
        name = str(metadata.get("name", "")).strip()
        values.setdefault("name_prefix", f"{name} seems " if name and language == "en" else "")
        return " ".join(template.format_map(_SafeFormat(values)).split())

    @staticmethod
    def _should_use_template(category: SpeechCategory, language: str) -> bool:
        if category == SpeechCategory.OCR_TEXT:
            return False
        if category in {SpeechCategory.LOW_PRIORITY_CONTEXT, SpeechCategory.SYSTEM_STATUS}:
            return language != "en"
        return True

    def _language_for(self, payload: Mapping[str, Any]) -> str:
        language = str(payload.get("language", self._settings.language)).strip().lower()
        if not language:
            language = self._settings.language.lower()
        return language.split("-", maxsplit=1)[0]

    @staticmethod
    def _parse_priority(payload: Mapping[str, Any]) -> AudioPriority | None:
        value = payload.get("priority_value", payload.get("priority"))
        try:
            if isinstance(value, int) and not isinstance(value, bool):
                return AudioPriority(value)
            if isinstance(value, str):
                clean = value.strip()
                if clean.isdigit():
                    return AudioPriority(int(clean))
                return AudioPriority[clean]
        except (KeyError, ValueError):
            return None
        return None

    @staticmethod
    def _parse_category(payload: Mapping[str, Any]) -> SpeechCategory | None:
        value = payload.get("category")
        if isinstance(value, SpeechCategory):
            return value
        if isinstance(value, str):
            try:
                return SpeechCategory(value)
            except ValueError:
                try:
                    return SpeechCategory[value.strip()]
                except KeyError:
                    return None
        return None

    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0.0 else default


class _SafeFormat(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return ""
