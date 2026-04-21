"""Child-process pyttsx3 speaker used to preserve interruptibility."""

from __future__ import annotations

import json
import sys
from typing import Any


def _set_voice(engine: Any, requested_voice: str) -> None:
    if not requested_voice:
        return
    requested = requested_voice.casefold()
    voices = engine.getProperty("voices") or []
    for voice in voices:
        voice_id = str(getattr(voice, "id", ""))
        name = str(getattr(voice, "name", ""))
        if requested in voice_id.casefold() or requested in name.casefold():
            engine.setProperty("voice", voice_id)
            return


def main() -> int:
    payload = json.loads(sys.stdin.read())
    text = " ".join(str(payload.get("text", "")).split())
    if not text:
        return 0

    import pyttsx3  # type: ignore[import-not-found]

    engine = pyttsx3.init()
    engine.setProperty("rate", int(payload.get("rate_wpm", 165)))
    engine.setProperty("volume", max(0.0, min(1.0, float(payload.get("volume", 1.0)))))
    _set_voice(engine, str(payload.get("voice", "")))
    engine.say(text)
    engine.runAndWait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
