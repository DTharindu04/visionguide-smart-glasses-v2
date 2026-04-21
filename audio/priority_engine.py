"""Audio interrupt and replacement policy."""

from __future__ import annotations

from config.constants import AudioPriority
from audio.types import QueueDecision, SpeechMessage


class PriorityInterruptEngine:
    """Applies the product priority rules for speech feedback."""

    def __init__(self, p1_interrupts_all: bool = True, p2_replaces_p3_p4: bool = True) -> None:
        self._p1_interrupts_all = p1_interrupts_all
        self._p2_replaces_p3_p4 = p2_replaces_p3_p4

    def decide(
        self,
        incoming: SpeechMessage,
        current: SpeechMessage | None,
        allow_interruptions: bool = True,
    ) -> QueueDecision:
        if incoming.priority == AudioPriority.P1_DANGER:
            return QueueDecision(
                accepted=True,
                interrupt_current=current is not None and self._p1_interrupts_all and allow_interruptions,
                drop_priorities=(
                    AudioPriority.P2_IDENTITY_OCR,
                    AudioPriority.P3_NAVIGATION,
                    AudioPriority.P4_CONTEXT,
                )
                if self._p1_interrupts_all
                else (),
                reason="p1_interrupts_all",
            )

        if incoming.priority == AudioPriority.P2_IDENTITY_OCR:
            interrupt_current = (
                self._p2_replaces_p3_p4
                and allow_interruptions
                and current is not None
                and current.priority in {
                    AudioPriority.P3_NAVIGATION,
                    AudioPriority.P4_CONTEXT,
                }
            )
            return QueueDecision(
                accepted=True,
                interrupt_current=interrupt_current,
                drop_priorities=(AudioPriority.P3_NAVIGATION, AudioPriority.P4_CONTEXT)
                if self._p2_replaces_p3_p4
                else (),
                reason="p2_replaces_p3_p4",
            )

        if current is not None and current.priority < incoming.priority:
            return QueueDecision(accepted=True, interrupt_current=False, reason="queued_behind_higher_priority")

        return QueueDecision(accepted=True, interrupt_current=False, reason="queued")
