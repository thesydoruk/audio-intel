"""Shared VAD data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechRegion:
    """Inclusive start / exclusive end of a speech span in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start
