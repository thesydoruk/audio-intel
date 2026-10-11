"""Shared transcript types for the HTTP API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from audio_intel.stats import TranscriptStats

SegmentKind = Literal["speech", "sound"]


@dataclass
class Word:
    """One aligned or Whisper-predicted token with sub-segment timestamps."""

    word: str
    start: float
    end: float
    probability: float = 0.0


@dataclass
class SpeechSegment:
    """One speech segment on the timeline (batch / dataset JSON schema)."""

    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    words: list[Word] = field(default_factory=list)
    language: str | None = None
    language_probability: float | None = None
    alignment_failed: bool = False
    speaker_id: str | None = None
    confidence: float | None = None

    @property
    def kind(self) -> SegmentKind:
        return "speech"


@dataclass
class TranscriptResult:
    """In-memory transcript used internally for CTC alignment."""

    video_id: str
    source_audio: str
    language: str
    duration_sec: float
    segments: list[SpeechSegment]
    transcribed_at: str
    aligned_at: str | None = None
    stats: TranscriptStats | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.stats:
            payload["stats"] = self.stats.to_dict()
        return payload
