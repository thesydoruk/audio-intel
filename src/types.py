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
class SoundSegment:
    """Non-speech AudioSet event (HTTP service timeline)."""

    start: float
    end: float
    label: str
    index: int
    score: float
    prompt_relevant: bool = True

    @property
    def kind(self) -> SegmentKind:
        return "sound"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "sound",
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "index": self.index,
            "score": self.score,
            "prompt_relevant": self.prompt_relevant,
        }


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


def speech_segment_to_api_dict(seg: SpeechSegment) -> dict[str, Any]:
    """Serialize a speech segment for the OpenAI-compatible HTTP API."""
    body: dict[str, Any] = {
        "kind": "speech",
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
    }
    if seg.confidence is not None:
        body["confidence"] = seg.confidence
    if seg.language:
        body["language"] = seg.language
    if seg.speaker_id:
        body["speaker_id"] = seg.speaker_id
    if seg.words:
        body["words"] = [
            {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
            for w in seg.words
        ]
    return body


def speech_segment_from_api_dict(data: dict[str, Any], seg_id: int = 0) -> SpeechSegment:
    """Parse one HTTP speech segment into the shared dataclass."""
    words = [
        Word(
            word=str(w.get("word", "")),
            start=float(w["start"]),
            end=float(w["end"]),
            probability=float(w.get("probability", w.get("score", 0.0))),
        )
        for w in data.get("words") or []
        if "start" in w and "end" in w
    ]
    return SpeechSegment(
        id=seg_id,
        start=float(data["start"]),
        end=float(data["end"]),
        text=str(data.get("text", "")),
        avg_logprob=float(data.get("avg_logprob", -0.5)),
        no_speech_prob=float(data.get("no_speech_prob", 0.0)),
        compression_ratio=float(data.get("compression_ratio", 1.0)),
        words=words,
        language=data.get("language"),
        confidence=data.get("confidence"),
        speaker_id=data.get("speaker_id"),
        alignment_failed=bool(data.get("alignment_failed", False)),
    )
