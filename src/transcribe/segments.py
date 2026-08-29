"""Build and convert speech segments from faster-whisper output.

Maps raw faster-whisper segments into :class:`~audio_intel.types.SpeechSegment`,
serializes them for the HTTP timeline, and merges CTC-aligned words back in.
"""

from __future__ import annotations

import math
from typing import Any

from audio_intel.quality.normalization import normalize_segment_text
from audio_intel.types import SpeechSegment, Word


def logprob_to_confidence(logprob: float | None) -> float:
    """Convert Whisper avg log-probability to a 0–1 confidence score."""
    if logprob is None:
        return 0.0
    return round(min(max(math.exp(logprob), 0.0), 1.0), 3)


def confidence_to_logprob(confidence: float | None) -> float:
    """Inverse of :func:`logprob_to_confidence` for timeline dict round-trips."""
    if confidence is None or confidence <= 0:
        return -0.5
    return math.log(max(min(confidence, 1.0), 1e-6))


def segment_from_whisper(
    seg: Any,
    *,
    seg_id: int,
    time_offset: float = 0.0,
    language: str | None = None,
    language_probability: float | None = None,
    normalize_text: bool = True,
) -> SpeechSegment:
    """Map one faster-whisper segment to a :class:`~audio_intel.types.SpeechSegment`."""
    raw_text = seg.text.strip()
    text = normalize_segment_text(raw_text) if normalize_text else raw_text
    words: list[Word] = []
    if seg.words:
        for word in seg.words:
            if not word.word:
                continue
            words.append(
                Word(
                    word=word.word.strip(),
                    start=round(float(word.start) + time_offset, 3),
                    end=round(float(word.end) + time_offset, 3),
                    probability=float(getattr(word, "probability", 0.0)),
                )
            )

    start = round(float(seg.start) + time_offset, 3)
    end = round(float(seg.end) + time_offset, 3)
    if words:
        start = words[0].start
        end = words[-1].end

    logprob = float(getattr(seg, "avg_logprob", -0.5))
    return SpeechSegment(
        id=seg_id,
        start=start,
        end=end,
        text=text,
        avg_logprob=logprob,
        no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0)),
        compression_ratio=float(getattr(seg, "compression_ratio", 1.0)),
        words=words,
        language=language,
        language_probability=language_probability,
        confidence=logprob_to_confidence(logprob),
    )


def speech_segment_to_timeline_dict(seg: SpeechSegment) -> dict[str, Any]:
    """Serialize a speech segment for the HTTP timeline (speech + sound events)."""
    payload: dict[str, Any] = {
        "kind": "speech",
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "confidence": seg.confidence,
        "avg_logprob": seg.avg_logprob,
        "no_speech_prob": seg.no_speech_prob,
        "compression_ratio": seg.compression_ratio,
    }
    if seg.language:
        payload["language"] = seg.language
    if seg.speaker_id:
        payload["speaker_id"] = seg.speaker_id
    if seg.words:
        payload["words"] = [
            {
                "word": word.word,
                "start": word.start,
                "end": word.end,
                **({"probability": round(word.probability, 3)} if word.probability else {}),
            }
            for word in seg.words
        ]
    if seg.alignment_failed:
        payload["alignment_failed"] = True
    return payload


def speech_dicts_to_segments(speech_segments: list[dict]) -> list[SpeechSegment]:
    """Rebuild typed segments from HTTP timeline dicts (``kind: speech`` only)."""
    speech_index = 0
    segments: list[SpeechSegment] = []
    for item in speech_segments:
        if item.get("kind") != "speech":
            continue
        confidence = item.get("confidence")
        words = [
            Word(
                word=str(word["word"]),
                start=float(word["start"]),
                end=float(word["end"]),
                probability=float(word.get("probability", 0.0)),
            )
            for word in item.get("words") or []
            if word.get("word")
        ]
        segments.append(
            SpeechSegment(
                id=speech_index,
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]),
                avg_logprob=float(item.get("avg_logprob", confidence_to_logprob(confidence))),
                no_speech_prob=float(item.get("no_speech_prob", 0.0)),
                compression_ratio=float(item.get("compression_ratio", 1.0)),
                words=words,
                language=item.get("language"),
                confidence=confidence,
                speaker_id=item.get("speaker_id"),
                alignment_failed=bool(item.get("alignment_failed", False)),
            )
        )
        speech_index += 1
    return segments


def apply_aligned_segments(
    original: list[dict],
    aligned: list[SpeechSegment],
) -> tuple[list[dict], int]:
    """Merge CTC-aligned words back into HTTP timeline dicts."""
    failed_count = 0
    updated: list[dict] = []
    speech_index = 0

    for item in original:
        if item.get("kind") != "speech":
            updated.append(item)
            continue

        if speech_index >= len(aligned):
            updated.append({**item, "alignment_failed": True})
            failed_count += 1
            continue

        merged = speech_segment_to_timeline_dict(aligned[speech_index])
        merged["alignment_failed"] = aligned[speech_index].alignment_failed
        merged["speaker_id"] = item.get("speaker_id") or merged.get("speaker_id")
        if aligned[speech_index].alignment_failed:
            failed_count += 1
        updated.append(merged)
        speech_index += 1

    return updated, failed_count


def overall_confidence(segments: list[dict]) -> float:
    """Duration-weighted mean confidence across speech timeline entries."""
    total_weight = 0.0
    acc = 0.0
    for seg in segments:
        if seg.get("kind") != "speech":
            continue
        weight = max(float(seg["end"]) - float(seg["start"]), 1e-3)
        acc += float(seg.get("confidence", 0.0)) * weight
        total_weight += weight
    return round(acc / total_weight, 3) if total_weight > 0 else 0.0
