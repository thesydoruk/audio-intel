"""Bridge HTTP speech segments and WhisperX CTC alignment.

Converts service timeline dicts into :class:`~audio_intel.types.TranscriptResult`
for :class:`~audio_intel.align.ctc.CTCAligner`, then merges aligned words back.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from audio_intel.audio.chunking import Chunk
from audio_intel.config import AlignmentConfig, Config
from audio_intel.transcribe.segments import (
    apply_aligned_segments,
    speech_dicts_to_segments,
)
from audio_intel.types import TranscriptResult

log = logging.getLogger("audio-intel")


def resolve_alignment_language(
    cfg: Config,
    languages: list[str],
    override: str | None,
) -> str:
    """Pick the language code for the WhisperX align model."""
    if cfg.alignment_language:
        return cfg.alignment_language
    if override and override.lower() not in {"", "auto", "unknown"}:
        return override
    for lang in languages:
        if lang and lang != "unknown":
            return lang
    return "en"


def alignment_preload_languages(cfg: Config) -> tuple[str, ...]:
    """Languages whose WhisperX align models should load at server startup."""
    if cfg.alignment_preload:
        return cfg.alignment_preload
    if cfg.alignment_language:
        return (cfg.alignment_language,)
    if cfg.forced_language:
        return (cfg.forced_language,)
    return ("uk", "en")


def build_alignment_config(cfg: Config) -> AlignmentConfig:
    """Map server env flags into an :class:`~audio_intel.config.AlignmentConfig`."""
    return AlignmentConfig(
        enabled=True,
        device=cfg.alignment_device or cfg.device,
        model=cfg.alignment_model,
        interpolate_method=cfg.alignment_interpolate_method,
    )


def segment_alignment_language(seg: dict, fallback: str) -> str:
    """Language for one speech segment; empty / unknown uses ``fallback``."""
    lang = seg.get("language")
    if not lang or str(lang).lower() in {"", "auto", "unknown"}:
        return fallback
    return str(lang)


def group_speech_indices_by_language(
    speech_segments: list[dict],
    fallback: str,
) -> dict[str, list[int]]:
    """Group timeline indexes by ``segment.language`` (fallback when missing)."""
    groups: dict[str, list[int]] = {}
    for index, seg in enumerate(speech_segments):
        lang = segment_alignment_language(seg, fallback)
        groups.setdefault(lang, []).append(index)
    return groups


def shift_speech_segments(segments: list[dict], offset: float) -> list[dict]:
    """Return copies with ``start`` / ``end`` / word times shifted by ``offset`` seconds."""
    shifted: list[dict] = []
    for seg in segments:
        item = dict(seg)
        item["start"] = round(float(seg["start"]) + offset, 3)
        item["end"] = round(float(seg["end"]) + offset, 3)
        words = seg.get("words")
        if isinstance(words, list):
            item["words"] = [
                {
                    **word,
                    **(
                        {"start": round(float(word["start"]) + offset, 3)}
                        if "start" in word
                        else {}
                    ),
                    **({"end": round(float(word["end"]) + offset, 3)} if "end" in word else {}),
                }
                for word in words
            ]
        shifted.append(item)
    return shifted


def _chunk_for_segment(seg: dict, chunks: list[Chunk]) -> Chunk | None:
    """Pick the VAD chunk that contains the segment midpoint, else max overlap."""
    start = float(seg["start"])
    end = float(seg["end"])
    mid = (start + end) / 2.0
    for chunk in chunks:
        if chunk.start <= mid <= chunk.end:
            return chunk
    best: Chunk | None = None
    best_overlap = 0.0
    for chunk in chunks:
        overlap = min(end, chunk.end) - max(start, chunk.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = chunk
    return best


def assign_segment_indices_to_chunks(
    speech_segments: list[dict],
    chunks: list[Chunk],
    indices: list[int],
) -> list[tuple[Chunk, list[int]]]:
    """Group language-subset indexes by the VAD chunk that covers each segment."""
    grouped: dict[tuple[float, float], list[int]] = {}
    chunk_by_span: dict[tuple[float, float], Chunk] = {}
    unassigned: list[int] = []
    for index in indices:
        chunk = _chunk_for_segment(speech_segments[index], chunks)
        if chunk is None:
            unassigned.append(index)
            continue
        key = (chunk.start, chunk.end)
        chunk_by_span[key] = chunk
        grouped.setdefault(key, []).append(index)
    assigned = [(chunk_by_span[key], grouped[key]) for key in grouped]
    if unassigned and chunks:
        assigned.append((chunks[0], unassigned))
    return assigned


def speech_segments_to_transcript(
    *,
    cfg: Config,
    path: str,
    speech_segments: list[dict],
    languages: list[str],
    duration_sec: float,
    language_override: str | None,
    language: str | None = None,
) -> TranscriptResult:
    """Convert service speech dicts into a :class:`~audio_intel.types.TranscriptResult` for CTC."""
    return TranscriptResult(
        video_id="request",
        source_audio=path,
        language=language or resolve_alignment_language(cfg, languages, language_override),
        duration_sec=duration_sec,
        segments=speech_dicts_to_segments(speech_segments),
        transcribed_at=datetime.now(timezone.utc).isoformat(),
    )


def merge_aligned_speech_segments(
    original: list[dict],
    aligned: TranscriptResult,
) -> tuple[list[dict], int]:
    """Apply aligned word timestamps back onto the service speech dicts."""
    return apply_aligned_segments(original, aligned.segments)
