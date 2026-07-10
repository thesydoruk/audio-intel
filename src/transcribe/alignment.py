"""Bridge HTTP speech segments and WhisperX CTC alignment.

Converts service timeline dicts into :class:`~audio_intel.types.TranscriptResult`
for :class:`~audio_intel.align.ctc.CTCAligner`, then merges aligned words back.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

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
    if cfg.alignment_language:
        return (cfg.alignment_language,)
    if cfg.forced_language:
        return (cfg.forced_language,)
    return ("en",)


def build_alignment_config(cfg: Config) -> AlignmentConfig:
    """Map server env flags into an :class:`~audio_intel.config.AlignmentConfig`."""
    return AlignmentConfig(
        enabled=True,
        device=cfg.alignment_device or cfg.device,
        model=cfg.alignment_model,
        interpolate_method=cfg.alignment_interpolate_method,
    )


def speech_segments_to_transcript(
    *,
    cfg: Config,
    path: str,
    speech_segments: list[dict],
    languages: list[str],
    duration_sec: float,
    language_override: str | None,
) -> TranscriptResult:
    """Convert service speech dicts into a :class:`~audio_intel.types.TranscriptResult` for CTC."""
    return TranscriptResult(
        video_id="request",
        source_audio=path,
        language=resolve_alignment_language(cfg, languages, language_override),
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
