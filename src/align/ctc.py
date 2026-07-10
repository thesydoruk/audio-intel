"""WhisperX CTC forced alignment for per-word timestamps.

Whisper produces segment-level timestamps; this module refines them with a
language-specific wav2vec2 model (via WhisperX). Alignment is best-effort:
failed segments keep Whisper timings and are flagged with ``alignment_failed``.

Used by the HTTP ``Transcriber`` when ``align=true`` in the request.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from audio_intel.common.model_pool import ModelPool
from audio_intel.config import AlignmentConfig
from audio_intel.types import SpeechSegment, TranscriptResult, Word

_align_failure_pattern = re.compile(
    r'Failed to align segment \("((?:[^"\\]|\\.)*)"\): .+resorting to original',
)


class _AlignFailureCollector(logging.Handler):
    """Capture WhisperX segments that fell back to original timestamps."""

    def __init__(self) -> None:
        super().__init__()
        self.failed_texts: set[str] = set()

    def emit(self, record: logging.LogRecord) -> None:
        match = _align_failure_pattern.search(record.getMessage())
        if match:
            self.failed_texts.add(match.group(1))


class CTCAligner:
    """Forced alignment via WhisperX wav2vec2 CTC."""

    def __init__(
        self,
        config: AlignmentConfig,
        *,
        language: str,
        device: str,
    ) -> None:
        self.config = config
        self.device = config.device or device
        self.language = language
        self._pool = ModelPool(self._create_model)

    def ensure_ready(self, num_workers: int = 1) -> None:
        """Pre-load ``num_workers`` WhisperX align models."""
        try:
            import whisperx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "WhisperX is required for CTC alignment. Install with: pip install -e '.[align]'"
            ) from exc
        self._pool.ensure_ready(num_workers)

    def align_transcript(
        self,
        transcript: TranscriptResult,
        *,
        audio_path: str | None = None,
    ) -> TranscriptResult:
        """Run CTC alignment and return a copy of ``transcript`` with word timings.

        ``audio_path`` overrides ``transcript.source_audio`` when the prepared
        16 kHz workspace WAV should be used instead of the original upload path.
        """
        try:
            import whisperx
        except ImportError as exc:
            raise ImportError(
                "WhisperX is required for CTC alignment. Install with: pip install -e '.[align]'"
            ) from exc

        if not transcript.segments:
            return transcript

        collector = _AlignFailureCollector()
        wx_logger = logging.getLogger("whisperx.alignment")
        wx_logger.addHandler(collector)
        try:
            align_model, metadata = self._pool.acquire()
            try:
                source_audio = audio_path or transcript.source_audio
                audio = whisperx.load_audio(source_audio)
                wx_segments = [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "avg_logprob": seg.avg_logprob,
                    }
                    for seg in transcript.segments
                ]

                aligned = whisperx.align(
                    wx_segments,
                    align_model,
                    metadata,
                    audio,
                    self.device,
                    interpolate_method=self.config.interpolate_method,
                    return_char_alignments=False,
                )
            finally:
                self._pool.release((align_model, metadata))
        finally:
            wx_logger.removeHandler(collector)

        segments = self._map_aligned_segments(
            transcript.segments,
            aligned["segments"],
            failed_texts=collector.failed_texts,
        )
        marked = sum(1 for seg in segments if seg.alignment_failed)
        stats = transcript.stats
        if stats is not None and marked:
            stats = replace(stats, marked_alignment_failed=stats.marked_alignment_failed + marked)
        return TranscriptResult(
            video_id=transcript.video_id,
            source_audio=transcript.source_audio,
            language=transcript.language,
            duration_sec=transcript.duration_sec,
            segments=segments,
            transcribed_at=transcript.transcribed_at,
            aligned_at=datetime.now(timezone.utc).isoformat(),
            stats=stats,
        )

    def _create_model(self) -> tuple[Any, Any]:
        import whisperx

        return whisperx.load_align_model(
            language_code=self.language,
            device=self.device,
            model_name=self.config.model,
        )

    def _map_aligned_segments(
        self,
        original_segments: list[SpeechSegment],
        aligned_segments: list[dict],
        *,
        failed_texts: set[str] | None = None,
    ) -> list[SpeechSegment]:
        """Merge WhisperX output back into :class:`~audio_intel.types.SpeechSegment` objects.

        A segment is marked ``alignment_failed`` when WhisperX logged a fallback
        for its text, or when aligned words are missing entirely.
        """
        failed_texts = failed_texts or set()
        fallback_by_index = {seg.id: seg for seg in original_segments}
        mapped: list[SpeechSegment] = []

        for idx, seg in enumerate(aligned_segments):
            words: list[Word] = []
            for word in seg.get("words", []):
                if "start" not in word or "end" not in word:
                    continue
                words.append(
                    Word(
                        word=str(word.get("word", "")).strip(),
                        start=float(word["start"]),
                        end=float(word["end"]),
                        probability=float(word.get("score", 0.0)),
                    )
                )

            fallback = fallback_by_index.get(idx) or (
                original_segments[idx] if idx < len(original_segments) else None
            )
            start = float(seg["start"])
            end = float(seg["end"])
            if words:
                start = words[0].start
                end = words[-1].end

            text = str(seg.get("text", "")).strip()
            alignment_failed = bool(text) and (not words or text in failed_texts)

            mapped.append(
                SpeechSegment(
                    id=idx,
                    start=start,
                    end=end,
                    text=text,
                    avg_logprob=float(
                        seg.get("avg_logprob", fallback.avg_logprob if fallback else -0.5)
                    ),
                    no_speech_prob=float(fallback.no_speech_prob if fallback else 0.0),
                    compression_ratio=float(fallback.compression_ratio if fallback else 1.0),
                    words=words,
                    language=fallback.language if fallback else None,
                    language_probability=(fallback.language_probability if fallback else None),
                    alignment_failed=alignment_failed,
                )
            )

        return mapped
