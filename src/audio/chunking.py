"""Group Silero VAD speech regions into bounded transcription chunks.

The point of chunking (vs. faster-whisper's built-in ``vad_filter``) is that each
chunk is a *contiguous* slice of the original audio cut at natural silences. We
transcribe each slice independently, which lets the model auto-detect language
per chunk (mixed-language media) and keeps timestamps anchored to real audio
time instead of a silence-stripped stream.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A contiguous audio span to transcribe, in seconds from file start."""

    start: float
    end: float


def build_chunks(
    speech: list[dict],
    sample_rate: int,
    total_samples: int,
    max_chunk_s: float,
    pad_s: float,
) -> list[Chunk]:
    """Merge consecutive VAD speech regions into chunks no longer than ``max_chunk_s``.

    Regions are merged greedily while the running span (first region start to the
    candidate region end) stays within ``max_chunk_s``; otherwise a new chunk is
    started. Each resulting chunk is padded by ``pad_s`` on both sides (clamped to
    the audio bounds) so word edges near the cut are not clipped.

    @param speech — VAD output: ``[{"start": samples, "end": samples}, ...]``.
    @returns time-ordered, non-empty chunks.
    """
    if not speech:
        return []

    total_s = total_samples / sample_rate
    merged: list[Chunk] = []
    cur_start: float | None = None
    cur_end: float | None = None

    for region in speech:
        start_s = region["start"] / sample_rate
        end_s = region["end"] / sample_rate
        if cur_start is None:
            cur_start, cur_end = start_s, end_s
        elif end_s - cur_start <= max_chunk_s:
            cur_end = end_s
        else:
            merged.append(Chunk(cur_start, cur_end))  # type: ignore[arg-type]
            cur_start, cur_end = start_s, end_s

    if cur_start is not None and cur_end is not None:
        merged.append(Chunk(cur_start, cur_end))

    padded: list[Chunk] = []
    for chunk in merged:
        start = max(0.0, chunk.start - pad_s)
        end = min(total_s, chunk.end + pad_s)
        if end > start:
            padded.append(Chunk(round(start, 3), round(end, 3)))
    return padded
