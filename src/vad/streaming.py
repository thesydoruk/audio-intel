"""Stream Silero VAD over disk-backed audio without loading the full waveform."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import soundfile as sf
from audio_intel.audio.decode import load_audio_window
from audio_intel.vad.regions import SpeechRegion

# Silero VAD expects 512-sample frames at 16 kHz (32 ms).
SILERO_WINDOW_SAMPLES = 512


def total_samples_from_duration(duration_s: float, sample_rate: int) -> int:
    """Round a duration in seconds to an integer sample count."""
    return max(0, int(round(duration_s * sample_rate)))


def filter_min_speech_duration(
    regions: list[SpeechRegion],
    *,
    min_speech_duration_ms: int,
) -> list[SpeechRegion]:
    """Drop speech regions shorter than ``min_speech_duration_ms``."""
    if min_speech_duration_ms <= 0:
        return regions
    min_duration_s = min_speech_duration_ms / 1000.0
    return [region for region in regions if region.duration >= min_duration_s]


def iter_vad_windows(
    path: str,
    *,
    sample_rate: int,
    total_samples: int,
    window_samples: int = SILERO_WINDOW_SAMPLES,
    block_duration_s: float = 30.0,
) -> Iterator[np.ndarray]:
    """Yield fixed-size mono PCM windows read sequentially from ``path``."""
    if total_samples <= 0:
        return

    block_samples = max(int(round(block_duration_s * sample_rate)), window_samples)
    handle = _open_sequential_wav(path, sample_rate)
    try:
        yield from _iter_vad_windows_blocks(
            path,
            sample_rate=sample_rate,
            total_samples=total_samples,
            window_samples=window_samples,
            block_samples=block_samples,
            handle=handle,
        )
    finally:
        if handle is not None:
            handle.close()


def _open_sequential_wav(path: str, sample_rate: int) -> sf.SoundFile | None:
    """Open a matching-rate WAV for sequential reads; ``None`` uses per-block decode."""
    try:
        handle = sf.SoundFile(path)
    except (OSError, RuntimeError, ValueError):
        return None
    if handle.samplerate != sample_rate:
        handle.close()
        return None
    return handle


def _read_vad_block(
    *,
    path: str,
    sample_rate: int,
    position: int,
    read_samples: int,
    handle: sf.SoundFile | None,
) -> np.ndarray:
    if handle is not None:
        data = handle.read(read_samples, dtype="float32", always_2d=False)
        if getattr(data, "size", 0) == 0:
            return np.array([], dtype=np.float32)
        if getattr(data, "ndim", 1) > 1:
            data = np.mean(data, axis=1, dtype=np.float32)
        return np.asarray(data, dtype=np.float32)
    return load_audio_window(
        path,
        sample_rate,
        position / sample_rate,
        read_samples / sample_rate,
    )


def _iter_vad_windows_blocks(
    path: str,
    *,
    sample_rate: int,
    total_samples: int,
    window_samples: int,
    block_samples: int,
    handle: sf.SoundFile | None,
) -> Iterator[np.ndarray]:
    position = 0
    carry = np.array([], dtype=np.float32)

    while position < total_samples:
        remaining = total_samples - position
        read_samples = min(block_samples, remaining)
        block = _read_vad_block(
            path=path,
            sample_rate=sample_rate,
            position=position,
            read_samples=read_samples,
            handle=handle,
        )
        # Source duration can overshoot the decoded WAV; empty decode means EOF.
        # Advancing by zero would spin forever on the same ffmpeg seek.
        if block.size == 0:
            break
        position += len(block)

        buffer = np.concatenate((carry, block)) if carry.size else block
        del block
        offset = 0
        while offset + window_samples <= len(buffer):
            yield buffer[offset : offset + window_samples].astype(np.float32, copy=False)
            offset += window_samples
        carry = np.array(buffer[offset:], dtype=np.float32, copy=True)
        del buffer

    if carry.size <= 0:
        return

    padded = np.zeros(window_samples, dtype=np.float32)
    padded[: carry.size] = carry
    yield padded


def regions_from_vad_iterator_events(
    events: list[dict[str, float]],
    *,
    duration_s: float,
    speech_active_at_end: bool,
) -> list[SpeechRegion]:
    """Convert VADIterator start/end events into contiguous speech regions."""
    regions: list[SpeechRegion] = []
    current_start: float | None = None

    for event in events:
        if "start" in event:
            current_start = float(event["start"])
        elif "end" in event and current_start is not None:
            regions.append(SpeechRegion(start=current_start, end=float(event["end"])))
            current_start = None

    if speech_active_at_end and current_start is not None:
        regions.append(SpeechRegion(start=current_start, end=round(duration_s, 3)))

    return regions
