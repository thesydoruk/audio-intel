from __future__ import annotations

import numpy as np
from audio_intel.separate.audio.timeline import MasterTimeline
from audio_intel.separate.types import TimeSpan


def raised_cosine_fade(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(n, 0), dtype=np.float32)
    return (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32))).astype(np.float32)


def blend_into_track(
    track: np.ndarray,
    timeline: MasterTimeline,
    span: TimeSpan,
    audio: np.ndarray,
    *,
    crossfade_ms: float = 40.0,
) -> None:
    """Write ``audio`` into ``track`` over ``span`` with raised-cosine edges.

    The centre of the span replaces whatever is there; the edges crossfade from
    the existing content into the new audio and back. Callers should pad
    consecutive segments so they overlap by at least the crossfade length: the
    later write then replaces the earlier fade-out, which yields a continuous
    signal with no dip or doubling at segment boundaries.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    sl = timeline.span_to_slice(span)
    n = sl.stop - sl.start
    if n <= 0:
        return
    if len(audio) < n:
        pad = np.zeros(n, dtype=np.float32)
        pad[: len(audio)] = audio
        audio = pad
    else:
        audio = audio[:n]

    fade = max(1, int(round(crossfade_ms * 0.001 * timeline.sample_rate)))
    fade = min(fade, max(1, n // 2))
    window = np.ones(n, dtype=np.float32)
    ramp = raised_cosine_fade(fade)
    window[:fade] *= ramp
    window[-fade:] *= ramp[::-1]

    existing = track[sl]
    track[sl] = existing * (1.0 - window) + audio * window
