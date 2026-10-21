from __future__ import annotations

import numpy as np
from audio_intel.separate.audio.resample import seconds_to_samples
from audio_intel.separate.types import TimeSpan


class MasterTimeline:
    """Sample-accurate master timeline at a fixed sample rate (typically 48 kHz)."""

    def __init__(self, duration: float, sample_rate: int = 48000) -> None:
        self.duration = float(duration)
        self.sample_rate = int(sample_rate)
        self.num_samples = seconds_to_samples(self.duration, self.sample_rate)

    def span_to_slice(self, span: TimeSpan) -> slice:
        start, end = span.clamp(0.0, self.duration).to_samples(self.sample_rate)
        start = max(0, min(self.num_samples, start))
        end = max(start, min(self.num_samples, end))
        return slice(start, end)

    def write(self, track: np.ndarray, span: TimeSpan, audio: np.ndarray) -> None:
        sl = self.span_to_slice(span)
        n = sl.stop - sl.start
        if n <= 0:
            return
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < n:
            pad = np.zeros(n, dtype=np.float32)
            pad[: len(audio)] = audio
            audio = pad
        elif len(audio) > n:
            audio = audio[:n]
        track[sl] = audio

    def add_crossfade(
        self,
        track: np.ndarray,
        span: TimeSpan,
        audio: np.ndarray,
        *,
        fade_samples: int,
        weight: float = 1.0,
    ) -> None:
        """Overlap-add with raised-cosine edges."""
        sl = self.span_to_slice(span)
        n = sl.stop - sl.start
        if n <= 0:
            return
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) != n:
            tmp = np.zeros(n, dtype=np.float32)
            m = min(n, len(audio))
            tmp[:m] = audio[:m]
            audio = tmp
        fade = min(fade_samples, max(1, n // 4))
        window = np.ones(n, dtype=np.float32)
        if fade > 1:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade, dtype=np.float32))
            window[:fade] = ramp
            window[-fade:] = ramp[::-1]
        track[sl] += audio * window * float(weight)
