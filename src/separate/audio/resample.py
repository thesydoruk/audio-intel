from __future__ import annotations

import numpy as np
import soxr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality resample. Preserves dtype float32."""
    if orig_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        out = soxr.resample(audio, orig_sr, target_sr, quality="HQ")
        return np.asarray(out, dtype=np.float32)
    channels = [
        soxr.resample(audio[:, c], orig_sr, target_sr, quality="HQ") for c in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def map_time(t: float, _from_sr: int, _to_sr: int) -> float:
    """Identity mapping: timeline is always in absolute seconds."""
    return float(t)


def seconds_to_samples(t: float, sample_rate: int) -> int:
    return int(round(float(t) * sample_rate))


def samples_to_seconds(n: int, sample_rate: int) -> float:
    return float(n) / float(sample_rate)
