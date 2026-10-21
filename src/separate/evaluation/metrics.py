"""Optional objective helpers for evaluation notebooks / future scoring."""

from __future__ import annotations

import numpy as np


def sync_ok(reference_frames: int, candidate_frames: int, sample_rate: int) -> bool:
    return reference_frames == candidate_frames and sample_rate == 48000


def active_ratio(audio: np.ndarray, thr: float = 0.01) -> float:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) == 0:
        return 0.0
    return float(np.mean(np.abs(audio) > thr))


def peak_normalize(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    m = float(np.max(np.abs(audio)) + 1e-12)
    if m > peak:
        return audio * (peak / m)
    return audio
