"""The background bed: the original mix wherever nobody speaks, the instrumental under speech.

A separation model's instrumental stem is a re-synthesis of the mix, and every second
of it that plays where there was no voice to remove is a second of needless artefacts.
:func:`gate_bed` keeps the source bit for bit outside speech and crosses over to the
instrumental only where the vocals stem has energy, with linear ramps so the seams are
inaudible. The vocals stem is the right gate: it marks where the *original* voice is
at sample precision, which is exactly where the bed must not contain it.
"""

from __future__ import annotations

import numpy as np

FRAME_MS = 20.0


def activity_mask(
    vocals: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float,
    hold_ms: float,
    frame_ms: float = FRAME_MS,
) -> np.ndarray:
    """Per-sample boolean mask: where the vocals stem is within ``threshold_db`` of its peak.

    Frame RMS over ``frame_ms`` windows (all channels together) is compared with the
    loudest frame; active frames are then held for ``hold_ms`` on both sides so onsets,
    tails and short pauses stay inside one region. A silent stem gives an all-False mask.
    """
    vocals = np.asarray(vocals, dtype=np.float64)
    if vocals.ndim == 1:
        vocals = vocals[:, None]
    n = len(vocals)
    frame = max(1, int(round(frame_ms * sample_rate / 1000.0)))
    n_frames = -(-n // frame)
    padded = np.zeros((n_frames * frame, vocals.shape[1]), dtype=np.float64)
    padded[:n] = vocals
    rms = np.sqrt(np.mean(padded.reshape(n_frames, -1) ** 2, axis=1))
    peak = float(rms.max()) if n_frames else 0.0
    if peak <= 0.0:
        return np.zeros(n, dtype=bool)
    active_frames = 20.0 * np.log10(rms / peak + 1e-20) > threshold_db
    mask = np.repeat(active_frames, frame)[:n]
    hold = int(round(hold_ms * sample_rate / 1000.0))
    return _dilate(mask, hold)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """True wherever ``mask`` is True within ``radius`` samples."""
    if radius <= 0 or not mask.any():
        return mask.copy()
    counts = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    idx = np.arange(len(mask))
    lo = np.clip(idx - radius, 0, len(mask))
    hi = np.clip(idx + radius + 1, 0, len(mask))
    return (counts[hi] - counts[lo]) > 0


def gain_curve(mask: np.ndarray, fade: int) -> np.ndarray:
    """``mask`` as a float gain with linear ramps of ``fade`` samples *outside* every region.

    Exactly 1.0 on the mask, exactly 0.0 further than ``fade`` samples from it, and a
    straight line in between (a moving average over the mask dilated by ``fade / 2``).
    """
    if fade <= 0:
        return mask.astype(np.float64)
    wide = _dilate(mask, fade // 2)
    counts = np.concatenate(([0], np.cumsum(wide, dtype=np.int64)))
    idx = np.arange(len(mask))
    half_lo = fade // 2
    half_hi = fade - half_lo
    lo = np.clip(idx - half_lo, 0, len(mask))
    hi = np.clip(idx + half_hi, 0, len(mask))
    g = (counts[hi] - counts[lo]) / float(fade)
    g[mask] = 1.0
    return np.clip(g, 0.0, 1.0)


def gate_bed(
    source: np.ndarray,
    instrumental: np.ndarray,
    vocals: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -45.0,
    hold_ms: float = 150.0,
    fade_ms: float = 150.0,
) -> np.ndarray:
    """``source`` where the vocals stem is quiet, ``instrumental`` where it is active.

    All three arrays are ``(samples, channels)`` (or 1-D) at ``sample_rate`` and the same
    length. Outside activity (plus ``fade_ms`` of ramp on each side) the result is the
    source sample for sample; on the held activity mask it is the instrumental sample for
    sample; the ramps mix the two linearly. The dtype of ``source`` is kept.
    """
    src = np.asarray(source)
    squeeze = src.ndim == 1
    src2 = src[:, None] if squeeze else src
    inst = np.asarray(instrumental, dtype=src2.dtype)
    inst = inst[:, None] if inst.ndim == 1 else inst
    if inst.shape != src2.shape:
        raise ValueError(f"instrumental shape {inst.shape} != source shape {src2.shape}")
    if len(np.asarray(vocals)) != len(src2):
        raise ValueError("vocals must have the same length as the source")

    mask = activity_mask(vocals, sample_rate, threshold_db=threshold_db, hold_ms=hold_ms)
    fade = int(round(fade_ms * sample_rate / 1000.0))
    g = gain_curve(mask, fade)[:, None]

    mixed = (src2.astype(np.float64) * (1.0 - g) + inst.astype(np.float64) * g).astype(src2.dtype)
    bed = np.where(g >= 1.0, inst, np.where(g <= 0.0, src2, mixed))
    return bed[:, 0] if squeeze else np.ascontiguousarray(bed)
