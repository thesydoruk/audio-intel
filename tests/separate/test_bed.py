"""The gated background bed: source outside speech, instrumental inside, linear ramps between."""

from __future__ import annotations

import numpy as np
import pytest
from audio_intel.separate.audio.bed import activity_mask, gain_curve, gate_bed

SR = 48000


def _signals(seconds: float = 4.0, stereo: bool = True, seed: int = 0):
    rng = np.random.default_rng(seed)
    shape = (int(seconds * SR), 2 if stereo else 1)
    source = rng.uniform(-0.5, 0.5, shape).astype(np.float32)
    instrumental = rng.uniform(-0.5, 0.5, shape).astype(np.float32)
    return source, instrumental


def _vocals_burst(n: int, start_s: float, end_s: float, level: float = 0.3) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SR
    v = np.zeros((n, 2), dtype=np.float32)
    sl = slice(int(start_s * SR), int(end_s * SR))
    v[sl] = (level * np.sin(2 * np.pi * 220 * t[sl]))[:, None]
    return v


def test_source_bit_exact_outside_activity_and_instrumental_inside():
    source, instrumental = _signals()
    n = len(source)
    vocals = _vocals_burst(n, 1.0, 2.0)
    bed = gate_bed(source, instrumental, vocals, SR, threshold_db=-45, hold_ms=150, fade_ms=150)

    assert bed.shape == source.shape and bed.dtype == source.dtype
    # hold (150 ms) + fade (150 ms) + one 20 ms frame of slack on each side
    before = int((1.0 - 0.32) * SR)
    after = int((2.0 + 0.32) * SR)
    assert np.array_equal(bed[:before], source[:before])
    assert np.array_equal(bed[after:], source[after:])
    # the held region, including the hold itself, is the instrumental sample for sample
    held = slice(int((1.0 - 0.14) * SR), int((2.0 + 0.14) * SR))
    assert np.array_equal(bed[held], instrumental[held])


def test_ramps_are_linear_monotonic_and_bounded():
    n = 2 * SR
    mask = np.zeros(n, dtype=bool)
    mask[SR // 2 : SR] = True
    fade = int(0.15 * SR)
    g = gain_curve(mask, fade)

    assert g.min() == 0.0 and g.max() == 1.0
    assert np.all(g[mask] == 1.0)
    assert np.all(g[: SR // 2 - fade] == 0.0)
    assert np.all(g[SR + fade :] == 0.0)
    rise = g[SR // 2 - fade : SR // 2]
    fall = g[SR : SR + fade]
    assert np.all(np.diff(rise) >= 0) and np.all(np.diff(fall) <= 0)
    # linear: constant slope of 1/fade (within rounding)
    assert np.allclose(np.diff(rise), 1.0 / fade, atol=1e-9)
    assert np.allclose(np.diff(fall), -1.0 / fade, atol=1e-9)


def test_ramp_region_is_a_convex_mix_of_source_and_instrumental():
    source, instrumental = _signals(stereo=False)
    n = len(source)
    vocals = _vocals_burst(n, 1.0, 2.0)
    bed = gate_bed(source, instrumental, vocals, SR, threshold_db=-45, hold_ms=150, fade_ms=150)
    # somewhere in the rising ramp: between the two inputs (per sample), neither of them
    ramp = slice(int((1.0 - 0.28) * SR), int((1.0 - 0.17) * SR))
    lo = np.minimum(source[ramp], instrumental[ramp])
    hi = np.maximum(source[ramp], instrumental[ramp])
    assert np.all(bed[ramp] >= lo - 1e-6) and np.all(bed[ramp] <= hi + 1e-6)
    assert not np.array_equal(bed[ramp], source[ramp])
    assert not np.array_equal(bed[ramp], instrumental[ramp])


def test_silent_vocals_leave_the_source_untouched():
    source, instrumental = _signals()
    bed = gate_bed(source, instrumental, np.zeros_like(source), SR)
    assert np.array_equal(bed, source)
    assert bed.dtype == source.dtype


def test_hold_extends_a_short_burst_on_both_sides():
    n = 2 * SR
    vocals = _vocals_burst(n, 1.0, 1.02)  # one 20 ms frame
    mask = activity_mask(vocals, SR, threshold_db=-45, hold_ms=100)
    on = np.flatnonzero(mask)
    assert on[0] == pytest.approx(int(0.9 * SR), abs=int(0.021 * SR))
    assert on[-1] == pytest.approx(int(1.12 * SR), abs=int(0.021 * SR))
    assert np.all(np.diff(on) == 1)  # one contiguous region


def test_threshold_is_relative_to_the_stem_peak():
    n = 2 * SR
    loud = _vocals_burst(n, 0.5, 0.7, level=0.5)
    quiet = _vocals_burst(n, 1.5, 1.7, level=0.5 * 10 ** (-50 / 20))  # 50 dB below
    vocals = loud + quiet
    mask = activity_mask(vocals, SR, threshold_db=-45, hold_ms=0)
    assert mask[int(0.6 * SR)]
    assert not mask[int(1.6 * SR)]
    assert activity_mask(vocals, SR, threshold_db=-55, hold_ms=0)[int(1.6 * SR)]


def test_mono_one_dimensional_input_round_trips():
    source, instrumental = _signals(stereo=False)
    vocals = _vocals_burst(len(source), 1.0, 2.0)[:, 0]
    bed = gate_bed(source[:, 0], instrumental[:, 0], vocals, SR)
    assert bed.ndim == 1 and len(bed) == len(source)
    assert np.array_equal(bed[: int(0.5 * SR)], source[: int(0.5 * SR), 0])
    assert np.array_equal(
        bed[int(1.2 * SR) : int(1.8 * SR)], instrumental[int(1.2 * SR) : int(1.8 * SR), 0]
    )


def test_shape_mismatches_are_errors():
    source, instrumental = _signals()
    with pytest.raises(ValueError):
        gate_bed(source, instrumental[:-1], np.zeros_like(source), SR)
    with pytest.raises(ValueError):
        gate_bed(source, instrumental, np.zeros(len(source) - 1), SR)
