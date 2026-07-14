"""Unit tests for streaming Silero VAD helpers."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.vad.regions import SpeechRegion
from audio_intel.vad.streaming import (  # noqa: E402
    SILERO_WINDOW_SAMPLES,
    filter_min_speech_duration,
    iter_vad_windows,
    regions_from_vad_iterator_events,
    total_samples_from_duration,
)


def _write_pcm_wav(path: str, samples: np.ndarray, sample_rate: int = 16000) -> None:
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        handle.writeframes(pcm.tobytes())


class StreamingVadHelpersTest(unittest.TestCase):
    def test_total_samples_from_duration(self) -> None:
        self.assertEqual(total_samples_from_duration(1.0, 16000), 16000)

    def test_filter_min_speech_duration(self) -> None:
        regions = [
            SpeechRegion(start=0.0, end=0.1),
            SpeechRegion(start=1.0, end=2.0),
        ]
        kept = filter_min_speech_duration(regions, min_speech_duration_ms=250)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0].start, 1.0)

    def test_regions_from_vad_iterator_events_closes_trailing_speech(self) -> None:
        regions = regions_from_vad_iterator_events(
            [{"start": 1.0}, {"end": 2.0}, {"start": 5.0}],
            duration_s=10.0,
            speech_active_at_end=True,
        )
        self.assertEqual(len(regions), 2)
        self.assertAlmostEqual(regions[0].end, 2.0)
        self.assertAlmostEqual(regions[1].start, 5.0)
        self.assertAlmostEqual(regions[1].end, 10.0)

    def test_iter_vad_windows_reads_without_full_decode(self) -> None:
        sample_rate = 16000
        total_samples = SILERO_WINDOW_SAMPLES * 10 + 100
        samples = np.random.default_rng(0).standard_normal(total_samples).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            _write_pcm_wav(wav_path, samples, sample_rate=sample_rate)
            windows = list(
                iter_vad_windows(
                    wav_path,
                    sample_rate=sample_rate,
                    total_samples=total_samples,
                    window_samples=SILERO_WINDOW_SAMPLES,
                    block_duration_s=0.05,
                )
            )
        finally:
            os.unlink(wav_path)

        self.assertEqual(len(windows), 11)
        self.assertEqual(windows[0].shape[0], SILERO_WINDOW_SAMPLES)
        self.assertEqual(windows[-1].shape[0], SILERO_WINDOW_SAMPLES)

    def test_iter_vad_windows_stops_when_decode_returns_empty(self) -> None:
        """Oversized total_samples must not spin when reads hit EOF."""
        sample_rate = 16000
        real_samples = SILERO_WINDOW_SAMPLES * 2
        samples = np.random.default_rng(1).standard_normal(real_samples).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            _write_pcm_wav(wav_path, samples, sample_rate=sample_rate)
            windows = list(
                iter_vad_windows(
                    wav_path,
                    sample_rate=sample_rate,
                    total_samples=real_samples + SILERO_WINDOW_SAMPLES * 50,
                    window_samples=SILERO_WINDOW_SAMPLES,
                    block_duration_s=0.05,
                )
            )
        finally:
            os.unlink(wav_path)

        self.assertGreaterEqual(len(windows), 2)
        self.assertLessEqual(len(windows), 4)


if __name__ == "__main__":
    unittest.main()
