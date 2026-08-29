"""Unit tests for WAV window reads that avoid loading the full file into RAM."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.audio.decode import (  # noqa: E402
    extract_wav_window,
    load_audio_spans,
    load_audio_window,
)


def _write_pcm_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        handle.writeframes(pcm.tobytes())


class WavWindowReadTest(unittest.TestCase):
    def test_load_audio_window_reads_slice_and_is_writable(self) -> None:
        sample_rate = 16000
        samples = np.linspace(-0.5, 0.5, sample_rate * 2, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            _write_pcm_wav(wav_path, samples, sample_rate)
            window = load_audio_window(wav_path, sample_rate, 0.5, 0.25)
        finally:
            os.unlink(wav_path)

        self.assertEqual(window.dtype, np.float32)
        self.assertTrue(window.flags.writeable)
        self.assertTrue(window.flags.c_contiguous)
        self.assertEqual(window.shape[0], int(round(0.25 * sample_rate)))
        expected = samples[int(0.5 * sample_rate) : int(0.75 * sample_rate)]
        np.testing.assert_allclose(window, expected, atol=2 / 32767)

    def test_load_audio_window_past_eof_is_empty(self) -> None:
        sample_rate = 16000
        samples = np.zeros(sample_rate, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            _write_pcm_wav(wav_path, samples, sample_rate)
            window = load_audio_window(wav_path, sample_rate, 2.0, 1.0)
        finally:
            os.unlink(wav_path)

        self.assertEqual(window.size, 0)

    def test_load_audio_spans_caps_duration(self) -> None:
        sample_rate = 16000
        samples = np.arange(sample_rate * 4, dtype=np.float32) / float(sample_rate * 4)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            _write_pcm_wav(wav_path, samples, sample_rate)
            clip = load_audio_spans(
                wav_path,
                [(0.0, 1.0), (2.0, 3.5)],
                sample_rate,
                max_duration_s=1.5,
            )
        finally:
            os.unlink(wav_path)

        self.assertEqual(clip.shape[0], int(round(1.5 * sample_rate)))

    def test_extract_wav_window_copies_slice_without_full_decode(self) -> None:
        sample_rate = 16000
        samples = np.linspace(-0.9, 0.9, sample_rate * 3, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src:
            src_path = src.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as dst:
            dst_path = dst.name
        try:
            _write_pcm_wav(src_path, samples, sample_rate)
            extract_wav_window(
                src_path,
                dst_path,
                sample_rate=sample_rate,
                start_s=1.0,
                duration_s=0.5,
            )
            extracted, extracted_sr = sf.read(dst_path, dtype="float32")
        finally:
            os.unlink(src_path)
            os.unlink(dst_path)

        self.assertEqual(extracted_sr, sample_rate)
        self.assertEqual(len(extracted), int(round(0.5 * sample_rate)))
        expected = samples[sample_rate : sample_rate + sample_rate // 2]
        np.testing.assert_allclose(extracted, expected, atol=2 / 32767)


if __name__ == "__main__":
    unittest.main()
