"""Speaker embedding helpers that must not load the full recording into RAM."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.config import load_config
from audio_intel.diarization.embeddings import SpeakerEmbedder  # noqa: E402


def _write_pcm_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        handle.writeframes(pcm.tobytes())


class SpeakerClipFromPathTest(unittest.TestCase):
    def test_clip_from_path_reads_only_budgeted_speech(self) -> None:
        sample_rate = 16000
        samples = np.linspace(-0.4, 0.4, sample_rate * 5, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            _write_pcm_wav(wav_path, samples, sample_rate)
            clip = SpeakerEmbedder._build_speaker_clip_from_path(
                wav_path,
                [(0.0, 2.0), (3.0, 5.0)],
                sample_rate=sample_rate,
                min_speech_s=0.5,
                max_clip_s=1.0,
            )
        finally:
            os.unlink(wav_path)

        assert clip is not None
        self.assertEqual(clip.shape[0], sample_rate)

    def test_clip_from_path_skips_short_speech(self) -> None:
        clip = SpeakerEmbedder._build_speaker_clip_from_path(
            "missing.wav",
            [(0.0, 0.2)],
            sample_rate=16000,
            min_speech_s=0.5,
            max_clip_s=30.0,
        )
        self.assertIsNone(clip)

    def test_embed_speakers_does_not_decode_full_file(self) -> None:
        cfg = load_config()
        embedder = SpeakerEmbedder(cfg)
        with (
            mock.patch("audio_intel.audio.decode.load_audio") as load_full,
            mock.patch.object(embedder, "_build_speaker_clip_from_path", return_value=None),
        ):
            out = embedder.embed_speakers(
                "audio.wav",
                [{"start": 0.0, "end": 4.0, "speaker_id": "spk_0"}],
                sample_rate=16000,
            )
        load_full.assert_not_called()
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main()
