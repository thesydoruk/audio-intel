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
from audio_intel.diarization.embeddings import SpeakerEmbedder, split_model_id  # noqa: E402


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


class EmbeddingModelIdTest(unittest.TestCase):
    def test_repo_with_subfolder(self) -> None:
        self.assertEqual(
            split_model_id("pyannote/speaker-diarization-community-1/embedding"),
            ("pyannote/speaker-diarization-community-1", "embedding"),
        )

    def test_plain_repo(self) -> None:
        self.assertEqual(split_model_id("pyannote/embedding"), ("pyannote/embedding", None))

    def test_local_path_is_kept_whole(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(split_model_id(root), (root, None))


class EmbeddingNormalizationTest(unittest.TestCase):
    def _embedder(self) -> SpeakerEmbedder:
        return SpeakerEmbedder(load_config())

    def test_dimension_comes_from_first_vector_when_model_does_not_say(self) -> None:
        embedder = self._embedder()
        out = embedder._normalize_embedding(np.full(256, 2.0, dtype=np.float32))
        assert out is not None
        self.assertEqual(len(out), 256)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=4)
        self.assertEqual(embedder.dimension, 256)

    def test_rejects_vector_of_another_length(self) -> None:
        embedder = self._embedder()
        embedder._dimension = 256
        self.assertIsNone(embedder._normalize_embedding(np.ones(512, dtype=np.float32)))

    def test_rejects_zero_and_non_finite_vectors(self) -> None:
        embedder = self._embedder()
        self.assertIsNone(embedder._normalize_embedding(np.zeros(256, dtype=np.float32)))
        bad = np.ones(256, dtype=np.float32)
        bad[3] = np.nan
        self.assertIsNone(embedder._normalize_embedding(bad))

    def test_model_id_follows_config(self) -> None:
        embedder = self._embedder()
        self.assertEqual(embedder.model_id, embedder.cfg.diarization_embedding_model)


if __name__ == "__main__":
    unittest.main()
