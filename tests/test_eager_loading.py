"""Tests for Whisper-first startup and lazy post-ASR model loading."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.config import load_config
from audio_intel.transcribe.alignment import alignment_preload_languages
from audio_intel.transcribe.service import Transcriber  # noqa: E402


class AlignmentPreloadLanguagesTest(unittest.TestCase):
    def test_uses_alignment_language_when_set(self) -> None:
        cfg = replace(load_config(), alignment_language="uk")
        self.assertEqual(alignment_preload_languages(cfg), ("uk",))

    def test_uses_forced_whisper_language_when_alignment_unset(self) -> None:
        cfg = replace(load_config(), alignment_language=None, forced_language="de")
        self.assertEqual(alignment_preload_languages(cfg), ("de",))

    def test_defaults_to_english(self) -> None:
        cfg = replace(load_config(), alignment_language=None, forced_language=None)
        self.assertEqual(alignment_preload_languages(cfg), ("en",))


class TranscriberLazyPostAsrLoadingTest(unittest.TestCase):
    def test_startup_loads_whisper_and_vad_only(self) -> None:
        cfg = replace(
            load_config(),
            aed_enabled=True,
            diarization_enabled=True,
            alignment_enabled=True,
            alignment_language="en",
        )
        vad = mock.MagicMock()
        aed = mock.MagicMock()
        diarizer = mock.MagicMock()
        embedder = mock.MagicMock()

        with (
            mock.patch("audio_intel.transcribe.service.WhisperEngine") as engine_cls,
            mock.patch("audio_intel.transcribe.service.VadPipeline") as vad_cls,
            mock.patch("audio_intel.transcribe.service.AudioEventDetector", return_value=aed),
            mock.patch("audio_intel.transcribe.service.SpeakerDiarizer", return_value=diarizer),
            mock.patch("audio_intel.transcribe.service.SpeakerEmbedder", return_value=embedder),
            mock.patch("audio_intel.align.CTCAligner") as aligner_cls,
        ):
            vad_cls.from_server.return_value = vad
            aligner = mock.MagicMock()
            aligner_cls.return_value = aligner

            transcriber = Transcriber(cfg)

        engine_cls.return_value.ensure_workers.assert_called_once()
        vad.ensure_ready.assert_called_once()
        aed.ensure_ready.assert_not_called()
        diarizer.ensure_ready.assert_not_called()
        embedder.ensure_ready.assert_not_called()
        aligner.ensure_ready.assert_called_once()
        self.assertIn("en", transcriber._aligners)
        self.assertIs(transcriber.aed, aed)
        self.assertIs(transcriber.diarizer, diarizer)

    def test_post_asr_helpers_load_models_once(self) -> None:
        cfg = replace(load_config(), aed_enabled=True, diarization_enabled=True)
        vad = mock.MagicMock()
        aed = mock.MagicMock()
        diarizer = mock.MagicMock()
        embedder = mock.MagicMock()

        with (
            mock.patch("audio_intel.transcribe.service.WhisperEngine"),
            mock.patch("audio_intel.transcribe.service.VadPipeline") as vad_cls,
            mock.patch("audio_intel.transcribe.service.AudioEventDetector", return_value=aed),
            mock.patch("audio_intel.transcribe.service.SpeakerDiarizer", return_value=diarizer),
            mock.patch("audio_intel.transcribe.service.SpeakerEmbedder", return_value=embedder),
        ):
            vad_cls.from_server.return_value = vad
            transcriber = Transcriber(cfg)

        transcriber._ensure_aed_ready()
        transcriber._ensure_aed_ready()
        aed.ensure_ready.assert_called_once()

        transcriber._ensure_speakers_ready()
        transcriber._ensure_speakers_ready()
        diarizer.ensure_ready.assert_called_once()
        embedder.ensure_ready.assert_called_once()


if __name__ == "__main__":
    unittest.main()
