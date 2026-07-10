"""Unit tests for per-request post-ASR feature flags."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.config import load_config
from audio_intel.transcribe.service import Transcriber  # noqa: E402


class RequestFeatureFlagsTest(unittest.TestCase):
    def test_run_skips_post_asr_when_disabled(self) -> None:
        cfg = load_config()

        with (
            mock.patch("audio_intel.transcribe.service.WhisperEngine") as engine_cls,
            mock.patch("audio_intel.transcribe.service.VadPipeline"),
            mock.patch("audio_intel.transcribe.service.AudioEventDetector"),
            mock.patch("audio_intel.transcribe.service.SpeakerDiarizer"),
            mock.patch("audio_intel.transcribe.service.SpeakerEmbedder"),
        ):
            transcriber = Transcriber(cfg)
            transcriber.aed = mock.MagicMock()
            transcriber.diarizer = mock.MagicMock()

            with (
                mock.patch.object(transcriber._vad, "prepare_chunks_for_server") as prepare_chunks,
                mock.patch.object(transcriber, "_transcribe_chunks") as transcribe_chunks,
                mock.patch.object(transcriber, "_apply_alignment") as apply_alignment,
                mock.patch.object(transcriber, "_apply_diarization") as apply_diarization,
                mock.patch.object(transcriber, "_detect_sounds") as detect_sounds,
                mock.patch("audio_intel.transcribe.service.MediaWorkspace.prepare") as prepare,
            ):
                media = mock.MagicMock()
                media.duration_s = 12.0
                media.whisper_path = "whisper.wav"
                prepare.return_value.__enter__.return_value = media

                prepare_chunks.return_value = [mock.MagicMock(start=0.0, end=1.0)]
                transcribe_chunks.return_value = (
                    [{"kind": "speech", "start": 0.0, "end": 1.0, "text": "hi", "confidence": 0.9}],
                    ["hi"],
                    ["en"],
                )

                transcriber.transcribe(
                    "clip.wav",
                    align=False,
                    diarize=False,
                    sound_events=False,
                )

                engine_cls.return_value.ensure_workers.assert_called_once()
                apply_alignment.assert_not_called()
                apply_diarization.assert_not_called()
                detect_sounds.assert_not_called()
                prepare.assert_called_once()
                self.assertFalse(prepare.call_args.kwargs["need_aed"])

    def test_run_parallelizes_diarization_and_sound_events(self) -> None:
        cfg = load_config()

        with (
            mock.patch("audio_intel.transcribe.service.WhisperEngine"),
            mock.patch("audio_intel.transcribe.service.VadPipeline"),
            mock.patch("audio_intel.transcribe.service.AudioEventDetector"),
            mock.patch("audio_intel.transcribe.service.SpeakerDiarizer"),
            mock.patch("audio_intel.transcribe.service.SpeakerEmbedder"),
            mock.patch("audio_intel.transcribe.service.ThreadPoolExecutor") as pool_cls,
        ):
            transcriber = Transcriber(cfg)
            transcriber.aed = mock.MagicMock()
            transcriber.diarizer = mock.MagicMock()

            pool = mock.MagicMock()
            pool_cls.return_value.__enter__.return_value = pool
            diar_future = mock.MagicMock()
            aed_future = mock.MagicMock()
            diar_future.result.return_value = (
                [{"kind": "speech", "start": 0.0, "end": 1.0, "text": "hi", "speaker_id": "S1"}],
                [{"id": "S1"}],
            )
            aed_future.result.return_value = (
                [{"kind": "sound", "start": 2.0, "end": 3.0, "label": "Dog"}],
                [],
            )
            pool.submit.side_effect = [diar_future, aed_future]

            with (
                mock.patch.object(transcriber._vad, "prepare_chunks_for_server") as prepare_chunks,
                mock.patch.object(transcriber, "_transcribe_chunks") as transcribe_chunks,
                mock.patch.object(transcriber, "_apply_alignment"),
                mock.patch.object(transcriber, "_apply_diarization") as apply_diarization,
                mock.patch.object(transcriber, "_detect_sounds") as detect_sounds,
                mock.patch("audio_intel.transcribe.service.MediaWorkspace.prepare") as prepare,
                mock.patch(
                    "audio_intel.transcribe.service._release_accelerator_memory"
                ) as release_mem,
            ):
                media = mock.MagicMock()
                media.duration_s = 12.0
                media.whisper_path = "whisper.wav"
                prepare.return_value.__enter__.return_value = media

                prepare_chunks.return_value = [mock.MagicMock(start=0.0, end=1.0)]
                transcribe_chunks.return_value = (
                    [{"kind": "speech", "start": 0.0, "end": 1.0, "text": "hi", "confidence": 0.9}],
                    ["hi"],
                    ["en"],
                )

                result = transcriber.transcribe(
                    "clip.wav",
                    diarize=True,
                    sound_events=True,
                )

                pool_cls.assert_called_once_with(max_workers=2)
                release_mem.assert_called_once()
                apply_diarization.assert_not_called()
                detect_sounds.assert_not_called()
                pool.submit.assert_any_call(
                    apply_diarization, media, transcribe_chunks.return_value[0]
                )
                pool.submit.assert_any_call(
                    detect_sounds,
                    media,
                    None,
                    0,
                )
                self.assertEqual(result["speakers"], [{"id": "S1"}])
                self.assertEqual(len(result["segments"]), 2)


if __name__ == "__main__":
    unittest.main()
