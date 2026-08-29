"""Unit tests for HTTP ↔ WhisperX alignment bridge."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.align.ctc import CTCAligner
from audio_intel.audio.chunking import Chunk
from audio_intel.config import AlignmentConfig, load_config
from audio_intel.transcribe.alignment import (  # noqa: E402
    assign_segment_indices_to_chunks,
    group_speech_indices_by_language,
    merge_aligned_speech_segments,
    resolve_alignment_language,
    shift_speech_segments,
    speech_segments_to_transcript,
)
from audio_intel.transcribe.service import Transcriber
from audio_intel.types import SpeechSegment, TranscriptResult, Word


class ResolveAlignmentLanguageTest(unittest.TestCase):
    def test_prefers_configured_language(self):
        cfg = load_config()
        object.__setattr__(cfg, "alignment_language", "uk")
        self.assertEqual(resolve_alignment_language(cfg, ["en"], None), "uk")

    def test_falls_back_to_detected_language(self):
        cfg = load_config()
        object.__setattr__(cfg, "alignment_language", None)
        self.assertEqual(resolve_alignment_language(cfg, ["uk", "en"], None), "uk")


class GroupSpeechIndicesByLanguageTest(unittest.TestCase):
    def test_groups_by_segment_language_and_fallback(self):
        speech = [
            {"kind": "speech", "language": "uk", "start": 0.0, "end": 1.0, "text": "а"},
            {"kind": "speech", "language": "en", "start": 1.0, "end": 2.0, "text": "b"},
            {"kind": "speech", "start": 2.0, "end": 3.0, "text": "в"},
            {"kind": "speech", "language": "unknown", "start": 3.0, "end": 4.0, "text": "г"},
        ]
        groups = group_speech_indices_by_language(speech, "uk")
        self.assertEqual(groups["uk"], [0, 2, 3])
        self.assertEqual(groups["en"], [1])


class ShiftSpeechSegmentsTest(unittest.TestCase):
    def test_shifts_segment_and_word_times(self):
        speech = [
            {
                "kind": "speech",
                "start": 10.0,
                "end": 12.0,
                "text": "hi",
                "words": [{"word": "hi", "start": 10.2, "end": 11.8}],
            }
        ]
        local = shift_speech_segments(speech, -10.0)
        self.assertAlmostEqual(local[0]["start"], 0.0)
        self.assertAlmostEqual(local[0]["end"], 2.0)
        self.assertAlmostEqual(local[0]["words"][0]["start"], 0.2)
        restored = shift_speech_segments(local, 10.0)
        self.assertAlmostEqual(restored[0]["start"], 10.0)
        self.assertAlmostEqual(restored[0]["words"][0]["end"], 11.8)


class AssignSegmentsToChunksTest(unittest.TestCase):
    def test_assigns_by_midpoint(self):
        speech = [
            {"kind": "speech", "start": 0.5, "end": 1.5, "text": "a"},
            {"kind": "speech", "start": 30.2, "end": 31.0, "text": "b"},
        ]
        chunks = [Chunk(0.0, 28.0), Chunk(28.0, 56.0)]
        assigned = assign_segment_indices_to_chunks(speech, chunks, [0, 1])
        by_start = {chunk.start: indices for chunk, indices in assigned}
        self.assertEqual(by_start[0.0], [0])
        self.assertEqual(by_start[28.0], [1])


class AlignmentBridgeTest(unittest.TestCase):
    def test_round_trip_preserves_alignment_failed_flag(self):
        speech = [
            {
                "kind": "speech",
                "start": 0.0,
                "end": 1.0,
                "text": "привіт",
                "confidence": 0.9,
                "words": [{"word": "привіт", "start": 0.0, "end": 1.0}],
            }
        ]
        transcript = speech_segments_to_transcript(
            cfg=load_config(),
            path="clip.wav",
            speech_segments=speech,
            languages=["uk"],
            duration_sec=1.0,
            language_override=None,
        )
        aligned = TranscriptResult(
            video_id="request",
            source_audio="clip.wav",
            language="uk",
            duration_sec=1.0,
            segments=[
                SpeechSegment(
                    id=0,
                    start=0.1,
                    end=0.9,
                    text="привіт",
                    avg_logprob=-0.2,
                    no_speech_prob=0.1,
                    compression_ratio=1.0,
                    words=[
                        Word(word="при", start=0.1, end=0.4, probability=0.8),
                        Word(word="віт", start=0.4, end=0.9, probability=0.7),
                    ],
                    alignment_failed=False,
                )
            ],
            transcribed_at=transcript.transcribed_at,
            aligned_at="2026-01-01T00:00:00+00:00",
        )

        merged, failed_count = merge_aligned_speech_segments(speech, aligned)
        self.assertEqual(failed_count, 0)
        self.assertEqual(len(merged[0]["words"]), 2)
        self.assertAlmostEqual(merged[0]["start"], 0.1, places=3)
        self.assertFalse(merged[0]["alignment_failed"])

    def test_marks_segment_when_alignment_failed(self):
        speech = [
            {
                "kind": "speech",
                "start": 0.0,
                "end": 1.0,
                "text": "test",
                "confidence": 0.5,
            }
        ]
        aligned = TranscriptResult(
            video_id="request",
            source_audio="clip.wav",
            language="en",
            duration_sec=1.0,
            segments=[
                SpeechSegment(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text="test",
                    avg_logprob=-0.5,
                    no_speech_prob=0.0,
                    compression_ratio=1.0,
                    words=[],
                    alignment_failed=True,
                )
            ],
            transcribed_at="2026-01-01T00:00:00+00:00",
            aligned_at="2026-01-01T00:00:00+00:00",
        )

        merged, failed_count = merge_aligned_speech_segments(speech, aligned)
        self.assertEqual(failed_count, 1)
        self.assertTrue(merged[0]["alignment_failed"])


class MapAlignedSegmentsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.aligner = CTCAligner(AlignmentConfig(), language="en", device="cpu")
        self.original = [
            SpeechSegment(
                id=0,
                start=0.0,
                end=1.0,
                text="hello",
                avg_logprob=-0.2,
                no_speech_prob=0.0,
                compression_ratio=1.0,
            )
        ]

    def test_empty_words_marks_failed(self) -> None:
        mapped = self.aligner._map_aligned_segments(
            self.original,
            [{"start": 0.0, "end": 1.0, "text": "hello", "words": []}],
        )
        self.assertTrue(mapped[0].alignment_failed)

    def test_words_without_timestamps_mark_failed(self) -> None:
        mapped = self.aligner._map_aligned_segments(
            self.original,
            [{"start": 0.0, "end": 1.0, "text": "hello", "words": [{"word": "hello"}]}],
        )
        self.assertTrue(mapped[0].alignment_failed)

    def test_words_with_timestamps_succeed(self) -> None:
        mapped = self.aligner._map_aligned_segments(
            self.original,
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "words": [{"word": "hello", "start": 0.1, "end": 0.9, "score": 0.8}],
                }
            ],
        )
        self.assertFalse(mapped[0].alignment_failed)
        self.assertEqual(len(mapped[0].words), 1)


class ApplyAlignmentPerLanguageTest(unittest.TestCase):
    def test_aligns_each_language_group_separately(self) -> None:
        cfg = replace(load_config(), alignment_enabled=True, alignment_language=None)

        with (
            mock.patch("audio_intel.transcribe.service.WhisperEngine"),
            mock.patch("audio_intel.transcribe.service.VadPipeline"),
            mock.patch("audio_intel.transcribe.service.AudioEventDetector"),
            mock.patch("audio_intel.transcribe.service.SpeakerDiarizer"),
            mock.patch("audio_intel.transcribe.service.SpeakerEmbedder"),
            mock.patch("audio_intel.align.CTCAligner"),
        ):
            transcriber = Transcriber(cfg)

        speech = [
            {"kind": "speech", "start": 0.0, "end": 1.0, "text": "привіт", "language": "uk"},
            {"kind": "speech", "start": 1.0, "end": 2.0, "text": "hello", "language": "en"},
        ]
        calls: list[tuple[str, list[str]]] = []

        def fake_align(language: str, subset: list[dict], **_kwargs):
            calls.append((language, [seg["text"] for seg in subset]))
            merged = [{**seg, "alignment_failed": False} for seg in subset]
            return merged, 0, "2026-01-01T00:00:00+00:00"

        media = mock.MagicMock()
        media.whisper_path = "whisper.wav"
        with mock.patch.object(transcriber, "_align_segment_subset", side_effect=fake_align):
            updated, meta = transcriber._apply_alignment(
                media,
                speech,
                ["uk", "en"],
                2.0,
                None,
            )

        self.assertTrue(meta["alignment_applied"])
        self.assertEqual({lang for lang, _ in calls}, {"uk", "en"})
        self.assertEqual(updated[0]["text"], "привіт")
        self.assertEqual(updated[1]["text"], "hello")


if __name__ == "__main__":
    unittest.main()
