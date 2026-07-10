"""Unit tests for HTTP ↔ WhisperX alignment bridge."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.config import load_config
from audio_intel.transcribe.alignment import (  # noqa: E402
    merge_aligned_speech_segments,
    resolve_alignment_language,
    speech_segments_to_transcript,
)
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


if __name__ == "__main__":
    unittest.main()
