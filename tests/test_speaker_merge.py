"""Tests for speaker ↔ segment overlap assignment."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.diarization.merge import (  # noqa: E402
    assign_speakers_to_segments,
    build_speaker_roster,
    overlap_seconds,
)


class SpeakerMergeTests(unittest.TestCase):
    def test_overlap_seconds(self):
        self.assertEqual(overlap_seconds(0.0, 5.0, 3.0, 8.0), 2.0)
        self.assertEqual(overlap_seconds(0.0, 2.0, 5.0, 8.0), 0.0)

    def test_assigns_speaker_with_max_overlap(self):
        segments = [
            {"kind": "speech", "start": 0.0, "end": 4.0, "text": "Hello"},
            {"kind": "speech", "start": 5.0, "end": 9.0, "text": "World"},
        ]
        intervals = [
            {"start": 0.0, "end": 4.5, "speaker_id": "spk_0"},
            {"start": 4.8, "end": 10.0, "speaker_id": "spk_1"},
        ]
        assigned, roster = assign_speakers_to_segments(segments, intervals)
        self.assertEqual(assigned[0]["speaker_id"], "spk_0")
        self.assertEqual(assigned[1]["speaker_id"], "spk_1")
        self.assertEqual(
            roster,
            [
                {"id": "spk_0", "speech_seconds": 4.0, "segment_count": 1},
                {"id": "spk_1", "speech_seconds": 4.0, "segment_count": 1},
            ],
        )

    def test_skips_assignment_when_no_overlap(self):
        segments = [{"kind": "speech", "start": 0.0, "end": 1.0, "text": "Hi"}]
        intervals = [{"start": 5.0, "end": 6.0, "speaker_id": "spk_0"}]
        assigned, roster = assign_speakers_to_segments(segments, intervals)
        self.assertNotIn("speaker_id", assigned[0])
        self.assertEqual(roster, [])

    def test_build_roster_aggregates_duration(self):
        segments = [
            {"start": 0.0, "end": 2.0, "speaker_id": "spk_0"},
            {"start": 2.0, "end": 5.5, "speaker_id": "spk_0"},
        ]
        self.assertEqual(
            build_speaker_roster(segments),
            [{"id": "spk_0", "speech_seconds": 5.5, "segment_count": 2}],
        )

    def test_splits_one_whisper_segment_at_speaker_change(self):
        segments = [
            {
                "kind": "speech",
                "start": 0.0,
                "end": 8.0,
                "text": "Hello there Yes",
                "confidence": 0.9,
                "words": [
                    {"start": 0.0, "end": 1.0, "word": " Hello"},
                    {"start": 1.0, "end": 2.0, "word": " there"},
                    {"start": 4.2, "end": 5.0, "word": " Yes"},
                ],
            }
        ]
        intervals = [
            {"start": 0.0, "end": 4.1, "speaker_id": "spk_0"},
            {"start": 4.1, "end": 8.0, "speaker_id": "spk_1"},
        ]
        assigned, roster = assign_speakers_to_segments(segments, intervals)
        self.assertEqual(len(assigned), 2)
        self.assertEqual(assigned[0]["speaker_id"], "spk_0")
        self.assertEqual(assigned[0]["text"], "Hello there")
        self.assertEqual(assigned[0]["start"], 0.0)
        self.assertEqual(assigned[0]["end"], 2.0)
        self.assertEqual(assigned[1]["speaker_id"], "spk_1")
        self.assertEqual(assigned[1]["text"], "Yes")
        self.assertNotIn("words", assigned[0])
        self.assertNotIn("words", assigned[1])
        self.assertEqual(
            roster,
            [
                {"id": "spk_0", "speech_seconds": 2.0, "segment_count": 1},
                {"id": "spk_1", "speech_seconds": 0.8, "segment_count": 1},
            ],
        )

    def test_rebuild_preserves_spacing_and_lowercases_all_caps(self) -> None:
        segments = [
            {
                "kind": "speech",
                "start": 0.0,
                "end": 4.0,
                "text": "ПРИВЕТ МИР",
                "words": [
                    {"start": 0.0, "end": 1.0, "word": " ПРИВЕТ"},
                    {"start": 1.0, "end": 2.0, "word": " МИР"},
                ],
            }
        ]
        intervals = [{"start": 0.0, "end": 4.0, "speaker_id": "spk_0"}]
        assigned, _ = assign_speakers_to_segments(segments, intervals)
        self.assertEqual(assigned[0]["text"], "привет мир")

    def test_rebuild_inserts_spaces_when_word_tokens_are_stripped(self) -> None:
        """Production path: segment_from_whisper strips leading spaces on words."""
        segments = [
            {
                "kind": "speech",
                "start": 0.0,
                "end": 5.0,
                "text": "Welcome everyone to another episode",
                "words": [
                    {"start": 0.0, "end": 0.5, "word": "Welcome"},
                    {"start": 0.5, "end": 1.2, "word": "everyone"},
                    {"start": 1.2, "end": 1.5, "word": "to"},
                    {"start": 1.5, "end": 2.2, "word": "another"},
                    {"start": 2.2, "end": 3.0, "word": "episode"},
                ],
            }
        ]
        intervals = [{"start": 0.0, "end": 5.0, "speaker_id": "spk_0"}]
        assigned, _ = assign_speakers_to_segments(segments, intervals)
        self.assertEqual(assigned[0]["text"], "Welcome everyone to another episode")

    def test_strips_words_when_intervals_missing(self):
        segments = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hi",
                "words": [{"start": 0.0, "end": 1.0, "word": " Hi"}],
            }
        ]
        assigned, roster = assign_speakers_to_segments(segments, [])
        self.assertEqual(len(assigned), 1)
        self.assertNotIn("words", assigned[0])
        self.assertEqual(roster, [])


if __name__ == "__main__":
    unittest.main()
