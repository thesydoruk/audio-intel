"""Unit tests for the post-decode hallucination filter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.audio.chunking import Chunk, build_chunks  # noqa: E402
from audio_intel.transcribe.hallucination import (  # noqa: E402
    HallucinationConfig,
    assess_segment,
    filter_segments,
    speech_coverage,
)
from audio_intel.transcribe.service import _chunk_speech_spans  # noqa: E402

CFG = HallucinationConfig(
    min_speech_coverage=0.3, max_no_speech_prob=0.8, max_no_speech_span_s=1.0, overrun_s=0.5
)


def seg(start: float, end: float, text: str, **extra) -> dict:
    return {"kind": "speech", "start": start, "end": end, "text": text, **extra}


class FeatureTests(unittest.TestCase):
    def test_coverage_counts_only_the_overlap(self) -> None:
        self.assertEqual(speech_coverage(0.0, 10.0, [(2.0, 4.0), (9.0, 12.0)]), 0.3)
        self.assertEqual(speech_coverage(0.0, 10.0, [(0.0, 10.0)]), 1.0)
        self.assertEqual(speech_coverage(0.0, 10.0, []), 0.0)
        self.assertEqual(speech_coverage(5.0, 5.0, [(0.0, 10.0)]), 0.0)


class VerdictTests(unittest.TestCase):
    def test_keeps_a_normal_segment(self) -> None:
        annotated, reason = assess_segment(
            seg(0.5, 3.0, "Привіт, мене звати Дмитро."),
            speech_spans=[(0.4, 3.1)],
            audio_end=5.9,
            cfg=CFG,
        )
        self.assertIsNone(reason)
        self.assertEqual(annotated["speech_coverage"], 1.0)

    def test_drops_text_over_music_without_vad_speech(self) -> None:
        _, reason = assess_segment(
            seg(0.0, 11.0, "Thank you.", no_speech_prob=0.71),
            speech_spans=[(3.0, 4.0)],  # the one region VAD fired on
            audio_end=11.0,
            cfg=CFG,
        )
        self.assertEqual(reason, "no_vad_speech")

    def test_drops_a_short_canned_phrase_whisper_doubts_itself(self) -> None:
        # The corpus case: "Thank you." over 0.3 s of music, VAD fired, no_speech_prob 0.87.
        _, reason = assess_segment(
            seg(90.5, 90.8, "Thank you.", no_speech_prob=0.87),
            speech_spans=[(90.4, 90.9)],
            audio_end=120.0,
            cfg=CFG,
        )
        self.assertEqual(reason, "no_speech_prob")

    def test_long_segment_survives_a_high_no_speech_prob(self) -> None:
        # A confidently decoded sentence is never dropped on no_speech_prob alone.
        _, reason = assess_segment(
            seg(
                170.0, 182.2, "Aircraft carrier is equipped with a flight deck", no_speech_prob=0.74
            ),
            speech_spans=[(170.0, 182.2)],
            audio_end=200.0,
            cfg=CFG,
        )
        self.assertIsNone(reason)

    def test_short_word_with_low_no_speech_prob_is_kept(self) -> None:
        _, reason = assess_segment(
            seg(0.0, 0.4, "Так.", no_speech_prob=0.3),
            speech_spans=[(0.0, 0.4)],
            audio_end=0.4,
            cfg=CFG,
        )
        self.assertIsNone(reason)

    def test_segment_on_the_padding_is_dropped(self) -> None:
        _, reason = assess_segment(
            seg(12.0, 29.98, "Thanks for watching!"),
            speech_spans=[(0.0, 11.0)],
            audio_end=11.0,
            cfg=CFG,
        )
        self.assertEqual(reason, "beyond_audio")

    def test_segment_running_onto_the_padding_is_clamped(self) -> None:
        annotated, reason = assess_segment(
            seg(
                8.0,
                29.98,
                "last words and padding",
                words=[
                    {"word": "last", "start": 8.0, "end": 8.4},
                    {"word": "words", "start": 8.5, "end": 9.0},
                    {"word": "and", "start": 14.0, "end": 14.5},
                    {"word": "padding", "start": 20.0, "end": 21.0},
                ],
            ),
            speech_spans=[(0.0, 11.0)],
            audio_end=11.0,
            cfg=CFG,
        )
        self.assertIsNone(reason)
        self.assertEqual(annotated["end"], 11.0)
        self.assertEqual([w["word"] for w in annotated["words"]], ["last", "words"])


class FilterTests(unittest.TestCase):
    def test_enabled_splits_kept_and_dropped(self) -> None:
        kept, dropped = filter_segments(
            [seg(0.0, 2.0, "справжня мова"), seg(5.0, 15.0, "Thank you.")],
            speech_spans=[(0.0, 2.0)],
            audio_end=15.0,
            cfg=CFG,
        )
        self.assertEqual([s["text"] for s in kept], ["справжня мова"])
        self.assertEqual(dropped[0]["hallucination_reason"], "no_vad_speech")

    def test_dry_run_keeps_everything_but_annotates(self) -> None:
        kept, dropped = filter_segments(
            [seg(5.0, 15.0, "Thank you.")],
            speech_spans=[(0.0, 2.0)],
            audio_end=15.0,
            cfg=HallucinationConfig(enabled=False),
        )
        self.assertEqual(dropped, [])
        self.assertEqual(kept[0]["hallucination_reason"], "no_vad_speech")


class ChunkSpansTests(unittest.TestCase):
    def test_build_chunks_keeps_the_vad_regions_of_each_chunk(self) -> None:
        sr = 16000
        regions = [
            {"start": 0, "end": 2 * sr},
            {"start": 3 * sr, "end": 5 * sr},
            {"start": 40 * sr, "end": 42 * sr},
        ]
        chunks = build_chunks(regions, sr, 60 * sr, max_chunk_s=28.0, pad_s=0.2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].speech, ((0.0, 2.0), (3.0, 5.0)))
        self.assertEqual(chunks[1].speech, ((40.0, 42.0),))
        # Padding widens the chunk, not the speech regions.
        self.assertEqual(chunks[1].start, 39.8)

    def test_chunk_without_regions_counts_as_all_speech(self) -> None:
        self.assertEqual(_chunk_speech_spans(Chunk(1.0, 4.0)), ((1.0, 4.0),))
        self.assertEqual(_chunk_speech_spans(Chunk(1.0, 4.0, ((1.5, 2.0),))), ((1.5, 2.0),))


if __name__ == "__main__":
    unittest.main()
