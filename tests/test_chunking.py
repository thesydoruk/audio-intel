"""Unit tests for VAD chunk grouping (dependency-free: chunking imports stdlib only).

Run: ``python -m pytest tests/test_chunking.py``
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.audio.chunking import build_chunks  # noqa: E402

SR = 16000


def region(start_s: float, end_s: float) -> dict:
    return {"start": int(start_s * SR), "end": int(end_s * SR)}


class BuildChunksTest(unittest.TestCase):
    def test_empty_speech_returns_no_chunks(self):
        self.assertEqual(build_chunks([], SR, total_samples=0, max_chunk_s=28, pad_s=0.2), [])

    def test_merges_close_regions_within_max(self):
        speech = [region(0.0, 1.0), region(1.5, 2.0)]
        chunks = build_chunks(speech, SR, total_samples=int(3 * SR), max_chunk_s=28, pad_s=0.2)
        self.assertEqual(len(chunks), 1)
        # Padded by 0.2s, clamped to [0, 3].
        self.assertAlmostEqual(chunks[0].start, 0.0, places=3)
        self.assertAlmostEqual(chunks[0].end, 2.2, places=3)

    def test_splits_when_span_exceeds_max(self):
        speech = [region(0.0, 1.5), region(1.8, 3.5)]
        chunks = build_chunks(speech, SR, total_samples=int(4 * SR), max_chunk_s=2, pad_s=0.0)
        self.assertEqual(len(chunks), 2)
        self.assertAlmostEqual(chunks[0].start, 0.0, places=3)
        self.assertAlmostEqual(chunks[0].end, 1.5, places=3)
        self.assertAlmostEqual(chunks[1].start, 1.8, places=3)
        self.assertAlmostEqual(chunks[1].end, 3.5, places=3)

    def test_padding_clamped_to_audio_bounds(self):
        speech = [region(0.0, 1.0)]
        chunks = build_chunks(speech, SR, total_samples=int(1 * SR), max_chunk_s=28, pad_s=0.5)
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0].start, 0.0, places=3)
        self.assertAlmostEqual(chunks[0].end, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
