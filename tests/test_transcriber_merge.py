"""Unit tests for parallel chunk merge helpers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.transcribe.service import (  # noqa: E402
    _ChunkTranscription,
    _merge_chunk_transcriptions,
)


class TranscriberMergeTests(unittest.TestCase):
    def test_merge_chunk_transcriptions_preserves_timeline_order(self) -> None:
        first = _ChunkTranscription(
            chunk_index=0,
            segments=[{"kind": "speech", "start": 0.0, "end": 1.0, "text": "one"}],
            text_parts=["one"],
            languages=["uk"],
        )
        second = _ChunkTranscription(
            chunk_index=1,
            segments=[{"kind": "speech", "start": 30.0, "end": 31.0, "text": "two"}],
            text_parts=["two"],
            languages=["en"],
        )

        segments, text_parts, languages = _merge_chunk_transcriptions([second, first])

        self.assertEqual([segment["text"] for segment in segments], ["one", "two"])
        self.assertEqual(text_parts, ["one", "two"])
        self.assertEqual(languages, ["uk", "en"])


if __name__ == "__main__":
    unittest.main()
