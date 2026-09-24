"""Speaker-embed request parsing and pyannote 3.x / 4.x output handling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.diarization.pyannote import _speaker_annotation  # noqa: E402
from audio_intel.server.transcription import (  # noqa: E402
    MAX_EMBED_SPEAKERS,
    parse_speaker_spans,
)


class ParseSpeakerSpansTest(unittest.TestCase):
    def test_parses_ids_and_spans(self) -> None:
        out = parse_speaker_spans('{"spk_0": [[0, 1.5], [3.0, 4]], "spk_1": []}')
        self.assertEqual(out, {"spk_0": [(0.0, 1.5), (3.0, 4.0)], "spk_1": []})

    def test_rejects_malformed_input(self) -> None:
        bad = [
            "not json",
            "[]",
            "{}",
            '{"a": [[1]]}',
            '{"a": [[2, 1]]}',
            '{"a": [[-1, 1]]}',
            '{"a": [["0", "1"]]}',
            '{"a": [[true, 1]]}',
            '{"a": {"start": 0}}',
            '{"a": [[0, NaN]]}',
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_speaker_spans(raw)

    def test_caps_speaker_count(self) -> None:
        many = "{" + ",".join(f'"s{i}": []' for i in range(MAX_EMBED_SPEAKERS + 1)) + "}"
        with self.assertRaises(ValueError):
            parse_speaker_spans(many)


class SpeakerAnnotationTest(unittest.TestCase):
    def test_prefers_exclusive_turns_from_pyannote_4(self) -> None:
        output = SimpleNamespace(
            speaker_diarization="regular", exclusive_speaker_diarization="excl"
        )
        self.assertEqual(_speaker_annotation(output), "excl")

    def test_falls_back_to_regular_turns(self) -> None:
        output = SimpleNamespace(speaker_diarization="regular")
        self.assertEqual(_speaker_annotation(output), "regular")

    def test_pyannote_3_annotation_passes_through(self) -> None:
        annotation = object()
        self.assertIs(_speaker_annotation(annotation), annotation)


if __name__ == "__main__":
    unittest.main()
