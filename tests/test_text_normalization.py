"""Unit tests for ASR segment text normalization."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.quality.normalization import (  # noqa: E402
    is_mostly_uppercase,
    normalize_for_matching,
    normalize_segment_text,
)


class NormalizeForMatchingTest(unittest.TestCase):
    def test_collapses_case_and_punctuation(self) -> None:
        self.assertEqual(
            normalize_for_matching("  Дякую за перегляд!  "),
            "дякую за перегляд",
        )


class IsMostlyUppercaseTest(unittest.TestCase):
    def test_detects_all_caps_cyrillic(self) -> None:
        self.assertTrue(is_mostly_uppercase("ПРЕЗИДЕНТ ЗАЯВИВ ПРО САНКЦІЇ"))

    def test_detects_all_caps_latin(self) -> None:
        self.assertTrue(is_mostly_uppercase("THE MINISTER ANNOUNCED AID"))

    def test_ignores_mixed_case(self) -> None:
        self.assertFalse(is_mostly_uppercase("Президент заявив про санкції"))

    def test_ignores_short_tokens(self) -> None:
        self.assertFalse(is_mostly_uppercase("OK"))


class NormalizeSegmentTextTest(unittest.TestCase):
    def test_lowercases_all_caps_segments(self) -> None:
        self.assertEqual(
            normalize_segment_text("  ПРЕЗИДЕНТ ЗАЯВИВ ПРО САНКЦІЇ  "),
            "президент заявив про санкції",
        )

    def test_keeps_mixed_case_segments(self) -> None:
        text = "Президент заявив про нові санкції."
        self.assertEqual(normalize_segment_text(text), text)

    def test_empty_returns_empty(self) -> None:
        self.assertEqual(normalize_segment_text("   "), "")


if __name__ == "__main__":
    unittest.main()
