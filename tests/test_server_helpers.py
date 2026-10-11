"""Tests for HTTP request helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.server.transcription import normalize_language  # noqa: E402


class NormalizeLanguageTest(unittest.TestCase):
    def test_auto_and_empty_mean_detect(self) -> None:
        self.assertIsNone(normalize_language(None))
        self.assertIsNone(normalize_language(""))
        self.assertIsNone(normalize_language("  auto "))

    def test_known_codes_are_lowercased(self) -> None:
        self.assertEqual(normalize_language("UK"), "uk")
        self.assertEqual(normalize_language(" en "), "en")

    def test_unknown_code_is_a_client_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            normalize_language("klingon")
        self.assertIn("klingon", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
