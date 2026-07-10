"""Tests for environment-driven configuration."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.config import load_config  # noqa: E402


class LoadConfigTest(unittest.TestCase):
    def test_reads_env_names(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "ASR_MODEL": "medium",
                "ASR_DEVICE": "cpu",
                "SOUND_EVENTS_ENABLED": "0",
                "SPEAKERS_ENABLED": "1",
                "WORD_ALIGN_ENABLED": "1",
                "LOG_LEVEL": "DEBUG",
                "VAD_SPEECH_THRESHOLD": "0.42",
            },
            clear=True,
        ):
            cfg = load_config()

        self.assertEqual(cfg.model, "medium")
        self.assertEqual(cfg.device, "cpu")
        self.assertFalse(cfg.aed_enabled)
        self.assertTrue(cfg.diarization_enabled)
        self.assertTrue(cfg.alignment_enabled)
        self.assertEqual(cfg.log_level, "DEBUG")
        self.assertAlmostEqual(cfg.vad_threshold, 0.42)


if __name__ == "__main__":
    unittest.main()
