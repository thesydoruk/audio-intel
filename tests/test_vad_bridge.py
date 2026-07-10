"""Unit tests for VAD region helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.audio.vad_bridge import cap_region_duration, regions_to_vad_samples  # noqa: E402
from audio_intel.vad.regions import SpeechRegion


class VadBridgeTest(unittest.TestCase):
    def test_regions_to_vad_samples(self):
        regions = [SpeechRegion(start=0.5, end=2.0)]
        samples = regions_to_vad_samples(regions, sample_rate=16000)
        self.assertEqual(samples, [{"start": 8000, "end": 32000}])

    def test_cap_region_duration_splits_long_region(self):
        regions = [SpeechRegion(start=0.0, end=60.0)]
        capped = cap_region_duration(regions, max_duration_s=28.0)
        self.assertEqual(len(capped), 3)
        self.assertAlmostEqual(capped[0].start, 0.0)
        self.assertAlmostEqual(capped[0].end, 28.0)
        self.assertAlmostEqual(capped[-1].end, 60.0)


if __name__ == "__main__":
    unittest.main()
