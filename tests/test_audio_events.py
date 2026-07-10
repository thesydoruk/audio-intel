"""Unit tests for the pure sound-event helpers (no torch / PANNs checkpoint).

Run: ``python -m pytest tests/test_audio_events.py``
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.events.panns import (  # noqa: E402
    events_from_framewise,
    is_speech_label,
    merge_sound_events,
)

LABELS = ["Explosion", "Speech", "Music"]


def _probs(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def _detect(probs, **kw):
    base = dict(
        min_score=0.35,
        top_k=3,
        merge_gap_s=0.0,
        min_duration_s=0.05,
        exclude_speech=True,
        prompt_allowlist=(),
    )
    base.update(kw)
    return events_from_framewise(probs, 0.1, LABELS, **base)


class AudioEventsTest(unittest.TestCase):
    def test_single_run(self) -> None:
        probs = _probs([[0.0, 0, 0]] * 2 + [[0.8, 0, 0]] * 4 + [[0.0, 0, 0]] * 2)
        events = _detect(probs)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["label"], "Explosion")
        self.assertEqual(ev["index"], 0)
        self.assertAlmostEqual(ev["start"], 0.2, places=3)
        self.assertAlmostEqual(ev["end"], 0.6, places=3)
        self.assertEqual(ev["score"], 0.8)
        self.assertTrue(ev["prompt_relevant"])

    def test_min_duration_drops_short(self) -> None:
        probs = _probs([[0.8, 0, 0]] + [[0.0, 0, 0]] * 5)
        self.assertEqual(_detect(probs, min_duration_s=0.5), [])

    def test_merge_gap(self) -> None:
        probs = _probs(
            [[0.8, 0, 0], [0.8, 0, 0], [0.0, 0, 0], [0.8, 0, 0], [0.8, 0, 0], [0.0, 0, 0]]
        )
        self.assertEqual(len(_detect(probs, merge_gap_s=0.0)), 2)
        merged = _detect(probs, merge_gap_s=0.15)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["start"], 0.0, places=3)
        self.assertAlmostEqual(merged[0]["end"], 0.5, places=3)

    def test_exclude_speech(self) -> None:
        probs = _probs([[0.0, 0.9, 0]] * 4)
        self.assertEqual(_detect(probs, exclude_speech=True), [])
        kept = _detect(probs, exclude_speech=False)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["label"], "Speech")

    def test_top_k_limits_simultaneous(self) -> None:
        probs = _probs([[0.9, 0.8, 0.7]] * 4)
        events = _detect(probs, top_k=1)
        self.assertEqual([e["index"] for e in events], [0])

    def test_prompt_allowlist(self) -> None:
        probs = _probs([[0.8, 0, 0.7]] * 4)
        events = _detect(probs, prompt_allowlist=("music",))
        by_label = {e["label"]: e["prompt_relevant"] for e in events}
        self.assertFalse(by_label["Explosion"])
        self.assertTrue(by_label["Music"])

    def test_is_speech_label(self) -> None:
        self.assertTrue(is_speech_label("Male speech, man speaking"))
        self.assertTrue(is_speech_label("Conversation"))
        self.assertFalse(is_speech_label("Explosion"))

    def test_merge_sound_events_across_gap(self) -> None:
        events = [
            {
                "kind": "sound",
                "start": 0.0,
                "end": 1.0,
                "label": "Explosion",
                "index": 0,
                "score": 0.8,
            },
            {
                "kind": "sound",
                "start": 1.4,
                "end": 2.0,
                "label": "Explosion",
                "index": 0,
                "score": 0.7,
            },
            {"kind": "sound", "start": 3.0, "end": 4.0, "label": "Music", "index": 2, "score": 0.6},
        ]
        merged = merge_sound_events(events, merge_gap_s=0.6)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[0]["end"], 2.0, places=3)
        self.assertEqual(merged[1]["label"], "Music")


if __name__ == "__main__":
    unittest.main()
