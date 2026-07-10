"""Unit tests for AED window planning and parallel orchestration."""

from __future__ import annotations

import sys
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.config import load_config
from audio_intel.events.panns import (  # noqa: E402
    AedWindowJob,
    AudioEventDetector,
    merge_debug_peaks,
    plan_aed_windows,
    trim_window_framewise_probs,
)


class PlanAedWindowsTest(unittest.TestCase):
    def test_short_audio_returns_single_window(self) -> None:
        sr = 32000
        jobs = plan_aed_windows(
            30 * sr,
            window_samples=60 * sr,
            overlap_samples=1 * sr,
            min_tail_samples=int(0.5 * sr),
        )
        self.assertEqual(jobs, [AedWindowJob(index=0, start=0, end=30 * sr, total_samples=30 * sr)])

    def test_long_audio_splits_with_overlap(self) -> None:
        sr = 32000
        total = 130 * sr
        window = 60 * sr
        overlap = 1 * sr
        jobs = plan_aed_windows(
            total,
            window_samples=window,
            overlap_samples=overlap,
            min_tail_samples=int(0.5 * sr),
        )
        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0].start, 0)
        self.assertEqual(jobs[0].end, 60 * sr)
        self.assertEqual(jobs[1].start, 59 * sr)
        self.assertEqual(jobs[-1].end, total)


class TrimWindowProbsTest(unittest.TestCase):
    def test_trims_overlap_margins(self) -> None:
        probs = np.ones((2000, 3), dtype=np.float32)
        trimmed, duration_s, offset_s = trim_window_framewise_probs(
            probs,
            segment_samples=60000,
            start_sample=59000,
            end_sample=119000,
            total_samples=130000,
            overlap_half_samples=16000,
            sample_rate=32000,
        )
        self.assertEqual(trimmed.shape[0], 934)
        self.assertGreater(duration_s, 0.0)
        self.assertGreater(offset_s, 1.8)


class MergeDebugPeaksTest(unittest.TestCase):
    def test_keeps_highest_score_per_class(self) -> None:
        merged = merge_debug_peaks(
            [
                {1: {"label": "Dog", "index": 1, "score": 0.4, "at": 1.0}},
                {1: {"label": "Dog", "index": 1, "score": 0.9, "at": 70.0}},
            ]
        )
        self.assertEqual(merged[1]["score"], 0.9)


def _completed_future(result) -> Future:
    future: Future = Future()
    future.set_result(result)
    return future


class AedParallelInferenceTest(unittest.TestCase):
    def test_run_windows_parallel_uses_thread_pool(self) -> None:
        cfg = load_config()
        detector = AudioEventDetector(cfg)
        jobs = [
            AedWindowJob(index=0, start=0, end=32000, total_samples=96000),
            AedWindowJob(index=1, start=31000, end=63000, total_samples=96000),
            AedWindowJob(index=2, start=62000, end=96000, total_samples=96000),
        ]

        with mock.patch.object(
            detector,
            "_run_window",
            side_effect=lambda **kwargs: ([{"kind": "sound", "start": 0.0, "end": 1.0}], {}),
        ) as run_window:
            with mock.patch("audio_intel.events.panns.ThreadPoolExecutor") as pool_cls:
                pool = mock.MagicMock()
                pool_cls.return_value.__enter__.return_value = pool

                def submit(func, **kwargs):
                    return _completed_future(func(**kwargs))

                pool.submit.side_effect = submit

                results = detector._run_windows_parallel(
                    path="audio.wav",
                    waveform=None,
                    jobs=jobs,
                    params=mock.MagicMock(),
                    sample_rate=32000,
                    overlap_half_samples=16000,
                    collect_debug=False,
                    workers=2,
                )

        pool_cls.assert_called_once_with(max_workers=2)
        self.assertEqual(run_window.call_count, 3)
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
