"""Tests for parallel HTTP request limiting."""

from __future__ import annotations

import sys
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.config import load_config
from audio_intel.transcribe.service import Transcriber  # noqa: E402


class ConcurrentRequestsTest(unittest.TestCase):
    def test_whisper_pool_sized_for_concurrent_requests(self) -> None:
        cfg = replace(
            load_config(),
            whisper_parallel_workers=2,
            max_concurrent_requests=3,
            aed_enabled=False,
        )

        with (
            mock.patch("audio_intel.transcribe.service.WhisperEngine") as engine_cls,
            mock.patch("audio_intel.transcribe.service.VadPipeline") as vad_cls,
        ):
            vad = mock.MagicMock()
            vad_cls.from_server.return_value = vad
            Transcriber(cfg)

        engine_cls.return_value.ensure_workers.assert_called_once_with(6)
        vad.ensure_ready.assert_called_once()

    def test_semaphore_limits_active_runs(self) -> None:
        cfg = replace(load_config(), max_concurrent_requests=2, aed_enabled=False)

        with (
            mock.patch("audio_intel.transcribe.service.WhisperEngine"),
            mock.patch("audio_intel.transcribe.service.VadPipeline") as vad_cls,
        ):
            vad = mock.MagicMock()
            vad_cls.from_server.return_value = vad
            transcriber = Transcriber(cfg)

        gate = threading.Event()
        release = threading.Event()
        active = threading.Lock()
        active_count = 0
        peak_active = 0

        def slow_run(*args, **kwargs):
            nonlocal active_count, peak_active
            with active:
                active_count += 1
                peak_active = max(peak_active, active_count)
            gate.set()
            release.wait(timeout=5.0)
            with active:
                active_count -= 1
            return {"text": "", "languages": [], "confidence": 0.0, "duration": 0.0, "segments": []}

        with mock.patch.object(transcriber, "_run", side_effect=slow_run):
            threads = [
                threading.Thread(target=transcriber.transcribe, args=("a.wav",)) for _ in range(3)
            ]
            for thread in threads:
                thread.start()

            self.assertTrue(gate.wait(timeout=5.0), "expected two concurrent runs to start")
            with active:
                self.assertEqual(active_count, 2)
                self.assertEqual(peak_active, 2)

            release.set()
            for thread in threads:
                thread.join(timeout=5.0)
                self.assertFalse(thread.is_alive())

        self.assertEqual(peak_active, 2)


if __name__ == "__main__":
    unittest.main()
