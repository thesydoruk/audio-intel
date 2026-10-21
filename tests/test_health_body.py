"""Unit tests for the /health body."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.config import load_config  # noqa: E402
from audio_intel.server.health import build_health_body  # noqa: E402


class FakeService:
    def snapshot(self) -> dict:
        return {
            "requests": {"in_flight": 1, "queued": 0, "max": 2, "queue_max": 8},
            "gpu_tokens": {"max": 2, "in_use": 1, "waiting": 0},
            "models": {"unload_idle": False, "idle_ttl_s": 600.0, "groups": {}},
        }


class HealthBodyTests(unittest.TestCase):
    def test_keeps_the_legacy_fields_and_adds_the_snapshot(self) -> None:
        cfg = load_config()
        body = build_health_body(version="x", cfg=cfg, service=FakeService(), gpu=None)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["model"], cfg.model)
        self.assertIn("diarization_enabled", body)
        self.assertIn("alignment_enabled", body)
        self.assertIsNone(body["gpu"])
        self.assertEqual(body["requests"]["in_flight"], 1)
        self.assertEqual(body["gpu_tokens"]["max"], 2)
        self.assertIn("groups", body["models"])

    def test_answers_before_the_service_exists(self) -> None:
        body = build_health_body(version="x", cfg=load_config(), service=None, gpu=None)
        self.assertEqual(body["status"], "ok")
        self.assertNotIn("requests", body)


if __name__ == "__main__":
    unittest.main()
