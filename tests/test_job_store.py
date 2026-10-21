"""Unit tests for the SQLite job store."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.jobs.store import JobStore  # noqa: E402


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class JobStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.store = JobStore(self.tmp.name, result_ttl_s=3600, clock=self.clock)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_create_get_and_lifecycle(self) -> None:
        job = self.store.create("separate", {"language": "uk"}, "clip.wav")
        self.assertEqual(job.status, "queued")
        self.assertTrue(self.store.job_dir(job.id).is_dir())
        self.store.mark_running(job.id)
        self.store.set_progress(job.id, "transcribing 1/2", 0.5)
        got = self.store.get(job.id)
        assert got is not None
        self.assertEqual(
            (got.status, got.stage, got.progress), ("running", "transcribing 1/2", 0.5)
        )
        self.store.write_result(job.id, {"text": "hi"})
        self.store.finish(job.id, "done")
        got = self.store.get(job.id)
        assert got is not None
        self.assertTrue(got.finished)
        self.assertEqual(got.expires_at, self.clock.now + 3600)
        self.assertEqual(self.store.read_result(job.id), {"text": "hi"})
        self.assertEqual(got.to_public()["status"], "done")

    def test_failed_job_keeps_the_error(self) -> None:
        job = self.store.create("separate", {}, "x.wav")
        self.store.finish(job.id, "failed", error="RuntimeError: boom")
        got = self.store.get(job.id)
        assert got is not None
        self.assertEqual(got.error, "RuntimeError: boom")
        self.assertEqual(got.progress, 0.0)

    def test_purge_removes_expired_rows_and_files(self) -> None:
        old = self.store.create("separate", {}, "a.wav")
        self.store.finish(old, "done") if False else self.store.finish(old.id, "done")
        fresh = self.store.create("separate", {}, "b.wav")
        self.clock.now += 3601
        self.store.finish(fresh.id, "done")
        purged = self.store.purge_expired()
        self.assertEqual(purged, [old.id])
        self.assertIsNone(self.store.get(old.id))
        self.assertFalse(self.store.job_dir(old.id).exists())
        self.assertIsNotNone(self.store.get(fresh.id))

    def test_restart_fails_jobs_left_running(self) -> None:
        job = self.store.create("separate", {}, "a.wav")
        self.store.mark_running(job.id)
        self.store.close()
        reopened = JobStore(self.tmp.name, result_ttl_s=3600, clock=self.clock)
        try:
            got = reopened.get(job.id)
            assert got is not None
            self.assertEqual(got.status, "failed")
            self.assertIn("restarted", got.error or "")
        finally:
            reopened.close()

    def test_counts_and_list_order(self) -> None:
        first = self.store.create("separate", {}, "a.wav")
        self.clock.now += 1
        second = self.store.create("separate", {}, "b.wav")
        self.store.finish(first.id, "done")
        self.assertEqual(self.store.counts(), {"done": 1, "queued": 1})
        self.assertEqual([j.id for j in self.store.list()], [second.id, first.id])
        self.assertTrue(self.store.delete(second.id))
        self.assertFalse(self.store.delete(second.id))


if __name__ == "__main__":
    unittest.main()
