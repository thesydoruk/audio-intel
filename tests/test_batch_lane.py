"""Unit tests for the batch lane (queue bound, cancellation, accounting)."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.jobs.lane import BatchBusyError, BatchLane  # noqa: E402


class BatchLaneTests(unittest.TestCase):
    def test_runs_jobs_in_order_with_one_worker(self) -> None:
        seen: list[str] = []
        done = threading.Event()

        def run(job_id: str, cancel: threading.Event) -> None:
            seen.append(job_id)
            if len(seen) == 3:
                done.set()

        lane = BatchLane(workers=1, queue_max=8, run=run)
        lane.start()
        for job_id in ("a", "b", "c"):
            lane.submit(job_id)
        self.assertTrue(done.wait(5))
        lane.stop()
        self.assertEqual(seen, ["a", "b", "c"])

    def test_queue_bound_and_snapshot(self) -> None:
        release = threading.Event()
        started = threading.Event()

        def run(job_id: str, cancel: threading.Event) -> None:
            started.set()
            release.wait(5)

        lane = BatchLane(workers=1, queue_max=1, run=run)
        lane.start()
        lane.submit("running")
        self.assertTrue(started.wait(5))
        lane.submit("waiting")
        with self.assertRaises(BatchBusyError):
            lane.submit("too-many")
        self.assertEqual(lane.snapshot(), {"running": 1, "queued": 1, "max": 1, "queue_max": 1})
        self.assertEqual(lane.cancel("waiting"), "queued")
        self.assertEqual(lane.cancel("running"), "running")
        self.assertIsNone(lane.cancel("unknown"))
        release.set()
        lane.stop()

    def test_cancel_signals_the_running_job(self) -> None:
        cancelled = threading.Event()
        started = threading.Event()

        def run(job_id: str, cancel: threading.Event) -> None:
            started.set()
            if cancel.wait(5):
                cancelled.set()

        lane = BatchLane(workers=1, queue_max=1, run=run)
        lane.start()
        lane.submit("j")
        self.assertTrue(started.wait(5))
        self.assertTrue(lane.is_running("j"))
        lane.cancel("j")
        self.assertTrue(cancelled.wait(5))
        lane.stop()

    def test_a_crashing_job_does_not_kill_the_worker(self) -> None:
        second = threading.Event()

        def run(job_id: str, cancel: threading.Event) -> None:
            if job_id == "bad":
                raise RuntimeError("boom")
            second.set()

        lane = BatchLane(workers=1, queue_max=4, run=run)
        lane.start()
        lane.submit("bad")
        lane.submit("good")
        self.assertTrue(second.wait(5))
        lane.stop()


if __name__ == "__main__":
    unittest.main()
