"""Unit tests for the priority-ordered GPU token pool."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.common.gpu_tokens import GpuTokens  # noqa: E402


class GpuTokensTests(unittest.TestCase):
    def test_zero_means_unlimited(self) -> None:
        tokens = GpuTokens(0)
        with tokens.hold(), tokens.hold(), tokens.hold():
            self.assertEqual(tokens.snapshot()["in_use"], 0)

    def test_limits_concurrency(self) -> None:
        tokens = GpuTokens(2)
        inside = threading.Semaphore(0)
        release = threading.Event()
        peak = []
        lock = threading.Lock()
        current = [0]

        def worker() -> None:
            with tokens.hold():
                with lock:
                    current[0] += 1
                    peak.append(current[0])
                inside.release()
                release.wait(5)
                with lock:
                    current[0] -= 1

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        inside.acquire(timeout=5)
        inside.acquire(timeout=5)
        self.assertEqual(tokens.snapshot(), {"max": 2, "in_use": 2, "waiting": 2})
        release.set()
        for t in threads:
            t.join(5)
        self.assertEqual(max(peak), 2)
        self.assertEqual(tokens.snapshot()["in_use"], 0)

    def test_lower_priority_number_goes_first(self) -> None:
        tokens = GpuTokens(1)
        order: list[str] = []
        holder_in = threading.Event()
        holder_release = threading.Event()
        queued = threading.Barrier(3, timeout=5)

        def holder() -> None:
            with tokens.hold(priority=0):
                holder_in.set()
                holder_release.wait(5)

        def waiter(name: str, priority: int) -> None:
            queued.wait()
            with tokens.hold(priority=priority):
                order.append(name)

        threads = [threading.Thread(target=holder)]
        threads[0].start()
        self.assertTrue(holder_in.wait(5))
        # batch arrives first, interactive second; interactive must still go first
        threads.append(threading.Thread(target=waiter, args=("batch", 1)))
        threads.append(threading.Thread(target=waiter, args=("interactive", 0)))
        threads[1].start()
        threads[2].start()
        queued.wait()
        # both are queued once waiting == 2
        for _ in range(50):
            if tokens.snapshot()["waiting"] == 2:
                break
            threading.Event().wait(0.02)
        holder_release.set()
        for t in threads:
            t.join(5)
        self.assertEqual(order, ["interactive", "batch"])


if __name__ == "__main__":
    unittest.main()
