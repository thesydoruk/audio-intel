"""Tests for the shared model pool."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.common.model_pool import ModelPool  # noqa: E402


class ModelPoolTest(unittest.TestCase):
    def test_single_copy_serializes_callers(self) -> None:
        pool: ModelPool[object] = ModelPool(object)
        pool.ensure_ready(1)

        first = pool.acquire()
        got_second = threading.Event()

        def take() -> None:
            pool.acquire()
            got_second.set()

        thread = threading.Thread(target=take, daemon=True)
        thread.start()
        self.assertFalse(got_second.wait(0.1))
        pool.release(first)
        self.assertTrue(got_second.wait(1.0))

    def test_growing_keeps_checked_out_copies(self) -> None:
        pool: ModelPool[object] = ModelPool(object)
        pool.ensure_ready(1)
        borrowed = pool.acquire()
        pool.ensure_ready(2)
        self.assertEqual(pool.size, 2)
        other = pool.acquire()
        self.assertIsNot(other, borrowed)
        pool.release(other)
        pool.release(borrowed)
        self.assertEqual({id(pool.acquire()), id(pool.acquire())}, {id(borrowed), id(other)})

    def test_ensure_ready_is_idempotent(self) -> None:
        created = []
        pool: ModelPool[object] = ModelPool(lambda: created.append(object()) or created[-1])
        pool.ensure_ready(2)
        pool.ensure_ready(2)
        pool.ensure_ready(1)
        self.assertEqual(len(created), 2)


if __name__ == "__main__":
    unittest.main()
