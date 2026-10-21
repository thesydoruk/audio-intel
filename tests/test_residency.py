"""Unit tests for model groups and the idle-unload janitor."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.common.residency import ModelGroup, ResidencyManager  # noqa: E402


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Stack:
    def __init__(self) -> None:
        self.loads = 0
        self.unloads = 0

    def load(self) -> None:
        self.loads += 1

    def unload(self) -> None:
        self.unloads += 1


class ModelGroupTests(unittest.TestCase):
    def test_loads_once_on_first_use(self) -> None:
        stack = Stack()
        group = ModelGroup("aed", load=stack.load, unload=stack.unload, clock=FakeClock())
        with group.use():
            self.assertTrue(group.loaded)
            self.assertEqual(group.users, 1)
        with group.use():
            pass
        self.assertEqual(stack.loads, 1)
        self.assertEqual(group.users, 0)

    def test_failed_load_is_retried_next_time(self) -> None:
        calls = []

        def flaky() -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("gated model")

        group = ModelGroup("diar", load=flaky, unload=lambda: None, clock=FakeClock())
        with self.assertRaises(RuntimeError):
            with group.use():
                pass
        self.assertFalse(group.loaded)
        with group.use():
            self.assertTrue(group.loaded)
        self.assertEqual(len(calls), 2)

    def test_unloads_only_when_idle_past_ttl_and_unused(self) -> None:
        clock = FakeClock()
        stack = Stack()
        group = ModelGroup("aed", load=stack.load, unload=stack.unload, clock=clock)
        with group.use():
            clock.now += 1000
            self.assertFalse(group.maybe_unload(10))  # in use
        self.assertFalse(group.maybe_unload(10))  # just released
        clock.now += 11
        self.assertTrue(group.maybe_unload(10))
        self.assertFalse(group.loaded)
        self.assertEqual(stack.unloads, 1)
        self.assertFalse(group.maybe_unload(10))  # nothing left to unload
        with group.use():
            pass
        self.assertEqual(stack.loads, 2)

    def test_pinned_group_never_unloads(self) -> None:
        clock = FakeClock()
        stack = Stack()
        group = ModelGroup("asr", load=stack.load, unload=stack.unload, pinned=True, clock=clock)
        group.mark_loaded()
        clock.now += 10_000
        self.assertFalse(group.maybe_unload(1))
        self.assertTrue(group.loaded)

    def test_concurrent_users_wait_for_one_load(self) -> None:
        started = threading.Event()
        release = threading.Event()
        loads = []

        def slow_load() -> None:
            loads.append(1)
            started.set()
            release.wait(5)

        group = ModelGroup("x", load=slow_load, unload=lambda: None)
        seen: list[int] = []

        def user() -> None:
            with group.use():
                seen.append(group.users)

        threads = [threading.Thread(target=user) for _ in range(3)]
        for t in threads:
            t.start()
        self.assertTrue(started.wait(5))
        release.set()
        for t in threads:
            t.join(5)
        self.assertEqual(len(loads), 1)
        self.assertEqual(len(seen), 3)

    def test_snapshot_reports_state(self) -> None:
        clock = FakeClock()
        group = ModelGroup("aed", load=lambda: None, unload=lambda: None, clock=clock)
        self.assertEqual(group.snapshot()["resident"], False)
        with group.use():
            self.assertEqual(group.snapshot()["in_use"], 1)
        clock.now += 5
        snap = group.snapshot()
        self.assertTrue(snap["resident"])
        self.assertEqual(snap["idle_s"], 5.0)
        self.assertEqual(snap["loads"], 1)


class ResidencyManagerTests(unittest.TestCase):
    def test_janitor_pass_unloads_idle_groups(self) -> None:
        clock = FakeClock()
        manager = ResidencyManager(unload_idle=True, idle_ttl_s=60, clock=clock)
        a, b = Stack(), Stack()
        ga = manager.group("a", load=a.load, unload=a.unload)
        gb = manager.group("b", load=b.load, unload=b.unload)
        with ga.use():
            pass
        clock.now += 30
        with gb.use():
            pass
        clock.now += 40  # a idle 70 s, b idle 40 s
        self.assertEqual(manager.unload_idle_groups(), ["a"])
        self.assertEqual((a.unloads, b.unloads), (1, 0))
        snap = manager.snapshot()
        self.assertFalse(snap["groups"]["a"]["resident"])
        self.assertTrue(snap["groups"]["b"]["resident"])

    def test_disabled_manager_starts_no_thread(self) -> None:
        manager = ResidencyManager(unload_idle=False, idle_ttl_s=60)
        manager.start()
        self.assertIsNone(manager._thread)
        manager.stop()


if __name__ == "__main__":
    unittest.main()
