"""A bounded, priority-ordered permit for GPU-bound stages.

Two models running kernels on one card at the same time barely finish sooner
than one after the other, but they do keep each other's activations in VRAM
and make every request slow at once. CPU work around them (ffmpeg, VAD,
resampling, stitching) overlaps with GPU work just fine. So every GPU-bound
stage takes a token from this pool and CPU stages run free.

The pool is priority-ordered: a waiter with a lower ``priority`` number goes
first, ties in arrival order. An interactive request (priority 0) therefore
overtakes a long batch job (priority 1) at its next stage boundary instead of
waiting out the whole job. ``max_parallel=0`` disables the limit.
"""

from __future__ import annotations

import heapq
import threading
from collections.abc import Iterator
from contextlib import contextmanager


class GpuTokens:
    """Priority semaphore for GPU stages."""

    def __init__(self, max_parallel: int) -> None:
        self.max_parallel = max(0, int(max_parallel))
        self._cond = threading.Condition()
        self._in_use = 0
        self._seq = 0
        self._waiting: list[tuple[int, int]] = []

    @contextmanager
    def hold(self, priority: int = 0) -> Iterator[None]:
        """Run the block holding one token; lower ``priority`` is served first."""
        if self.max_parallel == 0:
            yield
            return
        with self._cond:
            self._seq += 1
            ticket = (priority, self._seq)
            heapq.heappush(self._waiting, ticket)
            while self._in_use >= self.max_parallel or self._waiting[0] != ticket:
                self._cond.wait()
            heapq.heappop(self._waiting)
            self._in_use += 1
        try:
            yield
        finally:
            with self._cond:
                self._in_use -= 1
                self._cond.notify_all()

    def snapshot(self) -> dict:
        with self._cond:
            return {
                "max": self.max_parallel,
                "in_use": self._in_use,
                "waiting": len(self._waiting),
            }
