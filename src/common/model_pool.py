"""Thread-safe pool of lazily loaded model copies.

Used for Whisper and WhisperX align models when multiple threads or worker
processes need concurrent inference without sharing a single model instance.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class ModelPool(Generic[T]):
    """Load N copies of a model and hand them out via acquire/release.

    With ``num_workers == 1`` the pool keeps a single model and skips the queue.
    With ``num_workers > 1`` callers must pair every ``acquire()`` with ``release()``.
    """

    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._models: list[T] = []
        self._queue: queue.Queue[T] | None = None
        self._init_lock = threading.Lock()

    def ensure_ready(self, num_workers: int = 1) -> None:
        """Create models up to ``num_workers`` (idempotent, thread-safe)."""
        num_workers = max(1, num_workers)
        with self._init_lock:
            while len(self._models) < num_workers:
                self._models.append(self._factory())
                if len(self._models) < num_workers:
                    time.sleep(0.25)

            if num_workers > 1:
                self._queue = queue.Queue()
                for model in self._models[:num_workers]:
                    self._queue.put(model)
            else:
                self._queue = None

    def acquire(self) -> T:
        """Borrow a model; blocks when all copies are checked out (multi-worker mode)."""
        if self._queue is not None:
            return self._queue.get()
        if not self._models:
            self.ensure_ready(1)
        return self._models[0]

    def release(self, model: T) -> None:
        """Return a model to the pool (no-op in single-worker mode)."""
        if self._queue is not None:
            self._queue.put(model)
