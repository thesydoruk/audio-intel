"""Thread-safe pool of lazily loaded model copies.

Used for Whisper, Silero VAD, PANNs and WhisperX align models when several
threads need concurrent inference without sharing one model instance.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class ModelPool(Generic[T]):
    """Load N copies of a model and hand them out via acquire/release.

    Every ``acquire()`` must be paired with ``release()``. ``acquire()`` blocks
    while all copies are checked out, so a single-copy pool also serializes its
    callers instead of letting two threads drive the same model at once.
    """

    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._models: list[T] = []
        self._queue: queue.Queue[T] = queue.Queue()
        self._init_lock = threading.Lock()

    @property
    def size(self) -> int:
        """Number of loaded copies."""
        return len(self._models)

    def ensure_ready(self, num_workers: int = 1) -> None:
        """Grow the pool to ``num_workers`` copies (idempotent, thread-safe).

        Copies already checked out stay valid: new ones are only ever added.
        """
        num_workers = max(1, num_workers)
        with self._init_lock:
            while len(self._models) < num_workers:
                model = self._factory()
                self._models.append(model)
                self._queue.put(model)

    def acquire(self) -> T:
        """Borrow a model; blocks when all copies are checked out."""
        if not self._models:
            self.ensure_ready(1)
        return self._queue.get()

    def release(self, model: T) -> None:
        """Return a model to the pool."""
        self._queue.put(model)

    def clear(self) -> None:
        """Drop every copy. Only call when no copy is checked out (see ModelGroup)."""
        with self._init_lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._models.clear()
