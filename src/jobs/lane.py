"""The batch lane: a bounded queue of job ids and the threads that run them.

Separate from the interactive request semaphore on purpose: a queue of
minute-long separations must never hold the slots a two-second telegram
transcription needs. The two lanes share the model pools and meet only at the
GPU tokens, where batch work carries the lower priority.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable

log = logging.getLogger("audio-intel")

JobRunner = Callable[[str, threading.Event], None]


class BatchBusyError(RuntimeError):
    """More jobs are waiting than ``BATCH_QUEUE_MAX`` allows."""


class BatchLane:
    def __init__(self, *, workers: int, queue_max: int, run: JobRunner) -> None:
        self.workers = max(1, int(workers))
        self.queue_max = max(0, int(queue_max))
        self._run = run
        self._cond = threading.Condition()
        self._queue: deque[str] = deque()
        self._running: dict[str, threading.Event] = {}
        self._stop = False
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._threads:
            return
        for index in range(self.workers):
            thread = threading.Thread(target=self._worker, name=f"batch-{index}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            for event in self._running.values():
                event.set()
            self._cond.notify_all()
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads = []

    def submit(self, job_id: str) -> None:
        with self._cond:
            if len(self._queue) >= self.queue_max:
                raise BatchBusyError(
                    f"{len(self._queue)} job(s) already waiting (BATCH_QUEUE_MAX={self.queue_max})"
                )
            self._queue.append(job_id)
            self._cond.notify()

    def cancel(self, job_id: str) -> str | None:
        """Drop a queued job or signal a running one; returns what it was, or None."""
        with self._cond:
            if job_id in self._queue:
                self._queue.remove(job_id)
                return "queued"
            event = self._running.get(job_id)
            if event is not None:
                event.set()
                return "running"
        return None

    def is_running(self, job_id: str) -> bool:
        with self._cond:
            return job_id in self._running

    def snapshot(self) -> dict:
        with self._cond:
            return {
                "running": len(self._running),
                "queued": len(self._queue),
                "max": self.workers,
                "queue_max": self.queue_max,
            }

    def _worker(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                job_id = self._queue.popleft()
                cancel = threading.Event()
                self._running[job_id] = cancel
            try:
                self._run(job_id, cancel)
            except Exception:  # noqa: BLE001 — a job's failure is its own, the lane goes on
                log.exception("Batch job %s crashed outside its own error handling", job_id)
            finally:
                with self._cond:
                    self._running.pop(job_id, None)
                    self._cond.notify_all()
