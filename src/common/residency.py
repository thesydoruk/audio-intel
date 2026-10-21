"""Model groups that load on first use and may leave the GPU when idle.

The service holds several independent model stacks (Whisper, pyannote, PANNs,
wav2vec2 aligners). Loading any of them takes seconds and gigabytes of VRAM, so
they stay resident once loaded; on a GPU shared with other services that is
not always affordable. A :class:`ModelGroup` wraps one such stack:

- ``use()`` is the only way in. It loads the group if needed (one loader at a
  time, other callers wait for it) and counts the caller as a user for the
  duration of the ``with`` block.
- ``maybe_unload()`` lets the group go only when nobody is inside it and it has
  been idle for longer than the TTL. A pinned group never unloads.

:class:`ResidencyManager` owns the groups, runs the janitor thread that calls
``maybe_unload()`` on a timer when ``UNLOAD_IDLE_MODELS=1``, and renders the
snapshot ``/health`` reports: which groups are resident and when each was last
used, so an outside scheduler can see what the GPU is holding.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

log = logging.getLogger("audio-intel")

Clock = Callable[[], float]


class ModelGroup:
    """One loadable, optionally unloadable set of models."""

    def __init__(
        self,
        name: str,
        *,
        load: Callable[[], None],
        unload: Callable[[], None],
        pinned: bool = False,
        clock: Clock = time.monotonic,
    ) -> None:
        self.name = name
        self.pinned = pinned
        self._load = load
        self._unload = unload
        self._clock = clock
        self._cond = threading.Condition()
        self._loaded = False
        self._busy = False  # a load or unload is in progress
        self._users = 0
        self._last_used: float | None = None
        self._loads = 0
        self._unloads = 0
        self._load_seconds = 0.0

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def users(self) -> int:
        return self._users

    def mark_loaded(self) -> None:
        """Record that the models were loaded outside ``use()`` (eager startup)."""
        with self._cond:
            self._loaded = True
            self._last_used = self._clock()

    @contextmanager
    def use(self) -> Iterator[None]:
        """Hold the group loaded for the duration of the block."""
        with self._cond:
            while self._busy:
                self._cond.wait()
            if not self._loaded:
                self._busy = True
                started = self._clock()
                try:
                    # Loading takes seconds: release the lock so users of other
                    # groups and the health endpoint are not held up.
                    self._cond.release()
                    try:
                        self._load()
                    finally:
                        self._cond.acquire()
                    self._loaded = True
                    self._loads += 1
                    self._load_seconds = self._clock() - started
                finally:
                    self._busy = False
                    self._cond.notify_all()
            self._users += 1
        try:
            yield
        finally:
            with self._cond:
                self._users -= 1
                self._last_used = self._clock()
                self._cond.notify_all()

    def idle_seconds(self, now: float | None = None) -> float | None:
        """Seconds since the last user left; ``None`` while in use or never used."""
        if self._users > 0 or self._last_used is None:
            return None
        return (now if now is not None else self._clock()) - self._last_used

    def maybe_unload(self, ttl_s: float, now: float | None = None) -> bool:
        """Unload when loaded, idle for at least ``ttl_s`` and not pinned."""
        with self._cond:
            if self.pinned or not self._loaded or self._busy or self._users > 0:
                return False
            idle = self.idle_seconds(now)
            if idle is None or idle < ttl_s:
                return False
            self._busy = True
        try:
            log.info("Unloading %s models after %.0fs idle", self.name, idle)
            self._unload()
        finally:
            with self._cond:
                self._loaded = False
                self._busy = False
                self._unloads += 1
                self._cond.notify_all()
        return True

    def snapshot(self, now: float | None = None) -> dict:
        with self._cond:
            idle = self.idle_seconds(now)
            return {
                "resident": self._loaded,
                "pinned": self.pinned,
                "in_use": self._users,
                "idle_s": None if idle is None else round(idle, 1),
                "loads": self._loads,
                "unloads": self._unloads,
                "last_load_s": round(self._load_seconds, 1) if self._loads else None,
            }


class ResidencyManager:
    """Registry of :class:`ModelGroup` plus the idle-unload janitor."""

    def __init__(
        self,
        *,
        unload_idle: bool,
        idle_ttl_s: float,
        clock: Clock = time.monotonic,
    ) -> None:
        self.unload_idle = unload_idle
        self.idle_ttl_s = max(1.0, idle_ttl_s)
        self._clock = clock
        self._groups: dict[str, ModelGroup] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def group(
        self,
        name: str,
        *,
        load: Callable[[], None],
        unload: Callable[[], None],
        pinned: bool = False,
    ) -> ModelGroup:
        group = ModelGroup(name, load=load, unload=unload, pinned=pinned, clock=self._clock)
        self._groups[name] = group
        return group

    @property
    def groups(self) -> dict[str, ModelGroup]:
        return dict(self._groups)

    def unload_idle_groups(self, now: float | None = None) -> list[str]:
        """One janitor pass; returns the names of the groups it unloaded."""
        now = self._clock() if now is None else now
        return [name for name, g in self._groups.items() if g.maybe_unload(self.idle_ttl_s, now)]

    def start(self) -> None:
        """Start the janitor thread (no-op unless idle unloading is enabled)."""
        if not self.unload_idle or self._thread is not None:
            return
        interval = max(1.0, min(30.0, self.idle_ttl_s / 4))

        def run() -> None:
            while not self._stop.wait(interval):
                try:
                    self.unload_idle_groups()
                except Exception:  # noqa: BLE001 — the janitor must survive a bad unload
                    log.exception("Model janitor pass failed")

        self._thread = threading.Thread(target=run, name="model-janitor", daemon=True)
        self._thread.start()
        log.info(
            "Idle models are unloaded after %.0fs (checked every %.0fs)", self.idle_ttl_s, interval
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def snapshot(self) -> dict:
        now = self._clock()
        return {
            "unload_idle": self.unload_idle,
            "idle_ttl_s": self.idle_ttl_s,
            "groups": {name: g.snapshot(now) for name, g in self._groups.items()},
        }
