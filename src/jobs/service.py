"""What the HTTP layer talks to: submit, poll, download, cancel, and the janitor."""

from __future__ import annotations

import logging
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from audio_intel.jobs.lane import BatchBusyError, BatchLane
from audio_intel.jobs.separate_job import JobCancelled, run_separation_job
from audio_intel.jobs.store import JobStore

log = logging.getLogger("audio-intel")

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class JobService:
    def __init__(self, cfg: Any, transcriber: Any, *, clock=time.time) -> None:
        self.cfg = cfg
        self.transcriber = transcriber
        self._clock = clock
        self.store = JobStore(
            cfg.jobs_dir, result_ttl_s=cfg.jobs_result_ttl_h * 3600.0, clock=clock
        )
        self.lane = BatchLane(
            workers=cfg.batch_concurrent_jobs, queue_max=cfg.batch_queue_max, run=self._run
        )
        self._purge_lock = threading.Lock()
        self._last_purge = 0.0
        self.lane.start()
        log.info(
            "Batch lane started (workers=%d, queue=%d, jobs in %s, results kept %.0fh)",
            self.lane.workers,
            self.lane.queue_max,
            self.store.root,
            cfg.jobs_result_ttl_h,
        )

    def close(self) -> None:
        self.lane.stop()
        self.store.close()

    # ---------------------------------------------------------------- submit
    def submit_separation(self, upload_path: str | Path, filename: str, params: dict) -> Any:
        """Move an uploaded file into a new job and queue it; raises BatchBusyError."""
        self.purge()
        suffix = Path(filename or "").suffix.lower() or Path(str(upload_path)).suffix or ".bin"
        if not _SAFE_NAME.match(suffix.lstrip(".") or "bin"):
            suffix = ".bin"
        job = self.store.create(
            "separate", {**params, "input_file": f"input{suffix}"}, filename or "upload"
        )
        try:
            shutil.move(str(upload_path), self.store.job_dir(job.id) / f"input{suffix}")
            self.lane.submit(job.id)
        except BatchBusyError:
            self.store.delete(job.id)
            raise
        except Exception:
            self.store.delete(job.id)
            raise
        return job

    def _run(self, job_id: str, cancel: threading.Event) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        self.store.mark_running(job_id)
        try:
            result = run_separation_job(
                job, store=self.store, transcriber=self.transcriber, cancel=cancel
            )
            self.store.write_result(job_id, result)
            self.store.finish(job_id, "done")
            log.info("Job %s done in %ss", job_id, result.get("elapsed_s"))
        except JobCancelled:
            self.store.finish(job_id, "cancelled")
            log.info("Job %s cancelled", job_id)
        except Exception as exc:  # noqa: BLE001 — the job's failure is reported, not raised
            log.exception("Job %s failed", job_id)
            self.store.finish(job_id, "failed", error=f"{type(exc).__name__}: {exc}"[:500])

    # ------------------------------------------------------------------ read
    def get(self, job_id: str) -> dict[str, Any] | None:
        job = self.store.get(job_id)
        if job is None:
            return None
        body = job.to_public()
        if job.status == "done":
            result = self.store.read_result(job_id)
            if result is not None:
                body["result"] = result
        return body

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        self.purge()
        return [job.to_public() for job in self.store.list(limit)]

    def file_path(self, job_id: str, name: str) -> Path | None:
        """A file of a finished job's output directory, by its bare name."""
        if not _SAFE_NAME.match(name) or name.startswith("."):
            return None
        job = self.store.get(job_id)
        if job is None or job.status != "done":
            return None
        path = self.store.output_dir(job_id) / name
        return path if path.is_file() else None

    # -------------------------------------------------------------- cancel
    def cancel_or_delete(self, job_id: str) -> bool:
        """Stop a queued or running job; remove a finished one. False when unknown."""
        job = self.store.get(job_id)
        if job is None:
            return False
        where = self.lane.cancel(job_id)
        if where == "queued":
            self.store.finish(job_id, "cancelled")
            self.store.delete(job_id)
            return True
        if where == "running":
            return True  # the worker marks it cancelled at the next stage boundary
        self.store.delete(job_id)
        return True

    # -------------------------------------------------------------- janitor
    def purge(self, now: float | None = None) -> list[str]:
        now = self._clock() if now is None else now
        with self._purge_lock:
            if now - self._last_purge < 60 and now >= self._last_purge:
                return []
            self._last_purge = now
        ids = self.store.purge_expired(now)
        if ids:
            log.info("Purged %d expired job(s)", len(ids))
        return ids

    def snapshot(self) -> dict[str, Any]:
        return {"batch": {**self.lane.snapshot(), "jobs": self.store.counts()}}
