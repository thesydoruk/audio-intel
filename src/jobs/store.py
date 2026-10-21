"""Job rows in SQLite and one directory per job for its input, outputs and result.

SQLite rather than memory so finished results survive a restart (the files do
anyway) and a client that comes back later still finds its job. One
connection guarded by a lock is plenty: a handful of rows change per minute.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STATUSES = ("queued", "running", "done", "failed", "cancelled")
FINISHED = frozenset({"done", "failed", "cancelled"})


@dataclass
class Job:
    id: str
    kind: str
    status: str
    stage: str
    progress: float
    created_at: float
    started_at: float | None
    finished_at: float | None
    expires_at: float | None
    params: dict[str, Any]
    input_name: str
    error: str | None

    @property
    def finished(self) -> bool:
        return self.status in FINISHED

    def to_public(self, now: float | None = None) -> dict[str, Any]:
        """The JSON shape ``GET /v1/jobs/{id}`` returns (without the result)."""
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "expires_at": self.expires_at,
            "input": self.input_name,
            "params": self.params,
            "error": self.error,
        }


class JobStore:
    """Rows in ``<root>/jobs.sqlite``; files in ``<root>/<job id>/``."""

    def __init__(self, root: str | Path, *, result_ttl_s: float, clock=time.time) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.result_ttl_s = max(60.0, float(result_ttl_s))
        self._clock = clock
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.root / "jobs.sqlite", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS jobs ("
                " id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,"
                " stage TEXT NOT NULL DEFAULT '', progress REAL NOT NULL DEFAULT 0,"
                " created_at REAL NOT NULL, started_at REAL, finished_at REAL, expires_at REAL,"
                " params TEXT NOT NULL, input_name TEXT NOT NULL, error TEXT)"
            )
            self._conn.commit()
        # Rows left "running" by a crash will never finish: say so.
        self._fail_orphans()

    # ---------------------------------------------------------------- helpers
    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def output_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "out"

    def result_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "result.json"

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            kind=row["kind"],
            status=row["status"],
            stage=row["stage"],
            progress=float(row["progress"]),
            created_at=float(row["created_at"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            expires_at=row["expires_at"],
            params=json.loads(row["params"]),
            input_name=row["input_name"],
            error=row["error"],
        )

    def _fail_orphans(self) -> None:
        now = self._clock()
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status='failed', error=?, finished_at=?, expires_at=?"
                " WHERE status IN ('queued', 'running')",
                ("server restarted before the job finished", now, now + self.result_ttl_s),
            )
            self._conn.commit()

    # ------------------------------------------------------------------- CRUD
    def create(self, kind: str, params: dict[str, Any], input_name: str) -> Job:
        job_id = uuid.uuid4().hex[:16]
        now = self._clock()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs"
                " (id, kind, status, stage, progress, created_at, params, input_name)"
                " VALUES (?, ?, 'queued', 'queued', 0, ?, ?, ?)",
                (job_id, kind, now, json.dumps(params, ensure_ascii=False), input_name),
            )
            self._conn.commit()
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    def set_progress(self, job_id: str, stage: str, progress: float | None = None) -> None:
        with self._lock:
            if progress is None:
                self._conn.execute("UPDATE jobs SET stage=? WHERE id=?", (stage, job_id))
            else:
                self._conn.execute(
                    "UPDATE jobs SET stage=?, progress=? WHERE id=?",
                    (stage, max(0.0, min(1.0, float(progress))), job_id),
                )
            self._conn.commit()

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status='running', stage='starting', started_at=? WHERE id=?",
                (self._clock(), job_id),
            )
            self._conn.commit()

    def finish(self, job_id: str, status: str, *, error: str | None = None) -> None:
        assert status in FINISHED, status
        now = self._clock()
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, stage=?,"
                " progress=CASE WHEN ? = 'done' THEN 1.0 ELSE progress END,"
                " finished_at=?, expires_at=?, error=? WHERE id=?",
                (
                    status,
                    status,
                    status,
                    now,
                    now + self.result_ttl_s,
                    error,
                    job_id,
                ),
            )
            self._conn.commit()

    def write_result(self, job_id: str, result: dict[str, Any]) -> Path:
        path = self.result_path(job_id)
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return path

    def read_result(self, job_id: str) -> dict[str, Any] | None:
        path = self.result_path(job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def delete(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()
            deleted = cur.rowcount > 0
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        return deleted

    def purge_expired(self, now: float | None = None) -> list[str]:
        """Delete finished jobs whose results are past their TTL."""
        now = self._clock() if now is None else now
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM jobs WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,)
            ).fetchall()
        ids = [r["id"] for r in rows]
        for job_id in ids:
            self.delete(job_id)
        return ids

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def job_to_dict(job: Job) -> dict[str, Any]:
    return asdict(job)
