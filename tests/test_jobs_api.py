"""The job endpoints over FastAPI's test client, with the service faked."""

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - starlette raises RuntimeError without httpx
    TestClient = None  # type: ignore[assignment]

from audio_intel.jobs.lane import BatchBusyError  # noqa: E402


class FakeJob:
    def __init__(self, job_id: str) -> None:
        self.id = job_id
        self.status = "queued"


class FakeJobs:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.submitted: list[tuple[str, dict]] = []
        self.busy = False
        self.deleted: list[str] = []

    def submit_separation(self, upload_path, filename, params):
        if self.busy:
            raise BatchBusyError("2 job(s) already waiting")
        Path(upload_path).unlink()
        self.submitted.append((filename, params))
        return FakeJob("job1")

    def get(self, job_id):
        if job_id != "job1":
            return None
        return {"id": "job1", "status": "done", "result": {"tracks": {"speakers": {}}}}

    def list(self, limit=50):
        return [{"id": "job1", "status": "done"}]

    def file_path(self, job_id, name):
        path = self.root / name
        return path if job_id == "job1" and path.is_file() else None

    def cancel_or_delete(self, job_id):
        self.deleted.append(job_id)
        return job_id == "job1"

    def snapshot(self):
        return {"batch": {"running": 0, "queued": 0, "max": 1, "queue_max": 8, "jobs": {}}}

    def close(self):
        pass


@unittest.skipIf(TestClient is None, "fastapi test client not installed")
class JobsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # The package resolves ``app`` lazily through __getattr__, so a string
        # target would patch the FastAPI object on Python 3.10; patch the module.
        app_module = importlib.import_module("audio_intel.server.app")
        patches = [
            mock.patch.object(app_module, "Transcriber"),
            mock.patch.object(app_module, "JobService"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        self.app_module = app_module
        self.client = TestClient(app_module.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.jobs = FakeJobs(self.root)
        app_module.app.state.jobs = self.jobs

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_separate_true_queues_a_job_and_answers_202(self) -> None:
        response = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.wav", b"RIFF....", "audio/wav")},
            data={"separate": "true", "align": "true", "language": "uk"},
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.headers["location"], "/v1/jobs/job1")
        self.assertEqual(response.json()["url"], "/v1/jobs/job1")
        filename, params = self.jobs.submitted[0]
        self.assertEqual(filename, "clip.wav")
        self.assertEqual(params, {"language": "uk", "align": True, "sound_events": False})

    def test_full_queue_answers_503(self) -> None:
        self.jobs.busy = True
        response = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.wav", b"RIFF....", "audio/wav")},
            data={"separate": "true"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["retry-after"], "30")

    def test_job_read_file_and_delete(self) -> None:
        (self.root / "speaker_00_48k.wav").write_bytes(b"RIFF")
        self.assertEqual(self.client.get("/v1/jobs/job1").json()["status"], "done")
        self.assertEqual(self.client.get("/v1/jobs/nope").status_code, 404)
        self.assertEqual(self.client.get("/v1/jobs").json()["jobs"][0]["id"], "job1")
        got = self.client.get("/v1/jobs/job1/files/speaker_00_48k.wav")
        self.assertEqual((got.status_code, got.content), (200, b"RIFF"))
        self.assertEqual(self.client.get("/v1/jobs/job1/files/missing.wav").status_code, 404)
        self.assertEqual(
            self.client.delete("/v1/jobs/job1").json(), {"id": "job1", "deleted": True}
        )
        self.assertEqual(self.client.delete("/v1/jobs/nope").status_code, 404)

    def test_health_carries_the_batch_lane(self) -> None:
        self.assertIn("batch", self.client.get("/health").json())

    def test_jobs_disabled_answers_404(self) -> None:
        self.app_module.app.state.jobs = None
        self.assertEqual(self.client.get("/v1/jobs/job1").status_code, 404)


if __name__ == "__main__":
    unittest.main()
