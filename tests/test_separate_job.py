"""The separation job end to end on the stub separation backends and a fake transcriber."""

from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.config import load_config  # noqa: E402
from audio_intel.jobs.separate_job import (  # noqa: E402
    JobCancelled,
    SeparationEngine,
    run_separation_job,
)
from audio_intel.jobs.service import JobService  # noqa: E402
from audio_intel.jobs.store import JobStore  # noqa: E402


def write_two_speaker_clip(path: Path, sr: int = 48000) -> None:
    t = np.arange(int(6.0 * sr), dtype=np.float32) / sr
    audio = np.zeros_like(t)
    audio[int(0.5 * sr) : int(2.5 * sr)] = 0.2 * np.sin(2 * np.pi * 110 * t[: int(2.0 * sr)])
    audio[int(3.0 * sr) : int(5.0 * sr)] = 0.18 * np.sin(2 * np.pi * 180 * t[: int(2.0 * sr)])
    audio += 0.01 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, audio, sr)


STUB_SEPARATION = {
    "separation": {"backend": "residual"},
    "diarization": {"backend": "stub"},
    "embeddings": {"backend": "stub"},
    "overlap": {"backend": "stub"},
    "tse": {"backend": "adaptive_wiener"},
    "enhancement": {"enabled": False},
    "super_resolution": {"backend": "soxr_hq"},
    "device": {"device": "cpu"},
}


class FakeTranscriber:
    """Stands in for the service: records what the job asked for."""

    def __init__(self, cfg, out_root: Path) -> None:
        self.calls: list[tuple[str, str]] = []
        self.separation = SeparationEngine(cfg, providers={})
        # The engine's config comes from default.yaml; force the stub backends.
        base = self.separation.build_config
        self.separation.build_config = lambda: {**base(), **STUB_SEPARATION}  # type: ignore[method-assign]

    def separation_models(self):
        from contextlib import nullcontext

        return nullcontext()

    def transcribe_batch(self, path, *, override_language=None, align=False, priority=1):
        self.calls.append(("transcribe", Path(path).name))
        return {
            "segments": [
                {
                    "kind": "speech",
                    "start": 0.5,
                    "end": 2.4,
                    "text": f"from {Path(path).stem}",
                    "confidence": 0.9,
                }
            ],
            "languages": ["uk"],
            "dropped_segments": 1,
        }

    def embed_spans(self, path, spans_by_speaker, **kwargs):
        self.calls.append(("embed", Path(path).name))
        return {
            "embedding_model": "test/embedding",
            "speakers": [{"id": sid, "embedding": [0.1, 0.2]} for sid in spans_by_speaker],
        }

    def detect_sounds_file(self, path, *, priority=1):
        self.calls.append(("sounds", Path(path).name))
        return [{"kind": "sound", "start": 1.0, "end": 2.0, "label": "Music", "score": 0.5}]


class SeparationJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cfg = replace(load_config(), jobs_dir=str(root / "jobs"), separate_device="cpu")
        self.store = JobStore(self.cfg.jobs_dir, result_ttl_s=3600)
        self.transcriber = FakeTranscriber(self.cfg, root)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _job(self, **params):
        job = self.store.create(
            "separate", {"input_file": "input.wav", "sound_events": True, **params}, "clip.wav"
        )
        write_two_speaker_clip(self.store.job_dir(job.id) / "input.wav")
        return job

    def test_runs_every_stage_and_assembles_the_result(self) -> None:
        job = self._job()
        result = run_separation_job(
            job, store=self.store, transcriber=self.transcriber, cancel=threading.Event()
        )
        speakers = result["tracks"]["speakers"]
        self.assertGreaterEqual(len(speakers), 1)
        self.assertTrue(all(name.endswith(".wav") for name in speakers.values()))
        self.assertEqual(result["tracks"]["background"], "background_48k.wav")
        speech = [s for s in result["segments"] if s["kind"] == "speech"]
        self.assertEqual(len(speech), len(speakers))
        self.assertTrue(all(s["speaker_id"] in speakers for s in speech))
        self.assertEqual([s["kind"] for s in result["segments"]][0], "speech")
        self.assertIn("Music", [s.get("label") for s in result["segments"]])
        self.assertEqual(result["languages"], ["uk"])
        self.assertEqual(result["dropped_segments"], len(speakers))
        self.assertEqual({r["id"] for r in result["speakers"]}, set(speakers))
        self.assertTrue(all("embedding" in r for r in result["speakers"]))
        kinds = [kind for kind, _ in self.transcriber.calls]
        self.assertEqual(kinds.count("transcribe"), len(speakers))
        self.assertEqual(kinds.count("sounds"), 1)
        self.assertEqual(self.store.get(job.id).stage, "finishing")  # type: ignore[union-attr]
        for name in speakers.values():
            self.assertTrue((self.store.output_dir(job.id) / name).is_file())

    def test_cancel_before_transcription_stops_the_job(self) -> None:
        job = self._job()
        cancel = threading.Event()
        original = self.transcriber.separation.run

        def run_and_cancel(*args, **kwargs):
            out = original(*args, **kwargs)
            cancel.set()
            return out

        with mock.patch.object(self.transcriber.separation, "run", side_effect=run_and_cancel):
            with self.assertRaises(JobCancelled):
                run_separation_job(
                    job, store=self.store, transcriber=self.transcriber, cancel=cancel
                )
        self.assertEqual([k for k, _ in self.transcriber.calls], [])


class JobServiceTests(unittest.TestCase):
    def test_submit_runs_and_exposes_the_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = replace(load_config(), jobs_dir=str(root / "jobs"), separate_device="cpu")
            transcriber = FakeTranscriber(cfg, root)
            upload = root / "upload.wav"
            write_two_speaker_clip(upload)
            service = JobService(cfg, transcriber)
            try:
                job = service.submit_separation(upload, "clip.wav", {"sound_events": False})
                self.assertFalse(upload.exists())  # moved into the job directory
                for _ in range(300):
                    body = service.get(job.id)
                    assert body is not None
                    if body["status"] in ("done", "failed", "cancelled"):
                        break
                    threading.Event().wait(0.1)
                self.assertEqual(body["status"], "done", body.get("error"))
                self.assertIn("tracks", body["result"])
                name = next(iter(body["result"]["tracks"]["speakers"].values()))
                self.assertIsNotNone(service.file_path(job.id, name))
                self.assertIsNone(service.file_path(job.id, "../jobs.sqlite"))
                self.assertEqual(service.snapshot()["batch"]["jobs"], {"done": 1})
                self.assertTrue(service.cancel_or_delete(job.id))
                self.assertIsNone(service.get(job.id))
            finally:
                service.close()


if __name__ == "__main__":
    unittest.main()
