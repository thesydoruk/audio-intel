"""Tests for parallel chunked diarization orchestration."""

from __future__ import annotations

import sys
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.config import load_config
from audio_intel.diarization.chunks import DiarizationChunk
from audio_intel.diarization.pyannote import SpeakerDiarizer  # noqa: E402


def _completed_future(result) -> Future:
    future: Future = Future()
    future.set_result(result)
    return future


class ChunkedDiarizationParallelTest(unittest.TestCase):
    def test_diarize_chunks_parallel_uses_thread_pool(self) -> None:
        cfg = load_config()
        diarizer = SpeakerDiarizer(cfg)
        chunks = [
            DiarizationChunk(index=0, start_s=0.0, end_s=600.0),
            DiarizationChunk(index=1, start_s=570.0, end_s=1200.0),
            DiarizationChunk(index=2, start_s=1170.0, end_s=1800.0),
        ]

        with (
            mock.patch.object(diarizer, "_diarize_chunk_file") as diarize_chunk,
            mock.patch("audio_intel.diarization.pyannote.ThreadPoolExecutor") as pool_cls,
        ):
            diarize_chunk.side_effect = lambda _path, chunk: [
                {
                    "start": chunk.start_s + 1.0,
                    "end": chunk.start_s + 2.0,
                    "speaker_id": "spk_0",
                }
            ]
            pool = mock.MagicMock()
            pool_cls.return_value.__enter__.return_value = pool

            def submit(func, path, chunk):
                return _completed_future(func(path, chunk))

            pool.submit.side_effect = submit

            result = diarizer._diarize_chunks_parallel("audio.wav", chunks, workers=2)

        pool_cls.assert_called_once_with(max_workers=2)
        self.assertEqual(pool.submit.call_count, 3)
        self.assertEqual(len(result), 3)
        self.assertIn(0, result)
        self.assertIn(2, result)

    def test_chunked_path_links_after_parallel_inference(self) -> None:
        cfg = load_config()
        diarizer = SpeakerDiarizer(cfg)
        diarizer.cfg = mock.MagicMock(
            diarization_chunk_s=600.0,
            diarization_chunk_overlap_s=30.0,
            diarization_link_threshold=0.75,
            diarization_link_min_speech_s=0.5,
        )

        chunks = [
            DiarizationChunk(index=0, start_s=0.0, end_s=600.0),
            DiarizationChunk(index=1, start_s=570.0, end_s=900.0),
        ]

        with (
            mock.patch(
                "audio_intel.diarization.pyannote.plan_diarization_chunks",
                return_value=chunks,
            ),
            mock.patch.object(
                diarizer,
                "_diarize_chunks_parallel",
                return_value={
                    0: [{"start": 1.0, "end": 2.0, "speaker_id": "spk_0"}],
                    1: [{"start": 571.0, "end": 572.0, "speaker_id": "spk_0"}],
                },
            ) as parallel,
            mock.patch.object(diarizer, "_resolve_link_embedder") as resolve_embedder,
            mock.patch.object(diarizer, "_link_chunk_results") as link_chunks,
            mock.patch(
                "audio_intel.diarization.pyannote.merge_chunk_intervals",
                return_value=[{"start": 1.0, "end": 2.0, "speaker_id": "spk_0"}],
            ),
        ):
            embedder = mock.MagicMock()
            resolve_embedder.return_value = embedder
            link_chunks.return_value = [[{"start": 1.0, "end": 2.0, "speaker_id": "spk_0"}]]

            intervals = diarizer._diarize_chunked(
                "audio.wav",
                900.0,
                link_embedder=None,
            )

        parallel.assert_called_once()
        self.assertEqual(parallel.call_args.kwargs["workers"], diarizer._per_request_workers)
        link_chunks.assert_called_once()
        self.assertEqual(len(intervals), 1)


if __name__ == "__main__":
    unittest.main()
