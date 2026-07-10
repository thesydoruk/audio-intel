"""Speaker diarization via pyannote.audio (best-effort, optional)."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from audio_intel.audio.decode import SAMPLE_RATE, load_audio_window, write_temp_wav
from audio_intel.diarization.chunks import (
    DiarizationChunk,
    link_local_speakers,
    merge_chunk_intervals,
    offset_intervals,
    plan_diarization_chunks,
    remap_speaker_ids,
)

if TYPE_CHECKING:
    from audio_intel.config import Config
    from audio_intel.diarization.embeddings import SpeakerEmbedder

log = logging.getLogger("audio-intel")


class SpeakerDiarizer:
    """Pre-loaded pyannote diarization pipelines shared across concurrent requests."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._per_request_workers = cfg.diarization_pipeline_workers
        self._count = self._per_request_workers * cfg.max_concurrent_requests
        self._pipelines: list[object | None] = [None] * self._count
        self._locks = [threading.Lock() for _ in range(self._count)]
        self._init_lock = threading.Lock()
        self._rr = 0
        self._rr_lock = threading.Lock()
        self._link_embedder: SpeakerEmbedder | None = None

    def ensure_ready(self) -> None:
        """Load every pyannote diarization pipeline copy."""
        for index in range(self._count):
            self._load_pipeline(index)

    def _load_pipeline(self, index: int) -> None:
        if self._pipelines[index] is not None:
            return
        with self._init_lock:
            if self._pipelines[index] is not None:
                return
            import torch
            from audio_intel.hf_pretrained import pretrained_auth_kwargs
            from pyannote.audio import Pipeline

            token = self.cfg.hf_token
            if not token:
                raise ValueError("HF_TOKEN is required when SPEAKERS_ENABLED=1")

            log.info(
                "Loading pyannote diarization pipeline %d/%d (device=%s)",
                index + 1,
                self._count,
                self.cfg.diarization_device,
            )
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                **pretrained_auth_kwargs(token, Pipeline.from_pretrained),
            )
            device = torch.device(self.cfg.diarization_device)
            pipeline.to(device)
            self._pipelines[index] = pipeline
            log.info("Diarization pipeline %d/%d loaded", index + 1, self._count)

    def _pick_pipeline_index(self) -> int:
        with self._rr_lock:
            index = self._rr % self._count
            self._rr += 1
            return index

    def _normalize_pipeline_output(self, diarization) -> list[dict]:
        label_map: dict[str, str] = {}
        next_idx = 0
        intervals: list[dict] = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            raw = str(speaker)
            if raw not in label_map:
                label_map[raw] = f"spk_{next_idx}"
                next_idx += 1
            intervals.append(
                {
                    "start": round(float(turn.start), 3),
                    "end": round(float(turn.end), 3),
                    "speaker_id": label_map[raw],
                }
            )

        intervals.sort(key=lambda item: item["start"])
        return intervals

    def _run_pipeline(self, pipeline, path: str) -> list[dict]:
        diarization = pipeline(path)
        return self._normalize_pipeline_output(diarization)

    def _diarize_file(self, path: str) -> list[dict]:
        index = self._pick_pipeline_index()
        with self._locks[index]:
            self._load_pipeline(index)
            pipeline = self._pipelines[index]
            assert pipeline is not None
            log.info(
                "Diarization starting (pipeline %d/%d)",
                index + 1,
                self._count,
            )
            started = time.perf_counter()
            intervals = self._run_pipeline(pipeline, path)
            elapsed = time.perf_counter() - started
        log.info(
            "Diarization finished in %.1fs (pipeline %d/%d): %d intervals, %d speakers",
            elapsed,
            index + 1,
            self._count,
            len(intervals),
            len({item["speaker_id"] for item in intervals}),
        )
        return intervals

    def _resolve_link_embedder(self, link_embedder: SpeakerEmbedder | None) -> SpeakerEmbedder:
        if link_embedder is not None:
            return link_embedder
        if self._link_embedder is None:
            from audio_intel.diarization.embeddings import SpeakerEmbedder

            self._link_embedder = SpeakerEmbedder(self.cfg)
        return self._link_embedder

    def _speaker_embeddings_for_window(
        self,
        embedder: SpeakerEmbedder,
        audio,
        intervals: list[dict],
        *,
        window_start: float,
        window_end: float,
        time_offset: float = 0.0,
    ) -> dict[str, list[float]]:
        spans_by_speaker: dict[str, list[tuple[float, float]]] = {}
        for interval in intervals:
            start = float(interval["start"]) - time_offset
            end = float(interval["end"]) - time_offset
            overlap_start = max(start, window_start - time_offset)
            overlap_end = min(end, window_end - time_offset)
            if overlap_end <= overlap_start:
                continue
            sid = str(interval["speaker_id"])
            spans_by_speaker.setdefault(sid, []).append((overlap_start, overlap_end))

        embeddings: dict[str, list[float]] = {}
        for sid, spans in spans_by_speaker.items():
            embedding = embedder.embed_speaker_spans(
                audio,
                spans,
                sample_rate=SAMPLE_RATE,
                min_speech_s=self.cfg.diarization_link_min_speech_s,
            )
            if embedding is not None:
                embeddings[sid] = embedding
        return embeddings

    def _diarize_chunk_file(self, path: str, chunk: DiarizationChunk) -> list[dict]:
        chunk_path: str | None = None
        try:
            audio = load_audio_window(
                path,
                SAMPLE_RATE,
                chunk.start_s,
                chunk.duration_s,
            )
            chunk_path = write_temp_wav(audio, SAMPLE_RATE)
            local_intervals = self._diarize_file(chunk_path)
            return offset_intervals(local_intervals, chunk.start_s)
        finally:
            if chunk_path and os.path.exists(chunk_path):
                os.unlink(chunk_path)

    def _diarize_chunked(
        self,
        path: str,
        duration_s: float,
        *,
        link_embedder: SpeakerEmbedder | None,
    ) -> list[dict]:
        chunk_s = self.cfg.diarization_chunk_s
        overlap_s = self.cfg.diarization_chunk_overlap_s
        chunks = plan_diarization_chunks(
            duration_s,
            chunk_s=chunk_s,
            overlap_s=overlap_s,
        )
        if len(chunks) <= 1:
            return self._diarize_file(path)

        embedder = self._resolve_link_embedder(link_embedder)
        workers = min(len(chunks), self._per_request_workers)
        log.info(
            "Chunked diarization starting: %.1fs audio in %d chunk(s) "
            "(chunk_s=%.0fs, overlap_s=%.0fs, link_threshold=%.2f, workers=%d)",
            duration_s,
            len(chunks),
            chunk_s,
            overlap_s,
            self.cfg.diarization_link_threshold,
            workers,
        )
        started = time.perf_counter()

        infer_started = time.perf_counter()
        raw_by_index = self._diarize_chunks_parallel(path, chunks, workers=workers)
        log.info(
            "Chunked diarization inference finished in %.1fs (%d chunks, %d workers)",
            time.perf_counter() - infer_started,
            len(chunks),
            workers,
        )

        link_started = time.perf_counter()
        chunk_results = self._link_chunk_results(
            path,
            chunks,
            raw_by_index,
            embedder=embedder,
            overlap_s=overlap_s,
        )
        log.info(
            "Chunked diarization linking finished in %.1fs",
            time.perf_counter() - link_started,
        )

        merged = merge_chunk_intervals(chunk_results, chunks, overlap_s=overlap_s)
        elapsed = time.perf_counter() - started
        log.info(
            "Chunked diarization finished in %.1fs: %d chunks → %d intervals, %d speakers",
            elapsed,
            len(chunks),
            len(merged),
            len({item["speaker_id"] for item in merged}),
        )
        return merged

    def _diarize_chunks_parallel(
        self,
        path: str,
        chunks: list[DiarizationChunk],
        *,
        workers: int,
    ) -> dict[int, list[dict]]:
        """Run pyannote on each chunk concurrently (one pipeline copy per worker)."""
        raw_by_index: dict[int, list[dict]] = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._diarize_chunk_file, path, chunk): chunk for chunk in chunks
            }
            for future in as_completed(futures):
                chunk = futures[future]
                raw_by_index[chunk.index] = future.result()
                log.info(
                    "Diarization chunk %d/%d inferred: %d intervals",
                    chunk.index + 1,
                    len(chunks),
                    len(raw_by_index[chunk.index]),
                )

        return raw_by_index

    def _link_chunk_results(
        self,
        path: str,
        chunks: list[DiarizationChunk],
        raw_by_index: dict[int, list[dict]],
        *,
        embedder: SpeakerEmbedder,
        overlap_s: float,
    ) -> list[list[dict]]:
        """Assign global speaker ids across chunk boundaries (sequential)."""
        global_intervals: list[dict] = []
        chunk_results: list[list[dict]] = []
        next_global_index = 0

        for chunk in chunks:
            global_chunk_intervals = raw_by_index[chunk.index]
            if chunk.index == 0:
                mapping = {
                    str(sid): str(sid)
                    for sid in dict.fromkeys(item["speaker_id"] for item in global_chunk_intervals)
                }
                next_global_index = _next_speaker_index(global_chunk_intervals)
            else:
                overlap_end = chunk.start_s + overlap_s
                chunk_audio = load_audio_window(path, SAMPLE_RATE, chunk.start_s, chunk.duration_s)
                overlap_audio = load_audio_window(path, SAMPLE_RATE, chunk.start_s, overlap_s)
                local_embeddings = self._speaker_embeddings_for_window(
                    embedder,
                    chunk_audio,
                    global_chunk_intervals,
                    window_start=chunk.start_s,
                    window_end=overlap_end,
                    time_offset=chunk.start_s,
                )
                global_embeddings = self._speaker_embeddings_for_window(
                    embedder,
                    overlap_audio,
                    global_intervals,
                    window_start=chunk.start_s,
                    window_end=overlap_end,
                    time_offset=chunk.start_s,
                )
                mapping, next_global_index = link_local_speakers(
                    local_intervals=global_chunk_intervals,
                    previous_global_intervals=global_intervals,
                    chunk_start_s=chunk.start_s,
                    overlap_s=overlap_s,
                    local_embeddings=local_embeddings,
                    global_embeddings=global_embeddings,
                    next_global_index=next_global_index,
                    threshold=self.cfg.diarization_link_threshold,
                )

            remapped = remap_speaker_ids(global_chunk_intervals, mapping)
            chunk_results.append(remapped)
            global_intervals.extend(remapped)
            log.info(
                "Diarization chunk %d/%d linked: %d local speakers",
                chunk.index + 1,
                len(chunks),
                len(mapping),
            )

        return chunk_results

    def diarize(
        self,
        path: str,
        *,
        duration_s: float | None = None,
        link_embedder: SpeakerEmbedder | None = None,
    ) -> list[dict]:
        """Return normalized diarization intervals ``[{start, end, speaker_id}]``."""
        if duration_s is None or duration_s <= 0:
            from audio_intel.audio.decode import probe_media_duration

            duration_s = probe_media_duration(path)

        if self.cfg.diarization_chunk_s > 0 and duration_s > self.cfg.diarization_chunk_s:
            return self._diarize_chunked(path, duration_s, link_embedder=link_embedder)
        return self._diarize_file(path)


def _next_speaker_index(intervals: list[dict]) -> int:
    """Return the next free ``spk_N`` index after existing speaker labels."""
    max_idx = -1
    for interval in intervals:
        sid = str(interval["speaker_id"])
        if not sid.startswith("spk_"):
            continue
        try:
            max_idx = max(max_idx, int(sid[4:]))
        except ValueError:
            continue
    return max_idx + 1
