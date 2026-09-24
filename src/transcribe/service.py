"""Per-chunk transcription and timestamp-preserving stitching.

Pipeline:
  1. Prepare :class:`~audio_intel.audio.workspace.MediaWorkspace`: convert input once
     to 16 kHz (+ 32 kHz if AED).
  2. Run Silero VAD on the prepared 16 kHz WAV, then group regions into chunks.
  3. Transcribe each chunk by reading windows from the prepared WAV.
  4. Optionally align, diarize, and detect sound events — all reading from workspace files.
  5. Delete the workspace when the request completes.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from audio_intel.audio.chunking import Chunk
from audio_intel.audio.workspace import MediaWorkspace
from audio_intel.config import Config
from audio_intel.diarization.chunks import merge_speaker_ids_by_embedding, remap_segment_speaker_ids
from audio_intel.diarization.embeddings import SpeakerEmbedder
from audio_intel.diarization.merge import assign_speakers_to_segments, build_speaker_roster
from audio_intel.diarization.pyannote import SpeakerDiarizer
from audio_intel.events.panns import AedParams, AudioEventDetector
from audio_intel.transcribe.alignment import (
    alignment_preload_languages,
    assign_segment_indices_to_chunks,
    build_alignment_config,
    group_speech_indices_by_language,
    merge_aligned_speech_segments,
    resolve_alignment_language,
    shift_speech_segments,
    speech_segments_to_transcript,
)
from audio_intel.transcribe.engine import WhisperEngine, WhisperRuntime
from audio_intel.transcribe.segments import overall_confidence, speech_segment_to_timeline_dict
from audio_intel.vad.pipeline import VadPipeline

log = logging.getLogger("audio-intel")


def _release_accelerator_memory() -> None:
    """Best-effort GPU/host cleanup between heavy post-ASR stages."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@dataclass
class _ChunkTranscription:
    """Speech segments and metadata produced from one VAD chunk."""

    chunk_index: int
    segments: list[dict]
    text_parts: list[str]
    languages: list[str]


class Transcriber:
    """Holds loaded models and runs up to N HTTP pipeline requests in parallel."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        concurrent = cfg.max_concurrent_requests
        self._request_semaphore = threading.Semaphore(concurrent)
        per_request_workers = cfg.whisper_parallel_workers
        whisper_pool_size = per_request_workers * concurrent
        log.info(
            "Loading %d Whisper model cop%s of %s (device=%s, compute=%s, "
            "%d concurrent request(s), %d worker(s) per request)",
            whisper_pool_size,
            "y" if whisper_pool_size == 1 else "ies",
            cfg.model,
            cfg.device,
            cfg.compute_type,
            concurrent,
            per_request_workers,
        )
        self._engine = WhisperEngine(WhisperRuntime.from_server(cfg))
        self._engine.ensure_workers(whisper_pool_size)
        log.info("Whisper model%s loaded", "" if whisper_pool_size == 1 else " copies")

        self._vad = VadPipeline.from_server(cfg)
        self._vad.ensure_ready()

        # AED / pyannote stay lazy until post-ASR so Whisper peak VRAM matches the
        # older in-repo server (Whisper + VAD only at startup on shared GPUs).
        self.aed = AudioEventDetector(cfg) if cfg.aed_enabled else None
        self.diarizer = SpeakerDiarizer(cfg) if cfg.diarization_enabled else None
        self.embedder = SpeakerEmbedder(cfg) if cfg.diarization_enabled else None
        self._post_asr_ready_lock = threading.Lock()
        self._aed_ready = False
        self._speakers_ready = False

        self._aligners: dict[str, object] = {}
        self._aligners_lock = threading.Lock()
        if cfg.diarization_enabled:
            log.info(
                "Speaker diarization enabled (lazy load; device=%s, pipeline_workers=%d, "
                "embed_workers=%d, chunk_s=%.0fs)",
                cfg.diarization_device,
                cfg.diarization_pipeline_workers,
                cfg.diarization_embed_workers,
                cfg.diarization_chunk_s,
            )
        if cfg.aed_enabled:
            log.info(
                "Sound-event detection enabled (lazy load; device=%s)",
                cfg.aed_device,
            )
        if cfg.alignment_enabled:
            self._preload_alignment_models()
            log.info(
                "WhisperX alignment enabled (device=%s, preloaded=%s)",
                cfg.alignment_device or cfg.device,
                ", ".join(self._aligners),
            )

    def transcribe(
        self,
        path: str,
        override_language: str | None = None,
        *,
        align: bool = False,
        diarize: bool = False,
        sound_events: bool = False,
        aed_overrides: dict | None = None,
        aed_debug_top_n: int = 0,
    ) -> dict:
        """Transcribe a media file into a verbose-JSON-shaped dict."""
        with self._request_semaphore:
            return self._run(
                path,
                override_language,
                align=align,
                diarize=diarize,
                sound_events=sound_events,
                aed_overrides=aed_overrides,
                aed_debug_top_n=aed_debug_top_n,
            )

    def _run(
        self,
        path: str,
        override_language: str | None,
        *,
        align: bool = False,
        diarize: bool = False,
        sound_events: bool = False,
        aed_overrides: dict | None = None,
        aed_debug_top_n: int = 0,
    ) -> dict:
        cfg = self.cfg
        language = override_language or cfg.forced_language
        run_started = time.perf_counter()
        run_aed = sound_events and self.aed is not None

        with MediaWorkspace.prepare(
            path,
            need_aed=run_aed,
            aed_sample_rate=cfg.aed_sample_rate,
        ) as media:
            total_s = round(media.duration_s, 3)
            log.info(
                "Transcription starting: %.1fs audio, language=%s, "
                "align=%s, diarize=%s, sound_events=%s",
                total_s,
                language or "auto",
                align,
                diarize,
                sound_events,
            )

            chunks = self._vad.prepare_chunks_for_server(
                media.whisper_path,
                media.duration_s,
                cfg,
            )
            if not chunks:
                detected_sound_events: list[dict] = []
                aed_debug: list[dict] = []
                if run_aed:
                    detected_sound_events, aed_debug = self._detect_sounds(
                        media,
                        aed_overrides,
                        aed_debug_top_n,
                    )
                log.info(
                    "No speech detected (%.1fs audio), %d sound events",
                    total_s,
                    len(detected_sound_events),
                )
                result = {
                    "text": "",
                    "languages": [],
                    "confidence": 0.0,
                    "duration": total_s,
                    "segments": sorted(detected_sound_events, key=lambda s: s["start"]),
                }
                if align:
                    result.update(_empty_alignment_meta())
                if aed_debug:
                    result["aed_debug"] = aed_debug
                return result

            log.info(
                "VAD produced %d chunks (whisper workers=%d)",
                len(chunks),
                cfg.whisper_parallel_workers,
            )

            whisper_started = time.perf_counter()
            want_ctc = align and cfg.alignment_enabled
            speech_segments, text_parts, languages = self._transcribe_chunks(
                media,
                chunks,
                language,
                word_timestamps=not want_ctc,
            )
            log.info(
                "Whisper finished in %.1fs: %d speech segments, languages=%s",
                time.perf_counter() - whisper_started,
                len(speech_segments),
                languages,
            )

            alignment_meta = _empty_alignment_meta(requested=align)
            if align and speech_segments:
                speech_segments, alignment_meta = self._apply_alignment(
                    media,
                    speech_segments,
                    languages,
                    total_s,
                    override_language,
                    chunks=chunks,
                )

            run_diarization = diarize and self.diarizer is not None and bool(speech_segments)
            speakers: list[dict] = []
            sound_event_segments: list[dict] = []
            aed_debug: list[dict] = []

            post_started = time.perf_counter()
            if run_diarization and run_aed:
                _release_accelerator_memory()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    diar_future = pool.submit(
                        self._apply_diarization,
                        media,
                        speech_segments,
                    )
                    aed_future = pool.submit(
                        self._detect_sounds,
                        media,
                        aed_overrides,
                        aed_debug_top_n,
                    )
                    speech_segments, speakers = diar_future.result()
                    sound_event_segments, aed_debug = aed_future.result()
            elif run_diarization:
                _release_accelerator_memory()
                speech_segments, speakers = self._apply_diarization(media, speech_segments)
            elif run_aed:
                _release_accelerator_memory()
                sound_event_segments, aed_debug = self._detect_sounds(
                    media,
                    aed_overrides,
                    aed_debug_top_n,
                )
            if run_diarization or run_aed:
                log.info(
                    "Post-ASR phase finished in %.1fs (diarization=%s, sound_events=%s)",
                    time.perf_counter() - post_started,
                    run_diarization,
                    run_aed,
                )

            full_text = " ".join(text_parts).strip()
            confidence = overall_confidence(speech_segments)
            segments = sorted(speech_segments + sound_event_segments, key=lambda s: s["start"])
            log.info(
                "Done in %.1fs: %d speech + %d sound segments, %d speakers, confidence=%.2f",
                time.perf_counter() - run_started,
                len(speech_segments),
                len(sound_event_segments),
                len(speakers),
                confidence,
            )
            result = {
                "text": full_text,
                "languages": languages,
                "confidence": confidence,
                "duration": total_s,
                "segments": segments,
            }
            if align:
                result.update(alignment_meta)
            if speakers:
                result["speakers"] = speakers
            if aed_debug:
                result["aed_debug"] = aed_debug
            return result

    def _transcribe_chunks(
        self,
        media: MediaWorkspace,
        chunks: list[Chunk],
        language: str | None,
        *,
        word_timestamps: bool = True,
    ) -> tuple[list[dict], list[str], list[str]]:
        """Transcribe all VAD chunks, using parallel model copies when configured."""
        if not chunks:
            return [], [], []

        jobs = list(enumerate(chunks))
        worker_count = self.cfg.whisper_parallel_workers
        if worker_count <= 1 or len(jobs) <= 1:
            chunk_results = [
                result
                for chunk_index, chunk in jobs
                if (
                    result := self._transcribe_one_chunk(
                        chunk_index,
                        chunk,
                        media,
                        language,
                        word_timestamps=word_timestamps,
                    )
                )
                is not None
            ]
        else:
            log.info(
                "Whisper parallel transcription: %d chunks across %d model cop%s",
                len(jobs),
                worker_count,
                "y" if worker_count == 1 else "ies",
            )
            chunk_results: list[_ChunkTranscription] = []
            max_workers = min(worker_count, len(jobs))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        self._transcribe_one_chunk,
                        chunk_index,
                        chunk,
                        media,
                        language,
                        word_timestamps=word_timestamps,
                    )
                    for chunk_index, chunk in jobs
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        chunk_results.append(result)

        speech_segments, text_parts, languages = _merge_chunk_transcriptions(chunk_results)
        speech_segments.sort(key=lambda s: s["start"])
        return speech_segments, text_parts, languages

    def _transcribe_one_chunk(
        self,
        chunk_index: int,
        chunk: Chunk,
        media: MediaWorkspace,
        language: str | None,
        *,
        word_timestamps: bool = True,
    ) -> _ChunkTranscription | None:
        """Run faster-whisper on one contiguous VAD chunk."""
        clip = media.read_whisper_window(chunk.start, chunk.end)
        if clip.size == 0:
            return None

        try:
            transcription = self._engine.transcribe_array(
                clip,
                language=language,
                time_offset=chunk.start,
                word_timestamps=word_timestamps,
            )
        finally:
            del clip

        chunk_segments = [speech_segment_to_timeline_dict(seg) for seg in transcription.segments]
        text_parts = [seg.text for seg in transcription.segments]
        chunk_languages: list[str] = []
        if transcription.language != "unknown":
            chunk_languages.append(transcription.language)

        return _ChunkTranscription(
            chunk_index=chunk_index,
            segments=chunk_segments,
            text_parts=text_parts,
            languages=chunk_languages,
        )

    def _preload_alignment_models(self) -> None:
        from audio_intel.align import CTCAligner

        pool_size = max(self.cfg.max_concurrent_requests, self.cfg.whisper_parallel_workers)
        for language in alignment_preload_languages(self.cfg):
            log.info(
                "Loading WhisperX align model for language=%s (device=%s, workers=%d)",
                language,
                self.cfg.alignment_device or self.cfg.device,
                pool_size,
            )
            aligner = CTCAligner(
                build_alignment_config(self.cfg),
                language=language,
                device=self.cfg.device,
            )
            aligner.ensure_ready(pool_size)  # type: ignore[union-attr]
            self._aligners[language] = aligner

    def _ensure_aligner(self, language: str) -> object:
        aligner = self._aligners.get(language)
        if aligner is not None:
            return aligner
        with self._aligners_lock:
            aligner = self._aligners.get(language)
            if aligner is not None:
                return aligner
            log.warning(
                "WhisperX align model for language=%s was not preloaded at startup; loading now",
                language,
            )
            from audio_intel.align import CTCAligner

            aligner = CTCAligner(
                build_alignment_config(self.cfg),
                language=language,
                device=self.cfg.device,
            )
            aligner.ensure_ready(  # type: ignore[union-attr]
                max(self.cfg.max_concurrent_requests, self.cfg.whisper_parallel_workers)
            )
            self._aligners[language] = aligner
            return aligner

    def _apply_alignment(
        self,
        media: MediaWorkspace,
        speech_segments: list[dict],
        languages: list[str],
        duration_sec: float,
        override_language: str | None,
        chunks: list[Chunk] | None = None,
    ) -> tuple[list[dict], dict]:
        """Run WhisperX CTC alignment; never fail the whole transcription request."""
        meta = _empty_alignment_meta(requested=True)
        if not self.cfg.alignment_enabled:
            log.warning("Alignment requested but WORD_ALIGN_ENABLED=0; skipping")
            meta["alignment_failed"] = True
            return speech_segments, meta

        fallback = resolve_alignment_language(self.cfg, languages, override_language)
        if self.cfg.alignment_language:
            groups = {self.cfg.alignment_language: list(range(len(speech_segments)))}
        else:
            groups = group_speech_indices_by_language(speech_segments, fallback)

        jobs: list[tuple[str, Chunk | None, list[int]]] = []
        for language, indices in groups.items():
            if chunks:
                for chunk, chunk_indices in assign_segment_indices_to_chunks(
                    speech_segments, chunks, indices
                ):
                    jobs.append((language, chunk, chunk_indices))
            else:
                jobs.append((language, None, indices))

        started = time.perf_counter()
        updated = list(speech_segments)
        failed_count = 0
        aligned_at: str | None = None
        applied = False
        job_errors = 0

        def run_job(
            language: str, chunk: Chunk | None, indices: list[int]
        ) -> tuple[list[int], list[dict], int, str | None, bool]:
            subset = [speech_segments[index] for index in indices]
            audio = None
            time_offset = 0.0
            if chunk is not None:
                audio = media.read_whisper_window(chunk.start, chunk.end)
                time_offset = chunk.start
            try:
                merged, group_failed, group_aligned_at = self._align_segment_subset(
                    language,
                    subset,
                    audio_path=media.whisper_path,
                    duration_sec=duration_sec,
                    audio=audio,
                    time_offset=time_offset,
                )
            except ImportError:
                raise
            except Exception:  # noqa: BLE001 — one chunk/language must not fail the request
                log.exception(
                    "Alignment failed for language=%s chunk=%s after %.1fs",
                    language,
                    f"{chunk.start:.1f}-{chunk.end:.1f}" if chunk else "full",
                    time.perf_counter() - started,
                )
                return indices, subset, 0, None, False
            finally:
                del audio
            return indices, merged, group_failed, group_aligned_at, True

        results: list[tuple[list[int], list[dict], int, str | None, bool]] = []
        try:
            if len(jobs) <= 1:
                results = [run_job(*jobs[0])] if jobs else []
            else:
                workers = min(
                    len(jobs),
                    max(self.cfg.whisper_parallel_workers, self.cfg.max_concurrent_requests),
                )
                log.info(
                    "Alignment parallel: %d job(s) across %d worker(s)",
                    len(jobs),
                    workers,
                )
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [
                        pool.submit(run_job, language, chunk, indices)
                        for language, chunk, indices in jobs
                    ]
                    for future in as_completed(futures):
                        results.append(future.result())
        except ImportError:
            log.exception(
                "Alignment requested but WhisperX is not installed; "
                "install with: pip install -e '.[align]'"
            )
            meta["alignment_failed"] = True
            return speech_segments, meta

        for indices, merged, group_failed, group_aligned_at, ok in results:
            if not ok:
                job_errors += 1
                continue
            applied = True
            failed_count += group_failed
            aligned_at = group_aligned_at or aligned_at
            for index, new_seg in zip(indices, merged, strict=True):
                updated[index] = new_seg

        if not applied:
            meta["alignment_failed"] = True
            return speech_segments, meta

        meta.update(
            {
                "alignment_applied": True,
                "alignment_failed": job_errors > 0,
                "alignment_failed_segments": failed_count,
                "aligned_at": aligned_at,
            }
        )
        log.info(
            "Alignment finished in %.1fs: %d segments, %d job(s), %d failed",
            time.perf_counter() - started,
            len(updated),
            len(jobs),
            failed_count,
        )
        return updated, meta

    def _align_segment_subset(
        self,
        language: str,
        subset: list[dict],
        *,
        audio_path: str,
        duration_sec: float,
        audio: object | None = None,
        time_offset: float = 0.0,
    ) -> tuple[list[dict], int, str | None]:
        """Run CTC on one language (and optional chunk) group."""
        local = shift_speech_segments(subset, -time_offset) if time_offset else subset
        aligner = self._ensure_aligner(language)
        transcript = speech_segments_to_transcript(
            cfg=self.cfg,
            path=audio_path,
            speech_segments=local,
            languages=[language],
            duration_sec=duration_sec,
            language_override=None,
            language=language,
        )
        aligned = aligner.align_transcript(  # type: ignore[union-attr]
            transcript,
            audio_path=audio_path,
            audio=audio,
        )
        merged, failed_count = merge_aligned_speech_segments(local, aligned)
        if time_offset:
            merged = shift_speech_segments(merged, time_offset)
        return merged, failed_count, aligned.aligned_at

    def _ensure_aed_ready(self) -> None:
        """Load PANNs weights on first sound-event request."""
        if self.aed is None or self._aed_ready:
            return
        with self._post_asr_ready_lock:
            if self._aed_ready:
                return
            _release_accelerator_memory()
            log.info("Lazy-loading sound-event models")
            self.aed.ensure_ready()
            self._aed_ready = True

    def embed_speakers(
        self,
        path: str,
        spans_by_speaker: dict[str, list[tuple[float, float]]],
        *,
        min_speech_s: float,
        max_clip_s: float,
    ) -> dict:
        """Embed speakers from caller-supplied time spans, without ASR or diarization.

        Lets a client re-embed speakers it already knows (for example stored voice
        profiles after an embedding-model change) at a fraction of a full pass.
        """
        if self.embedder is None:
            raise RuntimeError("Speaker embeddings are disabled (SPEAKERS_ENABLED=0)")
        with self._request_semaphore:
            self._ensure_embedder_ready()
            with MediaWorkspace.prepare(path) as media:
                from audio_intel.audio.decode import SAMPLE_RATE

                intervals = [
                    {"start": start, "end": end, "speaker_id": sid}
                    for sid, spans in spans_by_speaker.items()
                    for start, end in spans
                ]
                embeddings = self.embedder.embed_speakers(
                    media.whisper_path,
                    intervals,
                    sample_rate=SAMPLE_RATE,
                    min_speech_s=min_speech_s,
                    max_clip_s=max_clip_s,
                )
                return {
                    "duration": round(media.duration_s, 3),
                    "embedding_model": self.embedder.model_id,
                    "dimension": self.embedder.dimension,
                    "speakers": [
                        {
                            "id": sid,
                            "speech_seconds": round(
                                sum(max(0.0, end - start) for start, end in spans), 3
                            ),
                            "embedding": embeddings.get(sid),
                        }
                        for sid, spans in spans_by_speaker.items()
                    ],
                }

    def _ensure_embedder_ready(self) -> None:
        """Load only the embedding models (the span endpoint needs no pipeline)."""
        if self.embedder is None or self._speakers_ready:
            return
        with self._post_asr_ready_lock:
            if self._speakers_ready:
                return
            self.embedder.ensure_ready()

    def _ensure_speakers_ready(self) -> None:
        """Load pyannote pipelines/embeddings on first diarization request."""
        if self.diarizer is None or self._speakers_ready:
            return
        with self._post_asr_ready_lock:
            if self._speakers_ready:
                return
            _release_accelerator_memory()
            log.info("Lazy-loading speaker diarization models")
            self.diarizer.ensure_ready()
            if self.embedder is not None:
                self.embedder.ensure_ready()
            self._speakers_ready = True

    def _apply_diarization(
        self,
        media: MediaWorkspace,
        speech_segments: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Run pyannote diarization with speaker embeddings and assign speaker ids."""
        self._ensure_speakers_ready()
        started = time.perf_counter()
        speakers: list[dict] = []
        updated_segments = speech_segments
        log.info("Speaker pipeline starting (%d speech segments)", len(speech_segments))
        try:
            intervals = self.diarizer.diarize(  # type: ignore[union-attr]
                media.whisper_path,
                duration_s=media.duration_s,
                link_embedder=self.embedder,
            )
            updated_segments, speakers = assign_speakers_to_segments(speech_segments, intervals)
            if self.embedder is not None and intervals:
                from audio_intel.audio.decode import SAMPLE_RATE

                embeddings = self.embedder.embed_speakers(
                    media.whisper_path,
                    intervals,
                    sample_rate=SAMPLE_RATE,
                )
                merge_map = merge_speaker_ids_by_embedding(
                    embeddings,
                    threshold=self.cfg.diarization_link_threshold,
                )
                merged_away = sum(1 for src, dst in merge_map.items() if src != dst)
                if merged_away:
                    updated_segments = remap_segment_speaker_ids(updated_segments, merge_map)
                    speakers = build_speaker_roster(updated_segments)
                    log.info(
                        "Speaker post-pass merge: %d duplicate id(s) collapsed "
                        "(%d → %d speakers, threshold=%.2f)",
                        merged_away,
                        len(merge_map),
                        len(speakers),
                        self.cfg.diarization_link_threshold,
                    )
                for row in speakers:
                    emb = embeddings.get(row["id"])
                    if emb is None:
                        for src, dst in merge_map.items():
                            if dst == row["id"] and src in embeddings:
                                emb = embeddings[src]
                                break
                    if emb:
                        row["embedding"] = emb
                        row["embedding_model"] = self.embedder.model_id
            tagged = sum(1 for seg in updated_segments if seg.get("speaker_id"))
            log.info(
                "Speaker pipeline finished in %.1fs: %d speakers, "
                "%d/%d segments tagged, %d embeddings",
                time.perf_counter() - started,
                len(speakers),
                tagged,
                len(updated_segments),
                sum(1 for row in speakers if row.get("embedding")),
            )
        except Exception:  # noqa: BLE001 — diarization is best-effort
            log.exception(
                "Speaker diarization failed after %.1fs; returning speech without speaker_id",
                time.perf_counter() - started,
            )
        return updated_segments, speakers

    def _detect_sounds(
        self,
        media: MediaWorkspace,
        aed_overrides: dict | None = None,
        aed_debug_top_n: int = 0,
    ) -> tuple[list[dict], list[dict]]:
        """Run PANNs sound-event detection at 32 kHz; never fails the request."""
        if self.aed is None or media.aed_path is None:
            return [], []
        self._ensure_aed_ready()
        started = time.perf_counter()
        try:
            # Keep full precision — rounding up to 0.1s overshoots the WAV and
            # can leave AED with a past-EOF tail window.
            duration_s = media.duration_s
            log.info(
                "Sound detection starting: %.1fs at %d Hz", duration_s, self.cfg.aed_sample_rate
            )
            params = AedParams.from_config(self.cfg).with_overrides(aed_overrides)
            events, debug = self.aed.detect_file(
                media.aed_path,
                params=params,
                debug_top_n=aed_debug_top_n,
                duration_s=duration_s,
            )
            log.info(
                "Sound detection finished in %.1fs: %d events",
                time.perf_counter() - started,
                len(events),
            )
            return events, debug
        except Exception:  # noqa: BLE001 — AED is best-effort, transcription must still return
            log.exception(
                "Sound-event detection failed after %.1fs; returning speech only",
                time.perf_counter() - started,
            )
            return [], []


def _empty_alignment_meta(*, requested: bool = True) -> dict:
    return {
        "alignment_requested": requested,
        "alignment_applied": False,
        "alignment_failed": False,
        "alignment_failed_segments": 0,
        "aligned_at": None,
    }


def _merge_chunk_transcriptions(
    chunk_results: list[_ChunkTranscription],
) -> tuple[list[dict], list[str], list[str]]:
    """Stitch parallel chunk outputs in timeline order."""
    chunk_results.sort(key=lambda item: item.chunk_index)
    speech_segments: list[dict] = []
    text_parts: list[str] = []
    languages: list[str] = []
    for result in chunk_results:
        speech_segments.extend(result.segments)
        text_parts.extend(result.text_parts)
        for lang in result.languages:
            if lang not in languages:
                languages.append(lang)
    return speech_segments, text_parts, languages
