from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from audio_intel.separate.audio.chunking import build_chunks
from audio_intel.separate.audio.io import (
    materialize_master_wav,
    probe_audio,
    read_audio,
    read_mono_resampled,
    read_segment,
    write_wav,
)
from audio_intel.separate.audio.resample import resample_audio
from audio_intel.separate.cache.store import CacheStore
from audio_intel.separate.config import ensure_cache_dir, load_config
from audio_intel.separate.embeddings.profile import SpeakerProfile, SpeakerProfileBank
from audio_intel.separate.models.base import estimate_snr_db, resolve_device
from audio_intel.separate.models.registry import (
    build_diarizer,
    build_embedder,
    build_enhancer,
    build_overlap_detector,
    build_separator,
    build_super_resolution,
    build_tse,
)
from audio_intel.separate.pipeline.metadata import build_summaries, write_metadata, write_rttm
from audio_intel.separate.report.html_report import write_html_report
from audio_intel.separate.stitching.stitcher import TrackStitcher
from audio_intel.separate.tracking.assigner import GlobalSpeakerTracker, LocalSpeakerStats
from audio_intel.separate.types import (
    DiarizationResult,
    DiarizationSegment,
    EmbeddingSample,
    ExtractedRegion,
    OverlapRegion,
    PipelineResult,
    SourceType,
    TimeSpan,
)
from tqdm import tqdm

log = logging.getLogger(__name__)

ProgressCb = Callable[[str], None]


# --------------------------------------------------------------------------- helpers
def merge_segments(
    segments: list[DiarizationSegment],
    *,
    min_duration: float = 0.0,
    merge_gap: float = 0.0,
) -> list[DiarizationSegment]:
    """Merge same-speaker turns separated by < merge_gap; drop turns shorter than min_duration."""
    by: dict[str, list[DiarizationSegment]] = {}
    for seg in sorted(segments, key=lambda s: (s.span.start, s.span.end)):
        by.setdefault(seg.local_speaker, []).append(seg)
    out: list[DiarizationSegment] = []
    for spk, segs in by.items():
        cur: DiarizationSegment | None = None
        for seg in segs:
            if cur is None:
                cur = DiarizationSegment(
                    spk, TimeSpan(seg.span.start, seg.span.end), seg.confidence
                )
                continue
            if seg.span.start <= cur.span.end + merge_gap:
                cur = DiarizationSegment(
                    spk,
                    TimeSpan(cur.span.start, max(cur.span.end, seg.span.end)),
                    min(cur.confidence, seg.confidence),
                )
            else:
                out.append(cur)
                cur = DiarizationSegment(
                    spk, TimeSpan(seg.span.start, seg.span.end), seg.confidence
                )
        if cur is not None:
            out.append(cur)
    out = [s for s in out if s.span.duration >= min_duration]
    out.sort(key=lambda s: (s.span.start, s.span.end))
    return out


def relabel_by_duration(
    segments: list[DiarizationSegment], prefix: str = "speaker_"
) -> list[DiarizationSegment]:
    """Map arbitrary diarizer labels to speaker_00, speaker_01, ... ordered by total speech time."""
    totals: dict[str, float] = {}
    for s in segments:
        totals[s.local_speaker] = totals.get(s.local_speaker, 0.0) + s.span.duration
    order = sorted(totals, key=lambda k: (-totals[k], k))
    mapping = {old: f"{prefix}{i:02d}" for i, old in enumerate(order)}
    return [DiarizationSegment(mapping[s.local_speaker], s.span, s.confidence) for s in segments]


def _merge_spans(spans: list[TimeSpan]) -> list[TimeSpan]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: s.start)
    out = [TimeSpan(spans[0].start, spans[0].end)]
    for s in spans[1:]:
        if s.start <= out[-1].end:
            out[-1] = TimeSpan(out[-1].start, max(out[-1].end, s.end))
        else:
            out.append(TimeSpan(s.start, s.end))
    return out


def split_by_overlap(
    span: TimeSpan,
    overlaps: list[OverlapRegion],
    *,
    min_part: float = 0.2,
) -> list[tuple[TimeSpan, bool]]:
    """Split a speech turn into (part, is_overlap) pieces.

    Single-speaker slivers shorter than ``min_part`` between two overlaps are
    absorbed into the overlap side, so TSE never has to stitch a 50 ms fragment.
    """
    inter = _merge_spans(
        [
            TimeSpan(max(span.start, o.span.start), min(span.end, o.span.end))
            for o in overlaps
            if span.overlap(o.span) > 0
        ]
    )
    if not inter:
        return [(span, False)]
    parts: list[tuple[TimeSpan, bool]] = []
    cursor = span.start
    for ov in inter:
        if ov.start - cursor > 1e-6:
            parts.append((TimeSpan(cursor, ov.start), False))
        parts.append((ov, True))
        cursor = ov.end
    if span.end - cursor > 1e-6:
        parts.append((TimeSpan(cursor, span.end), False))

    # Absorb tiny single slivers into neighbouring overlap parts
    merged: list[tuple[TimeSpan, bool]] = []
    for part, is_ov in parts:
        if not is_ov and part.duration < min_part and (merged and merged[-1][1]):
            prev, _ = merged[-1]
            merged[-1] = (TimeSpan(prev.start, part.end), True)
            continue
        if is_ov and merged and not merged[-1][1] and merged[-1][0].duration < min_part:
            prev, _ = merged.pop()
            part = TimeSpan(prev.start, part.end)
        merged.append((part, is_ov))
    # Coalesce adjacent parts of the same kind
    out: list[tuple[TimeSpan, bool]] = []
    for part, is_ov in merged:
        if out and out[-1][1] == is_ov and abs(out[-1][0].end - part.start) < 1e-6:
            out[-1] = (TimeSpan(out[-1][0].start, part.end), is_ov)
        else:
            out.append((part, is_ov))
    return out


def _slice_16k(audio16: np.ndarray, sr: int, span: TimeSpan) -> np.ndarray:
    a, b = span.to_samples(sr)
    a = max(0, min(len(audio16), a))
    b = max(a, min(len(audio16), b))
    return audio16[a:b]


# ------------------------------------------------------------------------ pipeline
class SpeakerExtractPipeline:
    """Recording -> per-speaker 48 kHz tracks + background track, all sample-aligned."""

    def __init__(
        self, cfg: dict[str, Any] | None = None, config_path: str | Path | None = None
    ) -> None:
        self.cfg = cfg or load_config(config_path)
        self.device = resolve_device(self.cfg.get("device", {}).get("device", "auto"))
        cache_root = ensure_cache_dir(self.cfg)
        self.cache = CacheStore(
            cache_root, enabled=bool(self.cfg.get("cache", {}).get("enabled", True))
        )
        self.bank = SpeakerProfileBank(self.cfg.get("embeddings", {}))
        self.tracker = GlobalSpeakerTracker(self.bank, self.cfg.get("tracking", {}))
        # Inside the transcription service this is its GPU-token guard; alone, a no-op.
        self._gpu = (self.cfg.get("_providers") or {}).get("gpu_guard") or nullcontext

        self._separator = None
        self._diarizer = None
        self._embedder = None
        self._overlap = None
        self._tse = None
        self._enh_48k = None
        self._enh_16k = None
        self._sr = None

    # ------------------------------------------------------------------ models
    def _unload_all(self) -> None:
        for h in (
            self._separator,
            self._diarizer,
            self._embedder,
            self._overlap,
            self._tse,
            self._enh_48k,
            self._enh_16k,
            self._sr,
        ):
            if h is not None and h.loaded:
                h.unload()

    def _maybe_unload(self, handle: Any) -> None:
        if (
            handle is not None
            and handle.loaded
            and self.cfg.get("device", {}).get("unload_idle_models", False)
        ):
            handle.unload()

    def separator(self):
        if self._separator is None:
            self._separator = build_separator(self.cfg, self.device)
        return self._separator

    def diarizer(self):
        if self._diarizer is None:
            self._diarizer = build_diarizer(self.cfg, self.device)
        return self._diarizer

    def embedder(self):
        if self._embedder is None:
            self._embedder = build_embedder(self.cfg, self.device)
        return self._embedder

    def overlap_detector(self):
        if self._overlap is None:
            self._overlap = build_overlap_detector(self.cfg, self.device)
        return self._overlap

    def tse(self):
        if self._tse is None:
            self._tse = build_tse(self.cfg, self.device, embedder=self.embedder())
        return self._tse

    def enhancer_48k(self):
        if self._enh_48k is None:
            self._enh_48k = build_enhancer(self.cfg, role="single_speaker", device=self.device)
        return self._enh_48k

    def enhancer_16k(self):
        if self._enh_16k is None:
            self._enh_16k = build_enhancer(self.cfg, role="post_tse", device=self.device)
        return self._enh_16k

    def super_resolution(self):
        if self._sr is None:
            self._sr = build_super_resolution(self.cfg, self.device)
        return self._sr

    # --------------------------------------------------------------------- run
    def run(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        progress: ProgressCb | None = None,
    ) -> PipelineResult:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_cb = progress or (lambda m: log.info(m))

        cfg = self.cfg
        audio_cfg = cfg.get("audio", {})
        pipe = cfg.get("pipeline", {})
        master_sr = int(audio_cfg.get("target_sample_rate", 48000))
        proc_sr = int(audio_cfg.get("processing_sample_rate", 16000))
        force = bool(pipe.get("force", False))
        resume = bool(pipe.get("resume", True)) and not force
        stop_after = pipe.get("stop_after")

        # 1) Master ------------------------------------------------------------
        log_cb("Materializing 48 kHz master WAV")
        self._input_name = input_path.stem
        master_path = output_dir / "_master_48k.wav"
        if force or not master_path.exists():
            info = materialize_master_wav(
                input_path,
                master_path,
                target_sr=master_sr,
                channels=audio_cfg.get("master_channels", "keep"),
                subtype=str(audio_cfg.get("master_subtype", "PCM_24")),
            )
        else:
            info = probe_audio(master_path)
        src_hash = CacheStore.file_hash(master_path)
        duration = info.duration

        # 2) Speech / background separation -------------------------------------
        sep_cfg = cfg.get("separation", {})
        sep_backend = str(sep_cfg.get("backend", "demucs"))
        if not pipe.get("run_separation", True):
            sep_backend = "none"
        speech_path = master_path
        background_path: Path | None = None
        if sep_backend == "demucs":
            speech_path, background_path = self._separate(
                master_path,
                output_dir,
                src_hash=src_hash,
                master_sr=master_sr,
                resume=resume,
                log_cb=log_cb,
            )
        elif sep_backend in {"residual", "none", "identity", "stub"}:
            log_cb(f"Separation backend '{sep_backend}': speech stem = master")
        else:
            raise ValueError(f"Unknown separation backend: {sep_backend}")
        if stop_after == "separation":
            return self._finish(
                output_dir,
                info,
                master_path,
                speech_path,
                background_path,
                [],
                [],
                [],
                {},
                {},
                stitcher=None,
            )

        # 3) Diarization (global labels) -----------------------------------------
        log_cb("Loading speech stem at 16 kHz for analysis")
        speech16 = read_mono_resampled(speech_path, proc_sr)
        diar_cfg = cfg.get("diarization", {})
        mode = str(diar_cfg.get("mode", "full"))
        extra: dict[str, Any] = {"diarization_mode": mode, "separation_backend": sep_backend}

        if mode == "chunked":
            segments = self._diarize_chunked(
                speech16, proc_sr, duration, src_hash, resume, log_cb, extra
            )
        else:
            segments = self._diarize_full(speech16, proc_sr, src_hash, resume, log_cb)
        segments = merge_segments(
            segments,
            min_duration=float(diar_cfg.get("min_segment_duration", 0.0)),
            merge_gap=float(diar_cfg.get("merge_gap", 0.0)),
        )
        if mode != "chunked":
            segments = relabel_by_duration(segments)
        speakers = sorted({s.local_speaker for s in segments})
        log_cb(f"Diarization: {len(segments)} turns, {len(speakers)} speaker(s)")
        if stop_after == "diarization":
            return self._finish(
                output_dir,
                info,
                master_path,
                speech_path,
                background_path,
                segments,
                [],
                [],
                {},
                extra,
                stitcher=None,
            )

        # 4) Overlaps --------------------------------------------------------------
        overlaps: list[OverlapRegion] = []
        if pipe.get("run_overlap", True):
            od = self.overlap_detector()
            od.ensure_loaded()
            with self._gpu():
                overlaps = od.detect(
                    speech16, proc_sr, offset=0.0, diarization=DiarizationResult(segments=segments)
                )
            self._maybe_unload(od)
        log_cb(f"Overlap regions: {len(overlaps)} ({sum(o.span.duration for o in overlaps):.1f} s)")
        if stop_after == "overlap":
            return self._finish(
                output_dir,
                info,
                master_path,
                speech_path,
                background_path,
                segments,
                overlaps,
                [],
                {},
                extra,
                stitcher=None,
            )

        # 5) Speaker profiles (embeddings + enrollment) ---------------------------------
        run_tse = bool(pipe.get("run_tse", True)) and bool(overlaps)
        if run_tse or pipe.get("run_embeddings", True):
            self._build_profiles(speech16, proc_sr, segments, overlaps, log_cb)
        for sid in speakers:
            if sid not in self.bank.profiles:
                self.bank.profiles[sid] = SpeakerProfile(speaker_id=sid)

        # 6) Extraction --------------------------------------------------------------
        stitch_cfg = cfg.get("stitching", {})
        stitcher = TrackStitcher(
            duration,
            sample_rate=master_sr,
            crossfade_ms=float(stitch_cfg.get("crossfade_ms", 40.0)),
            segment_pad_ms=float(stitch_cfg.get("segment_pad_ms", 120.0)),
        )
        for sid in speakers:
            stitcher.ensure_speaker(sid)
        regions = self._extract(
            speech_path,
            speech16,
            proc_sr,
            master_sr,
            segments,
            overlaps,
            stitcher,
            run_tse=run_tse,
            log_cb=log_cb,
        )

        # 7) Export ------------------------------------------------------------------
        result = self._finish(
            output_dir,
            info,
            master_path,
            speech_path,
            background_path,
            segments,
            overlaps,
            regions,
            {},
            extra,
            stitcher=stitcher,
        )
        log_cb("Done")
        return result

    # ------------------------------------------------------------------ stages
    def _separate(
        self,
        master_path: Path,
        output_dir: Path,
        *,
        src_hash: str,
        master_sr: int,
        resume: bool,
        log_cb: ProgressCb,
    ) -> tuple[Path, Path]:
        sep_cfg = self.cfg.get("separation", {})
        out_cfg = self.cfg.get("output", {})
        speech_path = output_dir / str(out_cfg.get("speech_stem_name", "_speech_48k.wav"))
        background_path = output_dir / str(out_cfg.get("background_name", "background_48k.wav"))
        key = self.cache.key(
            "separation",
            src_hash,
            sep_cfg.get("model"),
            sep_cfg.get("shifts"),
            sep_cfg.get("overlap"),
            sep_cfg.get("segment"),
        )
        cached = self.cache.load_json("separation", key) if resume else None
        if (
            cached
            and Path(cached.get("speech", "")).exists()
            and Path(cached.get("background", "")).exists()
        ):
            log_cb("Separation: reusing cached stems")
            return Path(cached["speech"]), Path(cached["background"])

        log_cb(
            f"Separating speech / background with Demucs ({sep_cfg.get('model', 'htdemucs_ft')})"
        )
        sep = self.separator()
        sep.ensure_loaded()
        master, sr = read_audio(master_path)
        with self._gpu():
            speech, background = sep.separate(master, sr)
        del master
        subtype = str(out_cfg.get("subtype", "PCM_24"))
        write_wav(speech_path, speech, sr, subtype=subtype)
        write_wav(background_path, background, sr, subtype=subtype)
        self._maybe_unload(sep)
        self.cache.save_json(
            "separation", key, {"speech": str(speech_path), "background": str(background_path)}
        )
        return speech_path, background_path

    def _diarize_full(
        self,
        speech16: np.ndarray,
        sr: int,
        src_hash: str,
        resume: bool,
        log_cb: ProgressCb,
    ) -> list[DiarizationSegment]:
        diar_cfg = self.cfg.get("diarization", {})
        num = diar_cfg.get("num_speakers")
        min_spk = num if num is not None else diar_cfg.get("min_speakers")
        max_spk = num if num is not None else diar_cfg.get("max_speakers")
        key = self.cache.key(
            "diarization-full",
            src_hash,
            self.cfg.get("separation", {}).get("backend"),
            self.cfg.get("separation", {}).get("model"),
            diar_cfg.get("backend"),
            diar_cfg.get("model_id"),
            min_spk,
            max_spk,
            diar_cfg.get("use_exclusive"),
        )
        cached = self.cache.load_json("diarization", key) if resume else None
        if cached:
            log_cb("Diarization: reusing cached result")
            return _segments_from_json(cached)

        log_cb("Diarizing the full recording")
        d = self.diarizer()
        d.ensure_loaded()
        with self._gpu():
            result = d.process(speech16, sr, min_speakers=min_spk, max_speakers=max_spk, offset=0.0)
        self._maybe_unload(d)
        self.cache.save_json("diarization", key, _segments_to_json(result.segments))
        return result.segments

    def _diarize_chunked(
        self,
        speech16: np.ndarray,
        sr: int,
        duration: float,
        src_hash: str,
        resume: bool,
        log_cb: ProgressCb,
        extra: dict[str, Any],
    ) -> list[DiarizationSegment]:
        """Chunked diarization with cross-chunk identity tracking (for extreme lengths)."""
        cfg = self.cfg
        diar_cfg = cfg.get("diarization", {})
        chunk_cfg = cfg.get("chunking", {})
        emb_cfg = cfg.get("embeddings", {})
        chunks = build_chunks(
            duration,
            chunk_duration=float(chunk_cfg.get("duration", 45.0)),
            overlap=float(chunk_cfg.get("overlap", 10.0)),
            min_chunk_duration=float(chunk_cfg.get("min_chunk_duration", 5.0)),
        )
        num = diar_cfg.get("num_speakers")
        min_spk = num if num is not None else diar_cfg.get("min_speakers")
        max_spk = num if num is not None else diar_cfg.get("max_speakers")
        od = self.overlap_detector()
        od.ensure_loaded()
        emb = self.embedder()
        emb.ensure_loaded()

        global_segments: list[DiarizationSegment] = []
        history: list[dict[str, Any]] = []
        prev_span: TimeSpan | None = None
        for chunk in tqdm(chunks, desc="chunks"):
            audio16 = _slice_16k(speech16, sr, chunk.span)
            key = self.cache.key(
                "diarization-chunk",
                src_hash,
                chunk.index,
                chunk.span.start,
                chunk.span.end,
                diar_cfg.get("backend"),
                diar_cfg.get("model_id"),
                min_spk,
                max_spk,
            )
            cached = self.cache.load_json("diarization", key) if resume else None
            if cached:
                segs = _segments_from_json(cached)
            else:
                d = self.diarizer()
                d.ensure_loaded()
                with self._gpu():
                    segs = d.process(
                        audio16,
                        sr,
                        min_speakers=min_spk,
                        max_speakers=max_spk,
                        offset=chunk.span.start,
                    ).segments
                self.cache.save_json("diarization", key, _segments_to_json(segs))
            with self._gpu():
                overlaps = od.detect(
                    audio16,
                    sr,
                    offset=chunk.span.start,
                    diarization=DiarizationResult(segments=segs),
                )

            # Local embeddings from the longest clean segment per local label
            reliable = self._reliable_segments(segs, overlaps)
            by_local: dict[str, list[DiarizationSegment]] = {}
            for s in reliable:
                by_local.setdefault(s.local_speaker, []).append(s)
            # A local speaker with only short/overlapped turns still needs an identity,
            # otherwise their speech would silently vanish from every track.
            for s in segs:
                if s.local_speaker not in by_local:
                    longest = max(
                        (x for x in segs if x.local_speaker == s.local_speaker),
                        key=lambda x: x.span.duration,
                    )
                    if longest.span.duration >= 0.3:
                        by_local[s.local_speaker] = [longest]
            local_stats: list[LocalSpeakerStats] = []
            best_clip: dict[str, tuple[np.ndarray, TimeSpan]] = {}
            for local_id, lsegs in by_local.items():
                best = max(lsegs, key=lambda s: s.span.duration)
                clip = _slice_16k(speech16, sr, best.span)
                if estimate_snr_db(clip) < float(emb_cfg.get("min_snr_db", 0.0)):
                    continue
                with self._gpu():
                    vec = emb.embed(clip, sr)
                all_local = [s for s in segs if s.local_speaker == local_id]
                total = sum(s.span.duration for s in all_local)
                ov_dur = sum(sum(s.span.overlap(o.span) for o in overlaps) for s in all_local)
                local_stats.append(
                    LocalSpeakerStats(
                        local_id=local_id,
                        embedding=vec,
                        quality=float(
                            np.clip(best.confidence * np.tanh(best.span.duration / 2.0), 0, 1)
                        ),
                        duration=total,
                        span=TimeSpan(
                            min(s.span.start for s in lsegs), max(s.span.end for s in lsegs)
                        ),
                        overlap_span_ratio=(ov_dur / total) if total > 0 else 0.0,
                    )
                )
                best_clip[local_id] = (clip, best.span)

            overlap_prev = None
            if prev_span is not None:
                a, b = max(prev_span.start, chunk.span.start), min(prev_span.end, chunk.span.end)
                if b > a:
                    overlap_prev = TimeSpan(a, b)
            assignments = self.tracker.assign_chunk(
                local_stats, chunk_span=chunk.span, overlap_with_prev=overlap_prev
            )
            local_to_global = {a.local_id: a for a in assignments}
            history.append(
                {
                    "chunk": chunk.index,
                    "mapping": {a.local_id: a.global_id for a in assignments},
                    "scores": {a.local_id: round(a.score, 3) for a in assignments},
                }
            )

            enroll_max = float(cfg.get("tse", {}).get("enrollment_max_duration", 8.0))
            for loc in local_stats:
                a = local_to_global.get(loc.local_id)
                if a is None:
                    continue
                clip, span = best_clip[loc.local_id]
                clip = clip[: int(enroll_max * sr)]
                self.bank.maybe_update(
                    a.global_id,
                    EmbeddingSample(
                        a.global_id, loc.embedding, span, loc.quality, snr_db=estimate_snr_db(clip)
                    ),
                    enrollment=clip,
                    sr=sr,
                )

            for s in segs:
                a = local_to_global.get(s.local_speaker)
                if a is None:
                    continue
                start, end = max(s.span.start, chunk.core.start), min(s.span.end, chunk.core.end)
                if end - start > 1e-4:
                    global_segments.append(
                        DiarizationSegment(a.global_id, TimeSpan(start, end), s.confidence)
                    )
            prev_span = chunk.span

        extra["chunk_maps"] = history
        self._maybe_unload(self._diarizer)
        return global_segments

    def _reliable_segments(
        self, segments: list[DiarizationSegment], overlaps: list[OverlapRegion]
    ) -> list[DiarizationSegment]:
        emb_cfg = self.cfg.get("embeddings", {})
        min_dur = float(emb_cfg.get("min_segment_duration", 0.8))
        max_dur = float(emb_cfg.get("max_segment_duration", 12.0))
        reject_overlap = bool(emb_cfg.get("reject_overlap", True))
        out: list[DiarizationSegment] = []
        for seg in segments:
            if seg.span.duration < min_dur:
                continue
            clipped = TimeSpan(seg.span.start, min(seg.span.end, seg.span.start + max_dur))
            if reject_overlap and any(clipped.overlap(o.span) > 0.05 for o in overlaps):
                continue
            out.append(DiarizationSegment(seg.local_speaker, clipped, seg.confidence))
        return out

    def _build_profiles(
        self,
        speech16: np.ndarray,
        sr: int,
        segments: list[DiarizationSegment],
        overlaps: list[OverlapRegion],
        log_cb: ProgressCb,
    ) -> None:
        """Embed the cleanest turns of every speaker and pick an enrollment clip for TSE."""
        emb_cfg = self.cfg.get("embeddings", {})
        tse_cfg = self.cfg.get("tse", {})
        max_segs = int(emb_cfg.get("max_profile_segments", 12))
        enroll_max = float(tse_cfg.get("enrollment_max_duration", 8.0))
        enroll_min = float(tse_cfg.get("enrollment_min_duration", 1.0))
        emb = self.embedder()
        emb.ensure_loaded()

        speakers = sorted({s.local_speaker for s in segments})
        reliable = self._reliable_segments(segments, overlaps)
        log_cb(f"Building speaker profiles for {len(speakers)} speaker(s)")
        for sid in speakers:
            clean = sorted(
                (s for s in reliable if s.local_speaker == sid), key=lambda s: -s.span.duration
            )
            if not clean:
                # Speaker never speaks alone long enough: fall back to their longest turn.
                clean = sorted(
                    (s for s in segments if s.local_speaker == sid), key=lambda s: -s.span.duration
                )[:1]
            enrollment: np.ndarray | None = None
            for seg in clean[:max_segs]:
                clip = _slice_16k(speech16, sr, seg.span)
                if len(clip) < int(0.3 * sr):
                    continue
                with self._gpu():
                    vec = emb.embed(clip, sr)
                sample = EmbeddingSample(
                    sid,
                    vec,
                    seg.span,
                    quality=float(np.clip(seg.confidence * np.tanh(seg.span.duration / 2.0), 0, 1)),
                    snr_db=estimate_snr_db(clip),
                )
                if sid not in self.bank.profiles:
                    self.bank.create(sample, enrollment=None, sr=sr, speaker_id=sid)
                else:
                    self.bank.profiles[sid].samples.append(sample)
                if enrollment is None and seg.span.duration >= enroll_min:
                    enrollment = clip[: int(enroll_max * sr)]
            if sid in self.bank.profiles and enrollment is not None:
                prof = self.bank.profiles[sid]
                prof.enrollment_audio = np.asarray(enrollment, dtype=np.float32)
                prof.enrollment_sr = sr
        self._maybe_unload(emb)

    def _extract(
        self,
        speech_path: Path,
        speech16: np.ndarray,
        proc_sr: int,
        master_sr: int,
        segments: list[DiarizationSegment],
        overlaps: list[OverlapRegion],
        stitcher: TrackStitcher,
        *,
        run_tse: bool,
        log_cb: ProgressCb,
    ) -> list[ExtractedRegion]:
        cfg = self.cfg
        pipe = cfg.get("pipeline", {})
        enh_cfg = cfg.get("enhancement", {})
        enhance = bool(enh_cfg.get("enabled", False)) and bool(pipe.get("run_enhancement", True))
        use_sr = bool(cfg.get("super_resolution", {}).get("enabled", True)) and bool(
            pipe.get("run_super_resolution", True)
        )
        min_part = float(cfg.get("overlap", {}).get("min_overlap_duration", 0.15))

        regions: list[ExtractedRegion] = []
        log_cb("Extracting per-speaker tracks")
        for seg in tqdm(segments, desc="segments", leave=False):
            sid = seg.local_speaker
            parts = (
                split_by_overlap(seg.span, overlaps, min_part=max(0.2, min_part))
                if overlaps
                else [(seg.span, False)]
            )
            for part, is_overlap in parts:
                padded = stitcher.padded(part)
                if is_overlap and run_tse:
                    out48 = self._extract_overlap(
                        speech16, proc_sr, master_sr, sid, padded, enhance=enhance, use_sr=use_sr
                    )
                    source = SourceType.OVERLAP
                else:
                    clip48, _ = read_segment(speech_path, padded.start, padded.end)
                    if enhance:
                        e48 = self.enhancer_48k()
                        e48.ensure_loaded()
                        clip48 = e48.enhance(clip48, master_sr)
                    out48 = clip48
                    source = SourceType.UNCERTAIN if is_overlap else SourceType.SINGLE_SPEAKER
                stitcher.write(sid, padded, out48)
                regions.append(
                    ExtractedRegion(
                        speaker_id=sid,
                        span=part,
                        source_type=source,
                        diarization_confidence=seg.confidence,
                        matching_confidence=1.0,
                        tse_confidence=(0.7 if source == SourceType.OVERLAP else None),
                        notes=""
                        if source != SourceType.UNCERTAIN
                        else "overlap left unseparated (TSE disabled)",
                    )
                )
        self._maybe_unload(self._tse)
        self._maybe_unload(self._enh_48k)
        self._maybe_unload(self._enh_16k)
        self._maybe_unload(self._sr)
        return regions

    def _extract_overlap(
        self,
        speech16: np.ndarray,
        proc_sr: int,
        master_sr: int,
        sid: str,
        span: TimeSpan,
        *,
        enhance: bool,
        use_sr: bool,
    ) -> np.ndarray:
        mix16 = _slice_16k(speech16, proc_sr, span)
        centroid = self.bank.centroid(sid) if sid in self.bank.profiles else None
        enroll, enroll_sr = (
            self.bank.enrollment(sid) if sid in self.bank.profiles else (None, proc_sr)
        )
        tse = self.tse()
        tse.ensure_loaded()
        with self._gpu():
            extracted = tse.extract(
                mix16,
                proc_sr,
                target_embedding=centroid,
                reference_audio=enroll,
                reference_sample_rate=enroll_sr,
            )
        if enhance:
            e16 = self.enhancer_16k()
            e16.ensure_loaded()
            extracted = e16.enhance(extracted, tse.sample_rate)
        if use_sr:
            srm = self.super_resolution()
            srm.ensure_loaded()
            out48 = srm.upscale(extracted, tse.sample_rate)
            if srm.output_sample_rate != master_sr:
                out48 = resample_audio(out48, srm.output_sample_rate, master_sr)
        else:
            out48 = resample_audio(extracted, tse.sample_rate, master_sr)
        return out48

    # ---------------------------------------------------------------- export
    def _finish(
        self,
        output_dir: Path,
        info,
        master_path: Path,
        speech_path: Path,
        background_path: Path | None,
        segments: list[DiarizationSegment],
        overlaps: list[OverlapRegion],
        regions: list[ExtractedRegion],
        similarities: dict[str, Any],
        extra: dict[str, Any],
        *,
        stitcher: TrackStitcher | None,
    ) -> PipelineResult:
        cfg = self.cfg
        out_cfg = cfg.get("output", {})
        sep_backend = str(cfg.get("separation", {}).get("backend", "demucs"))
        subtype = str(out_cfg.get("subtype", "PCM_24"))
        speaker_files: dict[str, str] = {}

        if stitcher is not None:
            speaker_files = stitcher.export(
                output_dir,
                naming=str(out_cfg.get("naming", "speaker_{id:02d}_48k.wav")),
                subtype=subtype,
            )
            if out_cfg.get("write_background", True) and sep_backend == "residual":
                background_path = output_dir / str(
                    out_cfg.get("background_name", "background_48k.wav")
                )
                master, sr = read_audio(master_path)
                speech_sum = stitcher.sum_tracks()
                n = min(len(master), len(speech_sum))
                residual = master[:n] - speech_sum[:n, None]
                write_wav(background_path, residual, sr, subtype=subtype)
        if (
            not out_cfg.get("write_background", True)
            and background_path is not None
            and sep_backend != "demucs"
        ):
            background_path = None
        if (
            not out_cfg.get("write_background", True)
            and sep_backend == "demucs"
            and background_path is not None
        ):
            # Stem was needed for separation; the user asked not to keep it.
            try:
                background_path.unlink()
            except OSError:
                pass
            background_path = None
        if sep_backend == "demucs" and not cfg.get("separation", {}).get("write_speech_stem", True):
            try:
                if speech_path != master_path:
                    speech_path.unlink()
            except OSError:
                pass

        speakers = sorted({s.local_speaker for s in segments} | set(speaker_files))
        result = PipelineResult(
            input_info=info,
            output_dir=str(output_dir),
            speaker_files=speaker_files,
            regions=regions,
            summaries=build_summaries(regions),
            global_speakers=speakers,
            segments=segments,
            overlaps=overlaps,
            background_path=str(background_path) if background_path else None,
            speech_stem_path=str(speech_path)
            if (speech_path != master_path and speech_path.exists())
            else None,
            master_path=str(master_path),
        )

        if out_cfg.get("write_rttm", True) and segments:
            result.rttm_path = str(
                write_rttm(
                    segments,
                    output_dir / "diarization.rttm",
                    uri=getattr(self, "_input_name", "audio"),
                )
            )
        if out_cfg.get("write_metadata", True):
            meta_path = output_dir / "metadata.json"
            payload_extra = dict(extra)
            if similarities:
                payload_extra["similarities"] = similarities
            write_metadata(result, meta_path, extra=payload_extra)
            result.metadata_path = str(meta_path)
        if out_cfg.get("write_html_report", True):
            report_path = output_dir / "report.html"
            write_html_report(
                report_path,
                duration=info.duration,
                speakers=speakers,
                regions=[r.to_dict() for r in regions],
                chunks=[],
                overlaps=[{"span": {"start": o.span.start, "end": o.span.end}} for o in overlaps],
                similarities=similarities,
                title=f"speaker-extract · {getattr(self, '_input_name', Path(info.path).stem)}",
                segments=[
                    {"speaker": s.local_speaker, "start": s.span.start, "end": s.span.end}
                    for s in segments
                ],
                background=str(background_path) if background_path else None,
            )
            result.report_path = str(report_path)
        return result


# ------------------------------------------------------------------ (de)serialization
def _segments_to_json(segments: list[DiarizationSegment]) -> dict[str, Any]:
    return {
        "segments": [
            {
                "speaker": s.local_speaker,
                "start": s.span.start,
                "end": s.span.end,
                "confidence": s.confidence,
            }
            for s in segments
        ]
    }


def _segments_from_json(data: dict[str, Any]) -> list[DiarizationSegment]:
    out = []
    for s in data.get("segments", []):
        out.append(
            DiarizationSegment(
                local_speaker=str(s.get("speaker", s.get("local_speaker"))),
                span=TimeSpan(float(s["start"]), float(s["end"])),
                confidence=float(s.get("confidence", 1.0)),
            )
        )
    return out
