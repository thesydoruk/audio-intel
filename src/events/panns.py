"""Non-speech sound detection (PANNs CNN14, AudioSet 527 classes).

The detector runs a framewise sound-event model over the full audio at 32 kHz
and converts the per-frame class probabilities into discrete, timestamped events
that live on the **same timeline** as the speech segments.

Noise is controlled entirely with thresholds (no class subset is hard-coded):
  - ``min_score``      — a class must clear this framewise probability.
  - ``top_k``          — at most this many classes may be active per frame.
  - ``merge_gap_s``    — same-label events closer than this are merged.
  - ``min_duration_s`` — events shorter than this are dropped.
  - ``exclude_speech`` — drop AudioSet speech-family classes (ASR covers them).

The pure helpers (:func:`events_from_framewise` and friends) take plain arrays
so they are unit-testable without torch / the PANNs checkpoint.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace

import numpy as np
from audio_intel.common.model_pool import ModelPool
from audio_intel.config import Config

log = logging.getLogger("audio-intel")


@dataclass(frozen=True)
class AedParams:
    """Effective detection thresholds for a single ``detect`` call.

    Seeded from the server :class:`~audio_intel.config.Config` and optionally overridden per request
    (debug / tuning) without mutating the global configuration.
    """

    min_score: float
    top_k: int
    merge_gap_s: float
    min_duration_s: float
    exclude_speech: bool

    @classmethod
    def from_config(cls, cfg: Config) -> AedParams:
        return cls(
            min_score=cfg.aed_min_score,
            top_k=cfg.aed_top_k,
            merge_gap_s=cfg.aed_merge_gap_s,
            min_duration_s=cfg.aed_min_duration_s,
            exclude_speech=cfg.aed_exclude_speech,
        )

    def with_overrides(self, overrides: dict | None) -> AedParams:
        """Return a copy with any non-``None`` override fields applied."""
        if not overrides:
            return self
        changes = {k: v for k, v in overrides.items() if v is not None and hasattr(self, k)}
        return replace(self, **changes) if changes else self


# AudioSet speech-family labels are recognised by these substrings; transcription
# already represents speech, so they are dropped from sound events by default.
_SPEECH_SUBSTRINGS = ("speech", "conversation", "narration", "monologue", "babbling")


def is_speech_label(label: str) -> bool:
    """True when an AudioSet label belongs to the speech family."""
    low = label.lower()
    return any(sub in low for sub in _SPEECH_SUBSTRINGS)


def _topk_active(probs: np.ndarray, min_score: float, top_k: int) -> np.ndarray:
    """Boolean ``[T, C]`` mask: class active when it clears ``min_score`` and is
    within the ``top_k`` highest-probability classes for that frame."""
    over = probs >= min_score
    num_classes = probs.shape[1]
    if top_k <= 0 or top_k >= num_classes:
        return over
    # Indices of the top_k classes per frame; mark them, then AND with threshold.
    top_idx = np.argpartition(-probs, top_k - 1, axis=1)[:, :top_k]
    top_mask = np.zeros_like(over)
    np.put_along_axis(top_mask, top_idx, True, axis=1)
    return over & top_mask


def _runs(active: np.ndarray, merge_gap_frames: int) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs in a 1-D boolean array as ``(start, end_exclusive)``.

    Runs separated by a gap of at most ``merge_gap_frames`` are merged.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(active)))

    if not runs or merge_gap_frames <= 0:
        return runs

    merged: list[tuple[int, int]] = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if s - pe <= merge_gap_frames:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def events_from_framewise(
    probs: np.ndarray,
    frame_dt: float,
    labels: Sequence[str],
    *,
    min_score: float,
    top_k: int,
    merge_gap_s: float,
    min_duration_s: float,
    exclude_speech: bool,
    prompt_allowlist: Sequence[str],
) -> list[dict]:
    """Convert framewise class probabilities into timestamped sound events.

    @param probs — ``[frames, classes]`` framewise probabilities (0–1).
    @param frame_dt — seconds per frame.
    @returns events ``{kind, start, end, label, index, score, prompt_relevant}``
      sorted by start time (then descending score).
    """
    if probs.ndim != 2 or probs.shape[0] == 0:
        return []

    allow = {a.strip().lower() for a in prompt_allowlist if a.strip()}
    merge_gap_frames = int(round(merge_gap_s / frame_dt)) if frame_dt > 0 else 0
    active = _topk_active(probs, min_score, top_k)

    events: list[dict] = []
    for cls in np.where(active.any(axis=0))[0]:
        label = labels[cls] if cls < len(labels) else str(cls)
        if exclude_speech and is_speech_label(label):
            continue
        column = active[:, cls]
        for run_start, run_end in _runs(column, merge_gap_frames):
            start = round(run_start * frame_dt, 3)
            end = round(run_end * frame_dt, 3)
            if end - start < min_duration_s:
                continue
            score = round(float(probs[run_start:run_end, cls].max()), 3)
            prompt_relevant = (not allow) or (label.lower() in allow)
            events.append(
                {
                    "kind": "sound",
                    "start": start,
                    "end": end,
                    "label": label,
                    "index": int(cls),
                    "score": score,
                    "prompt_relevant": prompt_relevant,
                }
            )

    events.sort(key=lambda e: (e["start"], -e["score"]))
    return events


def merge_sound_events(events: list[dict], merge_gap_s: float) -> list[dict]:
    """Merge same-label events separated by small gaps (including window seams)."""
    if not events:
        return []
    if merge_gap_s <= 0:
        return sorted(events, key=lambda e: (e["start"], -e["score"]))

    merged: list[dict] = []
    for ev in sorted(events, key=lambda e: e["start"]):
        if not merged:
            merged.append(ev)
            continue
        prev = merged[-1]
        if ev["label"] == prev["label"] and ev["start"] - prev["end"] <= merge_gap_s:
            merged[-1] = {
                **prev,
                "end": max(prev["end"], ev["end"]),
                "score": max(prev["score"], ev["score"]),
                "prompt_relevant": prev.get("prompt_relevant", True)
                or ev.get("prompt_relevant", True),
            }
        else:
            merged.append(ev)
    return merged


def peak_labels_from_framewise(
    probs: np.ndarray,
    frame_dt: float,
    labels: Sequence[str],
    *,
    top_n: int,
) -> list[dict]:
    """Top-``top_n`` AudioSet classes by peak framewise probability over the clip.

    Unfiltered diagnostic view (ignores ``min_score`` / ``top_k`` / duration /
    speech gates) so callers can see what the model scored — e.g. how confident
    PANNs was about ``Explosion`` even when it fell below the live thresholds.

    @returns ``{label, index, score, at}`` rows sorted by descending score,
      where ``at`` is the timestamp (s) of the peak frame.
    """
    if top_n <= 0 or probs.ndim != 2 or probs.shape[0] == 0:
        return []
    peak = probs.max(axis=0)
    peak_frame = probs.argmax(axis=0)
    order = np.argsort(-peak)[:top_n]
    out: list[dict] = []
    for cls in order:
        label = labels[cls] if cls < len(labels) else str(cls)
        out.append(
            {
                "label": label,
                "index": int(cls),
                "score": round(float(peak[cls]), 3),
                "at": round(float(peak_frame[cls] * frame_dt), 3),
            }
        )
    return out


@dataclass(frozen=True)
class AedWindowJob:
    """One PANNs inference window over the AED timeline (sample indices)."""

    index: int
    start: int
    end: int
    total_samples: int


def plan_aed_windows(
    total_samples: int,
    *,
    window_samples: int,
    overlap_samples: int,
    min_tail_samples: int,
) -> list[AedWindowJob]:
    """Split a waveform into overlapping AED inference windows."""
    if total_samples <= 0:
        return []

    if window_samples <= 0 or total_samples <= window_samples:
        return [AedWindowJob(index=0, start=0, end=total_samples, total_samples=total_samples)]

    overlap = max(overlap_samples, 0)
    overlap = min(overlap, window_samples - 1)
    step = window_samples - overlap

    jobs: list[AedWindowJob] = []
    start = 0
    index = 0
    while start < total_samples:
        end = min(start + window_samples, total_samples)
        if 0 < total_samples - end < min_tail_samples:
            end = total_samples
        jobs.append(AedWindowJob(index=index, start=start, end=end, total_samples=total_samples))
        if end >= total_samples:
            break
        start += step
        index += 1

    return jobs


def trim_window_framewise_probs(
    probs: np.ndarray,
    segment_samples: int,
    *,
    start_sample: int,
    end_sample: int,
    total_samples: int,
    overlap_half_samples: int,
    sample_rate: int,
) -> tuple[np.ndarray, float, float]:
    """Trim overlap margins from framewise probs and return global timing metadata."""
    frames = probs.shape[0]
    ratio = frames / segment_samples if segment_samples > 0 else 0.0
    half = overlap_half_samples
    left_trim = 0 if start_sample == 0 else int(round(half * ratio))
    right_trim = 0 if end_sample >= total_samples else int(round(half * ratio))
    if left_trim + right_trim < frames:
        probs = probs[left_trim : frames - right_trim]

    kept_samples = segment_samples
    if left_trim + right_trim < frames and ratio > 0:
        kept_samples = max(int(round((frames - left_trim - right_trim) / ratio)), 0)

    segment_duration_s = kept_samples / sample_rate if sample_rate > 0 else 0.0
    offset_s = (
        (start_sample + (left_trim / ratio if ratio > 0 else 0)) / sample_rate
        if sample_rate > 0
        else 0.0
    )
    return probs, segment_duration_s, offset_s


def merge_debug_peaks(peaks_list: Sequence[dict[int, dict]]) -> dict[int, dict]:
    """Keep the highest-scoring peak per AudioSet class across windows."""
    merged: dict[int, dict] = {}
    for peaks in peaks_list:
        for cls, row in peaks.items():
            current = merged.get(cls)
            if current is None or row["score"] > current["score"]:
                merged[cls] = row
    return merged


class AudioEventDetector:
    """Pre-loaded PANNs CNN14 sound-event detector."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._pool = ModelPool(self._create_model)
        self._labels: list[str] = []
        self._labels_lock = threading.Lock()
        self._models_loaded = 0
        self._models_loaded_lock = threading.Lock()

    def ensure_ready(self) -> None:
        """Pre-load every PANNs model copy for concurrent AED inference."""
        pool_size = self.cfg.aed_parallel_workers * self.cfg.max_concurrent_requests
        log.info(
            "Loading %d PANNs sound-event model cop%s (device=%s)",
            pool_size,
            "y" if pool_size == 1 else "ies",
            self.cfg.aed_device,
        )
        self._pool.ensure_ready(pool_size)
        log.info("PANNs model%s loaded", "" if pool_size == 1 else " copies")

    def _create_model(self) -> object:
        from panns_inference import SoundEventDetection, labels  # type: ignore

        with self._labels_lock:
            first = not self._labels
            if first:
                self._labels = list(labels)
        with self._models_loaded_lock:
            self._models_loaded += 1
            worker_index = self._models_loaded
        log.info(
            "Loading PANNs sound-event model %d (device=%s)",
            worker_index,
            self.cfg.aed_device,
        )
        model = SoundEventDetection(device=self.cfg.aed_device)
        if first:
            log.info("PANNs model loaded (%d classes)", len(self._labels))
        return model

    def _infer_segment(self, model: object, waveform: np.ndarray) -> np.ndarray:
        return np.asarray(model.inference(waveform[None, :]))[0]  # type: ignore[union-attr]

    def _events_for_probs(
        self,
        probs: np.ndarray,
        *,
        offset_s: float,
        segment_duration_s: float,
        params: AedParams,
    ) -> list[dict]:
        frames = probs.shape[0]
        frame_dt = segment_duration_s / frames if frames > 0 else 0.0
        events = events_from_framewise(
            probs,
            frame_dt,
            self._labels,
            min_score=params.min_score,
            top_k=params.top_k,
            merge_gap_s=params.merge_gap_s,
            min_duration_s=params.min_duration_s,
            exclude_speech=params.exclude_speech,
            prompt_allowlist=self.cfg.aed_prompt_allowlist,
        )
        if offset_s <= 0:
            return events
        return [
            {
                **event,
                "start": round(event["start"] + offset_s, 3),
                "end": round(event["end"] + offset_s, 3),
            }
            for event in events
        ]

    def _update_debug_peaks(
        self,
        peaks: dict[int, dict],
        probs: np.ndarray,
        *,
        offset_s: float,
        segment_duration_s: float,
    ) -> None:
        frames = probs.shape[0]
        frame_dt = segment_duration_s / frames if frames > 0 else 0.0
        for row in peak_labels_from_framewise(probs, frame_dt, self._labels, top_n=527):
            cls = int(row["index"])
            current = peaks.get(cls)
            if current is None or row["score"] > current["score"]:
                peaks[cls] = {
                    "label": row["label"],
                    "index": cls,
                    "score": row["score"],
                    "at": round(row["at"] + offset_s, 3),
                }

    def _detect_chunked(
        self,
        *,
        path: str | None,
        waveform: np.ndarray | None,
        duration_s: float,
        params: AedParams,
        debug_top_n: int,
    ) -> tuple[list[dict], list[dict]]:
        """Run PANNs window-by-window and emit events without a full probs matrix."""
        sr = self.cfg.aed_sample_rate
        n = int(round(duration_s * sr))
        window = int(self.cfg.aed_window_s * sr) if self.cfg.aed_window_s > 0 else 0
        overlap = max(int(self.cfg.aed_overlap_s * sr), 0)
        overlap = min(overlap, max(window - 1, 0))
        min_tail = int(0.5 * sr)
        jobs = plan_aed_windows(
            n,
            window_samples=window,
            overlap_samples=overlap,
            min_tail_samples=min_tail,
        )
        if not jobs:
            return [], []

        if len(jobs) == 1:
            return self._detect_single_window(
                path=path,
                waveform=waveform,
                job=jobs[0],
                params=params,
                debug_top_n=debug_top_n,
                sample_rate=sr,
                overlap_half_samples=overlap // 2,
            )

        workers = min(len(jobs), self.cfg.aed_parallel_workers)
        collect_debug = debug_top_n > 0
        log.info(
            "AED chunked inference: %.1fs audio in %d window(s) of %ds (overlap=%.1fs, workers=%d)",
            duration_s,
            len(jobs),
            self.cfg.aed_window_s,
            self.cfg.aed_overlap_s,
            workers,
        )

        if workers <= 1:
            window_results = [
                self._run_window(
                    path=path,
                    waveform=waveform,
                    job=job,
                    params=params,
                    sample_rate=sr,
                    overlap_half_samples=overlap // 2,
                    collect_debug=collect_debug,
                )
                for job in jobs
            ]
        else:
            window_results = self._run_windows_parallel(
                path=path,
                waveform=waveform,
                jobs=jobs,
                params=params,
                sample_rate=sr,
                overlap_half_samples=overlap // 2,
                collect_debug=collect_debug,
                workers=workers,
            )

        events: list[dict] = []
        debug_peak_chunks: list[dict[int, dict]] = []
        for job_events, peaks in window_results:
            events.extend(job_events)
            if collect_debug:
                debug_peak_chunks.append(peaks)

        merged = merge_sound_events(events, params.merge_gap_s)
        debug = sorted(
            merge_debug_peaks(debug_peak_chunks).values(),
            key=lambda row: (-row["score"], row["label"]),
        )[:debug_top_n]
        return merged, debug

    def _detect_single_window(
        self,
        *,
        path: str | None,
        waveform: np.ndarray | None,
        job: AedWindowJob,
        params: AedParams,
        debug_top_n: int,
        sample_rate: int,
        overlap_half_samples: int,
    ) -> tuple[list[dict], list[dict]]:
        events, peaks = self._run_window(
            path=path,
            waveform=waveform,
            job=job,
            params=params,
            sample_rate=sample_rate,
            overlap_half_samples=overlap_half_samples,
            collect_debug=debug_top_n > 0,
        )
        debug = sorted(peaks.values(), key=lambda row: (-row["score"], row["label"]))[:debug_top_n]
        return events, debug

    def _run_windows_parallel(
        self,
        *,
        path: str | None,
        waveform: np.ndarray | None,
        jobs: list[AedWindowJob],
        params: AedParams,
        sample_rate: int,
        overlap_half_samples: int,
        collect_debug: bool,
        workers: int,
    ) -> list[tuple[list[dict], dict[int, dict]]]:
        results: list[tuple[list[dict], dict[int, dict]] | None] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self._run_window,
                    path=path,
                    waveform=waveform,
                    job=job,
                    params=params,
                    sample_rate=sample_rate,
                    overlap_half_samples=overlap_half_samples,
                    collect_debug=collect_debug,
                ): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                results[job.index] = future.result()
        assert all(item is not None for item in results)
        return results  # type: ignore[return-value]

    def _run_window(
        self,
        *,
        path: str | None,
        waveform: np.ndarray | None,
        job: AedWindowJob,
        params: AedParams,
        sample_rate: int,
        overlap_half_samples: int,
        collect_debug: bool,
    ) -> tuple[list[dict], dict[int, dict]]:
        if waveform is None:
            from audio_intel.audio.decode import load_audio_window

            start_s = job.start / sample_rate
            seg_duration_s = (job.end - job.start) / sample_rate
            segment = load_audio_window(path, sample_rate, start_s, seg_duration_s)  # type: ignore[arg-type]
        else:
            segment = waveform[job.start : job.end]

        model = self._pool.acquire()
        try:
            probs = self._infer_segment(model, segment)
        finally:
            self._pool.release(model)

        probs, segment_duration_s, offset_s = trim_window_framewise_probs(
            probs,
            segment.shape[0],
            start_sample=job.start,
            end_sample=job.end,
            total_samples=job.total_samples,
            overlap_half_samples=overlap_half_samples,
            sample_rate=sample_rate,
        )
        events = self._events_for_probs(
            probs,
            offset_s=offset_s,
            segment_duration_s=segment_duration_s,
            params=params,
        )
        peaks: dict[int, dict] = {}
        if collect_debug:
            self._update_debug_peaks(
                peaks,
                probs,
                offset_s=offset_s,
                segment_duration_s=segment_duration_s,
            )
        return events, peaks

    def detect_file(
        self,
        path: str,
        *,
        params: AedParams | None = None,
        debug_top_n: int = 0,
        duration_s: float | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Detect sound events from a prepared AED WAV path."""
        p = params or AedParams.from_config(self.cfg)
        if duration_s is None:
            from audio_intel.audio.decode import probe_media_duration

            duration_s = probe_media_duration(path)
        duration_s = round(duration_s, 1)
        started = time.perf_counter()
        events, debug = self._detect_chunked(
            path=path,
            waveform=None,
            duration_s=duration_s,
            params=p,
            debug_top_n=debug_top_n,
        )
        log.info(
            "AED post-processing in %.1fs: %d events (min_score=%.2f, top_k=%d)",
            time.perf_counter() - started,
            len(events),
            p.min_score,
            p.top_k,
        )
        return events, debug

    def detect(
        self,
        waveform: np.ndarray,
        *,
        params: AedParams | None = None,
        debug_top_n: int = 0,
    ) -> tuple[list[dict], list[dict]]:
        """Detect non-speech sound events in a 32 kHz mono waveform.

        @param params — effective thresholds; defaults to the server config.
        @param debug_top_n — when > 0, also return the top-N peak-scoring classes
          (unfiltered) for diagnostics/tuning.
        @returns ``(events, debug_peaks)``. ``debug_peaks`` is empty unless
          ``debug_top_n`` > 0.
        """
        if waveform.size == 0:
            return [], []
        p = params or AedParams.from_config(self.cfg)
        started = time.perf_counter()
        duration = waveform.shape[0] / self.cfg.aed_sample_rate
        events, debug = self._detect_chunked(
            path=None,
            waveform=waveform,
            duration_s=duration,
            params=p,
            debug_top_n=debug_top_n,
        )
        log.info(
            "AED post-processing in %.1fs: %d events (min_score=%.2f, top_k=%d)",
            time.perf_counter() - started,
            len(events),
            p.min_score,
            p.top_k,
        )
        return events, debug
