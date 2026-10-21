"""A separation job: tracks per speaker, then the transcript of each track.

Stages, each a checkpoint for cancellation:

1. ``separating``   — the separation pipeline (Demucs, pyannote on the vocals
   stem, SepFormer in overlaps) writes one 48 kHz track per speaker and the
   background bed. Inside the service the pipeline borrows the service's
   loaded pyannote pipeline (and, with ``SEPARATE_EMBEDDINGS=wespeaker``, its
   embedding model) through ``_providers`` and takes GPU tokens at batch
   priority, so interactive requests get the card at the next stage boundary.
2. ``transcribing`` — Whisper on every speaker track separately, speaker known
   before decoding, no diarization. Music is gone from the track, so the
   hallucination filter has less to do.
3. ``embedding``    — the service's speaker vectors for the roster, from each
   speaker's own track, in the same space as the service's stored voices.
4. ``sound_events`` — PANNs on the background stem: effects without voices.
5. ``finishing``    — one timeline, the roster, and ``tracks`` with the file
   names ``GET /v1/jobs/{id}/files/{name}`` serves.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from audio_intel.config import Config
    from audio_intel.jobs.store import Job, JobStore

log = logging.getLogger("audio-intel")

Progress = Callable[[str, float | None], None]


class JobCancelled(Exception):
    """Raised at a stage boundary when the job was cancelled."""


class SeparationEngine:
    """The separation pipeline as the service runs it; its models form a model group."""

    def __init__(self, cfg: Config, providers: dict[str, Any] | None = None) -> None:
        self.cfg = cfg
        self.providers = dict(providers or {})
        self._pipeline: Any = None
        self._lock = threading.Lock()

    def build_config(self) -> dict[str, Any]:
        from audio_intel.separate.config import load_config

        overrides: dict[str, Any] = {
            "embeddings": {"backend": self.cfg.separate_embeddings},
            "cache": {"dir": str(Path(self.cfg.jobs_dir) / "cache")},
            "device": {"device": self.cfg.separate_device, "unload_idle_models": False},
            "output": {"write_html_report": False},
            "pipeline": {"resume": True},
            "_providers": self.providers,
        }
        return load_config(self.cfg.separate_config or None, overrides=overrides)

    def pipeline(self) -> Any:
        with self._lock:
            if self._pipeline is None:
                from audio_intel.separate.pipeline.orchestrator import SpeakerExtractPipeline

                self._pipeline = SpeakerExtractPipeline(self.build_config())
            return self._pipeline

    def load(self) -> None:
        """Model-group loader: bring up the stacks the pipeline owns itself."""
        pipeline = self.pipeline()
        pipeline.separator().ensure_loaded()
        pipeline.diarizer().ensure_loaded()
        pipeline.embedder().ensure_loaded()
        pipeline.tse().ensure_loaded()

    def unload(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                self._pipeline._unload_all()

    def run(self, input_path: Path, output_dir: Path, progress: Callable[[str], None]) -> Any:
        return self.pipeline().run(input_path, output_dir, progress=progress)


def _check(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise JobCancelled()


def _speaker_spans(result: Any, speaker: str) -> list[tuple[float, float]]:
    return [
        (float(seg.span.start), float(seg.span.end))
        for seg in result.segments
        if seg.local_speaker == speaker
    ]


def run_separation_job(
    job: Job,
    *,
    store: JobStore,
    transcriber: Any,
    cancel: threading.Event,
) -> dict[str, Any]:
    """Run one job to its result dict; raises :class:`JobCancelled` on Stop."""
    params = job.params
    input_path = store.job_dir(job.id) / params["input_file"]
    out_dir = store.output_dir(job.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    language = params.get("language")
    align = bool(params.get("align", False))
    sound_events = bool(params.get("sound_events", False))
    started = time.perf_counter()

    def progress(stage: str, fraction: float | None = None) -> None:
        store.set_progress(job.id, stage, fraction)

    # 1. separation --------------------------------------------------------------
    _check(cancel)
    progress("separating", 0.0)
    with transcriber.separation_models():
        sep = transcriber.separation.run(
            input_path, out_dir, progress=lambda m: log.info("[%s] %s", job.id, m)
        )
    log.info(
        "[%s] separation done in %.0fs: %d speakers",
        job.id,
        time.perf_counter() - started,
        len(sep.speaker_files),
    )

    # 2. ASR per speaker track -------------------------------------------------------
    speakers = sorted(sep.speaker_files)
    segments: list[dict[str, Any]] = []
    languages: list[str] = []
    dropped = 0
    texts: list[tuple[float, str]] = []
    for index, speaker in enumerate(speakers):
        _check(cancel)
        progress(
            f"transcribing {index + 1}/{len(speakers)}", 0.2 + 0.6 * index / max(1, len(speakers))
        )
        track = Path(sep.speaker_files[speaker])
        result = transcriber.transcribe_batch(str(track), override_language=language, align=align)
        for seg in result["segments"]:
            if seg.get("kind") == "speech":
                seg["speaker_id"] = speaker
                segments.append(seg)
                texts.append((float(seg["start"]), seg["text"]))
        for lang in result.get("languages", []):
            if lang not in languages:
                languages.append(lang)
        dropped += int(result.get("dropped_segments", 0))

    # 3. roster embeddings ------------------------------------------------------------
    _check(cancel)
    progress("embedding", 0.85)
    roster: list[dict[str, Any]] = []
    for speaker in speakers:
        spans = _speaker_spans(sep, speaker)
        row: dict[str, Any] = {
            "id": speaker,
            "speech_seconds": round(sum(e - s for s, e in spans), 3),
            "segment_count": sum(1 for seg in segments if seg.get("speaker_id") == speaker),
        }
        if spans:
            try:
                embedded = transcriber.embed_spans(sep.speaker_files[speaker], {speaker: spans})
                vector = embedded["speakers"][0].get("embedding") if embedded["speakers"] else None
                if vector:
                    row["embedding"] = vector
                    row["embedding_model"] = embedded["embedding_model"]
            except Exception:  # noqa: BLE001 — the roster is best-effort
                log.exception("[%s] embedding failed for %s", job.id, speaker)
        roster.append(row)

    # 4. sound events on the background -------------------------------------------------
    sounds: list[dict[str, Any]] = []
    if sound_events and sep.background_path:
        _check(cancel)
        progress("sound_events", 0.9)
        sounds = transcriber.detect_sounds_file(sep.background_path)

    # 5. assemble ------------------------------------------------------------------------
    progress("finishing", 0.95)
    timeline = sorted(segments + sounds, key=lambda s: float(s["start"]))
    texts.sort()
    tracks: dict[str, Any] = {
        "speakers": {spk: Path(path).name for spk, path in sep.speaker_files.items()},
        "background": Path(sep.background_path).name if sep.background_path else None,
        "speech": Path(sep.speech_stem_path).name if sep.speech_stem_path else None,
        "master": Path(sep.master_path).name if sep.master_path else None,
        "rttm": Path(sep.rttm_path).name if sep.rttm_path else None,
        "metadata": Path(sep.metadata_path).name if sep.metadata_path else None,
    }
    from audio_intel.transcribe.segments import overall_confidence

    return {
        "task": "transcribe",
        "language": languages[0] if languages else "unknown",
        "languages": languages,
        "confidence": overall_confidence(segments),
        "duration": round(float(sep.input_info.duration), 3),
        "text": " ".join(text for _, text in texts).strip(),
        "segments": timeline,
        "speakers": roster,
        "tracks": tracks,
        "overlaps": [
            {"start": round(float(o.span.start), 3), "end": round(float(o.span.end), 3)}
            for o in sep.overlaps
        ],
        **({"dropped_segments": dropped} if dropped else {}),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
