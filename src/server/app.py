"""FastAPI app exposing an OpenAI-compatible transcription endpoint.

Endpoints:
  - ``POST /v1/audio/transcriptions`` (multipart) → verbose JSON
  - ``POST /v1/speakers/embed`` (multipart) → speaker vectors for known time spans
  - ``GET /health`` → liveness

The response keeps the OpenAI ``language`` string (first detected language, for
client compatibility) and adds ``languages`` (distinct, in order of appearance).
``segments`` is a single timeline carrying both speech and non-speech sounds,
discriminated by ``kind``:
  - ``"speech"`` → ``text``, ``confidence``, optional per-chunk ``language``,
    optional ``speaker_id`` when diarization is enabled.
  - ``"sound"``  → AudioSet ``label`` (English), class ``index``, ``score`` and
    ``prompt_relevant`` (whether it should be fed to downstream LLMs).

Per-request form flags (default ``false``): ``align``, ``diarize``, ``sound_events``.
Server-side ``WORD_ALIGN_ENABLED``, ``SPEAKERS_ENABLED``, and ``SOUND_EVENTS_ENABLED`` must
also be on for the corresponding stage to run.

Top-level ``speakers`` roster (when diarization ran): ``id``, ``speech_seconds``,
``segment_count``, and — when enough speech — ``embedding`` (L2-normalized) with
``embedding_model``. Vectors from different models are not comparable.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import shutil
import tempfile

from audio_intel.config import load_config
from audio_intel.server.transcription import (
    build_aed_overrides,
    build_transcription_body,
    normalize_language,
    parse_speaker_spans,
)
from audio_intel.transcribe.service import Transcriber
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

cfg = load_config()
logging.basicConfig(
    level=getattr(logging, cfg.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("audio-intel")

try:
    VERSION = importlib.metadata.version("audio-intel")
except importlib.metadata.PackageNotFoundError:  # running from a bare source tree
    VERSION = "0+unknown"

app = FastAPI(title="audio-intel", version=VERSION)
transcriber = Transcriber(cfg)


@app.get("/health")
def health() -> dict:
    """Liveness probe used by the docker healthcheck."""
    return {
        "status": "ok",
        "version": VERSION,
        "model": cfg.model,
        "diarization_enabled": cfg.diarization_enabled,
        "speaker_embedding_model": (
            cfg.diarization_embedding_model if cfg.diarization_enabled else None
        ),
        "alignment_enabled": cfg.alignment_enabled,
    }


@app.post("/v1/audio/transcriptions")
def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: str = Form("verbose_json"),
    temperature: float = Form(0.0),
    language: str | None = Form(None),
    aed_min_score: float | None = Form(None),
    aed_top_k: int | None = Form(None),
    aed_merge_gap_s: float | None = Form(None),
    aed_min_duration_s: float | None = Form(None),
    aed_exclude_speech: bool | None = Form(None),
    aed_debug_top_n: int = Form(0),
    align: bool = Form(False),
    diarize: bool = Form(False),
    sound_events: bool = Form(False),
):
    """Transcribe an uploaded audio/video file.

    Sync handler so FastAPI runs it in a worker thread; excess requests wait on
    the transcriber's concurrency semaphore (``MAX_CONCURRENT_REQUESTS``).
    """
    del model, temperature  # OpenAI-compatible fields; ignored by this service.

    forced = normalize_language(language)
    aed_overrides = build_aed_overrides(
        aed_min_score=aed_min_score,
        aed_top_k=aed_top_k,
        aed_merge_gap_s=aed_merge_gap_s,
        aed_min_duration_s=aed_min_duration_s,
        aed_exclude_speech=aed_exclude_speech,
    )

    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp, length=1024 * 1024)
        tmp_path = tmp.name
    size_kb = os.path.getsize(tmp_path) / 1024
    log.info(
        "Transcription request: %s (%.1f KB, language=%s, align=%s, "
        "diarize=%s, sound_events=%s, aed_debug_top_n=%d)",
        file.filename or "upload",
        size_kb,
        forced or "auto",
        align,
        diarize,
        sound_events,
        aed_debug_top_n,
    )

    try:
        result = transcriber.transcribe(
            tmp_path,
            override_language=forced,
            align=align,
            diarize=diarize,
            sound_events=sound_events,
            aed_overrides=aed_overrides,
            aed_debug_top_n=aed_debug_top_n,
        )
        log.info(
            "Transcription response: %.1fs, %d segments, languages=%s, "
            "speakers=%d, align_failed=%s",
            result["duration"],
            len(result["segments"]),
            result["languages"],
            len(result.get("speakers") or []),
            result.get("alignment_failed"),
        )
    finally:
        os.unlink(tmp_path)

    if response_format == "text":
        return PlainTextResponse(result["text"])

    return JSONResponse(build_transcription_body(result))


@app.post("/v1/speakers/embed")
def embed_speakers(
    file: UploadFile = File(...),
    speakers: str = Form(...),
    min_speech_s: float = Form(3.0),
    max_clip_s: float = Form(30.0),
):
    """Embed speakers from known time spans, skipping ASR and diarization.

    ``speakers`` is JSON ``{"<id>": [[start_s, end_s], ...], ...}``. Each roster row
    returns ``embedding`` (``null`` below ``min_speech_s``) from the first
    ``max_clip_s`` seconds of its spans — the same clip rule transcription uses.
    """
    if not cfg.diarization_enabled:
        raise HTTPException(status_code=503, detail="SPEAKERS_ENABLED=0")
    try:
        spans_by_speaker = parse_speaker_spans(speakers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp, length=1024 * 1024)
        tmp_path = tmp.name
    try:
        result = transcriber.embed_speakers(
            tmp_path,
            spans_by_speaker,
            min_speech_s=min_speech_s,
            max_clip_s=max_clip_s,
        )
    finally:
        os.unlink(tmp_path)
    log.info(
        "Speaker embed response: %s, %d/%d speakers embedded",
        file.filename or "upload",
        sum(1 for row in result["speakers"] if row["embedding"]),
        len(result["speakers"]),
    )
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=cfg.port, log_level=cfg.log_level.lower())
