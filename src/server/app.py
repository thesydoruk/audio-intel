"""FastAPI app exposing an OpenAI-compatible transcription endpoint.

Endpoints:
  - ``POST /v1/audio/transcriptions`` (multipart) → verbose JSON
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
``segment_count``.
"""

from __future__ import annotations

import logging
import os
import tempfile

from audio_intel.config import load_config
from audio_intel.server.transcription import (
    build_aed_overrides,
    build_transcription_body,
    normalize_language,
)
from audio_intel.transcribe.service import Transcriber
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

cfg = load_config()
logging.basicConfig(
    level=getattr(logging, cfg.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("audio-intel")

app = FastAPI(title="audio-intel", version="1.0")
transcriber = Transcriber(cfg)


@app.get("/health")
def health() -> dict:
    """Liveness probe used by the docker healthcheck."""
    return {
        "status": "ok",
        "model": cfg.model,
        "diarization_enabled": cfg.diarization_enabled,
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
    payload = file.file.read()
    size_kb = len(payload) / 1024
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

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(payload)
        tmp_path = tmp.name

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=cfg.port, log_level=cfg.log_level.lower())
