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

Models load in the lifespan handler, so importing this module is cheap and the
server only accepts connections once Whisper and VAD are in memory. Requests
beyond ``MAX_CONCURRENT_REQUESTS`` wait in line; past ``MAX_QUEUED_REQUESTS``
they get ``503`` at once instead of tying up a server thread.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from audio_intel.config import load_config
from audio_intel.jobs.lane import BatchBusyError
from audio_intel.jobs.service import JobService
from audio_intel.server.health import build_health_body, gpu_snapshot
from audio_intel.server.transcription import (
    build_aed_overrides,
    build_transcription_body,
    normalize_language,
    parse_speaker_spans,
)
from audio_intel.transcribe.service import ServiceBusyError, Transcriber
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

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

# Copy uploads to disk in 1 MiB blocks (the whole file is never held in memory).
_UPLOAD_COPY_BLOCK = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.transcriber = Transcriber(cfg)
    app.state.jobs = JobService(cfg, app.state.transcriber) if cfg.separate_enabled else None
    try:
        yield
    finally:
        if app.state.jobs is not None:
            app.state.jobs.close()
        app.state.transcriber.close()


app = FastAPI(title="audio-intel", version=VERSION, lifespan=lifespan)


def _transcriber(request: Request) -> Transcriber:
    return request.app.state.transcriber


def _jobs(request: Request) -> JobService:
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(status_code=404, detail="SEPARATE_ENABLED=0: no job lane")
    return jobs


def _spool_upload(file: UploadFile) -> str:
    """Write the multipart upload to a temp file and return its path."""
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp, length=_UPLOAD_COPY_BLOCK)
        return tmp.name


def _busy(exc: ServiceBusyError) -> HTTPException:
    log.warning("Rejecting request: %s", exc)
    return HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "30"})


@app.get("/health")
async def health(request: Request) -> dict:
    """Liveness probe used by the docker healthcheck, plus what the process holds.

    Async on purpose: it must answer even while every worker thread of the
    server is busy with (or waiting on) a transcription. Beyond the flags, the
    body reports resident model groups, request lanes, GPU tokens and VRAM so
    an outside scheduler can see what this instance is doing.
    """
    body = build_health_body(
        version=VERSION,
        cfg=cfg,
        service=getattr(request.app.state, "transcriber", None),
        gpu=gpu_snapshot(),
    )
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is not None:
        body.update(jobs.snapshot())
    return body


@app.post("/v1/audio/transcriptions")
def transcribe(
    request: Request,
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
    separate: bool = Form(False),
):
    """Transcribe an uploaded audio/video file.

    Sync handler so FastAPI runs it in a worker thread; excess requests wait on
    the transcriber's concurrency semaphore (``MAX_CONCURRENT_REQUESTS``).

    With ``separate=true`` the file becomes a separation job instead: the
    answer is ``202`` with the job id, and ``GET /v1/jobs/{id}`` has the
    transcript per speaker track plus the tracks once it is done.
    """
    del model, temperature  # OpenAI-compatible fields; ignored by this service.

    try:
        forced = normalize_language(language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    aed_overrides = build_aed_overrides(
        aed_min_score=aed_min_score,
        aed_top_k=aed_top_k,
        aed_merge_gap_s=aed_merge_gap_s,
        aed_min_duration_s=aed_min_duration_s,
        aed_exclude_speech=aed_exclude_speech,
    )

    tmp_path = _spool_upload(file)
    if separate:
        jobs = _jobs(request)
        try:
            job = jobs.submit_separation(
                tmp_path,
                file.filename or "upload",
                {"language": forced, "align": align, "sound_events": sound_events},
            )
        except BatchBusyError as exc:
            os.unlink(tmp_path)
            raise _busy(ServiceBusyError(str(exc))) from exc
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        log.info("Separation job %s queued: %s", job.id, file.filename or "upload")
        return JSONResponse(
            {"id": job.id, "status": job.status, "url": f"/v1/jobs/{job.id}"},
            status_code=202,
            headers={"Location": f"/v1/jobs/{job.id}"},
        )
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
        result = _transcriber(request).transcribe(
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
    except ServiceBusyError as exc:
        raise _busy(exc) from exc
    finally:
        os.unlink(tmp_path)

    if response_format == "text":
        return PlainTextResponse(result["text"])

    return JSONResponse(build_transcription_body(result))


@app.post("/v1/speakers/embed")
def embed_speakers(
    request: Request,
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

    tmp_path = _spool_upload(file)
    try:
        result = _transcriber(request).embed_speakers(
            tmp_path,
            spans_by_speaker,
            min_speech_s=min_speech_s,
            max_clip_s=max_clip_s,
        )
    except ServiceBusyError as exc:
        raise _busy(exc) from exc
    finally:
        os.unlink(tmp_path)
    log.info(
        "Speaker embed response: %s, %d/%d speakers embedded",
        file.filename or "upload",
        sum(1 for row in result["speakers"] if row["embedding"]),
        len(result["speakers"]),
    )
    return JSONResponse(result)


@app.get("/v1/jobs")
async def list_jobs(request: Request, limit: int = 50) -> dict:
    return {"jobs": _jobs(request).list(max(1, min(limit, 500)))}


@app.get("/v1/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> dict:
    """State, stage and progress of a job; its result once ``status`` is ``done``."""
    body = _jobs(request).get(job_id)
    if body is None:
        raise HTTPException(status_code=404, detail="no such job")
    return body


@app.get("/v1/jobs/{job_id}/files/{name}")
async def get_job_file(request: Request, job_id: str, name: str):
    """One output file of a finished job, by the name ``result.tracks`` lists."""
    path = _jobs(request).file_path(job_id, name)
    if path is None:
        raise HTTPException(status_code=404, detail="no such file")
    return FileResponse(path, filename=name)


@app.delete("/v1/jobs/{job_id}")
async def delete_job(request: Request, job_id: str) -> dict:
    """Cancel a queued or running job, or remove a finished one and its files."""
    if not _jobs(request).cancel_or_delete(job_id):
        raise HTTPException(status_code=404, detail="no such job")
    return {"id": job_id, "deleted": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=cfg.port, log_level=cfg.log_level.lower())
