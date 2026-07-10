# audio-intel

HTTP service for audio transcription: speech recognition (faster-whisper), language detection,
and optional speaker diarization (pyannote), sound events (PANNs / AudioSet),
and word-level alignment (WhisperX CTC).

One file per request via an OpenAI-compatible API (`POST /v1/audio/transcriptions`).

## Features

| Component    | Technology     | Description                                  |
| ------------ | -------------- | -------------------------------------------- |
| ASR          | faster-whisper | VAD → chunks → transcription with timestamps |
| Alignment    | WhisperX CTC   | Refines word-level timestamps                |
| Diarization  | pyannote 3.x   | `speaker_id` on speech segments              |
| Sound events | PANNs CNN14    | Non-speech events on the same time axis      |

Input: any format supported by ffmpeg (wav, mp3, mp4, …).

## Requirements

- Python 3.10+
- **ffmpeg** and **ffprobe** on `PATH`
- For GPU: NVIDIA driver, CUDA-compatible `torch` + `ctranslate2`
- Docker: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Optional extras: `[server]`, `[align]`, `[vad]`, `[all]`.

## Quick start

### Local (virtual env)

```powershell
# Windows
.\scripts\setup.ps1
.\scripts\run.ps1
```

```bash
# Linux / macOS
chmod +x scripts/*.sh
./scripts/setup.sh
./scripts/run.sh
```

`setup` creates `.venv`, runs `pip install -e ".[all]"`, and copies `.env.example` → `.env`.

Manual run:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[all]"
cp .env.example .env
audio-intel
```

The server listens on `AUDIO_INTEL_PORT` (default `8080`).

### Docker Compose (GPU)

```bash
cp .env.example .env
docker compose up -d --build
```

Health check: `curl http://localhost:8080/health`

Model caches on the host: `./data/whisper`, `./data/panns`.

## Configuration

Variables are read from `.env` (via `python-dotenv`) or the environment; **shell variables take precedence**.

Template: [`.env.example`](.env.example). Main groups:

| Prefix                    | Purpose                                 |
| ------------------------- | --------------------------------------- |
| `AUDIO_INTEL_*`           | HTTP port                               |
| `LOG_LEVEL`               | Log level                               |
| `MAX_CONCURRENT_REQUESTS` | Parallel HTTP requests                  |
| `ASR_*`                   | Model, device, decode, parallel workers |
| `MODEL_CACHE_DIR`         | Model cache (HF + faster-whisper)       |
| `HF_TOKEN`                | Hugging Face token for pyannote         |
| `VAD_*`                   | Silero VAD and chunk splitting          |
| `SOUND_EVENTS_*`          | PANNs — sound events                    |
| `SPEAKERS_*`              | pyannote — speakers + embeddings        |
| `WORD_ALIGN_*`            | WhisperX CTC (requires `[align]` extra) |

`MODEL_CACHE_HOST_DIR`, `PANNS_CACHE_HOST_DIR`, and `CUDA_VISIBLE_DEVICES`
are used only by **Docker Compose** (volume mounts), not by the Python app.

## HTTP API

### `GET /health`

Liveness probe. Returns `status`, `model`, and flags for enabled capabilities.

### `POST /v1/audio/transcriptions`

OpenAI-compatible multipart endpoint. **One file per request.**

**Form fields:**

| Field                           | Default        | Description                          |
| ------------------------------- | -------------- | ------------------------------------ |
| `file`                          | —              | Audio/video file (required)          |
| `language`                      | auto           | Force language (`uk`, `en`, …)       |
| `response_format`               | `verbose_json` | `verbose_json` or `text`             |
| `align`                         | `false`        | WhisperX CTC alignment               |
| `diarize`                       | `false`        | Speaker diarization                  |
| `sound_events`                  | `false`        | PANNs sound events                   |
| `aed_min_score`, `aed_top_k`, … | server default | Per-request AED tuning               |
| `aed_debug_top_n`               | `0`            | Top-N AudioSet peaks for diagnostics |

Post-ASR stages require the matching `*_ENABLED=1` on the server **and** `=true` in the request.

**Examples:**

```bash
# Transcription only
curl -F "file=@audio.wav" http://localhost:8080/v1/audio/transcriptions

# Full analysis
curl -F "file=@audio.wav" \
  -F "align=true" -F "diarize=true" -F "sound_events=true" \
  http://localhost:8080/v1/audio/transcriptions
```

### Response format

A single timeline in `segments`, distinguished by the `kind` field:

**`speech`** — `text`, `confidence`, optionally `language`, `speaker_id`, `words[]`

**`sound`** — AudioSet `label`, `index`, `score`, `prompt_relevant`

Top-level fields:

- `languages` — all detected languages in order
- `confidence` — duration-weighted mean
- `speakers` — roster after diarization (`id`, `speech_seconds`, `segment_count`)
- with `align=true`: `alignment_requested`, `alignment_applied`, `alignment_failed`, `aligned_at`

## How it works

At the start of each request, the input is converted **once** into a `MediaWorkspace` (temporary directory):

| File              | Format      | Stages                           |
| ----------------- | ----------- | -------------------------------- |
| `whisper_16k.wav` | mono 16 kHz | VAD, Whisper, align, diarization |
| `aed_32k.wav`     | mono 32 kHz | PANNs (when `sound_events`)      |

16 kHz and 32 kHz conversion runs **in parallel** (two ffmpeg processes). Stages read windows from the prepared WAV files;
the directory is removed when the request finishes.

```
upload (1 file) → MediaWorkspace → VAD → Whisper (parallel chunks)
                → [align] → [diarize ∥ sound_events] → JSON
```

Diarization and sound events run in parallel when both are enabled in the request.

Long files (> `SPEAKERS_CHUNK_S`, default 600 s) are diarized in **chunks** with overlap;
speakers are linked across chunks via cosine similarity of embeddings in the overlap zone.
Chunk inference runs **in parallel** (`SPEAKERS_PARALLEL_WORKERS` pipeline copies).
`SPEAKERS_CHUNK_S=0` disables chunking (one pass over the full file).

AED inference over 60 s windows runs **in parallel** (`SOUND_EVENTS_PARALLEL_WORKERS` PANNs copies).

VAD reads `whisper_16k.wav` in a **streaming** fashion in blocks (`VAD_STREAM_BLOCK_S`), without loading the entire file into RAM.

## Project layout

```
src/                         # import: audio_intel.*
├── audio/                   # MediaWorkspace, decode, chunking
├── common/                  # ModelPool
├── vad/                     # Silero VAD, VadPipeline
├── transcribe/              # WhisperEngine, HTTP Transcriber
├── align/                   # WhisperX CTCAligner
├── quality/                 # text normalization
├── diarization/             # pyannote + speaker merge/embeddings
├── events/                  # PANNs AED
└── server/                  # FastAPI app
```

## pip extras

```bash
pip install -e .              # core: faster-whisper, decode
pip install -e ".[server]"    # + FastAPI, pyannote, PANNs
pip install -e ".[align]"     # + WhisperX
pip install -e ".[vad]"       # + silero-vad
pip install -e ".[all]"       # full stack
pip install -e ".[dev]"       # pytest, ruff, pre-commit
```

## Development

Recommended editor setup: [`.vscode/extensions.json`](.vscode/extensions.json) and [`.vscode/settings.json`](.vscode/settings.json) (Python, Ruff, pytest).

```bash
pip install -e ".[dev]"
pre-commit install          # git hook: ruff + mdformat
python -m pytest tests/
```

Before each commit, these run automatically:

- **ruff** — lint (`--fix`) and format for `*.py`
- **mdformat** — format `*.md`

Run manually on all files:

```bash
pre-commit run --all-files
```

Unit tests do not require a GPU (chunking, normalization, alignment bridge, AED helpers).

## License

MIT
