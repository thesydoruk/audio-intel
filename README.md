# audio-intel

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/thesydoruk/audio-intel/actions/workflows/ci.yml/badge.svg)](https://github.com/thesydoruk/audio-intel/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

An HTTP service that turns an audio or video file into a timeline: who said what and when,
in which language, plus non-speech sounds (applause, gunshots, music, …) on the same time
axis. It speaks the OpenAI transcription API, so existing clients work unchanged, and adds
speaker diarization, speaker voice vectors, sound events and word-level alignment on top.

| Stage           | Technology                 | What you get                                        |
| --------------- | -------------------------- | --------------------------------------------------- |
| ASR             | faster-whisper             | Text with timestamps, per-chunk language detection  |
| Alignment       | WhisperX CTC               | Precise word-level timestamps                       |
| Diarization     | pyannote 4.x (community-1) | `speaker_id` on each speech segment, speaker roster |
| Speaker vectors | WeSpeaker (community-1)    | A 256-d voice embedding per speaker                 |
| Sound events    | PANNs CNN14 / AudioSet     | Non-speech events with labels and scores            |

Input: anything ffmpeg can decode (wav, mp3, m4a, ogg, mp4, mov, …). One file per request.

## Contents

- [Choose how to install](#choose-how-to-install)
- [Install](#install) — [Docker](#docker-one-command) · [Docker Compose](#docker-compose) · [Local Python](#local-python)
- [Enable speaker diarization](#enable-speaker-diarization)
- [Uninstall](#uninstall)
- [Upgrade](#upgrade)
- [Configuration](#configuration)
- [HTTP API](#http-api)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Development](#development) · [Releases](#releases) · [License](#license)

## Choose how to install

| Way                                        | Pick it when                                                | You need                               |
| ------------------------------------------ | ----------------------------------------------------------- | -------------------------------------- |
| [Docker, one command](#docker-one-command) | You just want the service running on a GPU host             | Docker + NVIDIA Container Toolkit      |
| [Docker Compose](#docker-compose)          | You want settings in a `.env` file, or to build from source | The above + a clone of this repository |
| [Local Python](#local-python)              | You develop audio-intel or cannot use Docker                | Python 3.10+, ffmpeg, (CUDA for GPU)   |

**Hardware.** An NVIDIA GPU is strongly recommended: Whisper `large-v3` in float16 needs about
4.5 GB of VRAM per copy, and diarization adds about 2 GB. CPU works too (see
[CPU-only](#cpu-only)), but expect transcription to run many times slower than on a GPU.

**Disk.** The Docker image is about 6 GB to download (13 GB unpacked). Models are downloaded
on first use into a cache that survives restarts: Whisper `large-v3` about 3 GB, pyannote and
PANNs a few hundred MB.

## Install

### Docker (one command)

No clone needed. First check that Docker can see your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

If that prints your GPU, start the service:

```bash
docker run -d --name audio-intel --gpus all --restart unless-stopped \
  -p 8080:8080 \
  -v audio-intel-models:/var/lib/whisper \
  -v audio-intel-panns:/root/panns_data \
  -e HF_HOME=/var/lib/whisper \
  ghcr.io/thesydoruk/audio-intel:2.0.0
```

The two named volumes keep downloaded models across restarts and upgrades. On the first start
the container downloads Whisper before it answers, so give it a few minutes, then check:

```bash
curl http://localhost:8080/health
```

```json
{"status": "ok", "version": "2.0.0", "model": "large-v3", "diarization_enabled": false,
 "speaker_embedding_model": null, "alignment_enabled": false}
```

Transcribe a file:

```bash
curl -F "file=@interview.mp4" http://localhost:8080/v1/audio/transcriptions
```

Diarization is off by default because its models are gated on Hugging Face — see
[Enable speaker diarization](#enable-speaker-diarization). Any setting from
[`.env.example`](.env.example) can be passed with `-e NAME=value`.

### Docker Compose

```bash
git clone https://github.com/thesydoruk/audio-intel.git
cd audio-intel
cp .env.example .env          # edit it: model, devices, diarization, …
docker compose up -d          # pulls ghcr.io/thesydoruk/audio-intel
docker compose logs -f        # watch the first-start model download
curl http://localhost:8080/health
```

Models are cached in `./data/whisper` and `./data/panns` (override with
`MODEL_CACHE_HOST_DIR` / `PANNS_CACHE_HOST_DIR`). Pin another release with
`AUDIO_INTEL_VERSION=x.y.z` in `.env`.

To build the image from your checkout instead of pulling it (about 10 minutes):

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build
```

### Local Python

Needs Python 3.10+ and `ffmpeg` / `ffprobe` on `PATH`.

```bash
# Linux / macOS
./scripts/setup.sh
./scripts/run.sh
```

```powershell
# Windows
.\scripts\setup.ps1
.\scripts\run.ps1
```

`setup` creates `.venv`, installs the service with its dev tools and WhisperX alignment,
registers the pre-commit hook, and copies `.env.example` → `.env`. Re-running it is safe.

pip picks the default `torch` wheel for your platform: CUDA on Linux, **CPU-only on Windows**.
For a GPU build on Windows (or a specific CUDA version), point `setup` at the PyTorch index
first:

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu126 ./scripts/setup.sh
```

```powershell
$env:TORCH_INDEX = "https://download.pytorch.org/whl/cu126"; .\scripts\setup.ps1
```

Locally, `MODEL_CACHE_DIR` in `.env` decides where models go (default `./data/whisper`).

### CPU-only

Set these (in `.env`, or `-e` flags without `--gpus` for Docker):

```bash
ASR_DEVICE=cpu
ASR_COMPUTE_TYPE=int8
SOUND_EVENTS_DEVICE=cpu
SPEAKERS_DEVICE=cpu
ASR_CPU_THREADS=8          # roughly your physical core count
```

A smaller model (`ASR_MODEL=small` or `medium`) keeps CPU transcription usable.

## Enable speaker diarization

pyannote's models are free but gated: Hugging Face must know who downloads them.

1. Sign in to Hugging Face and open
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
   Accept the conditions (CC-BY-4.0) on that page.

1. Create a **read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   with the same account.

1. Turn diarization on and pass the token:

   ```bash
   # Docker: add to the docker run command
   -e SPEAKERS_ENABLED=1 -e HF_TOKEN=hf_xxx

   # Compose / local: in .env
   SPEAKERS_ENABLED=1
   HF_TOKEN=hf_xxx
   ```

1. Restart and check that `/health` shows `"diarization_enabled": true` and a
   `speaker_embedding_model`.

1. Ask for speakers per request with `diarize=true`:

   ```bash
   curl -F "file=@interview.mp4" -F "diarize=true" http://localhost:8080/v1/audio/transcriptions
   ```

The diarization models load lazily on the first `diarize=true` request, so that one request
takes a few extra seconds. If the token cannot fetch them, the request still returns its
transcript, only without `speaker_id`, and the log says why. Every later `diarize=true`
request retries the load, so once the terms are accepted no restart is needed.

## Uninstall

**Docker (one command)** — remove the container and image; add the volume line to also
delete the downloaded models:

```bash
docker rm -f audio-intel
docker rmi ghcr.io/thesydoruk/audio-intel:2.0.0
docker volume rm audio-intel-models audio-intel-panns     # models (optional)
```

**Docker Compose** — from the clone:

```bash
docker compose down --rmi all          # container, network and image
rm -rf data/whisper/* data/panns/*     # models (optional)
```

If you built locally, add `-f docker-compose.build.yml` to the `down` command so the
`audio-intel:local` image is removed too.

**Local Python** — undo `setup` (virtual environment, pre-commit hook, caches):

```bash
./scripts/uninstall.sh            # keeps models and .env
./scripts/uninstall.sh --purge    # also deletes data/whisper, data/panns and .env
```

```powershell
.\scripts\uninstall.ps1
.\scripts\uninstall.ps1 -Purge
```

A local install keeps Whisper in `MODEL_CACHE_DIR`, but pyannote goes to the Hugging Face
cache (`~/.cache/huggingface/hub`) unless `HF_HOME` is set. To reclaim that space too,
delete its `models--pyannote--*` directories.

## Upgrade

Pull the new tag and recreate the container; model volumes are reused:

```bash
docker pull ghcr.io/thesydoruk/audio-intel:<version>
docker rm -f audio-intel      # then re-run the docker run command with the new tag
# Compose: set AUDIO_INTEL_VERSION=<version> in .env, then
docker compose pull && docker compose up -d
```

Read the [release notes](https://github.com/thesydoruk/audio-intel/releases) first. In
particular, **2.0.0 changed the speaker embedding model**: 1.x vectors (512-d) are not
comparable with 2.x vectors (256-d), so any stored vectors must be re-embedded or dropped
(see [Speaker embedding models](#speaker-embedding-models)).

## Configuration

Settings come from environment variables or a `.env` file; **real environment variables win**
over `.env`. Every variable is documented with its values and effect in
[`.env.example`](.env.example). The ones you are most likely to change:

| Variable                  | Default    | What it changes                                                         |
| ------------------------- | ---------- | ----------------------------------------------------------------------- |
| `ASR_MODEL`               | `large-v3` | Whisper size: `large-v3-turbo` is much faster at slightly lower quality |
| `ASR_LANGUAGE`            | `auto`     | Force one language (`uk`, `en`, …) instead of detecting per chunk       |
| `ASR_DEVICE`              | `cuda`     | `cpu` to run without a GPU (pair with `ASR_COMPUTE_TYPE=int8`)          |
| `MAX_CONCURRENT_REQUESTS` | `2`        | Requests processed at once; the rest wait in line                       |
| `ASR_PARALLEL_WORKERS`    | `1`        | Whisper copies per request: faster on long files, more VRAM             |
| `SPEAKERS_ENABLED`        | `0`        | Load diarization (needs `HF_TOKEN`, see above)                          |
| `SOUND_EVENTS_ENABLED`    | `1`        | Load PANNs for sound events                                             |
| `WORD_ALIGN_ENABLED`      | `0`        | Load WhisperX CTC alignment models                                      |
| `AUDIO_INTEL_PORT`        | `8080`     | HTTP port                                                               |

Model copies multiply: `MAX_CONCURRENT_REQUESTS × ASR_PARALLEL_WORKERS` Whisper copies stay
loaded, and the same product applies to `SPEAKERS_PARALLEL_WORKERS` / `SPEAKERS_EMBED_WORKERS`.
Lower them first if you run out of VRAM.

Variable groups:

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
| `WORD_ALIGN_*`            | WhisperX CTC alignment                  |

`MODEL_CACHE_HOST_DIR`, `PANNS_CACHE_HOST_DIR`, `AUDIO_INTEL_VERSION` and
`CUDA_VISIBLE_DEVICES` are read only by Docker Compose, not by the Python app.

## HTTP API

### `GET /health`

Liveness probe. Returns `status`, `version`, `model`, flags for enabled capabilities, and
`speaker_embedding_model` when diarization is enabled.

### `POST /v1/audio/transcriptions`

OpenAI-compatible multipart endpoint. **One file per request.**

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

A post-ASR stage runs only when it is enabled on the server (`*_ENABLED=1`) **and** requested
(`=true`). Asking for a stage the server has disabled is not an error; it is skipped.

```bash
# Transcription only
curl -F "file=@audio.wav" http://localhost:8080/v1/audio/transcriptions

# Everything
curl -F "file=@audio.wav" \
  -F "align=true" -F "diarize=true" -F "sound_events=true" \
  http://localhost:8080/v1/audio/transcriptions
```

### Response

One timeline in `segments`, speech and sounds sorted by time and told apart by `kind`
(abridged):

```json
{
  "task": "transcribe",
  "language": "uk",
  "languages": ["uk", "en"],
  "confidence": 0.91,
  "duration": 312.4,
  "text": "Добрий день. …",
  "segments": [
    {"kind": "speech", "start": 0.52, "end": 3.1, "text": "Добрий день.",
     "confidence": 0.94, "language": "uk", "speaker_id": "spk_0"},
    {"kind": "sound", "start": 3.4, "end": 5.0, "label": "Applause",
     "index": 67, "score": 0.71, "prompt_relevant": true}
  ],
  "speakers": [
    {"id": "spk_0", "speech_seconds": 184.2, "segment_count": 41,
     "embedding": [0.031, -0.12, "…"],
     "embedding_model": "pyannote/speaker-diarization-community-1/embedding"}
  ]
}
```

- `speech` segments: `text`, `confidence`, optionally `language`, `speaker_id`, `words[]`
  (with `align=true`).
- `sound` segments: AudioSet `label` and class `index`, `score`, and `prompt_relevant`
  (whether it is worth feeding to a downstream LLM).
- `languages`: every detected language in order of appearance; `language` is the first,
  for OpenAI compatibility.
- `confidence`: duration-weighted mean over speech.
- `speakers` (with `diarize=true`): one row per speaker. Speakers with enough speech get an
  L2-normalized `embedding` and the `embedding_model` that produced it.
- With `align=true`: `alignment_requested`, `alignment_applied`, `alignment_failed`,
  `aligned_at`.

### `POST /v1/speakers/embed`

Speaker vectors for time spans the caller already knows — no ASR, no diarization.
Useful to re-embed stored speakers after `SPEAKERS_EMBEDDING_MODEL` changes. Needs
`SPEAKERS_ENABLED=1`.

| Field          | Default | Description                                               |
| -------------- | ------- | --------------------------------------------------------- |
| `file`         | —       | Audio/video file (required)                               |
| `speakers`     | —       | JSON `{"<id>": [[start_s, end_s], ...]}`, ≤ 256 speakers  |
| `min_speech_s` | `3.0`   | Below this much speech a speaker gets `embedding: null`   |
| `max_clip_s`   | `30.0`  | Only the first N seconds of each speaker's spans are used |

```bash
curl -F "file=@interview.mp4" \
  -F 'speakers={"host": [[0.5, 12.0], [40.2, 55.0]], "guest": [[12.3, 39.8]]}' \
  http://localhost:8080/v1/speakers/embed
```

Returns `embedding_model`, `dimension`, and `speakers[]` with `id`, `speech_seconds`,
`embedding`. The clip rule matches the transcription roster, so the vectors are comparable.

### Speaker embedding models

A speaker embedding is a fingerprint of a voice: two clips of the same person give vectors
with high cosine similarity. Vectors from **different models** live in different spaces
(and may differ in length), so never compare or average them. Every vector is tagged with
`embedding_model`, and a client that stores vectors must re-embed or drop them when that id
changes. The default is `pyannote/speaker-diarization-community-1/embedding` (256-d); 1.x
served `pyannote/embedding` (512-d).

Similarity thresholds are model-specific too. Community-1 scores the same voice higher than
the 1.x model, so a 1.x threshold maps up to keep the same false-accept rate: 0.70 → 0.80,
0.75 → 0.84, 0.82 → 0.90 (calibrated on 778 labeled speakers).

## How it works

```
upload (1 file) → MediaWorkspace → VAD → Whisper (parallel chunks)
                → [align] → [diarize ∥ sound_events] → JSON
```

1. **Decode once.** The upload is converted into a temporary `MediaWorkspace`: a 16 kHz mono
   WAV for speech stages and, with `sound_events`, a 32 kHz one for PANNs. Both conversions
   run in parallel, and every later stage reads windows from these files instead of
   re-decoding. The directory is deleted when the request finishes.
1. **Find speech.** Silero VAD streams through the 16 kHz WAV in blocks (`VAD_STREAM_BLOCK_S`)
   without loading the whole file, and cuts speech into chunks of at most `VAD_MAX_CHUNK_S`.
1. **Transcribe.** Whisper transcribes the chunks in parallel (`ASR_PARALLEL_WORKERS`),
   detecting the language per chunk, so mixed-language recordings come out right.
1. **Post-ASR stages**, as requested:
   - **Alignment** refines word timestamps with a WhisperX CTC model for the language.
   - **Diarization** runs pyannote. Files longer than `SPEAKERS_CHUNK_S` (default 600 s) are
     diarized in overlapping chunks in parallel (`SPEAKERS_PARALLEL_WORKERS`). Speakers are
     then linked across chunks by voice: first within the overlap, then against every
     speaker seen so far, and a final pass merges duplicate ids whose voices match
     (`SPEAKERS_LINK_THRESHOLD`). `SPEAKERS_CHUNK_S=0` diarizes the file in one pass.
   - **Sound events** run PANNs over 60 s windows in parallel
     (`SOUND_EVENTS_PARALLEL_WORKERS`); diarization and sound events run side by side.

Whisper, VAD, and the alignment models for `WORD_ALIGN_PRELOAD` languages load at startup.
PANNs, pyannote, and alignment models for other languages load on the first request that
needs them, which keeps startup VRAM low on shared GPUs.

## Troubleshooting

| Symptom                                                                                          | Cause and fix                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/health` does not answer for minutes after the first start                                      | Whisper is downloading (about 3 GB for `large-v3`). Follow `docker logs -f audio-intel`; the cache volume makes later starts fast.                                                |
| `whisper.device is 'cuda' but no CUDA GPU is available`                                          | The container has no GPU: add `--gpus all` and install the NVIDIA Container Toolkit, or switch to [CPU-only](#cpu-only). Locally, `torch` may be a CPU build (see `TORCH_INDEX`). |
| No `speaker_id` / `speakers` in the response                                                     | Needs both `SPEAKERS_ENABLED=1` on the server and `diarize=true` in the request. If both are set, look for `Speaker diarization` errors in the logs (next row).                   |
| Log: `Speaker diarization models failed to load` with `GatedRepoError 403` or `accept its terms` | The account behind `HF_TOKEN` has not accepted the model's conditions. Accept them on the model page; the next `diarize=true` request loads the models, no restart needed.        |
| `Could not load libtorchcodec` (local install)                                                   | pyannote 4 decodes audio through torchcodec, which needs FFmpeg 4–7 shared libraries. Install your distribution's ffmpeg libraries, or use the Docker image, which has them.      |
| `CUDA out of memory`                                                                             | Too many model copies: lower `ASR_PARALLEL_WORKERS`, `MAX_CONCURRENT_REQUESTS`, `SPEAKERS_*_WORKERS`, or use `ASR_MODEL=large-v3-turbo`.                                          |
| The first `diarize` / `sound_events` request is slower than the rest                             | Expected: those models load on first use.                                                                                                                                         |
| `bind: address already in use`                                                                   | Port 8080 is taken: map another one (`-p 9090:8080`, or `AUDIO_INTEL_PORT` in `.env`).                                                                                            |

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
scripts/                     # setup / run / uninstall, image smoke check
```

## pip extras

```bash
pip install -e .              # core: faster-whisper, decode
pip install -e ".[server]"    # + FastAPI, pyannote, PANNs
pip install -e ".[vad]"       # + silero-vad
pip install -e ".[all]"       # server + vad
pip install -e ".[dev]"       # pytest, ruff, pre-commit
# WhisperX pins pyannote.audio < 4 but its alignment never imports it, so it is installed
# without deps (scripts/setup.* and the Dockerfile do this for you):
pip install --no-deps whisperx==3.7.9 && pip install nltk pandas "transformers<5"
```

## Development

Recommended editor setup: [`.vscode/extensions.json`](.vscode/extensions.json) and [`.vscode/settings.json`](.vscode/settings.json) (Python, Ruff, pytest).

```bash
./scripts/setup.sh          # or .\scripts\setup.ps1 — installs dev tools and the git hook
python -m pytest tests/
```

Before each commit, these run automatically:

- **ruff** — lint (`--fix`) and format for `*.py`
- **mdformat** — format `*.md`

Run manually on all files:

```bash
pre-commit run --all-files
```

Unit tests need no GPU and download no models (chunking, normalization, alignment bridge,
AED and speaker helpers).

`scripts/image-smoke.sh <image>` checks a built image on a CPU-only machine: decodes audio
through torchcodec/FFmpeg, imports the native stack (torch, pyannote, WhisperX alignment,
CTranslate2), and runs the tests against the image's own dependencies. CI runs it whenever
the image recipe changes.

## Releases

Images are published to `ghcr.io/thesydoruk/audio-intel` by
[`release.yml`](.github/workflows/release.yml):

1. Bump `version` in `pyproject.toml` and merge to `main`.
1. Push a matching tag: `git tag v2.0.0 && git push origin v2.0.0`.
1. The workflow builds the image, runs the smoke check, pushes `:<version>` and `:latest`,
   then creates the GitHub Release. A tag that does not match `pyproject.toml` fails fast,
   and nothing is published unless the smoke check passes.

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for setup, tests, and pull-request
expectations. This project follows the [Contributor Covenant](.github/CODE_OF_CONDUCT.md).
Security reports go through [SECURITY.md](.github/SECURITY.md), not public issues.

## License

Copyright (c) 2026 [Valerii Sydoruk](https://github.com/thesydoruk).
This project's source code is released under the [MIT License](LICENSE).

Third-party libraries and pretrained models keep their own licenses and terms.
In particular, pyannote checkpoints on Hugging Face
(`pyannote/speaker-diarization-community-1`, CC-BY-4.0) are gated:
accept their conditions and set `HF_TOKEN` before enabling diarization.
Whisper, Silero VAD, WhisperX align models, and PANNs / AudioSet assets are
downloaded separately and are not redistributed in this repository.
