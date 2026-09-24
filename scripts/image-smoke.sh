#!/usr/bin/env bash
# Check a built image on a CPU-only machine: the native pieces that only fail at
# import time inside the real image (torch, torchcodec's FFmpeg loader, pyannote,
# whisperx alignment installed with --no-deps, CTranslate2), then the test suite
# against the image's own dependency set.
#
# Usage: scripts/image-smoke.sh <image>
set -euo pipefail

image="${1:?usage: scripts/image-smoke.sh <image>}"
root="$(cd "$(dirname "$0")/.." && pwd)"

docker run --rm -i -e CUDA_VISIBLE_DEVICES= --entrypoint python "${image}" - <<'PY'
import importlib.metadata as md

import ctranslate2
import faster_whisper
import numpy as np
import pyannote.audio
import soundfile as sf
import torch
import whisperx.alignment
from torchcodec.decoders import AudioDecoder

from audio_intel.diarization.embeddings import SpeakerEmbedder
from audio_intel.diarization.pyannote import SpeakerDiarizer

# torchcodec loads FFmpeg lazily: only a real decode proves the libraries resolve.
sf.write("/tmp/smoke.wav", np.zeros(16000, dtype=np.float32), 16000)
samples = AudioDecoder("/tmp/smoke.wav").get_all_samples()
assert samples.data.shape[-1] == 16000, samples.data.shape

for name in ("torch", "torchaudio", "torchcodec", "pyannote.audio", "whisperx",
             "faster-whisper", "ctranslate2", "audio-intel"):
    print(f"{name}=={md.version(name)}")
PY

docker run --rm -e CUDA_VISIBLE_DEVICES= \
    -v "${root}/tests:/app/tests:ro" \
    --entrypoint sh "${image}" -c \
    "pip install --quiet --no-cache-dir pytest && cd /app && python -m pytest -q -p no:cacheprovider tests"
