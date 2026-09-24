# audio-intel: per-VAD-chunk transcription (faster-whisper) + non-speech sound
# detection (PANNs/AudioSet) on a single timeline.
# CUDA 12.6 + cuDNN 9 runtime satisfies CTranslate2 (faster-whisper) GPU requirements.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv curl wget ca-certificates ffmpeg \
        # torchcodec's FFmpeg 6 loader links libpython3.12.so, absent from the venv python.
        libpython3.12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# pyannote.audio 4 needs torch >= 2.8 and torchcodec (0.7.x pairs with torch 2.8).
# whisperx pins pyannote.audio < 4 although its CTC alignment never imports
# pyannote, so it is installed without dependencies and its alignment deps are
# listed explicitly. The final reinstall keeps pip from drifting torch or
# ctranslate2 (cuDNN 9) away from this CUDA 12.6 runtime.
ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir \
        torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} \
        --index-url ${TORCH_INDEX} \
    && /opt/venv/bin/pip install --no-cache-dir -e ".[server,vad]" "torchcodec>=0.7,<0.8" \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps "whisperx==3.7.9" \
    && /opt/venv/bin/pip install --no-cache-dir \
        "nltk>=3.9.1" "pandas>=2.2.3" "transformers>=4.48.0,<5" \
        "ctranslate2>=4.5.0,<5" \
    && /opt/venv/bin/pip install --no-cache-dir \
        torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} \
        --index-url ${TORCH_INDEX}


EXPOSE 8080
CMD ["python", "-m", "audio_intel.server"]
