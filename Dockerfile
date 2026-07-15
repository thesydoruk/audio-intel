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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir \
        torch==2.5.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && /opt/venv/bin/pip install --no-cache-dir -e ".[all]" \
    # whisperx may pin ctranslate2==4.4 (cuDNN 8); this image ships cuDNN 9.
    && /opt/venv/bin/pip install --no-cache-dir "ctranslate2>=4.5.0,<5"

EXPOSE 8080
CMD ["python", "-m", "audio_intel.server"]
