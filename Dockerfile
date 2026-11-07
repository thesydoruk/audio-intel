# audio-intel: per-VAD-chunk transcription (faster-whisper) + non-speech sound
# detection (PANNs/AudioSet) on a single timeline.
# CUDA 12.8 + cuDNN 9 runtime: the first CUDA with Blackwell (sm_120, RTX 50xx)
# kernels, and what the torch cu128 wheels and CTranslate2 >= 4.6.3 are built for.
# Older cards (Ampere, Ada) run on the same image.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    # Every download lands under the models mount: faster-whisper and pyannote
    # through MODEL_CACHE_DIR / HF_HOME, SpeechBrain through torch.hub (TORCH_HOME),
    # the BS-RoFormer checkpoint through AUDIO_SEPARATOR_MODEL_DIR.
    MODEL_CACHE_DIR=/var/lib/whisper \
    HF_HOME=/var/lib/whisper \
    TORCH_HOME=/var/lib/whisper/torch \
    AUDIO_SEPARATOR_MODEL_DIR=/var/lib/whisper/audio-separator \
    # Separation jobs (inputs, tracks, results) get a mount of their own.
    JOBS_DIR=/var/lib/jobs

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv curl wget ca-certificates ffmpeg \
        # torchcodec's FFmpeg 6 loader links libpython3.12.so, absent from the venv python.
        libpython3.12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, from pyproject.toml alone, so editing src/ does not
# invalidate the multi-GB torch layer on every rebuild.
#
# constraints.txt pins torch/torchaudio to the cu128 wheels, torchcodec to the
# series that pairs with them, and ctranslate2 to a CUDA 12.8 build; every pip
# call below passes it, so no dependency can drag another torch in from PyPI.
# whisperx pins pyannote.audio < 4 although its CTC alignment never imports
# pyannote, so it is installed without dependencies and its alignment deps are
# listed explicitly.
# audio-separator (BS-RoFormer) declares the dependencies of every architecture it
# can run: demucs, diffq (a C extension with no wheel for this Python, and the
# image has no compiler), onnx2torch with torchvision, onnx. The RoFormer path
# imports none of them, so it too is installed without dependencies, followed by
# the packages that path does import (checked against 0.48.0; librosa, einops,
# soundfile, tqdm, pyyaml and requests are already present).
#
# SLIM=1 removes triton, which only torch.compile loads (~0.5 GB on disk). NCCL
# looks removable too but libtorch_cuda.so links it directly: torch does not
# import without it.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG SLIM=1
COPY pyproject.toml constraints.txt ./
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -c constraints.txt \
        torch torchaudio --index-url ${TORCH_INDEX} \
    && python3 -c 'import tomllib; p = tomllib.load(open("pyproject.toml", "rb"))["project"]; \
        print("\n".join(p["dependencies"] + p["optional-dependencies"]["server"] + p["optional-dependencies"]["vad"] \
            + p["optional-dependencies"]["separate-core"] \
            + [d for d in p["optional-dependencies"]["separate"] \
               if not d.startswith(("audio-intel", "audio-separator"))]))' \
        > /tmp/requirements.txt \
    && /opt/venv/bin/pip install --no-cache-dir -c constraints.txt \
        -r /tmp/requirements.txt torchcodec \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps "whisperx==3.7.9" \
    && /opt/venv/bin/pip install --no-cache-dir -c constraints.txt \
        "nltk>=3.9.1" "pandas>=2.2.3" "transformers>=4.48.0,<5" ctranslate2 \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps "audio-separator>=0.48.0" \
    && /opt/venv/bin/pip install --no-cache-dir -c constraints.txt \
        "beartype>=0.18.5,<0.19.0" "rotary-embedding-torch>=0.6.1,<0.7.0" \
        "onnxruntime" "pydub>=0.25" "ml_collections" "resampy>=0.4" "audioread" "six" \
    && if [ "${SLIM}" = "1" ]; then \
        /opt/venv/bin/pip uninstall -y triton; \
       fi \
    && rm /tmp/requirements.txt

COPY README.md LICENSE ./
COPY src ./src
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps -e .

EXPOSE 8080
CMD ["python", "-m", "audio_intel.server"]
