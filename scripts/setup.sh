#!/usr/bin/env bash
# Local install into .venv: the service, dev tools, and WhisperX alignment.
#
# For a CUDA build of torch, point TORCH_INDEX at the matching PyTorch wheel index
# before running, e.g. TORCH_INDEX=https://download.pytorch.org/whl/cu126
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi
for tool in ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$tool is required on PATH (install ffmpeg)" >&2
    exit 1
  fi
done

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment in .venv ..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
if [[ -n "${TORCH_INDEX:-}" ]]; then
  pip install "torch>=2.8" "torchaudio>=2.8" --index-url "$TORCH_INDEX"
fi
pip install -e ".[all,dev]"
# WhisperX pins pyannote.audio < 4 but its alignment never imports it: install it
# without dependencies, plus what alignment needs (same as the Dockerfile).
pip install --no-deps "whisperx==3.7.9"
pip install "nltk>=3.9.1" "pandas>=2.2.3" "transformers>=4.48.0,<5"
python -m pre_commit install

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

mkdir -p data/whisper data/panns

echo
echo "Setup complete."
echo "  Activate:   source .venv/bin/activate"
echo "  Run API:    ./scripts/run.sh"
echo "  Check:      curl http://localhost:8080/health"
echo "  Uninstall:  ./scripts/uninstall.sh   (add --purge to also delete models and .env)"
