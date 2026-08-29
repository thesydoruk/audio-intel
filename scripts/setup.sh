#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment in .venv ..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[all,dev]"
python -m pre_commit install

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

mkdir -p data/whisper data/panns

echo
echo "Setup complete."
echo "  Activate:  source .venv/bin/activate"
echo "  Run API:   ./scripts/run.sh"
echo "  Hooks:     pre-commit installed (runs on git commit)"
echo "  Docker:    docker compose up -d --build"
