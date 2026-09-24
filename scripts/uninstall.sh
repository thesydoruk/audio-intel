#!/usr/bin/env bash
# Undo scripts/setup.sh: remove the virtual environment, the pre-commit hook, and
# build/test caches. Downloaded models (data/whisper, data/panns) and .env are kept
# unless --purge is given. Docker resources are not touched — see the README.
#
# Usage: ./scripts/uninstall.sh [--purge]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

purge=0
case "${1:-}" in
  "") ;;
  --purge) purge=1 ;;
  *) echo "usage: $0 [--purge]" >&2; exit 2 ;;
esac

if [[ -x .venv/bin/python ]]; then
  .venv/bin/python -m pre_commit uninstall >/dev/null 2>&1 || true
fi

rm -rf .venv audio_intel.egg-info .pytest_cache .ruff_cache
find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
echo "Removed .venv, the pre-commit hook, and caches."

if [[ $purge -eq 1 ]]; then
  # Keep the .gitkeep placeholders the repository tracks.
  find data/whisper data/panns -mindepth 1 ! -name .gitkeep -exec rm -rf {} + 2>/dev/null || true
  rm -f .env
  echo "Purged downloaded models (data/whisper, data/panns) and .env."
else
  echo "Kept data/whisper, data/panns and .env (use --purge to delete them)."
fi
